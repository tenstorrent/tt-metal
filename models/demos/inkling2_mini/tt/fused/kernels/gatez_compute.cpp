// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate prep + down_proj, compute of K part s. Prep cores first run the gate-prep compute
// (attnres_compute with nq = 0: bf16 rms(v)). Then partial z_s = x_s @ down_s (fp32 DST); the root (s = 0) adds
// the other parts' partial rows, packed into nb tiles per z tile, with selection matmuls (S[r, g] = 1 where packed
// row g came from row r; partials enter the FPU as tf32, like the reduce of the ttnn.sum it replaces), packs z. The z
// part is all FPU: no SFPU op touches DST.
#define kernel_main prep_main
#include "attnres_compute.cpp"
#undef kernel_main

#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t kt_n = get_named_compile_time_arg_val("kt");
    constexpr uint32_t kz = get_named_compile_time_arg_val("kz");
    constexpr uint32_t n_split = get_named_compile_time_arg_val("n_split");
    constexpr uint32_t cb_x = 5, cb_w = 6, cb_recv = 7, cb_sel = 8, cb_zero = 9, cb_z = 18;
    constexpr uint32_t nb = get_named_compile_time_arg_val("nb");
    constexpr uint32_t G = 4;  // fp32 DST tiles per half-sync section (zeroed by the packer on release)

    const uint32_t s = get_arg_val<uint32_t>(0);
    if (get_arg_val<uint32_t>(1)) {  // 2: prep only (fused gate)
        prep_main();
        if (get_arg_val<uint32_t>(1) == 2) {
            return;
        }
    } else {
        compute_kernel_hw_startup(cb_x, cb_w, cb_z);
    }
    pack_reconfig_data_format(cb_z);
    DeviceZoneScopedN("C_z");
    cb_wait_front(cb_zero, 1);
    cb_wait_front(cb_x, kt_n);
    cb_wait_front(cb_w, kt_n * kz);
    if (s == 0) {
    }
    cb_reserve_back(cb_z, kz);
    for (uint32_t n0 = 0; n0 < kz; n0 += G) {
        const uint32_t ng = kz - n0 < G ? kz - n0 : G;
        tile_regs_acquire();
        reconfig_data_format_srca(cb_zero);
        copy_init(cb_zero);
        for (uint32_t n = 0; n < ng; ++n) {
            copy_tile(cb_zero, 0, n);
        }
        reconfig_data_format(cb_w, cb_x);
        matmul_init(cb_x, cb_w);
        for (uint32_t n = 0; n < ng; ++n) {
            for (uint32_t kt = 0; kt < kt_n; ++kt) {
                matmul_tiles(cb_x, cb_w, kt, kt * kz + n0 + n, n);
            }
        }
        if (s == 0) {  // z[r] += sum over senders of packed row (s' - 1) * rows + r: S_b @ P_b, b < nb
            cb_wait_front(cb_sel, nb);
            cb_wait_front(cb_recv, nb * kz);
            reconfig_data_format(cb_recv, cb_sel);
            matmul_init(cb_sel, cb_recv);
            for (uint32_t n = 0; n < ng; ++n) {
                for (uint32_t b = 0; b < nb; ++b) {
                    matmul_tiles(cb_sel, cb_recv, b, (n0 + n) * nb + b, n);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t n = 0; n < ng; ++n) {
            pack_tile(n, cb_z);
        }
        tile_regs_release();
    }
    cb_push_back(cb_z, kz);
}
