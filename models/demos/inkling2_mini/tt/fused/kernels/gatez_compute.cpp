// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate prep + down_proj (E10c), compute of split s. Prep cores first run the gate-prep compute
// (attnres_compute with nq = 0: bf16 rms(v)). Then partial z_s = x_s @ down_s (fp32 DST); the root (s = 0) adds
// the other splits' partials, received in cb_recv, as partial @ I on the FPU (same tf32 operand rounding as the
// reduce of ttnn.sum it replaces), and packs z. The z part is all FPU: no SFPU op touches DST there.
#define kernel_main prep_main
#include "attnres_compute.cpp"
#undef kernel_main

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t kt_n = get_named_compile_time_arg_val("kt");
    constexpr uint32_t kz = get_named_compile_time_arg_val("kz");
    constexpr uint32_t n_split = get_named_compile_time_arg_val("n_split");
    constexpr uint32_t cb_x = 5, cb_w = 6, cb_recv = 7, cb_eye = 8, cb_zero = 9, cb_z = 18;
    constexpr uint32_t G = 4;  // fp32 DST tiles per half-sync section (zeroed by the packer on release)

    const uint32_t s = get_arg_val<uint32_t>(0);
    if (get_arg_val<uint32_t>(1)) {
        prep_main();
    } else {
        compute_kernel_hw_startup(cb_x, cb_w, cb_z);
    }
    pack_reconfig_data_format(cb_z);
    cb_wait_front(cb_zero, 1);
    cb_wait_front(cb_x, kt_n);
    cb_wait_front(cb_w, kt_n * kz);
    if (s == 0) {
        cb_wait_front(cb_eye, 1);
        cb_wait_front(cb_recv, (n_split - 1) * kz);
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
        if (s == 0) {
            reconfig_data_format(cb_eye, cb_recv);
            matmul_init(cb_recv, cb_eye);
            for (uint32_t n = 0; n < ng; ++n) {
                for (uint32_t p = 0; p < n_split - 1; ++p) {
                    matmul_tiles(cb_recv, cb_eye, p * kz + n0 + n, 0, n);
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
