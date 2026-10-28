// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the decode tile's dense projection (tt/v41/decode_dense.py DecodeDense; kernels/dense_row_reader.cpp
// has the dataflow). Per unit (column j = b NT + n, segment s): one dot product over the segment's L tiles of the
// resident row block b (DST 0, fp32 accumulation, the fidelity the caller's); an owned unit's (s = 0) partial is
// packed as an fp32 tile into cb_own, another's into cb_part for the writer to send to the owner. Then, per owned
// unit, the own tile plus the S - 1 received fp32 tiles (exact fp32 adds) packed as the output tile.
//
// compile_time_args = [cb_x, cb_w, cb_part, cb_own, cb_recv, cb_out, KT, NT, B, L, BLK, PROBE, S]
// runtime args      = [n_units, n_owned, (j, s, owned) * n_units]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    const uint32_t n_owned = get_arg_val<uint32_t>(1);
    constexpr uint32_t UNITS = 2;

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_part = get_compile_time_arg_val(2);
    constexpr uint32_t cb_own = get_compile_time_arg_val(3);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t KT = get_compile_time_arg_val(6);
    constexpr uint32_t NT = get_compile_time_arg_val(7);
    constexpr uint32_t B = get_compile_time_arg_val(8);
    constexpr uint32_t L = get_compile_time_arg_val(9);
    constexpr uint32_t BLK = get_compile_time_arg_val(10);
    constexpr uint32_t PROBE = get_compile_time_arg_val(11);  // 2: no matmuls (a timing probe, wrong output)
    constexpr uint32_t S = get_compile_time_arg_val(12);
    constexpr uint32_t XT = B * KT;

    DataflowBuffer xb(cb_x), wb(cb_w), part(cb_part), own(cb_own), recv(cb_recv), out(cb_out);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    if (n_units == 0) {
        return;
    }
    xb.wait_front(XT);
    if (S > 1) {
        pack_reconfig_data_format(cb_part);  // the partials are fp32 tiles (cb_own the same format)
    }
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t j = get_arg_val<uint32_t>(UNITS + 3 * u), sg = get_arg_val<uint32_t>(UNITS + 3 * u + 1);
        const uint32_t owned = get_arg_val<uint32_t>(UNITS + 3 * u + 2);
        const uint32_t x0 = (j / NT) * KT + sg * L;  // the segment's first x tile
        tile_regs_acquire();
        matmul_init(cb_x, cb_w);
        for (uint32_t kb = 0; kb < L; kb += BLK) {
            wb.wait_front(BLK);
            for (uint32_t t = 0; t < BLK && (PROBE & 2) == 0; ++t) {
                matmul_tiles(cb_x, cb_w, x0 + kb + t, t, 0);  // DST[0] += x[b, k] @ W[b, k, n]
            }
            wb.pop_front(BLK);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (S == 1) {
            out.reserve_back(1);
            pack_tile(0, cb_out);
            out.push_back(1);
        } else if (owned) {
            own.reserve_back(1);
            pack_tile(0, cb_own);
            own.push_back(1);
        } else {
            part.reserve_back(1);
            pack_tile(0, cb_part);
            part.push_back(1);
        }
        tile_regs_release();
    }
    if (S > 1 && n_owned > 0) {
        own.wait_front(n_owned);
        recv.wait_front(n_owned * (S - 1));
        reconfig_data_format(cb_own, cb_recv);  // the unpacker from bf16 x bfp8 to fp32 tiles
        pack_reconfig_data_format(cb_out);
        for (uint32_t o = 0; o < n_owned; ++o) {
            tile_regs_acquire();
            add_init(cb_own, cb_recv);
            add_tiles(cb_own, cb_recv, o, o * (S - 1), 0);
            if (S > 2) {
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv);
                for (uint32_t t = 1; t < S - 1; ++t) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv, o * (S - 1) + t, 0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            out.reserve_back(1);
            pack_tile(0, cb_out);
            out.push_back(1);
            tile_regs_release();
        }
        own.pop_front(n_owned);
        recv.pop_front(n_owned * (S - 1));
    }
}
