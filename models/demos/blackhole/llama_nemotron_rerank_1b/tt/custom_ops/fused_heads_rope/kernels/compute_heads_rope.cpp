// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused head-split + RoPE compute. Per work unit the input CB holds q_per_kv Q heads,
// one K head and one V head (Wt tiles each). Every Q and K head gets
//   si  = (x @ T) * sin      -> CB 13   (T is the rotation tile; sin applied on the DST tile)
//   out = x * cos + si       -> output  (si added on the DST tile)
// which is what rotary_embedding_llama computes; V is copied through. The output CB
// carries the unit's tiles in the same order (Q heads, K, V).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_in = 0, cb_cos = 9, cb_sin = 10, cb_trans = 11, cb_si = 13, cb_out = 16;
constexpr auto D2B = EltwiseBinaryReuseDestType::DEST_TO_SRCB;  // DST -> SrcB, CB tile -> SrcA

template <uint32_t Wt>
inline void rotary_head(uint32_t off) {
    CircularBuffer si(cb_si);
    // si = (x @ T) * sin
    reconfig_data_format(cb_in, cb_trans);
    pack_reconfig_data_format(cb_si);
    matmul_init(cb_in, cb_trans);
    si.reserve_back(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        matmul_tiles(cb_in, cb_trans, off + j, 0, j);
    }
    reconfig_data_format_srca(cb_sin);
    mul_reuse_dest_init<D2B>(cb_sin);
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_reuse_dest_tiles<D2B>(cb_sin, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_si, j);
    }
    tile_regs_release();
    si.push_back(Wt);
    // out = x * cos + si
    reconfig_data_format(cb_in, cb_cos);
    pack_reconfig_data_format(cb_out);
    mul_init(cb_in, cb_cos);
    si.wait_front(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_tiles(cb_in, cb_cos, off + j, j, j);
    }
    reconfig_data_format_srca(cb_si);
    add_reuse_dest_init<D2B>(cb_si);
    for (uint32_t j = 0; j < Wt; ++j) {
        add_reuse_dest_tiles<D2B>(cb_si, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_out, off + j);
    }
    tile_regs_release();
    si.pop_front(Wt);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t q_per_kv = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t n_kv = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    constexpr uint32_t unit_tiles = (q_per_kv + 2) * Wt;

    CircularBuffer in(cb_in), out(cb_out), ccos(cb_cos), csin(cb_sin), ctr(cb_trans);
    compute_kernel_hw_startup(cb_in, cb_trans, cb_out);
    ctr.wait_front(1);
    // cos / sin stay at the CB front while consecutive units share a seq tile (mirrors the reader).
    uint32_t last_s = 0xFFFFFFFFu;
    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t s = ((unit_start + w) / n_kv) % St;
        if (s != last_s) {
            if (last_s != 0xFFFFFFFFu) {
                ccos.pop_front(Wt);
                csin.pop_front(Wt);
            }
            ccos.wait_front(Wt);
            csin.wait_front(Wt);
            last_s = s;
        }
        in.wait_front(unit_tiles);
        out.reserve_back(unit_tiles);
        for (uint32_t h = 0; h < q_per_kv + 1; ++h) {  // Q heads, then K
            rotary_head<Wt>(h * Wt);
        }
        // V passes through
        reconfig_data_format(cb_in, cb_in);
        pack_reconfig_data_format(cb_out);
        copy_init(cb_in);
        tile_regs_acquire();
        for (uint32_t j = 0; j < Wt; ++j) {
            copy_tile(cb_in, (q_per_kv + 1) * Wt + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Wt; ++j) {
            pack_tile(j, cb_out, (q_per_kv + 1) * Wt + j);
        }
        tile_regs_release();
        out.push_back(unit_tiles);
        in.pop_front(unit_tiles);
    }
    if (last_s != 0xFFFFFFFFu) {
        ccos.pop_front(Wt);
        csin.pop_front(Wt);
    }
    ctr.pop_front(1);
}
