// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the fused head-split + RMSNorm op. Loads the resident constants once
// (row-replicated gamma tiles for Q and K, the 1/head_dim reduce scaler, eps), then
// streams one work unit (Q | K | V tiles of one (batch, seq_tile, head_group)) per
// iteration into CB 0 with a single barrier per unit.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t gq_addr = get_arg_val<uint32_t>(1);
    const uint32_t gk_addr = get_arg_val<uint32_t>(2);
    const uint32_t scaler_addr = get_arg_val<uint32_t>(3);
    const uint32_t eps_addr = get_arg_val<uint32_t>(4);
    const uint32_t cos_addr = get_arg_val<uint32_t>(5);
    const uint32_t sin_addr = get_arg_val<uint32_t>(6);
    const uint32_t trans_addr = get_arg_val<uint32_t>(7);
    const uint32_t num_work_units = get_arg_val<uint32_t>(8);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(9);

    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(0);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(1);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t head_groups = get_compile_time_arg_val(5);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(6);
    constexpr uint32_t cache_rot = get_compile_time_arg_val(8);  // 1: cos/sin pushed only when the seq tile changes
    constexpr uint32_t q_split = get_compile_time_arg_val(9);    // 1, or 2: unit = half the Q heads + (K xor V)
    // 1: scaler / eps / rotation / cos / sin CBs alias a per-core L1 shard that already holds them; the reader only
    // reserves and pushes them, and reads gamma after the first unit is pushed (compute waits for it at first use).
    constexpr uint32_t resident = get_compile_time_arg_val(10);
    constexpr auto in0_args = TensorAccessorArgs<11>();
    constexpr auto gq_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto gk_args = TensorAccessorArgs<gq_args.next_compile_time_args_offset()>();
    constexpr auto sc_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();
    constexpr auto eps_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<eps_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr uint32_t fuse_rotary = get_compile_time_arg_val(7);  // 0/1; placeholders bound when 0

    constexpr uint32_t cb_id = 0;  // fused QKV tiles for compute
    constexpr uint32_t cb_gq = 1, cb_gk = 2, cb_scaler = 3, cb_eps = 4;
    constexpr uint32_t cb_cos = 9, cb_sin = 10, cb_trans = 11;

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);
    const auto sgq = TensorAccessor(gq_args, gq_addr);
    const auto sgk = TensorAccessor(gk_args, gk_addr);
    const auto ssc = TensorAccessor(sc_args, scaler_addr);
    const auto seps = TensorAccessor(eps_args, eps_addr);
    const auto scos = TensorAccessor(cos_args, cos_addr);
    const auto ssin = TensorAccessor(sin_args, sin_addr);
    const auto strans = TensorAccessor(trans_args, trans_addr);
    const uint32_t tile_size_bytes = get_tile_size(cb_id);
    const uint32_t const_tile_bytes = get_tile_size(cb_gq);

    Noc noc;
    CircularBuffer cb(cb_id);

    constexpr uint32_t group_q_tiles = heads_per_group * q_heads_per_kv * head_dim_tiles;
    constexpr uint32_t group_kv_tiles = heads_per_group * head_dim_tiles;
    constexpr uint32_t q_tiles_total = num_kv_heads * q_heads_per_kv * head_dim_tiles;
    constexpr uint32_t kv_tiles_total = num_kv_heads * head_dim_tiles;
    constexpr uint32_t sub_q_tiles = group_q_tiles / q_split;
    constexpr uint32_t kv_parts = (q_split == 1) ? 2 : 1;
    constexpr uint32_t unit_tiles = sub_q_tiles + kv_parts * group_kv_tiles;

    auto read_gamma = [&]() {
        CircularBuffer cgq(cb_gq), cgk(cb_gk);
        cgq.reserve_back(head_dim_tiles);
        cgk.reserve_back(head_dim_tiles);
        for (uint32_t i = 0; i < head_dim_tiles; ++i) {
            noc.async_read(sgq, cgq, const_tile_bytes, {.page_id = i}, {.offset_bytes = i * const_tile_bytes});
            noc.async_read(sgk, cgk, const_tile_bytes, {.page_id = i}, {.offset_bytes = i * const_tile_bytes});
        }
        noc.async_read_barrier();
        cgq.push_back(head_dim_tiles);
        cgk.push_back(head_dim_tiles);
    };

    // Resident constants (never popped by compute).
    if constexpr (resident) {
        CircularBuffer csc(cb_scaler), ceps(cb_eps), ct(cb_trans);
        csc.reserve_back(1);
        csc.push_back(1);
        ceps.reserve_back(1);
        ceps.push_back(1);
        ct.reserve_back(1);
        ct.push_back(1);
    } else {
        CircularBuffer cgq(cb_gq), cgk(cb_gk), csc(cb_scaler), ceps(cb_eps);
        cgq.reserve_back(head_dim_tiles);
        for (uint32_t i = 0; i < head_dim_tiles; ++i) {
            noc.async_read(sgq, cgq, const_tile_bytes, {.page_id = i}, {.offset_bytes = i * const_tile_bytes});
        }
        cgk.reserve_back(head_dim_tiles);
        for (uint32_t i = 0; i < head_dim_tiles; ++i) {
            noc.async_read(sgk, cgk, const_tile_bytes, {.page_id = i}, {.offset_bytes = i * const_tile_bytes});
        }
        csc.reserve_back(1);
        noc.async_read(ssc, csc, const_tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
        ceps.reserve_back(1);
        noc.async_read(seps, ceps, const_tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
        if constexpr (fuse_rotary) {
            CircularBuffer ct(cb_trans);
            ct.reserve_back(1);
            noc.async_read(strans, ct, const_tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            ct.push_back(1);
        }
        noc.async_read_barrier();
        cgq.push_back(head_dim_tiles);
        cgk.push_back(head_dim_tiles);
        csc.push_back(1);
        ceps.push_back(1);
    }

    uint32_t last_s_tile = 0xFFFFFFFFu;
    for (uint32_t w = 0; w < num_work_units; ++w) {
        const uint32_t work_unit = work_unit_start + w;
        const uint32_t sub = work_unit % q_split;  // which Q half (0 also carries K, 1 carries V)
        const uint32_t rest = work_unit / q_split;
        const uint32_t block = rest / head_groups;          // (batch, seq_tile) pair
        const uint32_t group = rest - block * head_groups;  // which head group
        const bool has_k = (q_split == 1) || sub == 0;
        const bool has_v = (q_split == 1) || sub == 1;
        const uint32_t s_tile = block % seq_tiles;
        const uint32_t batch = block / seq_tiles;
        const uint32_t block_base = batch * (seq_tiles * in0_w_tiles) + s_tile * in0_w_tiles;
        const uint32_t q_base_tile = block_base + group * group_q_tiles;
        const uint32_t k_base_tile = block_base + q_tiles_total + group * group_kv_tiles;
        const uint32_t v_base_tile = block_base + q_tiles_total + kv_tiles_total + group * group_kv_tiles;

        cb.reserve_back(unit_tiles);
        uint32_t l1_write_offset = 0;
        const uint32_t q_sub_base_tile = q_base_tile + sub * sub_q_tiles;
        for (uint32_t i = 0; i < sub_q_tiles; ++i) {
            noc.async_read(
                s0, cb, tile_size_bytes, {.page_id = q_sub_base_tile + i}, {.offset_bytes = l1_write_offset});
            l1_write_offset += tile_size_bytes;
        }
        if (has_k) {
            for (uint32_t i = 0; i < group_kv_tiles; ++i) {
                noc.async_read(
                    s0, cb, tile_size_bytes, {.page_id = k_base_tile + i}, {.offset_bytes = l1_write_offset});
                l1_write_offset += tile_size_bytes;
            }
        }
        if (has_v) {
            for (uint32_t i = 0; i < group_kv_tiles; ++i) {
                noc.async_read(
                    s0, cb, tile_size_bytes, {.page_id = v_base_tile + i}, {.offset_bytes = l1_write_offset});
                l1_write_offset += tile_size_bytes;
            }
        }
        bool load_rot = false;
        if constexpr (fuse_rotary) {
            load_rot = !cache_rot || s_tile != last_s_tile;
        }
        if constexpr (resident) {
            // cos/sin already sit in the aliased CBs (this core's units share one seq tile): push them to keep
            // the per-unit wait/pop lockstep with compute; the CB is exactly Wt tiles, so it wraps onto itself.
            CircularBuffer ccos(cb_cos), csin(cb_sin);
            ccos.reserve_back(head_dim_tiles);
            csin.reserve_back(head_dim_tiles);
            noc.async_read_barrier();
            ccos.push_back(head_dim_tiles);
            csin.push_back(head_dim_tiles);
        } else if (load_rot) {
            // cos/sin tiles for this seq tile (shared by every head in the unit). With cache_rot
            // they stay in the CB until the seq tile changes; the compute pops in lockstep.
            CircularBuffer ccos(cb_cos), csin(cb_sin);
            ccos.reserve_back(head_dim_tiles);
            csin.reserve_back(head_dim_tiles);
            for (uint32_t j = 0; j < head_dim_tiles; ++j) {
                noc.async_read(
                    scos,
                    ccos,
                    const_tile_bytes,
                    {.page_id = s_tile * head_dim_tiles + j},
                    {.offset_bytes = j * const_tile_bytes});
                noc.async_read(
                    ssin,
                    csin,
                    const_tile_bytes,
                    {.page_id = s_tile * head_dim_tiles + j},
                    {.offset_bytes = j * const_tile_bytes});
            }
            noc.async_read_barrier();
            ccos.push_back(head_dim_tiles);
            csin.push_back(head_dim_tiles);
            last_s_tile = s_tile;
        } else {
            noc.async_read_barrier();
        }
        cb.push_back(unit_tiles);
        if constexpr (resident) {
            if (w == 0) {
                read_gamma();
            }
        }
    }
}
