// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE stage 3 reader, one core: the key and query global norms (bf16 TILE [1,1,4,2560], Wt tiles each) into CBs 0 / 1,
// the scale tile (fp32, every element 2560^-0.5, the chain's scalar-filled operand) into CB 5, the value row (bf16
// TILE [1,1,1,640], Vt tiles; row 0 valid) repeated over the four branch rows into CB 9 (the chain's ttnn.repeat: rows
// 0-3 = row 0, the rest 0 -- each tile a NoC copy of the bf16 zero tile, then row 0's two face rows (32 bytes each)
// copied into rows 0-3 by the RISC), and the reduce scaler tile into CB 3 (waited on and ignored by the accurate fold).
// Compile-time args: 0 Wt, 1 Vt, then TensorAccessorArgs(key), (query), (value), (scale tile), (zero tile).
// Runtime args: the five buffer addresses in that order, 5 the first key/query tile (the lane's Wt block), 6 the first
// value tile (the lane's Vt block); the 1-row form passes 0, 0.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "../../kernels/zones.h"

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t Vt = get_compile_time_arg_val(1);
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096, BF16_FACE = 512, BF16_ROW = 32;
constexpr uint32_t c_key = 0, c_query = 1, c_scaler = 3, c_scale = 5, c_val = 9;

template <typename Acc>
FORCE_INLINE void read_tiles(
    Noc& noc, const Acc& acc, DataflowBuffer& dfb, uint32_t tiles, uint32_t tile_bytes, uint32_t first) {
    dfb.reserve_back(tiles);
    for (uint32_t t = 0; t < tiles; ++t) {
        noc.async_read(
            acc, dfb, tile_bytes, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = t * tile_bytes});
    }
}

void kernel_main() {
    constexpr auto a_key = TensorAccessorArgs<2>();
    constexpr auto a_query = TensorAccessorArgs<a_key.next_compile_time_args_offset()>();
    constexpr auto a_val = TensorAccessorArgs<a_query.next_compile_time_args_offset()>();
    constexpr auto a_scale = TensorAccessorArgs<a_val.next_compile_time_args_offset()>();
    constexpr auto a_zero = TensorAccessorArgs<a_scale.next_compile_time_args_offset()>();
    const auto key = TensorAccessor(a_key, get_arg_val<uint32_t>(0));
    const auto query = TensorAccessor(a_query, get_arg_val<uint32_t>(1));
    const auto value = TensorAccessor(a_val, get_arg_val<uint32_t>(2));
    const auto scale = TensorAccessor(a_scale, get_arg_val<uint32_t>(3));
    const auto zero = TensorAccessor(a_zero, get_arg_val<uint32_t>(4));
    const uint32_t first_kq = get_arg_val<uint32_t>(5);
    const uint32_t first_v = get_arg_val<uint32_t>(6);
    Noc noc;
    DataflowBuffer key_dfb(c_key), query_dfb(c_query), scale_dfb(c_scale), val_dfb(c_val);
    {
        FUSED_ZONE("fz_pl_gate_r_reads");
        read_tiles(noc, key, key_dfb, Wt, BF16_TILE, first_kq);
        read_tiles(noc, query, query_dfb, Wt, BF16_TILE, first_kq);
        scale_dfb.reserve_back(1);
        noc.async_read(scale, scale_dfb, FP32_TILE, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    }
    // value branches: zero tiles first, then row 0 of each value tile (face 0 and face 1 rows, 32 bytes each)
    val_dfb.reserve_back(Vt);
    const uint32_t vbase = val_dfb.get_write_ptr();
    {
        FUSED_ZONE("fz_pl_gate_r_zero_rows");
        for (uint32_t t = 0; t < Vt; ++t) {
            noc.async_read(
                zero, val_dfb, BF16_TILE, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = t * BF16_TILE});
        }
        noc.async_read_barrier();
        key_dfb.push_back(Wt);
        query_dfb.push_back(Wt);
        scale_dfb.push_back(1);
    }
    // row 0 of value tile t -> a 64-byte staging read per face row would overlap the zero copies: read the whole
    // value tiles into the scaler CB's neighbourhood instead? Simpler: read value tile t into row block 0 directly
    // (the zero copy has landed), then duplicate its row 0 into rows 1-3 on the RISC.
    {
        FUSED_ZONE("fz_pl_gate_r_value_rows");
        for (uint32_t t = 0; t < Vt; ++t) {
            noc.async_read(
                value, val_dfb, BF16_ROW, {.page_id = first_v + t, .offset_bytes = 0}, {.offset_bytes = t * BF16_TILE});
            noc.async_read(
                value,
                val_dfb,
                BF16_ROW,
                {.page_id = first_v + t, .offset_bytes = BF16_FACE},
                {.offset_bytes = t * BF16_TILE + BF16_FACE});
        }
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vbase);
        for (uint32_t t = 0; t < Vt; ++t) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t row0 = (t * BF16_TILE + face * BF16_FACE) / 4;  // 8 words per bf16 face row
                for (uint32_t r = 1; r < 4; ++r) {
                    for (uint32_t w = 0; w < 8; ++w) {
                        v[row0 + r * 8 + w] = v[row0 + w];
                    }
                }
            }
        }
        val_dfb.push_back(Vt);
    }
    dataflow_kernel_lib::prepare_reduce_scaler<c_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
}
