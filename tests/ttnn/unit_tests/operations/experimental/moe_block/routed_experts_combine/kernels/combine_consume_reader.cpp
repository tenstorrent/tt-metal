// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Combine consume microbench, reader (NCRISC): the accumulator reader's per-row work on rows landed in L1, through
// quantized_row::RowStager (routed_experts_combine kernels/dataflow/quantized_row_stage.hpp).
//
// At the start (not timed) the P wire rows of the input are read into a pool in L1 (the landing slots). Then n_rows
// rows go to the compute, row n from slot n % P, for token n % tokens: the row staged (cb_row, and Fp8E4m3's factor
// CB); Bf16 / Int8: the gate (trailer word 1, bf16 bits) into element 0 of a cb_w tile; meta (cb_meta): the token, the
// gate.
//
// The rows go through RowStager's two-phase path (issue: the NoC reads, the cells into a scratch; finish: the exponents
// / factors), as the accumulator reads a pool slot on another core.
//
// Compile-time args: format, n_rows, pool_slots, slot_bytes, H, producers, tokens, accumulator_protocol, then the
// input's TensorAccessorArgs. Runtime args: the input's address.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/routed_experts_combine/device/kernels/dataflow/quantized_row_stage.hpp"

void kernel_main() {
    constexpr auto format = static_cast<quantized_row::Format>(get_compile_time_arg_val(0));
    constexpr uint32_t n_rows = get_compile_time_arg_val(1);
    constexpr uint32_t pool_slots = get_compile_time_arg_val(2);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t producers = get_compile_time_arg_val(5);
    constexpr uint32_t tokens = get_compile_time_arg_val(6);
    // 1: the meta pages of routed_experts_combine's accumulator_compute.cpp (the real kernel): a first page
    // 0x80000000 | tokens, a token a row, 0xFFFFFFFF at the end; the gate tile for every format
    constexpr bool accumulator_protocol = get_compile_time_arg_val(7) != 0;
    constexpr auto in_args = TensorAccessorArgs<8>();
    constexpr uint32_t cb_row = 0, cb_w = 1, cb_fac = 2, cb_meta = 3, cb_pool = 10;
    constexpr uint32_t kTrailer = quantized_row::row_bytes(format, H, producers);

    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0), slot_bytes);
    const uint32_t pool = get_write_ptr(cb_pool);
    for (uint32_t s = 0; s < pool_slots; ++s) {
        noc_async_read(in.get_noc_addr(s), pool + s * slot_bytes, slot_bytes);
    }
    noc_async_read_barrier();
    quantized_row::RowStager<format, H, producers, cb_row, cb_fac> stager;
    stager.init();
    constexpr uint32_t kTiles = quantized_row::flat_tiles(H);
    // the cells' scratch: past the pool (its last slot's padding is too small: one more slot's room is in the CB)
    const uint32_t cells_scratch = pool + pool_slots * slot_bytes;
    auto meta_page = [&](uint32_t value) {
        cb_reserve_back(cb_meta, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = value;
        cb_push_back(cb_meta, 1);
    };
    if constexpr (accumulator_protocol) {
        meta_page(0x80000000u | tokens);
    }
    {
        DeviceZoneScopedN("CMB-RD");
        for (uint32_t n = 0; n < n_rows; ++n) {
            const uint32_t slot = pool + (n % pool_slots) * slot_bytes;
            const uint32_t gate = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + kTrailer)[1] & 0xFFFFu;
            // (the accumulator's pool path: the slot read over the NoC, its cells into a scratch, then finish)
            cb_reserve_back(cb_row, kTiles);
            uint32_t factor = 0;
            if constexpr (format == quantized_row::Format::Fp8E4m3) {
                cb_reserve_back(cb_fac, kTiles);
                factor = get_write_ptr(cb_fac);
            }
            stager.issue(get_noc_addr(slot), get_write_ptr(cb_row), cells_scratch);
            if constexpr (accumulator_protocol || format != quantized_row::Format::Fp8E4m3) {
                cb_reserve_back(cb_w, 1);
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w))[0] = gate;
                cb_push_back(cb_w, 1);
            }
            cb_reserve_back(cb_meta, 1);
            volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta));
            meta[0] = n % tokens;
            meta[1] = gate;
            noc_async_read_barrier();
            stager.finish(get_write_ptr(cb_row), cells_scratch, factor);
            stager.push();
            cb_push_back(cb_meta, 1);
        }
    }
    if constexpr (accumulator_protocol) {
        meta_page(0xFFFFFFFFu);
    }
}
