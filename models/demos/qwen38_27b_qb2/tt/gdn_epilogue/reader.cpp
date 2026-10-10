// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

// DFB_BINDINGS

void kernel_main() {
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(0);
    constexpr bool compact = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t heads = get_compile_time_arg_val(2);
    constexpr uint32_t gate_tile_offset = get_compile_time_arg_val(3);
    constexpr uint32_t input_padding = get_compile_time_arg_val(4);
    static_assert(input_padding <= 2);
    constexpr auto ra = TensorAccessorArgs<5>();
    constexpr auto ga = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    const auto raw = TensorAccessor(ra, get_arg_val<uint32_t>(0), 512);
    const auto gate = TensorAccessor(ga, get_arg_val<uint32_t>(1), 2048);
    const auto weight = TensorAccessor(wa, get_arg_val<uint32_t>(2), 2048);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t stride = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, 128>();
    DataflowBuffer epsilon(dfb::epsilon);
    generate_bcast_col_scalar(epsilon, epsilon_bits);
    cb_reserve_back(dfb::weight, 4);
    for (uint32_t tile = 0; tile < 4; ++tile) {
        noc_async_read_tile(tile, weight, get_write_ptr(dfb::weight) + tile * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(dfb::weight, 4);
    // Public output needs zero padding. Compact output writes only row zero
    // and independently clears its destination padding, so its opt-in path
    // may skip these 48 KiB of stores. All live FP32/BF16 words below are
    // overwritten before publication, including on both ring-buffer slots.
    // Poison mode checks that the unchanged per-row compute cannot leak
    // unused input rows into the live result, including after CB wrap.
    if constexpr (input_padding != 1) {
        auto* x_init = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(dfb::x));
        auto* g_init = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(dfb::gate));
        for (uint32_t i = 0; i < 8 * 1024; ++i) {
            x_init[i] = input_padding == 2 ? 0x7fc00000 : 0;
        }
        for (uint32_t i = 0; i < 8 * 512; ++i) {
            g_init[i] = input_padding == 2 ? 0x7fc07fc0 : 0;
        }
    }
    // CB11 stages one FP32 row and eight aligned BF16 row pairs. Blackhole
    // DRAM reads require equal low six address bits: compact odd users must
    // read an aligned pair, then select their half in L1.
    const uint32_t scratch = get_write_ptr(11);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t head = first + item * stride;
        const uint32_t user = head / heads;
        const uint32_t row_pair = (user / 16) * 1024 + ((user % 16) / 2) * 64;
        cb_reserve_back(dfb::x, 4);
        cb_reserve_back(dfb::gate, 4);
        const uint32_t x_base = get_write_ptr(dfb::x);
        const uint32_t g_base = get_write_ptr(dfb::gate);
        noc_async_read(raw.get_noc_addr(head), scratch, 512);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            for (uint32_t face = 0; face < 2; ++face) {
                if constexpr (compact) {
                    const uint32_t page = gate_tile_offset + (head % heads) * 4 + tile;
                    noc_async_read(
                        gate.get_noc_addr(page, row_pair + face * 512), scratch + 512 + (tile * 2 + face) * 64, 64);
                } else {
                    noc_async_read(
                        gate.get_noc_addr(head * 4 + tile, face * 512), g_base + tile * 2048 + face * 512, 32);
                }
            }
        }
        noc_async_read_barrier();
        if constexpr (compact) {
            const auto* pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 512);
            auto* gates = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_base);
            for (uint32_t tile = 0; tile < 4; ++tile) {
                for (uint32_t face = 0; face < 2; ++face) {
                    for (uint32_t word = 0; word < 8; ++word) {
                        gates[tile * 512 + face * 128 + word] = pairs[(tile * 2 + face) * 16 + (user % 2) * 8 + word];
                    }
                }
            }
        }
        const auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        auto* tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x_base);
        for (uint32_t value = 0; value < 128; ++value) {
            tiles[(value / 32) * 1024 + ((value % 32) / 16) * 256 + value % 16] = row[value];
        }
        cb_push_back(dfb::x, 4);
        cb_push_back(dfb::gate, 4);
    }
}
