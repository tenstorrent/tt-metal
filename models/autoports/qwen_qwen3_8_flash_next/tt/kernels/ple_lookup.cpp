// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Decode BFP4_b exactly as blockfloat_common.cpp, directly into BF16 bits.
static uint16_t decode(uint8_t nibble, uint8_t exponent) {
    uint32_t mantissa = nibble & 7;
    if (!mantissa) {
        return 0;
    }
    uint32_t shifts = 0;
    while (!(mantissa & 4)) {
        mantissa <<= 1;
        ++shifts;
    }
    mantissa = (mantissa << 1) & 7;
    return ((nibble >> 3) << 15) | ((exponent - shifts) << 7) | (mantissa << 4);
}

void kernel_main() {
    Noc noc;
    constexpr uint32_t rows_per_shard = get_compile_time_arg_val(0);
    constexpr uint32_t num_shards = get_compile_time_arg_val(1);
    constexpr auto indices_args = TensorAccessorArgs<2>();
    constexpr auto output_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto indices = TensorAccessor(indices_args, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1));
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t scratch = cb.get_write_ptr();
    auto ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto tile = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch + 64);
    auto out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 640);
    for (uint32_t token = start; token < start + count; ++token) {
        noc.async_read(indices, CoreLocalMem<uint32_t>(scratch), 64, {.page_id = token}, {});
        noc.async_read_barrier();
        for (uint32_t head = 0; head < 16; ++head) {
            const uint32_t index = ids[head];
            const uint32_t shard = index / rows_per_shard;
            ASSERT(shard < num_shards);
            const uint32_t row = index % rows_per_shard;
            const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(4 + shard));
            for (uint32_t col_tile = 0; col_tile < 5; ++col_tile) {
                noc.async_read(
                    table, CoreLocalMem<uint32_t>(scratch + 64), 576, {.page_id = (row / 32) * 5 + col_tile}, {});
                noc.async_read_barrier();
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t face = (row % 32 / 16) * 2 + c / 16;
                    const uint32_t face_row = row % 16;
                    const uint32_t elem = face * 256 + face_row * 16 + c % 16;
                    const uint8_t packed = tile[64 + elem / 2];
                    const uint8_t nibble = (packed >> (4 * (elem % 2))) & 15;
                    out[head * 160 + col_tile * 32 + c] = decode(nibble, tile[face * 16 + face_row]);
                }
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(scratch + 640), output, 5120, {}, {.page_id = token});
        noc.async_write_barrier();
    }
}
