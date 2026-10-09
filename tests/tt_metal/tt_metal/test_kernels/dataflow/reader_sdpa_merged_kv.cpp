// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr std::uint32_t rounds = get_compile_time_arg_val(0);
    constexpr std::uint32_t chunks = get_compile_time_arg_val(1);
    constexpr bool separate_v = get_compile_time_arg_val(2) == 1;
    constexpr std::uint32_t row_tiles = get_compile_time_arg_val(3);
    std::uint32_t addresses[3] = {
        get_arg_val<std::uint32_t>(0), get_arg_val<std::uint32_t>(1), get_arg_val<std::uint32_t>(2)};
    Noc noc;
    auto read_tiles = [&](std::uint32_t cb, std::uint32_t count) {
        CircularBuffer input(cb);
        const auto bytes = count * input.get_tile_size();
        input.reserve_back(count);
        noc.async_read(
            AllocatorBank<AllocatorBankType::DRAM>{}, input, bytes, {.bank_id = 0, .addr = addresses[cb]}, {});
        noc.async_read_barrier();
        input.push_back(count);
        addresses[cb] += bytes;
    };
    for (std::uint32_t round = 0; round < rounds; ++round) {
        read_tiles(tt::CBIndex::c_0, 2);
        for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
            read_tiles(tt::CBIndex::c_1, 2 * row_tiles);
            if constexpr (separate_v) {
                read_tiles(tt::CBIndex::c_2, 4);
            }
        }
    }
}
