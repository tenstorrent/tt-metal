// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr std::uint32_t rounds = get_compile_time_arg_val(0);
    constexpr std::uint32_t tiles = get_compile_time_arg_val(1);
    constexpr bool normalize = get_compile_time_arg_val(2);
    std::uint32_t output_address = get_arg_val<std::uint32_t>(0);
    std::uint32_t stats_address = get_arg_val<std::uint32_t>(1);
    CircularBuffer output(tt::CBIndex::c_16);
    CircularBuffer stats(tt::CBIndex::c_17);
    Noc noc;
    for (std::uint32_t round = 0; round < rounds; ++round) {
        if constexpr (!normalize) {
            stats.wait_front(1);
            noc.async_write(
                stats,
                AllocatorBank<AllocatorBankType::DRAM>{},
                stats.get_tile_size(),
                {},
                {.bank_id = 0, .addr = stats_address});
            noc.async_write_barrier();
            stats.pop_front(1);
            stats_address += stats.get_tile_size();
        }
        output.wait_front(tiles);
        noc.async_write(
            output,
            AllocatorBank<AllocatorBankType::DRAM>{},
            tiles * output.get_tile_size(),
            {},
            {.bank_id = 0, .addr = output_address});
        noc.async_write_barrier();
        output.pop_front(tiles);
        output_address += tiles * output.get_tile_size();
    }
}
