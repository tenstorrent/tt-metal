// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr std::uint32_t rounds = get_compile_time_arg_val(0);
    constexpr std::uint32_t tiles = get_compile_time_arg_val(1);
    std::uint32_t addresses[4] = {
        get_arg_val<std::uint32_t>(0),
        get_arg_val<std::uint32_t>(1),
        get_arg_val<std::uint32_t>(2),
        get_arg_val<std::uint32_t>(3)};
    Noc noc;
    for (std::uint32_t round = 0; round < rounds; ++round) {
        for (std::uint32_t index = 0; index < 4; ++index) {
            CircularBuffer input(index);
            const std::uint32_t count = index < 2 ? 1 : tiles;
            const std::uint32_t bytes = count * input.get_tile_size();
            input.reserve_back(count);
            noc.async_read(
                AllocatorBank<AllocatorBankType::DRAM>{}, input, bytes, {.bank_id = 0, .addr = addresses[index]}, {});
            noc.async_read_barrier();
            input.push_back(count);
            addresses[index] += bytes;
        }
    }
}
