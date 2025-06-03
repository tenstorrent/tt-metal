// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/sharded_accessor.h"

void kernel_main() {
    const uint32_t bank_base_address = get_arg_val<uint32_t>(0);

    constexpr DataFormat data_format = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t rank = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t base_idx = 4;

    using input_dspec = distribution_spec_t<base_idx, rank, num_banks>;

    auto sharded_accessor = ShardedAccessor<input_dspec, page_size>{.bank_base_address = bank_base_address};

    auto interleaved_accessor = InterleavedAddrGenFast</*DRAM=*/false>{
        .bank_base_address = bank_base_address, .page_size = page_size, .data_format = data_format};

    /* Benchmark get_noc_addr for both accessors
     * - get_noc_addr is a good proxy for page lookup logic
     * - Use volatile to prevent compiler from optimizing away the calls
     * - Some notes on tracy profiler:
     *   - Max number of markers is 250 (?), so we use a hard-coded loop count to make sure tests are consistent for
     * larger shapes
     *   - Cycles are fairly consistent across runs without much variation
     *   - There is a small overhead for tracy (~18 cycles?) for each marker
     *     * You can verify by inserting a dummy marker or removing the volatile since compiler will optimize out the
     * calls
     */
    constexpr size_t loop_count = 30;
    for (size_t i = 0; i < loop_count; ++i) {
        auto page_id = i % input_dspec::tensor_volume;
        {
            DeviceZoneScopedN("SHARDED_ACCESSOR");
            volatile auto _ = sharded_accessor.get_noc_addr(i);
        }
        {
            DeviceZoneScopedN("INTERLEAVED_ACCESSOR");
            volatile auto _ = interleaved_accessor.get_noc_addr(i);
        }
    }
}
