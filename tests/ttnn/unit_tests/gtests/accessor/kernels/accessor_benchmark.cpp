// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/sharded_accessor.h"

void kernel_main() {
    const uint32_t bank_base_address = get_common_arg_val<uint32_t>(0);

    constexpr DataFormat data_format = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t base_idx_cta = 2;

    using input_dspec_cta = nd_sharding::distribution_spec_t<base_idx_cta, 0>;
    // runtime tensor shape, runtime shard shape, and static bank coords
    constexpr uint32_t base_idx_cta_DDS = base_idx_cta + nd_sharding::compile_time_args_skip<input_dspec_cta>();
    constexpr uint32_t base_idx_crta_DDS = 0;
    using input_dspec_crta_DDS = nd_sharding::distribution_spec_t<base_idx_cta_DDS, base_idx_crta_DDS>;

    constexpr uint32_t base_idx_cta_SSD =
        base_idx_cta_DDS + nd_sharding::compile_time_args_skip<input_dspec_crta_DDS>();
    constexpr uint32_t base_idx_crta_SSD = base_idx_crta_DDS + nd_sharding::runtime_args_skip<input_dspec_crta_DDS>();
    using input_dspec_crta_SSD = nd_sharding::distribution_spec_t<base_idx_cta_SSD, base_idx_crta_SSD>;

    constexpr uint32_t base_idx_cta_DDD =
        base_idx_cta_SSD + nd_sharding::compile_time_args_skip<input_dspec_crta_SSD>();
    constexpr uint32_t base_idx_crta_DDD = base_idx_crta_SSD + nd_sharding::runtime_args_skip<input_dspec_crta_SSD>();
    using input_dspec_crta_DDD = nd_sharding::distribution_spec_t<base_idx_cta_DDD, base_idx_crta_DDD>;

    auto sharded_accessor_cta = nd_sharding::ShardedAccessor<input_dspec_cta, page_size>(bank_base_address);
    auto sharded_accessor_crta_DDS = nd_sharding::ShardedAccessor<input_dspec_crta_DDS, page_size>(bank_base_address);
    auto sharded_accessor_crta_SSD = nd_sharding::ShardedAccessor<input_dspec_crta_SSD, page_size>(bank_base_address);
    auto sharded_accessor_crta_DDD = nd_sharding::ShardedAccessor<input_dspec_crta_DDD, page_size>(bank_base_address);

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
        auto page_id = i % sharded_accessor_cta.get_dspec().get_tensor_volume();
        {
            DeviceZoneScopedN("SHARDED_ACCESSOR_CTA");
            volatile auto _ = sharded_accessor_cta.get_noc_addr(i);
        }
    }
    for (size_t i = 0; i < loop_count; ++i) {
        auto page_id = i % sharded_accessor_cta.get_dspec().get_tensor_volume();
        {
            DeviceZoneScopedN("SHARDED_ACCESSOR_CRTA_DDS");
            volatile auto _ = sharded_accessor_crta_DDS.get_noc_addr(i);
        }
    }
    for (size_t i = 0; i < loop_count; ++i) {
        auto page_id = i % sharded_accessor_cta.get_dspec().get_tensor_volume();
        {
            DeviceZoneScopedN("SHARDED_ACCESSOR_CRTA_SSD");
            volatile auto _ = sharded_accessor_crta_SSD.get_noc_addr(i);
        }
    }
    for (size_t i = 0; i < loop_count; ++i) {
        auto page_id = i % sharded_accessor_cta.get_dspec().get_tensor_volume();
        {
            DeviceZoneScopedN("SHARDED_ACCESSOR_CRTA_DDD");
            volatile auto _ = sharded_accessor_crta_DDD.get_noc_addr(i);
        }
    }
    for (size_t i = 0; i < loop_count; ++i) {
        auto page_id = i % sharded_accessor_cta.get_dspec().get_tensor_volume();
        {
            DeviceZoneScopedN("INTERLEAVED_ACCESSOR");
            volatile auto _ = interleaved_accessor.get_noc_addr(i);
        }
    }
}
