// Diagnostic only: TensorSpec construction implicitly initializes MetalContext/UMD.
// Do not run this executable concurrently with device work; it is NOT a device-free probe.
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <cassert>
#include <iostream>

using namespace tt::tt_metal;

int main() {
    for (const auto& [width, shard_width, cores] :
         {std::tuple<uint32_t, uint32_t, uint32_t>{32, 64, 1}, {4160, 480, 9}}) {
        const CoreRangeSet grid(CoreRange(CoreCoord{0, 0}, CoreCoord{cores - 1, 0}));
        const TensorSpec spec(
            Shape{1, 1, 32, width},
            TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::TILE),
                MemoryConfig(
                    TensorMemoryLayout::WIDTH_SHARDED,
                    BufferType::L1,
                    ShardSpec{grid, {32, shard_width}, ShardOrientation::ROW_MAJOR})));
        const auto shard_pages = spec.compute_buffer_sharding_args().shard_spec()->num_pages();
        const auto per_bank = spec.compute_consumed_memory_bytes_per_bank(16, 1);
        assert(spec.compute_packed_buffer_size_bytes() == 32 * width * 2);
        assert(shard_pages == shard_width / 32);
        assert(per_bank == 32 * shard_width * 2);
        std::cout << "N=" << width << " packed=" << spec.compute_packed_buffer_size_bytes()
                  << " shard_pages=" << shard_pages << " allocated_per_core=" << per_bank
                  << " allocated_total=" << per_bank * cores << '\n';
    }
}
