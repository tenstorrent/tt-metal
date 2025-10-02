// Quick debug version - minimal code to get DPRINT output
#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp>
#include <ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/nlp_concat_heads_boltz.hpp>

#include <iostream>
#include <chrono>
#include <thread>

using namespace ttnn;

int main() {
    try {
        std::cout << "Opening device..." << std::endl;
        auto device = ttnn::device::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
        std::cout << "Device opened" << std::endl;

        // Use same dimensions as working version
        const uint32_t num_heads = 4, seq = 32, head_dim = 32;
        Shape input_shape({num_heads, seq, seq, head_dim});
        std::cout << "Tensor shape: [" << num_heads << ", " << seq << ", " << seq << ", " << head_dim << "]" << std::endl;

        // Create interleaved tensor first
        auto interleaved_input = zeros(input_shape, DataType::BFLOAT16, TILE_LAYOUT, *device);
        std::cout << "Interleaved tensor created" << std::endl;

        // Convert to sharded - use exact same config as working version
        // For HEIGHT_SHARDED: shard_height = num_heads * seq = 4 * 32 = 128
        // shard_width = head_dim = 32 (must match physical width)
        uint32_t shard_height = num_heads * seq; // 128
        uint32_t shard_width = head_dim;         // 32
        std::cout << "Shard config: height=" << shard_height << ", width=" << shard_width << std::endl;

        auto core_range = CoreRangeSet({CoreRange({0, 0}, {7, 3})}); // 8x4 = 32 cores
        auto shard_spec = tt::tt_metal::ShardSpec(core_range, {shard_height, shard_width});
        auto sharded_mem_config = MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec);
        auto sharded_input = ttnn::interleaved_to_sharded(interleaved_input, sharded_mem_config, std::nullopt, std::nullopt);

        std::cout << "Running operation..." << std::endl;
        auto out = ttnn::experimental::nlp_concat_heads_boltz(sharded_input);
        std::cout << "Operation completed!" << std::endl;

        // Give DPRINT time to flush, then force exit
        std::this_thread::sleep_for(std::chrono::seconds(3));
        std::cout << "Debug output complete. Force exiting..." << std::endl;
        std::_Exit(0); // Force exit without cleanup

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::_Exit(1);
    }
}
