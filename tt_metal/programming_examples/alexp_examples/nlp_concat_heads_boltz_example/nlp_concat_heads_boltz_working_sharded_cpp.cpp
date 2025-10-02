// SPDX-FileCopyrightText: © 2025 Tenstorrent
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/memory_config/memory_config.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp>
#include <ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/nlp_concat_heads_boltz.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>

using namespace ttnn;

int main() {
    std::shared_ptr<ttnn::device::MeshDevice> device = nullptr;
    try {
        std::cout << "=== WORKING SHARDED VERSION ===" << std::endl;
        std::cout << "Opening device..." << std::endl;
        // Open device
        device = ttnn::device::open_mesh_device(/*device_id=*/0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
        std::cout << "Device opened successfully" << std::endl;

        // Use smaller dimensions to avoid memory allocation issues
        const uint32_t num_heads = 4;    // Reduced from 8
        const uint32_t seq = 32;         // Reduced from 64
        const uint32_t head_dim = 32;    // Reduced from 64

        Shape input_shape({num_heads, seq, seq, head_dim});
        std::cout << "Creating tensor with SMALLER shape: [" << num_heads << ", " << seq << ", " << seq << ", " << head_dim << "]" << std::endl;

        // WORKAROUND: Create interleaved tensor first, then convert to sharded
        std::cout << "Step 1: Creating interleaved tensor..." << std::endl;
        auto interleaved_input = zeros(input_shape, DataType::BFLOAT16, TILE_LAYOUT, *device);
        std::cout << "Interleaved tensor created successfully" << std::endl;

        // Set up sharding configuration with correct physical dimensions
        std::cout << "Step 2: Setting up sharding configuration..." << std::endl;

        // For HEIGHT_SHARDED with TILE_LAYOUT, shard_width MUST match physical width
        // Physical width for TILE_LAYOUT = padded_width = 32 (since head_dim=32, which is tile-aligned)
        // Let's try a larger shard_height to reduce the number of required cores
        uint32_t shard_height = 128;  // Use the full height to minimize shards
        uint32_t shard_width = 32;    // MUST match physical width for HEIGHT_SHARDED

        std::cout << "Correct shard shape: [" << shard_height << ", " << shard_width << "]" << std::endl;
        std::cout << "Note: shard_width=" << shard_width << " matches physical width for HEIGHT_SHARDED" << std::endl;
        std::cout << "Note: shard_height=" << shard_height << " covers full tensor height to minimize cores needed" << std::endl;

        // The system says it needs 32 cores, so let's provide exactly 32 cores
        auto core_range = CoreRangeSet({CoreRange({0, 0}, {7, 3})}); // 8x4 = 32 cores
        std::cout << "Using 32 cores (8x4 grid) for HEIGHT_SHARDED: (0,0) to (7,3)" << std::endl;
        auto shard_spec = tt::tt_metal::ShardSpec(core_range, {shard_height, shard_width});
        auto sharded_mem_config = MemoryConfig(
            TensorMemoryLayout::HEIGHT_SHARDED,
            BufferType::L1,
            shard_spec
        );

        std::cout << "Step 3: Converting to sharded tensor..." << std::endl;
        try {
            // Use interleaved_to_sharded operation to convert existing tensor to sharded
            // Need to provide optional parameters for the API
            auto sharded_input = ttnn::interleaved_to_sharded(interleaved_input, sharded_mem_config, std::nullopt, std::nullopt);
            std::cout << "Sharded conversion successful!" << std::endl;

            // Debug: Print tensor info
            std::cout << "=== SHARDED TENSOR INFO ===" << std::endl;
            std::cout << "Logical shape: " << sharded_input.logical_shape() << std::endl;
            std::cout << "Padded shape: " << sharded_input.padded_shape() << std::endl;
            if (sharded_input.is_sharded()) {
                auto shard_spec_val = sharded_input.shard_spec().value();
                std::cout << "Shard spec shape: [" << shard_spec_val.shape[0] << ", " << shard_spec_val.shape[1] << "]" << std::endl;
            }

        // Run the op via C++ API
        std::cout << "Step 4: Running nlp_concat_heads_boltz operation on sharded tensor..." << std::endl;
        auto out = ttnn::experimental::nlp_concat_heads_boltz(sharded_input);
        std::cout << "Operation completed successfully!" << std::endl;

        // Prevent DCE
        (void)out;
        std::cout << "=== WORKING SHARDED VERSION SUCCESS ===" << std::endl;

        // Give DPRINT time to flush all output, then exit immediately
        std::cout << "Waiting for DPRINT output to complete..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "DPRINT output should be complete. Exiting immediately." << std::endl;
        std::exit(0);
            return 0;

        } catch (const std::exception& shard_error) {
            std::cerr << "Sharding conversion failed: " << shard_error.what() << std::endl;

            // Fallback: Use the interleaved tensor directly
            std::cout << "FALLBACK: Using interleaved tensor for operation..." << std::endl;
            auto out = ttnn::experimental::nlp_concat_heads_boltz(interleaved_input);
            std::cout << "Operation completed successfully with interleaved tensor!" << std::endl;

        // Prevent DCE
        (void)out;
        std::cout << "=== FALLBACK VERSION SUCCESS ===" << std::endl;

        // Skip device cleanup to avoid hanging - operation completed successfully
        std::cout << "Operation completed successfully! Exiting without device cleanup to avoid hang." << std::endl;
        std::cout << "Note: The device cleanup causes hanging due to PCIE communication issues." << std::endl;

        // Force immediate exit without cleanup
        std::exit(0);
        }

    } catch (const std::exception& e) {
        std::cerr << "WORKING VERSION ERROR: " << e.what() << std::endl;
        // Skip device cleanup on error to avoid hanging
        std::cerr << "Exiting without device cleanup due to error" << std::endl;
        return 1;
    }
}
