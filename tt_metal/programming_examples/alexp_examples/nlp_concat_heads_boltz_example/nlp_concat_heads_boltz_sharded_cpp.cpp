// SPDX-FileCopyrightText: © 2025 Tenstorrent
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/memory_config/memory_config.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/nlp_concat_heads_boltz.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <array>
#include <cstdint>
#include <iostream>

using namespace ttnn;

int main() {
    std::shared_ptr<ttnn::device::MeshDevice> device = nullptr;
    try {
        std::cout << "=== SHARDED VERSION DEBUG ===" << std::endl;
        std::cout << "Opening device..." << std::endl;
        // Open device
        device = ttnn::device::open_mesh_device(/*device_id=*/0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
        std::cout << "Device opened successfully" << std::endl;

        // Example input shape: [num_heads, S, S, head_dim] in TILE layout
        // Choose dimensions that are multiples of tile size (32x32 for tiles)
        const uint32_t num_heads = 8;
        const uint32_t seq = 64;      // multiple of 32
        const uint32_t head_dim = 64; // multiple of 32

        Shape input_shape({num_heads, seq, seq, head_dim});
        std::cout << "Creating SHARDED tensor with shape: [" << num_heads << ", " << seq << ", " << seq << ", " << head_dim << "]" << std::endl;

        // Create sharded tensor to force using the sharded kernel
        std::cout << "Setting up sharding configuration..." << std::endl;

        // Fix shard dimensions to match physical requirements
        // For HEIGHT_SHARDED: shard_width must match physical width (512 for TILE_LAYOUT)
        uint32_t shard_height = 64;   // Match sequence length (padded_shape[-2])
        uint32_t shard_width = 512;   // Must match physical width for HEIGHT_SHARDED

        std::cout << "Shard shape: [" << shard_height << ", " << shard_width << "]" << std::endl;

        auto core_range = CoreRangeSet({CoreRange({0, 0}, {0, 0})}); // Single core
        std::cout << "Core range set to single core (0,0)" << std::endl;

        auto shard_spec = tt::tt_metal::ShardSpec(core_range, {shard_height, shard_width});
        std::cout << "ShardSpec created" << std::endl;

        auto sharded_mem_config = MemoryConfig(
            TensorMemoryLayout::HEIGHT_SHARDED,
            BufferType::L1,
            shard_spec
        );
        std::cout << "MemoryConfig created with HEIGHT_SHARDED" << std::endl;

            std::cout << "Creating sharded tensor..." << std::endl;
            std::cout << "DEBUG: About to call zeros() with sharded config" << std::endl;

            // Add memory debugging - let's check if we can catch the alignment issue
            std::cout << "DEBUG: MEMCPY_ALIGNMENT requirement: 16 bytes" << std::endl;

            ttnn::Tensor input;
            try {
                input = zeros(input_shape, DataType::BFLOAT16, TILE_LAYOUT, *device, sharded_mem_config);
                std::cout << "Sharded tensor created successfully!" << std::endl;
            } catch (const std::exception& inner_e) {
                std::cerr << "DETAILED ERROR during tensor creation: " << inner_e.what() << std::endl;

                // Let's try to understand what went wrong
                std::cout << "DEBUG: Tensor creation failed. This is likely due to:" << std::endl;
                std::cout << "1. Memory alignment issues (MEMCPY_ALIGNMENT=16)" << std::endl;
                std::cout << "2. Invalid sharded buffer page mapping" << std::endl;
                std::cout << "3. Insufficient L1 memory for shard allocation" << std::endl;
                throw;
            }

        // Debug: Print tensor shapes
        std::cout << "=== TENSOR SHAPE DEBUG ===" << std::endl;
        std::cout << "Logical shape: " << input.logical_shape() << std::endl;
        std::cout << "Padded shape: " << input.padded_shape() << std::endl;
        if (input.is_sharded()) {
            auto shard_spec = input.shard_spec().value();
            std::cout << "Shard spec shape: [" << shard_spec.shape[0] << ", " << shard_spec.shape[1] << "]" << std::endl;
            std::cout << "Padded shape[-2]: " << input.padded_shape()[-2] << std::endl;
            std::cout << "Padded shape[-1]: " << input.padded_shape()[-1] << std::endl;
        }

        // Run the op via C++ API
        std::cout << "Running nlp_concat_heads_boltz operation on sharded tensor..." << std::endl;
        auto out = ttnn::experimental::nlp_concat_heads_boltz(input);
        std::cout << "Operation completed successfully!" << std::endl;

        // Prevent DCE
        (void)out;
        std::cout << "=== SHARDED VERSION SUCCESS ===" << std::endl;

        // Skip device cleanup to avoid hanging
        std::cout << "Operation completed successfully! Exiting without device cleanup to avoid hang." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "SHARDED VERSION ERROR: " << e.what() << std::endl;
        // Skip device cleanup on error to avoid hanging
        return 1;
    }
}
