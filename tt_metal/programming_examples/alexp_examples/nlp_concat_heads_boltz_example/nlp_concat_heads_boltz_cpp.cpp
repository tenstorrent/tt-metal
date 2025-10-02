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
        std::cout << "Opening device..." << std::endl;
        // Open device
        device = ttnn::device::open_mesh_device(/*device_id=*/0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
        std::cout << "Device opened successfully" << std::endl;

        // Example input shape: [num_heads, S, S, head_dim] in TILE layout
        // Choose dimensions that are multiples of tile size (32x32 for tiles)
        const uint32_t num_heads = 8;
        const uint32_t seq = 64;      // multiple of 32
        const uint32_t head_dim = 64; // multiple of 32exitr

        Shape input_shape({num_heads, seq, seq, head_dim});
        std::cout << "Creating tensor with shape: [" << num_heads << ", " << seq << ", " << seq << ", " << head_dim << "]" << std::endl;

        // First, let's test with regular interleaved tensor to ensure basic operation works
        auto input = zeros(input_shape, DataType::BFLOAT16, TILE_LAYOUT, *device);
        std::cout << "Tensor created successfully" << std::endl;

        // Run the op via C++ API
        std::cout << "Running nlp_concat_heads_boltz operation..." << std::endl;
        auto out = ttnn::experimental::nlp_concat_heads_boltz(input);
        std::cout << "Operation completed successfully" << std::endl;

        // Prevent DCE
        (void)out;
        std::cout << "Program finished successfully" << std::endl;

        // Skip device cleanup to avoid hanging
        std::cout << "Operation completed successfully! Exiting without device cleanup to avoid hang." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        // Skip device cleanup on error to avoid hanging
        return 1;
    }
}
