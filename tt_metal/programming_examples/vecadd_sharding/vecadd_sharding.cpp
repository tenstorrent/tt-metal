// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This programming example is an advanced example, compared to the vecadd single core example in the
// contributed folder.  It illustrated sharding tensor inputs to L1 memory of multiple cores directly,
// then perform vector addition tile by tile. Because of sharding to L1, DRAM nor NoC is not involved.
// Data copy is avoided and reader and writer kernels are not needed.

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

struct DistributionConfig {
    TensorMemoryLayout layout;
    uint32_t num_cores_y;
    uint32_t num_cores_x;
};

std::string next_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Expected argument after " << argv[i] << std::endl;
        exit(1);
    }
    return argv[++i];
}

void help(std::string_view program_name) {
    fmt::print("Usage: {} [options]\n", program_name);
    fmt::print(
        "This program demonstrates how to add two vectors in sharding mode, using multiple cores and sharded L1 "
        "buffers.\n\n");
    fmt::print("Options:\n");
    fmt::print("  --device, -d <device_id>         Specify the device to run the program on. Default is 0.\n");
    fmt::print(
        "  --sharding_type, -s <sharding>   Specify the sharding type (options: height, width, block). Default is "
        "height.\n");
    exit(0);
}

int main(int argc, char** argv) {
    // This is an advanced example of vector addition with sharded L1 buffers. Sharding is a technique that allows
    // distributing data is specific patterns across multiple cores's L1(SRAM) memory, reducing or eliminating the
    // need for NoC bandwidth. This example demonstrates how to set up sharded L1 buffers and perform vector addition
    // across them.
    //
    // Sharding is quite percise and requires exact division of the data across the cores. The example tries to
    // distribute 64 (4x4) tiles across 4 cores. But in different sharding modes:
    // * height: Shard the tensors across the height dimension
    // * width: Shard the tensors across the width dimension
    // * block: Shard the tensors across both height and width dimensions
    //
    // In different modes, the data is distributed differently across the cores in different patterns. Each core:
    // * hight: Uses 4 cores in the y direction. Each core gets 1 rows of 4 tiles each.
    // * width: Uses 4 cores in the x direction. Each core gets 1 column of 4 tiles each.
    // * block: Uses a 2x2 grid of cores. Each core gets 2 rows and 2 columns of tiles, effectively sharding the tensor
    // into blocks.

    // used fixed seed for reproducibility and deterministic results
    int seed = 0x1234567;
    int device_id = 0;
    std::string sharding_type = "height";

    const std::unordered_map<std::string_view, DistributionConfig> test_configs{
        {"height", {TensorMemoryLayout::HEIGHT_SHARDED, 4, 1}},
        {"width", {TensorMemoryLayout::WIDTH_SHARDED, 1, 4}},
        {"block", {TensorMemoryLayout::BLOCK_SHARDED, 2, 2}},
    };

    // Quick and dirty argument parsing.
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "--device" || arg == "-d") {
            device_id = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--help" || arg == "-h") {
            help(argv[0]);
            return 0;
        } else if (arg == "--sharding_type" || arg == "-s") {
            sharding_type = next_arg(i, argc, argv);
            if (not test_configs.contains(sharding_type)) {
                std::cout << "Invalid sharding type: " << sharding_type << std::endl;
                help(argv[0]);
                return 1;
            }
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
            help(argv[0]);
        }
    }

    DeviceContext ctx(device_id);
    const auto& config = test_configs.at(sharding_type);

    // In this example we will distribute 16 tiles (4x4) across 4 cores.
    constexpr uint32_t n_tiles_y = 4;
    constexpr uint32_t n_tiles_x = 4;

    // Calculate the number of tiles per core in each dimension based on the configuration.
    const uint32_t num_tiles_per_core_x = n_tiles_x / config.num_cores_x;
    const uint32_t num_tiles_per_core_y = n_tiles_y / config.num_cores_y;
    const uint32_t num_tiles_per_core = num_tiles_per_core_x * num_tiles_per_core_y;

    fmt::print(
        "Sharding {}x{} tiles to {}x{} cores in {} mode\n",
        n_tiles_y,
        n_tiles_x,
        config.num_cores_y,
        config.num_cores_x,
        config.layout);
    fmt::print("Each core will handle {}x{} tiles\n\n", num_tiles_per_core_y, num_tiles_per_core_x);

    // The cores that the buffer will be sharded across.
    const CoreRange cores(CoreCoord(0, 0), CoreCoord(config.num_cores_y - 1, config.num_cores_x - 1));

    // Create the input and output buffers that live on L1(SRAM), sharded across cores.
    ShardConfig shard_config{
        .cores = CoreRangeSet(cores),
        .shard_shape =
            {num_tiles_per_core_x * tt::constants::TILE_HEIGHT, num_tiles_per_core_y * tt::constants::TILE_WIDTH},
        .tensor2d_shape_in_pages = {n_tiles_y, n_tiles_x},
        .layout = config.layout,
    };
    auto a = ctx.sharded_l1_buffer(shard_config);
    auto b = ctx.sharded_l1_buffer(shard_config);
    auto c = ctx.sharded_l1_buffer(shard_config);

    // Data to fill the input buffers.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    const size_t num_elements = n_tiles_x * n_tiles_y * tt::constants::TILE_HW;
    std::vector<bfloat16> a_data(num_elements);
    std::vector<bfloat16> b_data(num_elements);
    for (size_t i = 0; i < a_data.size(); ++i) {
        a_data[i] = bfloat16(dist(rng));
        b_data[i] = bfloat16(dist(rng));
    }

    // Copy data from host to L1 directly.
    ctx.write(a, a_data);
    ctx.write(b, b_data);

    // Build the program: 3 L1-backed circular buffers + compute kernel.
    // Because data is already in L1 via sharding, no reader/writer kernels are needed.
    // The CBs point directly at the sharded L1 buffer memory (no extra allocation or copy).
    auto program =
        ProgramBuilder(cores)
            .cb(tt::CBIndex::c_0, a, num_tiles_per_core)
            .cb(tt::CBIndex::c_1, b, num_tiles_per_core)
            .cb(tt::CBIndex::c_2, c, num_tiles_per_core)
            .compute(
                "tt_metal/programming_examples/vecadd_sharding/kernels/add_sharding.cpp",
                ComputeConfig{
                    .math_approx_mode = false,
                    .compile_args = {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2}})
            .runtime_args({num_tiles_per_core})
            .done()
            .build();

    ctx.run(std::move(program));

    fmt::print("Kernel execution finished. Reading results...\n");

    // Read the output buffer.
    auto c_data = ctx.read<bfloat16>(c);

    // Print partial results so we can see the output is correct (plus or minus
    // some error due to BFP16 precision)
    fmt::print("Partial results: (note we are running under BFP16. It's going to be less accurate)\n");
    size_t element_per_core = tt::constants::TILE_HW * num_tiles_per_core;
    size_t print_per_core = std::min((size_t)10, element_per_core);

    int core_idx = 0;
    for ([[maybe_unused]] auto& core : cores) {
        const auto core_offset = core_idx * element_per_core;
        fmt::print("Core {}:\n", core_idx);
        for (size_t index = 0; index < print_per_core; index++) {
            const auto i = core_offset + index;
            fmt::print(
                "index {}: {} + {} = {}\n",
                i,
                static_cast<float>(a_data[i]),
                static_cast<float>(b_data[i]),
                static_cast<float>(c_data[i]));
        }
        std::cout << std::endl;
        core_idx++;
    }

    // Verify the results
    bool pass = true;
    for (size_t i = 0; i < c_data.size(); i++) {
        float expected = static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]);
        if (std::abs(static_cast<float>(c_data[i]) - expected) > 0.2f) {  // Allow some error due to BFP16 precision
            fmt::print(
                stderr, "Mismatch at index {}: expected {}, got {}\n", i, expected, static_cast<float>(c_data[i]));
            pass = false;
        }
    }

    if (pass) {
        fmt::print("All results match expected values within tolerance.\n");
    } else {
        fmt::print(stderr, "Some results did not match expected values.\n");
    }

    // DeviceContext closes the device automatically when it goes out of scope.
    return 0;
}
