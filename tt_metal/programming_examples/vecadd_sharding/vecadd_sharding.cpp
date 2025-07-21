// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This programming example is an advanced example, compared to the vecadd single core example in the
// contributed folder.  It illustrated sharding tensor inputs to L1 memory of multiple cores directly,
// then perform vector addition tile by tile. Because of sharding to L1, DRAM is not involved.
// Data copy is avoided and reader and writer kernels are not needed.

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

// sharding configuration is defined by the following struct
struct L1Config {
    L1Config(TensorMemoryLayout layout, uint32_t cores_height, uint32_t cores_width) :
        buffer_layout(layout), num_cores_height(cores_height), num_cores_width(cores_width) {}

    TensorMemoryLayout buffer_layout;
    uint32_t num_cores_height;
    uint32_t num_cores_width;

    // following sharding parameters are hardcode for this example
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
    uint32_t element_size = 2;
    uint32_t num_tiles_per_core_height = 2;
    uint32_t num_tiles_per_core_width = 2;

    // following sharding parameters are calculated based on the above configuration
    uint32_t num_cores = num_cores_height * num_cores_width;
    uint32_t num_tiles_per_core = num_tiles_per_core_height * num_tiles_per_core_width;
    uint32_t size_bytes = num_cores_height * num_tiles_per_core_height * tt::constants::TILE_HEIGHT * num_cores_width *
                          num_tiles_per_core_width * tt::constants::TILE_WIDTH * element_size;
    uint32_t page_size_bytes = tt::constants::TILE_HW * element_size;
    CoreRange cores = CoreRange(CoreCoord(0, 0), CoreCoord(0, num_cores - 1));
    ShardSpecBuffer shard_spec() const {
        return ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({cores})),
            {(uint32_t)num_tiles_per_core_height * tt::constants::TILE_HEIGHT,
             (uint32_t)num_tiles_per_core_width * tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {num_cores_height * num_tiles_per_core_height * num_cores_height,
             num_tiles_per_core_width * num_cores_width});
    }
};

std::shared_ptr<Buffer> MakeShardedL1BufferBFP16(IDevice* device, const L1Config& test_config) {
    return CreateBuffer(tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = test_config.size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_layout = test_config.buffer_layout,
        .shard_parameters = test_config.shard_spec()});
}

CBHandle MakeCircularBufferBFP16(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles, const std::shared_ptr<Buffer>& l1_buf) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_HW;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(n_tiles * tile_size, {{cb, tt::DataFormat::Float16_b}})
                                              .set_page_size(cb, tile_size)
                                              // IMPORTANT: assign L1 buffer address to circular buffer directly so that
                                              // no extra allocation and data copy
                                              .set_globally_allocated_address(*l1_buf);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

std::string next_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Expected argument after " << argv[i] << std::endl;
        exit(1);
    }
    return argv[++i];
}

void help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "This program demonstrates how to add two vectors using "
                 "tt-Metalium.\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --device, -d <device_id>  Specify the device to run the "
                 "program on. Default is 0.\n";
    std::cout << "  --sharding_type, -s <sharding>  Specify the sharding type "
                 "options are height, width, or block. Default is height.\n";
    exit(0);
}

int main(int argc, char** argv) {
    // used fixed seed for reproducibility and deterministic results
    int seed = 0x1234567;
    int device_id = 0;
    std::string sharding_type = "height";

    // sharding configuration, 4x4 of tiles bfloat16, each core has 2x2 tiles, sharded to 4 core
    const std::unordered_map<std::string_view, L1Config> test_configs{
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

    IDevice* device = CreateDevice(device_id);
    Program program = CreateProgram();

    std::cout << "Sharding type: " << sharding_type << std::endl;
    const auto& test_config = test_configs.at(sharding_type);

    // Create the input and output buffers.
    auto a = MakeShardedL1BufferBFP16(device, test_config);
    auto b = MakeShardedL1BufferBFP16(device, test_config);
    auto c = MakeShardedL1BufferBFP16(device, test_config);

    std::mt19937 rng(seed);
    auto a_data = create_random_vector_of_bfloat16_native(test_config.size_bytes, 10, rng());
    auto b_data = create_random_vector_of_bfloat16_native(test_config.size_bytes, 10, rng());

    MakeCircularBufferBFP16(program, test_config.cores, tt::CBIndex::c_0, test_config.num_tiles_per_core, a);
    MakeCircularBufferBFP16(program, test_config.cores, tt::CBIndex::c_1, test_config.num_tiles_per_core, b);
    MakeCircularBufferBFP16(program, test_config.cores, tt::CBIndex::c_2, test_config.num_tiles_per_core, c);

    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/vecadd_sharding/kernels/add_sharding.cpp",
        test_config.cores,
        ComputeConfig{
            .math_approx_mode = false,
            // pass in compile time arguments
            .compile_args = {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2},
            .defines = {}});

    // copy data from host to L1 directly
    detail::WriteToBuffer(a, a_data);
    detail::WriteToBuffer(b, b_data);

    for (int i = 0; i < test_config.num_cores; ++i) {
        // Set runtime arguments for each core.
        CoreCoord core = {0, i};
        SetRuntimeArgs(program, compute, core, {test_config.num_tiles_per_core});
    }

    CommandQueue& cq = device->command_queue();
    // Enqueue the program
    EnqueueProgram(cq, program, true);

    std::cout << "Kernel execution finished" << std::endl;

    // Read the output buffer.
    std::vector<bfloat16> c_data;
    detail::ReadFromBuffer(c, c_data);

    // Print partial results so we can see the output is correct (plus or minus
    // some error due to BFP16 precision)
    std::cout << "Partial results: (note we are running under BFP16. It's going "
                 "to be less accurate)\n";
    size_t element_per_core = constants::TILE_HW * test_config.num_tiles_per_core;
    size_t print_per_core = std::min((size_t)10, element_per_core);

    for (int core = 0; core < test_config.num_cores; ++core) {
        const auto core_offset = core * element_per_core;
        std::cout << "Core (0, " << core << "):\n";
        for (int index = 0; index < print_per_core; index++) {
            const auto i = core_offset + index;
            std::cout << "index  " << i << "   " << a_data[i].to_float() << " + " << b_data[i].to_float() << " = "
                      << c_data[i].to_float() << "\n";
        }
        std::cout << std::endl;
    }
    std::cout << std::flush;

    // Finally, we close the device.
    CloseDevice(device);
    return 0;
}
