// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This programming example is an advanced example, compared to the vecadd single core example in the
// contributed folder.  It illustrated sharding tensor inputs to L1 memory of multiple cores directly,
// then perform vector addition tile by tile. Because of sharding to L1, DRAM nor NoC is not involved.
// Data copy is avoided and reader and writer kernels are not needed.

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/constants.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt_stl/fmt.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

using namespace tt::tt_metal;

struct DistributionConfig {
    DistributionConfig(TensorMemoryLayout layout, uint32_t num_cores_y, uint32_t num_cores_x) :
        layout(layout), num_cores_y(num_cores_y), num_cores_x(num_cores_x) {}

    TensorMemoryLayout layout;
    uint32_t num_cores_y;
    uint32_t num_cores_x;
};

// The tensor is a grid of n_tiles_y x n_tiles_x pages. Each page holds one tile worth of elements
// (TILE_HW * sizeof(bfloat16) bytes), and each row of a row-major tensor is one page.
TensorSpec MakeShardedL1TensorSpecBFP16(uint32_t n_tiles_y, uint32_t n_tiles_x, const NdShardSpec& shard_config) {
    return TensorSpec(
        Shape({n_tiles_y, n_tiles_x, tt::constants::TILE_HW}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), MemoryConfig(BufferType::L1, shard_config)));
}

experimental::DataflowBufferSpec MakeDataflowBufferBFP16(
    const experimental::DFBSpecName& name, uint32_t n_tiles, const experimental::TensorParamName& l1_tensor) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_HW;
    return experimental::DataflowBufferSpec{
        .unique_id = name,
        .entry_size = tile_size,
        .num_entries = n_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
        // IMPORTANT: assign L1 buffer address to dataflow buffer directly so that
        // no extra allocation and data copy
        .borrowed_from = l1_tensor,
    };
}

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
    // Sharding is quite precise and requires exact division of the data across the cores. The example tries to
    // distribute 64 (4x4) tiles across 4 cores. But in different sharding modes:
    // * height: Shard the tensors across the height dimension
    // * width: Shard the tensors across the width dimension
    // * block: Shard the tensors across both height and width dimensions
    //
    // In different modes, the data is distributed differently across the cores in different patterns. Each core:
    // * height: Uses 4 cores in the y direction. Each core gets 1 rows of 4 tiles each.
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

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const auto& config = test_configs.at(sharding_type);

    // In this example we will distribute 16 tiles (4x4) across 4 cores.
    constexpr uint32_t n_tiles_y = 4;
    constexpr uint32_t n_tiles_x = 4;

    // Calculate the number of tiles per core in each dimension based on the configuration.
    const uint32_t num_tiles_per_core_x = n_tiles_x / config.num_cores_x;
    const uint32_t num_tiles_per_core_y = n_tiles_y / config.num_cores_y;

    fmt::print(
        "Sharding {}x{} tiles to {}x{} cores in {} mode\n",
        n_tiles_y,
        n_tiles_x,
        config.num_cores_y,
        config.num_cores_x,
        config.layout);
    fmt::print("Each core will handle {}x{} tiles\n\n", num_tiles_per_core_y, num_tiles_per_core_x);

    // Construct the L1 configuration. The configuration is very detailed to enable maximum performance.
    // The cores that the buffer will be sharded across.
    const CoreRange cores(CoreCoord(0, 0), CoreCoord(config.num_cores_y - 1, config.num_cores_x - 1));
    // The tiles each core holds. Height sharding gives each core whole rows of tiles; width and block sharding give
    // each core a block of num_tiles_per_core_x rows by num_tiles_per_core_y columns of tiles.
    const bool height_sharded = config.layout == TensorMemoryLayout::HEIGHT_SHARDED;
    const uint32_t shard_tiles_y = height_sharded ? num_tiles_per_core_y : num_tiles_per_core_x;
    const uint32_t shard_tiles_x = height_sharded ? num_tiles_per_core_x : num_tiles_per_core_y;
    // The sharing configuration for the tensor
    NdShardSpec spec{
        // shard_shape: The size of each shard in the tensor, in (rows of tiles, columns of tiles, elements per tile)
        .shard_shape = Shape({shard_tiles_y, shard_tiles_x, tt::constants::TILE_HW}),
        // grid: The set of cores to shard the tensor across
        .grid = CoreRangeSet(cores),
        // orientation: The orientation of the shards in the tensor
        .orientation = ShardOrientation::ROW_MAJOR,
    };

    // Create the input and output tensors that lives on L1(SRAM)
    TensorSpec l1_tensor_spec = MakeShardedL1TensorSpecBFP16(n_tiles_y, n_tiles_x, spec);
    MeshTensor a = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);
    MeshTensor b = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);
    MeshTensor c = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);

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

    // Names identify the kernel, the dataflow buffers and the tensor parameters within the program spec.
    const experimental::KernelSpecName compute{"compute"};
    const experimental::DFBSpecName dfb_a{"a"};
    const experimental::DFBSpecName dfb_b{"b"};
    const experimental::DFBSpecName dfb_c{"c"};
    const experimental::TensorParamName a_param{"a"};
    const experimental::TensorParamName b_param{"b"};
    const experimental::TensorParamName c_param{"c"};

    size_t num_tiles_per_core = n_tiles_x * n_tiles_y / CoreRangeSet(cores).num_cores();
    // Create dataflow buffers so the compute APIs can access the data.
    // NOTE: These are special dataflow buffers that borrow the L1 memory of the tensors. As data already fully
    // resides in L1, we can simply point the dataflow buffers to the L1 tensors and avoid any extra allocation or data
    // copy. This is an important optimization for performance. But hyper specific to use cases like this one.
    experimental::DataflowBufferSpec dfb_a_spec = MakeDataflowBufferBFP16(dfb_a, num_tiles_per_core, a_param);
    experimental::DataflowBufferSpec dfb_b_spec = MakeDataflowBufferBFP16(dfb_b, num_tiles_per_core, b_param);
    experimental::DataflowBufferSpec dfb_c_spec = MakeDataflowBufferBFP16(dfb_c, num_tiles_per_core, c_param);

    // Create the kernel
    // There is no other kernel, so the compute kernel is both the producer and the consumer of each dataflow buffer.
    experimental::KernelSpec compute_spec{
        .unique_id = compute,
        .source = "tt_metal/programming_examples/vecadd_sharding/kernels/add_sharding.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ProducerOf(dfb_a, "in0"),
             experimental::ConsumerOf(dfb_a, "in0"),
             experimental::ProducerOf(dfb_b, "in1"),
             experimental::ConsumerOf(dfb_b, "in1"),
             experimental::ProducerOf(dfb_c, "out0"),
             experimental::ConsumerOf(dfb_c, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tile"}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .sfpu_precision_mode = Precision::Precise,
            },
    };

    experimental::ProgramSpec program_spec{
        .name = "vecadd_sharding",
        .kernels = {compute_spec},
        .dataflow_buffers = {dfb_a_spec, dfb_b_spec, dfb_c_spec},
        .tensor_parameters =
            {{.unique_id = a_param, .spec = l1_tensor_spec},
             {.unique_id = b_param, .spec = l1_tensor_spec},
             {.unique_id = c_param, .spec = l1_tensor_spec}},
        .work_units = {{.name = "vecadd_sharding", .kernels = {compute}, .target_nodes = cores}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // copy data from host to L1 directly
    // The host tensors must stay alive until the non-blocking writes are done.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    HostTensor a_host = HostTensor::from_vector(a_data, l1_tensor_spec);
    HostTensor b_host = HostTensor::from_vector(b_data, l1_tensor_spec);
    cq.enqueue_write_tensor(a_host, a);
    cq.enqueue_write_tensor(b_host, b);

    // Setup arguments and run the program.
    experimental::KernelRunArgs compute_run_args{.kernel = compute};
    for (const CoreCoord& core : cores) {
        experimental::AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, core, {{"num_tile", num_tiles_per_core}});
    }
    experimental::ProgramRunArgs run_args{
        .kernel_run_args = {compute_run_args},
        .tensor_args =
            {{a_param, experimental::TensorArgument{a}},
             {b_param, experimental::TensorArgument{b}},
             {c_param, experimental::TensorArgument{c}}},
    };
    experimental::SetProgramRunArgs(program, run_args);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    fmt::print("Kernel execution finished. Reading results...\n");

    // Read the output tensor.
    std::vector<bfloat16> c_data = cq.enqueue_read_tensor(c, /*blocking=*/true).to_vector<bfloat16>();

    // Print partial results so we can see the output is correct (plus or minus
    // some error due to BFP16 precision)
    fmt::print("Partial results: (note we are running under BFP16. It's going to be less accurate)\n");
    size_t element_per_core = tt::constants::TILE_HW * num_tiles_per_core;
    size_t print_per_core = std::min((size_t)10, element_per_core);

    int core_idx = 0;
    for ([[maybe_unused]] auto& core : cores) {
        const auto core_offset = core_idx * element_per_core;
        fmt::print("Core {}:\n", core_idx);
        for (int index = 0; index < print_per_core; index++) {
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

    // Finally, we close the device.
    mesh_device->close();
    return 0;
}
