// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// this programming example is based on the vecadd single core example in the
// contributed folder it illustarted using multiple cores to perform vector
// addition the program will use 4 cores to perform the vector addition
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

// Allocate a row-major BFP16 tensor of `size` bytes. Each row of a row-major tensor is one page of `page_size` bytes.
MeshTensor MakeMeshTensor(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t size, uint32_t page_size, bool sram) {
    TensorSpec spec(
        Shape({size / page_size, static_cast<uint32_t>(page_size / sizeof(bfloat16))}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, (sram ? BufferType::L1 : BufferType::DRAM)}));
    return MeshTensor::allocate_on_device(*mesh_device, spec);
}

// Allocate a tensor on DRAM or SRAM. Assuming the tensor holds BFP16 data.
// A tile on Tenstorrent is 32x32 elements, given us using BFP16, we need 2
// bytes per element. Making the tile size 32x32x2 = 2048 bytes.
// @param device: The device to allocate the tensor on.
// @param n_tiles: The number of tiles to allocate.
// @param sram: If true, allocate the tensor on SRAM, otherwise allocate it on
// DRAM.
MeshTensor MakeMeshTensorBFP16(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    // For simplicity, all DRAM tensors have page size = tile size.
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeMeshTensor(mesh_device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

experimental::DataflowBufferSpec MakeDataflowBuffer(
    const experimental::DFBSpecName& name, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    return experimental::DataflowBufferSpec{
        .unique_id = name,
        .entry_size = page_size,
        .num_entries = size / page_size,
        .data_format_metadata = format,
    };
}

// Dataflow buffers are Tenstorrent's way of communicating between the data
// movement and the compute kernels. kernels queue tiles into the dataflow
// buffer and takes them when they are ready. The dataflow buffer is backed by
// SRAM. There can be multiple dataflow buffers on a single Tensix core.
// A dataflow buffer is created on every core that runs a kernel bound to it.
// @param name: Which dataflow buffer to describe. This is just an ID that the
// kernel specs refer to.
// @param n_tiles: The number of tiles the dataflow buffer can hold.
experimental::DataflowBufferSpec MakeDataflowBufferBFP16(const experimental::DFBSpecName& name, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    return MakeDataflowBuffer(name, n_tiles * tile_size, tile_size, tt::DataFormat::Float16_b);
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
    fmt::print("This program demonstrates how to add two vectors using tt-Metalium.\n\n");
    fmt::print("Options:\n");
    fmt::print("  --device, -d <device_id>  Specify the device to run the program on. Default is 0.\n");
    fmt::print("  --seed, -s <seed>         Specify the seed for the random number generator. Default is random.\n");
    exit(0);
}

int main(int argc, char** argv) {
    int seed = 0x1234567;
    int device_id = 0;

    // Quick and dirty argument parsing.
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "--device" || arg == "-d") {
            device_id = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--seed" || arg == "-s") {
            seed = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--help" || arg == "-h") {
            help(argv[0]);
            return 0;
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
            help(argv[0]);
        }
    }
    // n_tiles is number of tiles of data for this programming example to add two vectors
    const uint32_t n_tiles = 640;

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    const uint32_t tile_size = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    std::map<CoreCoord, uint32_t> core_tile_idx;

    // Create 3 tensors on DRAM. These will hold the input and output data. A
    // and B are the input tensors, C is the output tensor.
    MeshTensor a = MakeMeshTensorBFP16(mesh_device, n_tiles, false);
    MeshTensor b = MakeMeshTensorBFP16(mesh_device, n_tiles, false);
    MeshTensor c = MakeMeshTensorBFP16(mesh_device, n_tiles, false);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, 10.0f);
    std::vector<bfloat16> a_data(tile_size * n_tiles);
    std::vector<bfloat16> b_data(tile_size * n_tiles);
    for (uint32_t i = 0; i < tile_size * n_tiles; i++) {
        a_data[i] = bfloat16(dist(rng));
        b_data[i] = bfloat16(dist(rng));
    }

    auto core_grid = mesh_device->compute_with_storage_grid_size();
    uint32_t num_cores_y = core_grid.y;

    // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
    const experimental::KernelSpecName reader{"reader"};
    const experimental::KernelSpecName writer{"writer"};
    const experimental::KernelSpecName compute{"compute"};
    const experimental::DFBSpecName dfb_0{"dfb_0"};
    const experimental::DFBSpecName dfb_1{"dfb_1"};
    const experimental::DFBSpecName dfb_2{"dfb_2"};
    const experimental::TensorParamName a_param{"a"};
    const experimental::TensorParamName b_param{"b"};
    const experimental::TensorParamName c_param{"c"};

    const uint32_t cir_buf_num_title = 4;
    experimental::DataflowBufferSpec dfb_0_spec = MakeDataflowBufferBFP16(dfb_0, cir_buf_num_title);
    experimental::DataflowBufferSpec dfb_1_spec = MakeDataflowBufferBFP16(dfb_1, cir_buf_num_title);
    experimental::DataflowBufferSpec dfb_2_spec = MakeDataflowBufferBFP16(dfb_2, cir_buf_num_title);

    // Calculate the work distribution across the cores. The work is split into 2 groups of cores. Primary and
    // secondary. The primary group will process more tiles pre core than the secondary group, in case the number of
    // work is not evenly divisible by the number of available cores. This function guarantees that the work is exactly
    // split across the cores. No more, no less.
    constexpr bool row_major =
        true;  // Distribute the work in row-major order (across the grid of cores). For this application, it doesn't
               // matter much But it may for kernels with more complex data access patterns.
    auto
        [num_cores,                   // number of cores utilized
         all_cores,                   // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(core_grid, n_tiles, row_major);

    // A Tensix core is made up with 5 processors. 2 data movement processors,
    // and 3 compute processors. The 2 data movement processors act independent
    // to other cores. And the 3 compute processors act together (hence 1 kernel
    // for compute). There is no need to explicitly parallelize the compute
    // kernels. Unlike traditional CPU/GPU style SPMD programming, the 3 compute
    // processors moves data from SRAM into the FPU(tensor engine)/SFPU(SIMD
    // engine), operates on the data, and move it back to SRAM. The data
    // movement processors moves data from the NoC, or in our case, the DRAM,
    // into the SRAM.
    //
    // Though the multi-Tensix parallelization strategy is SPMD for this example
    //
    // The vector add example consists of 3 kernels. `interleaved_tile_read`
    // reads tiles from the input tensors A and B into 2 dataflow buffers. `add`
    // reads tiles from the dataflow buffers, adds them together, and dumps the
    // result into a third dataflow buffer. `tile_write` reads tiles from the
    // third dataflow buffer and writes them to the output tensor C.
    experimental::KernelSpec reader_spec{
        .unique_id = reader,
        .source =
            "tt_metal/programming_examples/vecadd_multi_core/kernels/"
            "interleaved_tile_read_multi_core.cpp",
        .dfb_bindings = {experimental::ProducerOf(dfb_0, "in0"), experimental::ProducerOf(dfb_1, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = a_param, .accessor_name = "a"},
             {.tensor_parameter_name = b_param, .accessor_name = "b"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles", "start_tile_id"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };
    experimental::KernelSpec writer_spec{
        .unique_id = writer,
        .source =
            "tt_metal/programming_examples/vecadd_multi_core/kernels/"
            "tile_write_multi_core.cpp",
        .dfb_bindings = {experimental::ConsumerOf(dfb_2, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = c_param, .accessor_name = "c"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles", "start_tile_id"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };
    experimental::KernelSpec compute_spec{
        .unique_id = compute,
        .source =
            "tt_metal/programming_examples/vecadd_multi_core/"
            "kernels/add_multi_core.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(dfb_0, "in0"),
             experimental::ConsumerOf(dfb_1, "in1"),
             experimental::ProducerOf(dfb_2, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles", "start_tile_id"}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .sfpu_precision_mode = Precision::Precise,
            },
    };

    // The program spec describes the kernels, the dataflow buffers between them, the tensors they operate on, and the
    // cores the kernels run on.
    experimental::ProgramSpec program_spec{
        .name = "vecadd_multi_core",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {dfb_0_spec, dfb_1_spec, dfb_2_spec},
        .tensor_parameters =
            {{.unique_id = a_param, .spec = a.tensor_spec()},
             {.unique_id = b_param, .spec = b.tensor_spec()},
             {.unique_id = c_param, .spec = c.tensor_spec()}},
        .work_units = {{.name = "vecadd", .kernels = {reader, writer, compute}, .target_nodes = all_cores}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};
    // Set the runtime arguments for each core in the work groups. Give them a different starting and ending point to
    // achieve SPMD
    experimental::KernelRunArgs reader_run_args{.kernel = reader};
    experimental::KernelRunArgs writer_run_args{.kernel = writer};
    experimental::KernelRunArgs compute_run_args{.kernel = compute};
    uint32_t start_tile_id = 0;
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                experimental::AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"n_tiles", work_per_core}, {"start_tile_id", start_tile_id}});
                experimental::AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"n_tiles", work_per_core}, {"start_tile_id", start_tile_id}});
                experimental::AddRuntimeArgsForNode(
                    compute_run_args.runtime_arg_values,
                    core,
                    {{"n_tiles", work_per_core}, {"start_tile_id", start_tile_id}});
                core_tile_idx[core] = start_tile_id;  // Save the mapping so we can print the results later.

                start_tile_id += work_per_core;
            }
        }
    }
    // The tensors are bound to the tensor parameters along with the runtime arguments.
    experimental::ProgramRunArgs run_args{
        .kernel_run_args = {reader_run_args, writer_run_args, compute_run_args},
        .tensor_args =
            {{a_param, experimental::TensorArgument{a}},
             {b_param, experimental::TensorArgument{b}},
             {c_param, experimental::TensorArgument{c}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

    // The host tensors must stay alive until the non-blocking writes are done.
    HostTensor a_host = HostTensor::from_vector(a_data, a.tensor_spec());
    HostTensor b_host = HostTensor::from_vector(b_data, b.tensor_spec());
    cq.enqueue_write_tensor(a_host, a);
    cq.enqueue_write_tensor(b_host, b);
    // Enqueue the program
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::cout << "Kernel execution finished" << std::endl;

    // Read the output tensor.
    std::vector<bfloat16> c_data = cq.enqueue_read_tensor(c, /*blocking=*/true).to_vector<bfloat16>();

    // Print partial results so we can see the output is correct (plus or minus
    // some error due to BFP16 precision)
    std::cout << "Partial results: (note we are running under BFP16. It's going "
                 "to be less accurate)\n";
    // Print some results from some of the cores to illustrate the computation
    for (uint32_t core_y = 0; core_y < num_cores_y; core_y++) {
        CoreCoord core(0, core_y);
        if (!core_tile_idx.contains(core)) {
            continue;  // This core did not participate in the computation.
        }
        fmt::print("Core {}:\n", core);

        auto start_idx = core_tile_idx.at(core) * tile_size;
        for (uint32_t i = 0; i < 10; i++) {
            fmt::print(
                "Index {}: {} + {} = {}\n",
                start_idx + i,
                static_cast<float>(a_data[start_idx + i]),
                static_cast<float>(b_data[start_idx + i]),
                static_cast<float>(c_data[start_idx + i]));
        }
        fmt::print("\n");
    }

    // Check if the results match the expected values.
    bool pass = true;
    for (size_t i = 0; i < c_data.size(); i++) {
        float expected = static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]);
        if (std::abs(static_cast<float>(c_data[i]) - expected) > 0.3f) {  // Allow some tolerance due to BFP16 precision
            fmt::print(
                "Mismatch at index {}: {} + {} = {}, expected {}\n",
                i,
                static_cast<float>(a_data[i]),
                static_cast<float>(b_data[i]),
                static_cast<float>(c_data[i]),
                expected);
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
