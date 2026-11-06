// SPDX-FileCopyrightText: © 2024 Martin Chang
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
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

using namespace tt::tt_metal;

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;

// Describe a tensor on DRAM or SRAM that holds n_tiles tiles of BFP16 data. The BFP16 values are stored in pairs,
// packed into uint32 words. Each row of a row-major tensor is one page: one tile on DRAM, all tiles on SRAM.
TensorSpec MakeTensorSpecBFP16(uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    const uint32_t page_size = page_tiles * tile_size;
    return TensorSpec(
        Shape({n_tiles / page_tiles, page_size / static_cast<uint32_t>(sizeof(uint32_t))}),
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, (sram ? BufferType::L1 : BufferType::DRAM)}));
}

// Allocate a tensor on DRAM or SRAM. Assuming the tensor holds BFP16 data.
// A tile on Tenstorrent is 32x32 elements, given us using BFP16, we need 2 bytes per element.
// Making the tile size 32x32x2 = 2048 bytes.
// @param device: The device to allocate the tensor on.
// @param n_tiles: The number of tiles to allocate.
// @param sram: If true, allocate the tensor on SRAM, otherwise allocate it on DRAM.
// The buffer is allocated from the top of memory (bottom_up = false) and then wrapped in a tensor.
MeshTensor MakeTensorBFP16(distributed::MeshDevice& mesh_device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    const distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_tiles * tile_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM),
        .bottom_up = false};
    const distributed::ReplicatedBufferConfig buffer_config{.size = tile_size * n_tiles};
    return MeshTensor::from_buffer(
        std::move(*distributed::MeshBuffer::create(buffer_config, device_local_config, &mesh_device)),
        MakeTensorSpecBFP16(n_tiles, sram));
}

experimental::DataflowBufferSpec MakeDataflowBuffer(
    const experimental::DFBSpecName& dfb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    return experimental::DataflowBufferSpec{
        .unique_id = dfb, .entry_size = page_size, .num_entries = size / page_size, .data_format_metadata = format};
}

// Dataflow buffers are Tenstorrent's way of communicating between the data movement and the compute kernels.
// kernels queue tiles into the dataflow buffer and takes them when they are ready. The dataflow buffer is
// backed by SRAM. There can be multiple dataflow buffers on a single Tensix core.
// A dataflow buffer is created on the cores of the kernels that produce into it and consume from it.
// @param dfb: Which dataflow buffer to describe (in0, in1, out0, etc..). This is just an ID
// @param n_tiles: The number of tiles the dataflow buffer can hold.
experimental::DataflowBufferSpec MakeDataflowBufferBFP16(const experimental::DFBSpecName& dfb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    return MakeDataflowBuffer(dfb, n_tiles * tile_size, tile_size, tt::DataFormat::Float16_b);
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
    std::cout << "This program demonstrates how to add two vectors using tt-Metalium.\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --device, -d <device_id>  Specify the device to run the program on. Default is 0.\n";
    std::cout << "  --seed, -s <seed>         Specify the seed for the random number generator. Default is random.\n";
    exit(0);
}

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main(int argc, char** argv) {
    int seed = std::random_device{}();
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

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    distributed::MeshWorkload workload;
    // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
    experimental::NodeCoord core = {0, 0};

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    const uint32_t n_tiles = 64;
    const uint32_t tile_size = TILE_WIDTH * TILE_HEIGHT;
    // Create 3 tensors on DRAM. These will hold the input and output data. A and B are the input tensors, C is the
    // output tensor.
    MeshTensor a = MakeTensorBFP16(*mesh_device, n_tiles, false);
    MeshTensor b = MakeTensorBFP16(*mesh_device, n_tiles, false);
    MeshTensor c = MakeTensorBFP16(*mesh_device, n_tiles, false);

    std::mt19937 rng(seed);
    std::vector<uint32_t> a_data = create_random_vector_of_bfloat16(tile_size * n_tiles * 2, 10, rng());
    std::vector<uint32_t> b_data = create_random_vector_of_bfloat16(tile_size * n_tiles * 2, 10, rng());

    // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
    const experimental::KernelSpecName reader{"reader"};
    const experimental::KernelSpecName writer{"writer"};
    const experimental::KernelSpecName compute{"compute"};
    const experimental::DFBSpecName in0_dfb{"in0"};
    const experimental::DFBSpecName in1_dfb{"in1"};
    const experimental::DFBSpecName out0_dfb{"out0"};
    const experimental::TensorParamName a_tensor_param{"a"};
    const experimental::TensorParamName b_tensor_param{"b"};
    const experimental::TensorParamName c_tensor_param{"c"};

    const uint32_t tiles_per_cb = 4;
    // Describe 3 dataflow buffers. These will be used by the data movement kernels to stream data into the compute
    // cores and for the compute cores to stream data out.
    experimental::DataflowBufferSpec in0_dfb_spec = MakeDataflowBufferBFP16(in0_dfb, tiles_per_cb);
    experimental::DataflowBufferSpec in1_dfb_spec = MakeDataflowBufferBFP16(in1_dfb, tiles_per_cb);
    experimental::DataflowBufferSpec out0_dfb_spec = MakeDataflowBufferBFP16(out0_dfb, tiles_per_cb);

    // We're writing to tensors on a 1x1 MeshDevice, so the data goes to the device at MeshCoordinate 0, 0.
    // The writes do not block. The host tensors must stay alive until the writes are done.
    HostTensor a_host = HostTensor::from_vector(a_data, a.tensor_spec());
    HostTensor b_host = HostTensor::from_vector(b_data, b.tensor_spec());
    cq.enqueue_write_tensor(a_host, a);
    cq.enqueue_write_tensor(b_host, b);

    // A Tensix core is made up with 5 processors. 2 data movement processors, and 3 compute processors. The 2 data
    // movement processors act independent to other cores. And the 3 compute processors act together (hence 1 kernel for
    // compute). There is no need to explicitly parallelize the compute kernels. Unlike traditional CPU/GPU style SPMD
    // programming, the 3 compute processors moves data from SRAM into the FPU(tensor engine)/SFPU(SIMD engine),
    // operates on the data, and move it back to SRAM. The data movement processors moves data from the NoC, or in our
    // case, the DRAM, into the SRAM.
    //
    // The vector add example consists of 3 kernels. `interleaved_tile_read` reads tiles from the input tensors A and B
    // into 2 dataflow buffers. `add` reads tiles from the dataflow buffers, adds them together, and dumps the result
    // into a third dataflow buffer. `tile_write` reads tiles from the third dataflow buffer and writes them to the
    // output tensor C.
    experimental::KernelSpec reader_spec{
        .unique_id = reader,
        .source = OVERRIDE_KERNEL_PREFIX "contributed/vecadd/kernels/interleaved_tile_read.cpp",
        .dfb_bindings = {experimental::ProducerOf(in0_dfb, "in0"), experimental::ProducerOf(in1_dfb, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = a_tensor_param, .accessor_name = "a"},
             {.tensor_parameter_name = b_tensor_param, .accessor_name = "b"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };
    experimental::KernelSpec writer_spec{
        .unique_id = writer,
        .source = OVERRIDE_KERNEL_PREFIX "contributed/vecadd/kernels/tile_write.cpp",
        .dfb_bindings = {experimental::ConsumerOf(out0_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = c_tensor_param, .accessor_name = "c"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };
    experimental::KernelSpec compute_spec{
        .unique_id = compute,
        .source = OVERRIDE_KERNEL_PREFIX "contributed/vecadd/kernels/add.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(in0_dfb, "in0"),
             experimental::ConsumerOf(in1_dfb, "in1"),
             experimental::ProducerOf(out0_dfb, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config = experimental::ComputeHardwareConfig{.sfpu_precision_mode = Precision::Precise},
    };

    // A program spec describes the program: its kernels, the dataflow buffers between them, the tensors they
    // operate on, and the cores the kernels run on (the work units).
    experimental::ProgramSpec program_spec{
        .name = "vecadd",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in0_dfb_spec, in1_dfb_spec, out0_dfb_spec},
        .tensor_parameters =
            {{.unique_id = a_tensor_param, .spec = a.tensor_spec()},
             {.unique_id = b_tensor_param, .spec = b.tensor_spec()},
             {.unique_id = c_tensor_param, .spec = c.tensor_spec()}},
        .work_units = {{.name = "vecadd", .kernels = {reader, writer, compute}, .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // Set the runtime arguments for the kernels, and the tensors bound to the tensor parameters.
    experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = reader,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = writer,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = compute,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})}},
        .tensor_args =
            {{a_tensor_param, experimental::TensorArgument{a}},
             {b_tensor_param, experimental::TensorArgument{b}},
             {c_tensor_param, experimental::TensorArgument{c}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

    // We have setup the program. Now we can add it to the workload and queue it for execution.
    // The last argument to EnqueueMeshWorkload is a boolean that specifies whether
    // we wait for the program to finish execution before returning. I've set
    // it to true. But alternatively, you can set it to false and call
    // `Finish(cq)` to wait for all programs to finish.
    // But it shouldn't matter in this case since we block on reading the output
    // tensor.
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    // distributed::Finish(cq);
    std::cout << "Kernel execution finished" << std::endl;

    // Read the output tensor.
    // We're reading from the tensor on Device Coordinate 0, 0, since this is a 1x1 MeshDevice.
    // The read blocks until the data is on the host.
    std::vector<uint32_t> c_data = cq.enqueue_read_tensor(c, /*blocking=*/true).to_vector<uint32_t>();

    // Print partial results so we can see the output is correct (plus or minus some error due to BFP16 precision)
    std::cout << "Partial results: (note we are running under BFP16. It's going to be less accurate)\n";
    size_t n = std::min((size_t)10, (size_t)tile_size * n_tiles);
    bfloat16* a_bf16 = reinterpret_cast<bfloat16*>(a_data.data());
    bfloat16* b_bf16 = reinterpret_cast<bfloat16*>(b_data.data());
    bfloat16* c_bf16 = reinterpret_cast<bfloat16*>(c_data.data());
    for (int i = 0; i < n; i++) {
        std::cout << "  " << static_cast<float>(a_bf16[i]) << " + " << static_cast<float>(b_bf16[i]) << " = "
                  << static_cast<float>(c_bf16[i]) << "\n";
    }
    std::cout << std::flush;

    // Finally, we close the device.
    mesh_device->close();
    return 0;
}
