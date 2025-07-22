// SPDX-FileCopyrightText: © 2025 TenstorrentAI ULC
//
// SPDX-License-Identifier: Apache-2.0

// this programing example is based on the vecadd single core example in the
// contributed folder it illustarted using multiple cores to perform vector
// addition the program will use 4 cores to perform the vector addition
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include <algorithm>

using namespace tt;
using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, bool sram) {
    InterleavedBufferConfig config{
        .device = device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)};
    return CreateBuffer(config);
}

// Allocate a buffer on DRAM or SRAM. Assuming the buffer holds BFP16 data.
// A tile on Tenstorrent is 32x32 elements, given us using BFP16, we need 2
// bytes per element. Making the tile size 32x32x2 = 2048 bytes.
// @param device: The device to allocate the buffer on.
// @param n_tiles: The number of tiles to allocate.
// @param sram: If true, allocate the buffer on SRAM, otherwise allocate it on
// DRAM.
std::shared_ptr<Buffer> MakeBufferBFP16(IDevice* device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    // For simplicity, all DRAM buffers have page size = tile size.
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

// Circular buffers are Tenstorrent's way of communicating between the data
// movement and the compute kernels. kernels queue tiles into the circular
// buffer and takes them when they are ready. The circular buffer is backed by
// SRAM. There can be multiple circular buffers on a single Tensix core.
// @param program: The program to create the circular buffer on.
// @param core: The core to create the circular buffer on.
// @param cb: Which circular buffer to create (c_in0, c_in1, c_out0, c_out1,
// etc..). This is just an ID
// @param n_tiles: The number of tiles the circular buffer can hold.
CBHandle MakeCircularBufferBFP16(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float16_b);
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
    std::cout << "  --seed, -s <seed>         Specify the seed for the random "
                 "number generator. Default is random.\n";
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

    auto* device = CreateDevice(device_id);
    Program program = CreateProgram();

    CommandQueue& cq = device->command_queue();

    const uint32_t tile_size = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    std::vector<uint32_t> tiles_per_core;
    const uint32_t core_to_print = 4;

    // Create 3 buffers on DRAM. These will hold the input and output data. A
    // and B are the input buffers, C is the output buffer.
    auto a = MakeBufferBFP16(device, n_tiles, false);
    auto b = MakeBufferBFP16(device, n_tiles, false);
    auto c = MakeBufferBFP16(device, n_tiles, false);

    std::mt19937 rng(seed);
    std::vector<bfloat16> a_data = create_random_vector_of_bfloat16_native(tile_size * n_tiles * 2, 10, rng());
    std::vector<bfloat16> b_data = create_random_vector_of_bfloat16_native(tile_size * n_tiles * 2, 10, rng());

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    const uint32_t cir_buf_num_title = 4;
    MakeCircularBufferBFP16(program, all_device_cores, tt::CBIndex::c_0, cir_buf_num_title);
    MakeCircularBufferBFP16(program, all_device_cores, tt::CBIndex::c_1, cir_buf_num_title);
    MakeCircularBufferBFP16(program, all_device_cores, tt::CBIndex::c_2, cir_buf_num_title);

    // A Tensix core is made up with 5 processors. 2 data movement processors,
    // and 3 compute processors. The 2 data movement processors act independent
    // to other cores. And the 3 compute processors act together (hence 1 kerenl
    // for compute). There is no need to explicitly parallelize the compute
    // kernels. Unlike traditional CPU/GPU style SPMD programming, the 3 compute
    // processors moves data from SRAM into the FPU(tensor engine)/SFPU(SIMD
    // engine), operates on the data, and move it back to SRAM. The data
    // movement processors moves data from the NoC, or in our case, the DRAM,
    // into the SRAM.
    //
    // The vector add example consists of 3 kernels. `interleaved_tile_read`
    // reads tiles from the input buffers A and B into 2 circular buffers. `add`
    // reads tiles from the circular buffers, adds them together, and dumps the
    // result into a third circular buffer. `tile_write` reads tiles from the
    // third circular buffer and writes them to the output buffer C.
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1};
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)tt::CBIndex::c_2};
    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1, (std::uint32_t)tt::CBIndex::c_2};

    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/vecadd_multi_core/kernels/"
        "interleaved_tile_read_multi_core.cpp",
        all_device_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/vecadd_multi_core/kernels/"
        "tile_write_multi_core.cpp",
        all_device_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/vecadd_multi_core/"
        "kernels/add_multi_core.cpp",
        all_device_cores,
        ComputeConfig{.math_approx_mode = false, .compile_args = compute_compile_time_args, .defines = {}});

    constexpr bool row_major = true;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, n_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            SetRuntimeArgs(program, reader, core, std::array<uint32_t, 10>{0});
            SetRuntimeArgs(program, writer, core, std::array<uint32_t, 11>{0});
            SetRuntimeArgs(program, compute, core, std::array<uint32_t, 3>{0});
            continue;
        }
        if (i < core_to_print) {
            tiles_per_core.push_back(num_tiles_per_core);
        }
        SetRuntimeArgs(program, reader, core, {a->address(), b->address(), num_tiles_per_core, start_tile_id});
        SetRuntimeArgs(program, writer, core, {c->address(), num_tiles_per_core, start_tile_id});
        SetRuntimeArgs(program, compute, core, {num_tiles_per_core, start_tile_id});
        start_tile_id += num_tiles_per_core;
    }

    EnqueueWriteBuffer(cq, a, a_data, false);
    EnqueueWriteBuffer(cq, b, b_data, false);
    // Enqueue the program
    EnqueueProgram(cq, program, true);

    std::cout << "Kernel execution finished" << std::endl;

    // Read the output buffer.
    std::vector<bfloat16> c_data;
    EnqueueReadBuffer(cq, c, c_data, true);

    // Print partial results so we can see the output is correct (plus or minus
    // some error due to BFP16 precision)
    std::cout << "Partial results: (note we are running under BFP16. It's going "
                 "to be less accurate)\n";
    auto core_offset = 0;
    for (int core_index = 0; core_index < std::min(core_to_print, num_cores_total); ++core_index) {
        core_offset += core_index * tile_size * tiles_per_core[core_index];
        std::cout << "Core (0, " << core_index << "):\n";
        for (int index = 0; index < 10; index++) {
            const auto i = core_offset + index;
            std::cout << "index  " << i << "  " << a_data[i].to_float() << " + " << b_data[i].to_float() << " = "
                      << c_data[i].to_float() << "\n";
        }
        std::cout << std::endl;
    }
    std::cout << std::flush;

    // Finally, we close the device.
    CloseDevice(device);
    return 0;
}
