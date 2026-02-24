// SPDX-FileCopyrightText: © 2024 Martin Chang
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

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

    // DeviceContext wraps MeshDevice creation, command queue, and teardown in RAII.
    DeviceContext ctx(device_id);

    // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
    CoreCoord core = {0, 0};

    const uint32_t n_tiles = 64;
    const uint32_t tile_size = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

    // Create 3 DRAM tile buffers: A and B are inputs, C is the output.
    // A tile on Tenstorrent is 32x32 elements; with BFloat16, each tile occupies 2048 bytes.
    auto a = ctx.dram_tile_buffer(n_tiles);
    auto b = ctx.dram_tile_buffer(n_tiles);
    auto c = ctx.dram_tile_buffer(n_tiles);

    std::mt19937 rng(seed);
    std::vector<uint32_t> a_data = create_random_vector_of_bfloat16(tile_size * n_tiles * 2, 10, rng());
    std::vector<uint32_t> b_data = create_random_vector_of_bfloat16(tile_size * n_tiles * 2, 10, rng());

    // A Tensix core is made up with 5 processors. 2 data movement processors, and 3 compute processors. The 2 data
    // movement processors act independent to other cores. And the 3 compute processors act together (hence 1 kerenl for
    // compute). There is no need to explicitly parallelize the compute kernels. Unlike traditional CPU/GPU style SPMD
    // programming, the 3 compute processors moves data from SRAM into the FPU(tensor engine)/SFPU(SIMD engine),
    // operates on the data, and move it back to SRAM. The data movement processors moves data from the NoC, or in our
    // case, the DRAM, into the SRAM.
    //
    // The vector add example consists of 3 kernels. `interleaved_tile_read` reads tiles from the input buffers A and B
    // into 2 circular buffers. `add` reads tiles from the circular buffers, adds them together, and dumps the result
    // into a third circular buffer. `tile_write` reads tiles from the third circular buffer and writes them to the
    // output buffer C.
    //
    // Circular buffers are Tenstorrent's way of communicating between the data movement and the compute kernels.
    // Kernels queue tiles into the circular buffer and takes them when they are ready. The circular buffer is
    // backed by SRAM. There can be multiple circular buffers on a single Tensix core.
    constexpr uint32_t tiles_per_cb = 4;

    auto builder = ProgramBuilder(core);
    builder.cb(tt::CBIndex::c_0, tiles_per_cb)
        .cb(tt::CBIndex::c_1, tiles_per_cb)
        .cb(tt::CBIndex::c_16, tiles_per_cb);

    // Reader kernel: reads tiles from A and B into circular buffers c_0 and c_1.
    // The EZ API auto-generates TensorAccessorArgs from the buffer list {a, b}.
    auto& reader_ref = builder.reader(
        OVERRIDE_KERNEL_PREFIX "contributed/vecadd/kernels/interleaved_tile_read.cpp",
        {a, b});
    // Writer kernel: reads from c_16 and writes tiles to output buffer C.
    auto& writer_ref = builder.writer(
        OVERRIDE_KERNEL_PREFIX "contributed/vecadd/kernels/tile_write.cpp",
        {c});
    // Compute kernel: reads from c_0 and c_1, adds tiles, writes result to c_16.
    auto& compute_ref = builder.compute(
        OVERRIDE_KERNEL_PREFIX "contributed/vecadd/kernels/add.cpp");

    // Set the runtime arguments for the kernels.
    reader_ref.runtime_args({a->address(), b->address(), n_tiles});
    writer_ref.runtime_args({c->address(), n_tiles});
    compute_ref.runtime_args({n_tiles});

    // Upload input data (non-blocking), execute program, then read back the result.
    // The last write blocks to ensure data is ready before kernel execution.
    ctx.write(a, a_data);
    ctx.write(b, b_data);
    ctx.run(builder.build());

    std::cout << "Kernel execution finished" << std::endl;

    // Read the output buffer.
    auto c_data = ctx.read<uint32_t>(c);

    // Print partial results so we can see the output is correct (plus or minus some error due to BFP16 precision)
    std::cout << "Partial results: (note we are running under BFP16. It's going to be less accurate)\n";
    size_t n = std::min((size_t)10, (size_t)tile_size * n_tiles);
    bfloat16* a_bf16 = reinterpret_cast<bfloat16*>(a_data.data());
    bfloat16* b_bf16 = reinterpret_cast<bfloat16*>(b_data.data());
    bfloat16* c_bf16 = reinterpret_cast<bfloat16*>(c_data.data());
    for (size_t i = 0; i < n; i++) {
        std::cout << "  " << static_cast<float>(a_bf16[i]) << " + " << static_cast<float>(b_bf16[i]) << " = "
                  << static_cast<float>(c_bf16[i]) << "\n";
    }
    std::cout << std::flush;

    return 0;
}
