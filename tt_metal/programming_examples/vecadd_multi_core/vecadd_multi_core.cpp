// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This programming example is based on the vecadd single core example in the
// contributed folder. It illustrates using multiple cores to perform vector
// addition. The program will use all available cores to perform the vector addition.
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>
#include <tt-metalium/work_split.hpp>

#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

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

    // n_tiles is the number of tiles of data for this example to add two vectors.
    const uint32_t n_tiles = 640;

    DeviceContext ctx(device_id);

    const uint32_t tile_size = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    std::map<CoreCoord, uint32_t> core_tile_idx;

    // Create 3 DRAM tile buffers: A and B are inputs, C is the output.
    // A tile on Tenstorrent is 32x32 elements; with BFloat16, each tile occupies 2048 bytes.
    auto a = ctx.dram_tile_buffer(n_tiles);
    auto b = ctx.dram_tile_buffer(n_tiles);
    auto c = ctx.dram_tile_buffer(n_tiles);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, 10.0f);
    std::vector<bfloat16> a_data(tile_size * n_tiles);
    std::vector<bfloat16> b_data(tile_size * n_tiles);
    for (uint32_t i = 0; i < tile_size * n_tiles; i++) {
        a_data[i] = bfloat16(dist(rng));
        b_data[i] = bfloat16(dist(rng));
    }

    auto core_grid = ctx.device().compute_with_storage_grid_size();
    uint32_t num_cores_x = core_grid.x;
    uint32_t num_cores_y = core_grid.y;
    // CoreRange uses inclusive start and end coordinates: [0, 0] to [num_cores_x - 1, num_cores_y - 1].
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Calculate the work distribution across the cores. The work is split into 2 groups of cores.
    // Primary and secondary. The primary group will process more tiles per core than the secondary
    // group, in case the number of tiles is not evenly divisible by the number of available cores.
    // This function guarantees that the work is exactly split across the cores. No more, no less.
    constexpr bool row_major = true;
    auto
        [num_cores,                   // number of cores utilized
         all_cores,                   // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(core_grid, n_tiles, row_major);

    // A Tensix core is made up of 5 processors: 2 data movement processors and 3 compute processors.
    // The 2 data movement processors act independently. The 3 compute processors act together
    // (hence 1 kernel for compute). The data movement processors move data from the NoC (or DRAM)
    // into SRAM, and the compute processors move data from SRAM into the FPU/SFPU, operate on it,
    // and move it back.
    //
    // The vector add example consists of 3 kernels:
    //   interleaved_tile_read → reads tiles from A and B into circular buffers c_0 and c_1
    //   add_multi_core        → reads from c_0 and c_1, adds tiles, writes result to c_2
    //   tile_write             → reads from c_2 and writes tiles to output buffer C
    //
    // Circular buffers act as pipes between kernels on the same core. They are backed by L1 (SRAM)
    // memory. Each CB here holds 4 tiles for multi-buffering to overlap data movement and compute.
    //
    // The CB indices are passed as compile-time args before the auto-generated TensorAccessorArgs,
    // matching the kernels' expectation (e.g. TensorAccessorArgs<2>() at offset 2 in the reader).
    constexpr uint32_t cir_buf_num_tiles = 4;

    auto builder = ProgramBuilder(all_cores);
    builder.cb(tt::CBIndex::c_0, cir_buf_num_tiles)
        .cb(tt::CBIndex::c_1, cir_buf_num_tiles)
        .cb(tt::CBIndex::c_2, cir_buf_num_tiles);

    auto& reader_ref = builder
        .named_args({{"c_0", (uint32_t)tt::CBIndex::c_0}, {"c_1", (uint32_t)tt::CBIndex::c_1}})
        .reader(
            "tt_metal/programming_examples/vecadd_multi_core/kernels/interleaved_tile_read_multi_core.cpp",
            {a, b},
            {(uint32_t)tt::CBIndex::c_0, (uint32_t)tt::CBIndex::c_1});

    auto& writer_ref = builder.writer(
        "tt_metal/programming_examples/vecadd_multi_core/kernels/tile_write_multi_core.cpp",
        {c},
        {(uint32_t)tt::CBIndex::c_2});

    auto& compute_ref = builder
        .named_args({{"c_0", (uint32_t)tt::CBIndex::c_0},
                     {"c_1", (uint32_t)tt::CBIndex::c_1},
                     {"c_2", (uint32_t)tt::CBIndex::c_2}})
        .compute(
            "tt_metal/programming_examples/vecadd_multi_core/kernels/add_multi_core.cpp",
            MathFidelity::HiFi4,
            {(uint32_t)tt::CBIndex::c_0, (uint32_t)tt::CBIndex::c_1, (uint32_t)tt::CBIndex::c_2});

    // Set per-core runtime arguments to achieve SPMD — each core gets a different starting tile
    // and number of tiles to process.
    auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};
    uint32_t start_tile_id = 0;
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                reader_ref.runtime_args_at(core, {a->address(), b->address(), work_per_core, start_tile_id});
                writer_ref.runtime_args_at(core, {c->address(), work_per_core, start_tile_id});
                compute_ref.runtime_args_at(core, {work_per_core, start_tile_id});
                core_tile_idx[core] = start_tile_id;

                start_tile_id += work_per_core;
            }
        }
    }

    ctx.write(a, a_data);
    ctx.write(b, b_data);

    // Execute the program.
    ctx.run(builder.build());

    std::cout << "Kernel execution finished" << std::endl;

    // Read the output buffer.
    auto c_data = ctx.read<bfloat16>(c);

    // Print partial results so we can see the output is correct (plus or minus
    // some error due to BFP16 precision).
    std::cout << "Partial results: (note we are running under BFP16. It's going "
                 "to be less accurate)\n";
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
        if (std::abs(static_cast<float>(c_data[i]) - expected) > 0.3f) {
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

    return 0;
}
