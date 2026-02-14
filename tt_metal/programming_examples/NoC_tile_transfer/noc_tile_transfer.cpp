// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/ez/ez.hpp>

#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

int main() {
    // Check if the environment variable for kernel debug printing is set.
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            stderr,
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(0,1) to see the output of "
            "the Data Movement kernels. Command: export TT_METAL_DPRINT_CORES=(0,0),(0,1)\n");
    }

    DeviceContext ctx(0);

    // Core setup — this example uses two cores that communicate via the NoC.
    // Core 0 reads data from DRAM, sends it to Core 1 via NoC, and Core 1 writes it back to DRAM.
    constexpr CoreCoord core0 = {0, 0};
    constexpr CoreCoord core1 = {0, 1};
    const auto core0_physical = ctx.physical_core(core0);
    const auto core1_physical = ctx.physical_core(core1);

    // Input data preparation — a single tile of uint16 data.
    constexpr uint32_t single_tile_size = sizeof(uint16_t) * tt::constants::TILE_HW;
    auto src_dram_buffer = ctx.dram_buffer(single_tile_size, single_tile_size);
    auto dst_dram_buffer = ctx.dram_buffer(single_tile_size, single_tile_size);

    // Upload source data to DRAM.
    const uint16_t input_data = 14;
    std::vector<uint16_t> src_vec(1, input_data);
    ctx.write(src_dram_buffer, src_vec);

    // Build the program with kernels on two cores, coordinated by a semaphore:
    //
    //   Core 0:
    //     reader0 → reads tile from DRAM into cb_0
    //     writer0 → sends cb_0 data to Core 1's cb_1 via NoC, signals semaphore
    //
    //   Core 1:
    //     reader1 → waits on semaphore, receives data into cb_1
    //     writer1 → writes cb_1 data back to DRAM
    //
    // Circular buffers cb_0 and cb_1 are created on both cores (same core range).
    // The semaphore ensures Core 1 doesn't read before Core 0 has finished writing.
    //
    // For kernels that access DRAM buffers (reader0, writer1), the CB index is passed as a
    // compile-time arg before the auto-generated TensorAccessorArgs. For kernels that only do
    // inter-core NoC transfers (writer0, reader1), only CB indices are needed (no buffers).
    CoreRange sem_core_range = CoreRange(core0, core1);
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t src1_cb_index = CBIndex::c_1;

    // Build the program step by step so we can store the semaphore ID for reuse.
    auto builder = ProgramBuilder(sem_core_range);

    // Circular buffers on both cores (same config).
    builder.cb(tt::CBIndex::c_0, tt::DataFormat::UInt16, /*num_tiles=*/1, /*page_size=*/single_tile_size)
        .cb(tt::CBIndex::c_1, tt::DataFormat::UInt16, /*num_tiles=*/1, /*page_size=*/single_tile_size);

    // Create a single semaphore on both cores for synchronization.
    const uint32_t sem_id = builder.semaphore();

    // Core 0: reader reads from DRAM into cb_0.
    builder.on(core0)
        .reader(
            OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/reader0.cpp",
            {src_dram_buffer},
            {src0_cb_index})
        .runtime_args({src_dram_buffer->address()})

        // Core 0: writer sends cb_0 to Core 1's cb_1 via NoC.
        .on(core0)
        .writer(
            OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/writer0.cpp",
            {},  // No DRAM buffer access
            {src0_cb_index, src1_cb_index})
        .runtime_args({core1_physical.x, core1_physical.y, sem_id})

        // Core 1: reader waits for data from Core 0.
        .on(core1)
        .reader(
            OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/reader1.cpp",
            {},  // No DRAM buffer access
            {src0_cb_index, src1_cb_index})
        .runtime_args({core0_physical.x, core0_physical.y, sem_id})

        // Core 1: writer writes cb_1 to DRAM.
        .on(core1)
        .writer(
            OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/writer1.cpp",
            {dst_dram_buffer},
            {src1_cb_index})
        .runtime_args({dst_dram_buffer->address()});

    auto program = builder.build();

    ctx.run(std::move(program));

    // Read result back from DRAM.
    auto result_vec = ctx.read<uint16_t>(dst_dram_buffer);

    fmt::print("Result = {} : Expected = {}\n", result_vec[0], input_data);
}
