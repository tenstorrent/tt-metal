// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Simple NOC transfers between a Tensix core and a Blackhole L2CPU (x280) tile.
//
// A single Tensix data-movement kernel round-trips a data pattern through the L2CPU
// tile's 2 MiB L3-as-scratchpad ("LIM", x280 physical address 0x0800_0000):
//
//   host -> DRAM -> Tensix L1 --noc_async_write--> L2CPU LIM
//                   Tensix L1 <--noc_async_read--- L2CPU LIM -> DRAM -> host
//
// plus a noc_inline_dw_write patch of one word inside the region. The host verifies
// the round-tripped data, including the patched word.
//
// Notes:
//  * The L2CPU tile is a *passive* NOC endpoint in this test. tt-metal cannot place
//    kernels there (the x280s are RV64 cores with no launch/dispatch support), and the
//    harts stay in reset the whole time — inbound NOC access works regardless.
//  * The L2CPU tiles are not in tt-metal's Blackhole SoC descriptor, so their NOC0
//    coordinates are supplied here directly. Blackhole's coordinate translation maps
//    L2CPU translated coords 1:1 to NOC0, so raw (8,3)/(8,5)/(8,7)/(8,9) route correctly.
//  * Env overrides: L2CPU_X, L2CPU_Y (default 8,3), TT_L2CPU_TEST_ATOMIC=1 enables a
//    noc_semaphore_inc probe against LIM.
//
// Measured on a p100a: all four tiles pass the bulk + inline-write round-trip.
// The atomic probe HANGS (no atomic response from the L2CPU bridge) — NOC atomics
// are not supported inbound; protocols against the L2CPU must be built from plain
// reads/writes. The hang is recoverable: the next device init resets the Tensix.

#include <fmt/base.h>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
uint32_t env_or(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : fallback;
}
}  // namespace

int main() {
    bool pass = true;

    // NOC0 coordinates of the Blackhole L2CPU tiles: (8,3) (8,5) (8,7) (8,9).
    const uint32_t l2cpu_x = env_or("L2CPU_X", 8);
    const uint32_t l2cpu_y = env_or("L2CPU_Y", 3);
    // Land 64 KiB into the LIM, clear of anything boot code might touch.
    const uint32_t lim_addr = 0x0800'0000 + 0x1'0000;
    const bool test_atomic = env_or("TT_L2CPU_TEST_ATOMIC", 0) != 0;

    constexpr uint32_t transfer_size = 8192;  // bytes
    constexpr uint32_t num_words = transfer_size / sizeof(uint32_t);
    constexpr uint32_t patch_word_idx = num_words / 2;
    constexpr uint32_t atomic_word_idx = num_words / 4;

    try {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
            fmt::print(stderr, "This example targets Blackhole (L2CPU tiles only exist there).\n");
            return 1;
        }

        // Single-page DRAM buffers for input/output, and one L1 scratch buffer that the
        // kernel splits into a source half and a destination half. L1 allocations are
        // lockstep across banks, so the address is valid scratch on the executing core.
        distributed::DeviceLocalBufferConfig dram_config{.page_size = transfer_size, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig l1_config{.page_size = 2 * transfer_size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig dram_size{.size = transfer_size};
        distributed::ReplicatedBufferConfig l1_size{.size = 2 * transfer_size};

        auto src_dram = distributed::MeshBuffer::create(dram_size, dram_config, mesh_device.get());
        auto dst_dram = distributed::MeshBuffer::create(dram_size, dram_config, mesh_device.get());
        auto l1_scratch = distributed::MeshBuffer::create(l1_size, l1_config, mesh_device.get());

        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        std::vector<uint32_t> compile_args;
        TensorAccessorArgs(*src_dram->get_backing_buffer()).append_to(compile_args);
        TensorAccessorArgs(*dst_dram->get_backing_buffer()).append_to(compile_args);

        // Pin the kernel to NOC0: the L2CPU coordinates above are NOC0 coordinates, and
        // this avoids any dependence on NOC1 coordinate flipping.
        KernelHandle kernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "l2cpu_noc_transfer/kernels/l2cpu_rw.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});

        std::vector<uint32_t> input(num_words);
        for (uint32_t i = 0; i < num_words; i++) {
            input[i] = 0xA000'0000u + i;
        }
        distributed::EnqueueWriteMeshBuffer(cq, src_dram, input, /*blocking=*/false);

        SetRuntimeArgs(
            program,
            kernel,
            core,
            {static_cast<uint32_t>(l1_scratch->address()),
             static_cast<uint32_t>(src_dram->address()),
             static_cast<uint32_t>(dst_dram->address()),
             transfer_size,
             l2cpu_x,
             l2cpu_y,
             lim_addr,
             patch_word_idx,
             test_atomic ? 1u : 0u,
             atomic_word_idx});

        fmt::print(
            "Round-tripping {} bytes through L2CPU tile ({},{}) LIM @ 0x{:08x} (atomic probe: {})\n",
            transfer_size,
            l2cpu_x,
            l2cpu_y,
            lim_addr,
            test_atomic ? "on" : "off");

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);

        std::vector<uint32_t> result;
        distributed::EnqueueReadMeshBuffer(cq, result, dst_dram, /*blocking=*/true);

        // Expected: the input pattern, with the patched word replaced by the inline-write
        // value, and (if probed) the atomic word incremented by 5.
        std::vector<uint32_t> expected = input;
        expected[patch_word_idx] = 0xC0FFEE55;
        if (test_atomic) {
            expected[atomic_word_idx] += 5;
        }

        uint32_t mismatches = 0;
        for (uint32_t i = 0; i < num_words; i++) {
            if (result[i] != expected[i]) {
                if (mismatches < 8) {
                    fmt::print(stderr, "  word {:4d}: expected 0x{:08x}, got 0x{:08x}\n", i, expected[i], result[i]);
                }
                mismatches++;
            }
        }
        pass = (mismatches == 0);

        fmt::print(
            "bulk write+read: {} words | inline dw write @ word {}: 0x{:08x} | {}\n",
            num_words,
            patch_word_idx,
            result[patch_word_idx],
            mismatches == 0 ? "all match" : fmt::format("{} MISMATCHES", mismatches));
        if (test_atomic) {
            fmt::print(
                "atomic probe @ word {}: wrote 0x{:08x}, +5 => got 0x{:08x} ({})\n",
                atomic_word_idx,
                input[atomic_word_idx],
                result[atomic_word_idx],
                result[atomic_word_idx] == input[atomic_word_idx] + 5 ? "NOC atomics work"
                                                                      : "NOC atomics NOT confirmed");
        }

        if (!mesh_device->close()) {
            pass = false;
        }
    } catch (const std::exception& e) {
        fmt::print(stderr, "Failed with exception: {}\n", e.what());
        throw;
    }

    fmt::print("{}\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
