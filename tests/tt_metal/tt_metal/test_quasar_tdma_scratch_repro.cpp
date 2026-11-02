// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal repro for #60551. The TEN-4746 workaround in the Quasar matmul in0 reader does a 16-byte NOC
// loopback read into a function-local static scratch once per in0 block. In the ResNet-50 layer1 conv2 the
// watcher flags that read ("Local L1 overwrites mailboxes"), and with the program cache off the next launch
// sees run_mailbox = 0x3f. This test runs only that read (tdma_scratch_loopback_read.cpp) and checks:
//   * where the scratch lands for each DM thread, and that it holds the source bytes;
//   * that the source bytes appear nowhere else in the reserved (firmware) part of L1;
//   * that the node still launches programs after the reads (a corrupt mailbox stops the next launch).
// Env: TDMA_REPRO_THREADS (default 4), TDMA_REPRO_ITERS per launch (default 64), TDMA_REPRO_RUNS (default 3).

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#include <cstdlib>
#include <set>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace {
uint32_t env_or(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : fallback;
}
}  // namespace

TEST_F(QuasarMeshDeviceSingleCardFixture, TdmaScratchLoopbackRead) {
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator.";
    }
    auto mesh_device = devices_[0];
    // User DM threads run on DM2..DM7, so thread 0 is DM2, the processor in the ResNet failure.
    const uint32_t num_threads = env_or("TDMA_REPRO_THREADS", 4);
    const uint32_t num_iter = env_or("TDMA_REPRO_ITERS", 64);
    const uint32_t num_runs = env_or("TDMA_REPRO_RUNS", 3);

    const experimental::NodeCoord node{0, 0};
    const uint32_t unreserved = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t src_addr = unreserved;
    const uint32_t result_addr = unreserved + 1024;

    // bfloat16-looking words (0x3f.. high bytes, like the activations), distinct from anything firmware writes.
    std::vector<uint32_t> pattern = {0x3f813f82u, 0x3f833f84u, 0x3f853f86u, 0x3f873f88u};
    slow_dispatch::WriteToL1(this->device(), node, src_addr, pattern);
    std::vector<uint32_t> zeros(num_threads * 32, 0u);
    slow_dispatch::WriteToL1(this->device(), node, result_addr, zeros);

    auto scan_reserved = [&](const char* when) {
        std::vector<uint32_t> l1;
        slow_dispatch::ReadFromL1(this->device(), node, 0, unreserved, l1);
        std::vector<uint32_t> hits;
        for (uint32_t i = 0; i < l1.size(); i++) {
            for (uint32_t p : pattern) {
                if (l1[i] == p) {
                    hits.push_back(i * sizeof(uint32_t));
                }
            }
        }
        std::cout << "[SCRATCH] " << when << ": " << hits.size() << " pattern words in reserved L1 [0, 0x" << std::hex
                  << unreserved << ")";
        for (uint32_t a : hits) {
            std::cout << " 0x" << a;
        }
        std::cout << std::dec << std::endl;
        return hits;
    };
    ASSERT_TRUE(scan_reserved("before").empty()) << "pattern already in reserved L1 before the kernel ran";

    const experimental::KernelSpecName DM_KERNEL{"tdma_scratch"};
    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/tdma_scratch_loopback_read.cpp",
        .num_threads = num_threads,
        .runtime_arg_schema = {.common_runtime_arg_names = {"src_addr", "result_addr", "num_iter"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = node};
    experimental::ProgramSpec spec{.name = "tdma_scratch_repro", .kernels = {dm_kernel_spec}, .work_units = {wu}};

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    for (uint32_t run = 0; run < num_runs; run++) {
        // A new program each run, as with the program cache off: each launch goes through the mailbox again.
        Program program = experimental::MakeProgramFromSpec(this->device(), spec);
        experimental::ProgramRunArgs params;
        params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_KERNEL,
            .common_runtime_arg_values = {{"src_addr", src_addr}, {"result_addr", result_addr}, {"num_iter", num_iter}},
        }};
        experimental::SetProgramRunArgs(program, params);
        distributed::MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);
        std::cout << "[SCRATCH] run " << run << " done (" << num_threads << " threads x " << num_iter << " reads)"
                  << std::endl;
    }

    std::vector<uint32_t> res;
    slow_dispatch::ReadFromL1(this->device(), node, result_addr, num_threads * 128, res);
    std::set<uint32_t> scratch_words;
    uint32_t bad = 0;
    for (uint32_t t = 0; t < num_threads; t++) {
        const uint32_t* slot = &res[t * 32];
        const uint32_t scratch = slot[0];
        const bool ok = std::equal(pattern.begin(), pattern.end(), slot + 16);
        bad += ok ? 0 : 1;
        for (uint32_t w = 0; w < 4; w++) {
            scratch_words.insert(scratch + w * sizeof(uint32_t));
        }
        std::cout << "[SCRATCH] thread " << slot[1] << " scratch=0x" << std::hex << scratch << " data=" << slot[16]
                  << " " << slot[17] << " " << slot[18] << " " << slot[19] << std::dec << (ok ? " OK" : " MISMATCH")
                  << std::endl;
    }

    uint32_t leaks = 0;
    for (uint32_t a : scan_reserved("after")) {
        if (!scratch_words.contains(a)) {
            leaks++;
            std::cout << "[SCRATCH] LEAK: pattern word at 0x" << std::hex << a << std::dec
                      << ", outside every thread's scratch" << std::endl;
        }
    }
    EXPECT_EQ(bad, 0u) << "a thread's scratch does not hold the source bytes";
    EXPECT_EQ(leaks, 0u) << "the loopback read wrote outside the scratch, into reserved L1";
}
