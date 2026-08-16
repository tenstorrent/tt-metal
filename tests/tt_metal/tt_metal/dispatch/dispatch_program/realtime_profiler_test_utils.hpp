// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

namespace tt::tt_metal {

// A ~40k-NOP kernel body for any RISC: long enough that a record's window is unambiguous, short
// enough to enqueue by the hundreds. `marker_comment` embeds a distinct line for tests that
// assert on a record's kernel_sources; leaving it empty keeps one shared source so every program
// reuses a single JIT build instead of compiling per runtime_id.
inline std::string rt_profiler_nop_kernel_source(const std::string& marker_comment = {}) {
    return "#include <cstdint>\n" + (marker_comment.empty() ? std::string{} : "// " + marker_comment + "\n") +
           "void kernel_main() {\n"
           "    for (int i = 0; i < 200; i++) {\n"
           "#pragma GCC unroll 65534\n"
           "        for (int j = 0; j < 200; j++) {\n"
           "            asm(\"nop\");\n"
           "        }\n"
           "    }\n"
           "}\n";
}

inline Program make_rt_profiler_program(const std::string& kernel_src, const CoreRange& cores, uint32_t runtime_id) {
    Program program = CreateProgram();
    CreateKernelFromString(
        program,
        kernel_src,
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernelFromString(
        program,
        kernel_src,
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernelFromString(program, kernel_src, cores, ComputeConfig{});
    program.set_runtime_id(static_cast<uint64_t>(runtime_id));
    return program;
}

inline void enqueue_rt_profiler_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, Program program, bool blocking) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, blocking);
}

// Null when the RT profiler is inactive on this dispatch config (callers GTEST_SKIP).
inline std::shared_ptr<distributed::MeshDevice> open_profiler_unit_mesh(
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        /*device_id=*/0,
        DEFAULT_L1_SMALL_SIZE,
        trace_region_size,
        /*num_command_queues=*/1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    if (mesh_device == nullptr) {
        return nullptr;
    }
    if (!experimental::IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        return nullptr;
    }
    return mesh_device;
}

inline CoreRange all_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    return CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1});
}

template <typename Predicate>
void quiesce_and_wait_for(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Predicate delivered) {
    mesh_device->quiesce_devices();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!delivered() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

}  // namespace tt::tt_metal
