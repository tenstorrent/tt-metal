// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Reproducer for the device-profiler NoC-event capture of a long kernel: one launch on a few cores, BRISC and NCRISC
// each record LOOP_COUNT NoC reads, which makes the L1 marker ring quick-push to DRAM many times mid-kernel.
// Run with TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 and a program support count large enough
// that the per-RISC DRAM region does not overflow (LOOP_COUNT * 16 B per RISC, 48 B per support count unit).
//
// Env knobs (all optional), to mirror how a model harness drives the same path:
//   NOC_EVENT_QUICK_PUSH_LOOP_COUNT   reads per RISC (default 4000)
//   NOC_EVENT_QUICK_PUSH_DELAY_NOPS   nops between reads (default 0)
//   NOC_EVENT_QUICK_PUSH_ITER_ZONE    1 = a zone around every read (default 0)
//   NOC_EVENT_QUICK_PUSH_PUSH_EVERY   n > 0 = call kernel_profiler::quick_push() directly every n-th read (default 0)
//   NOC_EVENT_QUICK_PUSH_MCAST_EVERY  n > 0 = a linked multicast write to the core range every n-th read (default 0);
//                                     each one makes the NoC tracer quick-push pre-emptively, at any ring fill level
//   NOC_EVENT_QUICK_PUSH_WARMUP       N tiny programs launched (non-blocking) before the long one (default 0), so
//                                     the DRAM profiler region already holds many chunks
//   NOC_EVENT_QUICK_PUSH_BLOCKING     1 = blocking enqueue of the long program (default 1)
//   NOC_EVENT_QUICK_PUSH_SKIP_READ    1 = no explicit ReadMeshDeviceProfilerResults; only the read at device close
namespace {

uint32_t env_u32(const char* name, uint32_t default_value) {
    const char* value = std::getenv(name);
    return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : default_value;
}

Program make_program(
    const CoreRange& cores,
    uint32_t page_size,
    uint32_t loop_count,
    uint32_t delay_nops,
    uint32_t iter_zone,
    uint32_t push_every,
    uint32_t mcast_every,
    const std::vector<uint32_t>& runtime_args) {
    Program program = CreateProgram();

    constexpr uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, cores, cb_config);

    std::map<std::string, std::string> kernel_defines = {
        {"LOOP_COUNT", std::to_string(loop_count)},
        {"DELAY_NOPS", std::to_string(delay_nops)},
        {"ITER_ZONE", std::to_string(iter_zone)},
        {"PUSH_EVERY", std::to_string(push_every)},
        {"MCAST_EVERY", std::to_string(mcast_every)}};
    const std::string kernel_path = "tests/tt_metal/tools/profiler/kernels/noc_event_quick_push.cpp";
    KernelHandle brisc_kernel = CreateKernel(
        program,
        kernel_path,
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = kernel_defines});
    KernelHandle ncrisc_kernel = CreateKernel(
        program,
        kernel_path,
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = kernel_defines});
    SetRuntimeArgs(program, brisc_kernel, cores, runtime_args);
    SetRuntimeArgs(program, ncrisc_kernel, cores, runtime_args);
    return program;
}

}  // namespace

int main() {
    bool pass = true;

    try {
        const uint32_t loop_count = env_u32("NOC_EVENT_QUICK_PUSH_LOOP_COUNT", 4000);
        const uint32_t delay_nops = env_u32("NOC_EVENT_QUICK_PUSH_DELAY_NOPS", 0);
        const uint32_t iter_zone = env_u32("NOC_EVENT_QUICK_PUSH_ITER_ZONE", 0);
        const uint32_t push_every = env_u32("NOC_EVENT_QUICK_PUSH_PUSH_EVERY", 0);
        const uint32_t mcast_every = env_u32("NOC_EVENT_QUICK_PUSH_MCAST_EVERY", 0);
        const uint32_t warmup = env_u32("NOC_EVENT_QUICK_PUSH_WARMUP", 0);
        const bool blocking = env_u32("NOC_EVENT_QUICK_PUSH_BLOCKING", 1) != 0;
        const bool skip_read = env_u32("NOC_EVENT_QUICK_PUSH_SKIP_READ", 0) != 0;
        constexpr uint32_t page_size = 64;

        fmt::print(
            "noc_event_quick_push: loop_count={} delay_nops={} iter_zone={} push_every={} mcast_every={} warmup={} "
            "blocking={} skip_read={}\n",
            loop_count,
            delay_nops,
            iter_zone,
            push_every,
            mcast_every,
            warmup,
            blocking,
            skip_read);

        int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

        CoreRange cores({0, 0}, {1, 1});
        // Multicast range = the core range in NoC coordinates; each sender reaches the other 3 cores.
        const CoreCoord mcast_start = mesh_device->worker_core_from_logical_core(cores.start_coord);
        const CoreCoord mcast_end = mesh_device->worker_core_from_logical_core(cores.end_coord);
        const std::vector<uint32_t> runtime_args = {
            page_size,
            static_cast<uint32_t>(mcast_start.x),
            static_cast<uint32_t>(mcast_start.y),
            static_cast<uint32_t>(mcast_end.x),
            static_cast<uint32_t>(mcast_end.y),
            static_cast<uint32_t>(cores.size() - 1)};

        // Warm-up: many small programs on the same cores, each leaving a chunk per RISC in the DRAM profiler region.
        for (uint32_t i = 0; i < warmup; i++) {
            distributed::MeshWorkload workload;
            workload.add_program(device_range, make_program(cores, page_size, /*loop_count=*/1, 0, 0, 0, 0, runtime_args));
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        }

        distributed::MeshWorkload workload;
        workload.add_program(device_range, make_program(cores, page_size, loop_count, delay_nops, iter_zone, push_every, mcast_every, runtime_args));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, blocking);

        if (!skip_read) {
            ReadMeshDeviceProfilerResults(*mesh_device);
        }

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
