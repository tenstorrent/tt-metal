// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host driver for the x280 echo demo (run AFTER x280_boot has released the
// hart). Launches kernels/x280_echo_poll.cpp on Tensix {0,0} and interprets
// the diagnostics it returns. See ../x280/fw.c for the protocol.
//
// Env: L2CPU_X / L2CPU_Y (default 8,3), X280_VALUE, X280_TIMEOUT_ITERS.

#include <fmt/base.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>
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
    const uint32_t l2cpu_x = env_or("L2CPU_X", 8);
    const uint32_t l2cpu_y = env_or("L2CPU_Y", 3);
    const uint32_t mbox = 0x3010'0000;  // uncached GDDR alias (matches fw.c MBOX)
    const uint32_t value = env_or("X280_VALUE", 1000);
    const uint32_t timeout_iters = env_or("X280_TIMEOUT_ITERS", 100'000'000);
    const uint32_t seq = static_cast<uint32_t>(time(nullptr)) | 1;  // unique + nonzero per run

    bool pass = false;
    {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        distributed::DeviceLocalBufferConfig dram_config{.page_size = 128, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig l1_config{.page_size = 512, .buffer_type = BufferType::L1};
        auto dst_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = 128}, dram_config, mesh_device.get());
        auto l1_scratch = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = 512}, l1_config, mesh_device.get());

        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        std::vector<uint32_t> compile_args;
        TensorAccessorArgs(*dst_dram->get_backing_buffer()).append_to(compile_args);

        KernelHandle kernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "l2cpu_noc_transfer/kernels/x280_echo_poll.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});

        SetRuntimeArgs(
            program,
            kernel,
            core,
            {static_cast<uint32_t>(l1_scratch->address()),
             static_cast<uint32_t>(dst_dram->address()),
             l2cpu_x,
             l2cpu_y,
             mbox,
             seq,
             value,
             timeout_iters});

        fmt::print(
            "x280 echo: seq=0x{:08x} value={} -> expecting {} back from L2CPU ({},{})\n",
            seq,
            value,
            value * 3 + 7,
            l2cpu_x,
            l2cpu_y);

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);

        std::vector<uint32_t> r;
        distributed::EnqueueReadMeshBuffer(cq, r, dst_dram, /*blocking=*/true);

        const uint32_t found = r[0], response = r[1], iters = r[2];
        const uint32_t worker_x = r[3] & 0xffff, worker_y = r[3] >> 16;
        // r[4..27] = mailbox words 0x00..0x5c
        const uint64_t heartbeat = r[4] | (uint64_t(r[5]) << 32);
        const uint64_t fw_state = r[6] | (uint64_t(r[7]) << 32);
        const uint32_t hartid = r[8], traps = r[10];
        const uint32_t mcause = r[12], cmo_ok = r[14];
        const uint32_t resp_seq = r[20], resp_status = r[21];
        const uint32_t resp_probe = r[22], resp_result = r[23];

        fmt::print("worker NOC0 coords ({},{}) | polled {} iters\n", worker_x, worker_y, iters);
        fmt::print(
            "firmware: heartbeat={} state=0x{:x} hartid={} traps={} mcause=0x{:x} cmo_ok={}\n",
            heartbeat,
            fw_state,
            hartid,
            traps,
            mcause,
            cmo_ok);
        fmt::print(
            "fw response block: seq=0x{:08x} status={} probe=0x{:08x} result={}\n",
            resp_seq,
            resp_status,
            resp_probe,
            resp_result);

        if (found) {
            fmt::print("*** x280 -> Tensix NOC write received: {} (expected {}) ***\n", response, value * 3 + 7);
            pass = (response == value * 3 + 7);
        } else if (fw_state != 0xA11FE) {
            fmt::print("firmware never reached its main loop — hart not running?\n");
        } else if (resp_seq == seq && resp_status == 2) {
            fmt::print(
                "firmware processed the request but the probe read 0x{:08x} != magic — coordinate "
                "encoding mismatch on the x280's TLB window.\n",
                resp_probe);
        } else {
            fmt::print("firmware alive but request unseen/unanswered — inbound coherence issue?\n");
        }

        mesh_device->close();
    }
    fmt::print("{}\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
