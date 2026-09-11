// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NOC atomic-increment probe against a Blackhole L2CPU tile.
//
// Drives kernels/l2cpu_atomic_probe.cpp on one Tensix core: seed a word in the
// L2CPU with a plain write, fire one NOC atomic increment at it, wait a bounded
// time for the NIU atomic-response counter, then read the word back. Prints, per
// target, whether the increment landed and whether a response arrived.
//
// Env: DEVICE_ID (default 0), L2CPU_X/L2CPU_Y (default 8,3),
//      PROBE_ADDR_A (default LIM 0x0801_0100), PROBE_ADDR_B (default uncached GDDR
//      alias 0x3010_2000; 0 disables), PROBE_SPIN (default 20,000,000), PROBE_INCR (5).

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
    const int device_id = static_cast<int>(env_or("DEVICE_ID", 0));
    const uint32_t l2cpu_x = env_or("L2CPU_X", 8);
    const uint32_t l2cpu_y = env_or("L2CPU_Y", 3);
    const uint32_t addr_a = env_or("PROBE_ADDR_A", 0x0801'0100u);
    const uint32_t addr_b = env_or("PROBE_ADDR_B", 0x3010'2000u);
    const uint32_t spin_cap = env_or("PROBE_SPIN", 20'000'000u);
    const uint32_t incr = env_or("PROBE_INCR", 5);
    constexpr uint32_t out_size = 512;

    bool pass = true;
    try {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        distributed::DeviceLocalBufferConfig dram_cfg{.page_size = out_size, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig l1_cfg{.page_size = out_size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig buf_size{.size = out_size};
        auto out_dram = distributed::MeshBuffer::create(buf_size, dram_cfg, mesh_device.get());
        auto out_l1 = distributed::MeshBuffer::create(buf_size, l1_cfg, mesh_device.get());

        constexpr CoreCoord core{0, 0};
        Program program = CreateProgram();
        std::vector<uint32_t> ct_args;
        TensorAccessorArgs(*out_dram->get_backing_buffer()).append_to(ct_args);
        KernelHandle kernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "l2cpu_noc_transfer/kernels/l2cpu_atomic_probe.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = ct_args});
        SetRuntimeArgs(
            program,
            kernel,
            core,
            {static_cast<uint32_t>(out_l1->address()),
             static_cast<uint32_t>(out_dram->address()),
             out_size,
             l2cpu_x,
             l2cpu_y,
             addr_a,
             addr_b,
             spin_cap,
             incr});

        fmt::print(
            "NOC atomic probe: device {} -> L2CPU ({},{}) targets A=0x{:08x} B=0x{:08x}, incr {}, spin cap {}\n",
            device_id,
            l2cpu_x,
            l2cpu_y,
            addr_a,
            addr_b,
            incr,
            spin_cap);

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);

        std::vector<uint32_t> rec;
        distributed::EnqueueReadMeshBuffer(cq, rec, out_dram, /*blocking=*/true);

        for (uint32_t k = 0; k < 2; k++) {
            const uint32_t* r = rec.data() + k * 8;
            if (r[0] == 0) {
                continue;
            }
            const bool landed = (r[2] == r[1] + incr);
            const bool responded = (r[4] != r[3]);
            fmt::print(
                "  target 0x{:08x}: seed 0x{:08x} -> readback 0x{:08x} [{}] | atomic resp counter {} -> {} after {} "
                "spins [{}] | nonposted atomics started {} -> {}\n",
                r[0],
                r[1],
                r[2],
                landed ? "INCREMENT LANDED" : (r[2] == r[1] ? "unchanged" : "UNEXPECTED VALUE"),
                r[3],
                r[4],
                r[5],
                responded ? "response received" : "NO response",
                r[6],
                r[7]);
            pass = pass && landed;
        }

        pass = mesh_device->close() && pass;
    } catch (const std::exception& e) {
        fmt::print(stderr, "Failed with exception: {}\n", e.what());
        throw;
    }

    fmt::print(
        "{}\n", pass ? "Atomic increments landed on every target" : "Atomic increment did NOT land on every target");
    return pass ? 0 : 1;
}
