// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// LEGACY counterpart to programming_examples/named_kernel_args.
//
// Same schema (3 CTAs, 3 RTAs, 2 CRTAs) and identical DPRINT output, but built on the classic
// Metal 1.0 host API: CreateProgram / CreateKernel with positional compile_args, plus
// SetRuntimeArgs / SetCommonRuntimeArgs taking positional uint32_t vectors. The device kernel
// reads each arg by index. Compare with the TT_KERNEL example, where the host sets args by
// name and the kernel takes them as function/template parameters with a generated kernel_main().
//
// Run with DPRINT enabled:
//   export TT_METAL_DPRINT_CORES=0,0
//   ./metal_example_named_kernel_args_legacy

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

int main() {
    if (std::getenv("TT_METAL_DPRINT_CORES") == nullptr) {
        std::cerr << "WARNING: set TT_METAL_DPRINT_CORES=0,0 to see the kernel's DPRINT output.\n"
                  << "         e.g. export TT_METAL_DPRINT_CORES=0,0\n";
    }

    constexpr CoreCoord core = {0, 0};
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = CreateProgram();

    // Legacy positional args. CTAs are compile_args (read by index on device); the order here
    // is the contract with get_compile_time_arg_val(0..2) in the kernel.
    KernelHandle kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "named_kernel_args_legacy/kernels/dataflow/named_kernel_args_legacy_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {4, 2, 1},  // block_h, block_w, untilize
        });

    // The same schema on the compute (TRISC) path: ComputeConfig.compile_args are the CTAs, read
    // by index with get_compile_time_arg_val. Positional throughout, exactly like the DM kernel.
    KernelHandle compute_kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "named_kernel_args_legacy/kernels/compute/named_kernel_args_legacy_compute_kernel.cpp",
        core,
        ComputeConfig{
            .compile_args = {4, 2, 1},  // block_h, block_w, untilize
        });

    // RTAs (per-core) and CRTAs (common), both positional — order must match the kernel's
    // get_arg_val / get_common_arg_val indices. The DM and compute kernels take the same schema.
    SetRuntimeArgs(program, kernel, core, {0x10000, 0x20000, 64});  // src_addr, dst_addr, num_tiles
    SetCommonRuntimeArgs(program, kernel, {0x3f800000, 0x30000});   // scaler, sem_addr
    SetRuntimeArgs(program, compute_kernel, core, {0x10000, 0x20000, 64});  // src_addr, dst_addr, num_tiles
    SetCommonRuntimeArgs(program, compute_kernel, {0x3f800000, 0x30000});   // scaler, sem_addr

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    mesh_device->close();
    return 0;
}
