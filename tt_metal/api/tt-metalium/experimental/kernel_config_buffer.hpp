// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/vocabulary/hal_types.hpp>
#include <tt-metalium/vocabulary/mesh_coord.hpp>

// Experimental and subject to change: this header carries no API-stability guarantee.

namespace tt::tt_metal {

namespace distributed {
class MeshBuffer;
class MeshCommandQueue;
class MeshDevice;
class MeshWorkload;
}  // namespace distributed

namespace experimental {

class KernelConfigBufferImpl;

// A region of one core's L1: `size` bytes from `address`.
struct CoreL1Region {
    DeviceAddr address = 0;
    uint32_t size = 0;
};

// L1 that a workload's Tensix kernel binaries live in, instead of the HAL KERNEL_CONFIG window. The rest
// of the kernel config (runtime args, semaphores, CB and DFB configs) stays in the window. Each core has
// its own shard at its own address (per-core allocation), so its binaries fit wherever it has room. Slow
// dispatch only.
//
// Nothing is attached to a program: pass the buffer to each launch (EnqueueMeshWorkload below).
//
// Ownership: the buffer owns its L1, which is freed when the buffer is destroyed. It is move-only; moving it moves the
// L1 with it, and a moved-from buffer cannot be used. The caller keeps it alive while a launch that runs from it can
// still be executing.
class KernelConfigBuffer {
public:
    explicit KernelConfigBuffer(std::unique_ptr<KernelConfigBufferImpl> impl);
    ~KernelConfigBuffer();
    KernelConfigBuffer(KernelConfigBuffer&&) noexcept;
    KernelConfigBuffer& operator=(KernelConfigBuffer&&) noexcept;
    KernelConfigBuffer(const KernelConfigBuffer&) = delete;
    KernelConfigBuffer& operator=(const KernelConfigBuffer&) = delete;

    // Bytes of the largest shard on any device.
    uint32_t max_size() const;
    // The shard on `core` of the device at `device_coord`: sized for the binaries of the core's kernel group in that
    // device's program. Its address differs per core and per device.
    CoreL1Region region(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;

    KernelConfigBufferImpl& impl();
    const KernelConfigBufferImpl& impl() const;

private:
    std::unique_ptr<KernelConfigBufferImpl> impl_;
};

// Allocates a buffer on the Tensix cores of `workload`'s programs, on each of this host's devices. Compiles and lays
// out the programs first (program_preparation::compile), unless the caller already has. Their kernel binaries may
// outgrow the KERNEL_CONFIG window: launching them from it fails.
//
// Each core of each device gets a shard sized for its kernel group's binaries in that device's program, at its own
// address. `bottom_up` allocates the shards from the bottom of L1 instead of the top.
//
// Requires slow dispatch and the device opened with TT_METAL_ALLOCATOR_MODE_HYBRID=1.
KernelConfigBuffer CreateKernelConfigBuffer(
    distributed::MeshDevice& device, distributed::MeshWorkload& workload, bool bottom_up = false);

// distributed::EnqueueMeshWorkload, with the workload's Tensix kernel binaries written to and run from
// `buffer`, each core's at its own address. `buffer` must have a shard on each Tensix core of each device's
// program that fits the binaries of the core's kernel group. The rest of the kernel config must still fit
// the KERNEL_CONFIG window.
//
// Slow dispatch only.
void EnqueueMeshWorkload(
    distributed::MeshCommandQueue& mesh_cq,
    distributed::MeshWorkload& workload,
    const KernelConfigBuffer& buffer,
    bool blocking);

}  // namespace experimental
}  // namespace tt::tt_metal
