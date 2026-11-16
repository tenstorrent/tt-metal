// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>

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

// L1 that a workload's Tensix kernel-config blocks (runtime args, CB and DFB configs, kernel code)
// live in, instead of the HAL KERNEL_CONFIG window. Each core has its own shard at its own address
// (per-core allocation), so a block fits wherever a core has room. Slow dispatch only.
//
// Nothing is attached to a program: pass the buffer to each launch (EnqueueMeshWorkload below).
// Copies share one allocation; the L1 is freed by deallocate() or when the last copy is destroyed.
class KernelConfigBuffer {
public:
    explicit KernelConfigBuffer(std::shared_ptr<KernelConfigBufferImpl> impl);

    // Tensix cores the buffer has a shard on, on any device.
    const CoreRangeSet& cores() const;
    // Bytes of the largest shard on any device.
    uint32_t size() const;
    // Bytes of the shard on `core` of the device at `device_coord`: sized for the block of the core's kernel group in
    // that device's program.
    uint32_t size(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;
    // Shard address on `core` of the device at `device_coord`. Differs per core and per device; with a lockstep
    // address, every core of one device has its shard at one address.
    DeviceAddr address(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;
    // Frees the L1 now, for every copy. Launching with a deallocated buffer is an error.
    void deallocate();

    KernelConfigBufferImpl& impl() { return *impl_; }
    const KernelConfigBufferImpl& impl() const { return *impl_; }

private:
    std::shared_ptr<KernelConfigBufferImpl> impl_;
};

// Allocates a buffer on the Tensix cores of `workload`'s programs, on each of this host's devices. Compiles and lays
// out the programs first (program_preparation::compile), unless the caller already has. Their blocks may outgrow the
// KERNEL_CONFIG window: launching them from it fails.
//
// By default each core of each device gets a shard sized for its kernel group's block in that device's program, at
// its own address. With `lockstep_address`, every core of a device has its shard at one address, sized for the
// largest block on that device, so the programs may use Tensix program semaphores; it needs the cores' free L1 to
// start at the same address, and fails otherwise.
//
// Requires slow dispatch and the device opened with TT_METAL_ALLOCATOR_MODE_HYBRID=1.
KernelConfigBuffer CreateKernelConfigBuffer(
    distributed::MeshDevice& device, distributed::MeshWorkload& workload, bool lockstep_address = false);

// distributed::EnqueueMeshWorkload, with the workload's Tensix kernel-config blocks written to and
// run from `buffer`, each core's at its own address. `buffer` must have a shard on each Tensix core of each
// device's program that fits the block of the core's kernel group.
//
// Rejects Tensix program semaphores (CreateSemaphore) unless `buffer` has a lockstep address: a kernel finds a peer
// core's program semaphore at its own address, which per-core block addresses break.
// Slow dispatch only.
void EnqueueMeshWorkload(
    distributed::MeshCommandQueue& mesh_cq,
    distributed::MeshWorkload& workload,
    const KernelConfigBuffer& buffer,
    bool blocking);

}  // namespace experimental
}  // namespace tt::tt_metal
