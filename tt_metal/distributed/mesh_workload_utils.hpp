// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <host_api.hpp>
#include <stdint.h>
#include <optional>

#include "core_coord.hpp"
#include "sub_device_types.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/program/dispatch.hpp"

namespace tt::tt_metal {
class SystemMemoryManager;
}  // namespace tt::tt_metal

// Utility functions for dispatch MeshWorkloads
// Used by MeshCommandQueue
namespace tt::tt_metal::distributed {

class MeshDevice;

// Build the go-signal sequence in host-owned memory without submitting it.
// Used by offline command serialization; normal dispatch writes the same
// canonical sequence directly into CQ-owned hugepage memory below.
HostMemDeviceCommand build_go_signal_sequence(
    uint8_t cq_id,
    MeshDevice* mesh_device,
    SubDeviceId sub_device_id,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    bool send_mcast,
    bool send_unicasts,
    const program_dispatch::ProgramDispatchMetadata& dispatch_metadata,
    std::optional<uint32_t> config_ring_sync_count);

void write_go_signal_sequence(
    uint8_t cq_id,
    MeshDevice* mesh_device,
    SubDeviceId sub_device_id,
    SystemMemoryManager& sysmem_manager,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    bool send_mcast,
    bool send_unicasts,
    const program_dispatch::ProgramDispatchMetadata& dispatch_md,
    std::optional<uint32_t> config_ring_sync_count);

void write_rt_profiler_flush(
    uint8_t cq_id, SubDeviceId sub_device_id, SystemMemoryManager& sysmem_manager, uint32_t wait_count);

}  // namespace tt::tt_metal::distributed
