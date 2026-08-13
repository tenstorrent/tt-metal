// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <memory>
#include <optional>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {
class Program;
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_metal {

class IDevice;

namespace distributed {

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);

// Make the current thread block until the event is recorded by the associated MeshCommandQueue.
void EventSynchronize(const MeshEvent& event);

void Synchronize(
    MeshDevice* device, std::optional<uint8_t> cq_id, ttsl::Span<const SubDeviceId> sub_device_ids = {});

void Finish(MeshCommandQueue& mesh_cq, ttsl::Span<const SubDeviceId> sub_device_ids = {});

// Returns true if the distributed environment is initialized and world_size > 1.
bool UsingDistributedEnvironment();

}  // namespace distributed
}  // namespace tt::tt_metal
