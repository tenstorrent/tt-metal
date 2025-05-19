// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include "trace/trace_buffer.hpp"

namespace tt::tt_metal::distributed {

// MeshTrace capture consists of 3 steps:
// 1. Staging: Workload dispatch commands are recorded into MeshTraceNodes.
// 2. Assembly: On trace end, dispatch commands are generated for all MeshTraceNodes and stored in a
// MeshTraceDescriptor.
// 3. Commit to Mesh: Write assembled trace to DRAM buffer.

// Finalized/Consolidated dispatch commands on a device_range, corresponding
// to a trace
struct MeshTraceData {
    MeshCoordinateRange device_range = MeshCoordinateRange(MeshShape(0, 0));
    std::vector<uint32_t> data;
};

// Wrapper around the MeshTraceData. Captures the complete state of a MeshTrace
// (including the dispatch commands across devices, the SubDevices the trace runs on
// the size of the trace and the number of workers in the trace) on host
class MeshTraceDescriptor {
public:
    // Mapping of sub_device_id to descriptor
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> descriptors;
    // Store the keys of the map in a vector after descriptor has finished being populated
    // This is an optimization since we sometimes need to only pass the keys in a container
    std::vector<SubDeviceId> sub_device_ids;
    // Trace data per logical Device in a Mesh.
    std::vector<MeshTraceData> ordered_trace_data;
    uint32_t total_trace_size = 0;
};

// Ties a MeshTraceDescriptor (host side state) to a MeshBuffer (device side state)
struct MeshTraceBuffer {
    // The trace descriptor associated with a MeshTrace
    std::shared_ptr<MeshTraceDescriptor> desc = nullptr;
    // The MeshBuffer this trace will be serialized to, before being run on a
    // MeshDevice
    std::shared_ptr<MeshBuffer> mesh_buffer = nullptr;
};

// Top level class - Manages MeshTrace
class MeshTrace {
public:
    // Get global (unique) ID for trace
    static MeshTraceId next_id();
    // Create an empty MeshTraceBuffer, which needs to be populated
    // with a MeshTraceDescriptor and a MeshBuffer, to get tied to a MeshDevice.
    static std::shared_ptr<MeshTraceBuffer> create_empty_mesh_trace_buffer();
    // Once the Trace Data per logical device has been captured in the
    // MeshTraceDescriptor corresponding to this MeshTraceBuffer,
    // it can be binarized to a MeshDevice through a Command Queue.
    static void populate_mesh_buffer(MeshCommandQueue& mesh_cq, std::shared_ptr<MeshTraceBuffer>& trace_buffer);
};

}  // namespace tt::tt_metal::distributed
