// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {

// Forward decl to avoid including header
class Buffer;

struct TraceWorkerDescriptor {
    uint32_t num_completion_worker_cores = 0;
    uint32_t num_traced_programs_needing_go_signal_multicast = 0;
    uint32_t num_traced_programs_needing_go_signal_unicast = 0;
};

struct TraceDescriptor {
    // Mapping of sub_device_id to descriptor
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> descriptors;
    // Store the keys of the map in a vector after descriptor has finished being populated
    // This is an optimization since we sometimes need to only pass the keys in a container
    std::vector<SubDeviceId> sub_device_ids;
    std::vector<uint32_t> data;
};

struct TraceBuffer {
    std::shared_ptr<TraceDescriptor> desc;
    std::shared_ptr<Buffer> buffer;

    TraceBuffer(std::shared_ptr<TraceDescriptor> desc, std::shared_ptr<Buffer> buffer);
    ~TraceBuffer();

    void validate();
};

}  // namespace tt::tt_metal
