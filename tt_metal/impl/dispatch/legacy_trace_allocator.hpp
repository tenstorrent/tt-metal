// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <api/tt-metalium/device.hpp>

#include "trace/trace_node.hpp"

namespace tt::tt_metal {

class Hal;

// Trace allocator that uses WorkerConfigBufferMgr (the sequential ring-buffer
// allocator also used by the normal dispatch path) to assign kernel config
// buffer addresses.  This replicates the pre-SimpleTraceAllocator behavior
// where trace recording went through the same allocation logic as live
// dispatch, without any global Belady-like optimization or binary caching.
class LegacyTraceAllocator {
public:
    struct RingbufferConfig {
        uint32_t start;
        uint32_t size;
    };

    explicit LegacyTraceAllocator(const std::vector<RingbufferConfig>& ringbuffer_configs) :
        ringbuffer_configs_(ringbuffer_configs) {}

    void allocate_trace_programs(const Hal& hal, std::vector<TraceNode*>& trace_nodes);

private:
    void allocate_trace_programs_on_subdevice(
        const Hal& hal, std::vector<TraceNode*>& trace_nodes, SubDeviceId sub_device_id);

    std::vector<RingbufferConfig> ringbuffer_configs_;
};

}  // namespace tt::tt_metal
