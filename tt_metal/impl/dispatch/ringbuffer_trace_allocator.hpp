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
// allocator) like LegacyTraceAllocator, but with one optimization: when the
// same program_id appears on two consecutive dispatches, the second dispatch
// reuses the first dispatch's config/binary addresses exactly once, setting
// send_binary = false and only reserving launch message slots.
//
// The "exactly once" constraint prevents unbounded reuse chains: after a reuse,
// the next dispatch of the same program must do a full allocation.
class RingbufferTraceAllocator {
public:
    struct RingbufferConfig {
        uint32_t start;
        uint32_t size;
    };

    explicit RingbufferTraceAllocator(const std::vector<RingbufferConfig>& ringbuffer_configs) :
        ringbuffer_configs_(ringbuffer_configs) {}

    void allocate_trace_programs(const Hal& hal, std::vector<TraceNode*>& trace_nodes);

private:
    void allocate_trace_programs_on_subdevice(
        const Hal& hal, std::vector<TraceNode*>& trace_nodes, SubDeviceId sub_device_id);

    std::vector<RingbufferConfig> ringbuffer_configs_;
};

}  // namespace tt::tt_metal
