// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "core_coord.hpp"
#include <tt_stl/span.hpp>

namespace tt::tt_metal {
class HostCq;
enum NOC : uint8_t;
class Device;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

class HWCommandQueue {
public:
    HWCommandQueue(Device* device, uint32_t id, NOC noc_index);

    ~HWCommandQueue() = default;

    const CoreCoord& virtual_enqueue_program_dispatch_core() const;

    void set_dispatch_sems(uint32_t num_dispatch_sems, ttsl::Span<const uint32_t> workers_per_sub_device);

    uint32_t id() const;

    HostCq& host_cq();

    Device* device();

    // needed interface items
    void terminate();

private:
    uint32_t id_;
    HostCq& cq_;

    Device* device_;

    CoreCoord virtual_enqueue_program_dispatch_core_;
};

}  // namespace tt::tt_metal
