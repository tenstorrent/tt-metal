// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "core_coord.hpp"
#include <tt_stl/span.hpp>
#include <tt-umd/types/core_coordinates.hpp>

namespace tt::tt_metal {
class SystemMemoryManager;
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

    SystemMemoryManager& sysmem_manager();

    Device* device();

    // needed interface items
    void terminate();

private:
    uint32_t id_;
    SystemMemoryManager& manager_;

    Device* device_;

    CoreCoord virtual_enqueue_program_dispatch_core_;
};

}  // namespace tt::tt_metal
