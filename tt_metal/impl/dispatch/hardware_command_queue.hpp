// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "buffer.hpp"
#include "core_coord.hpp"
#include "dispatch_settings.hpp"
#include "event.hpp"
#include "launch_message_ring_buffer_state.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/span.hpp>
#include "sub_device_types.hpp"
#include "trace/trace_buffer.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include "vector_aligned.hpp"
#include "worker_config_buffer.hpp"

namespace tt::tt_metal {
class IDevice;
class SystemMemoryManager;
enum NOC : uint8_t;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

class HWCommandQueue {
public:
    HWCommandQueue(IDevice* device, uint32_t id, NOC noc_index);

    ~HWCommandQueue() = default;

    const CoreCoord& virtual_enqueue_program_dispatch_core() const;

    void set_go_signal_noc_data_and_dispatch_sems(
        uint32_t num_dispatch_sems, const vector_aligned<uint32_t>& noc_mcast_unicast_data);

    uint32_t id() const;

    SystemMemoryManager& sysmem_manager();

    IDevice* device();

    // needed interface items
    void terminate();

private:
    uint32_t id_;
    SystemMemoryManager& manager_;

    IDevice* device_;

    CoreCoord virtual_enqueue_program_dispatch_core_;
};

}  // namespace tt::tt_metal
