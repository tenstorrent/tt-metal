// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <device.hpp>

namespace tt::tt_metal {

namespace subdevice_dispatch {

void reset_worker_dispatch_state_on_device(
    IDevice* device,
    SystemMemoryManager& manager,
    uint8_t cq_id,
    CoreCoord dispatch_core,
    const std::array<uint32_t, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed,
    bool reset_launch_msg_state);

void set_num_worker_sems_on_dispatch(
    IDevice* device, SystemMemoryManager& manager, uint8_t cq_id, uint32_t num_worker_sems);

void set_go_signal_noc_data_on_dispatch(
    IDevice* device,
    const vector_memcpy_aligned<uint32_t>& go_signal_noc_data,
    SystemMemoryManager& manager,
    uint8_t cq_id);

}  // namespace subdevice_dispatch

}  // namespace tt::tt_metal
