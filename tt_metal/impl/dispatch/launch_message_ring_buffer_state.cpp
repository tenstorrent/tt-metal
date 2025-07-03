// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "launch_message_ring_buffer_state.hpp"

#include <cstdint>

#include "dev_msgs.h"

namespace tt::tt_metal {

void LaunchMessageRingBufferState::inc_mcast_wptr(uint32_t inc_val) {
    multicast_cores_launch_message_wptr_ =
        (multicast_cores_launch_message_wptr_ + inc_val) & (launch_msg_buffer_num_entries - 1);
}

void LaunchMessageRingBufferState::inc_unicast_wptr(uint32_t inc_val) {
    unicast_cores_launch_message_wptr_ =
        (unicast_cores_launch_message_wptr_ + inc_val) & (launch_msg_buffer_num_entries - 1);
}

void LaunchMessageRingBufferState::set_mcast_wptr(uint32_t val) {
    multicast_cores_launch_message_wptr_ = val & (launch_msg_buffer_num_entries - 1);
}

void LaunchMessageRingBufferState::set_unicast_wptr(uint32_t val) {
    unicast_cores_launch_message_wptr_ = val & (launch_msg_buffer_num_entries - 1);
}

uint32_t LaunchMessageRingBufferState::get_mcast_wptr() const { return multicast_cores_launch_message_wptr_; }

uint32_t LaunchMessageRingBufferState::get_unicast_wptr() const { return unicast_cores_launch_message_wptr_; }

void LaunchMessageRingBufferState::reset() {
    multicast_cores_launch_message_wptr_ = 0;
    unicast_cores_launch_message_wptr_ = 0;
}

}  // namespace tt::tt_metal
