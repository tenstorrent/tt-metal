// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace tt::tt_metal {

// Used to track the position of the current launch message to write to in the launch message ring buffer on the cores
// we dispatch programs to.
// The multicast cores and unicast cores are tracked separately to not need to modify state when we only need to
// dispatch to one of the core groups.
class LaunchMessageRingBufferState {
public:
    void inc_mcast_wptr(uint32_t inc_val);

    void inc_unicast_wptr(uint32_t inc_val);

    void set_mcast_wptr(uint32_t val);

    void set_unicast_wptr(uint32_t val);

    uint32_t get_mcast_wptr() const;

    uint32_t get_unicast_wptr() const;

    void reset();

private:
    uint32_t multicast_cores_launch_message_wptr_ = 0;
    uint32_t unicast_cores_launch_message_wptr_ = 0;
};

}  // namespace tt::tt_metal
