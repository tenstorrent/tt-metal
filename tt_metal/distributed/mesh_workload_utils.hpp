// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>

namespace tt::tt_metal::distributed {

namespace experimental {
// Utility functions for writing program dispatch commands
// and go signals through the per device CQ.
// Usage of these functions is temporary, until the MeshCQ
// can function independently and support MeshBuffer reads and
// writes.
void write_program_commands(
    CommandQueue& cq,
    ProgramCommandSequence& program_cmd_seq,
    uint32_t num_active_cores_in_program,
    SubDeviceId sub_device_id,
    bool stall_first,
    bool stall_before_program,
    bool blocking);

void write_go_signal(
    CommandQueue& cq,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    bool send_mcast,
    bool send_unicasts,
    int num_unicast_txns = -1);
}  // namespace experimental

}  // namespace tt::tt_metal::distributed
