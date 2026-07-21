// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal::distributed {

inline constexpr uint32_t kHDSocketConnectorStateVersion = 1;

// Persistent connector-side state for H2D/D2H sockets. Lives in the socket's
// named SHM region so that consecutive driver processes (which connect via
// H2DSocket::connect / D2HSocket::connect) can pick up where the previous
// process left off, rather than restarting counters at zero and desyncing
// from the device-resident pointers.
//
// Layout is fixed at 64 bytes (one cache line) so the trailing _pad never
// needs to grow when fields are added; future fields take from _pad and bump
// the version when semantics change.
struct alignas(64) HDSocketConnectorState {
    uint32_t version;         // kHDSocketConnectorStateVersion
    uint32_t page_size;       // 0 = unset
    uint32_t fifo_curr_size;  // 0 = unset
    uint32_t bytes_sent;      // H2D: live counter; D2H: unused
    uint32_t bytes_acked;     // D2H: live counter; H2D: unused
    uint32_t write_ptr;       // H2D only
    uint32_t read_ptr;        // D2H only
    // Set to 0 on open (owner construct stamps 1 since no connector is yet attached;
    // connect() writes 0 after reading the prior value); set to 1 in the destructor.
    // A connector that reads 0 here knows the previous process exited without running
    // its destructor (crash, _exit, kill).
    uint32_t clean_shutdown;
    uint32_t _pad[8];
};
static_assert(sizeof(HDSocketConnectorState) == 64, "HDSocketConnectorState must be 64 bytes");
static_assert(alignof(HDSocketConnectorState) == 64, "HDSocketConnectorState must be cache-line aligned");

}  // namespace tt::tt_metal::distributed
