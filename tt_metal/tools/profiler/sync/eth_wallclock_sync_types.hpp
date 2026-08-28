// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Layout shared by the eth wall-clock sync kernels and the host that reads their samples out of L1.
// Deliberately free of device-side includes so the host can include it directly.

#pragma once

#include <cstdint>

namespace tt::tt_metal::eth_sync {

constexpr uint32_t kEthSyncMagic = 0x53594E43;  // 'SYNC'

enum EthSyncStatus : uint32_t {
    ETH_SYNC_IDLE = 0,
    ETH_SYNC_RUNNING = 1,
    ETH_SYNC_DONE = 2,
    ETH_SYNC_TIMEOUT_HANDSHAKE = 3,  // peer never joined
    ETH_SYNC_TIMEOUT_TXQ = 4,        // local tx queue never drained
    ETH_SYNC_TIMEOUT_WAIT = 5,       // message or echo never arrived
};

// 32 B per sample so host-side indexing is a shift. The sender fills t0/t2, the receiver fills t1; each
// side leaves the other's fields zero, and the host joins them by index.
struct EthSyncSample {
    uint32_t t0_hi, t0_lo;
    uint32_t t2_hi, t2_lo;
    uint32_t t1_hi, t1_lo;
    uint32_t pad0, pad1;
};

struct EthSyncResult {
    uint32_t magic;      // kEthSyncMagic once the kernel starts: proves it ran at all
    uint32_t status;     // EthSyncStatus
    uint32_t n_samples;  // samples actually written
    uint32_t n_wanted;   // samples requested, so a partial run is obvious
    // Followed by n_wanted EthSyncSample entries.
};

}  // namespace tt::tt_metal::eth_sync
