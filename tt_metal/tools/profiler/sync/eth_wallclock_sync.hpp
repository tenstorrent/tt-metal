// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Device-to-device wall-clock sync over one ethernet link: Cristian's algorithm measured by the eriscs at
// both ends, with the samples left in eth L1 for the host to read directly.
//
// Per round trip the sender stamps t0 before the message leaves and t2 when the echo lands; the receiver
// stamps t1 when it arrives. Then
//
//     offset = t1 - (t0 + t2) / 2          rtt = t2 - t0
//
// The bracket midpoint cancels the link latency to first order, so the estimate is only as good as the
// SYMMETRY of the two directions -- which is why an eth link is the right place to measure this and a PCIe
// MMIO round trip is not. Many samples are taken so the host can filter and regress rather than trust one.
//
// WHERE THIS RUNS. At profiler init, before fabric is brought up and before any drainer is resident. That
// window matters three ways: the eth cores are still free (fabric routers take them later), no DRAM core is
// in stream mode yet, and the results can be read straight out of L1 over MMIO -- no drainer, no marker
// transport, no dependency on the profiler being able to capture anything at all.
//
// WHY NOT sync/sync_device_kernel_{sender,receiver}.cpp. Those measure each round trip as a profiler ZONE,
// so the numbers can only be recovered by reading zones back through the DRAM profiler -- the backend the
// streaming profiler stands down -- and they yield a duration per side rather than the (t0, t1, t2) triple
// the algorithm needs, paired by the unchecked assumption that the i-th zone on one side matches the i-th
// on the other. Worse, every wait in them is unbounded: a dropped message or an absent peer hangs the
// kernel forever. Measured on a 4-card box: the pair launches, never signals done, and teardown then fails
// with "Timed out while waiting for active ethernet core to become active again", needing a board reset.
// Here every wait carries a deadline, so a bad link costs a status word and a warning.

#pragma once

#include <cstdint>
#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "tools/profiler/sync/eth_wallclock_sync_types.hpp"

namespace tt::tt_metal::eth_sync {

// ---- Wall clock ---------------------------------------------------------------------------------------
//
// The SAME counter kernel zones are stamped with, read the same way: reading L latches H, so read L first
// and H's own latency does not matter. Local on purpose rather than including kernel_profiler.hpp -- these
// samples must not depend on the profiler being enabled, and this runs before any drainer exists.
constexpr uint32_t kWallClockL = 0xFFB121F0;
constexpr uint32_t kWallClockH = 0xFFB121F8;

inline __attribute__((always_inline)) void read_wall_clock(uint32_t& hi, uint32_t& lo) {
    lo = *reinterpret_cast<volatile uint32_t*>(kWallClockL);  // latches H
    hi = *reinterpret_cast<volatile uint32_t*>(kWallClockH);
}

inline __attribute__((always_inline)) uint64_t now64() {
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

namespace detail {

inline volatile EthSyncResult* result_at(uint32_t addr) { return reinterpret_cast<volatile EthSyncResult*>(addr); }
inline EthSyncSample* samples_at(uint32_t addr) {
    return reinterpret_cast<EthSyncSample*>(addr + sizeof(EthSyncResult));
}

inline void publish(uint32_t addr, uint32_t status, uint32_t n_done) {
    volatile EthSyncResult* r = result_at(addr);
    r->n_samples = n_done;
    r->status = status;  // written last: the host reads status, then samples
}

// Bounded tx-queue spin. Every send goes through it, so a wedged queue costs a status word, not the card.
inline bool txq_idle_bounded(uint64_t deadline) {
    while (eth_txq_is_busy()) {
        if (now64() >= deadline) {
            return false;
        }
    }
    return true;
}

// Bounded handshake, mirroring eth_setup_handshake() but with a deadline.
//
// Two details of the real primitives are load-bearing, and getting them wrong made both ends time out here
// on the first attempt. First, the handshake rides erisc_info->channels[0].bytes_sent -- the eth firmware's
// own channel state -- not an arbitrary L1 word. Second, every wait must call run_routing(): that is the
// firmware's link service, and without it the packets never move, so both sides sit waiting for something
// that will never arrive.
inline bool eth_wait_for_bytes_bounded(uint32_t num_bytes, uint64_t deadline) {
    while (erisc_info->channels[0].bytes_sent != num_bytes) {
        invalidate_l1_cache();
        run_routing();
        if (now64() >= deadline) {
            return false;
        }
    }
    return true;
}

inline bool eth_wait_for_receiver_done_bounded(uint64_t deadline) {
    internal_::eth_send_packet(
        0,
        ((uint32_t)(uintptr_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
        ((uint32_t)(uintptr_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
        1);
    while (erisc_info->channels[0].bytes_sent != 0) {
        invalidate_l1_cache();
        run_routing();
        if (now64() >= deadline) {
            return false;
        }
    }
    return true;
}

inline bool handshake_bounded(uint32_t handshake_addr, bool is_sender, uint64_t deadline) {
    if (is_sender) {
        if (!txq_idle_bounded(deadline)) {
            return false;
        }
        eth_send_bytes(handshake_addr, handshake_addr, 16);
        return eth_wait_for_receiver_done_bounded(deadline);
    }
    if (!eth_wait_for_bytes_bounded(16, deadline)) {
        return false;
    }
    eth_receiver_channel_done(0);
    return true;
}

}  // namespace detail

// ---- The two entry points -----------------------------------------------------------------------------
//
// One per link end, same n_samples on both. `result_addr` holds EthSyncResult + n_samples samples;
// `channel_addr` is the scratch the message moves through; `handshake_addr` is where the two sides meet.
// `timeout_cycles` bounds the WHOLE run, so a slow link yields fewer samples rather than a longer stall.

// gap_cycles spreads the samples in time. It exists because the RATE half of the solve is
// baseline-limited: back-to-back round trips span microseconds, and a frequency difference of a few ppm
// simply does not show up over that. The host<->device sync learned the same lesson -- at zero spacing its
// baseline was ~360 us and the fitted frequency carried ~1e-4 of error, which then grew with time since
// the anchor. Pacing costs nothing but wall time and buys a fit worth having.
inline bool eth_wallclock_sync_sender(
    uint32_t result_addr,
    uint32_t channel_addr,
    uint32_t handshake_addr,
    uint32_t n_samples,
    uint64_t timeout_cycles,
    uint32_t gap_cycles) {
    volatile EthSyncResult* res = detail::result_at(result_addr);
    EthSyncSample* samples = detail::samples_at(result_addr);
    res->magic = kEthSyncMagic;
    res->n_wanted = n_samples;
    res->n_samples = 0;
    res->status = ETH_SYNC_RUNNING;

    const uint64_t deadline = now64() + timeout_cycles;
    volatile eth_channel_sync_t* sync = reinterpret_cast<volatile eth_channel_sync_t*>(channel_addr);

    if (!detail::handshake_bounded(handshake_addr, /*is_sender=*/true, deadline)) {
        detail::publish(result_addr, ETH_SYNC_TIMEOUT_HANDSHAKE, 0);
        return false;
    }

    uint32_t done = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        // Start each iteration on an empty queue: an iteration that shares the queue with its predecessor
        // measures that predecessor's drain, not the link.
        if (!detail::txq_idle_bounded(deadline)) {
            detail::publish(result_addr, ETH_SYNC_TIMEOUT_TXQ, done);
            return false;
        }

        sync->bytes_sent = 1;
        sync->receiver_ack = 0;

        uint32_t t0_hi, t0_lo;
        read_wall_clock(t0_hi, t0_lo);
        eth_send_bytes_over_channel_payload_only(
            channel_addr,
            channel_addr,
            sizeof(eth_channel_sync_t),
            sizeof(eth_channel_sync_t),
            sizeof(eth_channel_sync_t) >> 4);

        // The receiver's echo clears bytes_sent. Stamp the instant the flip is OBSERVED, mirroring the
        // receiver's placement -- the midpoint only means anything if both sides stamp symmetrically.
        bool ok = true;
        while (sync->bytes_sent != 0) {
            invalidate_l1_cache();
            if (now64() >= deadline) {
                ok = false;
                break;
            }
        }
        uint32_t t2_hi, t2_lo;
        read_wall_clock(t2_hi, t2_lo);
        if (!ok) {
            detail::publish(result_addr, ETH_SYNC_TIMEOUT_WAIT, done);
            return false;
        }

        samples[i].t0_hi = t0_hi;
        samples[i].t0_lo = t0_lo;
        samples[i].t2_hi = t2_hi;
        samples[i].t2_lo = t2_lo;
        samples[i].t1_hi = 0;
        samples[i].t1_lo = 0;
        done = i + 1;

        // Pace to the next slot. Bounded by the same deadline, so pacing can never outlive the run.
        if (gap_cycles != 0 && i + 1 < n_samples) {
            const uint64_t next = ((static_cast<uint64_t>(t0_hi) << 32) | t0_lo) + gap_cycles;
            while (now64() < next) {
                if (now64() >= deadline) {
                    detail::publish(result_addr, ETH_SYNC_DONE, done);
                    return true;  // out of time, not a failure: the samples taken are still good
                }
            }
        }
    }

    detail::publish(result_addr, ETH_SYNC_DONE, done);
    return true;
}

inline bool eth_wallclock_sync_receiver(
    uint32_t result_addr,
    uint32_t channel_addr,
    uint32_t handshake_addr,
    uint32_t n_samples,
    uint64_t timeout_cycles) {
    volatile EthSyncResult* res = detail::result_at(result_addr);
    EthSyncSample* samples = detail::samples_at(result_addr);
    res->magic = kEthSyncMagic;
    res->n_wanted = n_samples;
    res->n_samples = 0;
    res->status = ETH_SYNC_RUNNING;

    const uint64_t deadline = now64() + timeout_cycles;
    volatile eth_channel_sync_t* sync = reinterpret_cast<volatile eth_channel_sync_t*>(channel_addr);
    sync->bytes_sent = 0;
    sync->receiver_ack = 0;

    if (!detail::handshake_bounded(handshake_addr, /*is_sender=*/false, deadline)) {
        detail::publish(result_addr, ETH_SYNC_TIMEOUT_HANDSHAKE, 0);
        return false;
    }

    uint32_t done = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        // ONE wait, stamped the instant it clears. The legacy receiver spins on channel 0 before its
        // per-channel loop and again inside it, so its stamp sits behind two polls while the sender's sits
        // behind one -- an asymmetry that lands straight in the offset.
        bool ok = true;
        while (sync->bytes_sent == 0) {
            invalidate_l1_cache();
            if (now64() >= deadline) {
                ok = false;
                break;
            }
        }
        uint32_t t1_hi, t1_lo;
        read_wall_clock(t1_hi, t1_lo);
        if (!ok) {
            detail::publish(result_addr, ETH_SYNC_TIMEOUT_WAIT, done);
            return false;
        }

        samples[i].t1_hi = t1_hi;
        samples[i].t1_lo = t1_lo;
        samples[i].t0_hi = 0;
        samples[i].t0_lo = 0;
        samples[i].t2_hi = 0;
        samples[i].t2_lo = 0;
        done = i + 1;

        // Echo through the SAME channel so the return path mirrors the outbound one; asymmetric paths
        // break the midpoint estimate.
        sync->bytes_sent = 0;
        sync->receiver_ack = 0;
        if (!detail::txq_idle_bounded(deadline)) {
            detail::publish(result_addr, ETH_SYNC_TIMEOUT_TXQ, done);
            return false;
        }
        eth_send_bytes_over_channel_payload_only(
            channel_addr,
            channel_addr,
            sizeof(eth_channel_sync_t),
            sizeof(eth_channel_sync_t),
            sizeof(eth_channel_sync_t) >> 4);
    }

    detail::publish(result_addr, ETH_SYNC_DONE, done);
    return true;
}

}  // namespace tt::tt_metal::eth_sync
