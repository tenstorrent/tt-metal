// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRISC side of the DRISC microbenchmark (see test_drisc_microbench.cpp for the phases and how to read the results).
// The peer is an ERISC on an active Ethernet core. NoC1 tests run only when this DRISC's NIU1 is in stream mode.

#include <cstdint>
#include <type_traits>

#include "experimental/drisc_mode.h"
#include "drisc_mb_device.hpp"

using namespace drisc_mb;

namespace {

#if DRISC_MB_PART == 1
void measure_drisc_local(Results& res, uint32_t base, uint32_t n) {
    const uint32_t streams[4] = {kDriscScratchStream, kDriscScratchStream2, kDriscDoorbellStream, kDriscSyncStream};
    measure_local(
        res,
        LocalEntries{D_TIMER_READ, D_L1_LOAD_DEP, D_L1_LOAD_INDEP4, D_STREAM_READ, D_STREAM_READ_INDEP4, D_SINK},
        base + kScratchOffset,
        streams,
        n);
    uint32_t sink = 0;

    uint32_t start = now();
    for (uint32_t i = 0; i < n; i++) {
        sink += get_timestamp_32b();
    }
    res.put(D_WALL_CLOCK_READ, now() - start, n);

    // Timer against the wall clock over ~64K wall-clock ticks (same counter if built with DRISC_MB_WALL_CLOCK).
    const uint32_t wall0 = get_timestamp_32b();
    const uint32_t t0 = now();
    while (get_timestamp_32b() - wall0 < (1u << 16)) {
    }
    const uint32_t t1 = now();
    const uint32_t wall1 = get_timestamp_32b();
    res.put(D_CALIBRATION, t1 - t0, wall1 - wall0);

    volatile tt_l1_ptr uint32_t* word = l1<uint32_t>(base + kScratchOffset + 32);
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        invalidate_l1_cache();
        sink += *word;
    }
    res.put(D_L1_POLL_FENCE, now() - start, n);

    start = now();
    for (uint32_t i = 0; i < n; i++) {
        stream_add_local(kDriscScratchStream, 1);
    }
    res.put(D_STREAM_LOCAL_UPDATE, now() - start, n);
    stream_reset(kDriscScratchStream);

    start = now();
    for (uint32_t i = 0; i < n; i++) {
        sink += noc_cmd_buf_ready(noc_index, write_cmd_buf);
    }
    res.put(D_CMD_BUF_READY, now() - start, n);
    res.put(D_SINK, res.words[2 * D_SINK] + sink, 0);
}

// DRISC -> ERISC writes and reads on one NoC. The ERISC idles meanwhile; its landing ring and streams absorb them.
void measure_writes(
    Results& res,
    uint32_t entry,
    uint8_t noc,
    uint32_t base,
    uint32_t eth_x,
    uint32_t eth_y,
    uint32_t eth_base,
    uint32_t payload,
    uint32_t n,
    uint32_t n_rt,
    uint32_t num_slots) {
    const uint32_t src = base + kBufferOffset;
    const uint64_t landing = get_noc_addr(eth_x, eth_y, eth_base + kEthLandingOffset, noc);
    const uint64_t eth_src = get_noc_addr(eth_x, eth_y, eth_base + kEthSrcOffset, noc);
    const uint64_t doorbell = stream_update_addr(eth_x, eth_y, kEthDoorbellStream, noc);
    const uint64_t bare_doorbell = stream_update_addr(eth_x, eth_y, kEthScratchStream, noc);

    // Back-to-back writes of one size: issue cost, then the time until every ack is back.
    auto writes = [&](uint32_t issue_entry, uint32_t size, auto posted) {
        uint32_t slot = 0;
        const uint32_t start = now();
        for (uint32_t i = 0; i < n; i++) {
            noc_async_write_one_packet<true, decltype(posted)::value>(src, landing + slot * kSlotBytes, size, noc);
            slot = next_slot(slot, kEthLandingSlots);
        }
        const uint32_t issued = now();
        if constexpr (decltype(posted)::value) {
            noc_async_posted_writes_flushed(noc);
        } else {
            noc_async_write_barrier(noc);
        }
        const uint32_t drained = now();
        res.put(issue_entry, issued - start, n);
        res.put(issue_entry + 1, drained - start, n);
    };
    writes(entry + DN_WR64_ISSUE, 64, std::false_type{});
    writes(entry + DN_WR_ISSUE, payload, std::false_type{});
    writes(entry + DN_WR_POSTED_ISSUE, payload, std::true_type{});

    uint32_t start = now();
    for (uint32_t i = 0; i < n; i++) {
        stream_add_remote(bare_doorbell, 1, noc);
    }
    uint32_t issued = now();
    noc_async_write_barrier(noc);
    res.put(entry + DN_DOORBELL_ISSUE, issued - start, n);
    res.put(entry + DN_DOORBELL_DRAIN, now() - start, n);

    // A dumb forwarder's copy, as the fabric forwards today: payload write with a transaction id, then the
    // downstream doorbell on the same NoC, which lands after the payload.
    uint32_t slot = 0;
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        noc_async_write_one_packet_with_trid(
            src, landing + slot * kSlotBytes, payload, slot_trid(slot), write_cmd_buf, noc);
        stream_add_remote(doorbell, 1, noc);
        slot = next_slot(slot, kEthLandingSlots);
    }
    issued = now();
    noc_async_write_barrier(noc);
    res.put(entry + DN_FWD_ISSUE, issued - start, n);
    res.put(entry + DN_FWD_DRAIN, now() - start, n);

    // The same copy as a forwarder with one fixed downstream would issue it: the destination is set once, so each
    // copy programs only addresses, size and transaction id. Doorbells use another command buffer, which keeps the
    // state. Then one copy at a time with the NoC idle: what a copy costs the CPU when the port is free.
    const uint32_t landing_l1 = eth_base + kEthLandingOffset;
    noc_async_write_one_packet_with_trid_set_state(landing, write_cmd_buf, noc);
    slot = 0;
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        noc_async_write_one_packet_with_trid_with_state(
            src, landing_l1 + slot * kSlotBytes, payload, slot_trid(slot), write_cmd_buf, noc);
        stream_add_remote(doorbell, 1, noc);
        slot = next_slot(slot, kEthLandingSlots);
    }
    issued = now();
    noc_async_write_barrier(noc);
    res.put(entry + DN_SF_FWD_ISSUE, issued - start, n);
    res.put(entry + DN_SF_FWD_DRAIN, now() - start, n);
    auto landing_slot = [=](uint32_t i) {
        const uint32_t s = i % kEthLandingSlots;
        return OpArgs{src, landing_l1 + s * kSlotBytes, 0, slot_trid(s)};
    };
    res.put(
        entry + DN_SF_FWD_ALONE,
        time_alone(
            n,
            noc,
            landing_slot,
            [=](const OpArgs& a) {
                noc_async_write_one_packet_with_trid_with_state(a.src, a.dst, payload, a.trid, write_cmd_buf, noc);
                stream_add_remote(doorbell, 1, noc);
            }),
        n);
    res.put(
        entry + DN_DOORBELL_ALONE,
        time_alone(n, noc, no_args, [=](const OpArgs&) { stream_add_remote(bare_doorbell, 1, noc); }),
        n);

    // The same copy with lean stateful issue: one command-buffer poll per write, and a stateful doorbell.
    lean_write_set_state(landing, payload, noc);
    lean_doorbell_set_state(doorbell, noc);
    slot = 0;
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        lean_trid_write(src, landing_l1 + slot * kSlotBytes, slot_trid(slot), noc);
        lean_doorbell(noc);
        slot = next_slot(slot, kEthLandingSlots);
    }
    issued = now();
    noc_async_write_barrier(noc);
    res.put(entry + DN_LEAN_FWD_ISSUE, issued - start, n);
    res.put(entry + DN_LEAN_FWD_DRAIN, now() - start, n);
    res.put(
        entry + DN_LEAN_FWD_ALONE,
        time_alone(
            n,
            noc,
            landing_slot,
            [=](const OpArgs& a) {
                lean_trid_write(a.src, a.dst, a.trid, noc);
                lean_doorbell(noc);
            }),
        n);

    start = now();
    for (uint32_t i = 0; i < n_rt; i++) {
        noc_async_write_one_packet(src, landing, payload, noc);
        noc_async_write_barrier(noc);
    }
    res.put(entry + DN_WR_RTT, now() - start, n_rt);

    // Pulls: read the ERISC's source buffer into the staging ring.
    slot = 0;
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        noc_async_read_one_packet(eth_src, base + kBufferOffset + slot * kSlotBytes, payload, noc);
        slot = next_slot(slot, num_slots);
    }
    issued = now();
    noc_async_read_barrier(noc);
    res.put(entry + DN_RD_ISSUE, issued - start, n);
    res.put(entry + DN_RD_DRAIN, now() - start, n);

    start = now();
    for (uint32_t i = 0; i < n_rt; i++) {
        noc_async_read_one_packet(eth_src, base + kBufferOffset, payload, noc);
        noc_async_read_barrier(noc);
    }
    res.put(entry + DN_RD_RTT, now() - start, n_rt);
}

#endif  // DRISC_MB_PART == 1

#if DRISC_MB_PART == 3
// Write command buffers in the order a forwarder would add them: the write buffer, the register-write buffer, then
// the atomic/inline buffer. Packed in nibbles.
constexpr uint32_t kCmdBufOrder = write_cmd_buf | (write_reg_cmd_buf << 4) | (write_at_cmd_buf << 8);
FORCE_INLINE uint32_t cmd_buf_at(uint32_t i) { return (kCmdBufOrder >> (4 * i)) & 0xF; }

// Back-to-back payload writes into the ERISC's landing ring on `noc`, round-robin over the first `k` command
// buffers (state already set). Puts {issue, drain} at `entry`, `entry + 1`.
void writes_round_robin(
    Results& res, uint32_t entry, uint8_t noc, uint32_t k, uint32_t src, uint32_t landing_l1, uint32_t n) {
    uint32_t c = 0, slot = 0;
    const uint32_t start = now();
    for (uint32_t i = 0; i < n; i++) {
        lean_trid_write(src, landing_l1 + slot * kSlotBytes, slot_trid(slot), noc, cmd_buf_at(c));
        c = c + 1 == k ? 0 : c + 1;
        slot = next_slot(slot, kEthLandingSlots);
    }
    const uint32_t issued = now();
    noc_async_write_barrier(noc);
    res.put(entry, issued - start, n);
    res.put(entry + 1, now() - start, n);
}

// How far the DRISC's command buffers let it run ahead of its NoC ports, and what its ports carry when used together.
void measure_cmd_bufs(
    Results& res,
    uint32_t base,
    uint32_t eth_x,
    uint32_t eth_y,
    uint32_t eth_base,
    uint32_t payload,
    uint32_t n,
    bool noc1_ok,
    uint32_t num_slots) {
    const uint32_t src = base + kBufferOffset;
    const uint32_t landing_l1 = eth_base + kEthLandingOffset;
    const uint32_t nocs = noc1_ok ? 2 : 1;
    for (uint8_t wn = 0; wn < nocs; wn++) {
        const uint64_t landing = get_noc_addr(eth_x, eth_y, landing_l1, wn);
        const uint32_t entry = D_NOC_BASE + wn * DCN_COUNT;
        for (uint32_t c = 0; c < kMaxWriteCmdBufs; c++) {
            lean_write_set_state(landing, payload, wn, cmd_buf_at(c));
        }
        // One write with the NoC idle, then wait until its command buffer accepts another.
        uint32_t hold = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t dst = landing_l1 + (i % kEthLandingSlots) * kSlotBytes;
            noc_async_write_barrier(wn);
            asm volatile("" : "+r"(dst));
            const uint32_t t0 = now();
            lean_trid_write(src, dst, 1, wn);
            while (!noc_cmd_buf_ready(wn, write_cmd_buf));
            hold += now() - t0;
        }
        noc_async_write_barrier(wn);
        res.put(entry + DCN_HOLD, hold, n);
        for (uint32_t k = 1; k <= kMaxWriteCmdBufs; k++) {
            writes_round_robin(res, entry + DCN_WR_K + 2 * (k - 1), wn, k, src, landing_l1, n);
        }
    }
    if (!noc1_ok) {
        return;
    }

    // Both NoCs at once: writes alternate NoC0 / NoC1, each NoC round-robin over k = 1, 2 command buffers.
    for (uint32_t k = 1; k <= 2; k++) {
        for (uint8_t wn = 0; wn < 2; wn++) {
            for (uint32_t c = 0; c < k; c++) {
                lean_write_set_state(get_noc_addr(eth_x, eth_y, landing_l1, wn), payload, wn, cmd_buf_at(c));
            }
        }
        uint32_t c = 0, slot = 0;
        const uint32_t start = now();
        for (uint32_t i = 0; i < n; i += 2) {
            lean_trid_write(src, landing_l1 + slot * kSlotBytes, slot_trid(slot), 0, cmd_buf_at(c));
            lean_trid_write(src, landing_l1 + slot * kSlotBytes, slot_trid(slot), 1, cmd_buf_at(c));
            c = c + 1 == k ? 0 : c + 1;
            slot = next_slot(slot, kEthLandingSlots);
        }
        const uint32_t issued = now();
        noc_async_write_barrier(0);
        noc_async_write_barrier(1);
        res.put(DC_BOTH + 2 * (k - 1), issued - start, n);
        res.put(DC_BOTH + 2 * (k - 1) + 1, now() - start, n);
    }

    // Pull + forward: reads of the ERISC's source buffer into our staging slots, interleaved with writes into its
    // landing ring. Variants: {read NoC, write NoC, write command buffers}.
    constexpr uint8_t kReadNoc[kDuplexVariants] = {0, 0, 0, 0};
    constexpr uint8_t kWriteNoc[kDuplexVariants] = {0, 0, 1, 1};
    constexpr uint32_t kWriteBufs[kDuplexVariants] = {1, 2, 1, 2};
    for (uint32_t v = 0; v < kDuplexVariants; v++) {
        const uint8_t rn = kReadNoc[v], wn = kWriteNoc[v];
        const uint32_t k = kWriteBufs[v];
        for (uint32_t c = 0; c < k; c++) {
            lean_write_set_state(get_noc_addr(eth_x, eth_y, landing_l1, wn), payload, wn, cmd_buf_at(c));
        }
        const uint64_t eth_src = get_noc_addr(eth_x, eth_y, eth_base + kEthSrcOffset, rn);
        uint32_t c = 0, slot = 0, rslot = 1;
        const uint32_t start = now();
        for (uint32_t i = 0; i < n; i++) {
            noc_async_read_one_packet(eth_src, base + kBufferOffset + rslot * kSlotBytes, payload, rn);
            lean_trid_write(src, landing_l1 + slot * kSlotBytes, slot_trid(slot), wn, cmd_buf_at(c));
            c = c + 1 == k ? 0 : c + 1;
            slot = next_slot(slot, kEthLandingSlots);
            rslot = rslot + 1 == num_slots ? 1 : rslot + 1;
        }
        const uint32_t issued = now();
        noc_async_read_barrier(rn);
        noc_async_write_barrier(wn);
        res.put(DC_DUPLEX + 2 * v, issued - start, n);
        res.put(DC_DUPLEX + 2 * v + 1, now() - start, n);
    }
}

// Pull from the ERISC we run with (A) and forward into other Ethernet cores' landing rings (B, C), per
// kFwdVariantList. The copies' source is our staging slot 0, which holds A's source pattern; pulls land in slots 1+.
void measure_forwarding(
    Results& res,
    uint32_t base,
    uint32_t a_x,
    uint32_t a_y,
    uint32_t eth_base,
    uint32_t payload,
    uint32_t n,
    bool noc1_ok,
    uint32_t num_slots,
    uint32_t fwd_eriscs,
    uint32_t b_x,
    uint32_t b_y,
    uint32_t c_x,
    uint32_t c_y) {
    const uint32_t src = base + kBufferOffset;
    const uint32_t landing_l1 = eth_base + kEthLandingOffset;
    const uint64_t a_src = get_noc_addr(a_x, a_y, eth_base + kEthSrcOffset, 0);
    noc_async_read_one_packet(a_src, src, payload, 0);
    noc_async_read_barrier(0);
    for (uint32_t v = 0; v < kFwdVariants; v++) {
        const FwdVariant fv = kFwdVariantList[v];
        if ((fv.c_noc >= 0 && fwd_eriscs < 2) || ((fv.b_noc == 1 || fv.c_noc == 1) && !noc1_ok)) {
            continue;
        }
        const uint8_t bn = fv.b_noc;
        const bool to_c = fv.c_noc >= 0;
        const uint8_t cn = to_c ? fv.c_noc : 0;
        // B and C on the same NoC need separate command buffers, since each keeps its destination.
        const uint32_t c_buf = to_c && cn == bn ? write_reg_cmd_buf : write_cmd_buf;
        lean_write_set_state(get_noc_addr(b_x, b_y, landing_l1, bn), payload, bn, write_cmd_buf);
        if (to_c) {
            lean_write_set_state(get_noc_addr(c_x, c_y, landing_l1, cn), payload, cn, c_buf);
        }
        uint32_t slot = 0, rslot = 1;
        const uint32_t start = now();
        for (uint32_t i = 0; i < n; i++) {
            if (fv.pull) {
                noc_async_read_one_packet(a_src, base + kBufferOffset + rslot * kSlotBytes, payload, 0);
                rslot = rslot + 1 == num_slots ? 1 : rslot + 1;
            }
            lean_trid_write(src, landing_l1 + slot * kSlotBytes, slot_trid(slot), bn, write_cmd_buf);
            if (to_c) {
                lean_trid_write(src, landing_l1 + slot * kSlotBytes, slot_trid(slot), cn, c_buf);
            }
            slot = next_slot(slot, kEthLandingSlots);
        }
        const uint32_t issued = now();
        noc_async_read_barrier(0);
        noc_async_write_barrier(0);
        if (noc1_ok) {
            noc_async_write_barrier(1);
        }
        res.put(DC_FWD + 2 * v, issued - start, n);
        res.put(DC_FWD + 2 * v + 1, now() - start, n);
    }
}
#endif  // DRISC_MB_PART == 3

}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(ARG_MY_BASE);
    const uint32_t eth_x = get_arg_val<uint32_t>(ARG_PEER_X);
    const uint32_t eth_y = get_arg_val<uint32_t>(ARG_PEER_Y);
    const uint32_t eth_base = get_arg_val<uint32_t>(ARG_PEER_BASE);
    const uint32_t payload = get_arg_val<uint32_t>(ARG_PAYLOAD);
    [[maybe_unused]] const uint32_t n = get_arg_val<uint32_t>(ARG_N);
    [[maybe_unused]] const uint32_t n_rt = get_arg_val<uint32_t>(ARG_N_RT);
    const bool noc1_ok = get_arg_val<uint32_t>(ARG_DRISC_NOC1) != 0;
    [[maybe_unused]] const uint32_t num_slots = get_arg_val<uint32_t>(ARG_DRISC_SLOTS);
    [[maybe_unused]] const uint32_t pipe_pkts = get_arg_val<uint32_t>(ARG_PIPE_PKTS);
    [[maybe_unused]] const uint32_t num_depths = get_arg_val<uint32_t>(ARG_NUM_DEPTHS);

    Results res{l1<uint32_t>(base + kResultsOffset), D_ERROR};
    const uint8_t noc = noc_index;  // NoC0: NIU0 is in stream mode on every DRISC Metal owns
    if (experimental::drisc_is_noc2axi_mode(noc)) {
        res.fail(100);
        return;
    }
    if (noc1_ok) {
        if (experimental::drisc_is_noc2axi_mode(1 - noc)) {
            res.fail(101);
            return;
        }
        // The kernel entry re-syncs only NOC_INDEX's transaction counters with the NIU; do the other NoC too.
        noc_local_state_init(1 - noc);
    }
    // On every exit: wait for our transactions, then leave the command buffers as firmware's noc_init set them
    // (it runs once at boot, and the NoC API relies on that state) with transaction id 0.
    struct RestoreNocs {
        uint8_t noc;
        bool noc1_ok;
        ~RestoreNocs() {
            for (uint8_t n = 0; n < NUM_NOCS; n++) {
                if (n != noc && !noc1_ok) {
                    continue;
                }
                noc_async_write_barrier(n);
                noc_async_read_barrier(n);
                noc_clear_packet_tags(n);
                noc_init_one(n, MEM_NOC_ATOMIC_RET_VAL_ADDR);
            }
        }
    } restore_nocs{noc, noc1_ok};

    const uint32_t my_streams[4] = {kDriscSyncStream, kDriscDoorbellStream, kDriscScratchStream, kDriscScratchStream2};
    for (uint32_t s : my_streams) {
        stream_reset(s);
    }
    if (!handshake(base, get_noc_addr(eth_x, eth_y, eth_base + kReadyOffset, noc), noc)) {
        res.fail(0);
        return;
    }
    // The peer is the ERISC, or in part 4 the other DRISC.
    constexpr uint32_t kPeerSyncStream = DRISC_MB_PART == 4 ? kDriscSyncStream : kEthSyncStream;
    Peer peer{stream_update_addr(eth_x, eth_y, kPeerSyncStream, noc), kDriscSyncStream, noc, 0};
    uint32_t phase = 1;
    auto barrier = [&]() {
        if (!peer.barrier()) {
            res.fail(phase);
            return false;
        }
        phase++;
        return true;
    };
    if (!barrier()) {
        return;
    }
    [[maybe_unused]] const uint64_t eth_doorbell = stream_update_addr(eth_x, eth_y, kEthDoorbellStream, noc);
    [[maybe_unused]] const uint64_t landing = get_noc_addr(eth_x, eth_y, eth_base + kEthLandingOffset, noc);

#if DRISC_MB_PART == 1
    // Phase: DRISC-local costs (the ERISC waits).
    measure_drisc_local(res, base, n);
    if (!barrier()) {
        return;
    }

    // Phase: DRISC -> ERISC writes and reads, per NoC.
    for (uint8_t wn = 0; wn < 2; wn++) {
        if (wn != noc && !noc1_ok) {
            continue;
        }
        measure_writes(res, D_NOC_BASE + wn * DN_COUNT, wn, base, eth_x, eth_y, eth_base, payload, n, n_rt, num_slots);
    }
    if (!barrier()) {  // the ERISC checks and resets its streams after this
        return;
    }

    // Phase: the ERISC pushes into this DRISC; check what landed, then reset before anyone rings again.
    if (!barrier()) {
        return;
    }
    // The ERISC rang the doorbell for each push: stateless, stateful and lean, each back to back and alone. Bare
    // doorbells: back to back, alone, lean alone.
    res.put(D_CHECK_PUSH_DOORBELLS, stream_value(kDriscDoorbellStream), 6 * n);
    res.put(D_CHECK_BARE_DOORBELLS, stream_value(kDriscScratchStream), 3 * n);
    stream_reset(kDriscDoorbellStream);
    stream_reset(kDriscScratchStream);
    if (!barrier()) {
        return;
    }

    // Phase: ping-pong. Answer each ERISC doorbell with one of ours; then the same with a payload each way.
    for (uint32_t r = 0; r < 2 * n_rt; r++) {
        if (!spin_until([&] { return stream_value(kDriscDoorbellStream) > r; })) {
            res.fail(phase);
            return;
        }
        if (r >= n_rt) {
            noc_async_write_one_packet(base + kBufferOffset, landing, payload, noc);
        }
        stream_add_remote(eth_doorbell, 1, noc);
    }
    noc_async_write_barrier(noc);
    if (!barrier()) {
        return;
    }
    stream_reset(kDriscDoorbellStream);
    if (!barrier()) {
        return;
    }

#elif DRISC_MB_PART == 2
    // Phase: pipeline, with stateless issue and then with lean stateful issue. The ERISC pushes packets into a ring
    // of `depth` staging slots and rings our doorbell; we forward each one back into its landing ring (payload +
    // doorbell) and return the staging slot once the write has left our L1. Credits go back in batches, as many as
    // have become free. With lean issue, payloads, landing doorbells and credits each keep their own command buffer.
    const uint64_t eth_credit = stream_update_addr(eth_x, eth_y, kEthCreditStream, noc);
    const uint32_t landing_l1 = eth_base + kEthLandingOffset;
    auto pipeline = [&](auto lean, uint32_t results) {
        constexpr bool kLean = decltype(lean)::value;
        for (uint32_t d = 0; d < num_depths; d++) {
            const uint32_t depth = get_arg_val<uint32_t>(ARG_DEPTHS + d);
            if constexpr (kLean) {
                // The phase barrier's doorbell reprograms write_at_cmd_buf, so set the state for every depth.
                lean_write_set_state(landing, payload, noc);
                lean_doorbell_set_state(eth_doorbell, noc);
                lean_doorbell_set_state(eth_credit, noc, write_reg_cmd_buf);
            }
            uint32_t forwarded = 0, credited = 0, work = 0, wait = 0, idle = 0;
            uint32_t fwd_slot = 0, credit_slot = 0, landing_slot = 0;
            uint32_t polls = 0;
            while (credited < pipe_pkts) {
                if (++polls == kSpinLimit) {
                    res.fail(phase);
                    return false;
                }
                const uint32_t arrived = stream_value(kDriscDoorbellStream);
                if (arrived == forwarded && credited == forwarded) {
                    idle++;
                    continue;
                }
                if (arrived != forwarded) {
                    polls = 0;
                }
                const uint32_t t0 = now();
                const uint32_t progress = forwarded + credited;
                while (forwarded < arrived) {
                    const uint32_t staged = base + kBufferOffset + fwd_slot * kSlotBytes;
                    if constexpr (kLean) {
                        lean_trid_write(staged, landing_l1 + landing_slot * kSlotBytes, slot_trid(fwd_slot), noc);
                        lean_doorbell(noc);
                    } else {
                        noc_async_write_one_packet_with_trid(
                            staged,
                            landing + landing_slot * kSlotBytes,
                            payload,
                            slot_trid(fwd_slot),
                            write_cmd_buf,
                            noc);
                        stream_add_remote(eth_doorbell, 1, noc);
                    }
                    forwarded++;
                    fwd_slot = next_slot(fwd_slot, depth);
                    landing_slot = next_slot(landing_slot, kEthLandingSlots);
                }
                uint32_t freed = 0;
                while (credited + freed < forwarded &&
                       ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, slot_trid(credit_slot))) {
                    freed++;
                    credit_slot = next_slot(credit_slot, depth);
                }
                if (freed) {
                    if constexpr (kLean) {
                        lean_doorbell_add(freed, noc, write_reg_cmd_buf);
                    } else {
                        stream_add_remote(eth_credit, freed, noc);
                    }
                    credited += freed;
                    polls = 0;
                }
                const uint32_t dt = now() - t0;
                if (forwarded + credited != progress) {
                    work += dt;
                } else {
                    wait += dt;
                }
            }
            noc_async_write_barrier(noc);
            res.put(results + d * DP_COUNT + DP_BUSY, work + wait, pipe_pkts);
            res.put(results + d * DP_COUNT + DP_WORK, work, pipe_pkts);
            res.put(results + d * DP_COUNT + DP_IDLE_POLLS, idle, pipe_pkts);
            if (!barrier()) {  // the ERISC has seen every copy and credit
                return false;
            }
            stream_reset(kDriscDoorbellStream);
            if (!barrier()) {
                return false;
            }
        }
        return true;
    };
    if (!pipeline(std::false_type{}, D_PIPE_BASE) || !pipeline(std::true_type{}, D_PIPE_BASE + kMaxDepths * DP_COUNT)) {
        return;
    }
#elif DRISC_MB_PART == 3
    // Phase: the DRISC's command buffers and NoC ports (the ERISC waits; its L1 is the source and destination).
    measure_cmd_bufs(res, base, eth_x, eth_y, eth_base, payload, n, noc1_ok, num_slots);
    const uint32_t fwd_eriscs = get_arg_val<uint32_t>(ARG_FWD_ERISCS);
    if (fwd_eriscs > 0) {
        measure_forwarding(
            res,
            base,
            eth_x,
            eth_y,
            eth_base,
            payload,
            n,
            noc1_ok,
            num_slots,
            fwd_eriscs,
            get_arg_val<uint32_t>(ARG_FWD0_X),
            get_arg_val<uint32_t>(ARG_FWD0_Y),
            get_arg_val<uint32_t>(ARG_FWD1_X),
            get_arg_val<uint32_t>(ARG_FWD1_Y));
    }
    if (!barrier()) {
        return;
    }

#else  // DRISC_MB_PART == 4
    // Phase: two DRISCs on one Ethernet tile; the peer is the other DRISC. Role 0 pulls 4 KiB packets from the
    // ERISC's source buffer, role 1 writes 4 KiB packets into its landing ring: each alone, then both at once.
    const uint32_t role = get_arg_val<uint32_t>(ARG_ROLE);
    const uint32_t t_x = get_arg_val<uint32_t>(ARG_TARGET_X);
    const uint32_t t_y = get_arg_val<uint32_t>(ARG_TARGET_Y);
    const uint32_t t_base = get_arg_val<uint32_t>(ARG_TARGET_BASE);
    const uint32_t ops = kTwoDriscOpsScale * n;
    const uint32_t mask = get_arg_val<uint32_t>(ARG_TWO_MASK);
    for (uint32_t v = 0; v < kTwoDriscVariants; v++) {
        if (!barrier()) {
            return;
        }
        const TwoDriscVariant tv = kTwoDriscList[v];
        const int8_t my_noc = role == 0 ? tv.pull_noc : tv.write_noc;
        if (!(mask & (1u << v)) || my_noc < 0 || (my_noc == 1 && !noc1_ok) ||
            (role == 0 && tv.own_write_noc == 1 && !noc1_ok)) {
            continue;
        }
        const uint8_t tn = my_noc;
        uint32_t slot = 0;
        const uint32_t start = now();
        if (role == 0 && tv.own_write_noc >= 0) {
            // One DRISC both ways: pulls from the source buffer interleaved with writes into the landing ring (part 3's
            // duplex, here with the other DRISC idle). Our staging slot 0 is the write source; pulls land in 1+.
            const uint8_t wn = tv.own_write_noc;
            const uint64_t t_src = get_noc_addr(t_x, t_y, t_base + kEthSrcOffset, tn);
            const uint32_t t_landing = t_base + kEthLandingOffset;
            lean_write_set_state(get_noc_addr(t_x, t_y, t_landing, wn), payload, wn);
            uint32_t rslot = 1;
            for (uint32_t i = 0; i < ops; i++) {
                noc_async_read_one_packet(t_src, base + kBufferOffset + rslot * kSlotBytes, payload, tn);
                lean_trid_write(base + kBufferOffset, t_landing + slot * kSlotBytes, slot_trid(slot), wn);
                rslot = rslot + 1 == num_slots ? 1 : rslot + 1;
                slot = next_slot(slot, kEthLandingSlots);
            }
            noc_async_read_barrier(tn);
            noc_async_write_barrier(wn);
        } else if (role == 0) {
            const uint64_t t_src = get_noc_addr(t_x, t_y, t_base + kEthSrcOffset, tn);
            for (uint32_t i = 0; i < ops; i++) {
                noc_async_read_one_packet(t_src, base + kBufferOffset + slot * kSlotBytes, payload, tn);
                slot = next_slot(slot, num_slots);
            }
            noc_async_read_barrier(tn);
        } else {
            const uint32_t t_landing = t_base + kEthLandingOffset;
            lean_write_set_state(get_noc_addr(t_x, t_y, t_landing, tn), payload, tn);
            for (uint32_t i = 0; i < ops; i++) {
                lean_trid_write(base + kBufferOffset, t_landing + slot * kSlotBytes, slot_trid(slot), tn);
                slot = next_slot(slot, kEthLandingSlots);
            }
            noc_async_write_barrier(tn);
        }
        res.put(D_NOC_BASE + v, now() - start, ops);
    }
    if (!barrier()) {
        return;
    }
    if (role == 0) {
        // Release the ERISC's polling kernel, if the host started one.
        stream_add_remote(stream_update_addr(t_x, t_y, kEthScratchStream, noc), 1, noc);
        noc_async_write_barrier(noc);
    }
#endif
}
