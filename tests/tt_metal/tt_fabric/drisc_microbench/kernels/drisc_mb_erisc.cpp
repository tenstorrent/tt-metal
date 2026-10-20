// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ERISC side of the DRISC microbenchmark (see test_drisc_microbench.cpp). Runs on ERISC1 of an active Ethernet core,
// the receiving router's core, and talks to the DRISC on this kernel's NoC.

#include <cstdint>
#include <type_traits>

#include "drisc_mb_device.hpp"

using namespace drisc_mb;

void kernel_main() {
#if DRISC_MB_PART == 4
    // Part 4: two DRISCs read from and write into our L1. Optionally (DRISC_MB_TWO_SPIN) we run meanwhile, polling a
    // stream register as a router's ERISC polls, until DRISC 1 rings it at the end.
    stream_reset(kEthScratchStream);
    spin_until([] { return stream_value(kEthScratchStream) != 0; });
#else
    const uint32_t base = get_arg_val<uint32_t>(ARG_MY_BASE);
    const uint32_t drisc_x = get_arg_val<uint32_t>(ARG_PEER_X);
    const uint32_t drisc_y = get_arg_val<uint32_t>(ARG_PEER_Y);
    const uint32_t drisc_base = get_arg_val<uint32_t>(ARG_PEER_BASE);
    const uint32_t payload = get_arg_val<uint32_t>(ARG_PAYLOAD);
    [[maybe_unused]] const uint32_t n = get_arg_val<uint32_t>(ARG_N);
    [[maybe_unused]] const uint32_t n_rt = get_arg_val<uint32_t>(ARG_N_RT);
    [[maybe_unused]] const bool drisc_noc1 = get_arg_val<uint32_t>(ARG_DRISC_NOC1) != 0;
    [[maybe_unused]] const uint32_t num_slots = get_arg_val<uint32_t>(ARG_DRISC_SLOTS);
    [[maybe_unused]] const uint32_t pipe_pkts = get_arg_val<uint32_t>(ARG_PIPE_PKTS);
    [[maybe_unused]] const uint32_t num_depths = get_arg_val<uint32_t>(ARG_NUM_DEPTHS);

    Results res{l1<uint32_t>(base + kResultsOffset), E_ERROR};
    const uint8_t noc = noc_index;
    [[maybe_unused]] const uint32_t src = base + kEthSrcOffset;
    [[maybe_unused]] const uint64_t staging = get_noc_addr(drisc_x, drisc_y, drisc_base + kBufferOffset, noc);
    [[maybe_unused]] const uint64_t drisc_doorbell = stream_update_addr(drisc_x, drisc_y, kDriscDoorbellStream, noc);
    [[maybe_unused]] const uint64_t drisc_bare_doorbell =
        stream_update_addr(drisc_x, drisc_y, kDriscScratchStream, noc);
    [[maybe_unused]] const uint32_t staging_l1 = drisc_base + kBufferOffset;
    [[maybe_unused]] uint32_t slot = 0;

    const uint32_t my_streams[4] = {kEthSyncStream, kEthDoorbellStream, kEthCreditStream, kEthScratchStream};
    for (uint32_t s : my_streams) {
        stream_reset(s);
    }
    if (!handshake(base, get_noc_addr(drisc_x, drisc_y, drisc_base + kReadyOffset, noc), noc)) {
        res.fail(0);
        return;
    }
    Peer peer{stream_update_addr(drisc_x, drisc_y, kDriscSyncStream, noc), kEthSyncStream, noc, 0};
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

#if DRISC_MB_PART == 1
    // Phase: the DRISC measures its local costs, then writes into our L1 and streams.
    if (!barrier() || !barrier()) {
        return;
    }
    // Per NoC the DRISC rang our doorbell for each copy (stateless back to back; stateful and lean, each back to back
    // and alone) and each bare doorbell (back to back, alone).
    const uint32_t nocs = drisc_noc1 ? 2 : 1;
    res.put(E_CHECK_FWD_DOORBELLS, stream_value(kEthDoorbellStream), 5 * nocs * n);
    res.put(E_CHECK_BARE_DOORBELLS, stream_value(kEthScratchStream), 2 * nocs * n);
    stream_reset(kEthDoorbellStream);
    stream_reset(kEthScratchStream);

    // Phase: our local costs, then pushes into the DRISC (it waits).
    const uint32_t streams[4] = {kEthScratchStream, kEthCreditStream, kEthDoorbellStream, kEthSyncStream};
    measure_local(
        res,
        LocalEntries{E_TIMER_READ, E_L1_LOAD_DEP, E_L1_LOAD_INDEP4, E_STREAM_READ, E_STREAM_READ_INDEP4, E_SINK},
        base + kScratchOffset,
        streams,
        n);

    slot = 0;
    uint32_t start = now();
    for (uint32_t i = 0; i < n; i++) {
        noc_async_write_one_packet(src, staging + slot * kSlotBytes, payload, noc);
        slot = next_slot(slot, num_slots);
    }
    uint32_t issued = now();
    noc_async_write_barrier(noc);
    res.put(E_WR_ISSUE, issued - start, n);
    res.put(E_WR_DRAIN, now() - start, n);

    start = now();
    for (uint32_t i = 0; i < n; i++) {
        stream_add_remote(drisc_bare_doorbell, 1, noc);
    }
    issued = now();
    noc_async_write_barrier(noc);
    res.put(E_DOORBELL_ISSUE, issued - start, n);
    res.put(E_DOORBELL_DRAIN, now() - start, n);

    slot = 0;
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        noc_async_write_one_packet(src, staging + slot * kSlotBytes, payload, noc);
        stream_add_remote(drisc_doorbell, 1, noc);
        slot = next_slot(slot, num_slots);
    }
    issued = now();
    noc_async_write_barrier(noc);
    res.put(E_PUSH_ISSUE, issued - start, n);
    res.put(E_PUSH_DRAIN, now() - start, n);

    // Pushes as a receiving ERISC with its own DRISC would issue them: the staging destination is set once, so each
    // push programs only addresses, size and transaction id, and nothing waits for acks. Doorbells use another
    // command buffer, which keeps the state.
    noc_async_write_one_packet_with_trid_set_state(staging, write_cmd_buf, noc);
    slot = 0;
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        noc_async_write_one_packet_with_trid_with_state(
            src, staging_l1 + slot * kSlotBytes, payload, slot_trid(slot), write_cmd_buf, noc);
        stream_add_remote(drisc_doorbell, 1, noc);
        slot = next_slot(slot, num_slots);
    }
    issued = now();
    noc_async_write_barrier(noc);
    res.put(E_SF_PUSH_ISSUE, issued - start, n);
    res.put(E_SF_PUSH_DRAIN, now() - start, n);

    // One at a time with the NoC idle: what each costs the CPU when the port is free.
    auto staging_slot = [=](uint32_t i) {
        const uint32_t s = i % num_slots;
        const uint64_t dst = staging + s * kSlotBytes;
        return OpArgs{src, static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32), slot_trid(s)};
    };
    res.put(
        E_SF_PUSH_ALONE,
        time_alone(
            n,
            noc,
            staging_slot,
            [=](const OpArgs& a) {
                noc_async_write_one_packet_with_trid_with_state(a.src, a.dst, payload, a.trid, write_cmd_buf, noc);
                stream_add_remote(drisc_doorbell, 1, noc);
            }),
        n);
    res.put(
        E_SF_WR_ALONE,
        time_alone(
            n,
            noc,
            staging_slot,
            [=](const OpArgs& a) {
                noc_async_write_one_packet_with_trid_with_state(a.src, a.dst, payload, a.trid, write_cmd_buf, noc);
            }),
        n);
    res.put(
        E_DOORBELL_ALONE,
        time_alone(n, noc, no_args, [=](const OpArgs&) { stream_add_remote(drisc_bare_doorbell, 1, noc); }),
        n);
    res.put(
        E_WR_ALONE,
        time_alone(
            n,
            noc,
            staging_slot,
            [=](const OpArgs& a) { noc_async_write_one_packet(a.src, a.noc_dst(), payload, noc); }),
        n);
    res.put(
        E_PUSH_ALONE,
        time_alone(
            n,
            noc,
            staging_slot,
            [=](const OpArgs& a) {
                noc_async_write_one_packet(a.src, a.noc_dst(), payload, noc);
                stream_add_remote(drisc_doorbell, 1, noc);
            }),
        n);

    // The same with lean stateful issue: one command-buffer poll per write, and a stateful doorbell.
    lean_write_set_state(staging, payload, noc);
    lean_doorbell_set_state(drisc_bare_doorbell, noc);
    res.put(E_LEAN_DOORBELL_ALONE, time_alone(n, noc, no_args, [=](const OpArgs&) { lean_doorbell(noc); }), n);
    lean_doorbell_set_state(drisc_doorbell, noc);
    res.put(
        E_LEAN_WR_ALONE,
        time_alone(n, noc, staging_slot, [=](const OpArgs& a) { lean_trid_write(a.src, a.dst, a.trid, noc); }),
        n);
    slot = 0;
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        lean_trid_write(src, staging_l1 + slot * kSlotBytes, slot_trid(slot), noc);
        lean_doorbell(noc);
        slot = next_slot(slot, num_slots);
    }
    issued = now();
    noc_async_write_barrier(noc);
    res.put(E_LEAN_PUSH_ISSUE, issued - start, n);
    res.put(E_LEAN_PUSH_DRAIN, now() - start, n);
    res.put(
        E_LEAN_PUSH_ALONE,
        time_alone(
            n,
            noc,
            staging_slot,
            [=](const OpArgs& a) {
                lean_trid_write(a.src, a.dst, a.trid, noc);
                lean_doorbell(noc);
            }),
        n);
    noc_clear_packet_tags(noc);

    start = now();
    for (uint32_t i = 0; i < n_rt; i++) {
        noc_async_write_one_packet(src, staging, payload, noc);
        noc_async_write_barrier(noc);
    }
    res.put(E_WR_RTT, now() - start, n_rt);

    // The DRISC checks and resets its streams between these two barriers.
    if (!barrier() || !barrier()) {
        return;
    }

    // Phase: ping-pong, doorbells only, then a payload each way.
    for (uint32_t pass = 0; pass < 2; pass++) {
        uint32_t total = 0;
        for (uint32_t r = 0; r < n_rt; r++) {
            const uint32_t expect = pass * n_rt + r + 1;
            const uint32_t t0 = now();
            if (pass == 1) {
                noc_async_write_one_packet(src, staging, payload, noc);
            }
            stream_add_remote(drisc_doorbell, 1, noc);
            if (!spin_until([&] { return stream_value(kEthDoorbellStream) >= expect; })) {
                res.fail(phase);
                return;
            }
            total += now() - t0;
        }
        res.put(pass == 0 ? E_PING_RTT : E_PING_DATA_RTT, total, n_rt);
    }
    noc_async_write_barrier(noc);
    if (!barrier()) {
        return;
    }
    stream_reset(kEthDoorbellStream);
    if (!barrier()) {
        return;
    }

#elif DRISC_MB_PART == 2
    // Phase: pipeline, with stateless issue and then with lean stateful issue. Push `pipe_pkts` packets through a
    // ring of `depth` DRISC staging slots, waiting for a freed slot (credit) when the ring is full. The DRISC
    // forwards each into our landing ring and rings our doorbell.
    auto pipeline = [&](auto lean, uint32_t results) {
        constexpr bool kLean = decltype(lean)::value;
        for (uint32_t d = 0; d < num_depths; d++) {
            const uint32_t depth = get_arg_val<uint32_t>(ARG_DEPTHS + d);
            if constexpr (kLean) {
                // The phase barrier's doorbell reprograms write_at_cmd_buf, so set the state for every depth.
                lean_write_set_state(staging, payload, noc);
                lean_doorbell_set_state(drisc_doorbell, noc);
            }
            uint32_t pushed = 0, busy = 0, waits = 0, polls = 0;
            slot = 0;
            const uint32_t t_start = now();
            while (pushed < pipe_pkts) {
                const uint32_t returned = stream_value(kEthCreditStream);
                if (pushed - returned >= depth) {
                    waits++;
                    if (++polls == kSpinLimit) {
                        res.fail(phase);
                        return false;
                    }
                    continue;
                }
                polls = 0;
                const uint32_t t0 = now();
                if constexpr (kLean) {
                    lean_trid_write(src, staging_l1 + slot * kSlotBytes, slot_trid(slot), noc);
                    lean_doorbell(noc);
                } else {
                    noc_async_write_one_packet(src, staging + slot * kSlotBytes, payload, noc);
                    stream_add_remote(drisc_doorbell, 1, noc);
                }
                pushed++;
                slot = next_slot(slot, depth);
                busy += now() - t0;
            }
            if (!spin_until([&] {
                    return stream_value(kEthCreditStream) >= pipe_pkts && stream_value(kEthDoorbellStream) >= pipe_pkts;
                })) {
                res.fail(phase);
                return false;
            }
            const uint32_t t_end = now();
            noc_async_write_barrier(noc);
            res.put(results + d * EP_COUNT + EP_TOTAL, t_end - t_start, pipe_pkts);
            res.put(results + d * EP_COUNT + EP_BUSY, busy, pipe_pkts);
            res.put(results + d * EP_COUNT + EP_CREDIT_WAITS, waits, pipe_pkts);
            stream_reset(kEthCreditStream);
            stream_reset(kEthDoorbellStream);
            if (!barrier() || !barrier()) {
                return false;
            }
        }
        return true;
    };
    if (!pipeline(std::false_type{}, E_PIPE_BASE) || !pipeline(std::true_type{}, E_PIPE_BASE + kMaxDepths * EP_COUNT)) {
        return;
    }
    noc_clear_packet_tags(noc);
#else  // DRISC_MB_PART == 3: the DRISC reads our source buffer and writes our landing ring
    if (!barrier()) {
        return;
    }
#endif
#endif  // DRISC_MB_PART == 4
}
