// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Transaction-ID sent latency on the router's receiving core (see test_erisc_load_overlap.cpp, TridSentLatency, and
// trid_sent_latency_common.hpp). Runs on ERISC1 with NOC1, as the router's receiver does. A packet's writes are issued
// as its local writes are: command buffer 0 preset once with the write's NOC_CTRL and this core as the source, then per
// write the transaction ID (ncrisc_noc_set_transaction_id, which waits for the buffer), source, length, destination and
// go (noc_write_with_state). The earlier write of the BEHIND context goes on buffer 2, preset the same way. The sent
// proof is the router's: the ID's outgoing-write count read 0 (ncrisc_noc_nonposted_write_with_transaction_id_sent).
// Every repetition starts with every earlier write acknowledged. The buffers' state is saved at the start and restored
// at the end; every wait is bounded, so a case that never completes ends the run with an error instead of hanging.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/trid_sent_latency_common.hpp"

using namespace trid_sent_latency;

namespace {

constexpr uint32_t kNoc = noc_index;
constexpr uint32_t kSpinLimit = 1u << 24;  // about 0.1 s
constexpr uint32_t kProbeBuf = 0;
constexpr uint32_t kOtherBuf = 2;
uint32_t self_coord;

FORCE_INLINE uint32_t cycles() {
    uint32_t c;
    asm volatile("csrr %0, mcycle" : "=r"(c));
    return c;
}

FORCE_INLINE bool nonposted_acked() {
    for (uint32_t i = 0; i < kSpinLimit; i++) {
        if (ncrisc_noc_nonposted_writes_flushed(kNoc)) {
            return true;
        }
    }
    return false;
}

template <uint32_t BUF>
void preset_write() {
    while (!noc_cmd_buf_ready(kNoc, BUF));
    noc_write_init_state<BUF, CQ_NOC_mkp>(kNoc, kVc);
    noc_cmd_buf_set_targ_addr_coordinate(kNoc, BUF, self_coord);
    noc_cmd_buf_clear_targ_addr_mid(kNoc, BUF);
    noc_cmd_buf_clear_ret_addr_mid(kNoc, BUF);
}

// One write as the router's local write.
template <uint32_t BUF>
FORCE_INLINE void local_write(uint32_t src, uint64_t dst, uint32_t size, uint32_t trid) {
    ncrisc_noc_set_transaction_id(kNoc, BUF, trid);
    noc_write_with_state<DM_DEDICATED_NOC, BUF, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_wait, true, false>(
        kNoc, src, dst, size);
}

// The packet's writes: `n` chunks of `size / n` bytes.
FORCE_INLINE void packet_writes(uint32_t src, uint64_t dst, uint32_t size, uint32_t n) {
    const uint32_t chunk = size / n;
    for (uint32_t k = 0; k < n; k++) {
        local_write<kProbeBuf>(src + k * chunk, dst + k * chunk, chunk, kProbeTrid);
    }
}

// Polls until every write of the probe's ID has left; returns the polls (0: timed out).
FORCE_INLINE uint32_t poll_sent() {
    for (uint32_t i = 1; i < kSpinLimit; i++) {
        if (ncrisc_noc_nonposted_write_with_transaction_id_sent(kNoc, kProbeTrid)) {
            return i;
        }
    }
    return 0;
}

bool run_case(volatile uint32_t* out, uint32_t kind, uint32_t context, uint32_t src, uint64_t dst, uint32_t other_src,
              uint64_t other_dst) {
    uint32_t issue_sum = 0, issue_min = ~0u, total_min = ~0u, polls_sum = 0;
    for (uint32_t rep = 0; rep < kReps; rep++) {
        if (!nonposted_acked()) {
            return false;
        }
        if (context == BEHIND) {
            local_write<kOtherBuf>(other_src, other_dst, kMaxBytes, kOtherTrid);
        }
        const uint32_t t0 = cycles();
        packet_writes(src, dst, kKindBytes[kind], kKindWrites[kind]);
        const uint32_t t1 = cycles();
        const uint32_t polls = poll_sent();
        const uint32_t t2 = cycles();
        if (polls == 0) {
            return false;
        }
        issue_sum += t1 - t0;
        issue_min = t1 - t0 < issue_min ? t1 - t0 : issue_min;
        total_min = t2 - t0 < total_min ? t2 - t0 : total_min;
        polls_sum += polls;
        out[kNumFields + rep] = t2 - t1;
    }
    out[ISSUE_SUM] = issue_sum;
    out[ISSUE_MIN] = issue_min;
    out[TOTAL_MIN] = total_min;
    out[POLLS] = polls_sum;
    return true;
}

}  // namespace

void kernel_main() {
    const uint32_t results = get_arg_val<uint32_t>(0);
    const uint32_t src_base = get_arg_val<uint32_t>(1);
    const uint32_t dst_x = get_arg_val<uint32_t>(2);
    const uint32_t dst_y = get_arg_val<uint32_t>(3);
    const uint32_t other_x = get_arg_val<uint32_t>(4);
    const uint32_t other_y = get_arg_val<uint32_t>(5);
    const uint32_t dst_l1 = get_arg_val<uint32_t>(6);
    volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(results);
    const uint64_t dst = get_noc_addr(dst_x, dst_y, dst_l1, kNoc);
    const uint64_t other_dst = get_noc_addr(other_x, other_y, dst_l1, kNoc);
    const uint32_t other_src = src_base + kMaxBytes;

    NocCmdBufState saved[4];
    for (uint32_t b = 0; b < 4; b++) {
        noc_cmd_buf_save_state(kNoc, b, &saved[b]);
    }
    ncrisc_noc_counters_init();
    self_coord =
        static_cast<uint32_t>(NOC_XY_ADDR(my_x[kNoc], my_y[kNoc], 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
    preset_write<kProbeBuf>();
    preset_write<kOtherBuf>();

    // One poll with nothing outstanding: the poll's own cost.
    uint32_t poll_cost = ~0u;
    nonposted_acked();
    for (uint32_t rep = 0; rep < kReps; rep++) {
        const uint32_t t0 = cycles();
        const bool sent = ncrisc_noc_nonposted_write_with_transaction_id_sent(kNoc, kProbeTrid);
        const uint32_t t1 = cycles();
        poll_cost = sent && t1 - t0 < poll_cost ? t1 - t0 : poll_cost;
    }
    out[kPollCost] = poll_cost;

    uint32_t failed = 0;
    for (uint32_t context = 0; context < kNumContexts && !failed; context++) {
        for (uint32_t kind = 0; kind < kNumKinds && !failed; kind++) {
            if (!run_case(out + case_index(kind, context), kind, context, src_base, dst, other_src, other_dst)) {
                failed = context * kNumKinds + kind + 1;
            }
        }
    }
    out[kFailed] = failed;

    nonposted_acked();
    for (uint32_t b = 0; b < 4; b++) {
        noc_cmd_buf_restore_state(kNoc, b, &saved[b]);
    }
    out[kDone] = kDoneValue;
}
