// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Ethernet TX queue probe kernel; one copy runs on ERISC0 of each end of a link (see test_eth_txq_probe.cpp).
//
// Sender: per packet, a payload write into the peer's landing slot, then a pointer update on the same queue, as the
// fabric router's sender does. On TXQ0 the update is a stream register write (the router's form); on TXQ1 it is a
// 16 B counter write (the form the router's credits use on TXQ1). With two data queues, even packets go on TXQ0 and
// odd ones on TXQ1.
// Receiver: takes packets in order, looking at the queue packet k was sent on (the merge), checks the landing slot,
// and optionally sends a 16 B count back every REVERSE_EVERY packets on REVERSE_QUEUE.
// Every wait gives up after ~1 s and records where, so a stuck queue fails the test instead of hanging it.
// RT_SEQ_UPDATE_TIMEOUT sets the receiver's TXQ0 keepalive period (ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT) for the run: with
// nothing else to send on TXQ0, that keepalive is what carries the acknowledgements for the sender's TXQ0 packets.
// The firmware value is restored at the end.

#include <cstdint>

#include "risc_common.h"
#include "internal/ethernet/tunneling.h"
#include "internal/ethernet/tt_eth_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h"
#include "tests/tt_metal/tt_fabric/eth_txq_probe/kernels/eth_txq_probe_common.hpp"

using namespace eth_txq_probe;

constexpr uint32_t NUM_DATA_QUEUES = get_compile_time_arg_val(CT_NUM_DATA_QUEUES);
constexpr uint32_t REVERSE_QUEUE = get_compile_time_arg_val(CT_REVERSE_QUEUE);
constexpr uint32_t REVERSE_EVERY = get_compile_time_arg_val(CT_REVERSE_EVERY);
constexpr bool GATED = get_compile_time_arg_val(CT_GATED) != 0;
constexpr bool USES_TXQ1 = NUM_DATA_QUEUES == 2 || (REVERSE_EVERY > 0 && REVERSE_QUEUE == 1);
constexpr uint32_t PTR_STREAM = 0;  // receiver's stream register counting packets sent on TXQ0

static_assert(NUM_DATA_QUEUES == 1 || NUM_DATA_QUEUES == 2);
static_assert(REVERSE_EVERY == 0 || REVERSE_QUEUE < kNumQueues);
static_assert(!GATED || REVERSE_EVERY > 0);

namespace {

constexpr uint32_t kSpinLimit = 100'000'000;
constexpr uint32_t kAcceptAhead =
    32;  // what the router sets on its TXQ1 (default_num_eth_txq_data_packet_accept_ahead)

constexpr uint32_t kTxqCounterOffsets[kTxqCounters] = {
    ETH_TXQ_TRANSFER_CNT, ETH_TXQ_PKT_START_CNT, ETH_TXQ_PKT_END_CNT, ETH_TXQ_WORD_CNT, ETH_TXQ_RESEND_CNT};
constexpr uint32_t kTxqConfigOffsets[kTxqConfig] = {
    ETH_TXQ_CTRL,
    ETH_TXQ_MAX_PKT_SIZE_BYTES,
    ETH_TXQ_REMOTE_SEQ_TIMEOUT,
    ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT,
    ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD};

volatile tt_l1_ptr uint32_t* results;

FORCE_INLINE uint32_t cycles() {
    uint32_t c;
    asm volatile("csrr %0, mcycle" : "=r"(c));
    return c;
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* l1_word(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

FORCE_INLINE void fail(uint32_t code, uint32_t index) {
    results[R_ERR_INDEX] = index;
    results[R_STATUS] = code;
}

template <typename Ready>
FORCE_INLINE bool wait_until(Ready ready, uint32_t code, uint32_t index) {
    for (uint32_t i = 0; !ready();) {
        if (++i == kSpinLimit) {
            fail(code, index);
            return false;
        }
    }
    return true;
}

FORCE_INLINE bool wait_txq_idle(uint32_t q, uint32_t code, uint32_t index) {
    return wait_until([q] { return !internal_::eth_txq_is_busy(q); }, code, index);
}

void sample_txq_counters(uint32_t at) {
    for (uint32_t q = 0; q < kNumQueues; q++) {
        for (uint32_t c = 0; c < kTxqCounters; c++) {
            results[at + q * kTxqCounters + c] = eth_txq_reg_read(q, kTxqCounterOffsets[c]);
        }
    }
}

void sample_rxq_drops(uint32_t at) {
    for (uint32_t q = 0; q < kNumQueues; q++) {
        results[at + q] = ETH_READ_REG(kRxq0RegsStart + q * kRxqRegsSize + kRxqPacketDropCnt);
    }
}

void sample_txq_config() {
    for (uint32_t q = 0; q < kNumQueues; q++) {
        for (uint32_t c = 0; c < kTxqConfig; c++) {
            results[R_TXQ_CONFIG + q * kTxqConfig + c] = eth_txq_reg_read(q, kTxqConfigOffsets[c]);
        }
    }
}

bool run_sender(uint32_t base, uint32_t payload, uint32_t n, uint32_t warm, uint32_t slots) {
    const uint32_t slot_mask = slots - 1;
    volatile tt_l1_ptr uint32_t* reverse_count = l1_word(base + kReverseLandOffset);
    uint32_t q1_count = 0;
    results[R_T_FIRST] = cycles();
    for (uint32_t k = 0; k < n; k++) {
        if (k == warm) {
            results[R_T_WARM] = cycles();
        }
        if constexpr (GATED) {
            auto has_slot = [&] {
                invalidate_l1_cache();
                return k - *reverse_count < slots;
            };
            if (!wait_until(has_slot, kErrSenderCredits, k)) {
                return false;
            }
        }
        const uint32_t slot = base + kSlotsOffset + (k & slot_mask) * kSlotBytes;
        *l1_word(slot) = k;
        if (NUM_DATA_QUEUES == 2 && (k & 1)) {
            if (!wait_txq_idle(1, kErrSenderTxq, k)) {
                return false;
            }
            internal_::eth_send_packet_bytes_unsafe(1, slot, slot, payload);
            const uint32_t src = base + kSmallRingOffset + (q1_count & (kSmallRingEntries - 1)) * 16;
            *l1_word(src) = ++q1_count;
            if (!wait_txq_idle(1, kErrSenderTxq, k)) {
                return false;
            }
            internal_::eth_send_packet_bytes_unsafe(1, src, base + kQ1CounterLandOffset, 16);
        } else {
            if (!wait_txq_idle(0, kErrSenderTxq, k)) {
                return false;
            }
            internal_::eth_send_packet_bytes_unsafe(0, slot, slot, payload);
            if (!wait_txq_idle(0, kErrSenderTxq, k)) {
                return false;
            }
            remote_update_ptr_val<PTR_STREAM, 0>(1);
        }
    }
    results[R_T_LAST] = cycles();

    auto receiver_done = [&] {
        invalidate_l1_cache();
        return *l1_word(base + kDoneLandOffset) != 0;
    };
    return wait_until(receiver_done, kErrSenderDone, n);
}

bool run_receiver(uint32_t base, uint32_t n, uint32_t warm, uint32_t slots) {
    const uint32_t slot_mask = slots - 1;
    volatile tt_l1_ptr uint32_t* q1_counter = l1_word(base + kQ1CounterLandOffset);
    uint32_t q1_seen = 0;
    uint32_t reverse_countdown = REVERSE_EVERY;
    uint32_t reverse_sent = 0;
    uint32_t merge_errors = 0;
    for (uint32_t k = 0; k < n; k++) {
        if (NUM_DATA_QUEUES == 2 && (k & 1)) {
            auto arrived = [&] {
                invalidate_l1_cache();
                return *q1_counter > q1_seen;
            };
            if (!wait_until(arrived, kErrReceiverQ1, k)) {
                return false;
            }
            q1_seen++;
        } else {
            if (!wait_until([] { return get_ptr_val<PTR_STREAM>() != 0; }, kErrReceiverQ0, k)) {
                return false;
            }
            increment_local_update_ptr_val<PTR_STREAM>(-1);
        }
        if (k == 0) {
            results[R_T_FIRST] = cycles();
        }
        if (k == warm) {
            results[R_T_WARM] = cycles();
        }
        // The sender may already have refilled this slot with a later packet when nothing gates it.
        invalidate_l1_cache();
        const uint32_t v = *l1_word(base + kSlotsOffset + (k & slot_mask) * kSlotBytes);
        merge_errors += (v < k || ((v - k) & slot_mask) != 0) ? 1 : 0;
        if constexpr (REVERSE_EVERY > 0) {
            if (--reverse_countdown == 0) {
                reverse_countdown = REVERSE_EVERY;
                const uint32_t src = base + kSmallRingOffset + (reverse_sent++ & (kSmallRingEntries - 1)) * 16;
                *l1_word(src) = k + 1;
                if (!wait_txq_idle(REVERSE_QUEUE, kErrReceiverTxq, k)) {
                    return false;
                }
                internal_::eth_send_packet_bytes_unsafe(REVERSE_QUEUE, src, base + kReverseLandOffset, 16);
            }
        }
    }
    results[R_T_LAST] = cycles();
    results[R_MERGE_ERRORS] = merge_errors;

    // Tell the sender it may finish; this goes out after the measurement.
    *l1_word(base + kDoneSrcOffset) = 1;
    if (!wait_txq_idle(0, kErrReceiverTxq, n)) {
        return false;
    }
    internal_::eth_send_packet_bytes_unsafe(0, base + kDoneSrcOffset, base + kDoneLandOffset, 16);
    return true;
}

}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(RT_BASE);
    const bool is_sender = get_arg_val<uint32_t>(RT_IS_SENDER) != 0;
    const uint32_t payload = get_arg_val<uint32_t>(RT_PAYLOAD);
    const uint32_t n = get_arg_val<uint32_t>(RT_NUM_PACKETS);
    const uint32_t warm = get_arg_val<uint32_t>(RT_WARMUP_PACKETS);
    const uint32_t slots = get_arg_val<uint32_t>(RT_NUM_SLOTS);
    const uint32_t seq_update_timeout = is_sender ? 0 : get_arg_val<uint32_t>(RT_SEQ_UPDATE_TIMEOUT);
    results = l1_word(base + kResultsOffset);

    const uint32_t firmware_seq_update_timeout = eth_txq_reg_read(0, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
    results[R_SEQ_TIMEOUT_FIRMWARE] = firmware_seq_update_timeout;
    if (seq_update_timeout != 0) {
        eth_txq_reg_write(0, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, seq_update_timeout);
    }

    init_ptr_val<PTR_STREAM>(0);
    if constexpr (USES_TXQ1) {
        // As the router sets up its TXQ1 in two-ERISC mode: resend mode, its own TX packet config entry, and the
        // receive side routing those packets to RXQ1. Both ends call it.
        eth_enable_packet_mode(1);
        eth_txq_reg_write(1, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, kAcceptAhead);
    }

    if (is_sender) {
        erisc::datamover::handshake::sender_side_handshake(base + kHandshakeOffset, 0, 0);
    } else {
        erisc::datamover::handshake::receiver_side_handshake(base + kHandshakeOffset, 0, 1);
    }
    sample_txq_counters(R_TXQ_BEFORE);
    sample_rxq_drops(R_RXQ_DROPS_BEFORE);

    const bool ok = is_sender ? run_sender(base, payload, n, warm, slots) : run_receiver(base, n, warm, slots);

    for (uint32_t q = 0; q < kNumQueues; q++) {
        if (!wait_txq_idle(q, is_sender ? kErrSenderTxq : kErrReceiverTxq, n)) {
            break;
        }
    }
    sample_txq_counters(R_TXQ_AFTER);
    sample_rxq_drops(R_RXQ_DROPS_AFTER);
    sample_txq_config();
    results[R_SEQ_TIMEOUT_USED] = eth_txq_reg_read(0, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
    if (seq_update_timeout != 0) {
        eth_txq_reg_write(0, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, firmware_seq_update_timeout);
    }
    if (ok && results[R_STATUS] == kRunning) {
        results[R_STATUS] = kDone;
    }
}
