// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Power-virus Ethernet link kernel (active ERISC0). The same kernel runs on
// both ends of every link that is up; each side sends
// payloads to its peer until the host sets the STOP word in this core's L1, and
// acks every payload it receives, so SerDes TX and RX toggle in both
// directions. Flow control and handshake follow the ethernet ubench kernels
// (tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/
//  ethernet_write_worker_latency_ubench_common.hpp).
//
// When STOP is seen a side stops sending and sends a FIN carrying its sent
// count. Both sides keep acking until the peer's
// FIN has arrived, all of the peer's payloads are received and all of their own
// payloads are acked. ETH packets are delivered in order, so FIN arrives after
// the last payload.
//
// Throttle knobs:
//   burst      – payloads per burst; 0 = continuous
//   idle_units – idle after each burst, units of 512 wall-clock ticks.
//                Incoming payloads are still acked while idle.
//
// L1 layout from runtime arg l1_base (unreserved ETH L1), see pv_common.hpp:
//   [0, 16)   handshake word (written by peer)
//   [16, 32)  timing record staging, persisted to DRAM at the end
//   [32, 48)  FIN word (written by peer): [magic, peer_sent, 0, 0]
//   [48, 64)  FIN staging (sent to the peer's FIN word)
//   [64, 80)  STOP word (written by the host)
//   [128, ...) region A: num_slots × (xfer_bytes + 16 B sync)
//   [...]     region B: same size
// Role 0 sends from region A and receives into region B; role 1 is mirrored,
// so a payload sent from slot i lands in the peer's receive slot i.
//
// Compile-time args:
//   0: num_slots     – ring depth per direction (power of two)
//   1: xfer_bytes    – payload bytes per message (multiple of 16)
//   2: burst
//   3: idle_units
//   4: startup_delay_ticks – optional wall-clock delay before the handshake
//   5: period_units  – phase-locked mode period (units of 512 ticks, power of
//                      two); 0 = off. When on, payloads are only sent while
//                      (wall_clock / 512) % period < active_units.
//   6: active_units
//
// Runtime args:
//   0: l1_base
//   1: role          – 0 or 1 (opposite on the peer)
//   2: dram_timing_bank_id
//   3: dram_timing_addr – DRAM address (in bank) for the 16 B timing record
//
// Timing record: [elapsed_lo, elapsed_hi, sent, received]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/tt_eth_api.h"
#include "risc_common.h"
#include "pv_common.hpp"

namespace {

struct slot_sync_t {
    volatile uint32_t bytes_sent;
    volatile uint32_t receiver_ack;
    volatile uint32_t reserved_0;
    volatile uint32_t reserved_1;
};
static_assert(sizeof(slot_sync_t) == 16);

constexpr uint32_t kIdleUnitTicks = 512;
constexpr uint32_t kHeaderBytes = pv::kEthHeaderBytes;
constexpr uint32_t kStopCheckInterval = 256;
constexpr uint32_t kRoutingInterval = 4096;
constexpr uint32_t kFinMagic = 0xF1F1F1F1u;

constexpr uint32_t num_slots = get_compile_time_arg_val(0);
constexpr uint32_t xfer_bytes = get_compile_time_arg_val(1);
constexpr uint32_t burst = get_compile_time_arg_val(2);
constexpr uint32_t idle_units = get_compile_time_arg_val(3);
constexpr uint32_t startup_delay_ticks = get_compile_time_arg_val(4);
constexpr uint32_t period_units = get_compile_time_arg_val(5);
constexpr uint32_t active_units = get_compile_time_arg_val(6);
static_assert(period_units == 0 || (period_units & (period_units - 1)) == 0, "period_units must be a power of two");
constexpr uint32_t slot_stride = xfer_bytes + sizeof(slot_sync_t);
static_assert((num_slots & (num_slots - 1)) == 0 && num_slots >= 2, "num_slots must be a power of two >= 2");
static_assert(xfer_bytes % 16 == 0, "xfer_bytes must be a multiple of 16");

FORCE_INLINE uint32_t next(uint32_t p) { return (p + 1) & (num_slots - 1); }

// Only the low wall-clock word is read: if another ERISC kernel runs, its
// interleaved reads of the latched 64-bit clock could pair a stale high word.
FORCE_INLINE uint32_t wall_clock_l() { return eth_risc_reg_read(ETH_RISC_WALL_CLOCK_0); }

FORCE_INLINE volatile slot_sync_t* sync_of(uint32_t slot_addr) {
    return reinterpret_cast<volatile slot_sync_t*>(slot_addr + xfer_bytes);
}

}  // namespace

void kernel_main() {
    const uint32_t l1_base = get_arg_val<uint32_t>(0);
    const uint32_t role = get_arg_val<uint32_t>(1);
    const uint32_t dram_timing_bank_id = get_arg_val<uint32_t>(2);
    const uint32_t dram_timing_addr = get_arg_val<uint32_t>(3);

    const uint32_t handshake_addr = l1_base + pv::kEthHandshake;
    volatile tt_l1_ptr uint32_t* timing = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + pv::kEthTiming);
    const uint32_t fin_addr = l1_base + pv::kEthFinIn;
    volatile tt_l1_ptr uint32_t* fin_in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fin_addr);
    volatile tt_l1_ptr uint32_t* fin_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + pv::kEthFinOut);
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + pv::kEthStop);
    const uint32_t region_a = l1_base + kHeaderBytes;
    const uint32_t region_b = region_a + num_slots * slot_stride;
    const uint32_t tx_base = role == 0 ? region_a : region_b;
    const uint32_t rx_base = role == 0 ? region_b : region_a;

    // TX slots start "busy" (bytes_sent != 0) until the peer acks by zeroing it;
    // a fresh slot is free, so mark them all acked (0). RX slots start empty.
    for (uint32_t i = 0; i < num_slots; i++) {
        const uint32_t tx_slot = tx_base + i * slot_stride;
        volatile tt_l1_ptr uint32_t* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tx_slot);
        for (uint32_t w = 0; w < xfer_bytes / sizeof(uint32_t); w++) {
            // Alternating pattern toggles SerDes lanes.
            payload[w] = (w & 1) ? 0xA5A5A5A5u : 0x5A5A5A5Au;
        }
        sync_of(tx_slot)->bytes_sent = 0;
        sync_of(rx_base + i * slot_stride)->bytes_sent = 0;
    }
    // The peer only sends FIN after the handshake, so clearing it here is safe.
    for (uint32_t w = 0; w < 4; w++) {
        fin_in[w] = 0;
        fin_out[w] = 0;
    }

    if constexpr (startup_delay_ticks != 0) {
        eth_wait_cycles(startup_delay_ticks);
    }

    // Handshake (also runs base-FW routing while waiting).
    if (role == 0) {
        eth_send_bytes(handshake_addr, handshake_addr, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }

    uint32_t prev_clk = wall_clock_l();
    uint64_t elapsed = 0;  // ticks since the handshake

    uint32_t tx_wr = 0;  // next TX slot to send
    uint32_t tx_rd = 0;  // oldest TX slot awaiting ack
    uint32_t tx_inflight = 0;
    uint32_t sent = 0;
    uint32_t rx_rd = 0;  // next RX slot expected
    uint32_t received = 0;

    bool sending = true;  // own send window still open
    bool fin_sent = false;
    uint32_t burst_left = burst;
    bool idling = false;
    uint64_t idle_end = 0;
    uint32_t iter = 0;
    uint32_t stop_check = 0;

    while (true) {
        const uint32_t now = wall_clock_l();
        elapsed += now - prev_clk;
        prev_clk = now;

        // --- host stop: announce FIN once the TX queue is free ---
        if (sending && ++stop_check == kStopCheckInterval) {
            stop_check = 0;
            invalidate_l1_cache();
            if (*stop != 0) {
                sending = false;
            }
        }
        if (!sending && !fin_sent && !eth_txq_is_busy()) {
            fin_out[0] = kFinMagic;
            fin_out[1] = sent;
            eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
                reinterpret_cast<uint32_t>(fin_out), fin_addr, 16);
            fin_sent = true;
        }

        // --- TX: send next payload if a slot is free and we are not idling ---
        if (idling && elapsed >= idle_end) {
            idling = false;
            burst_left = burst;
        }
        bool in_phase = true;
        if constexpr (period_units != 0) {
            // 2^32 / 512 is a multiple of period_units, so the low clock word
            // gives a consistent phase across wrap-around.
            in_phase = ((now / kIdleUnitTicks) & (period_units - 1)) < active_units;
        }
        if (sending && in_phase && !idling && tx_inflight < num_slots && !eth_txq_is_busy()) {
            const uint32_t slot = tx_base + tx_wr * slot_stride;
            sync_of(slot)->bytes_sent = xfer_bytes;
            eth_send_bytes_over_channel_payload_only_unsafe_one_packet(slot, slot, slot_stride);
            tx_wr = next(tx_wr);
            tx_inflight++;
            sent++;
            if constexpr (burst != 0 && idle_units != 0) {
                if (--burst_left == 0) {
                    idling = true;
                    idle_end = elapsed + static_cast<uint64_t>(idle_units) * kIdleUnitTicks;
                }
            }
        }

        // --- TX: retire acked slots (peer zeroes bytes_sent) ---
        if (tx_inflight != 0) {
            invalidate_l1_cache();
            if (sync_of(tx_base + tx_rd * slot_stride)->bytes_sent == 0) {
                tx_rd = next(tx_rd);
                tx_inflight--;
            }
        }

        // --- RX: ack incoming payload by zeroing its sync word on the peer ---
        invalidate_l1_cache();
        volatile slot_sync_t* rx_sync = sync_of(rx_base + rx_rd * slot_stride);
        if (rx_sync->bytes_sent != 0 && !eth_txq_is_busy()) {
            rx_sync->bytes_sent = 0;
            const uint32_t sync_addr = reinterpret_cast<uint32_t>(rx_sync);
            eth_send_bytes_over_channel_payload_only_unsafe_one_packet(sync_addr, sync_addr, sizeof(slot_sync_t));
            rx_rd = next(rx_rd);
            received++;
        }

        // --- done: own FIN out, own payloads all acked, peer finished and drained ---
        if (fin_sent && tx_inflight == 0 && fin_in[0] == kFinMagic && received == fin_in[1]) {
            break;
        }

        if (++iter == kRoutingInterval) {
            iter = 0;
            run_routing();
        }
    }

    // Let the final ack leave the TX queue before reusing L1 / exiting.
    while (eth_txq_is_busy()) {
    }

    // Persist timing in DRAM; active-ETH L1 is reclaimed after the kernel exits.
    timing[0] = static_cast<uint32_t>(elapsed);
    timing[1] = static_cast<uint32_t>(elapsed >> 32);
    timing[2] = sent;
    timing[3] = received;
    const uint64_t dst = get_noc_addr_from_bank_id<true>(dram_timing_bank_id, dram_timing_addr);
    noc_async_write(reinterpret_cast<uint32_t>(timing), dst, 16);
    eth_noc_async_write_barrier();
}
