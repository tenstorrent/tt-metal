// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Zero-copy host-side handle for an external CMAC port.
//
// Design overview
// ---------------
// erisc_cmac_simple brings the PCS/FEC link up.  Once train_status == 0 the
// host writes a mode-switch magic word to L1:kModeAddr; the firmware (or
// erisc_cmac_gw after reload) enters the data-path loop described below.
//
// TX (host → CMAC) — zero copy
//   1. Host writes raw Ethernet frame bytes directly to TX_BUF0 or TX_BUF1 in
//      erisc L1 via PCIe MMIO (no erisc memcpy involved).
//   2. Host writes frame size to kTxSizeAddr and buffer selector to kTxBufSelAddr.
//   3. Firmware sees kTxSizeAddr != 0, programs ETH_TXQ_TRANSFER_START_ADDR /
//      ETH_TXQ_TRANSFER_SIZE_BYTES, fires ETH_TXQ_CMD_START_RAW, clears kTxSizeAddr.
//   Erisc is in the control path only; data bytes never pass through erisc registers.
//
// RX (CMAC → host) — zero copy, ping-pong double-buffered
//   1. CMAC DMA writes incoming frame bytes into one of two ping-pong buffers
//      (kPacketBuf at 0x4000 or kPacketBuf1 at 0x6C00).
//   2. When a frame completes, firmware re-arms the RX queue to write into the
//      OTHER buffer, then writes kRxBufSelAddr (0 or 1) and kRxWpAddr (word count).
//   3. Host polls kRxWpAddr; when it changes, reads rx_buf_sel() to find which
//      buffer holds the frame, then reads frame bytes directly from that buffer
//      in erisc L1 via PCIe MMIO.
//   4. No ack from the host is required — firmware already switched buffers before
//      publishing, so CMAC DMA fills the next buffer while the host reads.
//   Data bytes travel CMAC DMA → erisc L1 → host PCIe read.  No erisc copy.
//
// L1 memory map (matches erisc_cmac_gw.cpp and boot_params_t layout)
//   0x1000–0x1F40  existing firmware data (boot_params, node_info, debug, results)
//   0x1F40  kTxSizeAddr   — host writes frame size (bytes); erisc clears after TX arm
//   0x1F44  kTxBufSelAddr — host writes 0 or 1 to select TX_BUF0 / TX_BUF1
//   0x1F48  kRxWpAddr     — erisc writes word count of completed frame
//   0x1F4C  kRxRpAddr     — (unused in ping-pong mode; kept for ABI compatibility)
//   0x1F50  kModeAddr     — host writes kModeMagic to switch firmware to data-path loop
//   0x1F54  kRxBufSelAddr — erisc writes 0 or 1: which RX buffer holds the frame
//   0x4000–0x6000  kPacketBuf   (RX DMA buf 0; host reads directly)
//   0x6000–0x6600  kTxBuf0      (host writes TX frame here)
//   0x6600–0x6C00  kTxBuf1      (TX ping-pong partner)
//   0x6C00–0x7C00  kPacketBuf1  (RX DMA buf 1; 4 KB; host reads directly)

#include <array>
#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>

#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <tt-metalium/core_coord.hpp>                     // CoreCoord

namespace tt::llrt {

// Per-op timing for the WQE-ring post_send hot path.
// Always-defined so callers can construct/dump regardless of build flavour.
// Population is gated by TT_METAL_CMAC_PROFILE — production builds get zeros.
struct PostSendProfile {
    struct OpStat {
        uint64_t count = 0;
        uint64_t total_ns = 0;
        uint64_t min_ns = UINT64_MAX;
        uint64_t max_ns = 0;
        // Log-bucket histogram: <100ns, <200, <500, <1µ, <2µ, <5µ, <10µ, <20µ, ≥20µ
        uint64_t buckets[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        void record(uint64_t ns);
        double mean_ns() const { return count ? static_cast<double>(total_ns) / static_cast<double>(count) : 0.0; }
    };
    OpStat payload_write;
    OpStat desc_pre_owned;  // 4 word writes for size+flags=0, seq, payload_off, cookie
    OpStat payload_drain;   // read-back of payload start (PCIe round-trip)
    OpStat owned_publish;   // word0 re-write with OWNED_BY_FW set
    OpStat producer_write;  // producer_idx publish
    OpStat producer_drain;  // read-back of producer_idx (PCIe round-trip)
    OpStat consumer_read;   // ring-full check at start of post_send (PCIe round-trip)
    OpStat total_post;      // end-to-end post_send

    void dump(std::ostream& os) const;
};

class ExternalIfaceSender {
public:
    ExternalIfaceSender(ChipId chip_id, CoreCoord virtual_eth_core);

    // Poll RESULTS_BUF train_status until 0 (PCS lock), then write kModeMagic.
    // Returns false on timeout (default 5 s).
    bool wait_for_link(uint32_t timeout_ms = 5000);

    // True if PCS lock has been achieved (train_status == 0 in erisc results).
    bool is_link_up() const;

    // Write a raw Ethernet frame (caller-owned) into the next TX buffer slot
    // directly in erisc L1, then arm TX via the mailbox.
    // buf.size() must be <= kMaxFrameBytes.
    // Returns false if the previous TX has not yet been consumed by firmware.
    bool send(std::span<const uint8_t> buf);

    // Poll kTxSizeAddr until firmware clears it (doorbell consumed → CMAC TX armed).
    // Doubles as the gw-mode-entry handshake: erisc_cmac_simple only clears the
    // doorbell from inside run_gateway_loop, so a successful return implies both
    // (a) firmware switched out of burst mode into the data-path loop, and
    // (b) the most recent send() has been pushed to ETH_TXQ_CMD_START_RAW.
    // Returns false on timeout. Default 500 ms covers the worst-case mode-poll
    // interval (~130 ms — 65k burst-loop iterations including tx_fire register
    // writes) with margin, plus DMA arm latency. Once gw mode is active,
    // subsequent calls return in microseconds.
    bool wait_tx_consumed(uint32_t timeout_ms = 500);

    // Returns the word count of the completed RX frame published by firmware.
    // A value different from the last call means a new frame is available.
    // After calling rx_watermark(), use rx_buf_sel() to determine which buffer
    // holds the frame, then call read_rx(rx_buf_sel(), ...) to read it.
    uint32_t rx_watermark() const;

    // Which PACKET_BUF holds the frame signalled by the last rx_watermark() call.
    // 0 → kPacketBuf (0x4000), 1 → kPacketBuf1 (0x6C00).
    // Read kRxBufSelAddr before kRxWpAddr changes again (i.e. before the next
    // rx_watermark() poll iteration overwrites it).
    uint32_t rx_buf_sel() const;

    // Read received frame bytes directly from the specified buffer in erisc L1.
    // buf_sel selects the buffer: 0 → kPacketBuf, 1 → kPacketBuf1.
    // word_offset is the starting word index within the selected buffer.
    // Call after rx_watermark() returns a new value.
    void read_rx(uint32_t buf_sel, uint32_t word_offset, std::span<uint8_t> out) const;

    // DEPRECATED: No longer needed in ping-pong mode.
    // Firmware switches RX buffers before publishing kRxWpAddr, so CMAC DMA
    // fills the next buffer immediately — no host ack is required.
    // Kept for ABI compatibility; calling this is a no-op in practice.
    [[deprecated("rx_consume() is not needed with ping-pong RX buffering")]]
    void rx_consume(uint32_t consumed_wp);

    // ── WQE-ring mode (v1) ─────────────────────────────────────────────────
    // Enables the new ring path. After this call, post_send / poll_completion /
    // wait_completion are the right primitives; send() / wait_tx_consumed()
    // still work but should not be mixed with ring-mode calls.
    //
    // CALL ORDER (load-bearing): enable_wqe_ring() MUST be called BEFORE
    // wait_for_link(). The firmware reads the mode bit only at gw-mode entry
    // (when wait_for_link() writes GW_MODE_MAGIC). If wait_for_link() runs
    // first, FW enters legacy run_gateway_loop and the mode bit is ignored;
    // post_send() will appear to succeed but no frames will leave the ring.
    // wait_for_link() polls fw_status after writing the magic to verify FW
    // actually entered run_wqe_ring_loop and returns false if it didn't.
    //
    // SINGLE PRODUCER: post_send() is single-threaded. producer_local_ and
    // next_seq_ are non-atomic; concurrent post_send() from multiple threads
    // races. Wrap calls in your own mutex if needed.
    bool enable_wqe_ring();

    // Post a frame to the ring. Returns assigned seq# on success, std::nullopt
    // if the ring is full. Zero-copy: writes payload to L1 slot, drains the
    // payload write-combining buffer with a read-back, then writes the
    // descriptor (with OWNED_BY_FW set last), then bumps producer_idx.
    std::optional<uint32_t> post_send(std::span<const uint8_t> buf, uint32_t cookie = 0);

    // Returns the latest completed seq# (= rcb.cq_head). Pure read, no PCIe write.
    uint32_t poll_completion();

    // Block until poll_completion() >= seq, or timeout. Returns false on timeout.
    bool wait_completion(uint32_t seq, uint32_t timeout_ms = 100);

    // Slots currently in flight (= producer_idx - consumer_idx).
    uint32_t inflight() const;

    // Force any pending posts (held back by the batched-drain optimization)
    // to publish to FW. Called automatically by wait_completion(); callers
    // who poll_completion() and want strict visibility should call this.
    // No-op if nothing is pending.
    void flush_pending();

    // Set the number of posts to accumulate before issuing a single payload
    // drain + batch OWNED publish. Default 16. Setting to 1 disables batching
    // (every post is fully published before returning). Setting to 0 is
    // treated as 1 (no batching).
    void set_drain_every(uint32_t n);

    // ── Phase 2: reliability layer ─────────────────────────────────────────
    // Enable the sliding-window ARQ flow control. After enabling:
    //   - post_send's ring-full check uses (producer - acked_idx), not
    //     (producer - consumer_idx). Slots can't be overwritten until acked.
    //   - The caller is expected to drive tick_retx() periodically (e.g. every
    //     1 ms) so the timeout-driven retx fires when needed.
    // Disabled by default: the host behaves like Phase 1 with no window-stall.
    // Useful so workloads that don't care about reliability don't pay the
    // bookkeeping cost.
    void enable_reliability(bool enabled = true);
    bool reliability_enabled() const { return reliability_enabled_; }

    // Read FW's current acked_idx (RCB +0x08). One PCIe RTT.
    uint32_t poll_acked();

    // Block until poll_acked() >= seq, or timeout. Returns false on timeout.
    // Auto-flushes pending posts so seq is visible to FW.
    bool wait_acked(uint32_t seq, uint32_t timeout_ms = 1000);

    // Drive the retx state machine. Should be called periodically (e.g. every
    // 1 ms) by a caller-owned timer thread or from a polling loop.
    //
    //   - Refreshes acked_idx_cached_ from RCB.
    //   - If acked_idx hasn't advanced for retx_timeout_us_ AND there's at
    //     least one un-acked in-flight slot, writes RCB.retx_pending =
    //     producer_local_ to trigger FW-side retx walk. Resets the timer.
    //
    // No-op if reliability is disabled. Returns true if a retx was triggered
    // by this call (caller may want to record metrics).
    bool tick_retx();

    // Last value of acked_idx read by poll_acked() or tick_retx(). Cheap
    // accessor for instrumentation.
    uint32_t acked_idx_cached() const { return acked_idx_cached_; }

    // Set the retx timeout in microseconds. Default 10000 (10 ms).
    void set_retx_timeout_us(uint32_t us) { retx_timeout_us_ = (us < 100) ? 100 : us; }

    // ── Phase 3: FW DMA-pull plumbing (probe / opt-in mode) ────────────────
    // Allocate a hugepage-backed payload region, map it for NoC DMA via UMD
    // (PCIDevice::map_hugepage_to_noc), publish the NoC base address into
    // RCB+0x20/+0x24, and flip RCB.mode to 2 (= dma-pull mode).
    //
    // CALL ORDER: same constraint as enable_wqe_ring() — must be called
    // BEFORE wait_for_link(), because FW reads RCB.mode only at gw-mode
    // entry. Calling it after wait_for_link() leaves FW in mode 1 (or 0).
    //
    // Phase 3.1 (probe): host pre-fills the first 64 B of the hugepage,
    // the FW probe path reads them into L1 0x3000, and the host verifies
    // via PCIe-MMIO reads. No CMAC TX involved.
    //
    // Returns false if hugepage allocation/mapping fails or if the device
    // does not support map_hugepage_to_noc (KMD too old, IOMMU absent on a
    // non-IOMMU build, etc.). On failure leaves dma_pull_enabled_ false and
    // the RCB unchanged so the caller can fall back to mode-1 wqe-ring.
    bool enable_fw_dma_pull();

    // P7 (2026-05-13): dynamically register an MR entry in the FW MR table.
    // `slot` selects which MR_TABLE row (0..15) to write. `base_noc_addr` is
    // the 64-bit NoC-encoded destination (upper 32 = noc_xy, lower 32 = L1
    // address). `length` is bytes. `rkey` must encode the generation byte
    // in the low byte for the validation check (see project-p2-mr-table-align).
    // `access_flags` is MR_ACCESS_* bitmask.
    //
    // Synchronous: writes staging slot + doorbell, then polls until FW clears
    // the doorbell. timeout_us bounds the wait; returns false on timeout.
    // Call after enable_wqe_ring() + wait_for_link() so the FW is servicing.
    bool register_mr_slot(
        uint32_t slot,
        uint64_t base_noc_addr,
        uint64_t length,
        uint32_t rkey,
        uint32_t access_flags,
        uint32_t timeout_us = 100000);

    // Direct accessor for the hugepage VA — used by Phase 3.1 to pre-load
    // the probe pattern, and by Phase 3.2+ for per-slot payload memcpy.
    // Returns nullptr if enable_fw_dma_pull() has not succeeded.
    void* dma_pull_buffer() { return host_payload_buf_; }
    size_t dma_pull_buffer_size() const { return host_payload_size_; }
    bool dma_pull_enabled() const { return dma_pull_enabled_; }

    // Per-op timing of the post_send hot path. Populated only if compiled
    // with -DTT_METAL_CMAC_PROFILE; otherwise fields stay zero.
    const PostSendProfile& profile() const { return profile_; }

    ChipId chip_id() const { return chip_id_; }
    CoreCoord virtual_core() const { return virtual_core_; }

    // Releases the hugepage mapping if enable_fw_dma_pull() ran. Safe to
    // construct/destruct without ever calling enable_fw_dma_pull().
    ~ExternalIfaceSender();

private:
    ChipId chip_id_;
    CoreCoord virtual_core_;
    uint8_t tx_buf_sel_{0};   // ping-pong index (0 → kTxBuf0, 1 → kTxBuf1)
    uint32_t last_rx_wp_{0};  // last rx_watermark() value seen by host

    // WQE-ring (v1) state
    bool wqe_ring_enabled_{false};
    uint32_t next_seq_{0};        // assigned to next post_send
    uint32_t producer_local_{0};  // mirrors RCB producer_idx (host's view)
    mutable PostSendProfile profile_{};

    // Batched-drain state (Phase 1.4):
    //   post_send writes payload + descriptor (with OWNED=0) and tracks the
    //   slot/size in pending_*. Once pending_count_ reaches drain_every_,
    //   flush_pending() is called: one read-back drain, then OWNED publish on
    //   all pending slots, then producer_idx bump. This amortizes the ~7 µs
    //   payload PCIe transit cost across drain_every_ posts.
    static constexpr uint32_t kPendingMax = 64;  // = kWqeRingN; can't accumulate more
    std::array<uint32_t, kPendingMax> pending_slot_{};
    std::array<uint16_t, kPendingMax> pending_size_{};
    uint32_t pending_count_{0};
    // Empirically optimal on the WH↔CX-5 rig (2026-05-09): drain_every=32
    // yields ~1.94 Gbps, well above 16 (~1.88) and 64 (collapses — ring fills
    // before FW can drain). 32 leaves ~half the ring as headroom for FW to
    // chew through while host fills the next half.
    uint32_t drain_every_{32};
    uint32_t last_pending_payload_off_{0};  // for the drain read-back

    // Cached FW consumer_idx. Only re-read when the cached value puts us close
    // to ring-full. Cached value is always ≤ true value (FW only advances),
    // so over-estimating fullness from a stale cache is safe — we'd return
    // nullopt and the caller retries, at which point we re-read.
    uint32_t cons_cached_{0};

    // Phase 2: reliability layer state. Inactive unless enable_reliability()
    // is called.
    bool reliability_enabled_{false};
    uint32_t acked_idx_cached_{0};
    std::chrono::steady_clock::time_point last_ack_progress_{};
    uint32_t retx_timeout_us_{10000};  // 10 ms default

    // Phase 3: FW-DMA-pull state. host_payload_buf_ is a host-VA mapping of
    // a hugepage; host_payload_noc_addr_ / _pa_ are the (NoC, PA/IOVA) pair
    // returned by PCIDevice::map_hugepage_to_noc. Inactive unless
    // enable_fw_dma_pull() succeeds.
    void* host_payload_buf_{nullptr};
    uint64_t host_payload_noc_addr_{0};
    uint64_t host_payload_pa_{0};
    size_t host_payload_size_{0};
    bool dma_pull_enabled_{false};

    // Mailbox layout in erisc L1 (byte addresses)
    static constexpr uint32_t kTxSizeAddr = 0x1F40;
    static constexpr uint32_t kTxBufSelAddr = 0x1F44;
    static constexpr uint32_t kRxWpAddr = 0x1F48;
    static constexpr uint32_t kRxRpAddr = 0x1F4C;  // unused in ping-pong mode; kept for ABI compat
    static constexpr uint32_t kModeAddr = 0x1F50;
    static constexpr uint32_t kRxBufSelAddr = 0x1F54;  // erisc→host: 0=kPacketBuf, 1=kPacketBuf1
    static constexpr uint32_t kModeMagic = 0xDA7ADA7Au;

    // L1 buffer addresses
    static constexpr uint32_t kPacketBuf = 0x4000;   // CMAC RX DMA buf 0 (8 KB)
    static constexpr uint32_t kPacketBuf1 = 0x6C00;  // CMAC RX DMA buf 1 (4 KB)
    // P1 (2026-05-12) — TX_BUF0/1 relocated past WQE_PAYLOAD region to free
    // 0x6000-0x6C00 for the 14 KB RX ring. Must match FW eth_cmac_init.h.
    static constexpr uint32_t kTxBuf0 = 0x29000;
    static constexpr uint32_t kTxBuf1 = 0x2A000;
    static constexpr uint32_t kMaxFrameBytes = 4080;  // jumbo target (CMAC max-pkt=4096)

    // test_results_t at RESULTS_BUF_ADDR (0x1E00). Order: heartbeat,
    // train_status, pcs_status, fec_status, link_inactive, mac_hi, mac_lo,
    // tx_count, rx_count.
    static constexpr uint32_t kHeartbeatAddr = 0x1E00;
    static constexpr uint32_t kTrainStatusAddr = 0x1E04;

    // ── WQE-ring v1 layout (must match firmware main_cmac.cc EXACTLY) ──────
    // 2026-05-12 jumbo: ring 32 × 4096 B = 128 KB at 0x9000-0x29000.
    static constexpr uint32_t kWqeRingLog2N = 5;
    static constexpr uint32_t kWqeRingN = 1u << kWqeRingLog2N;  // 32
    static constexpr uint32_t kWqeDescTableAddr = 0x8000;
    static constexpr uint32_t kWqeDescStride = 16;
    static constexpr uint32_t kWqeRcbAddr = 0x8400;
    static constexpr uint32_t kWqePayloadBase = 0x9000;
    static constexpr uint32_t kWqePayloadStride = 4096;
    static constexpr uint16_t kWqeFlagOwnedByFw = 0x0001;
    static constexpr uint16_t kWqeFlagLastBurst = 0x0002;
    static constexpr uint16_t kWqeFlagReqCompl = 0x0004;
    static constexpr uint16_t kWqeFlagDiagSingle = 0x0008;

    // RCB field offsets (8 × u32 starting at kWqeRcbAddr).
    static constexpr uint32_t kRcbProducerOff = 0x00;     // host writes
    static constexpr uint32_t kRcbConsumerOff = 0x04;     // FW writes
    static constexpr uint32_t kRcbAckedIdxOff = 0x08;     // Phase 2: FW writes (on ACK rx)
    static constexpr uint32_t kRcbFwStatusOff = 0x0C;     // FW writes
    static constexpr uint32_t kRcbCqHeadOff = 0x10;       // FW writes
    static constexpr uint32_t kRcbRetxPendingOff = 0x14;  // Phase 2: host writes (FW clears)
    static constexpr uint32_t kRcbRingLog2NOff = 0x18;    // informational hint;
                                                          // FW currently hardcodes N=64
                                                          // (kWqeRingN). Field reserved
                                                          // for future runtime-tunable N.
    static constexpr uint32_t kRcbModeOff = 0x1C;         // host writes 0/1/2
    // Phase 3: extension fields beyond the original 8-word RCB header.
    // Host writes the (lo, hi) halves of the NoC-encoded hugepage base
    // returned by PCIDevice::map_hugepage_to_noc; FW reads at gw-mode entry.
    // Diag counters used to live at +0x20; relocated to +0x40 in Phase 3
    // (kept here as a stale-read guard for any tooling that still parses
    // the old layout — see firmware comment).
    static constexpr uint32_t kRcbHostNocAddrLoOff = 0x20;
    static constexpr uint32_t kRcbHostNocAddrHiOff = 0x24;
    // P7: control-plane doorbell + slot index for CTRL_REG_MR / CTRL_DEREG_MR.
    // Host writes a 32 B mr_entry_t blob into kMrStagingAddr, the slot index
    // into kRcbCtrlSlotOff, then a non-zero sub-op into kRcbCtrlDoorbellOff.
    // FW services + clears the doorbell.
    static constexpr uint32_t kRcbCtrlDoorbellOff = 0x28;
    static constexpr uint32_t kRcbCtrlSlotOff = 0x2C;
    static constexpr uint32_t kMrStagingAddr = 0x8700;
    static constexpr uint32_t kCtrlSubopRegMr = 0x01;
    static constexpr uint32_t kCtrlSubopDeregMr = 0x02;

    // Mode field values — matches firmware's switch in main_cmac.cc.
    static constexpr uint32_t kRcbModeLegacy = 0;
    static constexpr uint32_t kRcbModeWqeRing = 1;
    static constexpr uint32_t kRcbModeDmaPull = 2;

    // Probe target in L1 (gap between mailbox and PACKET_BUF). Host reads
    // 64 B from here after enable_fw_dma_pull() + wait_for_link() to verify
    // FW pulled the pre-loaded pattern from the host hugepage.
    static constexpr uint32_t kDmaPullProbeDst = 0x3000;
    static constexpr uint32_t kDmaPullProbeLen = 64;
    // FW writes this magic into RCB.fw_status when the probe NoC read
    // completes, so the host can poll for completion.
    static constexpr uint32_t kFwStatusProbeOk = 0xFEEDFACEu;
};

}  // namespace tt::llrt
