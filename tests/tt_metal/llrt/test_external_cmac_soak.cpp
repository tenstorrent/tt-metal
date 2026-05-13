// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Sustained-rate soak for the WH external CMAC TX path — WQE-ring (v1).
//
// Pipelined post_send loop. Replaces the legacy per-frame send/wait_tx_consumed
// handshake with the host→FW ring (kWqeRingN slots): post_send returns
// immediately once the descriptor is published, and the ring naturally
// backpressures the host when full. A single wait_completion(last_seq) at the
// end drains the pipe — completion polling is amortised across all frames.
//
// Each frame embeds a 32-bit sequence number at payload offset 14 so a wire
// capture or echo receiver can verify ordering. CMAC prepends its own 14-byte
// L2 header, so the seq# lives at wire-byte offset 28.
//
// Continue+tally semantics: ring-full give-ups and completion timeouts do
// NOT abort the run. End-of-test summary reports posted / completed / drops.
//
// Knobs (env vars):
//   TT_METAL_EXTERNAL_CMAC_PORTS         "chip:chan" — same as the smoke test
//   TT_METAL_CMAC_POST_LINK_SETTLE_MS    settle after wait_for_link (default 2500)
//   TT_METAL_CMAC_SOAK_FRAMES            N iterations (default 100000)
//   TT_METAL_CMAC_SOAK_FRAME_SIZE        bytes (default 256, max 1500)
//   TT_METAL_CMAC_SOAK_FULL_BACKOFF_MS   max wait when ring is full before
//                                         dropping the frame (default 2000)
//   TT_METAL_CMAC_SOAK_DRAIN_TIMEOUT_MS  final wait_completion deadline (default 5000)
//   TT_METAL_CMAC_SOAK_LOG_EVERY         log progress every K iterations (default 10000)
//   TT_METAL_CMAC_SOAK_INTER_FRAME_US    optional inter-frame sleep, microseconds (default 0)
//
// Exit code is 0 if at least one frame completed; non-zero only if the FW
// never advanced cq_head (degenerate failure). Partial losses are reported
// in the summary but don't fail the test.

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <sstream>
#include <thread>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/external_iface_sender.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

namespace {

uint32_t env_uint(const char* name, uint32_t default_value) {
    const char* s = std::getenv(name);
    if (s == nullptr || s[0] == '\0') {
        return default_value;
    }
    return static_cast<uint32_t>(std::strtoul(s, nullptr, 0));
}

std::vector<uint8_t> build_frame(
    const std::array<uint8_t, 6>& /*dst_mac*/, const std::array<uint8_t, 6>& /*src_mac*/, uint32_t seq, uint32_t size) {
    // TT-RDMA v1 SEND frame. WH CMAC strips the wire L2 header on RX, so the
    // 32 B RDMA header lives at frame[0]. Layout per [[project-tt-rdma-v1-spec]]:
    //   +0  opcode = 0x01 (SEND)
    //   +1  version_flags = 0x01
    //   +2  tag = low-16 of seq
    //   +4  length = payload bytes after the 32 B header
    //   +8  seq
    //   +12 rkey = 0
    //   +16 remote_offset = 0
    //   +24 imm = 0
    //   +28 hdr_crc = 0
    std::vector<uint8_t> frame(size, 0);
    frame[0] = 0x01;
    frame[1] = 0x01;
    uint16_t tag = static_cast<uint16_t>(seq & 0xFFFFu);
    std::memcpy(frame.data() + 2, &tag, 2);
    uint32_t length = (size >= 32u) ? (size - 32u) : 0u;
    std::memcpy(frame.data() + 4, &length, 4);
    std::memcpy(frame.data() + 8, &seq, 4);
    return frame;
}

}  // namespace

int main() {
    const auto& rtopts = tt::tt_metal::MetalContext::instance().rtoptions();
    if (!rtopts.has_external_cmac_ports()) {
        log_info(tt::LogTest, "SKIP: TT_METAL_EXTERNAL_CMAC_PORTS unset");
        return 0;
    }
    const auto& ports = rtopts.get_external_cmac_ports();
    const auto [chip_id_signed, eth_chan] = *ports.begin();
    const tt::ChipId chip_id = static_cast<tt::ChipId>(chip_id_signed);

    const uint32_t soak_frames = env_uint("TT_METAL_CMAC_SOAK_FRAMES", 100000);
    const uint32_t frame_size = env_uint("TT_METAL_CMAC_SOAK_FRAME_SIZE", 256);
    const uint32_t full_backoff_ms = env_uint("TT_METAL_CMAC_SOAK_FULL_BACKOFF_MS", 2000);
    const uint32_t drain_timeout_ms = env_uint("TT_METAL_CMAC_SOAK_DRAIN_TIMEOUT_MS", 5000);
    const uint32_t log_every = env_uint("TT_METAL_CMAC_SOAK_LOG_EVERY", 10000);
    const uint32_t settle_ms = env_uint("TT_METAL_CMAC_POST_LINK_SETTLE_MS", 2500);
    const uint32_t inter_frame_us = env_uint("TT_METAL_CMAC_SOAK_INTER_FRAME_US", 0);

    log_info(
        tt::LogTest,
        "SOAK(wqe-ring v1): chip={}, eth_chan={}, frames={}, frame_size={}B, full_backoff={}ms, drain={}ms, "
        "settle={}ms",
        chip_id,
        eth_chan,
        soak_frames,
        frame_size,
        full_backoff_ms,
        drain_timeout_ms,
        settle_ms);

    tt::tt_metal::IDevice* device = tt::tt_metal::CreateDevice(chip_id);

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    CoreCoord logical_core = soc_desc.get_eth_core_for_channel(eth_chan, tt::CoordSystem::LOGICAL);
    CoreCoord virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, logical_core, tt::CoreType::ETH);
    log_info(tt::LogTest, "Resolved eth_chan={} → virtual ({},{})", eth_chan, virtual_core.x, virtual_core.y);

    tt::llrt::ExternalIfaceSender sender(chip_id, virtual_core);

    // Order is load-bearing: enable_wqe_ring() MUST precede wait_for_link().
    // FW samples the mode bit only at gw-mode entry (when wait_for_link writes
    // GW_MODE_MAGIC); writing it after means FW already entered legacy mode.
    TT_FATAL(sender.enable_wqe_ring(), "enable_wqe_ring() failed (FW already dispatched?)");

    // Phase 3.2: opt-in FW DMA-pull. Stacks on top of enable_wqe_ring() —
    // the only thing it changes is RCB.mode (1 → 2) plus the host_noc_addr
    // publish. enable_wqe_ring still does descriptor-table zeroing and TLB
    // window registration which both modes need.
    const bool dma_pull = env_uint("TT_METAL_CMAC_DMA_PULL", 0) != 0;
    if (dma_pull) {
        TT_FATAL(sender.enable_fw_dma_pull(), "enable_fw_dma_pull() failed (hugepages exhausted, or KMD < 2.0.0?)");
        log_info(tt::LogTest, "FW DMA-pull ENABLED — payload via NoC PCIe read, BAR carries desc only.");
    }
    {
        // DMA-pull mode removes the payload-drain cost, so the optimal
        // batch-size is smaller (no PCIe RTT to amortise). Default to 1
        // (= no batching) for DMA-pull and 32 for Phase 1.
        const uint32_t drain_default = dma_pull ? 1u : 32u;
        const uint32_t drain_every = env_uint("TT_METAL_CMAC_DRAIN_EVERY", drain_default);
        sender.set_drain_every(drain_every);
        log_info(tt::LogTest, "WQE-ring mode armed (drain_every={}).", drain_every);
    }

    // Phase 2: opt-in reliability layer. With TT_METAL_CMAC_RELIABILITY=1,
    // post_send's window-full check uses acked_idx (not consumer_idx); the
    // soak ticks tick_retx() periodically inside the post loop.
    const bool reliability = env_uint("TT_METAL_CMAC_RELIABILITY", 0) != 0;
    if (reliability) {
        sender.enable_reliability(true);
        const uint32_t retx_us = env_uint("TT_METAL_CMAC_RETX_TIMEOUT_US", 10000);
        sender.set_retx_timeout_us(retx_us);
        log_info(tt::LogTest, "Reliability ENABLED (retx_timeout={}us).", retx_us);
    }

    log_info(tt::LogTest, "Waiting for PCS link + WQE-ring entry (10s)...");
    bool link_up = sender.wait_for_link(10000);
    TT_FATAL(link_up, "wait_for_link timed out (or FW did not enter run_wqe_ring_loop)");
    log_info(tt::LogTest, "PCS up, FW in run_wqe_ring_loop.");

    if (settle_ms > 0) {
        log_info(tt::LogTest, "Settling {}ms for Mellanox eswitch FDB...", settle_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
    }

    std::array<uint8_t, 6> dst_mac = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    std::array<uint8_t, 6> src_mac = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x00};

    // Counters
    uint64_t tx_posted = 0;
    uint64_t tx_full_giveup = 0;
    uint64_t full_backoff_hits = 0;
    bool first_full_logged = false;

    uint32_t last_seq = 0;
    bool any_posted = false;

    auto t0 = std::chrono::steady_clock::now();
    auto last_log_t = t0;
    uint64_t last_log_posted = 0;
    auto last_retx_tick = t0;
    uint64_t retx_triggered_count = 0;

    for (uint32_t i = 0; i < soak_frames; ++i) {
        // Phase 2: tick the retx state machine ~1 ms granularity. Cheap (one
        // PCIe read) and only fires when reliability is enabled.
        if (reliability) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::microseconds>(now - last_retx_tick).count() >= 1000) {
                if (sender.tick_retx()) {
                    retx_triggered_count++;
                }
                last_retx_tick = now;
            }
        }
        std::vector<uint8_t> frame = build_frame(dst_mac, src_mac, i, frame_size);

        // Pipelined post: spin on ring-full up to full_backoff_ms, then drop.
        std::optional<uint32_t> seq;
        const auto retry_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(full_backoff_ms);
        bool hit_full = false;
        while (true) {
            seq = sender.post_send(std::span<const uint8_t>(frame));
            if (seq.has_value()) {
                break;
            }
            hit_full = true;
            if (std::chrono::steady_clock::now() >= retry_deadline) {
                break;
            }
            // While stuck waiting for ring drain, drive the retx state machine
            // every 1ms. Without this, a long full-spin (e.g. ACKs not arriving)
            // never triggers retx and we just give up after the backoff.
            if (reliability) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::microseconds>(now - last_retx_tick).count() >= 1000) {
                    if (sender.tick_retx()) {
                        retx_triggered_count++;
                    }
                    last_retx_tick = now;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }

        if (hit_full) {
            full_backoff_hits++;
        }

        if (!seq.has_value()) {
            tx_full_giveup++;
            if (!first_full_logged) {
                log_warning(tt::LogTest, "First ring-full give-up at iter {} (after {}ms)", i, full_backoff_ms);
                first_full_logged = true;
            }
            continue;
        }

        last_seq = *seq;
        any_posted = true;
        tx_posted++;

        if (inter_frame_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(inter_frame_us));
        }

        if (log_every > 0 && (i + 1) % log_every == 0) {
            auto now = std::chrono::steady_clock::now();
            auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_t).count();
            uint64_t batch = tx_posted - last_log_posted;
            double batch_fps = (dt_ms > 0) ? (batch * 1000.0 / static_cast<double>(dt_ms)) : 0.0;
            uint32_t cq_head = sender.poll_completion();
            uint32_t inflight = sender.inflight();
            log_info(
                tt::LogTest,
                "  iter {}/{}: posted={} full_giveup={} cq_head={} inflight={} post_fps={:.0f}",
                i + 1,
                soak_frames,
                tx_posted,
                tx_full_giveup,
                cq_head,
                inflight,
                batch_fps);
            last_log_t = now;
            last_log_posted = tx_posted;
        }
    }

    auto t_post_end = std::chrono::steady_clock::now();
    auto post_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_post_end - t0).count();

    // Drain: one wait_completion for the last seq we successfully posted.
    bool drained = false;
    if (any_posted) {
        log_info(tt::LogTest, "Draining: wait_completion(last_seq={}, timeout={}ms)...", last_seq, drain_timeout_ms);
        drained = sender.wait_completion(last_seq, drain_timeout_ms);
    }

    // Phase 2: if reliability enabled, also wait for acked_idx to catch up.
    bool acked = false;
    if (reliability && any_posted) {
        log_info(tt::LogTest, "Draining: wait_acked(last_seq={}, timeout={}ms)...", last_seq, drain_timeout_ms);
        // Drive retx ticks in parallel while waiting.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(drain_timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            uint32_t a = sender.poll_acked();
            if (static_cast<int32_t>(a - last_seq) >= 0) {
                acked = true;
                break;
            }
            sender.tick_retx();
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto total_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    uint32_t cq_head_final = sender.poll_completion();
    uint32_t inflight_final = sender.inflight();

    // tx_completed: cq_head is the last completed seq#. seqs start at 0, so
    // completed_count = cq_head + 1 — but only if at least one frame was ever
    // posted. Before the first completion, cq_head reads 0xFFFFFFFF (FW init),
    // so guard with any_posted.
    uint64_t tx_completed = 0;
    if (any_posted) {
        // Wrap-safe: completed = cq_head - first_seq + 1, where first_seq=0.
        tx_completed = static_cast<uint64_t>(cq_head_final) + 1ULL;
    }

    double avg_post_fps = (post_elapsed_ms > 0) ? (tx_posted * 1000.0 / static_cast<double>(post_elapsed_ms)) : 0.0;
    double avg_completion_fps =
        (total_elapsed_ms > 0) ? (tx_completed * 1000.0 / static_cast<double>(total_elapsed_ms)) : 0.0;
    double success_rate = (soak_frames > 0) ? (100.0 * static_cast<double>(tx_completed) / soak_frames) : 0.0;

    log_info(tt::LogTest, "=== SOAK SUMMARY (WQE-ring v1) ===");
    log_info(tt::LogTest, "  iterations:          {}", soak_frames);
    log_info(tt::LogTest, "  tx_posted:           {}", tx_posted);
    log_info(tt::LogTest, "  tx_full_giveup:      {}", tx_full_giveup);
    log_info(tt::LogTest, "  full_backoff_hits:   {}", full_backoff_hits);
    log_info(tt::LogTest, "  cq_head_final:       {}", cq_head_final);
    log_info(tt::LogTest, "  tx_completed:        {}", tx_completed);
    log_info(tt::LogTest, "  inflight_final:      {}", inflight_final);
    log_info(tt::LogTest, "  drain_done:          {}", drained ? "yes" : "TIMEOUT");
    if (reliability) {
        log_info(tt::LogTest, "  acked_idx_final:     {}", sender.acked_idx_cached());
        log_info(tt::LogTest, "  acked_drain_done:    {}", acked ? "yes" : "TIMEOUT");
        log_info(tt::LogTest, "  retx_triggered:      {}", retx_triggered_count);
    }

    {
        // FW writes dbg counters at WQE_RCB_DBG_BASE = WQE_RCB_ADDR + 0x40 = 0x8440
        // (relocated from +0x20 in Phase 3 to free space for host_noc_addr).
        constexpr uint32_t kDbgAddr = 0x8400 + 0x40;
        auto dbg = cluster.read_core(chip_id, virtual_core, kDbgAddr, 13);
        log_info(tt::LogTest, "  fw_dbg_rx_frames:    {}", dbg[0]);
        log_info(tt::LogTest, "  fw_dbg_ack_matches:  {}", dbg[1]);
        log_info(tt::LogTest, "  fw_dbg_retx_runs:    {}", dbg[2]);
        log_info(tt::LogTest, "  fw_dbg_first_word:   0x{:08X}", dbg[3]);
        log_info(tt::LogTest, "  fw_dbg_rxbuf[0..15]: {:08x} {:08x} {:08x} {:08x}", dbg[4], dbg[5], dbg[6], dbg[7]);
        log_info(tt::LogTest, "  fw_dbg_rxq1_buf:     {} (zero=no RXQ1 traffic)", dbg[8]);
        log_info(tt::LogTest, "  fw_dbg_mac_rx_cnt:   {}", dbg[9]);
        log_info(tt::LogTest, "  fw_dbg_rxq0_drop:    {}", dbg[10]);
        log_info(tt::LogTest, "  fw_dbg_rxq1_drop:    {}", dbg[11]);
        log_info(tt::LogTest, "  fw_dbg_last_rx_size: {} (BUF_PTR snapshot)", dbg[12]);
    }
    log_info(tt::LogTest, "  post_elapsed_ms:     {}", post_elapsed_ms);
    log_info(tt::LogTest, "  total_elapsed_ms:    {}", total_elapsed_ms);
    log_info(tt::LogTest, "  avg_post_fps:        {:.0f}", avg_post_fps);
    log_info(tt::LogTest, "  avg_completion_fps:  {:.0f}", avg_completion_fps);
    log_info(tt::LogTest, "  success_rate:        {:.2f}%", success_rate);

    // Per-op timing breakdown. Always dumped (zero-filled if not built with
    // TT_METAL_CMAC_PROFILE) so the output format stays identical across
    // build flavours and downstream tooling can scrape it.
    {
        std::ostringstream oss;
        sender.profile().dump(oss);
        log_info(tt::LogTest, "{}", oss.str());
    }

    tt::tt_metal::CloseDevice(device);
    // Degenerate failure: nothing ever completed.
    return tx_completed == 0 ? 2 : 0;
}
