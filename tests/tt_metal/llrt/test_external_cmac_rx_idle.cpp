// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// RX-only idle harness for measuring the pure-RX path ceiling.
//
// Companion to test_external_cmac_soak.cpp. The soak test drives WH TX at
// 11 Gbps and measures RX echoed from the Mellanox receiver, but the WH FW's
// run_wqe_ring_loop spends most of its time servicing TX descriptor publishes
// — only ~64% of CMAC-admitted RX frames make it through to dbg_rx_frames.
// To isolate the pure-RX ceiling, this harness puts the WH FW into
// run_wqe_ring_loop (via enable_wqe_ring + wait_for_link) but does NOT post
// any TX. The Mellanox side runs --tx-probe N to drive incoming frames.
//
// Outputs the deltas across the sleep window:
//   - CMAC RXQ0/RXQ1 PKT_END_CNT  : frames CMAC admitted off the wire
//   - CMAC RXQ0/RXQ1 PACKET_DROP  : frames CMAC dropped at the MAC layer
//   - FW dbg_rx_frames            : frames the FW poll loop actually saw
//
// The gap (PKT_END_CNT - dbg_rx_frames) with PACKET_DROP=0 is the FW-loop
// overhead — what we're trying to characterise.
//
// Knobs (env vars):
//   TT_METAL_EXTERNAL_CMAC_PORTS         "chip:chan" — same as the smoke test
//   TT_METAL_CMAC_POST_LINK_SETTLE_MS    settle after wait_for_link (default 2500)
//   TT_METAL_CMAC_RX_IDLE_DURATION_MS    sleep window in ms (default 30000)

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

// CMAC RX register MMIO addresses (visible from the erisc NOC, hence readable
// via cluster.read_core targeted at the erisc virtual core). Canonical map
// lives in budabackend-master/src/firmware/riscv/targets/erisc/src/api/eth_ss_regs.h.
struct CmacRxSnap {
    uint32_t rxq0_pkt_end;
    uint32_t rxq0_drop;
    uint32_t rxq0_outstd;
    uint32_t rxq1_pkt_end;
    uint32_t rxq1_drop;
};

CmacRxSnap read_cmac_rx(const tt::Cluster& cluster, tt::ChipId chip_id, const CoreCoord& core) {
    auto read = [&](uint32_t addr) -> uint32_t {
        auto v = cluster.read_core(chip_id, core, addr, sizeof(uint32_t));
        return v[0];
    };
    return CmacRxSnap{
        .rxq0_pkt_end = read(0xFFB92028),
        .rxq0_drop = read(0xFFB9204C),
        .rxq0_outstd = read(0xFFB92050),
        .rxq1_pkt_end = read(0xFFB93028),
        .rxq1_drop = read(0xFFB9304C),
    };
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

    const uint32_t duration_ms = env_uint("TT_METAL_CMAC_RX_IDLE_DURATION_MS", 30000);
    const uint32_t settle_ms = env_uint("TT_METAL_CMAC_POST_LINK_SETTLE_MS", 2500);

    log_info(
        tt::LogTest,
        "RX-IDLE: chip={}, eth_chan={}, duration={}ms, settle={}ms",
        chip_id,
        eth_chan,
        duration_ms,
        settle_ms);

    tt::tt_metal::IDevice* device = tt::tt_metal::CreateDevice(chip_id);

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    CoreCoord logical_core = soc_desc.get_eth_core_for_channel(eth_chan, tt::CoordSystem::LOGICAL);
    CoreCoord virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, logical_core, tt::CoreType::ETH);
    log_info(tt::LogTest, "Resolved eth_chan={} → virtual ({},{})", eth_chan, virtual_core.x, virtual_core.y);

    tt::llrt::ExternalIfaceSender sender(chip_id, virtual_core);

    // Same call order as soak: enable_wqe_ring() BEFORE wait_for_link() so FW
    // enters run_wqe_ring_loop (which polls CMAC RX). Without this, FW stays
    // in legacy run_gateway_loop and the RX path is different.
    TT_FATAL(sender.enable_wqe_ring(), "enable_wqe_ring() failed");

    // P5: enable DMA-pull mode so the host hugepage is mapped + its NoC
    // base is published to RCB+0x20/0x24. FW reuses this same hugepage as
    // the RX ring (slots at offset 128 KB onwards, 64 × 1536 B). With this
    // call, FW will NoC-push received SEND/SEND_IMM frames into the
    // hugepage in addition to the legacy GW_RX_BUFSEL path.
    bool rx_ring_ok = sender.enable_fw_dma_pull();
    if (!rx_ring_ok) {
        log_warning(
            tt::LogTest,
            "enable_fw_dma_pull failed — P5 RxWqeRing disabled "
            "(FW won't DMA-push SENDs to host). Continuing legacy.");
    } else {
        log_info(tt::LogTest, "DMA-pull on; RX ring at hugepage+128KB (64 × 1536 B).");
    }

    const uint32_t link_timeout_ms = env_uint("TT_METAL_CMAC_LINK_TIMEOUT_MS", 10000);
    log_info(tt::LogTest, "Waiting for PCS link + WQE-ring entry ({}ms)...", link_timeout_ms);
    bool link_up = sender.wait_for_link(link_timeout_ms);
    TT_FATAL(link_up, "wait_for_link timed out (or FW did not enter run_wqe_ring_loop)");
    log_info(tt::LogTest, "PCS up, FW in run_wqe_ring_loop.");

    if (settle_ms > 0) {
        log_info(tt::LogTest, "Settling {}ms for Mellanox eswitch FDB...", settle_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
    }

    // P7: register MR slot 2 dynamically via the host CTRL_REG_MR doorbell.
    // Picks a NoC self-encoded base at L1:0x40000 (free region past TX_BUFs +
    // MR staging), 32 KB, rkey 0x02 (slot encoded in high byte per
    // [[project-p2-mr-table-align]]: rkey>>24=slot, low byte=generation).
    // Slot 0 stays hardcoded (back-compat). Verifies the doorbell handshake +
    // dbg[29] increments. Read-back of MR_TABLE[2] confirms staging→table copy.
    {
        // Read NoC XY of this erisc to compose the local NoC address.
        // Slot 2 → L1:0x40000, length 0x8000, rkey = (slot<<24)|0x01.
        const uint64_t my_noc_base =
            (static_cast<uint64_t>(virtual_core.x) << 36) | (static_cast<uint64_t>(virtual_core.y) << 42) | 0x40000ull;
        const uint32_t mr2_rkey = (2u << 24) | 0x01u;
        const uint32_t mr2_access = 0x6u;  // REMOTE_WR | REMOTE_RD
        bool ok = sender.register_mr_slot(/*slot=*/2u, my_noc_base, /*length=*/0x8000u, mr2_rkey, mr2_access);
        log_info(tt::LogTest, "P7 register_mr_slot(2) → {}", ok ? "OK" : "TIMEOUT");
        auto mr2_w = cluster.read_core(chip_id, virtual_core, 0x8500u + 2u * 32u, 32u);
        // mr2_w is vector<uint32_t> of 8 elements. mr_entry_t bytes 0x1C/0x1D
        // (generation + state) pack into the low half of mr2_w[7].
        uint8_t mr2_gen = static_cast<uint8_t>(mr2_w[7] & 0xFFu);
        uint8_t mr2_state = static_cast<uint8_t>((mr2_w[7] >> 8) & 0xFFu);
        log_info(
            tt::LogTest,
            "P7 MR_TABLE[2]: base=0x{:08X}_{:08X} len=0x{:08X}_{:08X} rkey=0x{:08X} "
            "access=0x{:X} gen={} state={}",
            mr2_w[1],
            mr2_w[0],
            mr2_w[3],
            mr2_w[2],
            mr2_w[4],
            mr2_w[5],
            mr2_gen,
            mr2_state);
    }

    // Q(c) seed: write a known 16-word pattern into L1@0x30000 so a READ_REQ
    // can verify the FW NoC-read fetched the right bytes from MR slot 0's
    // target. Used to be unwritten (zeros from boot) until a prior WRITE
    // landed something there. With the pre-seed, READ_REQ validation works
    // independently of any prior WRITE.
    {
        std::vector<uint32_t> mr_seed = {
            0xDEADBEEFu,
            0xCAFEBABEu,
            0xFEEDFACEu,
            0x12345678u,
            0xABCDEF01u,
            0x55AA55AAu,
            0x99887766u,
            0x0BADC0DEu,
            0xC0FFEE00u,
            0xBABEFACEu,
            0x13371337u,
            0xFACEFEEDu,
            0x42424242u,
            0xA5A5A5A5u,
            0x5A5A5A5Au,
            0xFEEDBEEFu,
        };
        cluster.write_core(chip_id, virtual_core, std::span<uint32_t>(mr_seed), 0x30000u);
        log_info(tt::LogTest, "Q(c) seeded L1@0x30000 with 64B marker pattern");
    }

    // Phase I — exercise host-initiated READ if the environment opts in.
    // Set TT_METAL_CMAC_TEST_READ_INIT=1 (or N for N concurrent reads, <=16).
    // Drives smoke_mlx with --rx-echo-read to get synthetic READ_RESPs back.
    const uint32_t read_init_count = env_uint("TT_METAL_CMAC_TEST_READ_INIT", 0);
    constexpr uint32_t kLandingL1Base = 0x50000u;
    constexpr uint32_t kReadInitLen = 128u;  // bytes per response payload
    if (read_init_count > 0) {
        // Pre-zero the landing area so we can verify the echo wrote there.
        std::vector<uint32_t> zeros(read_init_count * kReadInitLen / 4u, 0u);
        cluster.write_core(chip_id, virtual_core, std::span<uint32_t>(zeros), kLandingL1Base);
        log_info(tt::LogTest, "Phase I: zeroed {} B at L1@0x{:X}", zeros.size() * 4u, kLandingL1Base);

        // Compose landing MR base NoC addr — same encoding as P7 above.
        const uint64_t landing_noc = (static_cast<uint64_t>(virtual_core.x) << 36) |
                                     (static_cast<uint64_t>(virtual_core.y) << 42) |
                                     static_cast<uint64_t>(kLandingL1Base);

        log_info(
            tt::LogTest,
            "Phase I: posting {} READs (length={} each, landing=0x{:08X}_{:08X})",
            read_init_count,
            kReadInitLen,
            static_cast<uint32_t>(landing_noc >> 32),
            static_cast<uint32_t>(landing_noc));

        std::vector<uint32_t> seqs;
        seqs.reserve(read_init_count);
        for (uint32_t i = 0; i < read_init_count; ++i) {
            auto s = sender.post_send_read(
                /*local_mr_noc=*/landing_noc,
                /*local_offset=*/i * kReadInitLen,
                /*remote_rkey=*/0x00000001u,  // hardcoded MR slot 0
                /*remote_offset=*/0,
                /*length=*/static_cast<uint16_t>(kReadInitLen),
                /*cookie=*/i);
            if (!s) {
                log_warning(tt::LogTest, "post_send_read({}) ring-full at i={}", read_init_count, i);
                break;
            }
            seqs.push_back(*s);
        }
        sender.flush_pending();
        log_info(
            tt::LogTest,
            "Phase I: {} READs posted (seqs {}..{})",
            seqs.size(),
            seqs.empty() ? 0u : seqs.front(),
            seqs.empty() ? 0u : seqs.back());

        // Don't wait_completion here — the cq_head bump happens when smoke_mlx
        // echoes back. The test's sleep window IS the wait. Verification of
        // landing happens after the window.
    }

    // FW dbg counter base — same offset as soak.
    constexpr uint32_t kDbgAddr = 0x8400 + 0x40;
    auto read_dbg = [&]() -> std::vector<uint32_t> {
        return cluster.read_core(chip_id, virtual_core, kDbgAddr, 32 * sizeof(uint32_t));
    };

    CmacRxSnap cmac_t0 = read_cmac_rx(cluster, chip_id, virtual_core);
    auto dbg_t0 = read_dbg();
    log_info(
        tt::LogTest,
        "T0 CMAC: rxq0_pkt_end={} rxq0_drop={} rxq0_outstd={} rxq1_pkt_end={} rxq1_drop={}",
        cmac_t0.rxq0_pkt_end,
        cmac_t0.rxq0_drop,
        cmac_t0.rxq0_outstd,
        cmac_t0.rxq1_pkt_end,
        cmac_t0.rxq1_drop);
    log_info(
        tt::LogTest,
        "T0 FW : dbg_rx_frames={} dbg_mac_rx_cnt={} dbg_rxq0_drop={} dbg_rxq1_drop={}",
        dbg_t0[0],
        dbg_t0[9],
        dbg_t0[10],
        dbg_t0[11]);
    log_info(tt::LogTest, "Sleeping {}ms — drive MLX --tx-probe NOW...", duration_ms);

    auto t_start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    auto t_end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    CmacRxSnap cmac_t1 = read_cmac_rx(cluster, chip_id, virtual_core);
    auto dbg_t1 = read_dbg();

    auto d = [](uint32_t a, uint32_t b) { return static_cast<int64_t>(b) - static_cast<int64_t>(a); };

    log_info(tt::LogTest, "=== RX-IDLE SUMMARY ===");
    log_info(tt::LogTest, "  elapsed_ms:            {}", elapsed_ms);
    log_info(tt::LogTest, "  cmac_rxq0_pkt_end_d:   {}", d(cmac_t0.rxq0_pkt_end, cmac_t1.rxq0_pkt_end));
    log_info(tt::LogTest, "  cmac_rxq0_drop_d:      {}", d(cmac_t0.rxq0_drop, cmac_t1.rxq0_drop));
    log_info(
        tt::LogTest,
        "  cmac_rxq0_outstd_d:    {} (instantaneous outstanding count)",
        d(cmac_t0.rxq0_outstd, cmac_t1.rxq0_outstd));
    log_info(tt::LogTest, "  cmac_rxq1_pkt_end_d:   {}", d(cmac_t0.rxq1_pkt_end, cmac_t1.rxq1_pkt_end));
    log_info(tt::LogTest, "  cmac_rxq1_drop_d:      {}", d(cmac_t0.rxq1_drop, cmac_t1.rxq1_drop));
    log_info(tt::LogTest, "  fw_dbg_rx_frames_d:    {}", d(dbg_t0[0], dbg_t1[0]));
    log_info(tt::LogTest, "  fw_dbg_mac_rx_cnt_d:   {}", d(dbg_t0[9], dbg_t1[9]));
    log_info(tt::LogTest, "  fw_dbg_rxq0_drop_d:    {}", d(dbg_t0[10], dbg_t1[10]));
    log_info(tt::LogTest, "  fw_dbg_rxq1_drop_d:    {}", d(dbg_t0[11], dbg_t1[11]));
    log_info(tt::LogTest, "  fw_dbg_last_rx_size:   {}", dbg_t1[12]);
    log_info(
        tt::LogTest,
        "  fw_dbg_mixed_size_d:   {} (BUF_PTR not divisible by frame count — stride-walk degraded)",
        d(dbg_t0[13], dbg_t1[13]));
    log_info(tt::LogTest, "  fw_dbg_last_pkt_delta: {} (frames stride-walked in most-recent service)", dbg_t1[14]);
    log_info(
        tt::LogTest,
        "  fw_dbg_op_send_count:  {} (Phase B: frames recognized as RDMA opcode SEND)",
        d(dbg_t0[15], dbg_t1[15]));
    log_info(
        tt::LogTest, "  fw_dbg_op_unknown_d:   {} (P2: batches dropped as non-RDMA noise)", d(dbg_t0[16], dbg_t1[16]));
    log_info(
        tt::LogTest,
        "  fw_dbg_ctrl_processed: {} (P2: CONTROL opcode 0xF0 frames serviced)",
        d(dbg_t0[17], dbg_t1[17]));
    log_info(
        tt::LogTest,
        "  fw_dbg_op_write:       {} (P3: inbound WRITE frames successfully landed)",
        d(dbg_t0[18], dbg_t1[18]));
    log_info(tt::LogTest, "  fw_dbg_write_bad_rkey: {} (P3)", d(dbg_t0[19], dbg_t1[19]));
    log_info(tt::LogTest, "  fw_dbg_write_bad_bnds: {} (P3)", d(dbg_t0[20], dbg_t1[20]));
    log_info(tt::LogTest, "  fw_dbg_write_dst_word: 0x{:08X} (P3 sanity: last destination's first u32)", dbg_t1[21]);

    // P3 sanity: read back 16 bytes from MR slot 0's L1 destination (0x30000)
    // to confirm a WRITE actually landed bytes there.
    {
        auto mr_dst = cluster.read_core(chip_id, virtual_core, 0x30000u, 16);
        log_info(
            tt::LogTest,
            "  L1@0x30000 (MR slot 0): {:08x} {:08x} {:08x} {:08x}",
            mr_dst[0],
            mr_dst[1],
            mr_dst[2],
            mr_dst[3]);
    }

    log_info(
        tt::LogTest, "  fw_dbg_rx_ring_push:   {} (P5: SENDs DMA-pushed to host RxWqeRing)", d(dbg_t0[22], dbg_t1[22]));
    log_info(tt::LogTest, "  fw_dbg_rx_slot_idx:    {} (P5: next slot FW will use)", dbg_t1[23]);
    log_info(tt::LogTest, "  fw_dbg_op_read_req:    {} (P6: READ_REQs validated + staged)", d(dbg_t0[24], dbg_t1[24]));
    log_info(tt::LogTest, "  fw_dbg_read_bad_rkey:  {} (P6)", d(dbg_t0[25], dbg_t1[25]));
    log_info(tt::LogTest, "  fw_dbg_read_bad_bnds:  {} (P6)", d(dbg_t0[26], dbg_t1[26]));
    log_info(tt::LogTest, "  fw_dbg_resp_word0:     0x{:08X} (P6: TX_BUF1 + 32, first payload u32)", dbg_t1[27]);
    log_info(
        tt::LogTest,
        "  fw_dbg_read_resp_tx:   {} (P6.1: READ_RESP frames fired on CMAC TXQ1)",
        d(dbg_t0[28], dbg_t1[28]));
    log_info(
        tt::LogTest,
        "  fw_dbg_ctrl_serviced:  T0={} T1={} (P7: CTRL doorbell REG/DEREG ops handled — T0=1 confirms pre-test "
        "register_mr fired)",
        dbg_t0[29],
        dbg_t1[29]);
    log_info(tt::LogTest, "  fw_dbg_op_send_imm:    {} (Q(b): SEND_IMM opcode 0x02 frames)", d(dbg_t0[30], dbg_t1[30]));
    log_info(
        tt::LogTest,
        "  fw_dbg_op_write_imm:   {} (Q(b): WRITE_IMM opcode 0x11 frames — payload to MR + 32B notify to host ring)",
        d(dbg_t0[31], dbg_t1[31]));
    // Phase R: ACK reception. dbg[1] increments on every opcode 0x40 frame
    // the FW dispatches. rcb[2] (acked_idx) is the highest seq# observed.
    log_info(tt::LogTest, "  fw_dbg_op_ack:         {} (Phase R: ACK frames serviced)", d(dbg_t0[1], dbg_t1[1]));
    {
        auto rcb_w = cluster.read_core(chip_id, virtual_core, 0x8400u, 32u);
        log_info(
            tt::LogTest,
            "  fw_rcb: producer={} consumer={} acked={} cq_head={} mode={}",
            rcb_w[0],
            rcb_w[1],
            rcb_w[2],
            rcb_w[4],
            rcb_w[7]);
    }

    // P6 sanity: read first 48 B of TX_BUF1 (= READ_RESP staging area).
    // Expect: [32-byte RDMA header with opcode 0x21] + [first 16 B of MR payload]
    {
        auto resp_dst = cluster.read_core(chip_id, virtual_core, 0x2A000u, 48);
        log_info(
            tt::LogTest,
            "  TX_BUF1 hdr u32[0..7]: {:08x} {:08x} {:08x} {:08x} | {:08x} {:08x} {:08x} {:08x}",
            resp_dst[0],
            resp_dst[1],
            resp_dst[2],
            resp_dst[3],
            resp_dst[4],
            resp_dst[5],
            resp_dst[6],
            resp_dst[7]);
        log_info(
            tt::LogTest,
            "  TX_BUF1 payload u32[0..3]: {:08x} {:08x} {:08x} {:08x}",
            resp_dst[8],
            resp_dst[9],
            resp_dst[10],
            resp_dst[11]);
    }

    // P5: peek the first 3 slots of the host RxWqeRing to confirm SENDs
    // landed in host hugepage. Slot 0 starts at hugepage + 128 KB.
    if (rx_ring_ok && sender.dma_pull_buffer() != nullptr) {
        const uint8_t* rx_ring_base = static_cast<const uint8_t*>(sender.dma_pull_buffer()) + (128u * 1024u);
        for (int slot = 0; slot < 3; ++slot) {
            const uint8_t* p = rx_ring_base + slot * 1536u;
            uint32_t hdr0 = *reinterpret_cast<const uint32_t*>(p + 0);
            uint32_t hdr_len = *reinterpret_cast<const uint32_t*>(p + 4);
            uint32_t hdr_seq = *reinterpret_cast<const uint32_t*>(p + 8);
            log_info(
                tt::LogTest,
                "  RxWqeRing slot {}: hdr[op,ver,tag]=0x{:08x} length={} seq={}",
                slot,
                hdr0,
                hdr_len,
                hdr_seq);
        }
    }

    int64_t cmac_total = d(cmac_t0.rxq0_pkt_end, cmac_t1.rxq0_pkt_end) + d(cmac_t0.rxq1_pkt_end, cmac_t1.rxq1_pkt_end);
    int64_t fw_seen = d(dbg_t0[0], dbg_t1[0]);
    if (cmac_total > 0) {
        double admit_rate = 100.0 * static_cast<double>(fw_seen) / static_cast<double>(cmac_total);
        log_info(tt::LogTest, "  fw_admit_pct:          {:.2f}% (fw_rx_frames / cmac_pkt_end_total)", admit_rate);
    }

    // Phase I diag — number of READ_REQs emitted, READ_RESPs landed/orphaned.
    log_info(
        tt::LogTest,
        "  fw_dbg_tx_fire_count:  T0={} T1={} (diag: CMAC fires in dma_pull pipeline)",
        dbg_t0[4],
        dbg_t1[4]);
    log_info(tt::LogTest, "  fw_dbg_last_flags:     0x{:04X} (diag: last flags_a seen at TX fire)", dbg_t1[3]);
    log_info(
        tt::LogTest,
        "  fw_dbg_read_req_tx:    T0={} T1={} (Phase I: initiator READ_REQs emitted)",
        dbg_t0[5],
        dbg_t1[5]);
    log_info(
        tt::LogTest, "  fw_dbg_read_resp_rx:   {} (Phase I: READ_RESPs landed in local MR)", d(dbg_t0[6], dbg_t1[6]));
    log_info(
        tt::LogTest,
        "  fw_dbg_read_resp_orph: {} (Phase I: READ_RESPs with no matching correlation entry)",
        d(dbg_t0[7], dbg_t1[7]));

    // Phase I verification — readback landing area and check the synthetic
    // pattern smoke_mlx emitted ((i ^ tag) & 0xFF).
    if (read_init_count > 0) {
        auto land = cluster.read_core(chip_id, virtual_core, kLandingL1Base, read_init_count * kReadInitLen);
        // Each READ used tag = seq & 0xFFFF. Reconstruct expected.
        // seqs go consecutively, and post_send_read uses next_seq_ which started
        // at... we don't have visibility. Just check that the first 16 bytes
        // of slot 0 are NON-ZERO (we pre-zeroed), as a smoke check.
        uint32_t nonzero = 0;
        for (size_t i = 0; i < std::min<size_t>(16, land.size()); ++i) {
            if (land[i] != 0) {
                nonzero++;
            }
        }
        log_info(
            tt::LogTest,
            "  Phase I landing L1@0x{:05X} first16w: {:08x} {:08x} {:08x} {:08x} (nonzero_words={}/16)",
            kLandingL1Base,
            land[0],
            land[1],
            land[2],
            land[3],
            nonzero);
    }

    tt::tt_metal::CloseDevice(device);
    return 0;
}
