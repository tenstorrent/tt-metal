// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Runs the wall-clock sync round trip on one Ethernet link with the Blackhole 1588 hardware sampled alongside
// (tools/profiler/sync/eth_ptp_sync_kernel.hpp), prints a summary and writes every stamp to a CSV.
//
//   test_eth_ptp_sync [--samples N] [--gap-us U] [--flags F] [--pair A B] [--csv PATH] [--force-aiclk D MHZ]
//   test_eth_ptp_sync --pair A B --peek ADDR[,ADDR...] [--poke ADDR VAL] [--peek-core X Y [--peek-idle]]
//   test_eth_ptp_sync --force-aiclk D MHZ --force-only        (leave D's AICLK forced; MHZ 0 releases it)
//
// --force-aiclk pins one chip's AICLK for the run (CMFW FORCE_AICLK, released at exit), so the two ERISCs run
// at different clocks and the software path's asymmetry can be measured against the hardware stamps.
//
// flags (PtpFlags in eth_ptp_sync_types.hpp); the default is PTP_FLAG_TCAM_LABEL alone: sync frames on TX
// queue 2 with their own header row and DA, one TCAM rule stamping only those frames, label-based selection,
// nothing else on the eth tile changed. The other bits are diagnostics and the older stamp-everything modes.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include <umd/device/arc/arc_messenger.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/blackhole_arc.hpp>
#include "impl/kernels/kernel.hpp"
#include "tools/profiler/sync/eth_ptp_sync_types.hpp"
#include "tools/profiler/sync/eth_wallclock_sync_solve.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::eth_sync;
using namespace tt::tt_metal::eth_ptp;

namespace {

uint64_t u64(uint32_t lo, uint32_t hi) { return (static_cast<uint64_t>(hi) << 32) | lo; }

struct Side {
    EthSyncResult hdr{};
    std::vector<EthSyncSample> sw;
    PtpResult ptp{};
    std::vector<PtpSample> hw;
};

Side read_side(
    tt::Cluster& cluster, int chip, const CoreCoord& core, uint32_t result_addr, uint32_t ptp_addr, uint32_t n) {
    Side s;
    cluster.read_core(&s.hdr, sizeof(s.hdr), tt_cxy_pair(chip, core), result_addr);
    cluster.read_core(&s.ptp, sizeof(s.ptp), tt_cxy_pair(chip, core), ptp_addr);
    const uint32_t got = std::min(n, std::min(s.hdr.n_samples, s.ptp.n_samples));
    s.sw.resize(got);
    s.hw.resize(got);
    if (got) {
        cluster.read_core(
            s.sw.data(), got * sizeof(EthSyncSample), tt_cxy_pair(chip, core), result_addr + sizeof(EthSyncResult));
        cluster.read_core(s.hw.data(), got * sizeof(PtpSample), tt_cxy_pair(chip, core), ptp_addr + sizeof(PtpResult));
    }
    return s;
}

void print_snapshot(const char* who, const PtpResult& r, double aiclk_ghz) {
    const double wall_s =
        static_cast<double>(u64(r.wall_end_lo, r.wall_end_hi) - u64(r.wall_start_lo, r.wall_start_hi)) /
        (aiclk_ghz * 1e9);
    const double cfr_ticks = static_cast<double>(u64(r.cfr_end_lo, r.cfr_end_hi) - u64(r.cfr_start_lo, r.cfr_start_hi));
    const double ptp_ns = static_cast<double>(u64(r.ptp_end_lo, r.ptp_end_hi) - u64(r.ptp_start_lo, r.ptp_start_hi));
    printf(
        "[ptp] %s: before: TIMER_CTRL=%u PTI_STAT=0x%06x TX_MAC_CFG=0x%04x RATE_SEL=0x%02x TH_STATUS=0x%08x | "
        "pti_ack=%u no_match_prev=0x%x override_prev=%u\n",
        who,
        r.timer_ctrl_before,
        r.pti_stat_before,
        r.tx_mac_cfg_before,
        r.rate_sel,
        r.th_status_before,
        r.pti_acked,
        r.no_match_prev,
        r.override_prev);
    printf(
        "[ptp] %s: TXQ0 CTRL=0x%x REMOTE_SEQ_TIMEOUT=0x%x LOCAL_SEQ_UPDATE_TIMEOUT=0x%x | TXQ1 CTRL=0x%x LSUT=0x%x | "
        "TXQ2 CTRL=0x%x LSUT=0x%x\n",
        who,
        r.pad[0],
        r.pad[1],
        r.pad[2],
        r.pad[3],
        r.pad[4],
        r.pad[5],
        r.pad[6]);
    printf(
        "[ptp] %s: over %.4f s of wall clock (aiclk %.3f GHz): CFR advanced %.0f ticks -> refclk %.4f MHz; PTP64NS "
        "advanced %.0f ns -> %.6f ns/ns\n",
        who,
        wall_s,
        aiclk_ghz,
        cfr_ticks,
        cfr_ticks / wall_s / 1e6,
        ptp_ns,
        ptp_ns / (wall_s * 1e9));
}

const char* status_name(uint32_t s) {
    switch (s) {
        case ETH_SYNC_IDLE: return "IDLE (kernel never ran)";
        case ETH_SYNC_RUNNING: return "RUNNING (did not finish)";
        case ETH_SYNC_DONE: return "DONE";
        case ETH_SYNC_TIMEOUT_HANDSHAKE: return "TIMEOUT_HANDSHAKE (peer never joined)";
        case ETH_SYNC_TIMEOUT_TXQ: return "TIMEOUT_TXQ";
        case ETH_SYNC_TIMEOUT_WAIT: return "TIMEOUT_WAIT";
        default: return "??";
    }
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t n_samples = 256, gap_us = 200, flags = PTP_FLAG_TCAM_LABEL;
    int pair_a = -1, pair_b = -1, force_chip = -1;
    uint32_t force_mhz = 0;
    std::string csv = "eth_ptp_sync.csv";
    std::vector<uint32_t> peek_addrs;
    uint32_t poke_addr = 0, poke_val = 0;
    bool peek_core_set = false, peek_idle = false, force_only = false;
    CoreCoord peek_core;
    for (int i = 1; i < argc; i++) {
        auto next = [&](uint32_t& v) {
            if (i + 1 < argc) {
                v = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
            }
        };
        if (!strcmp(argv[i], "--samples")) {
            next(n_samples);
        } else if (!strcmp(argv[i], "--gap-us")) {
            next(gap_us);
        } else if (!strcmp(argv[i], "--flags")) {
            next(flags);
        } else if (!strcmp(argv[i], "--pair") && i + 2 < argc) {
            pair_a = atoi(argv[++i]);
            pair_b = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--csv") && i + 1 < argc) {
            csv = argv[++i];
        } else if (!strcmp(argv[i], "--force-aiclk") && i + 2 < argc) {
            force_chip = atoi(argv[++i]);
            force_mhz = static_cast<uint32_t>(atoi(argv[++i]));
        } else if (!strcmp(argv[i], "--peek") && i + 1 < argc) {
            for (char* tok = strtok(argv[++i], ","); tok != nullptr; tok = strtok(nullptr, ",")) {
                peek_addrs.push_back(static_cast<uint32_t>(std::strtoul(tok, nullptr, 0)));
            }
        } else if (!strcmp(argv[i], "--poke") && i + 2 < argc) {
            poke_addr = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
            poke_val = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
        } else if (!strcmp(argv[i], "--peek-core") && i + 2 < argc) {
            const size_t px = static_cast<size_t>(atoi(argv[++i]));
            const size_t py = static_cast<size_t>(atoi(argv[++i]));
            peek_core = CoreCoord(px, py);
            peek_core_set = true;
        } else if (!strcmp(argv[i], "--peek-idle")) {
            peek_idle = true;
        } else if (!strcmp(argv[i], "--force-only")) {
            force_only = true;
        }
    }

    const auto shape = distributed::SystemMesh::instance().shape();
    auto mesh_device = distributed::MeshDevice::create(distributed::MeshDeviceConfig(shape));
    auto devices = mesh_device->get_devices();
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();

    IDevice* da = nullptr;
    IDevice* db = nullptr;
    CoreCoord ca, cb;
    for (IDevice* d : devices) {
        if (pair_a >= 0 && d->id() != pair_a) {
            continue;
        }
        for (const auto& [peer, cores] : cluster.get_ethernet_cores_grouped_by_connected_chips(d->id())) {
            if (pair_b >= 0 && static_cast<int>(peer) != pair_b) {
                continue;
            }
            for (IDevice* e : devices) {
                if (e->id() != peer) {
                    continue;
                }
                da = d;
                db = e;
                ca = cores.front();
                std::tie(std::ignore, cb) = d->get_connected_ethernet_core(ca);
                break;
            }
            if (da) {
                break;
            }
        }
        if (da) {
            break;
        }
    }
    if (!da) {
        fprintf(stderr, "no connected eth pair found\n");
        return 1;
    }
    // --peek/--poke: register access on the pair's two link cores (or --peek-core X Y on the first device,
    // --peek-idle for an idle eth core), then exit without running the sync.
    if (!peek_addrs.empty() || poke_addr != 0) {
        struct Target {
            IDevice* dev;
            CoreCoord core;
            bool idle;
        };
        std::vector<Target> targets;
        if (peek_core_set) {
            targets.push_back({da, peek_core, peek_idle});
        } else {
            targets.push_back({da, ca, false});
            targets.push_back({db, cb, false});
        }
        for (const Target& t : targets) {
            const auto ct = t.idle ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
            const uint32_t res = static_cast<uint32_t>(hal.get_dev_addr(ct, HalL1MemAddrType::UNRESERVED)) + 128;
            std::vector<uint32_t> rt = {static_cast<uint32_t>(peek_addrs.size()), poke_addr, poke_val};
            rt.insert(rt.end(), peek_addrs.begin(), peek_addrs.end());
            Program p = CreateProgram();
            EthernetConfig cfg{.noc = NOC::RISCV_0_default, .compile_args = {res}};
            if (t.idle) {
                cfg.eth_mode = Eth::IDLE;
                cfg.processor = DataMovementProcessor::RISCV_0;
            }
            auto kid = CreateKernel(p, "tt_metal/tools/profiler/sync/eth_reg_access.cpp", t.core, cfg);
            SetRuntimeArgs(p, kid, t.core, rt);
            const CoreCoord v = t.dev->virtual_core_from_logical_core(t.core, CoreType::ETH);
            const uint32_t zero = 0;
            cluster.write_core(&zero, sizeof(zero), tt_cxy_pair(t.dev->id(), v), res);
            tt::tt_metal::detail::CompileProgram(t.dev, p, /*force_slow_dispatch=*/true);
            tt::tt_metal::detail::WriteRuntimeArgsToDevice(t.dev, p, /*force_slow_dispatch=*/true);
            tt::tt_metal::detail::LaunchProgram(t.dev, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
            std::vector<uint32_t> out(1 + peek_addrs.size(), 0);
            for (int k = 0; k < 300 && out[0] != (0xD0E00000u | static_cast<uint32_t>(peek_addrs.size())); k++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                cluster.read_core(out.data(), out.size() * 4, tt_cxy_pair(t.dev->id(), v), res);
            }
            tt::tt_metal::detail::WaitProgramDone(t.dev, p, false);
            printf(
                "[peek] dev %d eth (%zu,%zu)%s%s\n",
                t.dev->id(),
                t.core.x,
                t.core.y,
                t.idle ? " [idle]" : "",
                out[0] == (0xD0E00000u | static_cast<uint32_t>(peek_addrs.size())) ? "" : "  (kernel did not finish)");
            if (poke_addr != 0) {
                printf("  wrote 0x%08x <- 0x%08x\n", poke_addr, poke_val);
            }
            for (size_t k = 0; k < peek_addrs.size(); k++) {
                printf("  0x%08x = 0x%08x (%u)\n", peek_addrs[k], out[1 + k], out[1 + k]);
            }
        }
        return 0;
    }
    printf(
        "[ptp] link: dev %d eth (%zu,%zu) -> dev %d eth (%zu,%zu), %u samples %u us apart, flags 0x%x\n",
        da->id(),
        ca.x,
        ca.y,
        db->id(),
        cb.x,
        cb.y,
        n_samples,
        gap_us,
        flags);

    const uint32_t base =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED));
    struct {
        uint32_t handshake, channel, result;
    } const lay{base, base + 64, base + 128};
    // Mo's result region is 144 B + 32 B per sample from base+128; ours is 128 B + 48 B per sample, then the raw
    // dumps at +0x20000/+0x21000. Everything must fit in the eth core's unreserved L1.
    const uint32_t ptp_addr = base + 0x18000;
    const uint32_t l1_size =
        static_cast<uint32_t>(hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED));
    TT_FATAL(n_samples <= 2048, "at most 2048 samples fit below the raw dump area");
    TT_FATAL(128u + 16u + 32u * n_samples <= 0x18000u, "software sample region would overrun the PTP region");
    TT_FATAL(
        0x18000u + 0x21000u + 0x1800u <= l1_size,
        "eth L1 unreserved region ({} B from {:#x}) too small",
        l1_size,
        base);
    const double ghz = cluster.get_device_aiclk(da->id()) / 1000.0;
    const uint32_t gap_cycles = static_cast<uint32_t>(gap_us * ghz * 1000.0);
    const uint64_t timeout_cycles =
        static_cast<uint64_t>(static_cast<double>(n_samples) * gap_cycles) + static_cast<uint64_t>(ghz * 2e9);

    std::vector<uint32_t> args = {
        n_samples,
        static_cast<uint32_t>(timeout_cycles),
        static_cast<uint32_t>(timeout_cycles >> 32),
        lay.result,
        lay.channel,
        lay.handshake,
        gap_cycles,
        ptp_addr,
        flags};
    Program p_snd = CreateProgram();
    Program p_rcv = CreateProgram();
    CreateKernel(
        p_snd,
        "tt_metal/tools/profiler/sync/eth_ptp_sync_sender.cpp",
        ca,
        EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = args});
    CreateKernel(
        p_rcv,
        "tt_metal/tools/profiler/sync/eth_ptp_sync_receiver.cpp",
        cb,
        EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = args});
    tt::tt_metal::detail::CompileProgram(db, p_rcv);
    tt::tt_metal::detail::CompileProgram(da, p_snd);
    {
        std::vector<uint32_t> zeros(0x1800 / 4, 0);
        const CoreCoord va0 = da->virtual_core_from_logical_core(ca, CoreType::ETH);
        const CoreCoord vb0 = db->virtual_core_from_logical_core(cb, CoreType::ETH);
        cluster.write_core(zeros.data(), zeros.size() * 4, tt_cxy_pair(da->id(), va0), ptp_addr + 0x20000);
        cluster.write_core(zeros.data(), zeros.size() * 4, tt_cxy_pair(db->id(), vb0), ptp_addr + 0x20000);
    }
    const auto force_aiclk = [&](int chip, uint32_t mhz) {
        const uint32_t rc = cluster.get_driver()->get_tt_device(chip)->get_arc_messenger()->send_message(
            static_cast<uint32_t>(tt::umd::blackhole::ArcMessageType::FORCE_AICLK), {mhz});
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        printf("[ptp] FORCE_AICLK dev %d -> %u MHz (0 = release): arc rc %u\n", chip, mhz, rc);
    };
    if (force_chip >= 0) {
        force_aiclk(force_chip, force_mhz);
    }
    if (force_only) {
        return 0;  // --force-only: leave the clock forced (0 releases) for a run of something else
    }
    printf(
        "[ptp] aiclk telemetry before launch: dev %d %d MHz, dev %d %d MHz\n",
        da->id(),
        cluster.get_device_aiclk(da->id()),
        db->id(),
        cluster.get_device_aiclk(db->id()));
    tt::tt_metal::detail::LaunchProgram(db, p_rcv, false, true);
    tt::tt_metal::detail::LaunchProgram(da, p_snd, false, true);

    const CoreCoord va = da->virtual_core_from_logical_core(ca, CoreType::ETH);
    const CoreCoord vb = db->virtual_core_from_logical_core(cb, CoreType::ETH);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
    while (std::chrono::steady_clock::now() < deadline) {
        EthSyncResult a{}, b{};
        cluster.read_core(&a, sizeof(a), tt_cxy_pair(da->id(), va), lay.result);
        cluster.read_core(&b, sizeof(b), tt_cxy_pair(db->id(), vb), lay.result);
        if (a.status >= ETH_SYNC_DONE && b.status >= ETH_SYNC_DONE) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    Side A = read_side(cluster, da->id(), va, lay.result, ptp_addr, n_samples);
    Side B = read_side(cluster, db->id(), vb, lay.result, ptp_addr, n_samples);
    tt::tt_metal::detail::WaitProgramDone(da, p_snd, false);
    tt::tt_metal::detail::WaitProgramDone(db, p_rcv, false);

    printf(
        "[ptp] status: sender %s | receiver %s | sw samples %u/%u | hw samples %u/%u\n",
        status_name(A.hdr.status),
        status_name(B.hdr.status),
        A.hdr.n_samples,
        B.hdr.n_samples,
        A.ptp.n_samples,
        B.ptp.n_samples);
    print_snapshot("sender  ", A.ptp, ghz);
    print_snapshot("receiver", B.ptp, cluster.get_device_aiclk(db->id()) / 1000.0);
    if (flags & PTP_FLAG_RAW_DUMP) {
        auto dump = [&](const char* who, int chip, const CoreCoord& core) {
            std::vector<uint32_t> mac(4 * 48), rx(6 * 48);
            cluster.read_core(mac.data(), mac.size() * 4, tt_cxy_pair(chip, core), ptp_addr + 0x20000);
            cluster.read_core(rx.data(), rx.size() * 4, tt_cxy_pair(chip, core), ptp_addr + 0x21000);
            printf("[ptp] %s raw MAC FIFO pops (w0 w1 w2 w3), rounds 0-1:\n", who);
            for (int k = 0; k < 48 && (mac[4 * k] | mac[4 * k + 1] | mac[4 * k + 2] | mac[4 * k + 3]); k++) {
                printf("   %2d: %08x %08x %08x %08x\n", k, mac[4 * k], mac[4 * k + 1], mac[4 * k + 2], mac[4 * k + 3]);
            }
            printf(
                "[ptp] %s raw RX FIFO pops (status_before lo hi label status_after_pop1 status_after_pop0), rounds "
                "0-1:\n",
                who);
            for (int k = 0; k < 48 && (rx[6 * k] | rx[6 * k + 1] | rx[6 * k + 2] | rx[6 * k + 3]); k++) {
                printf(
                    "   %2d: %08x %08x %08x %08x %08x %08x\n",
                    k,
                    rx[6 * k],
                    rx[6 * k + 1],
                    rx[6 * k + 2],
                    rx[6 * k + 3],
                    rx[6 * k + 4],
                    rx[6 * k + 5]);
            }
        };
        dump("sender  ", da->id(), va);
        dump("receiver", db->id(), vb);
    }

    const size_t n = std::min(A.hw.size(), B.hw.size());
    std::ofstream out(csv);
    out << "i,t0_wall,t2_wall,t1_wall,p0,p2,p1,p1b,snd_mac_tx,snd_mac_tag,snd_th_rx,snd_th_label,snd_diag,rcv_th_rx,"
           "rcv_th_label,rcv_mac_tx,rcv_mac_tag,rcv_diag\n";
    uint32_t tag_ok_a = 0, tag_ok_b = 0, rx_ok_a = 0, rx_ok_b = 0;
    std::vector<double> theta, rtt_wire, issue_a, observe_a, issue_b, observe_b;
    for (size_t i = 0; i < n; i++) {
        const auto& a = A.hw[i];
        const auto& b = B.hw[i];
        // FIFO words come out low-half first (run 1), so w2 is the timestamp's low word and w3 its high word.
        const uint64_t t0h = u64(a.mac_tx_lo, a.mac_tx_hi), t2h = u64(a.th_rx_lo, a.th_rx_hi);
        const uint64_t t1h = u64(b.th_rx_lo, b.th_rx_hi), t1bh = u64(b.mac_tx_lo, b.mac_tx_hi);
        const uint64_t p0 = u64(a.ptp_a_lo, a.ptp_a_hi), p2 = u64(a.ptp_b_lo, a.ptp_b_hi);
        const uint64_t p1 = u64(b.ptp_a_lo, b.ptp_a_hi), p1b = u64(b.ptp_b_lo, b.ptp_b_hi);
        out << i << ',' << u64(A.sw[i].t0_lo, A.sw[i].t0_hi) << ',' << u64(A.sw[i].t2_lo, A.sw[i].t2_hi) << ','
            << u64(B.sw[i].t1_lo, B.sw[i].t1_hi) << ',' << p0 << ',' << p2 << ',' << p1 << ',' << p1b << ',' << t0h
            << ',' << u64(a.mac_tag_lo, a.mac_tag_hi) << ',' << t2h << ',' << a.th_label << ',' << a.diag << ',' << t1h
            << ',' << b.th_label << ',' << t1bh << ',' << u64(b.mac_tag_lo, b.mac_tag_hi) << ',' << b.diag << '\n';
        const bool ok_a = (a.diag & (1u << 16)) && (a.diag & (1u << 17));
        const bool ok_b = (b.diag & (1u << 16)) && (b.diag & (1u << 17));
        tag_ok_a += (a.diag >> 16) & 1;
        rx_ok_a += (a.diag >> 17) & 1;
        tag_ok_b += (b.diag >> 16) & 1;
        rx_ok_b += (b.diag >> 17) & 1;
        if (ok_a && ok_b) {
            theta.push_back(
                (static_cast<double>(static_cast<int64_t>(t1h - t0h)) -
                 static_cast<double>(static_cast<int64_t>(t2h - t1bh))) /
                2.0);
            rtt_wire.push_back(
                static_cast<double>(static_cast<int64_t>(t2h - t0h)) -
                static_cast<double>(static_cast<int64_t>(t1bh - t1h)));
            issue_a.push_back(static_cast<double>(static_cast<int64_t>(t0h - p0)));
            observe_a.push_back(static_cast<double>(static_cast<int64_t>(p2 - t2h)));
            observe_b.push_back(static_cast<double>(static_cast<int64_t>(p1 - t1h)));
            issue_b.push_back(static_cast<double>(static_cast<int64_t>(t1bh - p1b)));
        }
    }
    printf(
        "[ptp] hw stamps found: sender mac %u/%zu rx %u/%zu | receiver mac %u/%zu rx %u/%zu | complete rounds %zu\n",
        tag_ok_a,
        n,
        rx_ok_a,
        n,
        tag_ok_b,
        n,
        rx_ok_b,
        n,
        theta.size());
    auto stats = [](const char* name, std::vector<double> v) {
        if (v.empty()) {
            printf("[ptp] %-34s n=0\n", name);
            return;
        }
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        double mean = 0;
        for (double x : v) {
            mean += x;
        }
        mean /= v.size();
        double var = 0;
        for (double x : v) {
            var += (x - mean) * (x - mean);
        }
        var /= v.size();
        printf(
            "[ptp] %-34s n=%zu min %.0f med %.0f max %.0f mean %.1f rms %.1f ns\n",
            name,
            v.size(),
            s.front(),
            s[s.size() / 2],
            s.back(),
            mean,
            std::sqrt(var));
    };
    stats("hw offset theta (rcv - snd)", theta);
    stats("hw wire rtt", rtt_wire);
    stats("sender issue->MAC egress", issue_a);
    stats("sender MAC ingress->observe", observe_a);
    stats("receiver MAC ingress->observe", observe_b);
    stats("receiver issue->MAC egress", issue_b);
    if (n >= 4) {
        auto trips = build_trips(A.sw, B.sw, n);
        auto sol = solve(trips);
        printf(
            "[ptp] sw solve: valid=%d offset %lld cyc rate %.9f rtt_min %llu cyc spread %lld residual_rms %.2f\n",
            (int)sol.valid,
            (long long)sol.offset,
            sol.rate,
            (unsigned long long)sol.rtt_min,
            (long long)sol.offset_spread,
            sol.residual_rms);
    }
    printf("[ptp] csv: %s\n", csv.c_str());
    if (force_chip >= 0) {
        force_aiclk(force_chip, 0);
        printf(
            "[ptp] aiclk telemetry after restore: dev %d %d MHz, dev %d %d MHz\n",
            da->id(),
            cluster.get_device_aiclk(da->id()),
            db->id(),
            cluster.get_device_aiclk(db->id()));
    }
    mesh_device->close();
    return 0;
}
