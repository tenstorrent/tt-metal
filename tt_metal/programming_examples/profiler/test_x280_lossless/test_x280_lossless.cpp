// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// X280 per-lane LOSSLESS harness. Proves the compact 8 B per-lane packet format + the acked per-lane
// ring D2H are lossless end to end under high-volume marker bursts. The ONLY fiction vs the real design
// is the producer: a synthetic per-RISC burst generator (finite loop) that emits far faster than a real
// kernel, to stress the pipeline. Everything else is the real design:
//   worker L1 SPSC ring  --(X280 reader==relay, per-lane, flow-controlled)-->  per-lane HOST ring
//     --(host consumer thread drains, advances `acked`)-->  reconstruct + verify.
// Flow control (the acked FIFO): the X280 blocks a lane's drain when its host ring is full
// (sent - acked == HRING_WORDS) -> back-pressure to the worker producer; a non-posted `sent` publish
// barriers the posted ring data so the host only reads what has landed. Fully lossless by construction.
//
// Build:  make -C tools/x280_bm build/lim_idle.bin build/profll.bin
//         cmake --build build_Release --target test_x280_lossless
// Run:    ./build_Release/programming_examples/test_x280_lossless --nmarkers 2000 --nharts 2

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <impl/context/metal_context.hpp>
#include <llrt/tt_cluster.hpp>
#include <tools/profiler/x280_driver.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "prof_packet.h"

using tt::Cluster;
using tt::CoreType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::MetalContext;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::profiler::X280Driver;

// --- LIM param map (MUST match tools/x280_bm/src/profll.c) ---
static constexpr uint64_t MBOX_PARAMS = 0x08011000ULL;
static constexpr uint64_t MBOX_COORDS = 0x08011200ULL;
static constexpr uint64_t P_STOP = MBOX_PARAMS + 0x28;
static constexpr uint64_t HSENT_BASE =
    0x08018000ULL;  // X280's authoritative per-lane `sent` (LIM); host seqlock-reads it
static constexpr uint64_t HACKED_BASE = 0x08017000ULL;  // host writes here (per-lane drained-words ack) (LIM)
static uint64_t harthb(int h) { return 0x08011040ULL + 0x100 + (uint64_t)h * 8; }

static constexpr uint64_t WIN_STRIDE = 0x200000ULL;  // 2 MiB host window
static constexpr int NRISC = 5;
static constexpr uint32_t HRING_WORDS = 512;  // per-lane host ring depth (MUST match profll.c)

template <typename T>
static void pack(std::vector<uint8_t>& buf, size_t off, T val) {
    std::memcpy(buf.data() + off, &val, sizeof(T));
}
static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open firmware: " + path);
    }
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (d.size() % 4) {
        d.push_back(0);
    }
    return d;
}

int main(int argc, char** argv) {
    using namespace tt::tt_metal;
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string idle_bin = "tools/x280_bm/build/lim_idle.bin";
    std::string active_bin = "tools/x280_bm/build/profll.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    uint64_t nmarkers = 2000, nharts = 2, ts_step = 0x1000000ull;
    uint32_t prog_id = 0xA5A5A5A5u;
    bool do_reset = false;
    uint32_t active_riscs =
        NRISC;  // producers per core; --onelane=1 (BRISC), --twolane=2 (+NCRISC) to isolate concurrency
    int cx0 = -1, cy0 = -1, cx1 = -1, cy1 = -1;
    uint64_t read_noc = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--idle-bin") {
            idle_bin = next();
        } else if (a == "--active-bin") {
            active_bin = next();
        } else if (a == "--device") {
            device_id = std::stoi(next());
        } else if (a == "--l2cpu") {
            l2cpu = std::stoi(next());
        } else if (a == "--pll") {
            pll = std::stoi(next());
        } else if (a == "--nmarkers") {
            nmarkers = std::stoull(next());
        } else if (a == "--nharts") {
            nharts = std::stoull(next());
        } else if (a == "--tsstep") {
            ts_step = std::stoull(next(), nullptr, 0);
        } else if (a == "--cx0") {
            cx0 = std::stoi(next());
        } else if (a == "--cy0") {
            cy0 = std::stoi(next());
        } else if (a == "--cx1") {
            cx1 = std::stoi(next());
        } else if (a == "--cy1") {
            cy1 = std::stoi(next());
        } else if (a == "--noc") {
            read_noc = std::stoull(next());
        } else if (a == "--reset") {
            do_reset = true;
        } else if (a == "--onelane") {
            active_riscs = 1;
        } else if (a == "--twolane") {
            active_riscs = 2;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    if (do_reset) {
        std::string cmd = "tt-smi -r " + std::to_string(device_id);
        printf("[boot] %s\n", cmd.c_str());
        if (std::system(cmd.c_str()) != 0) {
            fprintf(stderr, "tt-smi reset failed\n");
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    std::vector<uint8_t> idle = read_file(idle_bin), active = read_file(active_bin);
    printf(
        "[fw] idle %s (%zu B), active %s (%zu B)\n", idle_bin.c_str(), idle.size(), active_bin.c_str(), active.size());

    auto mesh = MeshDevice::create_unit_mesh(device_id);
    Cluster& cluster = MetalContext::instance().get_cluster();
    auto& hal = MetalContext::instance().hal();
    CoreCoord grid = mesh->compute_with_storage_grid_size();
    uint32_t gx = (uint32_t)grid.x, gy = (uint32_t)grid.y;
    if (cx0 < 0) {
        cx0 = 0;
        cy0 = 0;
        cx1 = (int)gx - 1;
        cy1 = (int)gy - 1;
    }
    uint32_t rgx = (uint32_t)(cx1 - cx0 + 1), num_cores = rgx * (uint32_t)(cy1 - cy0 + 1);
    uint32_t NL = num_cores * NRISC;  // number of lanes
    uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    printf(
        "[grid] full %ux%u; subrange (%d,%d)-(%d,%d) = %u cores (%u lanes); NoC%llu; %llu harts, "
        "%llu markers/lane, host ring %u words\n",
        gx,
        gy,
        cx0,
        cy0,
        cx1,
        cy1,
        num_cores,
        NL,
        (unsigned long long)read_noc,
        (unsigned long long)nharts,
        (unsigned long long)nmarkers,
        HRING_WORDS);

    // VIRTUAL coords for host UMD access; TRANSLATED coords for the X280 read-window table.
    std::vector<CoreCoord> vc(num_cores), logs(num_cores);
    std::vector<uint8_t> coords(num_cores * 8, 0);
    const auto& soc = cluster.get_soc_desc(device_id);
    uint32_t idx = 0;
    for (int ly = cy0; ly <= cy1; ly++) {
        for (int lx = cx0; lx <= cx1; lx++) {
            logs[idx] = CoreCoord{(uint32_t)lx, (uint32_t)ly};
            vc[idx] = cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logs[idx], CoreType::WORKER);
            tt::umd::CoreCoord tr = soc.translate_coord_to(
                {logs[idx], tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL}, tt::CoordSystem::TRANSLATED);
            pack<uint32_t>(coords, idx * 8 + 0, (uint32_t)tr.x);
            pack<uint32_t>(coords, idx * 8 + 4, (uint32_t)tr.y);
            idx++;
        }
    }

    // Zero the worker profiler control (head/tail start at 0) BEFORE boot.
    std::vector<uint8_t> zctrl(32 * 4, 0);
    for (uint32_t c = 0; c < num_cores; c++) {
        cluster.write_core(zctrl.data(), (uint32_t)zctrl.size(), tt_cxy_pair(device_id, vc[c]), prof_l1);
    }

    // Host D2H layout: per-lane rings [0 .. rings_bytes), then per-lane `sent` publish [rings_bytes ..).
    auto pcie_cores = cluster.get_soc_desc(device_id).get_cores(tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED);
    CoreCoord pc = pcie_cores.front();
    uint64_t pcie_enc = ((uint64_t)pc.x & 0x3f) | (((uint64_t)pc.y & 0x3f) << 6);
    uint64_t pcie_base = cluster.get_pcie_base_addr_from_device(device_id);
    uint64_t chan_sz = cluster.get_host_channel_size(device_id, 0);
    uint64_t data_off = (chan_sz / 2) & ~(WIN_STRIDE - 1);
    uint64_t host_base = pcie_base + data_off;
    uint64_t rings_bytes = (uint64_t)NL * HRING_WORDS * 4;
    (void)0;  // (host-mem `sent` publish removed; consumer reads authoritative HSENT from LIM)
    if (rings_bytes + NL * 4 > WIN_STRIDE) {
        fprintf(
            stderr,
            "[d2h] rings(%.2f MB)+sent exceed the 2 MB window; lower cores or HRING_WORDS\n",
            rings_bytes / 1e6);
        std::_Exit(2);
    }
    // Zero the rings + sent region in host memory.
    {
        std::vector<uint8_t> z(rings_bytes + NL * 4, 0);
        cluster.write_sysmem(z.data(), (uint32_t)z.size(), data_off, device_id, 0);
    }

    // ---- boot the X280 via the baked-in idle->active handoff ----
    X280Driver drv(cluster, device_id, l2cpu);
    bool half_broken = false;
    if (!drv.ensure_idle(idle, pll, std::chrono::milliseconds(3000), half_broken)) {
        fprintf(stderr, "[boot] idle FW not up (half_broken=%d) -- needs `tt-smi -r %d`\n", half_broken, device_id);
        std::_Exit(1);
    }
    {  // zero the per-lane HACKED ack pointers in LIM (host is the writer)
        std::vector<uint8_t> z(NL * 4, 0);
        drv.write_block(z.data(), (uint32_t)z.size(), HACKED_BASE);
    }
    drv.write_block(coords.data(), (uint32_t)coords.size(), MBOX_COORDS);
    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, pcie_enc);
    pack<uint64_t>(params, 0x08, host_base);
    pack<uint64_t>(params, 0x10, prof_l1);
    pack<uint64_t>(params, 0x18, (uint64_t)num_cores);
    pack<uint64_t>(params, 0x20, (uint64_t)HRING_WORDS);
    pack<uint64_t>(params, 0x28, 0);  // P_STOP
    pack<uint64_t>(params, 0x30, read_noc);
    pack<uint64_t>(params, 0x38, nharts);
    drv.write_block(params.data(), (uint32_t)params.size(), MBOX_PARAMS);
    for (uint64_t h = 0; h < nharts; h++) {
        drv.lim_wr_u64(harthb((int)h), 0);
    }
    if (!drv.handoff_to_active_fw(active, std::chrono::milliseconds(3000))) {
        fprintf(stderr, "[boot] active FW (profll) did not stamp RUNNING\n");
        std::_Exit(1);
    }
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        bool all = false;
        while (std::chrono::steady_clock::now() < deadline) {
            all = true;
            for (uint64_t h = 0; h < nharts; h++) {
                if (drv.lim_rd_u64(harthb((int)h)) != 3) {
                    all = false;
                }
            }
            if (all) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (!all) {
            fprintf(
                stderr,
                "[boot] not all %llu harts reached the work loop -- flaky boot, `tt-smi -r`\n",
                (unsigned long long)nharts);
            std::_Exit(1);
        }
    }
    printf("[boot] idle up, profll RUNNING, %llu harts draining\n", (unsigned long long)nharts);

    // ---- host consumer thread: drain per-lane rings, accumulate, advance `acked` ----
    std::atomic<bool> producers_done{false};
    std::vector<uint32_t> local_acked(NL, 0);
    std::vector<std::vector<uint32_t>> accum(NL);
    std::atomic<uint64_t> overflow{0};
    std::thread consumer([&]() {
        std::vector<uint32_t> lane_ring(HRING_WORDS);  // one lane's ring, read per-lane (not the whole buffer)
        int empty = 0;
        uint64_t words = 0;  // total words accumulated (progress / re-drain detector)
        auto start = std::chrono::steady_clock::now();
        auto next_log = start + std::chrono::seconds(1);
        for (;;) {
            auto now = std::chrono::steady_clock::now();
            if (now > next_log) {
                std::string per;
                for (uint32_t L = 0; L < NL && L < 8; L++) {
                    per += " " + std::to_string(local_acked[L]);
                }
                printf(
                    "  [consumer] words=%llu done=%d empty=%d  acked:%s\n",
                    (unsigned long long)words,
                    (int)producers_done.load(),
                    empty,
                    per.c_str());
                next_log = now + std::chrono::seconds(1);
            }
            if (now - start > std::chrono::seconds(30)) {
                printf("  [consumer] WALL TIMEOUT at %llu words -- bailing\n", (unsigned long long)words);
                break;
            }
            // Service each lane as its own socket: read ITS counter, read ITS ring slice, copy out, ack IT.
            // Small targeted reads of the stable [acked,sent) region avoid the whole-buffer read catching
            // the X280's concurrent frontier writes across other lanes.
            bool any = false;
            for (uint32_t L = 0; L < NL; L++) {
                uint32_t la = local_acked[L];
                uint32_t hs;
                drv.read_block(reinterpret_cast<uint8_t*>(&hs), 4, HSENT_BASE + L * 4);  // this lane's counter
                if (hs == la) {
                    continue;  // no new data on this lane
                }
                any = true;
                if ((int32_t)(hs - la) < 0) {
                    continue;  // impossible (sent<acked) -> ignore
                }
                uint32_t avail = hs - la;
                if (avail > HRING_WORDS) {  // X280 overran this ring (shouldn't happen) -> accept the loss
                    overflow.fetch_add(1);
                    local_acked[L] = hs;
                    continue;
                }
                // read only THIS lane's ring region, then re-read the counter (seqlock): if it moved while we
                // read the slice, retry next poll (when the ring fills the X280 blocks -> counter stabilizes).
                cluster.read_sysmem(
                    reinterpret_cast<uint8_t*>(lane_ring.data()),
                    HRING_WORDS * 4,
                    data_off + (uint64_t)L * HRING_WORDS * 4,
                    device_id,
                    0);
                uint32_t hs2;
                drv.read_block(reinterpret_cast<uint8_t*>(&hs2), 4, HSENT_BASE + L * 4);
                if (hs2 != hs) {
                    continue;  // lane advanced during our slice read -> snapshot untrustworthy, retry
                }
                for (uint32_t w = 0; w < avail; w++) {
                    accum[L].push_back(lane_ring[(la + w) % HRING_WORDS]);
                }
                local_acked[L] = hs;
                words += avail;
            }
            if (!any) {
                if (producers_done.load() && ++empty >= 100) {
                    break;  // producers finished AND no new data for ~20 ms -> everything drained
                }
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            empty = 0;
            drv.write_block(reinterpret_cast<uint8_t*>(local_acked.data()), NL * 4, HACKED_BASE);  // batched ack
        }
    });

    // ---- fast-dispatch the producers (finite burst); blocking enqueue returns when they finish ----
    auto mk_defs = [&](int proc_idx, bool with_proc) {
        std::map<std::string, std::string> d;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "0x%llxULL", (unsigned long long)prof_l1);
        d["PROF_L1"] = buf;
        d["N_MARKERS"] = std::to_string(nmarkers) + "u";
        std::snprintf(buf, sizeof(buf), "0x%llxull", (unsigned long long)ts_step);
        d["TS_STEP"] = buf;
        std::snprintf(buf, sizeof(buf), "0x%xu", prog_id);
        d["PROG_ID"] = buf;
        if (with_proc) {
            d["PROC_IDX"] = std::to_string(proc_idx);
        }
        return d;
    };
    const std::string kdir = "tt_metal/programming_examples/profiler/test_x280_lossless/kernels/";
    Program program = CreateProgram();
    CoreRange all_cores(CoreCoord{(uint32_t)cx0, (uint32_t)cy0}, CoreCoord{(uint32_t)cx1, (uint32_t)cy1});
    auto d0 = mk_defs(0, true), d1 = mk_defs(1, true), dc = mk_defs(0, false);
    CreateKernel(
        program,
        kdir + "producer_dm.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = d0});
    if (active_riscs >= 2) {  // NCRISC (r=1)
        CreateKernel(
            program,
            kdir + "producer_dm.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = d1});
    }
    if (active_riscs >= NRISC) {  // 3 TRISCs (r=2,3,4) as one compute kernel
        CreateKernel(program, kdir + "producer_compute.cpp", all_cores, ComputeConfig{.defines = dc});
    }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    printf("[run] dispatching producers (%llu markers/lane on %u lanes)...\n", (unsigned long long)nmarkers, NL);
    distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), wl, /*blocking=*/true);
    printf("[run] producers done; draining remainder\n");
    {  // diagnostic: what head/tail did the producers actually leave in worker L1?
        std::vector<uint8_t> wc(32 * 4);
        for (uint32_t c = 0; c < num_cores && c < 3; c++) {
            cluster.read_core(wc.data(), 128, tt_cxy_pair(device_id, vc[c]), prof_l1);
            const uint32_t* cw = reinterpret_cast<const uint32_t*>(wc.data());
            printf(
                "  [wctrl] core %u  head[0..4]=%u,%u,%u,%u,%u  tail[0..4]=%u,%u,%u,%u,%u\n",
                c,
                cw[0],
                cw[1],
                cw[2],
                cw[3],
                cw[4],
                cw[5],
                cw[6],
                cw[7],
                cw[8],
                cw[9]);
        }
    }

    // Signal P_STOP + producers_done; the consumer keeps acking so the X280 drains the rest, then the
    // consumer exits on its own (no L2CPU access from this thread while it runs). Only after join do we
    // touch the L2CPU again (wait_active_fw_returned), so there's no concurrent-UMD race.
    drv.lim_wr_u64(P_STOP, 1);
    producers_done.store(true);
    consumer.join();
    if (!drv.wait_active_fw_returned(std::chrono::seconds(5))) {
        fprintf(stderr, "[run] drainer did not return to idle after drain (unexpected)\n");
    }

    // ---- verify each lane's accumulated stream is complete + gap-free ----
    uint64_t ok_lanes = 0, total_markers = 0, total_stickies = 0, seq_gaps = 0, ts_bad = 0, prog_bad = 0,
             short_lanes = 0;
    const uint64_t TS_MASK44 = (1ull << 44) - 1;
    std::vector<std::string> bad;
    uint32_t active_lanes = num_cores * active_riscs;  // only r < active_riscs produce
    for (uint32_t L = 0; L < NL; L++) {
        uint32_t r = L % NRISC;
        if (r >= active_riscs) {
            continue;  // inactive lane
        }
        const auto& a = accum[L];
        bool lane_ok = true;
        uint32_t cur_hi = 0, cur_prog = 0;
        bool have_ctx = false;
        uint64_t exp_seq = 0, exp_ts = (((uint64_t)(r + 1) << 8)) & TS_MASK44, mkrs = 0;
        for (size_t p = 0; p + 1 < a.size(); p += 2) {
            uint32_t w0 = a[p], w1 = a[p + 1];
            if (pp_is_sticky(w0)) {
                cur_hi = pp_low27(w0) & 0xFFFu;
                cur_prog = pp_payload32(w1);
                have_ctx = true;
                total_stickies++;
                continue;
            }
            total_markers++;
            mkrs++;
            uint32_t seq = pp_low27(w0);
            uint64_t full = pp_full_ts(cur_hi, pp_payload32(w1)) & TS_MASK44;
            if (!have_ctx || cur_prog != prog_id) {
                prog_bad++;
                lane_ok = false;
            }
            if (seq != (exp_seq & 0x7FFFFFF)) {
                seq_gaps++;
                lane_ok = false;
            }
            if (full != exp_ts) {
                ts_bad++;
                lane_ok = false;
            }
            exp_seq++;
            exp_ts = (exp_ts + ts_step) & TS_MASK44;
        }
        if (mkrs != nmarkers) {  // lossless => every produced marker arrived
            lane_ok = false;
            short_lanes++;
        }
        if (lane_ok) {
            ok_lanes++;
        } else if (bad.size() < 12) {
            char b[128];
            std::snprintf(
                b,
                sizeof(b),
                "  BAD lane L=%u c=%u r=%u  markers=%llu/%llu",
                L,
                L / NRISC,
                r,
                (unsigned long long)mkrs,
                (unsigned long long)nmarkers);
            bad.emplace_back(b);
        }
    }
    for (auto& s : bad) {
        printf("%s\n", s.c_str());
    }

    printf("\n=== X280 per-lane lossless (acked ring D2H) [%u risc/core] ===\n", active_riscs);
    printf(
        "  lanes            : %llu ok / %u total%s\n",
        (unsigned long long)ok_lanes,
        active_lanes,
        (ok_lanes == active_lanes) ? "  ALL LOSSLESS" : "  *** LOSS ***");
    printf(
        "  markers/stickies : %llu / %llu (expected %llu markers)\n",
        (unsigned long long)total_markers,
        (unsigned long long)total_stickies,
        (unsigned long long)((uint64_t)active_lanes * nmarkers));
    printf(
        "  seq gaps         : %llu   short lanes: %llu\n",
        (unsigned long long)seq_gaps,
        (unsigned long long)short_lanes);
    printf(
        "  timestamp bad    : %llu   prog_id bad: %llu   ring overflow: %llu\n",
        (unsigned long long)ts_bad,
        (unsigned long long)prog_bad,
        (unsigned long long)overflow.load());
    std::fflush(stdout);
    std::_Exit((ok_lanes == active_lanes) ? 0 : 1);
}
