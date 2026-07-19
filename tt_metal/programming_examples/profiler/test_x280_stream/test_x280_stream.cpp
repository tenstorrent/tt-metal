// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// X280 LINEARIZED profiler pipeline test: 2 readers + 1 relay + SINGLE host ring, identity via
// STICKY-SRC. Producers are unchanged (per-RISC L1 SPSC, producer_common.h). The X280 readers drain the
// worker rings into per-reader LIM SPSCs, injecting a precomputed (core,risc) STICKY-SRC before each
// source's data; the relay round-robins the readers into ONE host ring (one sent/acked pair -> coherent).
// The host drains that one stream and demuxes by sticky: a STICKY-SRC sets the current (core,risc); every
// marker/meta after it binds to that lane until the next STICKY-SRC. Verifies each demuxed lane is
// gap-free -- proving the linearized + sticky design is lossless.
//
// Build:  make -C tools/x280_bm build/lim_idle.bin build/profstream.bin
//         cmake --build build_Release --target test_x280_stream
// Run:    ./build_Release/programming_examples/test_x280_stream --reset --nmarkers 300 --nread 2

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

// --- LIM map (MUST match tools/x280_bm/src/profstream.c) ---
static constexpr uint64_t MBOX_PARAMS = 0x08011000ULL;
static constexpr uint64_t MBOX_COORDS = 0x08011200ULL;
static constexpr uint64_t SRCLUT_BASE = 0x08014000ULL;
static constexpr uint64_t HSENT_ADDR = 0x08017000ULL;   // relay writes, host reads
static constexpr uint64_t HACKED_ADDR = 0x08017040ULL;  // host writes, relay reads
static constexpr uint64_t P_STOP = MBOX_PARAMS + 0x28;
static uint64_t harthb(int h) { return 0x08011040ULL + 0x100 + (uint64_t)h * 8; }

static constexpr uint64_t WIN_STRIDE = 0x200000ULL;
static constexpr int NRISC = 5;

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
    int device_id = 0, l2cpu = 0, pll = 1000;
    uint64_t nmarkers = 2000, nread = 2, ts_step = 0x1000000ull;
    uint32_t prog_id = 0xA5A5A5A5u, hring_words = 8192, prod_delay = 0;
    bool do_reset = false, direct = false;  // --direct: single-hart direct drain (no reader/relay split)
    uint32_t active_riscs = NRISC;
    int cx0 = -1, cy0 = -1, cx1 = -1, cy1 = -1;
    uint64_t read_noc = 0;

    std::string idle = "tools/x280_bm/build/lim_idle.bin", active = "tools/x280_bm/build/profstream.bin";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--nmarkers") {
            nmarkers = std::stoull(next());
        } else if (a == "--nread") {
            nread = std::stoull(next());
        } else if (a == "--hring") {
            hring_words = (uint32_t)std::stoul(next());
        } else if (a == "--proddelay") {
            prod_delay = (uint32_t)std::stoul(next());
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
        } else if (a == "--direct") {
            direct = true;
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
            fprintf(stderr, "reset failed\n");
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    std::vector<uint8_t> idle_fw = read_file(idle), active_fw = read_file(active);
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
    uint32_t NL = num_cores * NRISC;
    uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    printf(
        "[grid] subrange (%d,%d)-(%d,%d) = %u cores (%u lanes); %llu readers; %llu markers/lane; "
        "host ring %u words\n",
        cx0,
        cy0,
        cx1,
        cy1,
        num_cores,
        NL,
        (unsigned long long)nread,
        (unsigned long long)nmarkers,
        hring_words);

    // VIRTUAL coords for host UMD access; TRANSLATED for the X280 read-window table.
    std::vector<CoreCoord> vc(num_cores);
    std::vector<uint8_t> coords(num_cores * 8, 0);
    const auto& soc = cluster.get_soc_desc(device_id);
    uint32_t idx = 0;
    for (int ly = cy0; ly <= cy1; ly++) {
        for (int lx = cx0; lx <= cx1; lx++) {
            CoreCoord lg{(uint32_t)lx, (uint32_t)ly};
            vc[idx] = cluster.get_virtual_coordinate_from_logical_coordinates(device_id, lg, CoreType::WORKER);
            tt::umd::CoreCoord tr = soc.translate_coord_to(
                {lg, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL}, tt::CoordSystem::TRANSLATED);
            pack<uint32_t>(coords, idx * 8 + 0, (uint32_t)tr.x);
            pack<uint32_t>(coords, idx * 8 + 4, (uint32_t)tr.y);
            idx++;
        }
    }

    // Zero the worker profiler control (head/tail start at 0) before boot.
    std::vector<uint8_t> zctrl(32 * 4, 0);
    for (uint32_t c = 0; c < num_cores; c++) {
        cluster.write_core(zctrl.data(), (uint32_t)zctrl.size(), tt_cxy_pair(device_id, vc[c]), prof_l1);
    }

    // Single host ring at the mid-channel pinned region.
    auto pcie_cores = cluster.get_soc_desc(device_id).get_cores(tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED);
    CoreCoord pc = pcie_cores.front();
    uint64_t pcie_enc = ((uint64_t)pc.x & 0x3f) | (((uint64_t)pc.y & 0x3f) << 6);
    uint64_t pcie_base = cluster.get_pcie_base_addr_from_device(device_id);
    uint64_t chan_sz = cluster.get_host_channel_size(device_id, 0);
    uint64_t data_off = (chan_sz / 2) & ~(WIN_STRIDE - 1);
    uint64_t host_base = pcie_base + data_off;
    uint64_t hring_bytes = (uint64_t)hring_words * 4;
    if (hring_bytes > WIN_STRIDE) {
        fprintf(stderr, "[d2h] host ring exceeds 2 MB window\n");
        std::_Exit(2);
    }
    {  // zero the host ring
        std::vector<uint8_t> z(hring_bytes, 0);
        cluster.write_sysmem(z.data(), (uint32_t)z.size(), data_off, device_id, 0);
    }

    // ---- boot ----
    X280Driver drv(cluster, device_id, l2cpu);
    bool half_broken = false;
    if (!drv.ensure_idle(idle_fw, pll, std::chrono::milliseconds(3000), half_broken)) {
        fprintf(stderr, "[boot] idle FW not up (half_broken=%d) -- needs `tt-smi -r %d`\n", half_broken, device_id);
        std::_Exit(1);
    }
    {  // zero HACKED + HSENT + STAGECTL (PROD/CONS) BEFORE boot -- no reader/relay init race
        std::vector<uint8_t> z(512, 0);
        drv.write_block(z.data(), 8, HACKED_ADDR);
        drv.write_block(z.data(), 8, HSENT_ADDR);
        drv.write_block(z.data(), 512, 0x08018000ULL);  // STAGECTL
    }
    drv.write_block(coords.data(), (uint32_t)coords.size(), MBOX_COORDS);
    {  // precompute the STICKY-SRC lookup table: lane L -> 8 B packet
        std::vector<uint8_t> lut(NL * 8, 0);
        for (uint32_t L = 0; L < NL; L++) {
            pack<uint32_t>(lut, L * 8 + 0, pp_src_w0(L));
            pack<uint32_t>(lut, L * 8 + 4, pp_src_w1(L));
        }
        drv.write_block(lut.data(), (uint32_t)lut.size(), SRCLUT_BASE);
    }
    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, pcie_enc);
    pack<uint64_t>(params, 0x08, host_base);
    pack<uint64_t>(params, 0x10, prof_l1);
    pack<uint64_t>(params, 0x18, (uint64_t)num_cores);
    pack<uint64_t>(params, 0x20, (uint64_t)hring_words);
    pack<uint64_t>(params, 0x28, 0);  // P_STOP
    pack<uint64_t>(params, 0x30, read_noc | (direct ? 0x100ull : 0ull));  // NONCE bit 8 = direct drain
    pack<uint64_t>(params, 0x38, nread);
    drv.write_block(params.data(), (uint32_t)params.size(), MBOX_PARAMS);
    uint64_t nharts = direct ? 1 : nread + 1;
    for (uint64_t h = 0; h < nharts; h++) {
        drv.lim_wr_u64(harthb((int)h), 0);
    }
    if (!drv.handoff_to_active_fw(active_fw, std::chrono::milliseconds(3000))) {
        fprintf(stderr, "[boot] active FW (profstream) did not stamp RUNNING\n");
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
                "[boot] not all %llu harts reached work loop -- flaky boot, `tt-smi -r`\n",
                (unsigned long long)nharts);
            std::_Exit(1);
        }
    }
    printf("[boot] idle up, profstream RUNNING, %llu readers + 1 relay\n", (unsigned long long)nread);

    // ---- host consumer: drain the ONE host ring into a RAW capture buffer as fast as possible; the
    // sticky-demux + verify happen OFFLINE after the run, so the hot loop is just a bulk memcpy (keeps the
    // host ahead of the relay -> the ring never wraps -> no torn reads). This is also how a real profiler
    // would work: capture raw, post-process. ----
    std::atomic<bool> producers_done{false};
    std::vector<std::vector<uint32_t>> accum(NL);  // demuxed per-lane stream (filled offline)
    std::vector<uint32_t> capture;                 // the raw linearized stream, appended in order
    capture.reserve((size_t)NL * (nmarkers + 8) * 2 + 64);
    uint64_t total_words = 0;
    std::atomic<uint64_t> overflow{0};
    uint64_t h_polls = 0, h_us_hsent = 0, h_us_ring = 0, h_us_hacked = 0;  // consumer op timing (us)
    std::thread consumer([&]() {
        auto us = [](auto a, auto b) {
            return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
        };
        uint32_t acked = 0;  // words the host has consumed
        int empty = 0;
        auto start = std::chrono::steady_clock::now();
        auto next_log = start + std::chrono::seconds(1);
        for (;;) {
            auto now = std::chrono::steady_clock::now();
            if (now > next_log) {
                uint32_t hs = 0, ha = 0;
                drv.read_block(reinterpret_cast<uint8_t*>(&hs), 4, HSENT_ADDR);
                drv.read_block(reinterpret_cast<uint8_t*>(&ha), 4, HACKED_ADDR);
                printf(
                    "  [consumer] total=%llu hsent=%u acked=%u done=%d\n",
                    (unsigned long long)total_words,
                    hs,
                    ha,
                    (int)producers_done.load());
                next_log = now + std::chrono::seconds(1);
            }
            if (now - start > std::chrono::seconds(30)) {
                printf("  [consumer] WALL TIMEOUT at %llu words\n", (unsigned long long)total_words);
                break;
            }
            uint32_t hsent;
            auto ta = std::chrono::steady_clock::now();
            drv.read_block(reinterpret_cast<uint8_t*>(&hsent), 4, HSENT_ADDR);
            h_us_hsent += us(ta, std::chrono::steady_clock::now());
            if (hsent == acked) {
                // exit only after producers are done AND the stream has been quiet for ~500 ms (the relay
                // can lull mid-drain; exiting early truncates lane tails).
                if (producers_done.load() && ++empty >= 2500) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            empty = 0;
            uint32_t avail = hsent - acked;
            if (avail > hring_words) {
                overflow.fetch_add(1);
                acked = hsent;
                drv.write_block(reinterpret_cast<uint8_t*>(&acked), 4, HACKED_ADDR);
                continue;
            }
            uint32_t drain = avail & ~1u;  // 2-word aligned
            // read ONLY the new [acked, acked+drain) region straight into the capture buffer (1-2 contiguous
            // reads across the ring wrap) -- no whole-ring read, no per-marker decode.
            size_t base = capture.size();
            capture.resize(base + drain);
            uint32_t start = acked % hring_words;
            auto tr = std::chrono::steady_clock::now();
            if (start + drain <= hring_words) {
                cluster.read_sysmem(
                    reinterpret_cast<uint8_t*>(&capture[base]),
                    drain * 4,
                    data_off + (uint64_t)start * 4,
                    device_id,
                    0);
            } else {
                uint32_t first = hring_words - start;
                cluster.read_sysmem(
                    reinterpret_cast<uint8_t*>(&capture[base]),
                    first * 4,
                    data_off + (uint64_t)start * 4,
                    device_id,
                    0);
                cluster.read_sysmem(
                    reinterpret_cast<uint8_t*>(&capture[base + first]), (drain - first) * 4, data_off, device_id, 0);
            }
            h_us_ring += us(tr, std::chrono::steady_clock::now());
            h_polls++;
            acked += drain;
            total_words += drain;
            auto th = std::chrono::steady_clock::now();
            drv.write_block(reinterpret_cast<uint8_t*>(&acked), 4, HACKED_ADDR);
            h_us_hacked += us(th, std::chrono::steady_clock::now());
        }
    });

    // ---- dispatch producers (unchanged from the lossless test) ----
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
        d["PROD_DELAY"] = std::to_string(prod_delay) + "u";
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
    if (active_riscs >= 2) {
        CreateKernel(
            program,
            kdir + "producer_dm.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = d1});
    }
    if (active_riscs >= NRISC) {
        CreateKernel(program, kdir + "producer_compute.cpp", all_cores, ComputeConfig{.defines = dc});
    }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    printf("[run] dispatching producers (%llu markers/lane)...\n", (unsigned long long)nmarkers);
    distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), wl, /*blocking=*/true);
    printf("[run] producers done; draining remainder\n");

    drv.lim_wr_u64(P_STOP, 1);
    producers_done.store(true);
    consumer.join();
    if (!drv.wait_active_fw_returned(std::chrono::seconds(5))) {
        fprintf(stderr, "[run] pipeline did not return to idle (unexpected)\n");
    }

    // ---- read per-lane producer STALL stats (did the ring fill -> did the producer block?) ----
    std::vector<uint32_t> st_events(NL, 0);
    std::vector<uint64_t> st_spins(NL, 0);
    for (uint32_t c = 0; c < num_cores; c++) {
        std::vector<uint8_t> sb(NRISC * 16);
        cluster.read_core(sb.data(), NRISC * 16, tt_cxy_pair(device_id, vc[c]), prof_l1 + 0x2C00);
        const uint32_t* s = reinterpret_cast<const uint32_t*>(sb.data());
        for (uint32_t r = 0; r < NRISC; r++) {
            st_events[c * NRISC + r] = s[r * 4 + 0];
            st_spins[c * NRISC + r] = (uint64_t)s[r * 4 + 1] | ((uint64_t)s[r * 4 + 2] << 32);
        }
    }

    // ---- pipeline profile: where does each X280 hart spend its cycles? ----
    printf("\n--- pipeline profile (X280 cycles; copy%% = fraction of wall spent moving data) ---\n");
    uint64_t hmax = direct ? 0 : nread;  // direct: only hart 0 drains
    for (uint64_t h = 0; h <= hmax; h++) {
        std::vector<uint8_t> rs(0x40);
        drv.read_block(rs.data(), 0x40, 0x08011040ULL + h * 0x40);
        const uint64_t* v = reinterpret_cast<const uint64_t*>(rs.data());
        uint64_t bytes = v[0], t_copy = v[1], t_total = v[2], aux1 = v[4], aux2 = v[5], breach = v[6];
        uint64_t words = bytes / 4 + 1;
        double busy = t_total ? 100.0 * (double)t_copy / (double)t_total : 0.0;
        if (direct) {
            printf(
                "  drain : %6llu KB  wall=%lluM  copy=%5.1f%%  %.1f cyc/word  host-wait=%lluM cyc\n",
                (unsigned long long)(bytes / 1024),
                (unsigned long long)(t_total / 1000000),
                busy,
                (double)t_copy / (double)words,
                (unsigned long long)(aux1 / 1000000));
        } else if (h == nread) {
            printf(
                "  relay : %6llu KB  wall=%lluM  copy=%5.1f%%  %.1f cyc/word  hostfull=%llu idle=%llu "
                "OVERWRITE=%llu\n",
                (unsigned long long)(bytes / 1024),
                (unsigned long long)(t_total / 1000000),
                busy,
                (double)t_copy / (double)words,
                (unsigned long long)aux1,
                (unsigned long long)aux2,
                (unsigned long long)breach);
        } else {
            printf(
                "  reader%llu: %6llu KB  wall=%lluM  copy=%5.1f%%  %.1f cyc/word  spsc-wait=%lluM cyc\n",
                (unsigned long long)h,
                (unsigned long long)(bytes / 1024),
                (unsigned long long)(t_total / 1000000),
                busy,
                (double)t_copy / (double)words,
                (unsigned long long)(aux1 / 1000000));
        }
    }
    printf(
        "  host  : %llu polls  ring-read=%llums (%llu us/poll)  hsent-read=%llums  hacked-write=%llums\n",
        (unsigned long long)h_polls,
        (unsigned long long)(h_us_ring / 1000),
        (unsigned long long)(h_us_ring / (h_polls + 1)),
        (unsigned long long)(h_us_hsent / 1000),
        (unsigned long long)(h_us_hacked / 1000));

    // ---- offline demux: walk the raw captured stream, bind each marker/meta to the current STICKY-SRC ----
    {
        uint32_t cur_lane = 0xFFFFFFFF;
        for (size_t p = 0; p + 1 < capture.size(); p += 2) {
            uint32_t w0 = capture[p], w1 = capture[p + 1];
            if (pp_is_src(w0)) {
                cur_lane = pp_src_lane(w0);
            } else if (cur_lane < NL) {
                accum[cur_lane].push_back(w0);
                accum[cur_lane].push_back(w1);
            }
        }
    }
    printf("  [capture] %zu raw words drained\n", capture.size());

    // ---- verify each DEMUXED lane is complete + gap-free ----
    uint32_t active_lanes = num_cores * active_riscs;
    uint64_t ok_lanes = 0, total_markers = 0, total_stickies = 0, seq_gaps = 0, ts_bad = 0, prog_bad = 0,
             short_lanes = 0;
    const uint64_t TS_MASK44 = (1ull << 44) - 1;
    std::vector<std::string> bad;
    for (uint32_t L = 0; L < NL; L++) {
        uint32_t r = L % NRISC;
        if (r >= active_riscs) {
            continue;
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
        if (mkrs != nmarkers) {
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
    printf("\n=== X280 linearized profiler (2 readers + 1 relay + single host ring, sticky-src) ===\n");
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
    // producer stall summary: did the pipeline back-pressure the workload?
    uint64_t lanes_stalled = 0, tot_events = 0, tot_spins = 0, max_spins = 0;
    for (uint32_t L = 0; L < NL; L++) {
        if (L % NRISC >= active_riscs) {
            continue;
        }
        if (st_events[L]) {
            lanes_stalled++;
        }
        tot_events += st_events[L];
        tot_spins += st_spins[L];
        if (st_spins[L] > max_spins) {
            max_spins = st_spins[L];
        }
    }
    printf(
        "  producer stall   : %llu/%u lanes stalled  (delay=%u/marker)  events=%llu  spins=%llu  max=%llu/lane\n",
        (unsigned long long)lanes_stalled,
        active_lanes,
        prod_delay,
        (unsigned long long)tot_events,
        (unsigned long long)tot_spins,
        (unsigned long long)max_spins);
    std::fflush(stdout);
    std::_Exit((ok_lanes == active_lanes) ? 0 : 1);
}
