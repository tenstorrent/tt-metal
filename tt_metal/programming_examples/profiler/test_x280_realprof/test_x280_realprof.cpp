// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// X280 REAL kernel_profiler capture test. Unlike test_x280_stream (synthetic producer_common.h stand-in),
// this dispatches a REAL DeviceZoneScopedN workload (realprof_{dm,compute}.cpp), so the production path is
// exercised end to end: kernel_profiler.hpp emits 2-word markers + STICKY_PROG (runtime host-id, pushed by
// BRISC) + STICKY_TIMER (wall-clock high, on tick) into the per-RISC L1 SPSC rings; the X280 (profzone.bin)
// drains them, injecting a (core,risc) STICKY_SRC per source; the host demuxes the one linearized stream and
// reconstructs device records (lane, zone type, srcloc hash, full timestamp, prog). Verifies START/END
// balance, prog propagation, and per-lane timestamp monotonicity (real markers carry no synthetic seq).
//
// Flexible knobs pick what pushes and how fast: cores --cx0..--cy1, RISCs --onelane/--twolane/all,
// zones/lane --nmarkers, per-zone work (rate) --proddelay, runtime host-id --progid.
//
// Build:  make -C tools/x280_bm build/lim_idle.bin build/profzone.bin
//         cmake --build build_Release --target test_x280_realprof
// Run:    TT_METAL_DEVICE_PROFILER=1 TT_METAL_NO_RT_PROFILER=1 \
//           ./build_Release/programming_examples/test_x280_realprof --reset --nmarkers 300 --nread 2

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

// --pin: bind a std::thread to a single core so it can't be descheduled by other work. Used to give the
// host-ring flushers dedicated cores, isolating them from the live-Tracy machinery (see MPMC diagnostics).
static void pin_thread_to_core(std::thread& t, int core) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(set), &set);
    // Read back so we KNOW the pin took (a cgroup cpuset can silently confine us to a different set).
    cpu_set_t got;
    CPU_ZERO(&got);
    pthread_getaffinity_np(t.native_handle(), sizeof(got), &got);
    int actual = -1;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (CPU_ISSET(c, &got)) {
            actual = c;
            break;
        }
    }
    printf(
        "[pin]   requested core %d  set_rc=%d  now-allowed-first-core=%d  count=%d\n",
        core,
        rc,
        actual,
        CPU_COUNT(&got));
#else
    (void)t;
    (void)core;
#endif
}

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <impl/context/metal_context.hpp>
#include <llrt/tt_cluster.hpp>
#include <tools/profiler/x280_driver.hpp>
#include <tools/profiler/x280_profzone_boot.hpp>  // shared profzone bring-up (single source of truth)
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>   // D2HSocket (production D2H transport, --socket)
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>  // MeshCoreCoord
#include <umd/device/types/core_coordinates.hpp>

#include "prof_packet.h"

// --tracy: feed decoded records into the existing RealtimeProfilerTracyHandler (Yusuf's branch) so the
// zones render in Tracy. All no-ops unless the build is TRACY_ENABLE (build_Release is).
#if defined(TRACY_ENABLE)
#include <tt-metalium/experimental/realtime_profiler_packets.hpp>
#include "impl/dispatch/realtime_profiler_tracy_handler.hpp"
#include "impl/profiler/profiler.hpp"  // loadZoneSourceLocationsHashesReadOnly (zone hash -> name)
#endif

using tt::Cluster;
using tt::CoreType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::MetalContext;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::profiler::X280Driver;

// --- LIM map (MUST match tools/x280_bm/src/profstream.c). MBOX_COORDS/SRCLUT_BASE now live in the shared
//     bring-up header (x280_profzone_boot.hpp); only what the host hot-path still touches stays here. ---
static constexpr uint64_t MBOX_PARAMS = 0x08011000ULL;
// SENT pointer now lives in HOST SYSMEM (ring trailer), not LIM -- see sent_off(). Only HACKED stays in LIM.
static constexpr uint64_t HACKED_BASE = 0x08017200ULL;  // per-hart, host writes, X280 reads (stride 0x40)
static uint64_t HACKED_ADDR_H(uint64_t h) { return HACKED_BASE + h * 0x40; }
static constexpr uint64_t P_STOP = MBOX_PARAMS + 0x28;
static uint64_t harthb(int h) { return 0x08011040ULL + 0x100 + (uint64_t)h * 8; }

static constexpr uint64_t DEFAULT_WIN_STRIDE = 0x200000ULL;  // 2 MB default host-ring budget; --winmb overrides
static constexpr uint64_t NOC_2M_STRIDE = 0x200000ULL;       // X280 posted-write TLB window = HARD 2 MiB (noc.h)
static constexpr int WRITE_WIN_BASE = 200;   // raw host-ring posted write windows (MUST match profzone.c)
static constexpr int SOCKET_WIN_BASE = 208;  // socket windows start here -> raw rings must stay in [200,208)
static constexpr int NRISC = 5;
static constexpr uint32_t RING_CAP = 512;  // worker L1 ring depth (words) -- MUST match profstream.c/producer

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

// ---- real host pipeline (--mpmc): per-socket flush+demux -> device records -> MPMC -> M consumers ----
// A fully-decoded device record. The flusher (per socket, in-order) resolves lane/type/zone/ts/prog so the
// consumers are STATELESS (any consumer can process any batch -> a shared MPMC works). `type` is the packet
// type (PP_ZONE_START/END/TOTAL), `zone` the 16-bit srcloc hash naming the zone, `ts` the full device
// timestamp, `prog` the runtime host-id in effect. This is the record a Tracy emitter would consume.
struct Rec {
    uint32_t lane;
    uint32_t type;
    uint32_t zone;
    uint64_t ts;
    uint32_t prog;
};
using Batch = std::vector<Rec>;

// Bounded MPMC queue of record batches (mutex + condvars). Batches are large, so the lock is hit rarely
// (amortized) -- the point is the BOUND: when full, producers block = back-pressure up the pipeline.
struct BatchQ {
    std::mutex m;
    std::condition_variable cv_pop, cv_push;
    std::queue<Batch> q;
    size_t cap;
    bool closed = false;
    size_t peak = 0;           // DIAG: max occupancy ever reached (batches)
    uint64_t push_blocks = 0;  // DIAG: # pushes that found the queue full and had to wait (real MPMC back-pressure)
    explicit BatchQ(size_t c) : cap(c) {}
    void push(Batch&& b) {
        std::unique_lock<std::mutex> lk(m);
        if (q.size() >= cap) {
            push_blocks++;  // flusher is about to block on a full queue => this is where MPMC back-pressure starts
        }
        cv_push.wait(lk, [&] { return q.size() < cap; });
        q.push(std::move(b));
        if (q.size() > peak) {
            peak = q.size();
        }
        cv_pop.notify_one();
    }
    bool pop(Batch& out) {
        std::unique_lock<std::mutex> lk(m);
        cv_pop.wait(lk, [&] { return !q.empty() || closed; });
        if (q.empty()) {
            return false;  // closed and drained
        }
        out = std::move(q.front());
        q.pop();
        cv_push.notify_one();
        return true;
    }
    void close() {
        std::unique_lock<std::mutex> lk(m);
        closed = true;
        cv_pop.notify_all();
    }
};

int main(int argc, char** argv) {
    using namespace tt::tt_metal;
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int device_id = 0, l2cpu = 0, pll = 1000;
    uint64_t nmarkers = 2000, nread = 2, ts_step = 0x1000000ull, ndrain = 1;
    uint64_t win_stride = DEFAULT_WIN_STRIDE;  // --winmb: host-ring budget (rings share it); capped by chan_sz/2
    uint32_t prog_id = 0xA5A5A5A5u, hring_words = 1048576, prod_delay = 0;  // 4 MiB/ring (nwin=3: data spans 2
    // windows, the +64 B SENT trailer spills into a 3rd -> 3 TLB windows/ring, so nread<=2 at the default).
    // Sized for LIVE TRACY capture, the production case: with a connected tracy-capture the Tracy serialize+send
    // is a slow/bursty sink that fills the host ring, so ring size sets the knee. Sweep (nread=2 dualrelay, capture
    // connected): 256 KB knee ~4400, 1 MB ~2800, 2 MiB ~1000, 4 MiB ~850 == the no-Tracy drain floor (~830) -> live
    // Tracy is essentially free at 4 MiB. Below ~600 every size craters to the worker-L1-ring floor
    // (total_markers/256) that no host buffer can beat. (4 MiB-64 = 1048560 keeps it to nwin=2 -> nread<=4 if needed.)
    bool do_reset = false, direct = false;  // --direct: direct drain (no reader/relay split); --ndrain N: N drainers
    bool rr_consumer = false;  // --rrconsumer: one host thread round-robins all rings (else one thread per ring)
    bool split_noc = false;    // --splitnoc: drain hart h reads its slice over NoC (h&1) to relieve read contention
    bool wnoc1 = false;        // --wnoc1: route the posted PCIe write over NoC1 (reads stay NoC0)
    bool nodrain = false;      // --nodrain: diagnostic -- relay ignores host flow control + no host consumer
                               // (isolates whether the reader is throttled by the host sink; LOSSY on purpose)
    bool fullread = false;     // --fullread: "all buffers full" bench -- reader ignores tail, always drains a
                               // full RING_CAP buffer per (core,risc). Deterministic max-drain load; LOSSY.
    bool bulkcore = false;     // --bulkcore: one bulk NoC read of the whole core (all 5 rings, 2560 words) --
                               // amortizes NoC latency (rdrbench >2GB/s regime), drops per-risc round-robin. LOSSY.
    bool dualrelay = true;     // DEFAULT: one relay hart PER reader (2 D2H sockets, decouple the chip halves).
                               // Lifts the single-relay funnel -> the drain becomes NoC-read-bound, not
                               // relay-bound. --singlerelay reverts to one shared relay for A/B.
    bool adaptive = true;      // DEFAULT: per-core switch -- bulk read a core once it holds >= ADAPT_THRESH
                               // (4*RING_CAP) pending words, else per-risc. Lowest no-stall knee (~830 vs ~940
                               // per-risc) w/o bulk's constant NoC over-read. --noadaptive reverts to per-risc.
    int mpmc = 0;              // --mpmc M: real host pipeline -- 2 per-socket flush+demux threads -> record MPMC
                               // -> M consumer threads (0 = old offline capture+demux path)
    int cwork = 0;             // --cwork N: busy-wait N iters/record in each consumer (simulate Tracy-emit load)
    int mq_cap = 640;          // --mqcap N: depth (in batches) of the record MPMC between flushers and consumers.
                               // Deeper absorbs consumer bursts before back-pressuring the drain (64->640 took
                               // --csv from ~438 knee stalls to 0). ~BATCH_RECS*N*sizeof(Rec) max footprint.
    bool do_tracy = false;     // --tracy: emit decoded zones into Tracy (via RealtimeProfilerTracyHandler) so
                               // they visualize. Uses ONE consumer (per-lane START/END order must be serial).
    bool socket_mode = false;  // --socket: relay drains into one tt-metal D2HSocket per relay (the production
                               // D2H transport) instead of raw sysmem rings; A/B the knee vs the raw path.
    bool bringup = false;      // --bringup: STEP1 -- boot profzone as a PERSISTENT active FW and confirm it
                               // stays resident (idle FW never re-entered); no workload, no P_STOP, exit resident.
    bool do_pin = false;       // --pin: bind each flusher to its own core (0,1,...). Pair with `taskset -c 2-N`
                               // on the process so cores 0/1 are otherwise empty -> flushers can't be descheduled
                               // by the live-Tracy consumer/send-worker/capture (the measured stall source).
    bool do_csv = false;       // --csv [path]: instead of Tracy, hold decoded records in memory (cheap append on
                               // the hot path) and write them to CSV in large batches AFTER the drain -- so the
                               // consumer callback never back-pressures the X280 (goal: 0 stalls, like the sink).
    std::string csv_path = "tracy_captures/realprof.csv";
    uint32_t active_riscs = NRISC;
    int cx0 = -1, cy0 = -1, cx1 = -1, cy1 = -1;
    uint64_t read_noc = 0;

    std::string idle = "tools/x280_bm/build/lim_idle.bin", active = "tools/x280_bm/build/profzone.bin";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--nmarkers") {
            nmarkers = std::stoull(next());
        } else if (a == "--nread") {
            nread = std::stoull(next());
        } else if (a == "--hring") {
            hring_words = (uint32_t)std::stoul(next());
        } else if (a == "--winmb") {
            win_stride = (uint64_t)std::stoull(next()) << 20;  // host-ring budget in MB (rings share it)
        } else if (a == "--proddelay") {
            prod_delay = (uint32_t)std::stoul(next());
        } else if (a == "--progid") {
            prog_id = (uint32_t)std::stoul(next(), nullptr, 0);
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
        } else if (a == "--ndrain") {
            ndrain = std::stoull(next());
            direct = true;
        } else if (a == "--rrconsumer") {
            rr_consumer = true;
        } else if (a == "--splitnoc") {
            split_noc = true;
            direct = true;
        } else if (a == "--wnoc1") {
            wnoc1 = true;
            direct = true;
        } else if (a == "--socket") {
            socket_mode = true;
        } else if (a == "--nodrain") {
            nodrain = true;
        } else if (a == "--fullread") {
            fullread = true;
        } else if (a == "--bulkcore") {
            bulkcore = true;  // force the lossless bulk path for every core (drains real [head,tail))
        } else if (a == "--dualrelay") {
            dualrelay = true;  // default; kept explicit for clarity
        } else if (a == "--singlerelay") {
            dualrelay = false;  // opt out: one shared relay hart (the old funnel) for A/B comparison
        } else if (a == "--adaptive") {
            adaptive = true;  // default; kept explicit for clarity
        } else if (a == "--noadaptive") {
            adaptive = false;  // opt out: force plain per-risc drains (higher stall knee) for A/B
        } else if (a == "--mpmc") {
            mpmc = std::stoi(next());
        } else if (a == "--cwork") {
            cwork = std::stoi(next());
        } else if (a == "--mqcap") {
            mq_cap = std::stoi(next());
        } else if (a == "--tracy") {
            do_tracy = true;
        } else if (a == "--pin") {
            do_pin = true;
        } else if (a == "--bringup") {
            bringup = true;  // STEP1: boot profzone as a persistent active FW + confirm; no workload, no P_STOP
        } else if (a == "--csv") {
            do_csv = true;
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                csv_path = next();  // optional path arg
            }
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

    // VIRTUAL coords for host UMD access; TRANSLATED for the X280 read-window table; NOC0 for the Tracy
    // contexts (per-core row keys, matching the standard DeviceProfiler view -- see --tracy).
    std::vector<CoreCoord> vc(num_cores);
    std::vector<uint8_t> coords(num_cores * 8, 0);
    std::vector<uint32_t> noc0x(num_cores, 0), noc0y(num_cores, 0);
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
            tt::umd::CoreCoord n0 =
                soc.translate_coord_to({lg, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL}, tt::CoordSystem::NOC0);
            noc0x[idx] = (uint32_t)n0.x;
            noc0y[idx] = (uint32_t)n0.y;
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
    // Must be 2 MiB-aligned so ring 0 starts at in-window offset 0 (the whole 2 MiB-strided layout depends on it),
    // independent of --winmb. win_stride only widens the alignment further, never below 2 MiB.
    uint64_t data_off = (chan_sz / 2) & ~((win_stride > NOC_2M_STRIDE ? win_stride : NOC_2M_STRIDE) - 1);
    uint64_t host_base = pcie_base + data_off;
    uint64_t hring_bytes = (uint64_t)hring_words * 4;
    // The X280 reaches host sysmem through 2 MiB posted-write TLB windows (NOC_2M_STRIDE, hardware-fixed). A ring
    // spans nwin = ceil((data + 64 B trailer)/2 MiB) CONSECUTIVE windows mapping CONSECUTIVE host pages; because
    // window addresses are contiguous (win*STRIDE) and each covers exactly one page, a linear write across a
    // boundary routes transparently -> the ring behaves as one nwin*2 MiB region. Rings are STRIDED at nwin*2 MiB.
    // (Packing rings sub-2-MiB made ring h>=1 start at a nonzero in-window offset so a >256 KB ring's tail spilled
    // into an unconfigured window -> flusher WALL TIMEOUT: the old 1 MB wedge.) --winmb now only aligns data_off.
    uint64_t nwin = (hring_bytes + 64 + (NOC_2M_STRIDE - 1)) / NOC_2M_STRIDE;  // 2 MiB windows this ring spans
    uint64_t ring_stride = nwin * NOC_2M_STRIDE;
    uint64_t ndh = direct ? ndrain : nread;   // # host rings: direct = 1/drain hart; split = 1/reader (relay
                                              // writes reader h -> ring h), each drained by its own host thread
    // SENT pointer for ring h lives in sysmem at data_off + h*ring_stride + hring_bytes (the trailer).
    auto sent_off = [&](uint64_t h) { return data_off + h * ring_stride + hring_bytes; };
    // Raw write windows occupy [WRITE_WIN_BASE, SOCKET_WIN_BASE); each ring consumes nwin of them.
    if (!socket_mode && ndh * nwin > (uint64_t)(SOCKET_WIN_BASE - WRITE_WIN_BASE)) {
        fprintf(
            stderr,
            "[d2h] %llu rings x %llu windows (--hring %u = %llu B) exceed %d raw TLB windows -- lower "
            "--hring/--nread\n",
            (unsigned long long)ndh,
            (unsigned long long)nwin,
            hring_words,
            (unsigned long long)hring_bytes,
            SOCKET_WIN_BASE - WRITE_WIN_BASE);
        std::_Exit(2);
    }
    // data_off starts at chan_sz/2, so the rings have the whole second half of the PCIe channel to grow into;
    // the X280 writes them through the PCIe tile which maps the full channel, so win_stride is just a budget.
    if (!socket_mode && data_off + ring_stride * ndh > chan_sz) {
        fprintf(stderr, "[d2h] rings overflow host channel (%llu MB)\n", (unsigned long long)(chan_sz >> 20));
        std::_Exit(2);
    }
    // The real test runs through the production MPMC pipeline (flush+demux -> record MPMC -> consumers).
    // Default to one consumer per socket if the user did not pick a count; the offline capture path is the
    // synthetic benchmark's and is not used here.
    if (mpmc == 0) {
        mpmc = (int)ndh;
        printf("[realprof] defaulting to MPMC pipeline (%d consumer(s))\n", mpmc);
    }
    if (!socket_mode) {  // zero all host rings + their SENT trailers (raw path only; socket owns its own FIFO)
        std::vector<uint8_t> z(ring_stride * ndh, 0);
        cluster.write_sysmem(z.data(), (uint32_t)z.size(), data_off, device_id, 0);
    }

    // ---- boot: profzone as a RESIDENT active FW via the shared bring-up (single source of truth;
    //      see tools/profiler/x280_profzone_boot.hpp). Same params/SRCLUT/handoff/verify the manager uses. ----
    X280Driver drv(cluster, device_id, l2cpu);
    namespace pz = tt::tt_metal::profiler;
    pz::ProfzoneBootCfg bcfg;
    bcfg.idle_fw = idle_fw;
    bcfg.active_fw = active_fw;
    bcfg.pll_mhz = pll;
    bcfg.pcie_enc = pcie_enc;
    bcfg.host_base = host_base;
    bcfg.prof_l1 = prof_l1;
    bcfg.num_cores = (uint64_t)num_cores;
    bcfg.hring_words = hring_words;
    bcfg.ndh = ndh;
    bcfg.nread = nread;
    bcfg.ndrain = ndrain;
    bcfg.coords = coords.data();
    bcfg.coords_bytes = (uint32_t)coords.size();
    bcfg.read_noc = read_noc;
    bcfg.direct = direct;
    bcfg.split_noc = split_noc;
    bcfg.wnoc1 = wnoc1;
    bcfg.nodrain = nodrain;
    bcfg.fullread = fullread;
    bcfg.bulkcore = bulkcore;
    bcfg.dualrelay = dualrelay;
    bcfg.adaptive = adaptive;
    bcfg.socket = socket_mode;
    // --socket: create one D2HSocket per relay (the production D2H transport) at the LIM config addrs the
    // relay reads (0x0801A000 + h*0x1000). Must exist BEFORE boot so the sender_socket_md is resident when
    // relay_run_socket reads it. sender_is_l2cpu routes the config + bytes_acked writes to the X280 LIM. FIFO
    // is the same size as the raw ring so the knee A/B is apples-to-apples (same host-side buffering).
    std::vector<std::unique_ptr<distributed::D2HSocket>> socks;
    if (socket_mode) {
        const CoreCoord l2phys = pz::x280_l2cpu_tile(l2cpu);
        distributed::MeshCoordinate scoord = *distributed::MeshCoordinateRange(mesh->shape()).begin();
        const uint32_t cfg_sz = distributed::D2HSocket::required_config_buffer_size();
        uint64_t fifo_lo[2] = {0, 0};  // per-socket FIFO NoC addr (lo32; packed into P_HOST_BASE param)
        for (uint64_t h = 0; h < ndh; h++) {
            uint32_t caddr = 0x08019000u + (uint32_t)h * 0x100u;  // config buffer (bytes_acked lives here, +32)
            auto s = std::make_unique<distributed::D2HSocket>(
                mesh,
                distributed::MeshCoreCoord{scoord, l2phys},
                (uint32_t)hring_bytes,
                distributed::D2HSocket::ExternalConfigBuffer{.address = caddr, .sender_is_l2cpu = true});
            s->set_page_size(64);
            socks.push_back(std::move(s));
            // Read the FIFO NoC addr the socket wrote (get_noc_addr). The X280 can't read this config buffer
            // from LIM coherently (per-hart stale), so we pass the FIFO addr via the P_HOST_BASE param instead
            // (packed lo32/socket -- get_noc_addr hi=0 on BH). bytes_acked (config+32) is still read live.
            std::vector<uint8_t> cfgbuf(cfg_sz, 0);
            drv.read_block(cfgbuf.data(), cfg_sz, caddr);
            const uint32_t* c = reinterpret_cast<const uint32_t*>(cfgbuf.data());
            uint64_t fifo = (((uint64_t)c[13]) << 32) | c[4];
            if ((fifo >> 32) != 0 && h < 2) {
                fprintf(
                    stderr,
                    "[socket%llu] FIFO addr 0x%llx has nonzero hi -- lo32 packing is lossy!\n",
                    (unsigned long long)h,
                    (unsigned long long)fifo);
                std::_Exit(2);
            }
            if (h < 2) {
                fifo_lo[h] = fifo & 0xffffffffull;
            }
            printf(
                "[socket%llu cfg] fifo_noc=0x%llx fifo_total=%u is_d2h=%u enc[14]=0x%x cfg_sz=%u\n",
                (unsigned long long)h,
                (unsigned long long)fifo,
                c[5],
                c[6],
                c[14],
                cfg_sz);
        }
        // Pass BOTH sockets' FIFO lo32 offsets packed into the single P_HOST_BASE param: socket0 in [31:0],
        // socket1 in [63:32] (get_noc_addr hi=0 on BH, verified above -> lossless). profzone reconstructs each
        // as pbase|lo32 (bit60 PCIe-outbound). For nread=1, fifo_lo[1]=0 (unused). This wires the dual-relay
        // 2-socket path: relay hart hri drains reader hri into socket hri (sk_fifo[hri]). Note pbase|lo32 ==
        // pcie_base+lo32, so the single-socket case reconstructs identically to the old full-addr form.
        bcfg.host_base = (uint64_t)fifo_lo[0] | ((uint64_t)fifo_lo[1] << 32);
        printf(
            "[socket] created %llu D2HSocket(s), %llu B FIFO each, page 64 B; P_HOST_BASE packed=0x%llx\n",
            (unsigned long long)ndh,
            (unsigned long long)hring_bytes,
            (unsigned long long)bcfg.host_base);
    }
    uint64_t nharts = 0;
    bool half_broken = false;
    if (!pz::boot_profzone(drv, bcfg, nharts, half_broken)) {
        fprintf(
            stderr,
            "[boot] profzone bring-up failed (half_broken=%d) -- `tt-smi -r %d`\n",
            (int)half_broken,
            device_id);
        std::_Exit(1);
    }
    if (socket_mode) {
        uint32_t hb[2] = {0, 0};
        drv.read_block(hb, 8, 0x08011008ull);  // P_HOST_BASE in LIM (should == packed fifo addrs)
        printf(
            "[socket POST-BOOT] LIM P_HOST_BASE = 0x%08x%08x (expect 0x%llx)\n",
            hb[1],
            hb[0],
            (unsigned long long)bcfg.host_base);
    }
    uint64_t nrelay = dualrelay ? nread : 1;
    if (direct) {
        printf("[boot] idle up, profzone RUNNING, %llu direct drain hart(s)\n", (unsigned long long)ndrain);
    } else {
        printf(
            "[boot] idle up, profzone RUNNING, %llu readers + %llu relay(s)\n",
            (unsigned long long)nread,
            (unsigned long long)nrelay);
    }

    if (bringup) {
        // ---- STEP 1: confirm profzone is a PERSISTENT active FW ----
        // The idle FW is only the launch pad (reset is released ONCE -> JUMP -> profzone). With NO P_STOP,
        // profzone must stay in its drain loop indefinitely and must NEVER bounce back to idle. Verify via the
        // boot-phase word: RUNNING_ACTIVE_FW while resident, RETURNED_TO_IDLE if it bounced.
        constexpr uint64_t X280_BOOT_PHASE = 0x080160C0ull;  // X280_BOOT_HANDSHAKE_BASE(0x08016000)+0xC0
        constexpr uint64_t PHASE_RUNNING = 0x7E570001ull;    // X280_BOOT_PHASE_RUNNING_ACTIVE_FW
        constexpr uint64_t PHASE_RET_IDLE = 0x1D1E0002ull;   // X280_BOOT_PHASE_RETURNED_TO_IDLE
        auto phname = [&](uint64_t p) {
            return p == PHASE_RUNNING ? "RUNNING_ACTIVE_FW" : (p == PHASE_RET_IDLE ? "RETURNED_TO_IDLE" : "?");
        };
        auto harts_at_work = [&]() {
            for (uint64_t h = 0; h < nharts; h++) {
                if (drv.lim_rd_u64(harthb((int)h)) != 3) {
                    return false;
                }
            }
            return true;
        };
        uint64_t ph0 = drv.lim_rd_u64(X280_BOOT_PHASE);
        printf(
            "[bringup] booted: %llu harts @ work loop, phase=0x%llx (%s)\n",
            (unsigned long long)nharts,
            (unsigned long long)ph0,
            phname(ph0));
        for (int s = 1; s <= 3; s++) {  // hold WITHOUT P_STOP -> must remain resident
            std::this_thread::sleep_for(std::chrono::seconds(1));
            printf(
                "[bringup]  +%ds: phase=0x%llx harts@work=%s\n",
                s,
                (unsigned long long)drv.lim_rd_u64(X280_BOOT_PHASE),
                harts_at_work() ? "all" : "MISSING");
        }
        bool ok = (drv.lim_rd_u64(X280_BOOT_PHASE) == PHASE_RUNNING) && harts_at_work();
        printf(
            "[bringup] RESULT: %s (profzone left RESIDENT -- no P_STOP, no idle bounce)\n",
            ok ? "PASS -- persistent active FW" : "FAIL -- bounced to idle or a hart dropped");
        std::_Exit(ok ? 0 : 1);
    }

    // ---- host consumer: drain the ONE host ring into a RAW capture buffer as fast as possible; the
    // sticky-demux + verify happen OFFLINE after the run, so the hot loop is just a bulk memcpy (keeps the
    // host ahead of the relay -> the ring never wraps -> no torn reads). This is also how a real profiler
    // would work: capture raw, post-process. ----
    std::atomic<bool> producers_done{false};
    std::atomic<bool> device_done{false};          // set once the FW returned to idle => ALL data is in the host rings;
                                                   // drainers then exit at hsent==acked (authoritative, not time-based)
    std::vector<std::vector<uint32_t>> accum(NL);  // demuxed per-lane stream (filled offline)
    std::vector<std::vector<uint32_t>> caps(ndh);  // one RAW capture per drain hart / host ring
    for (auto& cap : caps) {
        cap.reserve((size_t)NL * (nmarkers + 8) * 2 / ndh + 64);
    }
    std::atomic<uint64_t> total_words{0};
    std::atomic<uint64_t> overflow{0};
    // per-ring op timing (summed for the report); disjoint indices -> lock-free across threads
    std::vector<uint64_t> h_polls_v(ndh, 0), h_us_hsent_v(ndh, 0), h_us_ring_v(ndh, 0), h_us_hacked_v(ndh, 0);
    auto us = [](auto a, auto b) {
        return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    // Drain ONE ring h to quiescence. Ring h touches only disjoint sysmem (ring @ data_off+h*ring_stride,
    // SENT @ sent_off(h)) and disjoint LIM HACKED_H(h) + writes only caps[h]/counters[h] -> N run concurrently,
    // one host thread each. The SENT pointer is now in HOST SYSMEM (read via read_sysmem = a memcpy from the
    // hugepage, ~ns) instead of device LIM (drv.read_block ~18 us/poll) -- that was the host-drain wall.
    auto drain_ring = [&](uint64_t h) {
        uint64_t hoff = data_off + h * ring_stride;
        uint64_t soff = sent_off(h);
        uint32_t acked = 0;
        auto start = std::chrono::steady_clock::now();
        auto next_log = start + std::chrono::seconds(1);
        for (;;) {
            auto now = std::chrono::steady_clock::now();
            if (h == 0 && now > next_log) {
                printf(
                    "  [consumer] total=%llu done=%d\n",
                    (unsigned long long)total_words.load(),
                    (int)producers_done.load());
                next_log = now + std::chrono::seconds(1);
            }
            if (now - start > std::chrono::seconds(30)) {
                printf("  [consumer %llu] WALL TIMEOUT\n", (unsigned long long)h);
                break;
            }
            uint32_t hsent;
            auto ta = std::chrono::steady_clock::now();
            cluster.read_sysmem(reinterpret_cast<uint8_t*>(&hsent), 4, soff, device_id, 0);  // SENT from sysmem
            h_us_hsent_v[h] += us(ta, std::chrono::steady_clock::now());
            if (hsent == acked) {
                // SPIN (no sleep): the sent read is a cheap sysmem poll now, so keep the relay fed. Exit only
                // when the DEVICE is fully drained (FW returned to idle => sent is final) and we've caught up.
                if (device_done.load()) {
                    break;
                }
                continue;
            }
            uint32_t avail = hsent - acked;
            if (avail > hring_words) {
                overflow.fetch_add(1);
                acked = hsent;
                drv.write_block(reinterpret_cast<uint8_t*>(&acked), 4, HACKED_ADDR_H(h));
                continue;
            }
            uint32_t drain = avail & ~1u;  // 2-word aligned
            // read ONLY the new [acked, acked+drain) region straight into this ring's capture buffer
            // (1-2 contiguous reads across the wrap) -- no whole-ring read, no per-marker decode.
            auto& cap = caps[h];
            size_t base = cap.size();
            cap.resize(base + drain);
            uint32_t st = acked % hring_words;
            auto tr = std::chrono::steady_clock::now();
            if (st + drain <= hring_words) {
                cluster.read_sysmem(
                    reinterpret_cast<uint8_t*>(&cap[base]), drain * 4, hoff + (uint64_t)st * 4, device_id, 0);
            } else {
                uint32_t first = hring_words - st;
                cluster.read_sysmem(
                    reinterpret_cast<uint8_t*>(&cap[base]), first * 4, hoff + (uint64_t)st * 4, device_id, 0);
                cluster.read_sysmem(
                    reinterpret_cast<uint8_t*>(&cap[base + first]), (drain - first) * 4, hoff, device_id, 0);
            }
            h_us_ring_v[h] += us(tr, std::chrono::steady_clock::now());
            h_polls_v[h]++;
            acked += drain;
            total_words.fetch_add(drain);
            auto th = std::chrono::steady_clock::now();
            drv.write_block(reinterpret_cast<uint8_t*>(&acked), 4, HACKED_ADDR_H(h));
            h_us_hacked_v[h] += us(th, std::chrono::steady_clock::now());
        }
    };
    // ---- --mpmc pipeline: per-socket flush+demux -> device-record MPMC -> M consumers ----
    const size_t BATCH_RECS = 4096;  // records per batch (amortizes the MPMC lock)
    BatchQ mq((size_t)mq_cap);       // bounded MPMC (--mqcap batches): full -> flusher blocks = back-pressure.
                                     // Deeper absorbs consumer bursts before stalling the drain (default 640).
    std::atomic<uint64_t> consumed{0}, sink_total{0};
    // per-flusher (per-socket) verify results
    std::vector<uint64_t> fl_mk(ndh, 0), fl_start(ndh, 0), fl_end(ndh, 0), fl_prog_ok(ndh, 0), fl_ts_bad(ndh, 0),
        fl_unbal(ndh, 0), fl_stall(ndh, 0);
    std::vector<uint64_t> fl_pages(ndh, 0);  // --socket: total pages the host read from this socket (reliable)
    // Flusher for socket h: drain ring h IN ORDER, demux (dispatch bulk/per-risc) + decode into fully-
    // resolved device records + per-lane seq verify (this socket owns its half's lanes), push record batches
    // to the MPMC. Demux MUST live here (sticky-src is in-order per stream); records are self-contained so
    // the M consumers are stateless.
    auto flusher = [&](uint64_t h) {
        uint64_t hoff = data_off + h * ring_stride, soff = sent_off(h);
        uint32_t acked = 0;
        std::vector<uint32_t> cur_hi(NL, 0);   // per-lane wall-clock high half (STICKY_TIMER)
        std::vector<int32_t> depth(NL, 0);     // per-lane zone nesting (START ++ / END --); 0 at end = balanced
        std::vector<uint64_t> last_ts(NL, 0);  // per-lane last timestamp (monotonicity check)
        uint32_t cur_prog = 0;                 // GLOBAL: runtime host-id (BRISC-only STICKY_PROG, program-global)
        uint64_t mk = 0, starts = 0, ends = 0, prog_ok = 0, ts_bad = 0, stall = 0;
        Batch batch;
        batch.reserve(BATCH_RECS);
        std::vector<uint32_t> buf;
        std::vector<uint32_t> resid;  // --socket: partial packet carried across reads (socket pages are 16-word
                                      // aligned, NOT packet-aligned like the raw ring's packet-boundary SENT)
        uint32_t cur_lane = 0xFFFFFFFF;
        auto emit = [&](uint32_t lane, uint32_t w0, uint32_t w1) {
            if (lane >= NL) {
                return;
            }
            uint32_t type = pp_type(w0);
            if (type == PP_STICKY_PROG) {  // program-global runtime host-id (host forward-fills onto markers)
                cur_prog = pp_prog_id(w1);
                return;
            }
            if (type == PP_STICKY_TIMER) {  // this lane's wall-clock high half
                cur_hi[lane] = pp_timer_hi(w0);
                return;
            }
            if (type == PP_STICKY_META) {  // legacy combined sticky (real producer never emits it -- be robust)
                cur_hi[lane] = pp_low27(w0);
                cur_prog = w1;
                return;
            }
            // marker: type = ZONE_START/END/TOTAL, zone = 16-bit srcloc hash, ts = full device timestamp
            uint32_t zone = pp_low27(w0) & 0xFFFFu;
            uint64_t ts = pp_full_ts(cur_hi[lane], w1);
            if (zone == 0x7FFFu && type == PP_ZONE_START) {
                stall++;  // X280-STALL zone (PROFILER_STALL_ZONE_ID) = producer back-pressure event
            }
            if (type == PP_ZONE_START) {
                depth[lane]++;
                starts++;
            } else if (type == PP_ZONE_END) {
                depth[lane]--;
                ends++;
            }
            if (cur_prog == prog_id) {
                prog_ok++;
            }
            if (ts < last_ts[lane]) {
                ts_bad++;
            }
            last_ts[lane] = ts;
            mk++;
            batch.push_back(Rec{lane, type, zone, ts, cur_prog});
            if (batch.size() >= BATCH_RECS) {
                mq.push(std::move(batch));
                batch = Batch();
                batch.reserve(BATCH_RECS);
            }
        };
        auto start = std::chrono::steady_clock::now();
        for (;;) {
            auto now = std::chrono::steady_clock::now();
            if (now - start > std::chrono::seconds(120)) {
                printf("  [flusher %llu] WALL TIMEOUT\n", (unsigned long long)h);
                break;
            }
            if (socket_mode) {
                // --socket: read whole 64 B pages from the D2HSocket FIFO (read() auto-acks the sender via
                // bytes_acked). Pages are 16-word aligned, NOT packet-aligned, so prepend the residual partial
                // packet carried from the last read; the walk below consumes whole packets and re-saves the tail.
                uint32_t np = socks[h]->pages_available();
                if (np == 0) {
                    if (device_done.load()) {  // FW idle => sender's final bytes_sent is in; FIFO drained => done
                        break;
                    }
                    continue;  // spin
                }
                // Guard: never request more than the FIFO holds. read() TT_FATALs if num_bytes > fifo_curr_size,
                // and pages_available() can transiently spike above the FIFO (a bytes_acked-vs-bytes_sent race
                // during concurrent draining). Bound each read to (fifo pages - 1); the loop takes the rest next
                // iteration. Without this a single spike walks the circular FIFO repeatedly -> runaway garbage.
                uint32_t fifo_pages = socks[h]->get_fifo_curr_size() / socks[h]->get_page_size();
                if (np >= fifo_pages) {
                    np = fifo_pages - 1u;
                }
                uint32_t dw = np * 16u;  // 64 B page = 16 words
                buf.resize(resid.size() + dw);
                if (!resid.empty()) {
                    std::copy(resid.begin(), resid.end(), buf.begin());
                }
                socks[h]->read(reinterpret_cast<void*>(buf.data() + resid.size()), np);  // notify_sender=true
                total_words.fetch_add(dw);
                fl_pages[h] += np;  // reliable host-side signal: did the relay deliver anything to this socket?
                resid.clear();
            } else {
                uint32_t hsent;
                cluster.read_sysmem(reinterpret_cast<uint8_t*>(&hsent), 4, soff, device_id, 0);  // SENT from sysmem
                if (hsent == acked) {
                    if (device_done.load()) {  // FW idle => sent is final; caught up => done
                        break;
                    }
                    continue;  // spin
                }
                uint32_t avail = hsent - acked;
                if (avail > hring_words) {
                    overflow.fetch_add(1);
                    acked = hsent;
                    drv.write_block(reinterpret_cast<uint8_t*>(&acked), 4, HACKED_ADDR_H(h));
                    continue;
                }
                // sent is always published on a packet boundary, so avail is a whole number of packet-words;
                // drain all of it (packets are now VARIABLE length -- SRC/TIMER are 1 word -- so no even-align).
                uint32_t drain = avail;
                buf.resize(drain);
                uint32_t st = acked % hring_words;
                if (st + drain <= hring_words) {
                    cluster.read_sysmem(
                        reinterpret_cast<uint8_t*>(buf.data()), drain * 4, hoff + (uint64_t)st * 4, device_id, 0);
                } else {
                    uint32_t first = hring_words - st;
                    cluster.read_sysmem(
                        reinterpret_cast<uint8_t*>(buf.data()), first * 4, hoff + (uint64_t)st * 4, device_id, 0);
                    cluster.read_sysmem(
                        reinterpret_cast<uint8_t*>(buf.data() + first), (drain - first) * 4, hoff, device_id, 0);
                }
                acked += drain;
                total_words.fetch_add(drain);
                drv.write_block(reinterpret_cast<uint8_t*>(&acked), 4, HACKED_ADDR_H(h));
            }
            // decode buf (whole frames): variable-length walk. SRC/TIMER are 1 word; PROG/markers 2; BULK
            // has its own framing. Advance by the decoded length so packet boundaries stay in sync.
            size_t p = 0, sz = buf.size();
            while (p < sz) {
                uint32_t w0 = buf[p];
                if (pp_is_bulkcore(w0)) {
                    if (p + 1 >= sz) {
                        break;
                    }
                    uint32_t core = pp_bulkcore_core(w0), rawn = buf[p + 1];
                    uint32_t prefix = 2u + (uint32_t)NRISC;
                    if (prefix & 1u) {
                        prefix++;
                    }
                    if (p + prefix + rawn > sz) {
                        break;
                    }
                    const uint32_t* meta = &buf[p + 2];
                    const uint32_t* raw = &buf[p + prefix];
                    for (uint32_t r = 0; r < (uint32_t)NRISC; r++) {
                        uint32_t head_mod = pp_bulk_head(meta[r]), run = pp_bulk_run(meta[r]);
                        uint32_t lane = core * (uint32_t)NRISC + r;
                        const uint32_t* ring = raw + (size_t)r * RING_CAP;
                        // worker ring holds variable-length packets too (1-word TIMER, 2-word marker/PROG);
                        // SRC never appears here (reader-injected). Walk by length.
                        uint32_t i = 0;
                        while (i < run) {
                            uint32_t rw0 = ring[(head_mod + i) % RING_CAP];
                            if (pp_is_timer(rw0)) {
                                if (lane < NL) {
                                    cur_hi[lane] = pp_timer_hi(rw0);
                                }
                                i += 1;
                                continue;
                            }
                            if (i + 1 >= run) {
                                break;
                            }
                            emit(lane, rw0, ring[(head_mod + i + 1) % RING_CAP]);
                            i += 2;
                        }
                    }
                    p += prefix + rawn;
                } else if (pp_is_src(w0)) {  // 1 word: sets the current lane
                    cur_lane = pp_src_lane(w0);
                    p += 1;
                } else if (pp_is_timer(w0)) {  // 1 word: refresh the current lane's wall-clock high half
                    if (cur_lane < NL) {
                        cur_hi[cur_lane] = pp_timer_hi(w0);
                    }
                    p += 1;
                } else {  // 2 words: PROG / marker (emit resolves)
                    if (p + 1 >= sz) {
                        break;
                    }
                    emit(cur_lane, w0, buf[p + 1]);
                    p += 2;
                }
            }
            if (socket_mode && p < sz) {
                // trailing partial packet (socket pages aren't packet-aligned) -> carry to the next read
                resid.assign(buf.begin() + (std::ptrdiff_t)p, buf.end());
            }
        }
        if (!batch.empty()) {
            mq.push(std::move(batch));
        }
        uint64_t unbal = 0;
        for (uint32_t L = 0; L < NL; L++) {
            if (depth[L] != 0) {
                unbal++;  // a lane whose START/END did not balance = a dropped/torn marker
            }
        }
        fl_mk[h] = mk;
        fl_start[h] = starts;
        fl_end[h] = ends;
        fl_prog_ok[h] = prog_ok;
        fl_ts_bad[h] = ts_bad;
        fl_unbal[h] = unbal;
        fl_stall[h] = stall;
    };
    // Consumer: pop record batches, do the "sink" work (a real profiler emits Tracy zones here). We do a
    // representative touch of every record so the compiler can't elide it and the cost is real-ish.
    auto consumer = [&]() {
        Batch b;
        uint64_t cnt = 0, sink = 0;
        while (mq.pop(b)) {
            for (auto& r : b) {
                sink += r.ts ^ ((uint64_t)r.lane << 32) ^ ((uint64_t)r.zone << 16) ^ r.type ^ r.prog;
                for (int d = 0; d < cwork; d++) {  // simulate per-record emit cost
                    __asm__ volatile("" ::: "memory");
                }
                cnt++;
            }
        }
        consumed.fetch_add(cnt);
        sink_total.fetch_add(sink);
    };

    // ---- CSV consumer (--csv): hold decoded records in memory and write them to CSV in large batches AFTER
    // the drain. The hot-path callback is a bulk vector-append (a memcpy of the whole batch) -- no formatting,
    // no I/O -- so it never back-pressures the X280 (like the sink). Pre-reserved so no realloc mid-run. A
    // SINGLE consumer owns the buffer (no lock). The CSV formatting + file write happen post-run (see below).
    // Append records into ONE pre-reserved contiguous buffer. Counterintuitively this beats moving whole
    // batch buffers: the popped batch is FREED after the copy so the allocator recycles it for the flusher's
    // next batch (bounded churn), whereas retaining batch buffers forces the flushers -- on the critical drain
    // path -- to malloc fresh each time (more pressure -> more stalls). Reserved up front so no realloc mid-run.
    std::vector<Rec> csv_recs;
    if (do_csv) {
        csv_recs.reserve((size_t)NL * ((size_t)nmarkers + 16) * 2);  // kernel markers + slack (FW/stall/2 words)
    }
    auto csv_consumer = [&]() {
        Batch b;
        uint64_t cnt = 0;
        while (mq.pop(b)) {
            csv_recs.insert(csv_recs.end(), b.begin(), b.end());  // copy into the pre-reserved buffer; batch freed
            cnt += b.size();
        }
        consumed.fetch_add(cnt);
    };

#if defined(TRACY_ENABLE)
    // ---- Tracy consumer (--tracy): feed decoded records into the EXISTING RealtimeProfilerTracyHandler so
    // the zones visualize. A SINGLE consumer -- per-lane START/END must be pushed in emission order (Tracy
    // nests by arrival), which the M stateless sink consumers would scramble. Contexts are pre-created here,
    // before draining (creation is ~ms; keep it off the hot path). ----
    std::unique_ptr<tt::tt_metal::RealtimeProfilerTracyHandler> tracy_handler;
    std::unordered_map<uint32_t, std::string> zone_names;  // hash -> name (stable storage backing name views)
    // Tracy's TT "frequency" is device cycles per NANOSECOND (GHz), NOT Hz. The handler does
    // gpuTime = round(ts / frequency) to turn the raw timestamp into ns-ticks (the context period is
    // 1 ns/tick). get_device_aiclk() is MHz -> /1000 = GHz (~1.0). Passing Hz (1e9) collapsed every marker's
    // ns to integer SECONDS -> all zones stacked on one tick with 0 duration. This is the Tensix wall-clock
    // rate (markers come from RISCV_DEBUG_REG_WALL_CLOCK); matches realtime_profiler_manager's sync_frequency.
    double tracy_freq = cluster.get_device_aiclk(device_id) / 1000.0;
    if (tracy_freq <= 0.0) {
        tracy_freq = 1.0;
    }
    if (do_tracy) {
        printf("[tracy] device aiclk = %.4f GHz (cyc/ns)\n", tracy_freq);
        tracy_handler = std::make_unique<tt::tt_metal::RealtimeProfilerTracyHandler>();
        // host_start must be in TRACY's clock domain (tracy::Profiler::GetTime()), NOT system_clock epoch --
        // otherwise the device zones land on a bogus multi-hour timeline. Provisional here; the first marker
        // re-anchors (CalibrateDevice) so device-first-ts maps to tracy-now with exact relative spacing.
        tracy_handler->AddDevice((uint32_t)device_id, tracy::Profiler::GetTime(), 0.0, tracy_freq);
        std::vector<std::pair<uint32_t, uint32_t>> worker_noc0;
        worker_noc0.reserve(num_cores);
        for (uint32_t c = 0; c < num_cores; c++) {
            worker_noc0.emplace_back(noc0x[c], noc0y[c]);
        }
        tracy_handler->PreCreateContexts((uint32_t)device_id, worker_noc0);
        for (auto& [h, md] : loadZoneSourceLocationsHashesReadOnly()) {  // recorded zone hash -> name
            zone_names[h] = md.marker_name;
        }
        zone_names[0x7FFFu] = "X280-STALL";  // PROFILER_STALL_ZONE_ID
        printf(
            "[tracy] handler up: %zu cores pre-created, %zu zone names loaded\n", (size_t)num_cores, zone_names.size());
    }
    auto tracy_consumer = [&]() {
        Batch b;
        uint64_t ts_base = 0;   // first device timestamp seen; markers are rebased to it so the device
        bool anchored = false;  // timeline starts at the capture origin (the context anchor's gpuTime=0),
        uint64_t cnt = 0;       // not ~device-ts (~seconds) into the trace. Durations are unaffected.
        while (mq.pop(b)) {
            for (auto& r : b) {
                uint32_t ci = r.lane / (uint32_t)NRISC, risc = r.lane % (uint32_t)NRISC;
                if (ci >= num_cores) {
                    continue;
                }
                if (!anchored) {
                    ts_base = r.ts;   // rebase origin. NOTE: no CalibrateDevice -- GPU drift-calibration is
                    anchored = true;  // intentionally off (it scaled durations ~9x); the AddDevice anchor
                }  // (gpuTime=0 <-> host_start) + this rebase give correct placement.
                auto it = zone_names.find(r.zone);
                if (it == zone_names.end()) {  // unnamed hash -> stable fallback string
                    char nb[24];
                    std::snprintf(nb, sizeof(nb), "Zone_0x%x", r.zone);
                    it = zone_names.emplace(r.zone, nb).first;
                }
                tt::tt_metal::experimental::WorkerZonePacket zp{};
                zp.chip_id = (uint32_t)device_id;
                zp.core_virtual_x = (uint32_t)vc[ci].x;
                zp.core_virtual_y = (uint32_t)vc[ci].y;
                zp.core_noc0_x = noc0x[ci];
                zp.core_noc0_y = noc0y[ci];
                zp.risc = risc;
                zp.timer_id = r.zone;
                zp.name = it->second;
                zp.timestamp = (r.ts >= ts_base) ? (r.ts - ts_base) : 0;  // rebased to capture origin (clamp)
                zp.is_start = (r.type == PP_ZONE_START);
                tracy_handler->HandleWorkerZone(zp);
                cnt++;
            }
        }
        consumed.fetch_add(cnt);
    };
#endif

    std::vector<std::thread> flushers, mconsumers;  // --mpmc threads

    // One host thread PER ring (default) -> ring 1 is never ack-starved by ring 0's servicing. --rrconsumer
    // falls back to a single thread round-robining all rings (the original behavior) for A/B comparison.
    std::vector<std::thread> consumers;
    if (mpmc > 0) {
        for (uint64_t h = 0; h < ndh; h++) {
            flushers.emplace_back(flusher, h);
        }
        if (do_pin) {
            for (uint64_t h = 0; h < ndh; h++) {
                pin_thread_to_core(flushers[h], (int)h);  // flusher h -> core h (0,1,...); leave rest to the OS
            }
            printf(
                "[pin] pinned %llu flusher(s) to cores 0..%llu (run under taskset -c 2-N to keep them empty)\n",
                (unsigned long long)ndh,
                (unsigned long long)(ndh - 1));
        }
        bool special_spawned = false;
        const char* consumer_desc = nullptr;
        if (do_csv) {
            mconsumers.emplace_back(csv_consumer);  // ONE consumer: bulk-append records to memory (write at end)
            special_spawned = true;
            consumer_desc = "1 CSV consumer (in-memory -> batched CSV at end)";
        }
#if defined(TRACY_ENABLE)
        if (!special_spawned && do_tracy) {
            mconsumers.emplace_back(tracy_consumer);  // ONE ordered consumer -> Tracy zones
            special_spawned = true;
            consumer_desc = "1 Tracy consumer";
        }
#endif
        if (!special_spawned) {
            if (do_tracy) {
                printf("[tracy] build lacks TRACY_ENABLE -- falling back to sink consumers (no zones emitted)\n");
            }
            for (int i = 0; i < mpmc; i++) {
                mconsumers.emplace_back(consumer);
            }
        }
        printf(
            "[mpmc] %llu flush+demux -> record MPMC -> %s\n",
            (unsigned long long)ndh,
            special_spawned ? consumer_desc : (std::to_string(mpmc) + " consumers").c_str());
    } else if (nodrain) {
        printf("[nodrain] host consumer DISABLED, relay ignores flow control -- diagnostic (lossy)\n");
    } else if (rr_consumer) {
        // one thread interleaves all rings in a single sweep loop (original behavior)
        consumers.emplace_back([&]() {
            std::vector<uint32_t> acked(ndh, 0);
            int empty = 0;
            auto start = std::chrono::steady_clock::now();
            auto next_log = start + std::chrono::seconds(1);
            for (;;) {
                auto now = std::chrono::steady_clock::now();
                if (now > next_log) {
                    printf(
                        "  [consumer] total=%llu done=%d\n",
                        (unsigned long long)total_words.load(),
                        (int)producers_done.load());
                    next_log = now + std::chrono::seconds(1);
                }
                if (now - start > std::chrono::seconds(30)) {
                    printf("  [consumer] WALL TIMEOUT\n");
                    break;
                }
                bool any = false;
                for (uint64_t h = 0; h < ndh; h++) {
                    uint64_t hoff = data_off + h * ring_stride;
                    uint32_t hsent;
                    auto ta = std::chrono::steady_clock::now();
                    cluster.read_sysmem(reinterpret_cast<uint8_t*>(&hsent), 4, sent_off(h), device_id, 0);
                    h_us_hsent_v[h] += us(ta, std::chrono::steady_clock::now());
                    if (hsent == acked[h]) {
                        continue;
                    }
                    any = true;
                    uint32_t avail = hsent - acked[h];
                    if (avail > hring_words) {
                        overflow.fetch_add(1);
                        acked[h] = hsent;
                        drv.write_block(reinterpret_cast<uint8_t*>(&acked[h]), 4, HACKED_ADDR_H(h));
                        continue;
                    }
                    uint32_t drain = avail & ~1u;
                    auto& cap = caps[h];
                    size_t base = cap.size();
                    cap.resize(base + drain);
                    uint32_t st = acked[h] % hring_words;
                    auto tr = std::chrono::steady_clock::now();
                    if (st + drain <= hring_words) {
                        cluster.read_sysmem(
                            reinterpret_cast<uint8_t*>(&cap[base]), drain * 4, hoff + (uint64_t)st * 4, device_id, 0);
                    } else {
                        uint32_t first = hring_words - st;
                        cluster.read_sysmem(
                            reinterpret_cast<uint8_t*>(&cap[base]), first * 4, hoff + (uint64_t)st * 4, device_id, 0);
                        cluster.read_sysmem(
                            reinterpret_cast<uint8_t*>(&cap[base + first]), (drain - first) * 4, hoff, device_id, 0);
                    }
                    h_us_ring_v[h] += us(tr, std::chrono::steady_clock::now());
                    h_polls_v[h]++;
                    acked[h] += drain;
                    total_words.fetch_add(drain);
                    auto th = std::chrono::steady_clock::now();
                    drv.write_block(reinterpret_cast<uint8_t*>(&acked[h]), 4, HACKED_ADDR_H(h));
                    h_us_hacked_v[h] += us(th, std::chrono::steady_clock::now());
                }
                if (!any) {
                    if (producers_done.load() && ++empty >= 2500) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                } else {
                    empty = 0;
                }
            }
        });
    } else {
        for (uint64_t h = 0; h < ndh; h++) {
            consumers.emplace_back(drain_ring, h);
        }
    }

    // ---- dispatch a REAL kernel_profiler workload ----
    // Each active RISC runs realprof_{dm,compute}.cpp: MARKER_COUNT DeviceZoneScopedN zones, WORK_SIZE nops per
    // zone (the rate knob). BRISC also pushes the runtime host-id (prog_id) as a STICKY_PROG via arg0. This
    // needs the SPSC profiler backend active -- run with TT_METAL_DEVICE_PROFILER=1 (emits -DPROFILE_KERNEL
    // markers) and TT_METAL_NO_RT_PROFILER=1 (so the built-in RT profiler does not also grab the X280; we
    // boot + drive profzone.bin above). The drainers were spawned before this dispatch, so the X280 is already
    // draining -- a full ring blocks the producing RISC (lossless), and the concurrent host drain unblocks it.
    auto mk_defs = [&]() {
        std::map<std::string, std::string> d;
        d["MARKER_COUNT"] = std::to_string(nmarkers) + "u";
        d["WORK_SIZE"] = std::to_string(prod_delay) + "u";
        return d;
    };
    const std::string kdir = "tt_metal/programming_examples/profiler/test_x280_realprof/kernels/";
    Program program = CreateProgram();
    CoreRange all_cores(CoreCoord{(uint32_t)cx0, (uint32_t)cy0}, CoreCoord{(uint32_t)cx1, (uint32_t)cy1});
    auto defs = mk_defs();
    auto brisc = CreateKernel(
        program,
        kdir + "realprof_dm.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defs});
    SetRuntimeArgs(program, brisc, all_cores, {prog_id});  // host pushes the runtime host-id -> BRISC STICKY_PROG
    if (active_riscs >= 2) {
        CreateKernel(
            program,
            kdir + "realprof_dm.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = defs});
    }
    if (active_riscs >= NRISC) {
        CreateKernel(program, kdir + "realprof_compute.cpp", all_cores, ComputeConfig{.defines = defs});
    }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    printf(
        "[run] dispatching REAL kernel (%llu zones/lane, work=%u, prog_id=0x%x)...\n",
        (unsigned long long)nmarkers,
        prod_delay,
        prog_id);
    distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), wl, /*blocking=*/true);
    printf("[run] kernel done; draining remainder\n");

    drv.lim_wr_u64(P_STOP, 1);
    producers_done.store(true);
    // Wait for the device FW to return to idle -- the drainers run CONCURRENTLY, so the relay can finish only
    // once the host has drained its ring. When this returns, ALL data is in the host rings (sent is final);
    // device_done lets the drainers exit at hsent==acked (authoritative, no premature time-based quit under
    // heavy back-pressure).
    if (!drv.wait_active_fw_returned(std::chrono::seconds(120))) {
        fprintf(stderr, "[run] pipeline did not return to idle (unexpected)\n");
    }
    device_done.store(true);
    if (socket_mode) {
        // Reliable host-side check (bypasses the flaky device RES readback): read the socket FIFO + bytes_sent
        // straight from the hugepage channel. FIFO nonzero => relay wrote marker data to the right addr;
        // bytes_sent nonzero => relay published it. Isolates relay-write vs bytes_sent-publish vs host-read.
        for (uint64_t h = 0; h < ndh; h++) {
            uint32_t cfgw[16] = {0};
            drv.read_block(cfgw, 64, 0x08019000ull + h * 0x100ull);
            uint32_t foff = cfgw[4];  // dn_fifo_lo (channel offset)
            uint32_t fw[8] = {0}, bs = 0;
            cluster.read_sysmem(reinterpret_cast<uint8_t*>(fw), 32, foff, device_id, 0);
            cluster.read_sysmem(reinterpret_cast<uint8_t*>(&bs), 4, foff + (uint32_t)hring_bytes, device_id, 0);
            printf(
                "[socket%llu POST-RUN] fifo_off=0x%x fifo[0..3]=%08x %08x %08x %08x  bytes_sent=%u\n",
                (unsigned long long)h,
                foff,
                fw[0],
                fw[1],
                fw[2],
                fw[3],
                bs);
        }
    }
    if (mpmc > 0) {
        for (auto& t : flushers) {
            t.join();  // flushers finish draining first
        }
        mq.close();  // then signal consumers no more batches are coming
        for (auto& t : mconsumers) {
            t.join();
        }
        if (do_csv) {
            // POST-drain CSV write, off the hot path: format rows into a reused buffer and fwrite in ~1 MB
            // chunks (the "large batches"). No back-pressure risk -- the device is already idle here.
            auto t0 = std::chrono::steady_clock::now();
            FILE* f = std::fopen(csv_path.c_str(), "w");
            if (!f) {
                fprintf(stderr, "[csv] cannot open %s\n", csv_path.c_str());
            } else {
                std::fputs("lane,core,risc,type,zone_hash,timestamp,prog\n", f);
                std::string buf;
                buf.reserve(1u << 20);
                char line[96];
                size_t total = 0;
                for (const auto& r : csv_recs) {
                    int n = std::snprintf(
                        line,
                        sizeof(line),
                        "%u,%u,%u,%u,0x%x,%llu,0x%x\n",
                        r.lane,
                        r.lane / (uint32_t)NRISC,
                        r.lane % (uint32_t)NRISC,
                        r.type,
                        r.zone,
                        (unsigned long long)r.ts,
                        r.prog);
                    buf.append(line, (size_t)n);
                    if (buf.size() >= (1u << 20)) {
                        std::fwrite(buf.data(), 1, buf.size(), f);
                        buf.clear();
                    }
                    total++;
                }
                if (!buf.empty()) {
                    std::fwrite(buf.data(), 1, buf.size(), f);
                }
                std::fclose(f);
                uint64_t ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
                printf(
                    "[csv] wrote %zu records to %s in %llums (post-drain, batched)\n",
                    total,
                    csv_path.c_str(),
                    (unsigned long long)ms);
            }
        }
#if defined(TRACY_ENABLE)
        if (do_tracy && tracy_handler) {
            tracy_handler->RemoveDevice((uint32_t)device_id);  // orphan-end summary + destroy contexts
            tracy_handler.reset();
            // The final std::_Exit is abrupt (skips atexit) -- give the in-process Tracy client a moment to
            // ship the queued zones to a connected tracy-capture/GUI before we bail.
            printf("[tracy] flushing zones to Tracy client (3s)...\n");
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
#endif
    } else {
        for (auto& c : consumers) {
            c.join();
        }
    }
    // sum the per-ring host op timers for the report
    uint64_t h_polls = 0, h_us_hsent = 0, h_us_ring = 0, h_us_hacked = 0;
    for (uint64_t h = 0; h < ndh; h++) {
        h_polls += h_polls_v[h];
        h_us_hsent += h_us_hsent_v[h];
        h_us_ring += h_us_ring_v[h];
        h_us_hacked += h_us_hacked_v[h];
    }

    // The real kernel_profiler does NOT write the synthetic per-lane stall-stats region (0x2C00); it records
    // back-pressure in-band as X280-STALL zones (PROFILER_STALL_ZONE_ID) in the marker stream instead. So
    // leave these zeroed -- the stall summary below reports 0 and the stalls surface as decoded zones.
    std::vector<uint32_t> st_events(NL, 0);
    std::vector<uint64_t> st_spins(NL, 0);

    // ---- pipeline profile: where does each X280 hart spend its cycles? ----
    printf("\n--- pipeline profile (X280 cycles; copy%% = fraction of wall spent moving data) ---\n");
    uint64_t hmax = direct ? (ndrain - 1) : (nread + nrelay - 1);  // direct: drain harts; split: readers + relay(s)
    for (uint64_t h = 0; h <= hmax; h++) {
        std::vector<uint8_t> rs(0x40);
        drv.read_block(rs.data(), 0x40, 0x08011040ULL + h * 0x40);
        const uint64_t* v = reinterpret_cast<const uint64_t*>(rs.data());
        uint64_t bytes = v[0], t_copy = v[1], t_total = v[2], aux1 = v[4], aux2 = v[5], breach = v[6], nbulk = v[7];
        uint64_t words = bytes / 4 + 1;
        double busy = t_total ? 100.0 * (double)t_copy / (double)t_total : 0.0;
        if (direct) {
            printf(
                "  drain%llu: %6llu KB  wall=%lluM  copy=%5.1f%%  %.1f cyc/word  host-wait=%lluM cyc\n",
                (unsigned long long)h,
                (unsigned long long)(bytes / 1024),
                (unsigned long long)(t_total / 1000000),
                busy,
                (double)t_copy / (double)words,
                (unsigned long long)(aux1 / 1000000));
        } else if (h >= nread) {
            printf(
                "  relay%llu: %6llu KB  wall=%lluM  copy=%5.1f%%  %.1f cyc/word  hostfull=%llu idle=%llu "
                "OVERWRITE=%llu\n",
                (unsigned long long)(h - nread),
                (unsigned long long)(bytes / 1024),
                (unsigned long long)(t_total / 1000000),
                busy,
                (double)t_copy / (double)words,
                (unsigned long long)aux1,
                (unsigned long long)aux2,
                (unsigned long long)breach);
            printf(
                "    [relay%llu cfg] fifo_addr=0x%llx bsent_addr=0x%llx\n",
                (unsigned long long)(h - nread),
                (unsigned long long)breach,
                (unsigned long long)nbulk);
        } else {
            uint64_t visits = aux2, polls = breach;  // reader: v[5]=visits (drains), v[6]=polls, v[7]=bulk cores
            printf(
                "  reader%llu: %6llu KB  wall=%lluM  copy=%5.1f%%  %.1f cyc/word  spsc-wait=%lluM  "
                "visits=%lluk avg-run=%llu  polls=%lluk (%.0f%% empty)  bulk=%llu cores\n",
                (unsigned long long)h,
                (unsigned long long)(bytes / 1024),
                (unsigned long long)(t_total / 1000000),
                busy,
                (double)t_copy / (double)words,
                (unsigned long long)(aux1 / 1000000),
                (unsigned long long)(visits / 1000),
                (unsigned long long)(visits ? words / visits : 0),
                (unsigned long long)(polls / 1000),
                polls ? 100.0 * (double)(polls - visits) / (double)polls : 0.0,
                (unsigned long long)nbulk);
        }
    }
    printf(
        "  host  : %llu polls  ring-read=%llums (%llu us/poll)  hsent-read=%llums  hacked-write=%llums\n",
        (unsigned long long)h_polls,
        (unsigned long long)(h_us_ring / 1000),
        (unsigned long long)(h_us_ring / (h_polls + 1)),
        (unsigned long long)(h_us_hsent / 1000),
        (unsigned long long)(h_us_hacked / 1000));

    uint32_t active_lanes = num_cores * active_riscs;  // shared by the mpmc + offline summaries below
    bool run_ok = false;                               // shared pass/fail for the exit code
    if (mpmc > 0) {
        // ---- --mpmc pipeline verify: the flushers already demuxed+decoded the real markers into device
        // records (they own their half's lanes, in order) and tallied per-lane checks; the consumers drained
        // the record MPMC. Real markers carry no synthetic seq, so "lossless" here means: every ZONE_START had
        // its ZONE_END (balanced per lane), each active lane produced at least its kernel zones, the prog id
        // propagated, and timestamps are monotonic per lane. ----
        uint64_t mk = 0, starts = 0, ends = 0, prog_ok = 0, ts_bad = 0, unbal = 0, stall = 0;
        for (uint64_t h = 0; h < ndh; h++) {
            mk += fl_mk[h];
            starts += fl_start[h];
            ends += fl_end[h];
            prog_ok += fl_prog_ok[h];
            ts_bad += fl_ts_bad[h];
            unbal += fl_unbal[h];
            stall += fl_stall[h];
        }
        // each DeviceZoneScopedN = 1 START + 1 END, so >= 2*nmarkers markers/active-lane from the kernel
        // (FW main zone + any X280-STALL zones add more -- a lower bound, not an equality). Tolerances: the
        // FW main zone's END may land after our P_STOP (an unclosed zone per active lane), and the first FW
        // markers precede the BRISC PROG sticky (prog=0), so allow up to one open zone per lane and require
        // prog only on the kernel-zone markers.
        uint64_t kernel_min = 2ull * (uint64_t)active_lanes * nmarkers;
        bool ok = (mk >= kernel_min) && (ts_bad == 0) && (consumed.load() == mk) && (starts >= ends) &&
                  ((starts - ends) <= (uint64_t)active_lanes) && (prog_ok >= kernel_min);
        run_ok = ok;
        printf(
            "\n=== X280 REAL profiler pipeline (--mpmc: %llu flush+demux -> record MPMC -> %d consumers) ===\n",
            (unsigned long long)ndh,
            mpmc);
        printf(
            "  markers      : %llu decoded / %llu consumed  (>= %llu kernel zones*2)%s\n",
            (unsigned long long)mk,
            (unsigned long long)consumed.load(),
            (unsigned long long)kernel_min,
            ok ? "  OK" : "  *** LOSS ***");
        if (socket_mode) {
            uint64_t tp = 0;
            for (uint64_t h = 0; h < ndh; h++) {
                tp += fl_pages[h];
            }
            printf(
                "  socket pages : %llu total read by host (%llu + %llu per socket) -- >0 => relay IS delivering "
                "to the FIFO\n",
                (unsigned long long)tp,
                (unsigned long long)fl_pages[0],
                (unsigned long long)(ndh > 1 ? fl_pages[1] : 0));
        }
        printf(
            "  zones        : %llu START / %llu END   unbalanced lanes: %llu\n",
            (unsigned long long)starts,
            (unsigned long long)ends,
            (unsigned long long)unbal);
        printf(
            "  X280-STALL   : %llu back-pressure zones  (delay=%u/marker)  [0 = drain fully kept up]\n",
            (unsigned long long)stall,
            prod_delay);
        printf(
            "  MPMC (mq)    : peak %zu / cap %zu batches   flusher push-blocks: %llu  [0 blocks => MPMC never the "
            "bottleneck]\n",
            mq.peak,
            mq.cap,
            (unsigned long long)mq.push_blocks);
        printf(
            "  prog id      : %llu/%llu markers carried 0x%x   ts regressions: %llu   ring overflow: %llu  (sink "
            "%llx)\n",
            (unsigned long long)prog_ok,
            (unsigned long long)mk,
            prog_id,
            (unsigned long long)ts_bad,
            (unsigned long long)overflow.load(),
            (unsigned long long)sink_total.load());
    } else {
        // ---- offline demux: each drain hart's stream is independent (its own sticky-src sequence over a
        // disjoint lane slice), so walk each capture separately and bind markers to the current STICKY-SRC ----
        size_t cap_words = 0;
        auto demux_t0 = std::chrono::steady_clock::now();
        for (auto& cap : caps) {
            cap_words += cap.size();
            uint32_t cur_lane = 0xFFFFFFFF;
            size_t p = 0, sz = cap.size();
            while (p + 1 < sz) {
                uint32_t w0 = cap[p];
                if (pp_is_bulkcore(w0)) {
                    // RAW-BULK frame: [hdr 2][NRISC meta][pad?][rawn RAW]. Split per-risc using the meta, taking
                    // only the valid [head_mod, head_mod+run) circular slice of each ring block (ignore over-read).
                    uint32_t core = pp_bulkcore_core(w0);
                    uint32_t rawn = cap[p + 1];
                    uint32_t prefix = 2u + (uint32_t)NRISC;
                    if (prefix & 1u) {
                        prefix++;
                    }
                    if (p + prefix + rawn > sz) {
                        break;  // truncated (shouldn't happen)
                    }
                    const uint32_t* meta = &cap[p + 2];
                    const uint32_t* raw = &cap[p + prefix];
                    for (uint32_t r = 0; r < (uint32_t)NRISC; r++) {
                        uint32_t head_mod = pp_bulk_head(meta[r]), run = pp_bulk_run(meta[r]);
                        uint32_t lane = core * (uint32_t)NRISC + r;
                        if (run == 0 || lane >= NL) {
                            continue;
                        }
                        const uint32_t* ring = raw + (size_t)r * RING_CAP;
                        uint32_t first = (head_mod + run > RING_CAP) ? (RING_CAP - head_mod) : run;  // wrap split
                        accum[lane].insert(accum[lane].end(), ring + head_mod, ring + head_mod + first);
                        if (first < run) {
                            accum[lane].insert(accum[lane].end(), ring, ring + (run - first));
                        }
                    }
                    p += prefix + rawn;
                } else if (pp_is_src(w0)) {
                    cur_lane = pp_src_lane(w0);
                    p += 2;
                } else {
                    if (cur_lane < NL) {
                        accum[cur_lane].push_back(w0);
                        accum[cur_lane].push_back(cap[p + 1]);
                    }
                    p += 2;
                }
            }
        }
        uint64_t demux_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - demux_t0)
                .count();
        double demux_gbs = demux_us ? (double)cap_words * 4.0 / 1e3 / (double)demux_us : 0.0;
        printf(
            "  [capture] %zu raw words across %llu ring(s); OFFLINE demux %llu ms = %.2f GB/s (1 thread, post-run)\n",
            cap_words,
            (unsigned long long)ndh,
            (unsigned long long)(demux_us / 1000),
            demux_gbs);

        // ---- verify each DEMUXED lane is complete + gap-free ----
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
        if (direct) {
            printf(
                "\n=== X280 linearized profiler (%llu direct drain hart(s), %llu host ring(s), sticky-src) ===\n",
                (unsigned long long)ndrain,
                (unsigned long long)ndh);
        } else {
            printf("\n=== X280 linearized profiler (2 readers + 1 relay + single host ring, sticky-src) ===\n");
        }
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
        run_ok = (ok_lanes == active_lanes);
    }  // end offline (non-mpmc) demux+verify
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
    std::_Exit(run_ok ? 0 : 1);
}
