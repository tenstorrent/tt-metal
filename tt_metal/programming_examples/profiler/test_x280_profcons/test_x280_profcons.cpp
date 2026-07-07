// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 TRUE CONSUMER for the SPSC kernel profiler (kernel_profiler.hpp). The X280
// runs continuously, draining every per-RISC L1 ring, advancing each ring's head
// (so producing RISCs unblock), and relaying drained markers to host pinned mem.
//
// The test: boot the consumer FIRST, then launch a workload that emits MORE
// markers than a ring holds. Without a consumer the producers would block forever
// (deadlock); with the consumer draining, the workload COMPLETES and every word is
// accounted for (drained == produced). That completion is the flow-control proof.
//
// Build:  make -C tools/x280_bm                               (profcons.bin)
//         cmake --build build_Release --target test_x280_profcons
// Run:    TT_METAL_DEVICE_PROFILER=1 \
//         ./build_Release/programming_examples/profiler/test_x280_profcons --loop 150
//
// Flags: --bin <profcons.bin> --device N --l2cpu N --pll MHZ --loop N --slice WORDS

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
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
#include <llrt/hal.hpp>
#include "tools/profiler/x280_driver.hpp"  // shared X280 boot driver (also used by RealtimeProfilerManager)
#include <umd/device/types/core_coordinates.hpp>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#include <tracy/TracyTTDevice.hpp>
#endif

using tt::Cluster;
using tt::CoreType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::MetalContext;
using tt::tt_metal::distributed::MeshDevice;

static constexpr uint64_t LIM_BASE = 0x08000000ULL;
static constexpr uint64_t RESET_UNIT_BASE = 0x80030000ULL;
static constexpr uint64_t L2CPU_RESET_REG = RESET_UNIT_BASE + 0x14;
static constexpr uint64_t L2CPU_REG_BASE = 0xFFFFF7FEFFF10000ULL;
static constexpr uint64_t PLL4_BASE = 0x80020500ULL;
static constexpr uint64_t PLL_CNTL_1 = PLL4_BASE + 0x4;
static constexpr uint64_t PLL_CNTL_5 = PLL4_BASE + 0x14;

static constexpr uint64_t MBOX_PARAMS = 0x08011000ULL;
static constexpr uint64_t MBOX_RESULTS = 0x08011040ULL;
static constexpr uint64_t MBOX_COORDS = 0x08011200ULL;
static constexpr uint64_t P_STOP = MBOX_PARAMS + 0x28;
static constexpr uint64_t BENCH_CFG = 0x08011600ULL;  // +0x00 words/ring/pass, +0x08 passes (0 = normal)
static constexpr uint64_t DONE_MAGIC = 0xC0570FFEE1ULL;
static constexpr uint64_t FOOTER_MAGIC = 0xC05D09F11E12345ULL;
static constexpr uint64_t WIN_STRIDE = 0x200000ULL;
static constexpr int NRISC = 5;
static constexpr uint32_t PROF_CTRL_WORDS = 32;  // control_vector length

static CoreCoord l2cpu_tile(int idx) {
    switch (idx) {
        case 1: return {8, 9};
        case 2: return {8, 5};
        case 3: return {8, 7};
        default: return {8, 3};
    }
}
static const CoreCoord ARC_TILE{8, 0};

struct PllSolution {
    int fbdiv;
    int postdiv[4];
};
static bool pll_solution(int mhz, PllSolution& out) {
    switch (mhz) {
        case 200: out = {128, {15, 15, 15, 15}}; return true;
        case 800: out = {64, {1, 1, 1, 1}}; return true;
        case 1000: out = {80, {1, 1, 1, 1}}; return true;
        case 1750: out = {140, {1, 1, 1, 1}}; return true;
        default: return false;
    }
}

class X280 {
public:
    X280(Cluster& cluster, int chip, int l2cpu) : cluster_(cluster), chip_(chip), l2cpu_(l2cpu) {
        l2_ = virt(l2cpu_tile(l2cpu_));
        arc_ = virt(ARC_TILE);
    }
    uint32_t reg_rd(const tt_cxy_pair& t, uint64_t a) const {
        uint32_t v = 0;
        cluster_.read_reg(&v, t, a);
        return v;
    }
    void reg_wr(const tt_cxy_pair& t, uint64_t a, uint32_t v) const { cluster_.write_reg(&v, t, a); }
    uint64_t lim_rd_u64(uint64_t a) const {
        uint64_t v = 0;
        cluster_.read_core(&v, sizeof(v), l2_, a);
        return v;
    }
    void lim_wr_u64(uint64_t a, uint64_t v) const { cluster_.write_core(&v, sizeof(v), l2_, a); }
    void assert_reset() const {
        uint32_t v = reg_rd(arc_, L2CPU_RESET_REG);
        reg_wr(arc_, L2CPU_RESET_REG, v & ~(1u << (l2cpu_ + 4)));
    }
    void release_reset() const {
        uint32_t v = reg_rd(arc_, L2CPU_RESET_REG);
        reg_wr(arc_, L2CPU_RESET_REG, v | (1u << (l2cpu_ + 4)));
        (void)reg_rd(arc_, L2CPU_RESET_REG);
    }
    void load_lim(const std::vector<uint8_t>& bin) const {
        cluster_.write_core(bin.data(), static_cast<uint32_t>(bin.size()), l2_, LIM_BASE);
    }
    void write_block(const std::vector<uint8_t>& block, uint64_t addr) const {
        cluster_.write_core(block.data(), static_cast<uint32_t>(block.size()), l2_, addr);
    }
    void set_reset_vectors(uint64_t entry) const {
        uint32_t lo = static_cast<uint32_t>(entry & 0xFFFFFFFF);
        uint32_t hi = static_cast<uint32_t>(entry >> 32);
        for (int core = 0; core < 4; core++) {
            reg_wr(l2_, L2CPU_REG_BASE + core * 8, lo);
            reg_wr(l2_, L2CPU_REG_BASE + core * 8 + 4, hi);
        }
    }
    void set_pll(int mhz) const {
        PllSolution sol;
        if (!pll_solution(mhz, sol)) {
            throw std::runtime_error("no PLL solution for " + std::to_string(mhz) + " MHz");
        }
        uint32_t c5 = reg_rd(arc_, PLL_CNTL_5);
        uint8_t pd[4] = {
            uint8_t(c5 & 0xFF), uint8_t((c5 >> 8) & 0xFF), uint8_t((c5 >> 16) & 0xFF), uint8_t((c5 >> 24) & 0xFF)};
        uint32_t c1 = reg_rd(arc_, PLL_CNTL_1);
        uint32_t c1_low = c1 & 0x0000FFFF;
        uint16_t fb = uint16_t((c1 >> 16) & 0xFFFF);
        auto write_c5 = [&]() {
            uint32_t v = pd[0] | (uint32_t(pd[1]) << 8) | (uint32_t(pd[2]) << 16) | (uint32_t(pd[3]) << 24);
            reg_wr(arc_, PLL_CNTL_5, v);
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };
        auto write_c1 = [&]() {
            reg_wr(arc_, PLL_CNTL_1, c1_low | (uint32_t(fb) << 16));
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };
        for (int i = 0; i < 4; i++) {
            while (pd[i] < sol.postdiv[i]) {
                pd[i]++;
                write_c5();
            }
        }
        while (fb != sol.fbdiv) {
            fb += (sol.fbdiv > fb) ? 1 : -1;
            write_c1();
        }
        for (int i = 0; i < 4; i++) {
            while (pd[i] > sol.postdiv[i]) {
                pd[i]--;
                write_c5();
            }
        }
    }
    tt_cxy_pair l2() const { return l2_; }

private:
    tt_cxy_pair virt(CoreCoord phys) const {
        CoreCoord v = cluster_.get_virtual_coordinate_from_physical_coordinates(chip_, phys);
        return tt_cxy_pair(chip_, v);
    }
    Cluster& cluster_;
    int chip_;
    int l2cpu_;
    tt_cxy_pair l2_;
    tt_cxy_pair arc_;
};

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open firmware: " + path);
    }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (data.size() % 4) {
        data.push_back(0);
    }
    return data;
}

template <typename T>
static void pack(std::vector<uint8_t>& buf, size_t off, T val) {
    std::memcpy(buf.data() + off, &val, sizeof(T));
}

int main(int argc, char** argv) {
    using namespace tt::tt_metal;
    std::string bin_path = "tools/x280_bm/build/profcons.bin";
    int device_id = 0, l2cpu = 0, pll = 1000, loop = 150;
    uint64_t slice_words = 768, nharts = 2;  // fastest read setup: 2 harts x ILP 4
    int bench = 0;                           // --bench N: isolated drain benchmark, N passes (no producers)
    int bench_ro = 0;                        // --bench-ro: read-only (skip relay) to isolate read vs relay
    int emit_tracy = 0;                      // --tracy: after the run, push relayed markers to Tracy zones
    double dev_ghz = 1.35;                   // --freq: device wall-clock GHz for cycle->ns (Blackhole ~1.35)
    int no_reset = 0;                        // --no-reset: skip tt-smi -r, boot X280 via L2CPU reset only
    int derisk_socket = 0;                   // --derisk-socket: can a D2HSocket target the X280 L2CPU as sender?
    int socktest = 0;                        // --socktest: X280 pushes pages through a real D2H socket; measure BW
    uint64_t sock_npages = 200000;           // --npages: pages for the socktest bench (64 B each)
    uint64_t sock_batch = 16;                // --batch: pages pushed per notify (amortizes reserve+notify)
    int sock_zones = 0;                      // --sockzones: push device-zone pages + emit them to Tracy
    int realzones = 0;                       // --realzones: real workload -> profzone pairs -> socket -> Tracy
    int primeecc = 0;                        // --primeecc: one-time fresh-board L3 LIM ECC prime (then tt-smi -r)
    int wayprobe = 0;  // --wayprobe: does the L2CPU reset clear L3 WayEnable? (metal-only prime feasibility)
    int eccprobe = 0;  // --eccprobe: read-only dump of L3 Cache Controller ECC state (detector, no injection)
    int eccinject = 0;      // --eccinject: STEP 2 safe single-bit (correctable) ECC inject; proves induce->detect
    uint32_t inj_bit = 0;   // --injbit: which data bit index to toggle for --eccinject (default 0)
    int inj_double = 0;     // --injdouble: attempt a persistent UNCORRECTABLE (2-bit) error at --injaddr
    uint64_t inj_addr = 0;  // --injaddr: target LIM line for the inject (default LIM_BASE+0x40000)
    int eccread = 0;        // --eccread: read --injaddr N times, report L3 fix/fail counter deltas (persistence/scan)
    int eccpoke = 0;        // --eccpoke: boot a probe FW that reads --injaddr; does the hart HALT (uncorrectable)?
    int eccscrub = 0;       // --eccscrub: probe FW — does the X280 have Zicboz cbo.zero (no-read line-zero)?
    int dmaprime = 0;       // --dmaprime: bring complex up w/ harts parked so the DMAC is accessible (unlock)
    int dmascrub = 0;       // --dmascrub: host-program the L2CPU PDMA to zero-fill LIM (no WayEnable/reset)
    uint64_t dma_src = 0x0A000000ULL;  // --dmasrc: PDMA source (zero device); default the safe-zero region
    uint64_t dma_dst = 0x08050000ULL;  // --dmadst: PDMA dest in LIM; default a scratch word inside the primed region
    uint64_t dma_bytes = 64ULL;        // --dmabytes: bytes to copy (default 1 line; must be x32; <=131040 single-block)

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--bin") {
            bin_path = next();
        } else if (a == "--device") {
            device_id = std::stoi(next());
        } else if (a == "--l2cpu") {
            l2cpu = std::stoi(next());
        } else if (a == "--pll") {
            pll = std::stoi(next());
        } else if (a == "--loop") {
            loop = std::stoi(next());
        } else if (a == "--slice") {
            slice_words = std::stoull(next());
        } else if (a == "--nharts") {
            nharts = std::stoull(next());
        } else if (a == "--bench") {
            bench = std::stoi(next());
        } else if (a == "--bench-ro") {
            bench_ro = 1;
        } else if (a == "--tracy") {
            emit_tracy = 1;
        } else if (a == "--freq") {
            dev_ghz = std::stod(next());
        } else if (a == "--no-reset") {
            no_reset = 1;
        } else if (a == "--derisk-socket") {
            derisk_socket = 1;
        } else if (a == "--socktest") {
            socktest = 1;
        } else if (a == "--npages") {
            sock_npages = std::stoull(next());
        } else if (a == "--batch") {
            sock_batch = std::stoull(next());
        } else if (a == "--sockzones") {
            socktest = 1;
            sock_zones = 1;
            if (sock_npages == 200000) {
                sock_npages = 2000;  // fewer for the zone demo
            }
        } else if (a == "--realzones") {
            realzones = 1;
        } else if (a == "--primeecc") {
            primeecc = 1;
        } else if (a == "--wayprobe") {
            wayprobe = 1;
        } else if (a == "--eccprobe") {
            eccprobe = 1;
        } else if (a == "--eccinject") {
            eccinject = 1;
        } else if (a == "--injbit") {
            inj_bit = static_cast<uint32_t>(std::stoul(next()));
        } else if (a == "--eccread") {
            eccread = 1;
        } else if (a == "--eccpoke") {
            eccpoke = 1;
        } else if (a == "--injdouble") {
            inj_double = 1;
        } else if (a == "--injaddr") {
            inj_addr = std::stoull(next(), nullptr, 0);
        } else if (a == "--eccscrub") {
            eccscrub = 1;
        } else if (a == "--dmaprime") {
            dmaprime = 1;
        } else if (a == "--dmascrub") {
            dmascrub = 1;
        } else if (a == "--dmasrc") {
            dma_src = std::stoull(next(), nullptr, 0);
        } else if (a == "--dmadst") {
            dma_dst = std::stoull(next(), nullptr, 0);
        } else if (a == "--dmabytes") {
            dma_bytes = std::stoull(next(), nullptr, 0);
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }
    if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
        fprintf(stderr, "[warn] TT_METAL_DEVICE_PROFILER not set -- kernels won't emit markers.\n");
    }

    std::vector<uint8_t> bin;
    if (!primeecc && !wayprobe && !eccprobe && !eccinject && !dmascrub && !dmaprime && !eccscrub && !eccread &&
        !eccpoke) {  // reg/DMA modes read their own or no fw
        bin = read_file(bin_path);
        printf("[fw] %s (%zu bytes)\n", bin_path.c_str(), bin.size());
    }
    if (!no_reset) {
        std::string cmd = "tt-smi -r " + std::to_string(device_id);
        printf("[boot] %s\n", cmd.c_str());
        if (std::system(cmd.c_str()) != 0) {
            fprintf(stderr, "tt-smi reset failed\n");
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
    } else {
        printf("[boot] --no-reset: booting X280 via L2CPU reset only (device-init scenario)\n");
    }

    auto mesh = MeshDevice::create_unit_mesh(device_id);
    Cluster& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();

    if (wayprobe) {
        // Feasibility probe for a metal-only ECC prime: is the L3 WayEnable (set by --primeecc, the
        // thing that "needs an ASIC reset") cleared by the *L2CPU* reset that metal already does in
        // its X280 boot? If yes, metal could scrub + L2CPU-reset with no external tt-smi -r.
        X280 x(cluster, device_id, l2cpu);
        tt_cxy_pair l2 = x.l2();
        constexpr uint64_t L3_WAYENABLE = 0x02010008ULL;
        uint32_t before = x.reg_rd(l2, L3_WAYENABLE);
        x.reg_wr(l2, L3_WAYENABLE, 0xF);
        uint32_t after_set = x.reg_rd(l2, L3_WAYENABLE);
        x.assert_reset();
        x.release_reset();
        uint32_t after_l2cpu_reset = x.reg_rd(l2, L3_WAYENABLE);
        printf(
            "[wayprobe] WayEnable: before=0x%x after_set=0x%x after_L2CPU_reset=0x%x  => L2CPU reset %s WayEnable "
            "(so a metal-only prime is %s; a full tt-smi -r would restore it either way)\n",
            before,
            after_set,
            after_l2cpu_reset,
            (after_l2cpu_reset != after_set) ? "CLEARS/CHANGES" : "does NOT clear",
            (after_l2cpu_reset != after_set) ? "PLAUSIBLE" : "NOT viable via L2CPU reset");
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (eccprobe) {
        // Read-only dump of the L3 Cache Controller ECC state (base 0x02010000, from the SiFive X280
        // MC manual §14.3/§14.4). Two purposes: (1) validate our NoC register path — L3 Config MUST
        // read 0x06091004 (Banks=4, Ways=16, lgSets=9, lgBlockBytes=6); (2) report the data-array ECC
        // counters, our non-intrusive detector for LIM (L3 data SRAM) ECC health. Reads only — no
        // injection, no writes. NOTE: run with --no-reset to leave the current X280 state undisturbed.
        // Reading a *Count register clears its ECC interrupt (manual §14.4.4/5) but not device memory.
        X280 x(cluster, device_id, l2cpu);
        tt_cxy_pair l2 = x.l2();
        constexpr uint64_t L3_BASE = 0x02010000ULL;
        auto rd = [&](uint64_t off) { return x.reg_rd(l2, L3_BASE + off); };
        uint32_t cfg = rd(0x000);       // Config
        uint32_t wayen = rd(0x008);     // WayEnable (0 => LIM mode; >0 => that many ways are cache)
        uint32_t dir_fix = rd(0x108);   // DirECCFixCount  (correctable metadata)
        uint32_t dir_fail = rd(0x128);  // DirECCFailCount (uncorrectable metadata)
        uint32_t dat_fix = rd(0x148);   // DatECCFixCount  (correctable data)
        uint32_t dat_fail_lo = rd(0x160), dat_fail_hi = rd(0x164);
        uint32_t dat_fail = rd(0x168);  // DatECCFailCount (UNCORRECTABLE data == the fresh-board fault)
        const bool base_ok = (cfg == 0x06091004u);
        printf(
            "[eccprobe] L3 Config = 0x%08x  Banks=%u Ways=%u lgSets=%u lgBlockBytes=%u  %s\n",
            cfg,
            cfg & 0xff,
            (cfg >> 8) & 0xff,
            (cfg >> 16) & 0xff,
            (cfg >> 24) & 0xff,
            base_ok ? "[OK: NoC reg path + L3 base 0x02010000 confirmed]"
                    : "[!! expected 0x06091004 -- base/NoC path wrong, ignore counters below]");
        printf("[eccprobe] WayEnable = 0x%x  (%s)\n", wayen, wayen == 0 ? "LIM mode" : "some ways enabled as cache");
        printf(
            "[eccprobe] L3 DATA array: DatECCFixCount(corr)=%u  DatECCFailCount(UNCORR)=%u  lastFailAddr=0x%08x%08x\n",
            dat_fix,
            dat_fail,
            dat_fail_hi,
            dat_fail_lo);
        printf("[eccprobe] L3 METADATA:   DirECCFixCount=%u  DirECCFailCount=%u\n", dir_fix, dir_fail);
        printf(
            "[eccprobe] => %s\n",
            (dat_fail == 0 && dir_fail == 0) ? "no uncorrectable ECC errors observed (LIM ECC healthy)"
                                             : "UNCORRECTABLE ECC errors present!");
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (eccpoke) {
        // Boot a probe FW from the primed low LIM, then have the hart read --injaddr. If that line is
        // uncorrectable (unprimed/defective), the read faults and the hart halts: STATUS stays BOOTED and
        // never reaches SURVIVED. This is the deterministic "no heartbeat" bad state we can then heal.
        X280 x(cluster, device_id, l2cpu);
        uint64_t target = inj_addr ? inj_addr : (LIM_BASE + 0x100000ULL);
        auto bin = read_file("tools/x280_bm/build/eccpoke.bin");
        constexpr uint64_t A_STATUS = 0x08010000ULL, A_RESULT = 0x08010010ULL, A_PARAM = 0x08010020ULL,
                           A_MCAUSE = 0x0800FFE0ULL;
        printf(
            "[eccpoke] probe FW %zu bytes; boot LIM 0x%llx, hart reads target=0x%llx\n",
            bin.size(),
            (unsigned long long)LIM_BASE,
            (unsigned long long)target);
        x.assert_reset();
        x.load_lim(bin);
        x.lim_wr_u64(A_PARAM, target);  // param AFTER load so the FW image doesn't clobber it
        x.set_reset_vectors(LIM_BASE);
        x.release_reset();
        uint64_t status = 0;
        for (int i = 0; i < 200; i++) {
            status = x.lim_rd_u64(A_STATUS);
            if (status == 0xEC00000002ULL) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        uint64_t result = x.lim_rd_u64(A_RESULT), mcause = x.lim_rd_u64(A_MCAUSE);
        printf(
            "[eccpoke] status=0x%llx result=0x%llx trap_mcause=0x%llx\n",
            (unsigned long long)status,
            (unsigned long long)result,
            (unsigned long long)mcause);
        printf(
            "[eccpoke] => %s\n",
            (status == 0xEC00000002ULL)
                ? "SURVIVED -- the read did not fault (target line is not uncorrectable)"
                : "HALTED at the read -- uncorrectable ECC fault reproduced (this is the bad state)");
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (eccread) {
        // Read a target LIM line N times and report L3 fix/fail counter deltas. Two uses: (1) after a warm
        // reset, does a previously-injected defect still tick counters (did it SURVIVE the reset)? (2) scan
        // high LIM addresses to find genuinely-unprimed regions (uncorrectable on first read).
        X280 x(cluster, device_id, l2cpu);
        tt_cxy_pair l2 = x.l2();
        constexpr uint64_t L3_BASE = 0x02010000ULL;
        auto rd = [&](uint64_t off) { return x.reg_rd(l2, L3_BASE + off); };
        uint64_t target = inj_addr ? inj_addr : LIM_BASE;
        uint32_t fix0 = rd(0x148), fail0 = rd(0x168);
        volatile uint64_t sink = 0;
        uint64_t first = x.lim_rd_u64(target);
        for (int i = 0; i < 16; i++) {
            sink += x.lim_rd_u64(target);
        }
        (void)sink;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        uint32_t fix1 = rd(0x148), fail1 = rd(0x168);
        printf(
            "[eccread] target=0x%llx readback=0x%llx  DatECCFixCount %u->%u (+%u)  DatECCFailCount %u->%u (+%u)\n",
            (unsigned long long)target,
            (unsigned long long)first,
            fix0,
            fix1,
            fix1 - fix0,
            fail0,
            fail1,
            fail1 - fail0);
        printf(
            "[eccread] => %s\n",
            (fail1 > fail0) ? "UNCORRECTABLE on read (unprimed or 2-bit defect present)"
            : (fix1 > fix0) ? "CORRECTABLE ticks (a 1-bit defect is present at this line)"
                            : "clean (no ECC events -- line is primed and error-free)");
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (eccinject && inj_double) {
        // STEP 3 (induce the "unprimed-like" bad state): try to plant a PERSISTENT UNCORRECTABLE (2-bit)
        // ECC error at a chosen LIM line, and measure whether it (a) registers as uncorrectable
        // (DatECCFailCount++) and (b) survives — i.e. whether the injector can stand in for genuinely
        // unprimed LIM in an end-to-end Option-A test. Strategy: arm a bit + write the line (store a 1-bit
        // error), then arm a second bit + write again, then read it back several times.
        X280 x(cluster, device_id, l2cpu);
        tt_cxy_pair l2 = x.l2();
        constexpr uint64_t L3_BASE = 0x02010000ULL;
        auto rd = [&](uint64_t off) { return x.reg_rd(l2, L3_BASE + off); };
        uint64_t target = inj_addr ? inj_addr : (LIM_BASE + 0x40000ULL);
        uint32_t fix0 = rd(0x148), fail0 = rd(0x168);
        // Two store-time toggles on the same line, different bits, to try to accumulate 2 bits in SRAM.
        x.reg_wr(l2, L3_BASE + 0x040, (inj_bit & 0xffu));  // arm bit A (data)
        x.lim_wr_u64(target, 0xA5A5A5A5A5A5A5A5ULL);
        x.reg_wr(l2, L3_BASE + 0x040, ((inj_bit + 1) & 0xffu));  // arm bit B (data)
        x.lim_wr_u64(target, 0xA5A5A5A5A5A5A5A5ULL);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // Read it back a few times (each read is ECC-checked).
        volatile uint64_t sink = 0;
        for (int i = 0; i < 8; i++) {
            sink += x.lim_rd_u64(target);
        }
        (void)sink;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        uint32_t fix1 = rd(0x148), fail1 = rd(0x168);
        uint32_t faillo = rd(0x160), failhi = rd(0x164);
        uint64_t readback = x.lim_rd_u64(target);
        printf(
            "[injdouble] target=0x%llx  DatECCFixCount %u->%u  DatECCFailCount %u->%u  failAddr=0x%x%08x  "
            "readback=0x%llx\n",
            (unsigned long long)target,
            fix0,
            fix1,
            fail0,
            fail1,
            failhi,
            faillo,
            (unsigned long long)readback);
        printf(
            "[injdouble] => %s\n",
            (fail1 > fail0) ? "UNCORRECTABLE induced (DatECCFailCount ticked) -- viable bad state; next: check it "
                              "survives a warm reset"
            : (fix1 > fix0) ? "only CORRECTABLE (fix ticked, fail did not) -- injector can't make a persistent 2-bit "
                              "line; need unprimed-region induction"
                            : "no counter change -- toggle did not land on these host accesses");
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (eccinject) {
        // STEP 2 (SAFE): inject a SINGLE-bit ECC error into the L3 data array and confirm it registers
        // as a *correctable* event (DatECCFixCount ticks up) with NO halt and NO reset. This proves the
        // induce->detect loop end-to-end before we go double-bit (step 3, which halts the tile).
        //
        // ECCInjectError @ 0x02010040: [7:0]=ECCToggleBit, bit16=ECCToggleType (0=data, 1=directory).
        // The toggle fires on the "next cache operation" — so we leave the X280 running (mesh init booted
        // the RT drainer, whose harts continuously touch LIM) to generate one, and also issue a few host
        // reads of LIM as a backup. A single flipped bit is always correctable, so the drainer keeps
        // running and no data is lost.
        X280 x(cluster, device_id, l2cpu);
        tt_cxy_pair l2 = x.l2();
        constexpr uint64_t L3_BASE = 0x02010000ULL;
        auto rd = [&](uint64_t off) { return x.reg_rd(l2, L3_BASE + off); };
        uint32_t fix_before = rd(0x148);
        uint32_t fail_before = rd(0x168);
        // Program the injector: toggle one data bit on the next cache operation.
        const uint32_t inj = (inj_bit & 0xffu) | (0u << 16);  // ECCToggleType=0 => data array
        x.reg_wr(l2, L3_BASE + 0x040, inj);
        // Generate cache operations to trigger the toggle (host reads + let the running drainer tick).
        volatile uint64_t sink = 0;
        for (int i = 0; i < 16; i++) {
            sink += x.lim_rd_u64(LIM_BASE + (uint64_t)(i * 64));
        }
        (void)sink;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        uint32_t fix_after = rd(0x148);
        uint32_t fail_after = rd(0x168);
        printf(
            "[eccinject] single-bit (bit %u, data): DatECCFixCount %u -> %u   DatECCFailCount %u -> %u\n",
            inj_bit,
            fix_before,
            fix_after,
            fail_before,
            fail_after);
        printf(
            "[eccinject] => %s\n",
            (fix_after > fix_before)
                ? "CORRECTABLE error induced + detected -- induce/detect loop PROVEN (no halt, no reset)"
                : "no counter change: the toggle did not fire on a host access -- 'next cache op' likely "
                  "needs a hart-side access (retry with the drainer definitely running / target a hart-touched line)");
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (eccscrub) {
        // PROBE: does the X280 implement a no-read line-zero (Zicboz `cbo.zero`)? That is the clean
        // primitive for a hart to stamp ECC into UNPRIMED LIM without a read-modify-write fault. Boots
        // the probe FW from PRIMED LIM (safe), runs cbo.zero on a separate primed line, reports outcome.
        X280 x(cluster, device_id, l2cpu);
        auto bin = read_file("tools/x280_bm/build/eccscrub.bin");
        printf("[eccscrub] probe FW %zu bytes; booting from LIM 0x%llx\n", bin.size(), (unsigned long long)LIM_BASE);
        x.assert_reset();
        x.load_lim(bin);
        x.set_reset_vectors(LIM_BASE);
        x.release_reset();
        constexpr uint64_t A_STATUS = 0x08010000ULL, A_READBACK = 0x08010008ULL, A_DONE = 0x08010010ULL,
                           A_EFFECT = 0x08010018ULL, A_MCAUSE = 0x0800FFE0ULL;
        uint64_t done = 0, status = 0;
        for (int i = 0; i < 300; i++) {
            done = x.lim_rd_u64(A_DONE);
            status = x.lim_rd_u64(A_STATUS);
            if (done == 0xEC000000FFULL) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        uint64_t rb = x.lim_rd_u64(A_READBACK), eff = x.lim_rd_u64(A_EFFECT);
        uint64_t mcause = x.lim_rd_u64(A_MCAUSE), mepc = x.lim_rd_u64(A_MCAUSE + 8),
                 mtval = x.lim_rd_u64(A_MCAUSE + 16);
        printf(
            "[eccscrub] status=0x%llx done=0x%llx readback=0x%llx effect=%llu\n",
            (unsigned long long)status,
            (unsigned long long)done,
            (unsigned long long)rb,
            (unsigned long long)eff);
        printf(
            "[eccscrub] trap-diag: mcause=0x%llx mepc=0x%llx mtval=0x%llx\n",
            (unsigned long long)mcause,
            (unsigned long long)mepc,
            (unsigned long long)mtval);
        const char* verdict;
        if (done == 0xEC000000FFULL) {
            verdict = eff == 1 ? "cbo.zero WORKS (line zeroed, no trap) -> clean hart-scrub primitive available"
                               : "cbo.zero decoded as NO-OP (survived, line unchanged) -> not usable as-is";
        } else if (status == 0xEC00000002ULL) {
            verdict =
                "TRAPPED at cbo.zero (mcause=2 => illegal) -> X280 has NO Zicboz; hart-scrub needs uncached-store path";
        } else {
            verdict = "inconclusive -- FW did not reach DONE (LIM boot failed or hung before the probe)";
        }
        printf("[eccscrub] => %s\n", verdict);
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (dmaprime) {
        // UNLOCK STEP: bring the L2CPU complex OUT of reset (so the DMAC becomes accessible) WITHOUT any
        // hart fetching uninitialized LIM. The DMAC reads all-ones while the complex is in reset (proven
        // earlier), so an ECC scrub is impossible until the complex is up. Two independent guards keep the
        // harts from executing garbage on a fresh board:
        //   (1) reset-PC -> a `j .` spin in the general-purpose scratch (x280-phys 0x20010100, non-ECC,
        //       executable per the ISA doc's RNMI-handler note), so even an un-suppressed hart just spins.
        //   (2) suppress-instruction-fetch flags (tt-isa-documentation BlackholeA0/L2CPUTile/MemoryMap.md:
        //       32-bit word at NoC 0xFFFFF7FEFFF10400, per-hart suppress in bits [19:16], "only applicable
        //       when coming out of reset" -> set BEFORE release).
        // Mirrors Umair's x280_boot.cpp release sequence (PLL-low -> release ARC bit l2cpu+4). No DMA writes
        // here -- this only proves the unlock; the scrub is --dmascrub once the DMAC reads sane.
        X280 x(cluster, device_id, l2cpu);
        tt_cxy_pair l2 = x.l2();
        constexpr uint64_t kScratch = 0xFFFFF7FEFFF10100ULL;      // L2CPU_REG_BASE + 0x100 (NoC view of 0x20010100)
        constexpr uint64_t kSuppressReg = 0xFFFFF7FEFFF10400ULL;  // L2CPU_REG_BASE + 0x400
        constexpr uint64_t kScratchPhys = 0x20010100ULL;          // x280-physical pc for the reset vector
        constexpr uint64_t CH0 = 0xFFFFF7FEFFF80000ULL, CTL0 = 0x18;
        x.set_pll(200);                       // README: release at LOW speed
        x.assert_reset();                     // ensure harts in reset before arming
        x.reg_wr(l2, kScratch, 0x0000006fu);  // `j .` (jal x0, 0) -> infinite self-loop
        x.set_reset_vectors(kScratchPhys);    // all 4 harts boot to the scratch spin
        uint32_t sup_before = x.reg_rd(l2, kSuppressReg);
        x.reg_wr(l2, kSuppressReg, (sup_before & 0xFFFFu) | (0xFu << 16));  // suppress fetch, all 4 harts
        uint32_t sup_after = x.reg_rd(l2, kSuppressReg);
        x.release_reset();  // complex up; harts parked (spin + suppressed)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        uint32_t ctl_lo0 = x.reg_rd(l2, CH0 + CTL0), ctl_hi0 = x.reg_rd(l2, CH0 + CTL0 + 4);
        // DECISIVE TEST: is the DMAC clock-gated? Umair's boot notes 0x02010008=0xf "ungates the L2CPU
        // cache-controller clock domain" -- but that's the WayEnable=cache-mode write, irreversible until
        // tt-smi -r. If the DMAC only reads sane AFTER this, then the DMA scrub can't avoid the reset either.
        uint32_t wayen_before = x.reg_rd(l2, 0x02010008ULL);
        x.reg_wr(l2, 0x02010008ULL, 0xF);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        uint32_t ctl_lo = x.reg_rd(l2, CH0 + CTL0), ctl_hi = x.reg_rd(l2, CH0 + CTL0 + 4);
        printf(
            "[dmaprime] DMAC CTL0 pre-ungate = 0x%08x%08x ; WayEnable(0x02010008) was 0x%x\n",
            ctl_hi0,
            ctl_lo0,
            wayen_before);
        bool dmac_up = !(ctl_lo == 0xffffffffu && ctl_hi == 0xffffffffu);
        printf(
            "[dmaprime] suppress 0x%llx: 0x%08x -> 0x%08x (bits19:16=per-hart suppress)\n",
            (unsigned long long)kSuppressReg,
            sup_before,
            sup_after);
        printf(
            "[dmaprime] after PLL-low + reset-vec->scratch-spin + suppress + release: DMAC CTL0=0x%08x%08x\n",
            ctl_hi,
            ctl_lo);
        printf(
            "[dmaprime] => %s\n",
            dmac_up ? "DMAC ACCESSIBLE -- complex is up with harts parked. UNLOCK ACHIEVED (scrub is now possible)."
                    : "DMAC still all-ones -- complex did not come up as expected; inspect before scrubbing.");
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (dmascrub) {
        // STEP 3 (corrected): host-drive the L2CPU **Synopsys DesignWare DMAC** (NOT a SiFive PDMA --
        // my earlier attempt used the wrong register model and faulted the device) to copy zeros from
        // the L3 Zero Device (0x0A000000, "always returns zero on reads") into LIM (0x08000000). The
        // DMAC's write is an internal-master access, so it lays down valid data+ECC (manual §18.1.1's
        // recommended ECC init) without touching WayEnable -> no `tt-smi -r`.
        //
        // Register model + sequence ported from our own working tools/x280_bm/dma_engine.h / dma_probe.c
        // (which already do NoC<->LIM), with two changes for a fully on-tile copy:
        //   - transfer type MEM->MEM (tt_fc=0): DMAC is the flow controller and moves the whole block
        //     autonomously -- NO software handshake (no REQSRC/REQDST). That's what makes it safe to
        //     drive host-side over NoC (no per-burst timing).
        //   - both masters = L2 (on-tile L2 cache port): zero device and LIM are both on-tile, so no
        //     DMA_TLB / NoC-window setup at all.
        // The DMAC is at 0xFFFFF7FEFFF80000 in the NoC/host register view (== 0x2FF80000 hart view);
        // channel-0 offsets (SAR/DAR/LLP/CTL/CFG, INT +0x2C0, MISC +0x398) match dma_engine.h.
        if (dma_bytes % 32 != 0 || dma_bytes == 0 || dma_bytes > 4095ull * 32) {
            fprintf(
                stderr,
                "[dmascrub] --dmabytes must be a nonzero multiple of 32 and <= %u (single block)\n",
                4095u * 32u);
            return 2;
        }
        X280 x(cluster, device_id, l2cpu);
        tt_cxy_pair l2 = x.l2();
        // NOTE: do NOT assert_reset here -- the L2CPU reset gates the whole complex including the DMAC,
        // which makes its registers read back all-ones (inaccessible). The DMAC must stay clocked.
        x.set_pll(1000);  // ensure the L2CPU complex is clocked (ARC-side PLL program; does not reset the complex)
        constexpr uint64_t L3_BASE = 0x02010000ULL;
        constexpr uint64_t CH0 = 0xFFFFF7FEFFF80000ULL;  // Synopsys DW DMAC channel-0 block (host/NoC view)
        constexpr uint64_t SAR0 = 0x00, DAR0 = 0x08, LLP0 = 0x10, CTL0 = 0x18, CFG0 = 0x40;
        constexpr uint64_t INT = 0x2C0, MISC = 0x398;
        constexpr uint64_t RAWTFR = 0x00, RAWERR = 0x20, MASKTFR = 0x50, CLEARTFR = 0x78, CLEARBLK = 0x80,
                           CLEARSRCT = 0x88, CLEARDSTT = 0x90, CLEARERR = 0x98;
        constexpr uint64_t DMACFG = 0x00, CHEN = 0x08;
        auto wr64 = [&](uint64_t off, uint64_t v) {
            x.reg_wr(l2, CH0 + off, (uint32_t)(v & 0xffffffffu));
            x.reg_wr(l2, CH0 + off + 4, (uint32_t)(v >> 32));
        };
        auto rd32 = [&](uint64_t off) { return x.reg_rd(l2, CH0 + off); };

        // Accessibility guard: if the DMAC block reads all-ones it's not reachable/clocked -- bail out
        // BEFORE programming it blindly (which is how we faulted the device earlier).
        uint32_t probe_ctl_lo = rd32(CTL0), probe_ctl_hi = rd32(CTL0 + 4);
        uint32_t probe_cfg_lo = rd32(CFG0);
        printf("[dmascrub] DMAC probe: CTL0=0x%08x%08x CFG0lo=0x%08x\n", probe_ctl_hi, probe_ctl_lo, probe_cfg_lo);
        if (probe_ctl_lo == 0xffffffffu && probe_ctl_hi == 0xffffffffu) {
            printf(
                "[dmascrub] => DMAC reads all-ones (inaccessible). Aborting before programming. "
                "Complex may need to be clocked/out-of-reset first.\n");
            std::fflush(stdout);
            std::_Exit(0);
        }

        const uint64_t block_ts = dma_bytes / 32;  // 32-byte (WORD_32) words
        uint32_t fail_before = x.reg_rd(l2, L3_BASE + 0x168);
        printf(
            "[dmascrub] Synopsys DMAC MEM->MEM: %llu B (block_ts=%llu x32B)  src=0x%llx (zero dev) -> dst=0x%llx "
            "(LIM)\n",
            (unsigned long long)dma_bytes,
            (unsigned long long)block_ts,
            (unsigned long long)dma_src,
            (unsigned long long)dma_dst);

        // --- reset channel: disable+enable DMAC, clear + unmask interrupts ---
        wr64(MISC + DMACFG, 0);
        wr64(MISC + DMACFG, 1);
        for (uint64_t o : {CLEARTFR, CLEARBLK, CLEARSRCT, CLEARDSTT, CLEARERR}) {
            wr64(INT + o, 1);
        }
        for (uint64_t o : {MASKTFR, (uint64_t)0x58, (uint64_t)0x60, (uint64_t)0x68, (uint64_t)0x70}) {
            wr64(INT + o, 0x0101);  // unmask (bit0=mask, bit8=we)
        }

        // --- CTL0: int_en=1, done=1, 32B words, INCR/INCR, msize=BURST_8, tt_fc=0 (MEM->MEM),
        //     sms=dms=L2(1), block_ts ---
        uint64_t ctl = 0;
        ctl |= 1ull << 0;         // int_en
        ctl |= (uint64_t)5 << 1;  // dst_tr_width = WORD_32
        ctl |= (uint64_t)5 << 4;  // src_tr_width = WORD_32
        // dinc (7-8)=INCR=0, sinc (9-10)=INCR=0
        ctl |= (uint64_t)2 << 11;  // dest_msize = BURST_8
        ctl |= (uint64_t)2 << 14;  // src_msize  = BURST_8
        // tt_fc (20-22) = 0  => MEM->MEM, DMAC flow controller
        ctl |= 1ull << 23;      // dms = L2
        ctl |= 1ull << 25;      // sms = L2
        ctl |= block_ts << 32;  // block_ts (bits 32-43)
        ctl |= 1ull << 44;      // done (pre-arm, per init_ctl)
        wr64(CTL0, ctl);

        // --- CFG0: ch_prior=1 (bits 5-7); hs_sel_* left 0 (unused in MEM->MEM) ---
        wr64(CFG0, 1ull << 5);

        wr64(SAR0, dma_src);
        wr64(DAR0, dma_dst);
        wr64(LLP0, 0);
        for (uint64_t o : {CLEARTFR, CLEARBLK, CLEARSRCT, CLEARDSTT, CLEARERR}) {
            wr64(INT + o, 1);
        }

        // --- enable channel 0 (CHEN bit0=ch_en, bit8=we) and poll RAWTFR/RAWERR (no start_burst) ---
        wr64(MISC + CHEN, 0x0101);
        bool done = false, err = false;
        for (int i = 0; i < 20000; i++) {
            if (rd32(INT + RAWERR) & 1) {
                err = true;
                break;
            }
            if (rd32(INT + RAWTFR) & 1) {
                done = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        uint32_t ctl_lo = rd32(CTL0), ctl_hi = rd32(CTL0 + 4);
        uint64_t w0 = x.lim_rd_u64(dma_dst);
        uint64_t w1 = x.lim_rd_u64(dma_dst + 8);
        uint32_t fail_after = x.reg_rd(l2, L3_BASE + 0x168);
        printf(
            "[dmascrub] done=%d err=%d  CTL0=0x%08x%08x (done bit44=%d)  readback dst[0]=0x%016llx dst[1]=0x%016llx  "
            "DatECCFailCount %u -> %u\n",
            done,
            err,
            ctl_hi,
            ctl_lo,
            (ctl_hi >> 12) & 1,
            (unsigned long long)w0,
            (unsigned long long)w1,
            fail_before,
            fail_after);
        printf(
            "[dmascrub] => %s\n",
            (done && !err && w0 == 0 && w1 == 0 && fail_after == fail_before)
                ? "MEM->MEM DMA landed, dst zeroed, no uncorrectable ECC -- host-driven DMA scrub WORKS (no WayEnable, "
                  "no tt-smi -r)"
                : "did not fully verify -- inspect done/err/CTL/readback above");
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (primeecc) {
        // One-time fresh-board L3 LIM ECC prime (mirrors tt-llm-engine loader.prime_lim_ecc /
        // x280_tester.cpp): route L3 through the cache controller so it writes valid data+ECC
        // into the physical SRAM backing LIM. WayEnable=0xF is irreversible until ASIC reset, so
        // LIM is a cache (unusable) until the caller runs `tt-smi -r` AFTER this — the ECC then
        // persists across that reset and LIM_BASE becomes valid SRAM. Run once per power cycle.
        X280 x(cluster, device_id, l2cpu);
        tt_cxy_pair l2 = x.l2();
        constexpr uint64_t L3_WAYENABLE = 0x02010008ULL;     // highest-enabled-way, increase-only
        constexpr uint64_t L3_WAYMASK_BASE = 0x02010800ULL;  // WayMask0..37, stride 8
        constexpr uint32_t L3_NUM_MASTERS = 38;
        constexpr uint64_t ZERO_DEVICE_BASE = 0x0A000000ULL;  // safe-zero source lines
        const uint64_t prime_bytes = 0x60000;                 // FW + mailboxes + profzone stacks/SP
        x.reg_wr(l2, L3_WAYENABLE, 0xF);
        for (uint32_t m = 0; m < L3_NUM_MASTERS; m++) {
            x.reg_wr(l2, L3_WAYMASK_BASE + (uint64_t)m * 8, 0x8000);  // force alloc into Way 15
        }
        for (uint64_t off = 0; off < prime_bytes; off += 64) {
            x.reg_wr(l2, ZERO_DEVICE_BASE + off, 0);  // touch each 64B line -> fetch+merge+writeback ECC
        }
        printf(
            "[primeecc] primed 0x%llx bytes L3 LIM ECC on L2CPU %d (WayEnable=0xF). "
            "NOW run: tt-smi -r %d  (ECC persists across that reset; LIM valid after)\n",
            (unsigned long long)prime_bytes,
            l2cpu,
            device_id);
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (derisk_socket) {
        // DE-RISK: can a tt-metal D2HSocket use the X280 L2CPU as its sender_core, and does
        // the host write the sender_socket_md config into the X280's LIM? Both are built
        // around a Tensix sender; this probes the two suspected mismatches (L2CPU CoreCoord,
        // and the uint32 config address vs LIM's 0x08000000 base).
        using tt::tt_metal::distributed::D2HSocket;
        using tt::tt_metal::distributed::MeshCoordinate;
        using tt::tt_metal::distributed::MeshCoreCoord;
        printf("[derisk] required_config_buffer_size = %u\n", D2HSocket::required_config_buffer_size());
        CoreCoord l2phys = l2cpu_tile(l2cpu);
        CoreCoord l2virt = cluster.get_virtual_coordinate_from_physical_coordinates(device_id, l2phys);
        tt_cxy_pair l2v(device_id, l2virt);
        printf(
            "[derisk] X280 L2CPU phys=(%u,%u) virt=(%u,%u)\n",
            (unsigned)l2phys.x,
            (unsigned)l2phys.y,
            (unsigned)l2virt.x,
            (unsigned)l2virt.y);
        const uint32_t cfg_addr = 0x08019000u;  // LIM: above STAGECTL, below STAGE_BASE
        std::vector<uint8_t> zc(256, 0);
        cluster.write_core(zc.data(), (uint32_t)zc.size(), l2v, cfg_addr);  // zero so we can see a write land
        try {
            MeshCoreCoord sender{MeshCoordinate(0, 0), l2phys};
            D2HSocket sock(
                mesh, sender, 4096u, D2HSocket::ExternalConfigBuffer{.address = cfg_addr, .sender_is_l2cpu = true});
            sock.set_page_size(64);
            printf("[derisk] D2HSocket CONSTRUCTED OK; config_buffer_address=0x%x\n", sock.get_config_buffer_address());
            uint32_t md[7] = {0};
            cluster.read_core(md, sizeof(md), l2v, cfg_addr);
            printf(
                "[derisk] sender_socket_md@LIM: bytes_sent=%u num_downstreams=%u write_ptr=%u "
                "dn_bytes_sent_addr=0x%x dn_fifo_addr=0x%x fifo_total=%u is_d2h=%u\n",
                md[0],
                md[1],
                md[2],
                md[3],
                md[4],
                md[5],
                md[6]);
            printf("[derisk] config landed in LIM: %s\n", (md[5] != 0 || md[4] != 0) ? "YES" : "NO");
        } catch (const std::exception& e) {
            printf("[derisk] D2HSocket construction FAILED: %s\n", e.what());
        }
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (socktest) {
        // STEP 2 BW BENCH: the X280 (profsock.bin) pushes `sock_npages` synthetic 64 B pages
        // through a REAL D2HSocket (sender = X280 L2CPU) to the host FIFO; the host drains via
        // socket.read() (which acks back to the X280 LIM, driving flow control). The FW times its
        // push -> sustained D2H-socket BW, vs the raw-relay ~1.2 GB/s.
        using tt::tt_metal::distributed::D2HSocket;
        using tt::tt_metal::distributed::MeshCoordinate;
        using tt::tt_metal::distributed::MeshCoreCoord;
        auto pcie_cores = cluster.get_soc_desc(device_id).get_cores(tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED);
        auto pc = pcie_cores.front();
        const uint32_t cfg_addr = 0x08019000u;
        const uint64_t SOCK_DONE = 0x5005C0FFEEULL;

        X280 x280(cluster, device_id, l2cpu);
        x280.assert_reset();
        auto bin = read_file("tools/x280_bm/build/profsock.bin");
        x280.load_lim(bin);
        std::vector<uint8_t> params(64, 0), zres(64, 0);
        pack<uint64_t>(params, 0x00, (uint64_t)cfg_addr);
        pack<uint64_t>(params, 0x08, (uint64_t)pc.x);
        pack<uint64_t>(params, 0x10, (uint64_t)pc.y);
        pack<uint64_t>(params, 0x18, sock_npages);
        pack<uint64_t>(params, 0x20, sock_batch);
        pack<uint64_t>(params, 0x28, (uint64_t)sock_zones);  // 0=raw BW pages, 1=device-zone pages
        x280.write_block(params, MBOX_PARAMS);
        x280.write_block(zres, MBOX_RESULTS);

        CoreCoord l2phys = l2cpu_tile(l2cpu);
        MeshCoreCoord sender{MeshCoordinate(0, 0), l2phys};
        D2HSocket sock(
            mesh, sender, 4096u, D2HSocket::ExternalConfigBuffer{.address = cfg_addr, .sender_is_l2cpu = true});
        sock.set_page_size(64);

#if defined(TRACY_ENABLE)
        static const tracy::RiscType kRisc[NRISC] = {
            tracy::RiscType::BRISC,
            tracy::RiscType::NCRISC,
            tracy::RiscType::TRISC_0,
            tracy::RiscType::TRISC_1,
            tracy::RiscType::TRISC_2};
        std::map<std::pair<uint32_t, uint32_t>, TracyTTCtx> ctxs;  // one Tracy context per (core_x,core_y)
        if (sock_zones) {
            printf("[sockzones] waiting up to 30s for tracy-capture to connect...\n");
            std::fflush(stdout);
            for (int w = 0; w < 600 && !tracy::GetProfiler().IsConnected(); w++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
#endif

        x280.set_reset_vectors(LIM_BASE);
        x280.set_pll(pll);
        x280.release_reset();
        printf(
            "[socktest] X280 pushing %llu pages (batch=%llu, zones=%d) through D2H socket; host draining...\n",
            (unsigned long long)sock_npages,
            (unsigned long long)sock_batch,
            sock_zones);

        std::vector<uint32_t> buf(64 * 64 / sizeof(uint32_t));  // up to one FIFO (64 pages, 16 u32 each)
        uint64_t pages_read = 0, zones_emitted = 0;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        bool done = false;
        while (std::chrono::steady_clock::now() < deadline) {
            uint32_t avail = sock.pages_available();
            if (avail > 0) {
                if (avail > 64) {
                    avail = 64;
                }
                sock.read(buf.data(), avail);
                pages_read += avail;
#if defined(TRACY_ENABLE)
                if (sock_zones) {
                    for (uint32_t j = 0; j < avail; j++) {
                        uint32_t* pg = buf.data() + j * 16;  // 64 B page = 16 u32
                        uint64_t st = ((uint64_t)pg[0] << 32) | pg[1];
                        uint64_t en = ((uint64_t)pg[2] << 32) | pg[3];
                        uint32_t cx = pg[4], cy = pg[5], risc = pg[6] % NRISC, tid = pg[7];
                        auto key = std::make_pair(cx, cy);
                        auto it = ctxs.find(key);
                        if (it == ctxs.end()) {
                            TracyTTCtx ctx = TracyTTContext();
                            TracyTTContextPopulate(ctx, 0, 0.0, 1.0);  // synthetic ts already in ns (freq 1.0)
                            it = ctxs.emplace(key, ctx).first;
                        }
                        tracy::TTDeviceMarker m;
                        m.chip_id = (uint64_t)device_id;
                        m.core_x = cx;
                        m.core_y = cy;
                        m.risc = kRisc[risc];
                        m.runtime_host_id = 0;
                        m.marker_name = "x280_socket_zone_" + std::to_string(tid);
                        m.file = "x280_socket";
                        m.line = 0;
                        m.timestamp = st;
                        m.marker_type = tracy::TTDeviceMarkerType::ZONE_START;
                        TracyTTPushStartMarker(it->second, m);
                        m.timestamp = en;
                        m.marker_type = tracy::TTDeviceMarkerType::ZONE_END;
                        TracyTTPushEndMarker(it->second, m);
                        zones_emitted++;
                    }
                }
#endif
            }
            if (x280.lim_rd_u64(MBOX_RESULTS + 0x18) == SOCK_DONE && pages_read >= sock_npages) {
                done = true;
                break;
            }
        }
        uint64_t fw_bytes = x280.lim_rd_u64(MBOX_RESULTS + 0x00);
        uint64_t cyc = x280.lim_rd_u64(MBOX_RESULTS + 0x08);
        double mbps = cyc ? (double)fw_bytes / 1e6 / ((double)cyc / ((double)pll * 1e6)) : 0.0;
        x280.assert_reset();  // clean halt
        printf(
            "[socktest] done=%d pages_read=%llu fw_bytes=%llu cycles=%llu zones=%llu\n",
            done,
            (unsigned long long)pages_read,
            (unsigned long long)fw_bytes,
            (unsigned long long)cyc,
            (unsigned long long)zones_emitted);
        if (sock_zones) {
            printf(
                "[sockzones] emitted %llu device zones to Tracy; draining to capture...\n",
                (unsigned long long)zones_emitted);
            std::fflush(stdout);
            std::this_thread::sleep_for(std::chrono::seconds(3));
            return done ? 0 : 1;  // normal return so TracyClient flushes (use TRACY_NO_EXIT=1)
        }
        printf(
            "[socktest] D2H-SOCKET PUSH BW: %.0f MB/s  (64B pages, per-page notify; vs raw-relay ~1.2 GB/s)\n", mbps);
        std::fflush(stdout);
        std::_Exit(done ? 0 : 1);
    }

    CoreCoord grid = mesh->compute_with_storage_grid_size();
    uint32_t gx = (uint32_t)grid.x, gy = (uint32_t)grid.y;
    uint32_t num_cores = gx * gy;
    uint64_t prof_l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::PROFILER);
    printf(
        "[cfg] %ux%u=%u cores; profiler_msg_t @ L1 0x%lx; slice=%lu words; loop=%d\n",
        gx,
        gy,
        num_cores,
        (unsigned long)prof_l1,
        (unsigned long)slice_words,
        loop);

    // translated coords for every worker core + pre-zero each core's control vector
    std::vector<uint8_t> coords(num_cores * 8, 0);
    std::vector<CoreCoord> vc(num_cores);
    std::vector<uint8_t> zero_ctrl(PROF_CTRL_WORDS * 4, 0);
    for (uint32_t ly = 0; ly < gy; ly++) {
        for (uint32_t lx = 0; lx < gx; lx++) {
            uint32_t idx = ly * gx + lx;
            CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord{lx, ly}, CoreType::WORKER);
            vc[idx] = v;
            pack<uint32_t>(coords, idx * 8 + 0, (uint32_t)v.x);
            pack<uint32_t>(coords, idx * 8 + 4, (uint32_t)v.y);
            cluster.write_core(zero_ctrl.data(), (uint32_t)zero_ctrl.size(), tt_cxy_pair(device_id, v), prof_l1);
        }
    }

    // D2H relay region: slice (c*5+r) at host_base + (c*5+r)*slice_words*4
    auto pcie_cores = cluster.get_soc_desc(device_id).get_cores(tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED);
    auto pc = pcie_cores.front();
    uint32_t pcie_enc = ((uint32_t)pc.x & 0x3f) | (((uint32_t)pc.y & 0x3f) << 6);
    uint64_t pcie_base = cluster.get_pcie_base_addr_from_device(device_id);
    uint64_t chan_sz = cluster.get_host_channel_size(device_id, 0);
    uint64_t region = (uint64_t)num_cores * NRISC * slice_words * 4;
    if (region + nharts * 64 > WIN_STRIDE) {
        fprintf(stderr, "relay region %lu B > one 2MB window; reduce --slice or cores\n", (unsigned long)region);
        return 1;
    }
    uint64_t data_off = (chan_sz / 2) & ~(WIN_STRIDE - 1);
    uint64_t host_base = pcie_base + data_off;
    printf(
        "[d2h] PCIe enc=0x%x host_base=0x%lx region=%lu KB\n",
        pcie_enc,
        (unsigned long)host_base,
        (unsigned long)(region >> 10));
    {
        uint64_t z = 0;
        for (uint64_t h = 0; h < nharts; h++) {
            cluster.write_sysmem(&z, sizeof(z), data_off + region + h * 64, device_id, 0);
        }
    }  // footer slots

#if defined(TRACY_ENABLE)
    if (realzones) {
        // STEP 3 real markers: run a SMALL profiled workload (markers fit the rings, so it
        // completes with no concurrent drainer), then boot profzone to drain+pair the rings
        // on-device and push complete zones through a D2H socket; host emits them to Tracy.
        using tt::tt_metal::distributed::D2HSocket;
        using tt::tt_metal::distributed::MeshCoordinate;
        using tt::tt_metal::distributed::MeshCoreCoord;
        const uint32_t cfg_addr = 0x08019000u;
        const uint64_t ZDONE = 0x20E50FFEE1ULL;
        uint32_t zloop = (loop > 100) ? 30u : (uint32_t)loop;  // keep markers < 512-word ring
        {
            Program program = CreateProgram();
            CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});
            std::map<std::string, std::string> defs = {
                {"LOOP_COUNT", std::to_string(zloop)}, {"LOOP_SIZE", std::to_string(200)}};
            CreateKernel(
                program,
                "tests/tt_metal/tools/profiler/kernels/full_buffer.cpp",
                all_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defs});
            CreateKernel(
                program,
                "tests/tt_metal/tools/profiler/kernels/full_buffer.cpp",
                all_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = defs});
            CreateKernel(
                program,
                "tests/tt_metal/tools/profiler/kernels/full_buffer_compute.cpp",
                all_cores,
                ComputeConfig{.compile_args = {}, .defines = defs});
            distributed::MeshWorkload wl;
            wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
            printf("[realzones] launching workload (loop=%u, fits rings) ...\n", zloop);
            distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), wl, /*blocking=*/true);
            printf("[realzones] workload done; real markers in the rings.\n");
        }
        printf("[realzones] waiting up to 30s for tracy-capture...\n");
        std::fflush(stdout);
        for (int w = 0; w < 600 && !tracy::GetProfiler().IsConnected(); w++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        X280 zx(cluster, device_id, l2cpu);
        zx.assert_reset();
        auto zbin = read_file("tools/x280_bm/build/profzone.bin");
        zx.load_lim(zbin);
        zx.write_block(coords, MBOX_COORDS);
        std::vector<uint8_t> zp(64, 0), zr(64, 0);
        pack<uint64_t>(zp, 0x00, (uint64_t)cfg_addr);
        pack<uint64_t>(zp, 0x08, (uint64_t)pc.x);
        pack<uint64_t>(zp, 0x10, (uint64_t)pc.y);
        pack<uint64_t>(zp, 0x18, prof_l1);
        pack<uint64_t>(zp, 0x20, (uint64_t)num_cores);
        zx.write_block(zp, MBOX_PARAMS);
        zx.write_block(zr, MBOX_RESULTS);
        CoreCoord l2phys = l2cpu_tile(l2cpu);
        MeshCoreCoord sender{MeshCoordinate(0, 0), l2phys};
        D2HSocket sock(
            mesh, sender, 4096u, D2HSocket::ExternalConfigBuffer{.address = cfg_addr, .sender_is_l2cpu = true});
        sock.set_page_size(64);
        zx.set_reset_vectors(LIM_BASE);
        zx.set_pll(pll);
        zx.release_reset();
        zx.lim_wr_u64(MBOX_PARAMS + 0x28, 1);  // P_STOP: rings are static post-workload -> drain then exit
        printf("[realzones] profzone draining+pairing rings -> pushing zones through the socket...\n");
        static const tracy::RiscType kRisc[NRISC] = {
            tracy::RiscType::BRISC,
            tracy::RiscType::NCRISC,
            tracy::RiscType::TRISC_0,
            tracy::RiscType::TRISC_1,
            tracy::RiscType::TRISC_2};
        std::map<std::pair<uint32_t, uint32_t>, TracyTTCtx> ctxs;
        std::vector<uint32_t> buf(64 * 16);
        uint64_t zones = 0;
        bool zdone = false;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while (std::chrono::steady_clock::now() < deadline) {
            uint32_t avail = sock.pages_available();
            if (avail > 0) {
                if (avail > 64) {
                    avail = 64;
                }
                sock.read(buf.data(), avail);
                for (uint32_t j = 0; j < avail; j++) {
                    uint32_t* pgp = buf.data() + j * 16;
                    uint64_t st = ((uint64_t)pgp[0] << 32) | pgp[1];
                    uint64_t en = ((uint64_t)pgp[2] << 32) | pgp[3];
                    uint32_t cx = pgp[4], cy = pgp[5], risc = pgp[6] % NRISC, tid = pgp[7];
                    auto key = std::make_pair(cx, cy);
                    auto it = ctxs.find(key);
                    if (it == ctxs.end()) {
                        TracyTTCtx ctx = TracyTTContext();
                        TracyTTContextPopulate(ctx, 0, 0.0, dev_ghz);  // device cycles -> ns @ dev_ghz
                        it = ctxs.emplace(key, ctx).first;
                    }
                    tracy::TTDeviceMarker m;
                    m.chip_id = (uint64_t)device_id;
                    m.core_x = cx;
                    m.core_y = cy;
                    m.risc = kRisc[risc];
                    m.runtime_host_id = 0;
                    m.marker_name = "x280_kzone_" + std::to_string(tid);
                    m.file = "x280_kernel";
                    m.line = 0;
                    m.timestamp = st;
                    m.marker_type = tracy::TTDeviceMarkerType::ZONE_START;
                    TracyTTPushStartMarker(it->second, m);
                    m.timestamp = en;
                    m.marker_type = tracy::TTDeviceMarkerType::ZONE_END;
                    TracyTTPushEndMarker(it->second, m);
                    zones++;
                }
            }
            bool fwdone = zx.lim_rd_u64(MBOX_RESULTS + 0x18) == ZDONE;
            if (fwdone && avail == 0 && zones >= zx.lim_rd_u64(MBOX_RESULTS + 0x00)) {
                zdone = true;
                break;
            }
        }
        uint64_t total = zx.lim_rd_u64(MBOX_RESULTS + 0x00);
        zx.assert_reset();
        printf(
            "[realzones] profzone paired %llu zones; host emitted %llu to Tracy (done=%d).\n",
            (unsigned long long)total,
            (unsigned long long)zones,
            zdone);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        return zdone ? 0 : 1;
    }
#endif

    // --- boot the consumer FIRST (so it is draining before any producer runs) ---
    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);
    x280.write_block(coords, MBOX_COORDS);
    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, (uint64_t)pcie_enc);
    pack<uint64_t>(params, 0x08, host_base);
    pack<uint64_t>(params, 0x10, prof_l1);
    pack<uint64_t>(params, 0x18, (uint64_t)num_cores);
    pack<uint64_t>(params, 0x20, slice_words);
    pack<uint64_t>(params, 0x28, 0);  // stop = 0
    pack<uint64_t>(params, 0x30, 0xC05ULL);
    pack<uint64_t>(params, 0x38, nharts);
    x280.write_block(params, MBOX_PARAMS);
    // bench config (0 = normal continuous consumer; else N drain passes of a full ring)
    x280.lim_wr_u64(BENCH_CFG + 0x00, bench ? 512ULL : 0ULL);
    x280.lim_wr_u64(BENCH_CFG + 0x08, bench ? (uint64_t)bench : 0ULL);
    x280.lim_wr_u64(BENCH_CFG + 0x10, (uint64_t)bench_ro);
    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();
    printf(
        "[cons] X280 consumer: %llu harts x ILP 4 draining %u cores x %d rings ...\n",
        (unsigned long long)nharts,
        num_cores,
        NRISC);

    // --- BENCH MODE: no producers; the FW times `bench` drain passes of a full ring ---
    if (bench) {
        printf("[bench] %d passes x %u rings x 512 words via fast path (no producers) ...\n", bench, num_cores * NRISC);
        auto bdl = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        bool bd = false;
        while (std::chrono::steady_clock::now() < bdl) {
            bool all = true;
            for (uint64_t h = 0; h < nharts; h++) {
                if (x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x18) != DONE_MAGIC) {
                    all = false;
                    break;
                }
            }
            if (all) {
                bd = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!bd) {
            fprintf(stderr, "[bench] timed out\n");
            std::fflush(stdout);
            std::_Exit(1);
        }
        uint64_t tot_bytes = 0, max_cyc = 0;
        for (uint64_t h = 0; h < nharts; h++) {
            uint64_t b = x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x00);
            uint64_t cyc = x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x08);
            double hmbps = cyc ? (double)b / 1e6 / ((double)cyc / ((double)pll * 1e6)) : 0.0;
            printf(
                "  hart %llu: %llu B in %llu cycles -> %.0f MB/s\n",
                (unsigned long long)h,
                (unsigned long long)b,
                (unsigned long long)cyc,
                hmbps);
            tot_bytes += b;
            if (cyc > max_cyc) {
                max_cyc = cyc;
            }
        }
        double agg = max_cyc ? (double)tot_bytes / 1e6 / ((double)max_cyc / ((double)pll * 1e6)) : 0.0;
        printf("  -------------------------------------------------\n");
        printf(
            "  AGGREGATE drain+relay : %.0f MB/s   (%llu harts x ILP4; gridilp read ref ~1534 MB/s)\n",
            agg,
            (unsigned long long)nharts);
        std::fflush(stdout);
        std::_Exit(0);
    }

    // --- launch a workload that OVERFLOWS the rings (would deadlock w/o consumer) ---
    {
        Program program = CreateProgram();
        CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});
        std::map<std::string, std::string> defs = {
            {"LOOP_COUNT", std::to_string(loop)}, {"LOOP_SIZE", std::to_string(200)}};
        CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/full_buffer.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defs});
        CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/full_buffer.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = defs});
        CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/full_buffer_compute.cpp",
            all_cores,
            ComputeConfig{.compile_args = {}, .defines = defs});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
        printf(
            "[prod] launching workload (loop=%d -> ~%d marker-words/ring, ring cap 512) ...\n", loop, (loop + 4) * 4);
        distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, /*blocking=*/true);
        printf("[prod] workload COMPLETED -- producers never permanently blocked (consumer kept rings drained)\n");
    }

    // --- stop the consumer, wait for ALL harts to finish draining ---
    x280.lim_wr_u64(P_STOP, 1);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool all = true;
        for (uint64_t h = 0; h < nharts; h++) {
            if (x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x18) != DONE_MAGIC) {
                all = false;
                break;
            }
            uint64_t foot = 0;
            cluster.read_sysmem(&foot, sizeof(foot), data_off + region + h * 64, device_id, 0);
            if (foot != FOOTER_MAGIC) {
                all = false;
                break;
            }
        }
        if (all) {
            done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!done) {
        fprintf(stderr, "[cons] timed out waiting for all %llu harts' done + footers\n", (unsigned long long)nharts);
        std::fflush(stdout);
        std::_Exit(1);
    }

    uint64_t total_drained = 0, loops = 0, max_out = 0;
    for (uint64_t h = 0; h < nharts; h++) {
        total_drained += x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x00);
        uint64_t hl = x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x08);
        uint64_t hm = x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x10);
        if (hl > loops) {
            loops = hl;
        }
        if (hm > max_out) {
            max_out = hm;
        }
    }

    // ground truth: sum each ring's final tail (DEVICE_BUFFER_END_INDEX) over all cores
    uint64_t total_produced = 0;
    uint32_t overflow_rings = 0;
    std::vector<uint32_t> tail_per(num_cores * NRISC, 0);  // per-(core,risc) final tail (words relayed)
    std::vector<uint8_t> ctrl(PROF_CTRL_WORDS * 4);
    for (uint32_t c = 0; c < num_cores; c++) {
        cluster.read_core(ctrl.data(), (uint32_t)ctrl.size(), tt_cxy_pair(device_id, vc[c]), prof_l1);
        for (int r = 0; r < NRISC; r++) {
            uint32_t tail;
            std::memcpy(&tail, ctrl.data() + (5 + r) * 4, 4);  // DEVICE_BUFFER_END_INDEX_BR_ER + r
            tail_per[c * NRISC + r] = tail;
            total_produced += tail;
            if (tail > slice_words) {
                overflow_rings++;
            }
        }
    }

    printf("\n=== X280 SPSC consumer ===\n");
    printf("  consumer loops          : %llu\n", (unsigned long long)loops);
    printf(
        "  max outstanding (words) : %llu  (ring cap 512 -> %s)\n",
        (unsigned long long)max_out,
        max_out > 512 ? "OVERRUN!" : "ok, flow-controlled");
    printf("  produced (sum of tails) : %llu words\n", (unsigned long long)total_produced);
    printf("  drained by X280         : %llu words\n", (unsigned long long)total_drained);
    printf("  lossless drain          : %s\n", total_drained == total_produced ? "YES ✓" : "NO <-- MISMATCH");
    printf(
        "  rings exceeding slice    : %u / %u (relay clamped there; drain still complete)\n",
        overflow_rings,
        num_cores * NRISC);
    bool ring_overflowed = max_out > 512 ? false : (total_produced / (num_cores * NRISC) > 256);
    printf(
        "  ring overflow exercised : %s\n",
        ring_overflowed ? "YES (producers relied on consumer)" : "no (workload fit in ring)");
    // Clean teardown: halt the X280 (assert L2CPU reset) so it's left in a re-bootable
    // state for the next --no-reset boot (the device-close hook will do this).
    x280.assert_reset();
    std::fflush(stdout);

#if defined(TRACY_ENABLE)
    if (emit_tracy) {
        // Read every relayed (core,risc) slice back from host sysmem, parse the 2-word
        // markers, and push them to Tracy as device zones (Strategy B: direct emit, like
        // realtime_profiler_tracy_handler). One Tracy context per core; all contexts share
        // a single (cpu=0, gpu=global_min_ts) anchor so cross-core timing stays aligned.
        printf("[tracy] waiting up to 30s for tracy-capture to connect ...\n");
        std::fflush(stdout);
        for (int w = 0; w < 600 && !tracy::GetProfiler().IsConnected(); w++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!tracy::GetProfiler().IsConnected()) {
            fprintf(stderr, "[tracy] no capture connected (start tracy-capture first) — skipping emit.\n");
            return 1;
        }
        static const tracy::RiscType kRisc[NRISC] = {
            tracy::RiscType::BRISC,
            tracy::RiscType::NCRISC,
            tracy::RiscType::TRISC_0,
            tracy::RiscType::TRISC_1,
            tracy::RiscType::TRISC_2};
        // Pass 1: pull all slices into host memory, find the global min timestamp.
        std::vector<std::vector<uint32_t>> slices(num_cores * NRISC);
        uint64_t min_ts = ~0ULL;
        for (uint32_t c = 0; c < num_cores; c++) {
            for (int r = 0; r < NRISC; r++) {
                uint32_t valid = std::min<uint32_t>(tail_per[c * NRISC + r], (uint32_t)slice_words);
                if (valid < 2) {
                    continue;
                }
                auto& s = slices[c * NRISC + r];
                s.resize(valid);
                cluster.read_sysmem(
                    s.data(), valid * 4, data_off + (uint64_t)(c * NRISC + r) * slice_words * 4, device_id, 0);
                for (uint32_t i = 0; i + 1 < valid; i += 2) {
                    if ((s[i] & 0x80000000u) == 0) {
                        continue;
                    }
                    uint64_t ts = ((uint64_t)(s[i] & 0xFFF) << 32) | s[i + 1];
                    if (ts && ts < min_ts) {
                        min_ts = ts;
                    }
                }
            }
        }
        if (min_ts == ~0ULL) {
            min_ts = 0;
        }
        // Stats pass: pair START/END per (core,risc) in ring order (= time order) to PROVE
        // we extracted real zones with sane durations -- headless evidence independent of
        // Tracy GUI rendering (tracy-csvexport only dumps CPU zones).
        {
            uint64_t nstart = 0, nend = 0, npair = 0, sumdur = 0, mindur = ~0ULL, maxdur = 0;
            for (uint32_t cr = 0; cr < num_cores * NRISC; cr++) {
                std::vector<uint64_t> stk;
                auto& s = slices[cr];
                for (uint32_t i = 0; i + 1 < s.size(); i += 2) {
                    uint32_t w0 = s[i];
                    if ((w0 & 0x80000000u) == 0) {
                        continue;
                    }
                    uint32_t ptype = ((w0 >> 12) >> 16) & 0x7;
                    if (ptype > 1) {
                        continue;
                    }
                    uint64_t ts = ((uint64_t)(w0 & 0xFFF) << 32) | s[i + 1];
                    if (ptype == 0) {
                        nstart++;
                        stk.push_back(ts);
                    } else {
                        nend++;
                        if (!stk.empty()) {
                            uint64_t st = stk.back();
                            stk.pop_back();
                            if (ts >= st) {
                                uint64_t d = ts - st;
                                npair++;
                                sumdur += d;
                                mindur = std::min(mindur, d);
                                maxdur = std::max(maxdur, d);
                            }
                        }
                    }
                }
            }
            printf(
                "[tracy] parsed zones: %llu START / %llu END -> %llu matched pairs\n",
                (unsigned long long)nstart,
                (unsigned long long)nend,
                (unsigned long long)npair);
            if (npair) {
                printf(
                    "[tracy] zone duration (cycles->ns @ %.3f GHz): min %.0f / mean %.0f / max %.0f ns\n",
                    dev_ghz,
                    mindur / dev_ghz,
                    (double)sumdur / npair / dev_ghz,
                    maxdur / dev_ghz);
            }
        }

        // Pass 2: emit. ZONE_START(0)/ZONE_END(1) only (2-word markers) for this first cut.
        // Markers MUST be pushed in timestamp order per context, else Tracy drops the
        // out-of-order device zones (mirrors tt-metal's getSortedDeviceMarkersVector). So
        // gather a whole core's markers across all 5 RISCs, sort by timestamp, then emit.
        uint64_t emitted = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            std::vector<tracy::TTDeviceMarker> ms;
            for (int r = 0; r < NRISC; r++) {
                auto& s = slices[c * NRISC + r];
                for (uint32_t i = 0; i + 1 < s.size(); i += 2) {
                    uint32_t w0 = s[i];
                    if ((w0 & 0x80000000u) == 0) {
                        continue;
                    }
                    uint32_t timer_id = (w0 >> 12) & 0x7FFFF;
                    uint32_t ptype = (timer_id >> 16) & 0x7;
                    if (ptype > 1) {
                        continue;  // skip non-zone packets for now
                    }
                    tracy::TTDeviceMarker m;
                    m.chip_id = (uint64_t)device_id;
                    m.core_x = (uint64_t)vc[c].x;
                    m.core_y = (uint64_t)vc[c].y;
                    m.risc = kRisc[r];
                    m.timestamp = ((uint64_t)(w0 & 0xFFF) << 32) | s[i + 1];
                    m.runtime_host_id = 0;
                    m.marker_name = "x280_zone_" + std::to_string(timer_id & 0xFFFF);
                    m.file = "x280_relayed";
                    m.line = 0;
                    m.marker_type =
                        (ptype == 0) ? tracy::TTDeviceMarkerType::ZONE_START : tracy::TTDeviceMarkerType::ZONE_END;
                    ms.push_back(std::move(m));
                }
            }
            if (ms.empty()) {
                continue;
            }
            std::stable_sort(ms.begin(), ms.end(), [](const auto& a, const auto& b) {
                if (a.timestamp != b.timestamp) {
                    return a.timestamp < b.timestamp;
                }
                return a.marker_type == tracy::TTDeviceMarkerType::ZONE_START;  // START before END on ties
            });
            TracyTTCtx ctx = TracyTTContext();
            TracyTTContextPopulate(ctx, 0, (double)min_ts, dev_ghz);
            for (auto& m : ms) {
                if (m.marker_type == tracy::TTDeviceMarkerType::ZONE_START) {
                    TracyTTPushStartMarker(ctx, m);
                } else {
                    TracyTTPushEndMarker(ctx, m);
                }
                emitted++;
            }
        }
        printf(
            "[tracy] emitted %llu zone markers (min_ts=%llu, %.3f GHz); draining to capture ...\n",
            (unsigned long long)emitted,
            (unsigned long long)min_ts,
            dev_ghz);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        // normal return so the TracyClient destructor flushes (use TRACY_NO_EXIT=1 to block
        // until tracy-capture has drained everything).
        return (total_drained == total_produced && max_out <= 512) ? 0 : 1;
    }
#endif

    std::_Exit((total_drained == total_produced && max_out <= 512) ? 0 : 1);
}
