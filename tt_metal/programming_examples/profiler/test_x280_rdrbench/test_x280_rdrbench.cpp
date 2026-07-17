// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 READER-DRAIN microbenchmark. Isolates the round-robin reader (no relay/D2H): 2 reader harts
// each own half the core grid and, per core, POLL the 16-word control region then DRAIN K markers
// from that core's L1 buffer into the reader's OWN streaming 256 KiB LIM SPSC ring. Sweeps K =
// {0,4,8,...,4096,5000} at ILP=1 in ONE boot (K=0 = poll-only lower bound = round-robin overhead)
// and reports drain GB/s + ns/marker per K. No live producer: each core's L1 is pre-filled static.
//
// Build:  make -C tools/x280_bm                               (rdrbench.bin)
//         cmake --build build_Release --target test_x280_rdrbench
// Run:    ./build_Release/programming_examples/profiler/test_x280_rdrbench \
//             --nharts 2 --nrounds 200
//
// Flags: --bin <rdrbench.bin> --device N --l2cpu N --pll MHZ --nharts N(<=4)
//        --nrounds N --no-reset --no-boot

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
static constexpr uint64_t MBOX_COORDS = 0x08011200ULL;
static constexpr uint64_t DONE_MAGIC = 0x5EADE5511BULL;  // matches rdrbench.c

static constexpr uint64_t SRC_L1 = 0x80000ULL;          // worker L1 scratch: ctrl @ +0, markers @ +128
static constexpr uint64_t DST_BASE = 0x08040000ULL;     // LIM sink region (per-hart 128 KiB)
static constexpr uint64_t BENCH_ENTRY = 0x08001000ULL;  // FW links at 0x08001000 (x280-lim.ld); boot there
static constexpr uint64_t MW = 4;                       // marker words (self-describing 4-word marker)
static const uint32_t KLIST[] = {
    0u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 5000u};  // matches rdrbench.c
static constexpr int NK = 13;
static const uint32_t ILPLIST[] = {1u};  // matches rdrbench.c
static constexpr int NILP = 1;
static constexpr uint32_t MAX_K = 5000u;

static constexpr uint64_t BENCH_RES = 0x08013000ULL;  // 2D grid results (dedicated LIM region)
static constexpr uint64_t RES_STRIDE = 0x180ULL;      // NILP*NK u64 cycles + done, per hart
static uint64_t res_slot(int h) { return BENCH_RES + (uint64_t)h * RES_STRIDE; }
static uint64_t res_cell(int h, int ii, int ki) { return res_slot(h) + (uint64_t)(ii * NK + ki) * 8; }
static uint64_t res_done(int h) { return res_slot(h) + (uint64_t)(NILP * NK) * 8; }

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
    void lim_read(void* dst, uint32_t n, uint64_t a) const { cluster_.read_core(dst, n, l2_, a); }
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
    std::string bin_path = "tools/x280_bm/build/rdrbench.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    uint64_t nrounds = 200, nharts = 2;  // 2 readers, per the benchmark spec
    uint64_t noc_split = 0;              // --nocsplit: reader h reads over NOC (h & 1)
    bool do_reset = true, do_boot = true;

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
        } else if (a == "--nharts") {
            nharts = std::stoull(next());
        } else if (a == "--nrounds") {
            nrounds = std::stoull(next());
        } else if (a == "--nocsplit") {
            noc_split = 1;
        } else if (a == "--no-reset") {
            do_reset = false;
        } else if (a == "--no-boot") {
            do_boot = false;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    std::vector<uint8_t> bin = read_file(bin_path);
    printf("[fw] %s (%zu bytes)\n", bin_path.c_str(), bin.size());
    if (do_reset) {
        std::string cmd = "tt-smi -r " + std::to_string(device_id);
        printf("[boot] %s\n", cmd.c_str());
        if (std::system(cmd.c_str()) != 0) {
            fprintf(stderr, "tt-smi reset failed\n");
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    auto mesh = MeshDevice::create_unit_mesh(device_id);
    Cluster& cluster = MetalContext::instance().get_cluster();
    CoreCoord grid = mesh->compute_with_storage_grid_size();
    uint32_t gx = (uint32_t)grid.x, gy = (uint32_t)grid.y;
    uint32_t num_cores = gx * gy;
    printf(
        "[grid] worker grid %ux%u = %u cores; %llu readers; nrounds=%llu\n",
        gx,
        gy,
        num_cores,
        (unsigned long long)nharts,
        (unsigned long long)nrounds);

    // Build the coord table + PRE-FILL each core's L1 with a static buffer: 16-word control region +
    // MAX_K markers. There is NO live producer kernel -- the benchmark measures pure reader read
    // throughput over static data, so the same fill serves every K (the FW drains a fixed K <= MAX_K).
    std::vector<uint8_t> coords(num_cores * 8, 0);
    std::vector<uint8_t> fill(128 + (size_t)MAX_K * MW * 4, 0);
    for (size_t w = 0; w < fill.size() / 4; w++) {
        uint32_t v = 0x80000000u | (uint32_t)w;  // marker valid-bit + index pattern (bad reads would show)
        std::memcpy(fill.data() + w * 4, &v, 4);
    }
    for (uint32_t ly = 0; ly < gy; ly++) {
        for (uint32_t lx = 0; lx < gx; lx++) {
            uint32_t idx = ly * gx + lx;
            CoreCoord logical{lx, ly};
            CoreCoord virt =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical, CoreType::WORKER);
            pack<uint32_t>(coords, idx * 8 + 0, (uint32_t)virt.x);
            pack<uint32_t>(coords, idx * 8 + 4, (uint32_t)virt.y);
            cluster.write_core(fill.data(), (uint32_t)fill.size(), tt_cxy_pair(device_id, virt), SRC_L1);
        }
    }
    printf(
        "[fill] wrote %zu B static buffer (%u markers) to %u cores at L1 0x%llx\n",
        fill.size(),
        MAX_K,
        num_cores,
        (unsigned long long)SRC_L1);

    if (!do_boot) {
        printf("[done] --no-boot.\n");
        std::fflush(stdout);
        std::_Exit(0);
    }

    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.write_block(bin, BENCH_ENTRY);  // FW linked at 0x08001000 (x280-lim.ld)
    x280.write_block(coords, MBOX_COORDS);

    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, num_cores);
    pack<uint64_t>(params, 0x08, SRC_L1);
    pack<uint64_t>(params, 0x10, MW);
    pack<uint64_t>(params, 0x18, nrounds);
    pack<uint64_t>(params, 0x20, nharts);
    pack<uint64_t>(params, 0x28, DST_BASE);
    pack<uint64_t>(params, 0x30, noc_split);
    x280.write_block(params, MBOX_PARAMS);
    printf(
        "[cfg] noc_split=%llu (%s)\n",
        (unsigned long long)noc_split,
        noc_split ? "reader h -> NOC (h&1)" : "all readers -> NOC0");

    // Clear the results + DONE region so a FW that fails to (re)boot is caught as a timeout, not read
    // as stale data from a prior run (re-asserting reset on a wfi'd L2CPU does NOT cleanly restart it;
    // each boot needs a fresh tt-smi -r).
    std::vector<uint8_t> zero_res(RES_STRIDE * 4, 0);
    x280.write_block(zero_res, BENCH_RES);

    x280.set_reset_vectors(BENCH_ENTRY);
    x280.set_pll(pll);
    x280.release_reset();
    printf("[bench] %llu readers draining %u cores, sweeping K...\n", (unsigned long long)nharts, num_cores);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool all = true;
        for (uint64_t h = 0; h < nharts; h++) {
            if (x280.lim_rd_u64(res_done((int)h)) != DONE_MAGIC) {
                all = false;
                break;
            }
        }
        if (all) {
            done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!done) {
        fprintf(stderr, "[bench] timed out waiting for hart done flags\n");
        std::fflush(stdout);
        std::_Exit(1);
    }

    // 2D ILP x K grid. The slower reader hart bounds a sweep; us_sweep = cycles/nrounds/pll (pll MHz).
    // Cell = aggregate drain GB/s at that (ILP, K). K=0 column = poll-only lower bound (µs/sweep).
    auto us_at = [&](int ii, int ki) {
        uint64_t maxc = 0;
        for (uint64_t h = 0; h < nharts; h++) {
            uint64_t c = x280.lim_rd_u64(res_cell((int)h, ii, ki));
            if (c > maxc) {
                maxc = c;
            }
        }
        return (double)maxc / (double)nrounds / (double)pll;
    };
    printf(
        "\n=== X280 reader-drain ILP x K sweep (%llu readers, %u cores, %llu rounds, %llu-word markers) ===\n",
        (unsigned long long)nharts,
        num_cores,
        (unsigned long long)nrounds,
        (unsigned long long)MW);

    // Grid A: aggregate drain GB/s (higher = better). Rows = ILP, cols = K (markers/core).
    printf("\n  drain GB/s   |");
    for (int ki = 1; ki < NK; ki++) {
        printf(" K=%-4u", KLIST[ki]);
    }
    printf("     (poll-only K=0 us/sweep)\n");
    printf("  -------------+");
    for (int ki = 1; ki < NK; ki++) {
        printf("-------");
    }
    printf("\n");
    for (int ii = 0; ii < NILP; ii++) {
        printf("  ILP=%-2u       |", ILPLIST[ii]);
        for (int ki = 1; ki < NK; ki++) {
            double us = us_at(ii, ki);
            double gbps = us > 0 ? ((double)KLIST[ki] * num_cores * MW * 4.0) / (us * 1e-6) / 1e9 : 0.0;
            printf(" %-6.2f", gbps);
        }
        printf("     %.2f\n", us_at(ii, 0));
    }

    // Grid B: ns per marker (lower = better) -- shows ILP amortizing per-core latency, esp. at small K.
    printf("\n  ns/marker    |");
    for (int ki = 1; ki < NK; ki++) {
        printf(" K=%-4u", KLIST[ki]);
    }
    printf("\n  -------------+");
    for (int ki = 1; ki < NK; ki++) {
        printf("-------");
    }
    printf("\n");
    for (int ii = 0; ii < NILP; ii++) {
        printf("  ILP=%-2u       |", ILPLIST[ii]);
        for (int ki = 1; ki < NK; ki++) {
            double us = us_at(ii, ki);
            double nspm = ((double)KLIST[ki] * num_cores) > 0 ? (us * 1000.0) / ((double)KLIST[ki] * num_cores) : 0.0;
            printf(" %-6.1f", nspm);
        }
        printf("\n");
    }
    std::fflush(stdout);
    std::_Exit(0);
}
