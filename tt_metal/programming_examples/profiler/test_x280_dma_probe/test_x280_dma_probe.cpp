// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 DMA validation (DMA arc, step 1). Proves the bare-metal X280 can drive
// its Synopsys DW DMAC to pull a Tensix L1 region over the NOC into local LIM,
// with no core loads/stores.
//
// Flow (all via tt-metal UMD):
//   1. tt-smi -r, open the chip (MeshDevice), grab the low-level Cluster.
//   2. Host writes a known pattern (word[i] = 0xA5A50000 | i) into Tensix (0,0)
//      L1 at src_l1 (no kernel needed -- a static buffer the DMA will read).
//   3. Boot the X280 dma_probe FW, passing it (target physical NOC coord,
//      src L1 addr, dst LIM addr, nbytes) via a LIM mailbox. The FW runs
//      dma_engine_noc_to_x280() -- one DMAC NOC->LIM transfer.
//   4. Read the FW's result (rc/cycles/first/last) from LIM, then read the LIM
//      destination over the NOC and byte-compare it to the pattern.
//
// Build:  make -C tools/x280_bm                               (dma_probe.bin)
//         cmake --build build_Release --target test_x280_dma_probe
// Run (repo root, libs on the path):
//   export TT_METAL_HOME=$PWD
//   export LD_LIBRARY_PATH=$(find build_Release -name '*.so*' -type f \
//       -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
//   ./build_Release/programming_examples/profiler/test_x280_dma_probe
//
// Flags: --bin <dma_probe.bin> --device N --l2cpu N --pll MHZ
//        --core X Y (Tensix source core, default 0 0) --src 0xADDR
//        --dst 0xLIMADDR --nbytes N --no-reset --no-boot

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>

#include <impl/context/metal_context.hpp>
#include <llrt/tt_cluster.hpp>

using tt::Cluster;
using tt::CoreType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::MetalContext;
using tt::tt_metal::distributed::MeshDevice;

static constexpr uint64_t LIM_BASE = 0x08000000ULL;
static constexpr uint64_t RESET_UNIT_BASE = 0x80030000ULL;
static constexpr uint64_t L2CPU_RESET_REG = RESET_UNIT_BASE + 0x14;
static constexpr uint64_t L2CPU_REG_BASE = 0xFFFFF7FEFFF10000ULL;
static constexpr uint64_t PLL4_BASE = 0x80020500ULL;
static constexpr uint64_t PLL_CNTL_1 = PLL4_BASE + 0x4;
static constexpr uint64_t PLL_CNTL_5 = PLL4_BASE + 0x14;

// ---- LIM mailbox (must match tools/x280_bm/src/dma_probe.c) ----
static constexpr uint64_t MBOX_PARAMS = 0x08011000ULL;
static constexpr uint64_t MBOX_RESULTS = 0x08011040ULL;
static constexpr uint64_t R_DONE = MBOX_RESULTS + 0x18;
static constexpr uint64_t DONE_MAGIC = 0xDDA9C0FFEEULL;

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
    uint32_t lim_rd_u32(uint64_t a) const {
        uint32_t v = 0;
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
    void write_mailbox(const std::vector<uint8_t>& block, uint64_t addr) const {
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
    std::string bin_path = "tools/x280_bm/build/dma_probe.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    int src_x = 0, src_y = 0;
    uint64_t src_l1 = 0x80000, dst_lim = 0x08012000, nbytes = 256, repeats = 1;
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
        } else if (a == "--core") {
            src_x = std::stoi(next());
            src_y = std::stoi(next());
        } else if (a == "--src") {
            src_l1 = std::stoull(next(), nullptr, 0);
        } else if (a == "--dst") {
            dst_lim = std::stoull(next(), nullptr, 0);
        } else if (a == "--nbytes") {
            nbytes = std::stoull(next());
        } else if (a == "--repeats") {
            repeats = std::stoull(next());
        } else if (a == "--no-reset") {
            do_reset = false;
        } else if (a == "--no-boot") {
            do_boot = false;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }
    nbytes &= ~uint64_t(3);  // keep word-aligned

    std::vector<uint8_t> bin;
    if (do_boot) {
        bin = read_file(bin_path);
        printf("[fw] %s (%zu bytes)\n", bin_path.c_str(), bin.size());
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

    auto mesh = MeshDevice::create_unit_mesh(device_id);
    Cluster& cluster = MetalContext::instance().get_cluster();

    // ---- Write a known pattern into the Tensix source L1 (no kernel) ----
    CoreCoord src_logical{(size_t)src_x, (size_t)src_y};
    CoreCoord src_virt =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, src_logical, CoreType::WORKER);
    CoreCoord src_phys =
        cluster.get_physical_coordinate_from_logical_coordinates(device_id, src_logical, CoreType::WORKER);
    tt_cxy_pair src_v(device_id, src_virt);
    printf(
        "[src] Tensix logical (%d,%d) -> physical (%u,%u); L1 0x%llx, %llu bytes\n",
        src_x,
        src_y,
        (unsigned)src_phys.x,
        (unsigned)src_phys.y,
        (unsigned long long)src_l1,
        (unsigned long long)nbytes);

    std::vector<uint8_t> pattern(nbytes, 0);
    for (uint64_t i = 0; i + 4 <= nbytes; i += 4) {
        uint32_t w = 0xA5A50000u | (uint32_t)(i / 4);
        std::memcpy(pattern.data() + i, &w, 4);
    }
    cluster.write_core(pattern.data(), (uint32_t)nbytes, src_v, src_l1);
    std::vector<uint8_t> rb(nbytes, 0);
    cluster.read_core(rb.data(), (uint32_t)nbytes, src_v, src_l1);
    printf("[src] pattern written+verified on Tensix L1: %s\n", rb == pattern ? "OK" : "READBACK MISMATCH");

    if (!do_boot) {
        printf("[done] --no-boot: source-pattern only.\n");
        std::fflush(stdout);
        std::_Exit(0);
    }

    // ---- Boot the DMA probe FW; pass it the target via the LIM mailbox ----
    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);

    std::vector<uint8_t> params(64, 0);
    pack<uint32_t>(params, 0x00, (uint32_t)src_phys.x);
    pack<uint32_t>(params, 0x04, (uint32_t)src_phys.y);
    pack<uint64_t>(params, 0x08, src_l1);
    pack<uint64_t>(params, 0x10, dst_lim);
    pack<uint64_t>(params, 0x18, nbytes);
    pack<uint64_t>(params, 0x20, repeats);
    x280.write_mailbox(params, MBOX_PARAMS);

    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();
    printf(
        "[dma] X280 DMA pull: Tensix (%u,%u) L1 0x%llx -> LIM 0x%llx, %llu bytes ...\n",
        (unsigned)src_phys.x,
        (unsigned)src_phys.y,
        (unsigned long long)src_l1,
        (unsigned long long)dst_lim,
        (unsigned long long)nbytes);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (x280.lim_rd_u64(R_DONE) == DONE_MAGIC) {
            done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!done) {
        fprintf(stderr, "[dma] timed out waiting for FW done flag\n");
        std::fflush(stdout);
        std::_Exit(1);
    }

    uint64_t rc = x280.lim_rd_u64(MBOX_RESULTS + 0x00);
    uint64_t cycles = x280.lim_rd_u64(MBOX_RESULTS + 0x08);
    uint32_t fw_first = x280.lim_rd_u32(MBOX_RESULTS + 0x10);
    uint32_t fw_last = x280.lim_rd_u32(MBOX_RESULTS + 0x14);
    uint64_t retrig_avg = x280.lim_rd_u64(MBOX_RESULTS + 0x20);
    uint64_t retrig_min = x280.lim_rd_u64(MBOX_RESULTS + 0x28);

    // ---- Read the LIM destination over the NOC and compare to the pattern ----
    std::vector<uint8_t> got(nbytes, 0);
    x280.lim_read(got.data(), (uint32_t)nbytes, dst_lim);
    size_t mism = 0, first_mism = 0;
    for (size_t i = 0; i < nbytes; i++) {
        if (got[i] != pattern[i]) {
            if (mism == 0) {
                first_mism = i;
            }
            mism++;
        }
    }

    const char* rcs = (rc == 0) ? "success" : (rc == 1) ? "DMA error" : (rc == 2) ? "timeout" : "?";
    printf("\n=== X280 DMA NOC->LIM validation ===\n");
    printf("  dma rc       : %llu (%s)\n", (unsigned long long)rc, rcs);
    printf("  dma cycles   : %llu  (first transfer, full setup; @ %d MHz)\n", (unsigned long long)cycles, pll);
    if (repeats > 1 && retrig_avg > 0) {
        double first_bw = (double)nbytes * pll / (double)cycles;   // MB/s
        double rt_bw = (double)nbytes * pll / (double)retrig_min;  // MB/s
        printf(
            "  re-trigger   : avg %llu, min %llu cycles  (setup-once, x%llu)\n",
            (unsigned long long)retrig_avg,
            (unsigned long long)retrig_min,
            (unsigned long long)(repeats - 1));
        printf(
            "  setup saved  : %lld cycles/transfer (%.1fx faster: %.0f -> %.0f MB/s)\n",
            (long long)cycles - (long long)retrig_min,
            (double)cycles / (double)retrig_min,
            first_bw,
            rt_bw);
    }
    printf("  fw readback  : first=0x%08x last=0x%08x\n", fw_first, fw_last);
    uint32_t exp_first = 0xA5A50000u, exp_last = 0xA5A50000u | (uint32_t)(nbytes / 4 - 1);
    printf("  expected     : first=0x%08x last=0x%08x\n", exp_first, exp_last);
    printf(
        "  host compare : %zu / %llu bytes mismatched%s",
        mism,
        (unsigned long long)nbytes,
        mism ? "" : "  -> DMA MOVED THE DATA ✓\n");
    if (mism) {
        printf("  (first mismatch at byte %zu)\n", first_mism);
    }
    std::fflush(stdout);
    std::_Exit(rc == 0 && mism == 0 ? 0 : 1);
}
