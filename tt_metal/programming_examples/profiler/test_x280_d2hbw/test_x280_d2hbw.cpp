// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 -> HOST device-to-host (D2H) write bandwidth, bare-metal. The X280
// fabricates fake 64 B packets and blasts posted NoC writes through the PCIe tile
// into host pinned memory (sysmem channel 0); this measures the peak export BW.
//
// Host side (low-level Cluster): derive the PCIe tile's TRANSLATED coord +
// get_pcie_base_addr_from_device for the host IOVA, place per-hart 2 MiB regions
// in the host channel, boot the FW, then verify arrival with a FOOTER handshake
// (read_sysmem -- a host-side read of the hugepage, NOT a device read through the
// PCIe tile, which would hang the hart) and report BW.
//
// Build:  make -C tools/x280_bm                               (d2hbw.bin)
//         cmake --build build_Release --target test_x280_d2hbw
// Run:    ./build_Release/programming_examples/profiler/test_x280_d2hbw \
//             --nharts 2 --ilp 4 --bytes 1048576 --nrounds 64
//
// Flags: --bin <d2hbw.bin> --device N --l2cpu N --pll MHZ --nharts N(<=4)
//        --ilp 1|2|4|8 --bytes B(per hart/round) --nrounds N --no-reset --no-boot

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
#include <umd/device/types/core_coordinates.hpp>

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

static constexpr uint64_t MBOX_PARAMS = 0x08011000ULL;
static constexpr uint64_t MBOX_RESULTS = 0x08011040ULL;
static constexpr uint64_t DONE_MAGIC = 0xD2A011BDDEULL;
static constexpr uint64_t FOOTER_MAGIC = 0xF007E2D2D2D2D2D2ULL;
static constexpr uint64_t WIN_STRIDE = 0x200000ULL;  // 2 MiB per hart

static uint64_t res_slot(int h) { return MBOX_RESULTS + (uint64_t)h * 0x40; }

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
    std::string bin_path = "tools/x280_bm/build/d2hbw.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    uint64_t nharts = 2, ilp = 4, bytes = 1048576, nrounds = 64;
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
        } else if (a == "--ilp") {
            ilp = std::stoull(next());
        } else if (a == "--bytes") {
            bytes = std::stoull(next());
        } else if (a == "--nrounds") {
            nrounds = std::stoull(next());
        } else if (a == "--no-reset") {
            do_reset = false;
        } else if (a == "--no-boot") {
            do_boot = false;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }
    if (bytes > WIN_STRIDE - 4096) {
        bytes = WIN_STRIDE - 4096;  // leave room for the footer at win_stride-64
    }

    auto bin = read_file(bin_path);
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

    // --- D2H addressing: PCIe tile translated coord + host IOVA (sysmem ch 0) ---
    auto pcie_cores = cluster.get_soc_desc(device_id).get_cores(tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED);
    if (pcie_cores.empty()) {
        fprintf(stderr, "no PCIE cores in soc desc\n");
        return 1;
    }
    auto pc = pcie_cores.front();
    uint32_t pcie_enc = ((uint32_t)pc.x & 0x3f) | (((uint32_t)pc.y & 0x3f) << 6);
    uint64_t pcie_base = cluster.get_pcie_base_addr_from_device(device_id);
    uint64_t chan_sz = cluster.get_host_channel_size(device_id, 0);
    uint64_t need = nharts * WIN_STRIDE;
    // Place our buffer high in the channel, away from the dispatch CQ near base.
    uint64_t data_off = (chan_sz / 2) & ~(WIN_STRIDE - 1);
    if (data_off + need > chan_sz) {
        data_off = (chan_sz - need) & ~(WIN_STRIDE - 1);
    }
    uint64_t host_base = pcie_base + data_off;
    printf(
        "[d2h] PCIe tile translated (%u,%u) enc=0x%x | pcie_base=0x%lx chan0=%lu MB | data_off=0x%lx host_base=0x%lx\n",
        (unsigned)pc.x,
        (unsigned)pc.y,
        pcie_enc,
        (unsigned long)pcie_base,
        (unsigned long)(chan_sz >> 20),
        (unsigned long)data_off,
        (unsigned long)host_base);
    if (host_base & (WIN_STRIDE - 1)) {
        fprintf(
            stderr, "host_base 0x%lx not 2MB aligned (pcie_base unaligned); extend tool\n", (unsigned long)host_base);
        return 1;
    }

    uint64_t nonce = 0xD2D2000000000000ULL | (uint64_t)(bytes & 0xFFFF) << 16 | (nharts << 8) | ilp;

    // Zero each hart's footer slot in sysmem so a stale value can't false-trigger.
    {
        uint64_t zero = 0;
        for (uint64_t h = 0; h < nharts; h++) {
            uint64_t foff = data_off + h * WIN_STRIDE + (WIN_STRIDE - 64);
            cluster.write_sysmem(&zero, sizeof(zero), foff, device_id, 0);
        }
    }

    printf(
        "[run] %llu harts x ILP %llu, %llu B/hart x %llu rounds -> host (nonce 0x%lx)\n",
        (unsigned long long)nharts,
        (unsigned long long)ilp,
        (unsigned long long)bytes,
        (unsigned long long)nrounds,
        (unsigned long)nonce);

    if (!do_boot) {
        printf("[done] --no-boot.\n");
        std::fflush(stdout);
        std::_Exit(0);
    }

    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);

    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, (uint64_t)pcie_enc);
    pack<uint64_t>(params, 0x08, host_base);
    pack<uint64_t>(params, 0x10, WIN_STRIDE);
    pack<uint64_t>(params, 0x18, bytes);
    pack<uint64_t>(params, 0x20, nharts);
    pack<uint64_t>(params, 0x28, ilp);
    pack<uint64_t>(params, 0x30, nrounds);
    pack<uint64_t>(params, 0x38, nonce);
    x280.write_block(params, MBOX_PARAMS);

    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();

    // Wait for: (a) all FW done flags (LIM read, safe) AND (b) all footers landed
    // in sysmem (host-side read_sysmem -> all posted data writes have arrived).
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool all = true;
        for (uint64_t h = 0; h < nharts; h++) {
            if (x280.lim_rd_u64(res_slot((int)h) + 0x18) != DONE_MAGIC) {
                all = false;
                break;
            }
            uint64_t foot = 0;
            uint64_t foff = data_off + h * WIN_STRIDE + (WIN_STRIDE - 64);
            cluster.read_sysmem(&foot, sizeof(foot), foff, device_id, 0);
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
        fprintf(stderr, "[run] timed out waiting for done flags + footers (data may not have landed)\n");
        std::fflush(stdout);
        std::_Exit(1);
    }

    // Verify a sample of each hart's data in host memory matches the fake pattern.
    uint32_t verify_ok = 0;
    for (uint64_t h = 0; h < nharts; h++) {
        uint64_t dpat = nonce ^ (h * 0x0101010101010101ULL);
        uint64_t got_lo = 0, got_mid = 0;
        cluster.read_sysmem(&got_lo, sizeof(got_lo), data_off + h * WIN_STRIDE + 0, device_id, 0);
        cluster.read_sysmem(&got_mid, sizeof(got_mid), data_off + h * WIN_STRIDE + (bytes / 2 & ~7ULL), device_id, 0);
        if (got_lo == dpat && got_mid == dpat) {
            verify_ok++;
        }
    }

    uint64_t max_cycles = 0, total_bytes = 0;
    printf(
        "\n=== X280 D2H write BW (%llu harts x ILP %llu, 64 B flits) ===\n",
        (unsigned long long)nharts,
        (unsigned long long)ilp);
    for (uint64_t h = 0; h < nharts; h++) {
        uint64_t cyc = x280.lim_rd_u64(res_slot((int)h) + 0x00);
        uint64_t b = x280.lim_rd_u64(res_slot((int)h) + 0x08);
        double mbps = cyc ? (double)b / 1e6 / ((double)cyc / ((double)pll * 1e6)) : 0.0;
        printf(
            "  hart %llu: %llu B in %llu cycles -> %.0f MB/s\n",
            (unsigned long long)h,
            (unsigned long long)b,
            (unsigned long long)cyc,
            mbps);
        if (cyc > max_cycles) {
            max_cycles = cyc;
        }
        total_bytes += b;
    }
    double wall_s = (double)max_cycles / ((double)pll * 1e6);
    double agg = wall_s > 0 ? (double)total_bytes / 1e6 / wall_s : 0.0;
    printf("  -------------------------------------------------\n");
    printf("  AGGREGATE D2H : %.0f MB/s   (Linux 2-hart ref ~268 MB/s)\n", agg);
    printf(
        "  verify        : %u / %llu harts pattern-correct in host mem%s\n",
        verify_ok,
        (unsigned long long)nharts,
        verify_ok == nharts ? "  ✓" : "  <-- MISMATCH");
    std::fflush(stdout);
    std::_Exit(verify_ok == nharts ? 0 : 1);
}
