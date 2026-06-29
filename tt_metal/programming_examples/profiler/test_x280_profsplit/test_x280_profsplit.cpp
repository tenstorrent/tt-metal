// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 profiler consumer with READER/RELAY-HART SPLIT (throughput bench). Reader
// harts drain device rings into per-reader LIM staging rings; a dedicated relay
// hart posted-writes the staged flits to host. Measures end-to-end read->relay
// throughput with no producers (relay hart times the whole concurrent window).
//
// Build:  make -C tools/x280_bm                               (profcons_split.bin)
//         (host: one-file compile against libtt_metal.so)
// Run:    ./build_Release/programming_examples/profiler/test_x280_profsplit \
//             --nharts 3 --nread 2 --reps 200

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
#include <llrt/hal.hpp>
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
static constexpr uint64_t MBOX_COORDS = 0x08011200ULL;
static constexpr uint64_t STAGECTL = 0x08018000ULL;
static constexpr uint64_t DONE_MAGIC = 0x5717C0FFEEULL;
static constexpr uint64_t FOOTER_MAGIC = 0xF00DD2C0FFEEULL;
static constexpr uint64_t WIN_STRIDE = 0x200000ULL;
static constexpr int NRISC = 5;
static constexpr uint64_t HOST_SLICE = 2048;

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
    std::string bin_path = "tools/x280_bm/build/profcons_split.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    uint64_t nharts = 3, nread = 2, reps = 200;
    int mode = 0;  // 0 = full read->relay; 1 = read-only scatter; 2 = read-only contiguous

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
        } else if (a == "--nread") {
            nread = std::stoull(next());
        } else if (a == "--reps") {
            reps = std::stoull(next());
        } else if (a == "--ro") {
            mode = 1;  // read-only, quarter-scatter
        } else if (a == "--ro-contig") {
            mode = 2;  // read-only, contiguous (old pattern) -- A/B baseline
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }
    if (nread >= nharts) {
        fprintf(stderr, "need at least 1 relay hart (nread < nharts)\n");
        return 2;
    }

    auto bin = read_file(bin_path);
    printf("[fw] %s (%zu bytes)\n", bin_path.c_str(), bin.size());
    {
        std::string cmd = "tt-smi -r " + std::to_string(device_id);
        printf("[boot] %s\n", cmd.c_str());
        if (std::system(cmd.c_str()) != 0) {
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    auto mesh = MeshDevice::create_unit_mesh(device_id);
    Cluster& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();

    CoreCoord grid = mesh->compute_with_storage_grid_size();
    uint32_t gx = (uint32_t)grid.x, gy = (uint32_t)grid.y;
    uint32_t num_cores = gx * gy;
    uint64_t prof_l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::PROFILER);

    std::vector<uint8_t> coords(num_cores * 8, 0);
    for (uint32_t ly = 0; ly < gy; ly++) {
        for (uint32_t lx = 0; lx < gx; lx++) {
            uint32_t idx = ly * gx + lx;
            CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord{lx, ly}, CoreType::WORKER);
            pack<uint32_t>(coords, idx * 8 + 0, (uint32_t)v.x);
            pack<uint32_t>(coords, idx * 8 + 4, (uint32_t)v.y);
        }
    }

    auto pcie_cores = cluster.get_soc_desc(device_id).get_cores(tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED);
    auto pc = pcie_cores.front();
    uint32_t pcie_enc = ((uint32_t)pc.x & 0x3f) | (((uint32_t)pc.y & 0x3f) << 6);
    uint64_t pcie_base = cluster.get_pcie_base_addr_from_device(device_id);
    uint64_t chan_sz = cluster.get_host_channel_size(device_id, 0);
    uint64_t region = (uint64_t)num_cores * NRISC * HOST_SLICE;
    if (region > WIN_STRIDE) {
        fprintf(stderr, "relay region %lu B > 2MB window\n", (unsigned long)region);
        return 1;
    }
    uint64_t data_off = (chan_sz / 2) & ~(WIN_STRIDE - 1);
    uint64_t host_base = pcie_base + data_off;
    printf(
        "[cfg] %ux%u=%u cores; prof_l1=0x%lx; %llu harts (%llu readers + %llu relay); reps=%llu; region=%lu KB\n",
        gx,
        gy,
        num_cores,
        (unsigned long)prof_l1,
        (unsigned long long)nharts,
        (unsigned long long)nread,
        (unsigned long long)(nharts - nread),
        (unsigned long long)reps,
        (unsigned long)(region >> 10));

    {
        uint64_t z = 0;
        cluster.write_sysmem(&z, sizeof(z), data_off + region, device_id, 0);
    }  // footer slot

    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);
    x280.write_block(coords, MBOX_COORDS);
    std::vector<uint8_t> zero(256, 0);
    x280.write_block(zero, STAGECTL);  // pre-zero staging prod/cons/rdone (no init race)
    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, (uint64_t)pcie_enc);
    pack<uint64_t>(params, 0x08, host_base);
    pack<uint64_t>(params, 0x10, prof_l1);
    pack<uint64_t>(params, 0x18, (uint64_t)num_cores);
    pack<uint64_t>(params, 0x20, nharts);
    pack<uint64_t>(params, 0x28, nread);
    pack<uint64_t>(params, 0x30, reps);
    pack<uint64_t>(params, 0x38, (uint64_t)mode);
    x280.write_block(params, MBOX_PARAMS);
    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();
    printf("[run] readers draining -> LIM staging -> relay hart -> host ...\n");

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool all = true;
        for (uint64_t h = 0; h < nharts; h++) {
            if (x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x18) != DONE_MAGIC) {
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
        fprintf(stderr, "[run] timed out\n");
        std::fflush(stdout);
        std::_Exit(1);
    }

    if (mode != 0) {
        uint64_t ro_bytes = 0, ro_cyc = 0;
        for (uint64_t h = 0; h < nread; h++) {
            ro_bytes += x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x00);
            uint64_t c = x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x08);
            if (c > ro_cyc) {
                ro_cyc = c;
            }
        }
        double mbps = ro_cyc ? (double)ro_bytes / 1e6 / ((double)ro_cyc / ((double)pll * 1e6)) : 0.0;
        printf("\n=== X280 read-only bench (%s) ===\n", mode == 1 ? "quarter-scatter" : "contiguous");
        printf(
            "  readers read : %llu B in %llu cycles (max)\n", (unsigned long long)ro_bytes, (unsigned long long)ro_cyc);
        printf("  READ BW      : %.0f MB/s   (%llu readers, no relay)\n", mbps, (unsigned long long)nread);
        std::fflush(stdout);
        std::_Exit(0);
    }

    uint64_t footer = 0;
    cluster.read_sysmem(&footer, sizeof(footer), data_off + region, device_id, 0);
    bool noc1_ok = (footer == FOOTER_MAGIC);

    uint64_t reader_bytes = 0, relay_bytes = 0, relay_cyc = 0;
    for (uint64_t h = 0; h < nread; h++) {
        reader_bytes += x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x00);
    }
    for (uint64_t h = nread; h < nharts; h++) {
        relay_bytes += x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x00);
        uint64_t cyc = x280.lim_rd_u64(MBOX_RESULTS + h * 0x40 + 0x08);
        if (cyc > relay_cyc) {
            relay_cyc = cyc;
        }
    }
    double mbps = relay_cyc ? (double)relay_bytes / 1e6 / ((double)relay_cyc / ((double)pll * 1e6)) : 0.0;

    printf("\n=== X280 reader/relay-split consumer (bench) ===\n");
    printf("  readers read  : %llu B\n", (unsigned long long)reader_bytes);
    printf("  relay shipped : %llu B in %llu cycles\n", (unsigned long long)relay_bytes, (unsigned long long)relay_cyc);
    printf("  consistent    : %s\n", reader_bytes == relay_bytes ? "YES" : "NO <-- mismatch");
    printf("  NOC1 relay->host landed : %s\n", noc1_ok ? "YES (footer present)" : "NO <-- NOC1->PCIe write dropped!");
    printf("  -------------------------------------------------\n");
    printf(
        "  END-TO-END    : %.0f MB/s   (%llu readers + %llu relay on NOC1)\n",
        mbps,
        (unsigned long long)nread,
        (unsigned long long)(nharts - nread));
    std::fflush(stdout);
    std::_Exit((reader_bytes == relay_bytes && noc1_ok) ? 0 : 1);
}
