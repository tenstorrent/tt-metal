// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 4-hart vector poll split across NOC0/NOC1 to test whether the 2 NIUs beat
// the 530 MB/s single-NIU ceiling. --noc1 N puts the last N harts on NOC1.
//
// Build:  make -C tools/x280_bm  ;  cmake --build build_Release --target test_x280_poll4n1
// Run:    ./build_Release/programming_examples/profiler/test_x280_poll4n1 --noc1 2 --nrounds 2000

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
static constexpr uint64_t MBOX_RESULTS = 0x08011040ULL;
static constexpr uint64_t MBOX_COORDS = 0x08011200ULL;
static constexpr uint64_t DONE_MAGIC = 0x4A0117C0DEULL;
static constexpr uint64_t SRC_L1 = 0x80000ULL;
static constexpr uint64_t DST_BASE = 0x08012000ULL;
static constexpr uint64_t BYTES = 64;

static uint64_t res_slot(int h) { return MBOX_RESULTS + (uint64_t)h * 0x40; }
static const CoreCoord ARC_TILE{8, 0};
struct PllSolution {
    int fbdiv;
    int postdiv[4];
};
static bool pll_solution(int mhz, PllSolution& out) {
    switch (mhz) {
        case 1000: out = {80, {1, 1, 1, 1}}; return true;
        case 200: out = {128, {15, 15, 15, 15}}; return true;
        case 800: out = {64, {1, 1, 1, 1}}; return true;
        case 1750: out = {140, {1, 1, 1, 1}}; return true;
        default: return false;
    }
}

class X280 {
public:
    X280(Cluster& c, int chip, int l2) : cluster_(c), chip_(chip), l2cpu_(l2) {
        l2_ = virt({8, 3});
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
    void load_lim(const std::vector<uint8_t>& b) const {
        cluster_.write_core(b.data(), (uint32_t)b.size(), l2_, LIM_BASE);
    }
    void write_block(const std::vector<uint8_t>& b, uint64_t a) const {
        cluster_.write_core(b.data(), (uint32_t)b.size(), l2_, a);
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
    void set_reset_vectors(uint64_t e) const {
        uint32_t lo = (uint32_t)(e & 0xFFFFFFFF), hi = (uint32_t)(e >> 32);
        for (int c = 0; c < 4; c++) {
            reg_wr(l2_, L2CPU_REG_BASE + c * 8, lo);
            reg_wr(l2_, L2CPU_REG_BASE + c * 8 + 4, hi);
        }
    }
    void set_pll(int mhz) const {
        PllSolution sol;
        if (!pll_solution(mhz, sol)) {
            throw std::runtime_error("pll");
        }
        uint32_t c5 = reg_rd(arc_, PLL_CNTL_5);
        uint8_t pd[4] = {
            uint8_t(c5 & 0xFF), uint8_t((c5 >> 8) & 0xFF), uint8_t((c5 >> 16) & 0xFF), uint8_t((c5 >> 24) & 0xFF)};
        uint32_t c1 = reg_rd(arc_, PLL_CNTL_1), c1_low = c1 & 0x0000FFFF;
        uint16_t fb = uint16_t((c1 >> 16) & 0xFFFF);
        auto wc5 = [&]() {
            uint32_t v = pd[0] | (uint32_t(pd[1]) << 8) | (uint32_t(pd[2]) << 16) | (uint32_t(pd[3]) << 24);
            reg_wr(arc_, PLL_CNTL_5, v);
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };
        auto wc1 = [&]() {
            reg_wr(arc_, PLL_CNTL_1, c1_low | (uint32_t(fb) << 16));
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };
        for (int i = 0; i < 4; i++) {
            while (pd[i] < sol.postdiv[i]) {
                pd[i]++;
                wc5();
            }
        }
        while (fb != sol.fbdiv) {
            fb += (sol.fbdiv > fb) ? 1 : -1;
            wc1();
        }
        for (int i = 0; i < 4; i++) {
            while (pd[i] > sol.postdiv[i]) {
                pd[i]--;
                wc5();
            }
        }
    }
    tt_cxy_pair l2() const { return l2_; }

private:
    tt_cxy_pair virt(CoreCoord p) const {
        return tt_cxy_pair(chip_, cluster_.get_virtual_coordinate_from_physical_coordinates(chip_, p));
    }
    Cluster& cluster_;
    int chip_, l2cpu_;
    tt_cxy_pair l2_, arc_;
};

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        throw std::runtime_error("open " + p);
    }
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (d.size() % 4) {
        d.push_back(0);
    }
    return d;
}
template <typename T>
static void pack(std::vector<uint8_t>& b, size_t o, T v) {
    std::memcpy(b.data() + o, &v, sizeof(T));
}

int main(int argc, char** argv) {
    using namespace tt::tt_metal;
    std::string bin_path = "tools/x280_bm/build/poll4n1.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    uint64_t nharts = 4, nrounds = 2000, noc1_harts = 2, flits = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--bin") {
            bin_path = argv[++i];
        } else if (a == "--device") {
            device_id = std::stoi(argv[++i]);
        } else if (a == "--pll") {
            pll = std::stoi(argv[++i]);
        } else if (a == "--nharts") {
            nharts = std::stoull(argv[++i]);
        } else if (a == "--nrounds") {
            nrounds = std::stoull(argv[++i]);
        } else if (a == "--noc1") {
            noc1_harts = std::stoull(argv[++i]);
        } else if (a == "--flits") {
            flits = std::stoull(argv[++i]);
        }
    }
    uint64_t bytes_per_read = flits * BYTES;
    auto bin = read_file(bin_path);
    printf("[fw] %s (%zu bytes); noc1_harts=%llu\n", bin_path.c_str(), bin.size(), (unsigned long long)noc1_harts);
    if (std::system(("tt-smi -r " + std::to_string(device_id)).c_str()) != 0) {
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));

    auto mesh = MeshDevice::create_unit_mesh(device_id);
    Cluster& cluster = MetalContext::instance().get_cluster();
    CoreCoord grid = mesh->compute_with_storage_grid_size();
    uint32_t gx = (uint32_t)grid.x, gy = (uint32_t)grid.y, num_cores = gx * gy;
    printf(
        "[grid] %ux%u = %u cores; %llu harts, %llu on NOC1\n",
        gx,
        gy,
        num_cores,
        (unsigned long long)nharts,
        (unsigned long long)noc1_harts);

    {
        Program program = CreateProgram();
        std::map<std::string, std::string> defs = {{"COUNTER_ADDR", std::to_string(SRC_L1)}};
        CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});
        CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_x280_poll_rate/kernels/brisc_counter.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defs});
        distributed::MeshWorkload wl;
        wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), wl, false);
    }

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

    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);
    x280.write_block(coords, MBOX_COORDS);
    {
        std::vector<uint8_t> z(nharts * 0x40, 0);
        x280.write_block(z, MBOX_RESULTS);
    }  // clear stale done
    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, num_cores);
    pack<uint64_t>(params, 0x08, SRC_L1);
    pack<uint64_t>(params, 0x10, BYTES);
    pack<uint64_t>(params, 0x18, DST_BASE);
    pack<uint64_t>(params, 0x20, nrounds);
    pack<uint64_t>(params, 0x28, nharts);
    pack<uint64_t>(params, 0x30, noc1_harts);
    pack<uint64_t>(params, 0x38, flits);
    x280.write_block(params, MBOX_PARAMS);
    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();
    printf("[run] split poll, %llu rounds ...\n", (unsigned long long)nrounds);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool all = true;
        for (uint64_t h = 0; h < nharts; h++) {
            if (x280.lim_rd_u64(res_slot((int)h) + 0x10) != DONE_MAGIC) {
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
        fprintf(stderr, "timeout\n");
        std::_Exit(1);
    }

    uint64_t total_reads = 0, max_cycles = 0;
    printf(
        "\n=== X280 dual-NoC poll (%llu harts, %llu on NOC1, %llu flits/read) ===\n",
        (unsigned long long)nharts,
        (unsigned long long)noc1_harts,
        (unsigned long long)flits);
    for (uint64_t h = 0; h < nharts; h++) {
        uint64_t reads = x280.lim_rd_u64(res_slot((int)h) + 0x00);
        uint64_t cyc = x280.lim_rd_u64(res_slot((int)h) + 0x08);
        double mbps = (double)reads * bytes_per_read / 1e6 / ((double)cyc / ((double)pll * 1e6));
        int noc = (h >= (nharts - noc1_harts)) ? 1 : 0;
        printf(
            "  hart %llu (NOC%d): %llu reads, %.0f MB/s\n",
            (unsigned long long)h,
            noc,
            (unsigned long long)reads,
            mbps);
        total_reads += reads;
        if (cyc > max_cycles) {
            max_cycles = cyc;
        }
    }
    double agg = (double)total_reads * bytes_per_read / 1e6 / ((double)max_cycles / ((double)pll * 1e6));
    printf("  -------------------------------------------------\n");
    printf("  AGGREGATE: %.0f MB/s   (all-NOC0 ref ~530 MB/s)\n", agg);
    std::fflush(stdout);
    std::_Exit(0);
}
