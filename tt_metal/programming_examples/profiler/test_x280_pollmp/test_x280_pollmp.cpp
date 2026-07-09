// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 long-shot levers vs the 530 MB/s mesh-ingress wall. Each hart STREAMS a
// large distinct linear region from one Tensix core's L1 over the NoC, through
// either the uncached System Port or the cacheable Memory Port (#3), optionally
// pinning a static NoC VC (#4). Single linear pass = real NoC traffic on every
// line (no cache reuse) and the best case for the L2 prefetcher. The question is
// whether a cached port lets ONE hart hold several NoC reads in flight and so
// exceed the ~135 MB/s single-hart / 530 MB/s aggregate System-Port ceiling.
//
// Build:  make -C tools/x280_bm                               (pollmp.bin)
//         cmake --build build_Release --target test_x280_pollmp
// Run:    ./build_Release/programming_examples/profiler/test_x280_pollmp \
//             --nharts 1 --memport 1 --span 524288
//
// Flags: --bin <pollmp.bin> --device N --l2cpu N --pll MHZ --nharts N(<=4)
//        --span BYTES --memport 0|1 --vc N(0..7) --vc-spread --no-reset --no-boot

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
static constexpr uint64_t DONE_MAGIC = 0x4D90111CDEULL;

static constexpr uint64_t SRC_L1 = 0x80000ULL;
static constexpr uint64_t DST_BASE = 0x08012000ULL;

static constexpr uint32_t VC_NONE = 0xFFFFFFFFu;
static constexpr uint32_t VC_SPREAD = 0xFFFFFFFEu;

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
    std::string bin_path = "tools/x280_bm/build/pollmp.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    uint64_t nharts = 1, span = 524288, memport = 1, ilp = 4;
    uint32_t vc = VC_NONE;
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
        } else if (a == "--span") {
            span = std::stoull(next());
        } else if (a == "--memport") {
            memport = std::stoull(next());
        } else if (a == "--ilp") {
            ilp = std::stoull(next());
        } else if (a == "--vc") {
            vc = (uint32_t)std::stoul(next());
        } else if (a == "--vc-spread") {
            vc = VC_SPREAD;
        } else if (a == "--no-reset") {
            do_reset = false;
        } else if (a == "--no-boot") {
            do_boot = false;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

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

    CoreCoord grid = mesh->compute_with_storage_grid_size();
    uint32_t gx = (uint32_t)grid.x, gy = (uint32_t)grid.y;
    uint32_t num_cores = gx * gy;
    const char* port_name = memport ? "Memory Port (cached)" : "System Port (uncached)";
    std::string vc_name =
        (vc == VC_NONE) ? "none" : (vc == VC_SPREAD ? "spread(vc=hartid)" : ("vc=" + std::to_string(vc)));
    printf(
        "[grid] %ux%u = %u cores; %llu harts stream %llu B each via %s; ILP=%llu; static VC: %s\n",
        gx,
        gy,
        num_cores,
        (unsigned long long)nharts,
        (unsigned long long)span,
        port_name,
        (unsigned long long)ilp,
        vc_name.c_str());

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
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, /*blocking=*/false);
        printf("[prod] BRISC counter launched on all %u cores (L1 has live data to stream)\n", num_cores);
    }

    std::vector<uint8_t> coords(num_cores * 8, 0);
    for (uint32_t ly = 0; ly < gy; ly++) {
        for (uint32_t lx = 0; lx < gx; lx++) {
            uint32_t idx = ly * gx + lx;
            CoreCoord logical{lx, ly};
            CoreCoord v = cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical, CoreType::WORKER);
            pack<uint32_t>(coords, idx * 8 + 0, (uint32_t)v.x);
            pack<uint32_t>(coords, idx * 8 + 4, (uint32_t)v.y);
        }
    }

    if (!do_boot) {
        printf("[done] --no-boot.\n");
        std::fflush(stdout);
        std::_Exit(0);
    }

    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);
    x280.write_block(coords, MBOX_COORDS);

    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, num_cores);
    pack<uint64_t>(params, 0x08, SRC_L1);
    pack<uint64_t>(params, 0x10, span);
    pack<uint64_t>(params, 0x18, DST_BASE);
    pack<uint64_t>(params, 0x20, nharts);
    pack<uint64_t>(params, 0x28, memport);
    pack<uint64_t>(params, 0x30, (uint64_t)vc);
    pack<uint64_t>(params, 0x38, ilp);
    x280.write_block(params, MBOX_PARAMS);

    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();
    printf("[run] %llu harts streaming ...\n", (unsigned long long)nharts);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool all = true;
        for (uint64_t h = 0; h < nharts; h++) {
            if (x280.lim_rd_u64(res_slot((int)h) + 0x18) != DONE_MAGIC) {
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
        fprintf(stderr, "[run] timed out (a hart may have faulted on the port/VC config)\n");
        std::fflush(stdout);
        std::_Exit(1);
    }

    uint64_t max_cycles = 0, total_bytes = 0;
    printf(
        "\n=== X280 streaming read: %s, %llu hart(s), VC %s ===\n",
        port_name,
        (unsigned long long)nharts,
        vc_name.c_str());
    for (uint64_t h = 0; h < nharts; h++) {
        uint64_t cyc = x280.lim_rd_u64(res_slot((int)h) + 0x00);
        uint64_t bytes = x280.lim_rd_u64(res_slot((int)h) + 0x08);
        double mbps = cyc ? (double)bytes / 1e6 / ((double)cyc / ((double)pll * 1e6)) : 0.0;
        printf(
            "  hart %llu: %llu B in %llu cycles -> %.0f MB/s\n",
            (unsigned long long)h,
            (unsigned long long)bytes,
            (unsigned long long)cyc,
            mbps);
        if (cyc > max_cycles) {
            max_cycles = cyc;
        }
        total_bytes += bytes;
    }
    double wall_s = (double)max_cycles / ((double)pll * 1e6);
    double agg_mb = wall_s > 0 ? (double)total_bytes / 1e6 / wall_s : 0.0;
    printf("  -------------------------------------------------\n");
    printf("  AGGREGATE BW  : %.0f MB/s   (System-Port 4-hart ref ~530 MB/s)\n", agg_mb);
    std::fflush(stdout);
    std::_Exit(0);
}
