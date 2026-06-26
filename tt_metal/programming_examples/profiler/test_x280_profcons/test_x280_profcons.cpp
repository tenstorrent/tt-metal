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
#include <umd/device/types/core_coordinates.hpp>

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
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }
    if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
        fprintf(stderr, "[warn] TT_METAL_DEVICE_PROFILER not set -- kernels won't emit markers.\n");
    }

    auto bin = read_file(bin_path);
    printf("[fw] %s (%zu bytes)\n", bin_path.c_str(), bin.size());
    {
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
    const auto& hal = MetalContext::instance().hal();

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
    std::vector<uint8_t> ctrl(PROF_CTRL_WORDS * 4);
    for (uint32_t c = 0; c < num_cores; c++) {
        cluster.read_core(ctrl.data(), (uint32_t)ctrl.size(), tt_cxy_pair(device_id, vc[c]), prof_l1);
        for (int r = 0; r < NRISC; r++) {
            uint32_t tail;
            std::memcpy(&tail, ctrl.data() + (5 + r) * 4, 4);  // DEVICE_BUFFER_END_INDEX_BR_ER + r
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
    std::fflush(stdout);
    std::_Exit((total_drained == total_produced && max_out <= 512) ? 0 : 1);
}
