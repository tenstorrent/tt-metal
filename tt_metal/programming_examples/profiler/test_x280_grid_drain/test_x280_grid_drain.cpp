// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 grid drain (DMA arc): the bare-metal X280 DMA-pulls N flits from EVERY
// Tensix worker core per round, while a BRISC counter runs on every core.
//
// Flow (all via tt-metal UMD):
//   1. tt-smi -r, open the chip, grab the Cluster, read the worker grid size.
//   2. Launch brisc_counter on ALL worker cores (each spins, incrementing a u32
//      at L1 word[0] = 0x80000).
//   3. Build a physical-coord table for every core, and write a per-core identity
//      tag (0xC0DE0000 | idx) into each core's L1 word[1] (0x80004) -- the kernel
//      never touches word[1], so it survives as a "which core is this" stamp.
//   4. Boot the grid_drain FW with the coord table + params. It does one full DMA
//      setup, then per round walks all cores: reprogram NoC TLB to that core +
//      re-trigger a 256 B (4-flit) DMA into a per-core LIM slot.
//   5. Read the LIM destination back: verify slot i has tag(i) (identity) and a
//      non-zero advancing counter (liveness), and report per-round time + agg BW.
//
// Build:  make -C tools/x280_bm                               (grid_drain.bin)
//         cmake --build build_Release --target test_x280_grid_drain
// Run (repo root, libs on the path):
//   export TT_METAL_HOME=$PWD
//   export LD_LIBRARY_PATH=$(find build_Release -name '*.so*' -type f \
//       -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
//   ./build_Release/programming_examples/profiler/test_x280_grid_drain --nrounds 100
//
// Flags: --bin <grid_drain.bin> --device N --l2cpu N --pll MHZ
//        --bytes N (per core, mult of 32; default 256) --nrounds N --no-reset --no-boot

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

// ---- LIM layout (must match tools/x280_bm/src/grid_drain.c) ----
static constexpr uint64_t MBOX_PARAMS = 0x08011000ULL;
static constexpr uint64_t MBOX_RESULTS = 0x08011040ULL;
static constexpr uint64_t MBOX_COORDS = 0x08011200ULL;
static constexpr uint64_t R_DONE = MBOX_RESULTS + 0x18;
static constexpr uint64_t DONE_MAGIC = 0x6D1DD4A1E0ULL;

static constexpr uint64_t SRC_L1 = 0x80000ULL;       // counter region base in each core's L1
static constexpr uint64_t DST_BASE = 0x08012000ULL;  // per-core LIM slots

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
    std::string bin_path = "tools/x280_bm/build/grid_drain.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    uint64_t bytes = 256, nrounds = 100;
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
    bytes &= ~uint64_t(31);  // 32B-aligned (single-block re-trigger)

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
    printf(
        "[grid] worker grid %ux%u = %u cores; fetching %llu B (%llu flits)/core/round\n",
        gx,
        gy,
        num_cores,
        (unsigned long long)bytes,
        (unsigned long long)(bytes / 64));

    // ---- Launch the BRISC counter on ALL worker cores ----
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
        printf("[prod] BRISC counter launched on all %u cores\n", num_cores);
    }

    // ---- Build the physical-coord table + write a per-core identity tag ----
    std::vector<uint8_t> coords(num_cores * 8, 0);
    for (uint32_t ly = 0; ly < gy; ly++) {
        for (uint32_t lx = 0; lx < gx; lx++) {
            uint32_t idx = ly * gx + lx;
            CoreCoord logical{lx, ly};
            // Use the VIRTUAL/translated coord for BOTH the X280's NoC read and
            // the host's tag write, so they reference the identical tile (the
            // physical coord diverges from virtual past the harvested rows, and
            // the X280 NoC routes translated coords).
            CoreCoord virt =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical, CoreType::WORKER);
            pack<uint32_t>(coords, idx * 8 + 0, (uint32_t)virt.x);
            pack<uint32_t>(coords, idx * 8 + 4, (uint32_t)virt.y);
            uint32_t tag = 0xC0DE0000u | idx;
            cluster.write_core(&tag, 4, tt_cxy_pair(device_id, virt), SRC_L1 + 4);  // word[1]
        }
    }
    printf("[prod] wrote identity tags + built coord table for %u cores\n", num_cores);

    if (!do_boot) {
        printf("[done] --no-boot.\n");
        std::fflush(stdout);
        std::_Exit(0);
    }

    // ---- Boot the grid_drain FW ----
    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);
    x280.write_block(coords, MBOX_COORDS);

    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, num_cores);
    pack<uint64_t>(params, 0x08, SRC_L1);
    pack<uint64_t>(params, 0x10, bytes);
    pack<uint64_t>(params, 0x18, DST_BASE);
    pack<uint64_t>(params, 0x20, nrounds);
    x280.write_block(params, MBOX_PARAMS);

    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();
    printf("[drain] X280 draining %u cores x %llu rounds ...\n", num_cores, (unsigned long long)nrounds);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (x280.lim_rd_u64(R_DONE) == DONE_MAGIC) {
            done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!done) {
        fprintf(stderr, "[drain] timed out waiting for FW done flag\n");
        std::fflush(stdout);
        std::_Exit(1);
    }

    uint64_t rc = x280.lim_rd_u64(MBOX_RESULTS + 0x00);
    uint64_t total = x280.lim_rd_u64(MBOX_RESULTS + 0x08);
    uint32_t rounds = x280.lim_rd_u32(MBOX_RESULTS + 0x10);
    uint32_t cores = x280.lim_rd_u32(MBOX_RESULTS + 0x14);
    uint64_t round_min = x280.lim_rd_u64(MBOX_RESULTS + 0x20);
    uint32_t c0_r0 = x280.lim_rd_u32(MBOX_RESULTS + 0x28);
    uint32_t c0_rlast = x280.lim_rd_u32(MBOX_RESULTS + 0x2C);
    uint32_t fail_core = x280.lim_rd_u32(MBOX_RESULTS + 0x30);
    uint32_t fail_x = x280.lim_rd_u32(MBOX_RESULTS + 0x34);
    uint32_t fail_y = x280.lim_rd_u32(MBOX_RESULTS + 0x38);
    if (fail_core != 0xFFFFFFFFu) {
        uint32_t lx = fail_core % gx, ly = fail_core / gx;
        printf(
            "[fail] core idx %u = logical (%u,%u) -> virtual (%u,%u) DMA timed out\n",
            fail_core,
            lx,
            ly,
            fail_x,
            fail_y);
        printf("[coords] first 12 virtual: ");
        for (uint32_t i = 0; i < 12 && i < num_cores; i++) {
            uint32_t cx, cy;
            std::memcpy(&cx, coords.data() + i * 8 + 0, 4);
            std::memcpy(&cy, coords.data() + i * 8 + 4, 4);
            printf("%u:(%u,%u) ", i, cx, cy);
        }
        printf("\n");
    }

    // ---- Read the LIM destination back; verify identity + liveness ----
    std::vector<uint8_t> got(num_cores * bytes, 0);
    x280.lim_read(got.data(), (uint32_t)(num_cores * bytes), DST_BASE);
    uint32_t id_ok = 0, live = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        uint32_t w0, w1;
        std::memcpy(&w0, got.data() + i * bytes + 0, 4);  // counter
        std::memcpy(&w1, got.data() + i * bytes + 4, 4);  // identity tag
        if (w1 == (0xC0DE0000u | i)) {
            id_ok++;
        }
        if (w0 != 0) {
            live++;
        }
    }

    double rounds_d = rounds ? (double)rounds : 1.0;
    double avg_round = (double)total / rounds_d;        // cycles/round
    double per_core = avg_round / (cores ? cores : 1);  // cycles/core
    double secs = (double)total / ((double)pll * 1e6);  // total seconds
    double agg_mb = (double)cores * bytes * rounds / 1e6 / (secs > 0 ? secs : 1);

    const char* rcs = (rc == 0) ? "success" : (rc == 1) ? "DMA error" : (rc == 2) ? "timeout" : "?";
    printf("\n=== X280 grid drain (DMA, single channel, %llu B/core) ===\n", (unsigned long long)bytes);
    printf("  rc            : %llu (%s)\n", (unsigned long long)rc, rcs);
    printf("  cores x rounds: %u x %u\n", cores, rounds);
    printf("  total cycles  : %llu  (@ %d MHz)\n", (unsigned long long)total, pll);
    printf(
        "  per round     : %.0f cycles (%.1f us); fastest round %llu cycles\n",
        avg_round,
        avg_round / pll,
        (unsigned long long)round_min);
    printf("  per core      : %.0f cycles\n", per_core);
    printf("  aggregate BW  : %.0f MB/s\n", agg_mb);
    printf(
        "  identity      : %u / %u slots have correct tag%s\n",
        id_ok,
        cores,
        id_ok == cores ? "  ✓" : "  <-- MISMATCH");
    printf(
        "  liveness      : %u / %u slots nonzero; core0 counter round0=%u last=%u (delta=%u -> %s)\n",
        live,
        cores,
        c0_r0,
        c0_rlast,
        c0_rlast - c0_r0,
        c0_rlast != c0_r0 ? "advancing ✓" : "STALE");
    std::fflush(stdout);
    std::_Exit((rc == 0 && id_ok == cores) ? 0 : 1);
}
