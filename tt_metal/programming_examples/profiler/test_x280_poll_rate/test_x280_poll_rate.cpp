// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 NOC poll-rate probe (step 2). Measures how fast the bare-metal X280 can
// poll a single Tensix L1 word over the NOC -- the core pull-profiler primitive.
//
// Flow (all via tt-metal UMD, single access path):
//   1. tt-smi -r, open the chip (MeshDevice), grab the low-level Cluster.
//   2. Launch a BRISC free-running counter on logical Tensix core (0,0)
//      (non-blocking; it spins forever writing L1 0x80000).
//   3. Verify the producer: read L1 0x80000 a few times via the Cluster and
//      confirm it's advancing.
//   4. Boot the X280 poller FW (tools/x280_bm/build/poller.bin) into L2CPU LIM,
//      passing it (target physical NOC coord, L1 addr, read count) through a LIM
//      mailbox. The poller programs a 2 MiB NOC TLB window to (0,0) and reads
//      the counter as fast as it can.
//   5. Read the poller's results from LIM and print the poll rate (ns/read,
//      reads/s, MB/s) plus first/last (liveness: the X280 saw the counter move).
//
// Build:  make -C tools/x280_bm                                   (counter.bin + poller.bin)
//         cmake --build build_Release --target test_x280_poll_rate
// Run (repo root, libs on the path):
//   export TT_METAL_HOME=$PWD
//   export LD_LIBRARY_PATH=$(find build_Release -name '*.so*' -type f \
//       -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
//   ./build_Release/programming_examples/profiler/test_x280_poll_rate
//
// Flags: --bin <poller.bin> --device N --l2cpu N --pll MHZ --nreads N
//        --core X Y (producer logical core, default 0 0) --no-reset --no-boot

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
using tt::tt_metal::MetalContext;
using tt::tt_metal::distributed::MeshDevice;

// ---- X280 / boot constants (see test_x280_counter) ----
static constexpr uint64_t LIM_BASE = 0x08000000ULL;
static constexpr uint64_t RESET_UNIT_BASE = 0x80030000ULL;
static constexpr uint64_t L2CPU_RESET_REG = RESET_UNIT_BASE + 0x14;
static constexpr uint64_t L2CPU_REG_BASE = 0xFFFFF7FEFFF10000ULL;
static constexpr uint64_t PLL4_BASE = 0x80020500ULL;
static constexpr uint64_t PLL_CNTL_1 = PLL4_BASE + 0x4;
static constexpr uint64_t PLL_CNTL_5 = PLL4_BASE + 0x14;

// ---- LIM mailbox (must match tools/x280_bm/src/poller.c) ----
static constexpr uint64_t MBOX_PARAMS = 0x08011000ULL;
static constexpr uint64_t MBOX_RESULTS = 0x08011040ULL;
static constexpr uint64_t R_DONE = MBOX_RESULTS + 0x18;
static constexpr uint64_t DONE_MAGIC = 0xD09EC0FFEEULL;

// ---- Tensix producer L1 counter address (must match brisc_counter.cpp) ----
static constexpr uint64_t COUNTER_ADDR = 0x80000ULL;

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
    std::string bin_path = "tools/x280_bm/build/poller.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    int prod_x = 0, prod_y = 0;
    uint64_t nreads = 2000000;
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
        } else if (a == "--nreads") {
            nreads = std::stoull(next());
        } else if (a == "--core") {
            prod_x = std::stoi(next());
            prod_y = std::stoi(next());
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

    // ---- Step 2: launch the BRISC free-running counter on the producer core ----
    CoreCoord prod_logical{(size_t)prod_x, (size_t)prod_y};
    {
        Program program = CreateProgram();
        std::map<std::string, std::string> defs = {{"COUNTER_ADDR", std::to_string(COUNTER_ADDR)}};
        CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_x280_poll_rate/kernels/brisc_counter.cpp",
            prod_logical,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defs});
        distributed::MeshWorkload workload;
        auto range = distributed::MeshCoordinateRange(mesh->shape());
        workload.add_program(range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, /*blocking=*/false);
        printf(
            "[prod] launched BRISC counter on logical (%d,%d), L1 0x%llx\n",
            prod_x,
            prod_y,
            (unsigned long long)COUNTER_ADDR);
    }

    // ---- Step 3: verify the producer is advancing (host reads its L1) ----
    CoreCoord prod_virt =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, prod_logical, CoreType::WORKER);
    CoreCoord prod_phys =
        cluster.get_physical_coordinate_from_logical_coordinates(device_id, prod_logical, CoreType::WORKER);
    tt_cxy_pair prod_v(device_id, prod_virt);
    printf(
        "[prod] logical (%d,%d) -> physical (%u,%u), virtual (%u,%u)\n",
        prod_x,
        prod_y,
        (unsigned)prod_phys.x,
        (unsigned)prod_phys.y,
        (unsigned)prod_virt.x,
        (unsigned)prod_virt.y);
    uint32_t pv_prev = 0;
    bool advanced = false;
    for (int s = 0; s < 12; s++) {
        uint32_t pv = 0;
        cluster.read_core(&pv, 4, prod_v, COUNTER_ADDR);
        printf("[prod] sample %2d: %u\n", s, pv);
        if (s > 0 && pv != pv_prev) {
            advanced = true;
        }
        pv_prev = pv;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    printf("[prod] %s\n", advanced ? "advancing OK" : "NOT advancing!");

    if (!do_boot) {
        printf("[done] --no-boot: producer verification only.\n");
        std::fflush(stdout);
        std::_Exit(0);  // skip device close: the spinning producer would hang teardown
    }

    // ---- Step 4: boot the X280 poller, pass it the target via LIM mailbox ----
    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);

    std::vector<uint8_t> params(64, 0);
    pack<uint32_t>(params, 0x00, (uint32_t)prod_phys.x);
    pack<uint32_t>(params, 0x04, (uint32_t)prod_phys.y);
    pack<uint64_t>(params, 0x08, COUNTER_ADDR);
    pack<uint64_t>(params, 0x10, nreads);
    x280.write_mailbox(params, MBOX_PARAMS);

    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();
    printf(
        "[poll] X280 polling Tensix physical (%u,%u) L1 0x%llx, %llu reads ...\n",
        (unsigned)prod_phys.x,
        (unsigned)prod_phys.y,
        (unsigned long long)COUNTER_ADDR,
        (unsigned long long)nreads);

    // ---- Step 5: wait for the poller, read + report results ----
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (x280.lim_rd_u64(R_DONE) == DONE_MAGIC) {
            done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!done) {
        fprintf(stderr, "[poll] timed out waiting for poller done flag\n");
        return 1;
    }

    uint64_t reads = x280.lim_rd_u64(MBOX_RESULTS + 0x00);
    uint64_t cycles = x280.lim_rd_u64(MBOX_RESULTS + 0x08);
    uint32_t first = x280.lim_rd_u32(MBOX_RESULTS + 0x10);
    uint32_t last = x280.lim_rd_u32(MBOX_RESULTS + 0x14);

    double ns_per_read = (double)cycles * 1000.0 / (double)pll / (double)reads;  // cycles @ pll MHz
    double reads_per_s = 1e9 / ns_per_read;
    double mb_per_s = reads_per_s * 4.0 / 1e6;

    printf("\n=== X280 poll rate (single uncached u32 reads over NOC) ===\n");
    printf("  reads        : %llu\n", (unsigned long long)reads);
    printf("  cycles       : %llu  (@ %d MHz X280 PLL)\n", (unsigned long long)cycles, pll);
    printf("  ns / read    : %.2f\n", ns_per_read);
    printf("  reads / sec  : %.2f M\n", reads_per_s / 1e6);
    printf("  bandwidth    : %.1f MB/s\n", mb_per_s);
    printf(
        "  liveness     : first=%u last=%u (delta=%u -> %s)\n",
        first,
        last,
        last - first,
        last != first ? "saw live counter" : "STALE!");
    std::fflush(stdout);
    std::_Exit(0);  // skip device close: the spinning producer would hang teardown
}
