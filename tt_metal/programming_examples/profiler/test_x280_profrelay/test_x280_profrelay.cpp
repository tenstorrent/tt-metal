// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// CLOSE THE LOOP: a normal workload makes all 5 RISCs on all cores emit device
// profiler FW+kernel timestamps into L1; the X280 (bare-metal) drains those L1
// profiler buffers over the NoC and relays them to host pinned memory through the
// PCIe tile -- replacing tt-metal's profiler readback. The host then validates the
// relayed bytes against a direct read of one core's profiler buffer (ground truth)
// and parses the guaranteed FW/kernel markers for every core x RISC.
//
// Build:  make -C tools/x280_bm                               (profrelay.bin)
//         cmake --build build_Release --target test_x280_profrelay
// Run:    TT_METAL_DEVICE_PROFILER=1 \
//         ./build_Release/programming_examples/profiler/test_x280_profrelay --loop 1
//
// Flags: --bin <profrelay.bin> --device N --l2cpu N --pll MHZ --loop N --no-reset

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
static constexpr uint64_t DONE_MAGIC = 0x9D0F11A1DEULL;
static constexpr uint64_t FOOTER_MAGIC = 0xF007D09F11E12345ULL;
static constexpr uint64_t WIN_STRIDE = 0x200000ULL;  // 2 MiB host write window

// profiler_msg_t layout (Blackhole Tensix): control_vector[32] (128B) + 5 RISC
// buffers of 2048B; guaranteed markers at word idx 4/5,6/7,8/9,10/11 in each buf.
static constexpr uint64_t PROF_CTRL_BYTES = 128;
static constexpr uint64_t PROF_RISC_BYTES = 2048;
static constexpr int PROF_NRISC = 5;
static constexpr uint64_t PROF_MSG_BYTES = PROF_CTRL_BYTES + PROF_NRISC * PROF_RISC_BYTES;  // 10368
static constexpr uint64_t BYTES_PER_CORE = 10496;  // PROF_MSG_BYTES rounded up to 256
static const char* RISC_NAME[PROF_NRISC] = {"BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"};

static uint64_t res_off(int w) { return MBOX_RESULTS + (uint64_t)0 * 0x40 + w; }

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

// Reconstruct a 44-bit timestamp from a guaranteed marker pair at word index wi.
static inline bool marker_valid(const uint8_t* core_base, int risc, int wi, uint64_t& ts_out) {
    const uint32_t* w =
        reinterpret_cast<const uint32_t*>(core_base + PROF_CTRL_BYTES + (uint64_t)risc * PROF_RISC_BYTES) + wi;
    uint32_t H = w[0], L = w[1];
    ts_out = ((uint64_t)(H & 0xFFF) << 32) | L;
    return (H & 0x80000000u) != 0;  // sentinel bit set => marker written
}

int main(int argc, char** argv) {
    using namespace tt::tt_metal;
    std::string bin_path = "tools/x280_bm/build/profrelay.bin";
    int device_id = 0, l2cpu = 0, pll = 1000, loop = 1;
    bool do_reset = true;

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
        } else if (a == "--no-reset") {
            do_reset = false;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }
    if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
        fprintf(stderr, "[warn] TT_METAL_DEVICE_PROFILER not set -- kernels won't emit markers. Set it to 1.\n");
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
    const auto& hal = MetalContext::instance().hal();

    CoreCoord grid = mesh->compute_with_storage_grid_size();
    uint32_t gx = (uint32_t)grid.x, gy = (uint32_t)grid.y;
    uint32_t num_cores = gx * gy;
    printf(
        "[grid] worker grid %ux%u = %u cores; launching all-5-RISC profiler workload (loop=%d)\n",
        gx,
        gy,
        num_cores,
        loop);

    // --- launch a workload so all 5 RISCs on all cores emit FW+kernel markers ---
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
        distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, /*blocking=*/true);
        printf("[prod] workload finished; FW+kernel markers now in each core's L1 profiler buffer\n");
    }

    // profiler_msg_t L1 address (NOT hardcoded -- same accessor the metal readback uses)
    uint64_t prof_l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::PROFILER);
    printf(
        "[src] profiler_msg_t @ L1 0x%lx  (%lu B/core relayed)\n",
        (unsigned long)prof_l1,
        (unsigned long)BYTES_PER_CORE);

    // translated Tensix coords for every worker core (X280 reads these)
    std::vector<uint8_t> coords(num_cores * 8, 0);
    std::vector<CoreCoord> virt_coords(num_cores);
    for (uint32_t ly = 0; ly < gy; ly++) {
        for (uint32_t lx = 0; lx < gx; lx++) {
            uint32_t idx = ly * gx + lx;
            CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord{lx, ly}, CoreType::WORKER);
            virt_coords[idx] = v;
            pack<uint32_t>(coords, idx * 8 + 0, (uint32_t)v.x);
            pack<uint32_t>(coords, idx * 8 + 4, (uint32_t)v.y);
        }
    }

    // --- D2H sysmem target (same addressing as test_x280_d2hbw) ---
    auto pcie_cores = cluster.get_soc_desc(device_id).get_cores(tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED);
    if (pcie_cores.empty()) {
        fprintf(stderr, "no PCIE cores\n");
        return 1;
    }
    auto pc = pcie_cores.front();
    uint32_t pcie_enc = ((uint32_t)pc.x & 0x3f) | (((uint32_t)pc.y & 0x3f) << 6);
    uint64_t pcie_base = cluster.get_pcie_base_addr_from_device(device_id);
    uint64_t chan_sz = cluster.get_host_channel_size(device_id, 0);
    uint64_t need = (uint64_t)num_cores * BYTES_PER_CORE;
    if (need + 64 > WIN_STRIDE) {
        fprintf(stderr, "relay region %lu B exceeds one 2MB window; reduce cores or split\n", (unsigned long)need);
        return 1;
    }
    uint64_t data_off = (chan_sz / 2) & ~(WIN_STRIDE - 1);
    uint64_t host_base = pcie_base + data_off;
    if (host_base & (WIN_STRIDE - 1)) {
        fprintf(stderr, "host_base not 2MB aligned\n");
        return 1;
    }
    printf(
        "[d2h] PCIe enc=0x%x host_base=0x%lx (sysmem off 0x%lx)\n",
        pcie_enc,
        (unsigned long)host_base,
        (unsigned long)data_off);

    // zero the footer slot so a stale value can't false-trigger
    {
        uint64_t z = 0;
        cluster.write_sysmem(&z, sizeof(z), data_off + need, device_id, 0);
    }

    // --- boot the relay FW ---
    uint64_t nonce = 0x9D0F0000ULL | num_cores;
    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);
    x280.write_block(coords, MBOX_COORDS);
    std::vector<uint8_t> params(64, 0);
    pack<uint64_t>(params, 0x00, (uint64_t)pcie_enc);
    pack<uint64_t>(params, 0x08, host_base);
    pack<uint64_t>(params, 0x10, prof_l1);
    pack<uint64_t>(params, 0x18, BYTES_PER_CORE);
    pack<uint64_t>(params, 0x20, (uint64_t)num_cores);
    pack<uint64_t>(params, 0x28, 4);  // ilp
    pack<uint64_t>(params, 0x30, nonce);
    x280.write_block(params, MBOX_PARAMS);
    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();
    printf("[relay] X280 draining %u cores' profiler L1 -> host D2H ...\n", num_cores);

    // wait done flag + footer in sysmem
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (x280.lim_rd_u64(res_off(0x18)) == DONE_MAGIC) {
            uint64_t foot = 0;
            cluster.read_sysmem(&foot, sizeof(foot), data_off + need, device_id, 0);
            if (foot == FOOTER_MAGIC) {
                done = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!done) {
        fprintf(stderr, "[relay] timed out waiting for done flag + footer\n");
        std::fflush(stdout);
        std::_Exit(1);
    }

    uint64_t cyc = x280.lim_rd_u64(res_off(0x00));
    uint64_t bytes = x280.lim_rd_u64(res_off(0x08));
    double mbps = cyc ? (double)bytes / 1e6 / ((double)cyc / ((double)pll * 1e6)) : 0.0;
    printf("[relay] done: %lu B in %lu cycles -> %.0f MB/s relay\n", (unsigned long)bytes, (unsigned long)cyc, mbps);

    // read the relayed region out of host pinned memory
    std::vector<uint8_t> relayed(need, 0);
    cluster.read_sysmem(relayed.data(), (uint32_t)need, data_off, device_id, 0);

    // --- ground-truth cross-check: read core 0's profiler buffer DIRECTLY over NoC ---
    std::vector<uint8_t> truth(PROF_MSG_BYTES, 0);
    cluster.read_core(truth.data(), (uint32_t)PROF_MSG_BYTES, tt_cxy_pair(device_id, virt_coords[0]), prof_l1);
    bool gt_match = std::memcmp(truth.data(), relayed.data(), PROF_MSG_BYTES) == 0;
    printf("[verify] relayed core-0 vs direct read_core: %s\n", gt_match ? "EXACT MATCH ✓" : "MISMATCH <--");

    // --- parse guaranteed FW/kernel markers for every core x RISC ---
    uint32_t valid_pairs = 0, total_pairs = num_cores * PROF_NRISC;
    for (uint32_t c = 0; c < num_cores; c++) {
        const uint8_t* cb = relayed.data() + (uint64_t)c * BYTES_PER_CORE;
        for (int r = 0; r < PROF_NRISC; r++) {
            uint64_t fs = 0, fe = 0, ks = 0, ke = 0;
            bool v = marker_valid(cb, r, 4, fs) && marker_valid(cb, r, 6, fe) && marker_valid(cb, r, 8, ks) &&
                     marker_valid(cb, r, 10, ke);
            if (v && fs <= ks && ks <= ke && ke <= fe) {
                valid_pairs++;
            }
        }
    }

    // sample: core 0, all 5 RISCs, FW + kernel durations (cycles)
    printf("\n=== relayed profiler timestamps (core 0 sample) ===\n");
    const uint8_t* c0 = relayed.data();
    for (int r = 0; r < PROF_NRISC; r++) {
        uint64_t fs, fe, ks, ke;
        marker_valid(c0, r, 4, fs);
        marker_valid(c0, r, 6, fe);
        marker_valid(c0, r, 8, ks);
        marker_valid(c0, r, 10, ke);
        printf(
            "  %-6s  FW[%llu..%llu] %llu cyc | KERNEL[%llu..%llu] %llu cyc\n",
            RISC_NAME[r],
            (unsigned long long)fs,
            (unsigned long long)fe,
            (unsigned long long)(fe - fs),
            (unsigned long long)ks,
            (unsigned long long)ke,
            (unsigned long long)(ke - ks));
    }

    // dump the relayed buffer for downstream Tracy ingestion (increment 3)
    const char* pdir = std::getenv("TT_METAL_PROFILER_DIR");
    std::string out = (pdir ? std::string(pdir) : std::string("/tmp")) + "/x280_relayed_profiler.bin";
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(relayed.data()), (std::streamsize)need);
    of.close();

    printf("  -------------------------------------------------\n");
    printf(
        "  cores x RISCs with valid FW+kernel markers : %u / %u%s\n",
        valid_pairs,
        total_pairs,
        valid_pairs == total_pairs ? "  ✓" : "  <-- some missing");
    printf("  relayed bytes dumped to                    : %s\n", out.c_str());
    std::fflush(stdout);
    std::_Exit((gt_match && valid_pairs == total_pairs) ? 0 : 1);
}
