// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole power virus: runs constant load in four domains, each enabled and
// throttled on its own, until a duration elapses or Ctrl-C.
//
//   TRISC  – MVMUL loop from pre-filled L1 (the MaxUtil compute kernel)
//   DRAM   – BRISC on every worker reads round-robin across all DRAM banks
//   NOC    – NCRISC posted L1-to-L1 writes along long row/column ring routes
//   ETH    – active ERISC0 bidirectional traffic on every link between opened chips
//
// Each Tensix core has a small L1 control block (kernels/pv_common.hpp). The
// host launches the programs without waiting, sleeps, then writes STOP to every
// core and waits for the programs to finish. Every generator writes a
// [elapsed, count] record that the host turns into utilization and bandwidth.
//
// On each Tensix core BRISC and NCRISC use opposite NOCs: in dedicated NOC mode
// they share command buffer 0 and the transaction counters of a NOC.
//
// Knobs are environment variables, documented in tests/didt/README.md.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_metal/test_utils/stimulus.hpp>
#include <distributed/mesh_device_impl.hpp>
#include "multi_device_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"
#include "impl/buffers/buffer_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/tt_cluster.hpp"

#include "kernels/pv_common.hpp"
#include "noc_virus_routes.hpp"

namespace tt::tt_metal {

namespace unit_tests::didt::power_virus {

using std::shared_ptr;

constexpr const char* kKernelDir = "tests/didt/power_virus/kernels/";
constexpr uint32_t kNumTiles = 8;
constexpr uint32_t kTileBytes = 32 * 32 * 2;  // float16_b
constexpr uint32_t kUsefulFpuCyclesPerTile = 8 * 16;
constexpr uint32_t kDramPagesPerBank = 16;
constexpr uint32_t kDramMaxXferBytes = 16384;
constexpr uint32_t kNocMaxXferBytes = 65536;
constexpr uint32_t kEthMaxXferBytes = 8192;
constexpr uint32_t kEthDefaultSlots = 8;
constexpr uint32_t kEthTimingPageBytes = 64;
constexpr uint32_t kDefaultPeriodUnits = 256;  // ~97 µs at 1.35 GHz
constexpr uint32_t kMaxIdleUnits = 1u << 20;

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------

std::optional<int64_t> parse_int(const char* text) {
    const std::string_view value(text);
    int64_t parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return std::nullopt;
    }
    return parsed;
}

/// Reads an integer env var in [lo, hi]; nullopt (with a warning if set but invalid) otherwise.
std::optional<uint32_t> env_u32(const std::string& name, uint32_t lo, uint32_t hi) {
    const char* env = std::getenv(name.c_str());
    if (env == nullptr) {
        return std::nullopt;
    }
    const auto val = parse_int(env);
    if (val.has_value() && *val >= lo && *val <= hi) {
        return static_cast<uint32_t>(*val);
    }
    log_warning(LogTest, "{}='{}' is not an integer in [{}, {}] – ignored", name, env, lo, hi);
    return std::nullopt;
}

bool is_pow2(uint32_t v) { return v != 0 && (v & (v - 1)) == 0; }

/// One generator's resolved knobs. Kernels take them as compile-time args.
struct Throttle {
    bool on = false;
    uint32_t pct = 0;
    uint32_t xfer_bytes = 0;    // 0 for TRISC
    uint32_t burst = 0;         // raw mode: units per burst (TRISC: loops of 8 tiles)
    uint32_t idle_units = 0;    // raw mode: idle after each burst, units of 512 ticks
    uint32_t period_units = 0;  // phase-locked mode; 0 = off
    uint32_t active_units = 0;

    std::string describe() const {
        if (!on) {
            return "off";
        }
        std::string s = xfer_bytes != 0 ? fmt::format("{} B transfers, ", xfer_bytes) : "";
        if (period_units != 0) {
            return s + fmt::format("{}% phase-locked ({}/{} × 512 ticks)", pct, active_units, period_units);
        }
        if (burst != 0) {
            return s + fmt::format("burst {} then idle {} × 512 ticks", burst, idle_units);
        }
        return s + "continuous";
    }
};

/// <prefix> = utilization % (0 = off); <prefix>_BURST / _IDLE_512 select raw
/// burst/idle mode; otherwise <prefix>_PERIOD_512 sets the
/// phase-locked period. max_xfer == 0 means the generator has no transfer size.
Throttle read_throttle(
    const std::string& prefix, uint32_t default_pct, uint32_t default_xfer, uint32_t min_xfer, uint32_t max_xfer) {
    Throttle t;
    t.pct = env_u32(prefix, 0, 100).value_or(default_pct);
    if (max_xfer != 0) {
        t.xfer_bytes = default_xfer;
        if (auto v = env_u32(prefix + "_XFER_BYTES", 0, max_xfer)) {
            if (*v == 0 || (is_pow2(*v) && *v >= min_xfer)) {
                t.xfer_bytes = *v;
            } else {
                log_warning(
                    LogTest,
                    "{}_XFER_BYTES={} must be 0 or a power of two in [{}, {}] – ignored",
                    prefix,
                    *v,
                    min_xfer,
                    max_xfer);
            }
        }
        if (t.xfer_bytes == 0) {
            t.pct = 0;
        }
    }
    t.on = t.pct != 0;
    if (!t.on) {
        return t;
    }

    const auto burst = env_u32(prefix + "_BURST", 0, 1u << 24);
    const auto idle = env_u32(prefix + "_IDLE_512", 0, kMaxIdleUnits);
    if (burst.has_value() || idle.has_value()) {
        t.burst = burst.value_or(0);
        t.idle_units = t.burst == 0 ? 0 : idle.value_or(0);
        if (t.pct != 100) {
            log_warning(LogTest, "{}={} is ignored because {}_BURST/_IDLE_512 are set", prefix, t.pct, prefix);
        }
        return t;
    }
    if (t.pct < 100) {
        t.period_units = kDefaultPeriodUnits;
        if (auto p = env_u32(prefix + "_PERIOD_512", 2, 1u << 20)) {
            if (is_pow2(*p)) {
                t.period_units = *p;
            } else {
                log_warning(LogTest, "{}_PERIOD_512={} is not a power of two – using {}", prefix, *p, t.period_units);
            }
        }
        t.active_units = std::max<uint32_t>(1, (t.period_units * t.pct + 50) / 100);
    }
    return t;
}

struct PvConfig {
    uint32_t duration_s = 10;  // 0 = until Ctrl-C
    bool super_sync = false;
    Throttle trisc;
    Throttle dram;
    Throttle noc;
    Throttle eth;
    uint32_t eth_startup_delay_ticks = 0;
};

PvConfig read_config() {
    PvConfig cfg;
    cfg.duration_s = env_u32("PV_DURATION_S", 0, 1u << 30).value_or(10);
    if (const char* s = std::getenv("PV_SUPER_SYNC")) {
        const std::string v(s);
        cfg.super_sync = !v.empty() && v != "0" && v != "false";
    }
    cfg.trisc = read_throttle("PV_TRISC", 100, 0, 0, 0);
    cfg.dram = read_throttle("PV_DRAM", 0, 8192, 64, kDramMaxXferBytes);
    cfg.noc = read_throttle("PV_NOC", 0, kNocMaxXferBytes, 64, kNocMaxXferBytes);
    cfg.eth = read_throttle("PV_ETH", 0, kEthMaxXferBytes, 64, kEthMaxXferBytes);
    cfg.eth_startup_delay_ticks = env_u32("PV_ETH_STARTUP_DELAY_TICKS", 0, UINT32_MAX).value_or(0);
    if (cfg.super_sync && !cfg.trisc.on && !cfg.dram.on && !cfg.noc.on) {
        cfg.super_sync = false;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Per-device setup
// ---------------------------------------------------------------------------

struct NocRoute {
    uint32_t noc;
    CoreCoord dst;  // logical
};

struct EthLink {
    CoreCoord core;  // logical ETH core
    ChipId peer_chip;
    CoreCoord peer_core;
    uint32_t role;  // 0 or 1; opposite on the peer
};

struct DeviceRun {
    shared_ptr<distributed::MeshDevice> mesh_device;
    IDevice* device = nullptr;
    CoreRange grid{{0, 0}, {0, 0}};

    // Tensix L1 layout
    uint32_t ctrl = 0;
    uint32_t in0 = 0;
    uint32_t in1 = 0;
    uint32_t out = 0;
    uint32_t dram_staging = 0;
    uint32_t noc_buffer = 0;

    uint32_t num_banks = 0;
    shared_ptr<Buffer> dram_buffer;
    std::map<CoreCoord, NocRoute> routes;  // source core -> route

    std::vector<EthLink> links;
    uint32_t eth_l1_base = 0;
    uint32_t eth_num_slots = kEthDefaultSlots;
    shared_ptr<Buffer> eth_timing_buffer;

    Program program;
};

std::vector<uint32_t> random_bf16_tiles(uint32_t num_tiles, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    std::vector<bfloat16> values(num_tiles * 32 * 32);
    std::generate(values.begin(), values.end(), [&]() { return bfloat16(dis(gen)); });
    return tt::test_utils::pack_vector<uint32_t, bfloat16>(values);
}

/// xorshift pattern for the NOC buffer, so transfers toggle data.
std::vector<uint32_t> xorshift_words(uint32_t num_words, uint32_t seed) {
    std::vector<uint32_t> words(num_words);
    uint32_t rng = seed;
    for (auto& w : words) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        w = rng;
    }
    return words;
}

void layout_l1(DeviceRun& run, const PvConfig& cfg) {
    const auto& hal = MetalContext::instance().hal();
    const uint32_t base = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t size = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    auto align = [](uint32_t a) { return (a + 63) & ~63u; };
    uint32_t addr = align(base);
    run.ctrl = addr;
    addr += pv::kControlBytes;
    run.in0 = align(addr);
    run.in1 = run.in0 + kNumTiles * kTileBytes;
    run.out = run.in1 + kNumTiles * kTileBytes;
    run.dram_staging = align(run.out + kNumTiles * kTileBytes);
    run.noc_buffer = align(run.dram_staging + kDramMaxXferBytes);
    const uint32_t end = run.noc_buffer + std::max<uint32_t>(cfg.noc.xfer_bytes, 64);
    TT_FATAL(end <= base + size, "Power-virus L1 layout ({} B) exceeds unreserved L1 ({} B)", end - base, size);
}

/// Writes the control block, the MVMUL inputs and the NOC buffer to every worker.
void init_tensix_l1(DeviceRun& run, const PvConfig& cfg) {
    std::vector<uint32_t> ctrl(pv::kControlBytes / sizeof(uint32_t), 0);
    ctrl[pv::kGo / 4] = cfg.super_sync ? 0 : 1;
    ctrl[pv::kFinal / 4] = pv::kFinalUnset;

    std::vector<uint32_t> inputs = random_bf16_tiles(kNumTiles, 42);
    const auto in1 = random_bf16_tiles(kNumTiles, 43);
    inputs.insert(inputs.end(), in1.begin(), in1.end());
    const auto noc_words = cfg.noc.on ? xorshift_words(cfg.noc.xfer_bytes / 4, 0xaaaaaaaau) : std::vector<uint32_t>{};

    for (const auto& core : run.grid) {
        detail::WriteToDeviceL1(run.device, core, run.ctrl, ctrl);
        if (cfg.trisc.on) {
            detail::WriteToDeviceL1(run.device, core, run.in0, inputs);
        }
        if (cfg.noc.on) {
            auto words = noc_words;
            detail::WriteToDeviceL1(run.device, core, run.noc_buffer, words);
        }
    }
}

void setup_dram(DeviceRun& run, const PvConfig& cfg) {
    if (!cfg.dram.on) {
        return;
    }
    run.num_banks = run.device->allocator()->get_num_banks(BufferType::DRAM);
    const uint32_t xfer = cfg.dram.xfer_bytes;
    run.dram_buffer = BufferImpl::create(run.device, run.num_banks * kDramPagesPerBank * xfer, xfer, BufferType::DRAM);
    std::vector<uint32_t> pattern(run.dram_buffer->size() / sizeof(uint32_t));
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = (i & 1) ? 0xA5A5A5A5u : 0x5A5A5A5Au;  // toggles every bit
    }
    detail::WriteToBuffer(run.dram_buffer, pattern);
}

/// Maps the NOC routes (physical NOC0 coordinates) onto this device's
/// worker grid. Routes whose destination is outside the grid are dropped.
void setup_noc_routes(DeviceRun& run, const PvConfig& cfg) {
    if (!cfg.noc.on) {
        return;
    }
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(run.device->id());
    std::map<std::pair<uint32_t, uint32_t>, CoreCoord> noc0_to_logical;
    std::map<CoreCoord, std::pair<uint32_t, uint32_t>> logical_to_noc0;
    uint32_t live = 0;
    for (const auto& core : run.grid) {
        const auto n = soc.translate_coord_to(
            {core.x, core.y, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL}, tt::CoordSystem::NOC0);
        noc0_to_logical[{n.x, n.y}] = core;
        logical_to_noc0[core] = {n.x, n.y};
        live |= 1u << n.x;
    }
    uint32_t per_noc[2] = {0, 0};
    for (const auto& core : run.grid) {
        const auto [x, y] = logical_to_noc0.at(core);
        noc_virus::Route r{};
        if (!noc_virus::find_route(live, x, y, &r)) {
            continue;
        }
        const auto dst = noc0_to_logical.find({r.dst_x, r.dst_y});
        if (dst == noc0_to_logical.end()) {
            continue;
        }
        run.routes[core] = {r.noc, dst->second};
        per_noc[r.noc]++;
    }
    log_info(
        LogTest,
        "Device {}: NOC virus – {} routes ({} on NOC0, {} on NOC1), {}",
        run.device->id(),
        run.routes.size(),
        per_noc[0],
        per_noc[1],
        cfg.noc.describe());
}

/// Finds every link that is up and whose peer chip is also being run.
void setup_eth_links(DeviceRun& run, const PvConfig& cfg, const std::set<ChipId>& dispatched_chips) {
    if (!cfg.eth.on) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const ChipId chip = run.device->id();
    uint32_t links_up = 0;
    for (const auto& core : run.device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true)) {
        if (!cluster.is_ethernet_link_up(chip, core)) {
            continue;
        }
        ++links_up;
        // Skip links that leave this cluster (e.g. Galaxy exit links).
        const auto chan = cluster.get_soc_desc(chip).logical_eth_core_to_chan_map.at(core);
        const auto& internal = cluster.get_ethernet_connections();
        if (!internal.contains(chip) || !internal.at(chip).contains(chan)) {
            continue;
        }
        const auto [peer_chip, peer_core] = run.device->get_connected_ethernet_core(core);
        if (!dispatched_chips.contains(peer_chip)) {
            continue;
        }
        const bool lower =
            chip < peer_chip ||
            (chip == peer_chip && (core.x < peer_core.x || (core.x == peer_core.x && core.y < peer_core.y)));
        run.links.push_back({core, peer_chip, peer_core, lower ? 0u : 1u});
    }
    if (run.links.empty()) {
        log_info(
            LogTest,
            "Device {}: ETH link traffic skipped – {} link(s) up, none to a chip in this run{}",
            chip,
            links_up,
            dispatched_chips.size() == 1 ? " (use PowerVirus_AllDevices)" : "");
        return;
    }

    const auto& hal = MetalContext::instance().hal();
    run.eth_l1_base = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t eth_l1_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    auto l1_bytes = [&](uint32_t slots) { return pv::kEthHeaderBytes + 2 * slots * (cfg.eth.xfer_bytes + 16); };
    while (run.eth_num_slots > 2 && l1_bytes(run.eth_num_slots) > eth_l1_size) {
        run.eth_num_slots /= 2;
    }
    TT_FATAL(
        l1_bytes(run.eth_num_slots) <= eth_l1_size,
        "ETH link buffers ({} B) do not fit in {} B of ETH L1; reduce PV_ETH_XFER_BYTES",
        l1_bytes(run.eth_num_slots),
        eth_l1_size);

    run.eth_timing_buffer =
        BufferImpl::create(run.device, run.links.size() * kEthTimingPageBytes, kEthTimingPageBytes, BufferType::DRAM);
    std::vector<uint32_t> zeros(run.eth_timing_buffer->size() / sizeof(uint32_t), 0);
    detail::WriteToBuffer(run.eth_timing_buffer, zeros);
    for (const auto& link : run.links) {
        std::vector<uint32_t> stop{0};
        detail::WriteToDeviceL1(run.device, link.core, run.eth_l1_base + pv::kEthStop, stop, CoreType::ETH);
    }
    log_info(
        LogTest,
        "Device {}: ETH links – {} of {} up, {} slots, {}",
        chip,
        run.links.size(),
        links_up,
        run.eth_num_slots,
        cfg.eth.describe());
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

std::vector<uint32_t> throttle_args(const Throttle& t) {
    return {t.burst, t.idle_units, t.period_units, t.active_units};
}

void build_program(DeviceRun& run, const PvConfig& cfg) {
    IDevice* device = run.device;
    run.program = CreateProgram();
    Program& program = run.program;
    const CoreRangeSet all_cores(run.grid);

    if (cfg.trisc.on) {
        CreateKernel(
            program,
            std::string(kKernelDir) + "pv_compute.cpp",
            all_cores,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .compile_args =
                    {run.in0,
                     run.in1,
                     run.out,
                     kNumTiles,
                     run.ctrl,
                     cfg.super_sync ? 1u : 0u,
                     cfg.trisc.period_units,
                     cfg.trisc.active_units,
                     cfg.trisc.burst,
                     cfg.trisc.idle_units},
            });
    }

    // NOC per core: NCRISC follows its route; BRISC takes the other NOC. Cores
    // without a route split BRISC across NOCs in a checkerboard.
    std::set<CoreRange> brisc_cores[2];
    std::set<CoreRange> ncrisc_cores[2];
    std::map<CoreCoord, uint32_t> brisc_noc;
    for (const auto& core : run.grid) {
        const auto route = run.routes.find(core);
        uint32_t b = (core.x + core.y) & 1;
        if (route != run.routes.end()) {
            ncrisc_cores[route->second.noc].insert(CoreRange(core, core));
            b = 1 - route->second.noc;
        }
        brisc_noc[core] = b;
        brisc_cores[b].insert(CoreRange(core, core));
    }

    // BRISC: super-sync barrier and/or DRAM reads.
    if (cfg.super_sync || cfg.dram.on) {
        uint32_t sender_sem = 0;
        uint32_t receiver_sem = 0;
        if (cfg.super_sync) {
            sender_sem = CreateSemaphore(program, all_cores, INVALID);
            receiver_sem = CreateSemaphore(program, all_cores, INVALID);
        }
        std::vector<uint32_t> args = {
            run.ctrl,
            cfg.super_sync ? 1u : 0u,
            cfg.trisc.on ? 1u : 0u,
            sender_sem,
            receiver_sem,
            cfg.dram.on ? 1u : 0u,
            std::max<uint32_t>(run.num_banks, 1),
            kDramPagesPerBank,
            std::max<uint32_t>(cfg.dram.xfer_bytes, 64),
        };
        const auto t = throttle_args(cfg.dram);
        args.insert(args.end(), t.begin(), t.end());

        const CoreCoord sender = run.grid.start_coord;
        const CoreCoord sender_v = device->worker_core_from_logical_core(sender);
        const CoreCoord tl = device->worker_core_from_logical_core(run.grid.start_coord);
        const CoreCoord br = device->worker_core_from_logical_core(run.grid.end_coord);
        // Multicast rectangles are given end-first on NOC1.
        const bool sender_noc1 = brisc_noc.at(sender) == 1;
        const CoreCoord mc_start = sender_noc1 ? br : tl;
        const CoreCoord mc_end = sender_noc1 ? tl : br;
        const uint32_t num_dests = run.grid.size() - 1;

        uint32_t core_idx = 0;
        for (uint32_t noc = 0; noc < 2; ++noc) {
            if (brisc_cores[noc].empty()) {
                continue;
            }
            auto kernel = CreateKernel(
                program,
                std::string(kKernelDir) + "pv_brisc.cpp",
                CoreRangeSet(brisc_cores[noc]),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = noc == 0 ? NOC::NOC_0 : NOC::NOC_1,
                    .compile_args = args,
                });
            for (const auto& range : brisc_cores[noc]) {
                const CoreCoord core = range.start_coord;
                const CoreCoord v = device->worker_core_from_logical_core(core);
                SetRuntimeArgs(
                    program,
                    kernel,
                    core,
                    {core == sender ? 1u : 0u,
                     static_cast<uint32_t>(sender_v.x),
                     static_cast<uint32_t>(sender_v.y),
                     static_cast<uint32_t>(mc_start.x),
                     static_cast<uint32_t>(mc_start.y),
                     static_cast<uint32_t>(mc_end.x),
                     static_cast<uint32_t>(mc_end.y),
                     num_dests,
                     static_cast<uint32_t>(v.x),
                     static_cast<uint32_t>(v.y),
                     run.dram_buffer ? run.dram_buffer->address() : 0u,
                     run.dram_staging,
                     core_idx++});
            }
        }
    }

    // NCRISC: NOC virus on cores with a route.
    if (cfg.noc.on) {
        std::vector<uint32_t> args = {run.ctrl, cfg.super_sync ? 1u : 0u, cfg.noc.xfer_bytes};
        const auto t = throttle_args(cfg.noc);
        args.insert(args.end(), t.begin(), t.end());
        for (uint32_t noc = 0; noc < 2; ++noc) {
            if (ncrisc_cores[noc].empty()) {
                continue;
            }
            auto kernel = CreateKernel(
                program,
                std::string(kKernelDir) + "pv_noc.cpp",
                CoreRangeSet(ncrisc_cores[noc]),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = noc == 0 ? NOC::NOC_0 : NOC::NOC_1,
                    .compile_args = args,
                });
            for (const auto& range : ncrisc_cores[noc]) {
                const CoreCoord dst = device->worker_core_from_logical_core(run.routes.at(range.start_coord).dst);
                SetRuntimeArgs(
                    program,
                    kernel,
                    range.start_coord,
                    {run.noc_buffer, static_cast<uint32_t>(dst.x), static_cast<uint32_t>(dst.y)});
            }
        }
    }

    // ERISC0: link traffic.
    if (!run.links.empty()) {
        std::set<CoreRange> link_cores;
        for (const auto& link : run.links) {
            link_cores.insert(CoreRange(link.core, link.core));
        }
        EthernetConfig eth_cfg{
            .eth_mode = Eth::SENDER,
            .noc = NOC::NOC_0,
            .processor = DataMovementProcessor::RISCV_0,
            .compile_args =
                {run.eth_num_slots,
                 cfg.eth.xfer_bytes,
                 cfg.eth.burst,
                 cfg.eth.idle_units,
                 cfg.eth_startup_delay_ticks,
                 cfg.eth.period_units,
                 cfg.eth.active_units},
        };
        eth_test_common::set_arch_specific_eth_config(eth_cfg);
        auto kernel =
            CreateKernel(program, std::string(kKernelDir) + "pv_eth_link.cpp", CoreRangeSet(link_cores), eth_cfg);
        const uint32_t num_banks = device->allocator()->get_num_banks(BufferType::DRAM);
        for (uint32_t i = 0; i < run.links.size(); ++i) {
            // Timing record i is page i of the interleaved timing buffer.
            SetRuntimeArgs(
                program,
                kernel,
                run.links[i].core,
                {run.eth_l1_base,
                 run.links[i].role,
                 i % num_banks,
                 run.eth_timing_buffer->address() + (i / num_banks) * kEthTimingPageBytes});
        }
    }
}

// ---------------------------------------------------------------------------
// Run and report
// ---------------------------------------------------------------------------

std::atomic<bool> g_interrupted{false};

extern "C" void on_sigint(int) { g_interrupted.store(true); }

struct Record {
    uint64_t elapsed = 0;
    uint64_t count = 0;
};

Record read_record(IDevice* device, const CoreCoord& core, uint32_t addr) {
    std::vector<uint32_t> rec;
    detail::ReadFromDeviceL1(device, core, addr, 16, rec, CoreType::WORKER);
    if (rec.size() != 4) {
        return {};
    }
    return {(static_cast<uint64_t>(rec[1]) << 32) | rec[0], (static_cast<uint64_t>(rec[3]) << 32) | rec[2]};
}

/// Logs one generator's aggregate and returns false if any core reported nothing.
bool report_movers(
    const DeviceRun& run,
    const char* name,
    uint32_t record_offset,
    uint32_t xfer_bytes,
    const std::vector<CoreCoord>& cores,
    double seconds) {
    uint64_t bytes = 0;
    uint64_t max_elapsed = 0;
    uint32_t idle_cores = 0;
    for (const auto& core : cores) {
        const Record r = read_record(run.device, core, run.ctrl + record_offset);
        if (r.count == 0) {
            ++idle_cores;
        }
        bytes += r.count * xfer_bytes;
        max_elapsed = std::max(max_elapsed, r.elapsed);
    }
    log_info(
        LogTest,
        "Device {}: {} – {:.1f} GB/s over {} cores ({:.2f} GB/s per core), {:.2f} GB in {:.1f} s",
        run.device->id(),
        name,
        bytes / seconds / 1e9,
        cores.size(),
        cores.empty() ? 0.0 : bytes / seconds / 1e9 / cores.size(),
        bytes / 1e9,
        seconds);
    if (idle_cores != 0) {
        log_warning(LogTest, "Device {}: {} – {} core(s) reported no transfers", run.device->id(), name, idle_cores);
    }
    return idle_cores == 0;
}

bool report(const DeviceRun& run, const PvConfig& cfg, double seconds) {
    bool ok = true;
    std::vector<CoreCoord> all_cores;
    for (const auto& core : run.grid) {
        all_cores.push_back(core);
    }

    if (cfg.trisc.on) {
        double min_pct = 1e9, max_pct = 0, sum_pct = 0;
        uint64_t max_elapsed = 0;
        uint32_t idle_cores = 0;
        for (const auto& core : all_cores) {
            const Record r = read_record(run.device, core, run.ctrl + pv::kFpuRecord);
            if (r.count == 0 || r.elapsed == 0) {
                ++idle_cores;
                continue;
            }
            const double pct = 100.0 * r.count * kNumTiles * kUsefulFpuCyclesPerTile / r.elapsed;
            min_pct = std::min(min_pct, pct);
            max_pct = std::max(max_pct, pct);
            sum_pct += pct;
            max_elapsed = std::max(max_elapsed, r.elapsed);
        }
        const uint32_t n = all_cores.size() - idle_cores;
        log_info(
            LogTest,
            "Device {}: TRISC – FPU utilization avg {:.1f}% (min {:.1f}%, max {:.1f}%) over {} cores; "
            "average AICLK ≈ {:.0f} MHz",
            run.device->id(),
            n ? sum_pct / n : 0.0,
            n ? min_pct : 0.0,
            max_pct,
            n,
            max_elapsed / seconds / 1e6);
        if (idle_cores != 0) {
            log_warning(LogTest, "Device {}: TRISC – {} core(s) reported no loops", run.device->id(), idle_cores);
            ok = false;
        }
    }
    if (cfg.dram.on) {
        ok &= report_movers(run, "DRAM reads", pv::kDramRecord, cfg.dram.xfer_bytes, all_cores, seconds);
    }
    if (cfg.noc.on) {
        std::vector<CoreCoord> sources;
        for (const auto& [core, route] : run.routes) {
            sources.push_back(core);
        }
        ok &= report_movers(run, "NOC writes", pv::kNocRecord, cfg.noc.xfer_bytes, sources, seconds);
    }
    if (!run.links.empty()) {
        std::vector<uint32_t> readback;
        detail::ReadFromBuffer(run.eth_timing_buffer, readback);
        constexpr size_t words_per_page = kEthTimingPageBytes / sizeof(uint32_t);
        double total = 0;
        for (size_t i = 0; i < run.links.size(); ++i) {
            const auto& link = run.links[i];
            const size_t off = i * words_per_page;
            const uint32_t sent = readback.size() >= off + 4 ? readback[off + 2] : 0;
            const uint32_t received = readback.size() >= off + 4 ? readback[off + 3] : 0;
            const bool complete = sent != 0 && received != 0;
            ok &= complete;
            const double gbps = static_cast<double>(sent) * cfg.eth.xfer_bytes / seconds / 1e9;
            total += gbps;
            log_info(
                LogTest,
                "Device {} ETH ({},{}) ↔ chip {} ({},{}): TX {:.1f} GB/s ({} sent, {} received){}",
                run.device->id(),
                link.core.x,
                link.core.y,
                link.peer_chip,
                link.peer_core.x,
                link.peer_core.y,
                gbps,
                sent,
                received,
                complete ? "" : "  INCOMPLETE");
        }
        log_info(
            LogTest,
            "Device {}: ETH – {:.1f} GB/s TX over {} links (RX is symmetric)",
            run.device->id(),
            total,
            run.links.size());
    }
    return ok;
}

bool run_power_virus(const std::vector<shared_ptr<distributed::MeshDevice>>& mesh_devices) {
    const PvConfig cfg = read_config();
    log_info(
        LogTest,
        "Power virus: duration={}, super_sync={}\n  TRISC: {}\n  DRAM:  {}\n  NOC:   {}\n  ETH:   {}",
        cfg.duration_s == 0 ? std::string("until Ctrl-C") : fmt::format("{} s", cfg.duration_s),
        cfg.super_sync,
        cfg.trisc.describe(),
        cfg.dram.describe(),
        cfg.noc.describe(),
        cfg.eth.describe());

    std::set<ChipId> dispatched_chips;
    for (const auto& mesh_device : mesh_devices) {
        dispatched_chips.insert(mesh_device->impl().get_devices().at(0)->id());
    }

    std::vector<DeviceRun> runs(mesh_devices.size());
    for (size_t i = 0; i < mesh_devices.size(); ++i) {
        DeviceRun& run = runs[i];
        run.mesh_device = mesh_devices[i];
        run.device = mesh_devices[i]->impl().get_devices().at(0);
        const auto grid = run.device->compute_with_storage_grid_size();
        run.grid = CoreRange({0, 0}, {grid.x - 1, grid.y - 1});
        layout_l1(run, cfg);
        init_tensix_l1(run, cfg);
        setup_dram(run, cfg);
        setup_noc_routes(run, cfg);
        setup_eth_links(run, cfg, dispatched_chips);
        build_program(run, cfg);
        log_info(LogTest, "Device {}: ready, {} Tensix cores", run.device->id(), run.grid.size());
    }

    // Launch everything without waiting; links handshake once both ends run.
    for (auto& run : runs) {
        slow_dispatch::LaunchProgramAsync(*run.device, run.program, /*force_slow_dispatch=*/false);
    }
    const auto t_start = std::chrono::steady_clock::now();
    log_info(
        LogTest, "Power virus running on {} device(s){}", runs.size(), cfg.duration_s == 0 ? " – Ctrl-C to stop" : "");

    g_interrupted.store(false);
    auto previous_handler = std::signal(SIGINT, on_sigint);
    auto last_report = t_start;
    while (!g_interrupted.load()) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t_start).count();
        if (cfg.duration_s != 0 && elapsed >= cfg.duration_s) {
            break;
        }
        if (now - last_report >= std::chrono::seconds(10)) {
            last_report = now;
            log_info(LogTest, "Power virus: {:.0f} s", elapsed);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // A second Ctrl-C while stopping falls back to the previous handler.
    std::signal(SIGINT, previous_handler);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    log_info(LogTest, "Stopping after {:.1f} s{}", seconds, g_interrupted.load() ? " (interrupted)" : "");

    std::vector<uint32_t> stop{1};
    for (auto& run : runs) {
        for (const auto& core : run.grid) {
            detail::WriteToDeviceL1(run.device, core, run.ctrl + pv::kStop, stop);
        }
        for (const auto& link : run.links) {
            detail::WriteToDeviceL1(run.device, link.core, run.eth_l1_base + pv::kEthStop, stop, CoreType::ETH);
        }
    }
    for (auto& run : runs) {
        slow_dispatch::WaitProgramDone(*run.device, run.program);
    }
    log_info(LogTest, "All generators stopped");

    bool ok = true;
    for (const auto& run : runs) {
        ok &= report(run, cfg, seconds);
    }
    return ok;
}

}  // namespace unit_tests::didt::power_virus

// ---------------------------------------------------------------------------
// GTest entry points
// ---------------------------------------------------------------------------

namespace {
bool skip_reason(const MeshDispatchFixture& fixture, std::string& reason) {
    if (MetalContext::instance().hal().get_arch() != ARCH::BLACKHOLE) {
        reason = "The power virus targets Blackhole";
        return true;
    }
    if (!fixture.IsSlowDispatch()) {
        reason = "The power virus needs slow dispatch: set TT_METAL_SLOW_DISPATCH_MODE=1";
        return true;
    }
    return false;
}
}  // namespace

TEST_F(MeshDispatchFixture, PowerVirus_SingleDevice) {
    std::string reason;
    if (skip_reason(*this, reason)) {
        GTEST_SKIP() << reason;
    }
    EXPECT_TRUE(unit_tests::didt::power_virus::run_power_virus({devices_.at(0)}));
}

TEST_F(MeshDispatchFixture, PowerVirus_AllDevices) {
    std::string reason;
    if (skip_reason(*this, reason)) {
        GTEST_SKIP() << reason;
    }
    EXPECT_TRUE(unit_tests::didt::power_virus::run_power_virus(devices_));
}

}  // namespace tt::tt_metal
