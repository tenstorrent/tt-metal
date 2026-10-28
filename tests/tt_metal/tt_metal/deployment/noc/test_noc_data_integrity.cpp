// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/eth/common.hpp"
#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "tt_metal/tt_metal/deployment/kernels/noc_integrity_types.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <thread>

#include "command_queue_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/allocator/allocator.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"

namespace tt::tt_metal {
namespace deployment_noc {

using namespace std;

constexpr uint32_t MAX_DESTS = 64;
constexpr const char* KERNEL_PATH = "tests/tt_metal/tt_metal/deployment/kernels/noc_integrity_kernel.cpp";

struct params {
    uint32_t iterations;
    uint32_t slice_size;
    uint32_t timeout_ms;
    uint32_t dram_all_nius;
    uint32_t packet_size;
    uint32_t burst;
    uint32_t vc;
    uint32_t tensix_rows;
};

static params get_params() {
    TEST_PARAM(uint32_t, iterations, 1, "NOC_TEST_ITERATIONS");
    TEST_PARAM(uint32_t, slice_size, 4096, "NOC_TEST_SLICE_SIZE");
    TEST_PARAM(uint32_t, timeout_ms, 2000, "NOC_TEST_TIMEOUT_MS");
    TEST_PARAM(uint32_t, dram_all_nius, 0, "NOC_TEST_DRAM_ALL_NIUS");
    // Defaults mirror fabric EDM local writes: 2 KB packets on VC 2, many in flight.
    TEST_PARAM(uint32_t, packet_size, 2048, "NOC_TEST_PACKET_SIZE");
    TEST_PARAM(uint32_t, burst, 8, "NOC_TEST_BURST");
    TEST_PARAM(uint32_t, vc, 2, "NOC_TEST_VC");
    TEST_PARAM(uint32_t, tensix_rows, 0, "NOC_TEST_TENSIX_ROWS");
    TT_FATAL(slice_size >= 64 && (slice_size % 64) == 0, "NOC_TEST_SLICE_SIZE must be a multiple of 64");
    TT_FATAL(packet_size >= 16 && (packet_size % 16) == 0, "NOC_TEST_PACKET_SIZE must be a multiple of 16");
    TT_FATAL(burst >= 1, "NOC_TEST_BURST must be at least 1");
    TT_FATAL(vc < 4, "NOC_TEST_VC must be a unicast VC (0-3)");
    return {iterations, slice_size, timeout_ms, dram_all_nius, packet_size, burst, vc, tensix_rows};
}

struct core_layout {
    uint32_t progress;
    uint32_t results;
    uint32_t src;
    uint32_t readback;
};

struct dest_layout {
    core_layout core;
    uint32_t dest_region;
};

// Unreserved L1 that is safe on both active (base firmware running) and idle Ethernet cores.
static l1_allocator new_common_eth_allocator() {
    const auto& hal = MetalContext::instance().hal();
    uint32_t active_start = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    uint32_t active_end =
        active_start + hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    uint32_t idle_start = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    uint32_t idle_end = idle_start + hal.get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    return {.start = max(active_start, idle_start), .end = min(active_end, idle_end)};
}

static core_layout alloc_core_layout(l1_allocator* alloc, uint32_t slice_size) {
    core_layout l{};
    l.progress = l1_alloc(alloc, NOC_PROGRESS_WORDS * sizeof(uint32_t));
    l.results = l1_alloc(alloc, MAX_DESTS * NOC_RESULT_WORDS * sizeof(uint32_t));
    l.src = l1_alloc(alloc, slice_size);
    l.readback = l1_alloc(alloc, slice_size);
    return l;
}

static dest_layout make_layout(l1_allocator alloc, uint32_t slice_size, uint32_t num_slots) {
    dest_layout l{};
    l.core = alloc_core_layout(&alloc, slice_size);
    l.dest_region = l1_alloc(&alloc, num_slots * slice_size);
    return l;
}

struct source {
    shared_ptr<distributed::MeshDevice> mesh;
    IDevice* device;
    CoreCoord logical;
    CoreCoord virt;
    CoreCoord noc0;
    bool is_eth;
    bool active;
    int chan;
    uint32_t slot;
};

struct dest {
    CoreCoord noc0;
    tt::umd::CoreCoord umd_noc0;
    CoreCoord logical;
    bool is_tensix;
    bool is_dram;
    bool active;
    int chan;
    int subchan;
    uint32_t base_addr;  // start of the slot region; each source adds slot * slice_size
    uint8_t dram_noc_mask;
};

struct failure {
    uint32_t noc;
    source src;
    dest dst;
    uint32_t addr;
    vector<uint32_t> record;
};

static string chip_str(IDevice* device) {
    return fmt::format("chip {} ({}, {})", device->id(), pci_bdf_for_device_id(device->id()), get_ubb(device));
}

static string src_str(const source& s) {
    if (s.is_eth) {
        return fmt::format(
            "src eth ch {} [noc0 {}-{}, logical {}, {}]",
            s.chan,
            s.noc0.x,
            s.noc0.y,
            s.logical,
            s.active ? "active" : "idle");
    }
    return fmt::format("src tensix [noc0 {}-{}, logical {}]", s.noc0.x, s.noc0.y, s.logical);
}

static string dst_str(const dest& d) {
    if (d.is_dram) {
        return fmt::format("dst dram ch {} sub {} [noc0 {}-{}]", d.chan, d.subchan, d.noc0.x, d.noc0.y);
    }
    if (d.is_tensix) {
        return fmt::format("dst tensix [noc0 {}-{}, logical {}]", d.noc0.x, d.noc0.y, d.logical);
    }
    return fmt::format("dst eth ch {} [noc0 {}-{}, {}]", d.chan, d.noc0.x, d.noc0.y, d.active ? "active" : "idle");
}

static vector<CoreCoord> all_eth_cores(IDevice* device) {
    vector<CoreCoord> cores;
    for (const auto& c : device->get_active_ethernet_cores(true)) {
        cores.push_back(c);
    }
    for (const auto& c : device->get_inactive_ethernet_cores()) {
        cores.push_back(c);
    }
    const auto& chan_map =
        MetalContext::instance().get_cluster().get_soc_desc(device->id()).logical_eth_core_to_chan_map;
    sort(cores.begin(), cores.end(), [&](const CoreCoord& a, const CoreCoord& b) {
        return chan_map.at(a) < chan_map.at(b);
    });
    return cores;
}

static CoreCoord to_noc0(IDevice* device, const CoreCoord& virt, CoreType type) {
    const auto& soc_d = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    auto c =
        soc_d.translate_coord_to(tt::umd::CoreCoord(virt.x, virt.y, type, CoordSystem::TRANSLATED), CoordSystem::NOC0);
    return {c.x, c.y};
}

static vector<source> get_eth_sources(const shared_ptr<distributed::MeshDevice>& mesh) {
    auto* const device = mesh->impl().get_devices()[0];
    const auto& soc_d = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto active = device->get_active_ethernet_cores(true);

    vector<source> sources;
    uint32_t slot = 0;
    for (const auto& logical : all_eth_cores(device)) {
        CoreCoord virt = device->ethernet_core_from_logical_core(logical);
        sources.push_back({
            .mesh = mesh,
            .device = device,
            .logical = logical,
            .virt = virt,
            .noc0 = to_noc0(device, virt, CoreType::ETH),
            .is_eth = true,
            .active = active.contains(logical),
            .chan = soc_d.logical_eth_core_to_chan_map.at(logical),
            .slot = slot++,
        });
    }
    return sources;
}

static vector<source> get_tensix_sources(const shared_ptr<distributed::MeshDevice>& mesh) {
    auto* const device = mesh->impl().get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();
    set<CoreCoord> corners = {
        {0, 0},
        {grid.x - 1, 0},
        {0, grid.y - 1},
        {grid.x - 1, grid.y - 1},
    };

    vector<source> sources;
    uint32_t slot = 0;
    for (const auto& logical : corners) {
        CoreCoord virt = device->worker_core_from_logical_core(logical);
        sources.push_back({
            .mesh = mesh,
            .device = device,
            .logical = logical,
            .virt = virt,
            .noc0 = to_noc0(device, virt, CoreType::TENSIX),
            .is_eth = false,
            .active = false,
            .chan = -1,
            .slot = slot++,
        });
    }
    return sources;
}

// Tensix cores (all worker rows unless NOC_TEST_TENSIX_ROWS is set), plus Ethernet L1 and
// DRAM endpoints in NOC rows 0/1. PCIe, ARC and router-only tiles are never targeted.
static vector<dest> get_dests(
    IDevice* device,
    const dest_layout& eth_layout,
    const dest_layout& tensix_layout,
    const shared_ptr<distributed::MeshBuffer>& dram_buffer,
    const params& p) {
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& soc_d = cluster.get_soc_desc(device->id());
    const auto active = device->get_active_ethernet_cores(true);

    vector<dest> dests;
    CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t rows = p.tensix_rows ? min<uint32_t>(p.tensix_rows, grid.y) : grid.y;
    for (uint32_t y = 0; y < rows; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            CoreCoord logical(x, y);
            CoreCoord noc0 = to_noc0(device, device->worker_core_from_logical_core(logical), CoreType::TENSIX);
            dests.push_back({
                .noc0 = noc0,
                .umd_noc0 = tt::umd::CoreCoord(noc0.x, noc0.y, CoreType::TENSIX, CoordSystem::NOC0),
                .logical = logical,
                .is_tensix = true,
                .is_dram = false,
                .active = false,
                .chan = -1,
                .subchan = -1,
                .base_addr = tensix_layout.dest_region,
                .dram_noc_mask = 0,
            });
        }
    }

    for (const auto& logical : all_eth_cores(device)) {
        CoreCoord noc0 = to_noc0(device, device->ethernet_core_from_logical_core(logical), CoreType::ETH);
        if (noc0.y > 1) {
            continue;
        }
        dests.push_back({
            .noc0 = noc0,
            .umd_noc0 = tt::umd::CoreCoord(noc0.x, noc0.y, CoreType::ETH, CoordSystem::NOC0),
            .logical = logical,
            .is_tensix = false,
            .is_dram = false,
            .active = active.contains(logical),
            .chan = soc_d.logical_eth_core_to_chan_map.at(logical),
            .subchan = -1,
            .base_addr = eth_layout.dest_region,
            .dram_noc_mask = 0,
        });
    }

    const auto& allocator = dram_buffer->device()->allocator();
    map<uint32_t, uint32_t> view_to_bank;
    for (uint32_t bank = 0; bank < allocator->get_num_banks(BufferType::DRAM); bank++) {
        view_to_bank[allocator->impl().get_dram_channel_from_bank_id(bank)] = bank;
    }

    for (const auto& core : soc_d.get_cores(CoreType::DRAM, CoordSystem::NOC0)) {
        if (core.y > 1) {
            continue;
        }
        auto [chan, subchan] = soc_d.get_dram_channel_for_core(core, CoordSystem::NOC0);
        optional<uint32_t> bank;
        for (size_t view = 0; view < soc_d.get_num_dram_views(); view++) {
            if (soc_d.get_channel_for_dram_view(view) == static_cast<size_t>(chan) && view_to_bank.contains(view)) {
                bank = view_to_bank.at(view);
                break;
            }
        }
        if (!bank) {
            continue;
        }
        auto translated = soc_d.translate_coord_to(core, CoordSystem::TRANSLATED);
        dests.push_back({
            .noc0 = {core.x, core.y},
            .umd_noc0 = core,
            .logical = {},
            .is_tensix = false,
            .is_dram = true,
            .active = false,
            .chan = chan,
            .subchan = subchan,
            .base_addr =
                static_cast<uint32_t>(dram_buffer->address() + allocator->get_bank_offset(BufferType::DRAM, *bank)),
            .dram_noc_mask = soc_d.get_dram_endpoint_noc_mask({translated.x, translated.y}),
        });
    }
    return dests;
}

// With DRISC firmware, NIUs that are not a DRAM view's preferred endpoint are in stream mode and don't reach GDDR.
static bool dram_niu_usable(const dest& d, uint32_t noc, const params& p) {
    if (d.dram_noc_mask & (1u << noc)) {
        return true;
    }
    const bool drisc = MetalContext::instance().hal().has_programmable_core_type(HalProgrammableCoreType::DRAM);
    return p.dram_all_nius && !drisc;
}

struct launched {
    const source* src;
    core_layout layout;
    vector<const dest*> dests;
    vector<uint32_t> addrs;
};

static vector<uint32_t> read_words(const source& s, uint32_t addr, uint32_t words) {
    return MetalContext::instance().get_cluster().read_core<uint32_t>(
        s.device->id(), s.virt, addr, words * sizeof(uint32_t));
}

struct wedged_chip {
    IDevice* device;
    uint32_t noc;
    size_t batch;
    string reason;
};

struct poll_result {
    vector<vector<uint32_t>> last_prog;  // per run; empty if the chip could not be read
    vector<bool> stalled;                // per run: kernel stopped making progress
    map<ChipId, string> wedged;          // chips to stop using, with the reason
    set<ChipId> unreadable;              // subset of wedged whose host reads hang or fail
};

// One polling thread per chip: a host read that never returns only blocks its own chip.
static poll_result wait_for_sources(const vector<launched>& runs, const params& p) {
    using clock = chrono::steady_clock;
    const int64_t stall_limit_ms = 2 * p.timeout_ms + 5000;
    auto now_ms = [] { return chrono::duration_cast<chrono::milliseconds>(clock::now().time_since_epoch()).count(); };

    struct chip_poll {
        ChipId chip;
        vector<size_t> idx;
        vector<pair<CoreCoord, uint32_t>> targets;  // copied so a detached thread never touches `runs`
        atomic<int64_t> read_start_ms{-1};
        atomic<size_t> reading{0};
        atomic<bool> finished{false};
        vector<vector<uint32_t>> prog;
        vector<bool> stalled;
        string error;
    };

    map<ChipId, shared_ptr<chip_poll>> polls;
    for (size_t i = 0; i < runs.size(); i++) {
        ChipId chip = runs[i].src->device->id();
        auto& cp = polls[chip];
        if (!cp) {
            cp = make_shared<chip_poll>();
            cp->chip = chip;
        }
        cp->idx.push_back(i);
        cp->targets.emplace_back(runs[i].src->virt, runs[i].layout.progress);
    }

    map<ChipId, thread> threads;
    for (auto& [chip, cp] : polls) {
        cp->prog.resize(cp->targets.size());
        cp->stalled.resize(cp->targets.size(), false);
        threads[chip] = thread([cp, stall_limit_ms, now_ms] {
            const size_t n = cp->targets.size();
            vector<uint32_t> last_hb(n, ~0u);
            vector<int64_t> last_change(n, now_ms());
            for (;;) {
                bool running = false;
                for (size_t j = 0; j < n; j++) {
                    if (cp->stalled[j]) {
                        continue;
                    }
                    cp->reading = j;
                    cp->read_start_ms = now_ms();
                    try {
                        cp->prog[j] = MetalContext::instance().get_cluster().read_core<uint32_t>(
                            cp->chip,
                            cp->targets[j].first,
                            cp->targets[j].second,
                            NOC_PROGRESS_WORDS * sizeof(uint32_t));
                    } catch (const std::exception& e) {
                        cp->error = fmt::format("host read failed: {}", e.what());
                        cp->finished = true;
                        return;
                    }
                    cp->read_start_ms = -1;
                    const auto& prog = cp->prog[j];
                    if (prog[NOC_PROGRESS_STATE] == NOC_STATE_DONE || prog[NOC_PROGRESS_STATE] == NOC_STATE_ABORTED) {
                        continue;
                    }
                    if (prog[NOC_PROGRESS_HEARTBEAT] != last_hb[j]) {
                        last_hb[j] = prog[NOC_PROGRESS_HEARTBEAT];
                        last_change[j] = now_ms();
                        running = true;
                    } else if (now_ms() - last_change[j] > stall_limit_ms) {
                        cp->stalled[j] = true;
                    } else {
                        running = true;
                    }
                }
                if (!running) {
                    cp->finished = true;
                    return;
                }
                this_thread::sleep_for(chrono::milliseconds(10));
            }
        });
    }

    poll_result res;
    res.last_prog.resize(runs.size());
    res.stalled.resize(runs.size(), false);

    set<ChipId> pending;
    for (const auto& [chip, _] : polls) {
        pending.insert(chip);
    }
    while (!pending.empty()) {
        this_thread::sleep_for(chrono::milliseconds(50));
        for (auto it = pending.begin(); it != pending.end();) {
            const ChipId chip = *it;
            auto& cp = polls[chip];
            if (cp->finished) {
                threads[chip].join();
                if (!cp->error.empty()) {
                    res.wedged[chip] = cp->error;
                    res.unreadable.insert(chip);
                } else {
                    size_t n_stalled = 0;
                    for (size_t j = 0; j < cp->idx.size(); j++) {
                        res.last_prog[cp->idx[j]] = cp->prog[j];
                        res.stalled[cp->idx[j]] = cp->stalled[j];
                        n_stalled += cp->stalled[j];
                    }
                    if (n_stalled) {
                        res.wedged[chip] = fmt::format("{} source(s) stopped making progress", n_stalled);
                    }
                }
                it = pending.erase(it);
                continue;
            }
            const int64_t t = cp->read_start_ms;
            if (t >= 0 && now_ms() - t > stall_limit_ms) {
                const auto& r = runs[cp->idx[cp->reading]];
                res.wedged[chip] =
                    fmt::format("host read of {} progress did not return after {} ms", src_str(*r.src), now_ms() - t);
                res.unreadable.insert(chip);
                threads[chip].detach();
                it = pending.erase(it);
                continue;
            }
            ++it;
        }
    }
    return res;
}

template <typename FIXTURE>
static vector<failure> run_phase(
    FIXTURE* fixture,
    uint32_t noc,
    const vector<vector<source>>& sources_per_chip,
    const map<ChipId, vector<dest>>& dests_per_chip,
    const dest_layout& eth_l,
    const dest_layout& tensix_l,
    const params& p,
    map<ChipId, wedged_chip>& wedged) {
    /* ==================== */
    const auto& hal = MetalContext::instance().hal();
    const uint32_t idle_eriscs = hal.get_num_risc_processors(HalProgrammableCoreType::IDLE_ETH);

    struct plan {
        const source* src;
        core_layout layout;
        vector<uint32_t> compile_args;
        uint32_t seed;
        vector<const dest*> dests;
        vector<uint32_t> addrs;
        vector<CoreCoord> xys;
    };

    vector<plan> plans;
    size_t max_dests = 0;
    for (const auto& sources : sources_per_chip) {
        for (const auto& s : sources) {
            if (s.is_eth && s.active) {
                EthernetConfig probe{
                    .noc = static_cast<NOC>(noc), .processor = static_cast<DataMovementProcessor>(noc)};
                eth_test_common::set_arch_specific_eth_config(probe);
                if (static_cast<uint32_t>(probe.noc) != noc) {
                    log_warning(
                        tt::LogTest, "{} {}: cannot run on noc {}, skipping", chip_str(s.device), src_str(s), noc);
                    continue;
                }
            }

            const core_layout& layout = s.is_eth ? eth_l.core : tensix_l.core;
            plan pl{
                .src = &s,
                .layout = layout,
                .compile_args =
                    {
                        layout.progress,
                        layout.results,
                        layout.src,
                        layout.readback,
                        p.slice_size,
                        p.iterations,
                        p.timeout_ms,
                        p.packet_size,
                        p.burst,
                        p.vc,
                    },
                .seed = 0x9E3779B9u ^ (s.device->id() << 24) ^ (s.slot << 8) ^ noc,
            };
            const auto& soc_d = MetalContext::instance().get_cluster().get_soc_desc(s.device->id());
            for (const auto& d : dests_per_chip.at(s.device->id())) {
                if (!d.is_dram && d.noc0 == s.noc0) {
                    continue;
                }
                if (d.is_dram && !dram_niu_usable(d, noc, p)) {
                    continue;
                }
                auto xy = soc_d.translate_chip_coord_to_translated_coord(
                    d.umd_noc0, noc ? tt::umd::NocId::NOC1 : tt::umd::NocId::NOC0);
                pl.dests.push_back(&d);
                pl.addrs.push_back(d.base_addr + s.slot * p.slice_size);
                pl.xys.emplace_back(xy.x, xy.y);
            }
            max_dests = max(max_dests, pl.dests.size());
            plans.push_back(std::move(pl));
        }
    }

    // Ethernet kernels are limited in runtime args, so destinations run in batches of MAX_DESTS.
    const size_t num_batches = (max_dests + MAX_DESTS - 1) / MAX_DESTS;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    vector<failure> failures;
    for (size_t batch = 0; batch < num_batches; batch++) {
        const size_t first = batch * MAX_DESTS;
        vector<launched> runs;
        map<shared_ptr<distributed::MeshDevice>, shared_ptr<Program>> programs;

        for (const auto& pl : plans) {
            if (first >= pl.dests.size()) {
                continue;
            }
            const source& s = *pl.src;
            if (wedged.contains(s.device->id())) {
                continue;
            }
            const size_t last = min(first + MAX_DESTS, pl.dests.size());

            launched run{.src = &s, .layout = pl.layout};
            vector<uint32_t> rt_args = {pl.seed, static_cast<uint32_t>(last - first)};
            for (size_t i = first; i < last; i++) {
                rt_args.insert(
                    rt_args.end(),
                    {static_cast<uint32_t>(pl.xys[i].x), static_cast<uint32_t>(pl.xys[i].y), pl.addrs[i]});
                run.dests.push_back(pl.dests[i]);
                run.addrs.push_back(pl.addrs[i]);
            }

            if (!programs.contains(s.mesh)) {
                programs[s.mesh] = make_shared<Program>();
            }
            Program& program = *programs[s.mesh];

            KernelHandle kernel;
            if (s.is_eth) {
                EthernetConfig config{
                    .eth_mode = s.active ? Eth::SENDER : Eth::IDLE,
                    .noc = static_cast<NOC>(noc),
                    .processor = static_cast<DataMovementProcessor>(s.active || idle_eriscs > 1 ? noc : 0),
                    .compile_args = pl.compile_args,
                };
                if (s.active) {
                    eth_test_common::set_arch_specific_eth_config(config);
                }
                kernel = CreateKernel(program, KERNEL_PATH, s.logical, config);
            } else {
                DataMovementConfig config{
                    .processor = noc ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0,
                    .noc = static_cast<NOC>(noc),
                    .compile_args = pl.compile_args,
                };
                kernel = CreateKernel(program, KERNEL_PATH, s.logical, config);
            }
            SetRuntimeArgs(program, kernel, s.logical, rt_args);

            // Stale progress from a previous batch must not look like completion.
            MetalContext::instance().get_cluster().write_core(
                s.device->id(), s.virt, vector<uint32_t>(NOC_PROGRESS_WORDS, 0), pl.layout.progress);
            runs.push_back(std::move(run));
        }
        if (runs.empty()) {
            continue;
        }

        log_info(
            tt::LogTest,
            "  noc {} batch {}/{}: compiling and launching {} kernels on {} chips",
            noc,
            batch + 1,
            num_batches,
            runs.size(),
            programs.size());
        map<shared_ptr<distributed::MeshDevice>, shared_ptr<distributed::MeshWorkload>> workloads;
        for (auto& [mesh, program] : programs) {
            program->impl().compile(mesh.get());
            workloads[mesh] = make_shared<distributed::MeshWorkload>();
            workloads[mesh]->add_program(device_range, std::move(*program));
        }
        for (auto& [mesh, workload] : workloads) {
            fixture->RunProgram(mesh, *workload, true);
        }
        log_info(tt::LogTest, "  noc {} batch {}/{}: running", noc, batch + 1, num_batches);

        poll_result poll = wait_for_sources(runs, p);

        for (auto& [mesh, _] : workloads) {
            if (!poll.wedged.contains(mesh->get_device_ids()[0])) {
                fixture->FinishCommands(mesh);
            }
        }

        size_t batch_failures = 0;
        for (size_t i = 0; i < runs.size(); i++) {
            const auto& r = runs[i];
            if (poll.unreadable.contains(r.src->device->id())) {
                continue;
            }
            auto results = read_words(*r.src, r.layout.results, r.dests.size() * NOC_RESULT_WORDS);
            const auto& prog = poll.last_prog[i];
            for (size_t d = 0; d < r.dests.size(); d++) {
                vector<uint32_t> rec(
                    results.begin() + d * NOC_RESULT_WORDS, results.begin() + (d + 1) * NOC_RESULT_WORDS);
                if (poll.stalled[i] && d == prog[NOC_PROGRESS_DEST]) {
                    rec[NOC_RESULT_STATUS] = NOC_STATUS_HANG;
                    rec[NOC_RESULT_PATTERN] = prog[NOC_PROGRESS_PATTERN];
                    rec[NOC_RESULT_ITERATION] = prog[NOC_PROGRESS_ITERATION];
                }
                if (rec[NOC_RESULT_STATUS] == NOC_STATUS_DONE && rec[NOC_RESULT_MISMATCHES] == 0) {
                    continue;
                }
                failures.push_back({noc, *r.src, *r.dests[d], r.addrs[d], std::move(rec)});
                batch_failures++;
            }
        }

        for (const auto& [chip, reason] : poll.wedged) {
            IDevice* device = nullptr;
            for (const auto& r : runs) {
                if (r.src->device->id() == chip) {
                    device = r.src->device;
                    break;
                }
            }
            log_critical(
                tt::LogTest,
                "  {} wedged in noc {} batch {}/{}: {}; skipping it for the rest of the test",
                chip_str(device),
                noc,
                batch + 1,
                num_batches,
                reason);
            wedged[chip] = {device, noc, batch + 1, reason};
        }
        log_info(
            tt::LogTest,
            "  noc {} batch {}/{}: {} sources ran, {} failing source/destination pairs",
            noc,
            batch + 1,
            num_batches,
            runs.size(),
            batch_failures);
    }
    return failures;
}

static void print_failure_summary(const vector<failure>& failures, const map<ChipId, wedged_chip>& wedged) {
    using entry = pair<pair<size_t, size_t>, string>;  // (noc0 y, x) for ordering, label
    struct group {
        IDevice* device;
        size_t pairs = 0;
        map<string, vector<entry>> by_src;
        map<string, vector<entry>> by_dst;
    };
    map<pair<ChipId, uint32_t>, group> groups;
    for (const auto& f : failures) {
        auto& g = groups[{f.src.device->id(), f.noc}];
        g.device = f.src.device;
        g.pairs++;
        g.by_src[src_str(f.src)].push_back({{f.dst.noc0.y, f.dst.noc0.x}, dst_str(f.dst)});
        g.by_dst[dst_str(f.dst)].push_back({{f.src.noc0.y, f.src.noc0.x}, src_str(f.src)});
    }

    auto recurring = [](const map<string, vector<entry>>& m) {
        vector<pair<string, vector<entry>>> out;
        for (const auto& [k, v] : m) {
            if (v.size() > 1) {
                out.emplace_back(k, v);
                sort(out.back().second.begin(), out.back().second.end());
            }
        }
        stable_sort(
            out.begin(), out.end(), [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });
        return out;
    };

    log_critical(tt::LogTest, "==================== NOC FAILURE SUMMARY ====================");
    for (const auto& [key, g] : groups) {
        log_critical(
            tt::LogTest,
            "{} noc {}: {} failing pair(s), {} source(s), {} destination(s)",
            chip_str(g.device),
            key.second,
            g.pairs,
            g.by_src.size(),
            g.by_dst.size());

        auto srcs = recurring(g.by_src);
        if (!srcs.empty()) {
            log_critical(
                tt::LogTest, "  Recurring sources (one source -> many destinations: suspect the source's path):");
        }
        for (const auto& [src, dsts] : srcs) {
            log_critical(tt::LogTest, "    {} -> {} destinations", src, dsts.size());
            for (const auto& [_, d] : dsts) {
                log_critical(tt::LogTest, "        {}", d);
            }
        }

        auto dsts = recurring(g.by_dst);
        if (!dsts.empty()) {
            log_critical(
                tt::LogTest,
                "  Recurring destinations (many sources -> one destination: suspect the destination's path):");
        }
        for (const auto& [dst, srcs_of_dst] : dsts) {
            log_critical(tt::LogTest, "    {} <- {} sources", dst, srcs_of_dst.size());
            for (const auto& [_, s] : srcs_of_dst) {
                log_critical(tt::LogTest, "        {}", s);
            }
        }

        vector<string> isolated;
        for (const auto& [src, dsts_of_src] : g.by_src) {
            if (dsts_of_src.size() == 1 && g.by_dst.at(dsts_of_src[0].second).size() == 1) {
                isolated.push_back(fmt::format("{} -> {}", src, dsts_of_src[0].second));
            }
        }
        if (!isolated.empty()) {
            log_critical(tt::LogTest, "  Isolated pairs (no recurring source or destination):");
            for (const auto& s : isolated) {
                log_critical(tt::LogTest, "    {}", s);
            }
        }
    }
    if (!wedged.empty()) {
        log_critical(tt::LogTest, "Wedged chips (not tested after this point):");
        for (const auto& [_, w] : wedged) {
            log_critical(tt::LogTest, "  {}: noc {} batch {}: {}", chip_str(w.device), w.noc, w.batch, w.reason);
        }
    }
    log_critical(tt::LogTest, "=============================================================");
}

struct test_result {
    vector<failure> failures;
    map<ChipId, wedged_chip> wedged;
};

static void report(const test_result& result) {
    const auto& failures = result.failures;
    for (const auto& f : failures) {
        const auto& rec = f.record;
        string chip = chip_str(f.src.device);
        string prefix = fmt::format("{} {} -> {} @0x{:x} noc {}", chip, src_str(f.src), dst_str(f.dst), f.addr, f.noc);
        switch (rec[NOC_RESULT_STATUS]) {
            case NOC_STATUS_WRITE_TIMEOUT:
            case NOC_STATUS_READ_TIMEOUT:
                log_critical(
                    tt::LogTest,
                    "NOC timeout: {}: {} barrier timed out ({}, iteration {})",
                    prefix,
                    rec[NOC_RESULT_STATUS] == NOC_STATUS_WRITE_TIMEOUT ? "write" : "read",
                    noc_pattern_name(rec[NOC_RESULT_PATTERN]),
                    rec[NOC_RESULT_ITERATION]);
                break;
            case NOC_STATUS_NOT_RUN:
                log_critical(tt::LogTest, "NOC not run: {}: source stopped before this destination", prefix);
                break;
            case NOC_STATUS_HANG:
                log_critical(
                    tt::LogTest,
                    "NOC hang: {}: source stopped making progress ({}, iteration {})",
                    prefix,
                    noc_pattern_name(rec[NOC_RESULT_PATTERN]),
                    rec[NOC_RESULT_ITERATION]);
                break;
            default:
                log_critical(
                    tt::LogTest,
                    "NOC mismatch: {}: {} words in {} transfers, first at +0x{:x}, expected 0x{:08x}, got 0x{:08x} "
                    "({}, iteration {})",
                    prefix,
                    rec[NOC_RESULT_MISMATCHES],
                    rec[NOC_RESULT_TRANSFERS],
                    rec[NOC_RESULT_FIRST_OFFSET],
                    rec[NOC_RESULT_EXPECTED],
                    rec[NOC_RESULT_ACTUAL],
                    noc_pattern_name(rec[NOC_RESULT_PATTERN]),
                    rec[NOC_RESULT_ITERATION]);
                break;
        }
    }

    if (failures.empty() && result.wedged.empty()) {
        return;
    }
    log_critical(tt::LogTest, "{} failing source/destination pairs in total", failures.size());
    print_failure_summary(failures, result.wedged);
}

// Closing a wedged chip hangs, so skip normal teardown once the report is printed.
static void exit_if_wedged(const test_result& result) {
    if (result.wedged.empty()) {
        return;
    }
    log_critical(
        tt::LogTest,
        "{} chip(s) wedged; exiting without closing devices. Reset with tt-smi -glx_reset before the next run.",
        result.wedged.size());
    fflush(stdout);
    fflush(stderr);
    _Exit(1);
}

template <typename FIXTURE>
static test_result run_test(
    FIXTURE* fixture, const vector<shared_ptr<distributed::MeshDevice>>& devices, bool tensix_sources) {
    /* ==================== */
    const params p = get_params();

    vector<vector<source>> sources_per_chip;
    uint32_t num_slots = 0;
    for (const auto& mesh : devices) {
        sources_per_chip.push_back(tensix_sources ? get_tensix_sources(mesh) : get_eth_sources(mesh));
        num_slots = max<uint32_t>(num_slots, sources_per_chip.back().size());
    }

    const dest_layout eth_l = make_layout(new_common_eth_allocator(), p.slice_size, num_slots);
    const dest_layout tensix_l = make_layout(new_tensix_allocator(), p.slice_size, num_slots);

    log_info(
        tt::LogTest,
        "slice {} B, packet {} B, burst {}, vc {}, {} slots, iterations {}, tensix rows {}, "
        "eth L1 dest region 0x{:x}, tensix L1 dest region 0x{:x}, DRISC firmware {}",
        p.slice_size,
        p.packet_size,
        p.burst,
        p.vc,
        num_slots,
        p.iterations,
        p.tensix_rows,
        eth_l.dest_region,
        tensix_l.dest_region,
        MetalContext::instance().hal().has_programmable_core_type(HalProgrammableCoreType::DRAM) ? "on" : "off");

    vector<shared_ptr<distributed::MeshBuffer>> dram_buffers;
    map<ChipId, vector<dest>> dests_per_chip;
    for (const auto& mesh : devices) {
        auto* const device = mesh->impl().get_devices()[0];
        const uint32_t num_banks = mesh->allocator()->get_num_banks(BufferType::DRAM);
        const uint32_t page_size = num_slots * p.slice_size;
        distributed::DeviceLocalBufferConfig local{.page_size = page_size, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig global{.size = static_cast<DeviceAddr>(page_size) * num_banks};
        auto buffer = distributed::MeshBuffer::create(global, local, mesh.get());
        dram_buffers.push_back(buffer);

        auto dests = get_dests(device, eth_l, tensix_l, buffer, p);
        for (const auto& d : dests) {
            if (d.is_tensix) {
                continue;
            }
            log_info(
                tt::LogTest,
                "  {}: {}{}",
                chip_str(device),
                dst_str(d),
                d.is_dram ? fmt::format(" (noc mask 0x{:x})", d.dram_noc_mask) : "");
        }
        dests_per_chip[device->id()] = std::move(dests);
    }

    test_result result;
    for (uint32_t noc = 0; noc < 2; noc++) {
        auto f = run_phase(fixture, noc, sources_per_chip, dests_per_chip, eth_l, tensix_l, p, result.wedged);
        result.failures.insert(result.failures.end(), f.begin(), f.end());
    }
    return result;
}

// Parses `--ubb N` / `--ubb=N` (N in 1..4) from the command line; 0 means all trays.
static uint32_t get_ubb_arg() {
    const auto args = ::testing::internal::GetArgvs();
    for (size_t i = 1; i < args.size(); i++) {
        string val;
        if (args[i] == "--ubb") {
            TT_FATAL(i + 1 < args.size(), "--ubb requires a value (1-4)");
            val = args[i + 1];
        } else if (args[i].starts_with("--ubb=")) {
            val = args[i].substr(6);
        } else {
            continue;
        }
        uint32_t ubb = 0;
        try {
            ubb = stoul(val);
        } catch (const std::exception&) {
            TT_THROW("--ubb value '{}' is not a number", val);
        }
        TT_FATAL(ubb >= 1 && ubb <= 4, "--ubb must be between 1 and 4, got {}", ubb);
        return ubb;
    }
    return 0;
}

static vector<shared_ptr<distributed::MeshDevice>> filter_by_ubb(
    const vector<shared_ptr<distributed::MeshDevice>>& devices) {
    const uint32_t ubb = get_ubb_arg();
    if (!ubb) {
        return devices;
    }
    umd::ClusterDescriptor* cluster_desc = MetalContext::instance().get_cluster().get_cluster_desc();
    vector<shared_ptr<distributed::MeshDevice>> out;
    for (const auto& mesh : devices) {
        auto* const device = mesh->impl().get_devices()[0];
        if (tt::tt_fabric::get_ubb_id(*cluster_desc, device->id()).tray_id == ubb) {
            out.push_back(mesh);
        }
    }
    log_info(tt::LogTest, "--ubb {}: testing {} of {} chips", ubb, out.size(), devices.size());
    return out;
}

// PCIe bus-number high nibble per tray (index = tray - 1); must match tt_fabric::get_ubb_id.
static const map<string, vector<uint32_t>> ubb_tray_buses = {
    {"0xb140", {0x00, 0x40, 0xC0, 0x80}},  // Blackhole
    {"0x401e", {0xC0, 0x80, 0x00, 0x40}},  // Wormhole
};

// Limits UMD to the tray's PCIe devices so chips on other trays are never opened.
class ubb_visible_devices_env : public ::testing::Environment {
public:
    void SetUp() override {
        const uint32_t ubb = get_ubb_arg();
        if (!ubb) {
            return;
        }
        TT_FATAL(std::getenv("TT_VISIBLE_DEVICES") == nullptr, "--ubb cannot be combined with TT_VISIBLE_DEVICES");

        namespace fs = std::filesystem;
        vector<string> bdfs;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator("/sys/class/tenstorrent", ec)) {
            const fs::path pci = fs::canonical(entry.path() / "device", ec);
            if (ec) {
                continue;
            }
            const string bdf = pci.filename().string();
            const auto it = ubb_tray_buses.find(read_text_file_trimmed((pci / "device").string()));
            const auto colon = bdf.find(':');
            if (it == ubb_tray_buses.end() || colon == string::npos) {
                continue;
            }
            const uint32_t bus = stoul(bdf.substr(colon + 1, 2), nullptr, 16);
            if ((bus & 0xF0) == it->second[ubb - 1]) {
                bdfs.push_back(bdf);
            }
        }
        TT_FATAL(!bdfs.empty(), "--ubb {}: no Tenstorrent PCIe devices found on that tray", ubb);
        sort(bdfs.begin(), bdfs.end());

        string visible;
        for (const auto& bdf : bdfs) {
            visible += (visible.empty() ? "" : ",") + bdf;
        }
        log_info(tt::LogTest, "--ubb {}: TT_VISIBLE_DEVICES={}", ubb, visible);
        setenv("TT_VISIBLE_DEVICES", visible.c_str(), 1);
    }
};

[[maybe_unused]] static ::testing::Environment* const ubb_env =
    ::testing::AddGlobalTestEnvironment(new ubb_visible_devices_env);

}  // namespace deployment_noc

TEST_F(MeshDispatchFixture, TensixDeploymentNoc00EthSourceIntegrity) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    print_detected_devices();

    auto devices = deployment_noc::filter_by_ubb(devices_);
    if (devices.empty()) {
        GTEST_SKIP() << "no chips on the requested UBB tray";
    }
    auto result = deployment_noc::run_test(this, devices, false);
    deployment_noc::report(result);
    deployment_noc::exit_if_wedged(result);
    ASSERT_TRUE(result.failures.empty());
}

TEST_F(MeshDispatchFixture, TensixDeploymentNoc01TensixSourceIntegrity) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "Ethernet destinations require slow dispatch";
    }
    print_detected_devices();

    auto devices = deployment_noc::filter_by_ubb(devices_);
    if (devices.empty()) {
        GTEST_SKIP() << "no chips on the requested UBB tray";
    }
    auto result = deployment_noc::run_test(this, devices, true);
    deployment_noc::report(result);
    deployment_noc::exit_if_wedged(result);
    ASSERT_TRUE(result.failures.empty());
}

}  // namespace tt::tt_metal
