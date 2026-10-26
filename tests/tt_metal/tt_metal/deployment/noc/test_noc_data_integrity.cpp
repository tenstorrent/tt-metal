// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/eth/common.hpp"
#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "tt_metal/tt_metal/deployment/kernels/noc_integrity_types.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <chrono>
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

static void wait_for_sources(const vector<launched>& runs, uint32_t noc, const params& p) {
    using clock = chrono::steady_clock;
    const auto stall_limit = chrono::milliseconds(2 * p.timeout_ms + 5000);

    vector<uint32_t> last_hb(runs.size(), ~0u);
    vector<clock::time_point> last_change(runs.size(), clock::now());

    for (;;) {
        bool all_done = true;
        vector<size_t> stuck;
        for (size_t i = 0; i < runs.size(); i++) {
            auto prog = read_words(*runs[i].src, runs[i].layout.progress, NOC_PROGRESS_WORDS);
            if (prog[NOC_PROGRESS_STATE] == NOC_STATE_DONE || prog[NOC_PROGRESS_STATE] == NOC_STATE_ABORTED) {
                continue;
            }
            all_done = false;
            if (prog[NOC_PROGRESS_HEARTBEAT] != last_hb[i]) {
                last_hb[i] = prog[NOC_PROGRESS_HEARTBEAT];
                last_change[i] = clock::now();
            } else if (clock::now() - last_change[i] > stall_limit) {
                stuck.push_back(i);
            }
        }
        if (all_done) {
            return;
        }
        if (!stuck.empty()) {
            for (size_t i : stuck) {
                const auto& r = runs[i];
                auto prog = read_words(*r.src, r.layout.progress, NOC_PROGRESS_WORDS);
                uint32_t d = prog[NOC_PROGRESS_DEST];
                log_critical(
                    tt::LogTest,
                    "NOC hang: {} {} -> {} noc {}: pattern {}, iteration {}, state {:08x}",
                    chip_str(r.src->device),
                    src_str(*r.src),
                    d < r.dests.size() ? dst_str(*r.dests[d]) : fmt::format("dest #{}", d),
                    noc,
                    noc_pattern_name(prog[NOC_PROGRESS_PATTERN]),
                    prog[NOC_PROGRESS_ITERATION],
                    prog[NOC_PROGRESS_STATE]);
            }
            log_critical(tt::LogTest, "Timed out! You probably need to reset the device (tt-smi -glx_reset)");
            exit(1);
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }
}

template <typename FIXTURE>
static vector<failure> run_phase(
    FIXTURE* fixture,
    uint32_t noc,
    const vector<vector<source>>& sources_per_chip,
    const map<ChipId, vector<dest>>& dests_per_chip,
    const dest_layout& eth_l,
    const dest_layout& tensix_l,
    const params& p) {
    /* ==================== */
    const auto& hal = MetalContext::instance().hal();
    const uint32_t idle_eriscs = hal.get_num_risc_processors(HalProgrammableCoreType::IDLE_ETH);

    struct plan {
        const source* src;
        core_layout layout;
        vector<uint32_t> compile_args;
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
                        0x9E3779B9u ^ (s.device->id() << 24) ^ (s.slot << 8) ^ noc,
                        p.packet_size,
                        p.burst,
                        p.vc,
                    },
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
            const size_t last = min(first + MAX_DESTS, pl.dests.size());

            launched run{.src = &s, .layout = pl.layout};
            vector<uint32_t> rt_args = {static_cast<uint32_t>(last - first)};
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

        map<shared_ptr<distributed::MeshDevice>, shared_ptr<distributed::MeshWorkload>> workloads;
        for (auto& [mesh, program] : programs) {
            program->impl().compile(mesh.get());
            workloads[mesh] = make_shared<distributed::MeshWorkload>();
            workloads[mesh]->add_program(device_range, std::move(*program));
        }
        for (auto& [mesh, workload] : workloads) {
            fixture->RunProgram(mesh, *workload, true);
        }

        wait_for_sources(runs, noc, p);

        for (auto& [mesh, _] : workloads) {
            fixture->FinishCommands(mesh);
        }

        size_t batch_failures = 0;
        for (const auto& r : runs) {
            auto results = read_words(*r.src, r.layout.results, r.dests.size() * NOC_RESULT_WORDS);
            for (size_t d = 0; d < r.dests.size(); d++) {
                vector<uint32_t> rec(
                    results.begin() + d * NOC_RESULT_WORDS, results.begin() + (d + 1) * NOC_RESULT_WORDS);
                if (rec[NOC_RESULT_STATUS] == NOC_STATUS_DONE && rec[NOC_RESULT_MISMATCHES] == 0) {
                    continue;
                }
                failures.push_back({noc, *r.src, *r.dests[d], r.addrs[d], std::move(rec)});
                batch_failures++;
            }
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

static void report(const vector<failure>& failures) {
    map<string, uint32_t> by_src;
    map<string, uint32_t> by_dst;

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
                log_critical(tt::LogTest, "NOC not run: {}: source aborted before this destination", prefix);
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
        by_src[fmt::format("{} {} noc {}", chip, src_str(f.src), f.noc)]++;
        by_dst[fmt::format("{} {} noc {}", chip, dst_str(f.dst), f.noc)]++;
    }

    if (failures.empty()) {
        return;
    }
    log_critical(tt::LogTest, "Failing sources:");
    for (const auto& [k, n] : by_src) {
        log_critical(tt::LogTest, "\t{}: {} failing destinations", k, n);
    }
    log_critical(tt::LogTest, "Failing destinations:");
    for (const auto& [k, n] : by_dst) {
        log_critical(tt::LogTest, "\t{}: {} failing sources", k, n);
    }
    log_critical(tt::LogTest, "{} failing source/destination pairs in total", failures.size());
}

template <typename FIXTURE>
static vector<failure> run_test(
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

    vector<failure> failures;
    for (uint32_t noc = 0; noc < 2; noc++) {
        auto f = run_phase(fixture, noc, sources_per_chip, dests_per_chip, eth_l, tensix_l, p);
        failures.insert(failures.end(), f.begin(), f.end());
    }
    return failures;
}

}  // namespace deployment_noc

TEST_F(MeshDispatchFixture, TensixDeploymentNoc00EthSourceIntegrity) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    print_detected_devices();

    auto failures = deployment_noc::run_test(this, devices_, false);
    deployment_noc::report(failures);
    ASSERT_TRUE(failures.empty());
}

TEST_F(MeshDispatchFixture, TensixDeploymentNoc01TensixSourceIntegrity) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "Ethernet destinations require slow dispatch";
    }
    print_detected_devices();

    auto failures = deployment_noc::run_test(this, devices_, true);
    deployment_noc::report(failures);
    ASSERT_TRUE(failures.empty());
}

}  // namespace tt::tt_metal
