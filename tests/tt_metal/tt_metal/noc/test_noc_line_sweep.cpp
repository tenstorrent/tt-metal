// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOC line sweep: a per-chip NOC data-integrity test.
//
// Purpose: verify that every core of a chip can move data to every other core over each NOC without corruption or
// loss, using plain NOC reads and writes with host-generated data and host-side verification. It is intended to
// run on each chip after board bring-up and before any fabric or multi-chip test, so that a bad NOC path is found
// at this level and reported as a data mismatch rather than surfacing later as a hang in a higher layer.
//
// Faults this is designed to expose:
//   * A data line stuck or flipping on one NOC path: shows up as the same XOR mask on every bad word.
//   * A fault tied to a position inside a NOC flit: shows up as bad words concentrated on one flit-word index.
//   * A fault that depends on the source, the destination, or the NOC: shows up as a bad (source, destination,
//     NOC) subset while the same cores are clean in every other combination.
//   * A write or read that never completes: shows up as a slot that still holds the poison pattern.
//
// Structure:
//   * A "line" is an ordered set of cores: a logical tensix row, a logical tensix column, the active-Ethernet
//     cores, or the idle-Ethernet cores of a chip.
//   * One "step" takes an active line and a passive line. In PUSH mode every core of the active line runs the
//     kernel and writes its own payload into a dedicated slot on every core of the passive line. In PULL mode
//     every core of the active line reads every passive core's payload into a dedicated local slot.
//   * A sweep iterates: for each active line, for each other passive line, for each NOC (and RISC) combination.
//     Row-major sweeps iterate tensix rows, column-major sweeps iterate columns, and the Ethernet variants pair
//     Ethernet lines with tensix rows and columns.
//   * After every step the host reads every slot back and compares it to the expected payload. Nothing on the
//     device ever waits for data from another core, so corruption cannot turn into a hang.
//
// Environment knobs (all optional):
//   TT_NOC_SWEEP_TRANSFER_BYTES  bytes per (source, destination) pair, multiple of 64     (default 1024)
//   TT_NOC_SWEEP_CHUNK_BYTES     bytes per NOC transaction, 0 = whole transfer per call   (default 0)
//   TT_NOC_SWEEP_ITERS           how many times each step re-sends the same data         (default 1)
//   TT_NOC_SWEEP_SEED            payload RNG seed                                        (default 0xC0FFEE)
//   TT_NOC_SWEEP_NOCS            comma list of NOCs to test, e.g. "1" or "0,1"            (default "0,1")
//   TT_NOC_SWEEP_DEVICES         comma list of chip ids to test, empty = all              (default all)
//   TT_NOC_SWEEP_MAX_REPORT      max bad pairs logged per step                            (default 16)
//   TT_NOC_SWEEP_VERBOSE         1 = log every step and the cores of every line; 0 = one summary line per
//                                sweep per chip plus a progress line every few seconds   (default 0)
//   TT_NOC_SWEEP_INJECT_XOR      checker self-test: OR this mask into the payload the host writes to every
//                                source core while the expected data is left untouched, so every pair must be
//                                reported bad with exactly this signature. 0 = off (default).
//   TT_NOC_SWEEP_INJECT_FLIT_WORD  which 32-bit word of every 64-byte flit receives the injected mask
//                                (default 15).
//
// Run (slow dispatch recommended so every tensix core and every Ethernet core is available):
//   TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_noc --gtest_filter='NocLineSweep*'

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>

#include "env_lib.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/tt_cluster.hpp"
#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal {
namespace {

constexpr uint32_t kPoison = 0xDEADBEEF;
constexpr uint32_t kFlitBytes = 64;
constexpr const char* kKernelPath = "tests/tt_metal/tt_metal/test_kernels/dataflow/noc_line_sweep.cpp";

enum class LineKind { TensixRow, TensixCol, ActiveEth, IdleEth };

struct Line {
    LineKind kind;
    std::string name;
    std::vector<CoreCoord> cores;  // logical coordinates
    HalProgrammableCoreType prog_type;
    CoreType host_type;

    bool is_eth() const { return kind == LineKind::ActiveEth || kind == LineKind::IdleEth; }
};

struct RiscNoc {
    DataMovementProcessor proc;
    NOC noc;
};

// How often a progress line is logged while a sweep runs in non-verbose mode.
constexpr auto kProgressInterval = std::chrono::seconds(10);

struct SweepConfig {
    uint32_t transfer_bytes = 1024;
    uint32_t chunk_bytes = 0;
    uint32_t num_iters = 1;
    uint32_t seed = 0xC0FFEE;
    std::vector<NOC> nocs = {NOC::NOC_0, NOC::NOC_1};
    std::vector<ChipId> devices;  // empty = all
    uint32_t max_report = 16;
    bool verbose = false;
    uint32_t inject_xor = 0;  // checker self-test, see header
    uint32_t inject_flit_word = 15;
};

// Per (source core, destination core) verification result.
struct PairResult {
    CoreCoord src;  // logical
    CoreCoord dst;  // logical
    uint32_t bad_words = 0;
    uint32_t total_words = 0;
    uint32_t xor_or = 0;                                    // OR of (expected ^ got) over all bad words
    uint32_t zero_to_one = 0;                               // bad words where got has bits set that expected does not
    uint32_t one_to_zero = 0;                               // bad words where got lost bits that expected has
    std::array<uint32_t, kFlitBytes / 4> flit_word_hist{};  // histogram of (byte offset % 64) / 4
    bool undelivered = false;                               // entire slot still holds the poison pattern
    uint32_t first_bad_off = 0;
    uint32_t first_exp = 0;
    uint32_t first_got = 0;
};

std::vector<uint32_t> parse_list(const std::string& s) {
    std::vector<uint32_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back(static_cast<uint32_t>(std::stoul(tok, nullptr, 0)));
        }
    }
    return out;
}

SweepConfig config_from_env() {
    SweepConfig cfg;
    cfg.transfer_bytes = tt::parse_env<uint32_t>("TT_NOC_SWEEP_TRANSFER_BYTES", cfg.transfer_bytes);
    cfg.transfer_bytes = std::max<uint32_t>(kFlitBytes, cfg.transfer_bytes - (cfg.transfer_bytes % kFlitBytes));
    cfg.chunk_bytes = tt::parse_env<uint32_t>("TT_NOC_SWEEP_CHUNK_BYTES", cfg.chunk_bytes);
    cfg.chunk_bytes -= cfg.chunk_bytes % kFlitBytes;
    cfg.num_iters = std::max<uint32_t>(1, tt::parse_env<uint32_t>("TT_NOC_SWEEP_ITERS", cfg.num_iters));
    cfg.seed = tt::parse_env<uint32_t>("TT_NOC_SWEEP_SEED", cfg.seed);
    cfg.max_report = tt::parse_env<uint32_t>("TT_NOC_SWEEP_MAX_REPORT", cfg.max_report);
    cfg.verbose = tt::parse_env<uint32_t>("TT_NOC_SWEEP_VERBOSE", 0) != 0;
    cfg.inject_xor = tt::parse_env<uint32_t>("TT_NOC_SWEEP_INJECT_XOR", cfg.inject_xor);
    cfg.inject_flit_word =
        tt::parse_env<uint32_t>("TT_NOC_SWEEP_INJECT_FLIT_WORD", cfg.inject_flit_word) % (kFlitBytes / 4);
    const std::string nocs = tt::parse_env<std::string>("TT_NOC_SWEEP_NOCS", "0,1");
    cfg.nocs.clear();
    for (uint32_t n : parse_list(nocs)) {
        cfg.nocs.push_back(n == 0 ? NOC::NOC_0 : NOC::NOC_1);
    }
    for (uint32_t d : parse_list(tt::parse_env<std::string>("TT_NOC_SWEEP_DEVICES", ""))) {
        cfg.devices.push_back(static_cast<ChipId>(d));
    }
    return cfg;
}

struct Region {
    uint32_t base;
    uint32_t size;
};

Region unreserved_region(HalProgrammableCoreType t) {
    const auto& hal = MetalContext::instance().hal();
    const auto addr_type =
        t == HalProgrammableCoreType::TENSIX ? HalL1MemAddrType::DEFAULT_UNRESERVED : HalL1MemAddrType::UNRESERVED;
    return {static_cast<uint32_t>(hal.get_dev_addr(t, addr_type)), hal.get_dev_size(t, addr_type)};
}

// Layout inside the unreserved region of every core: [payload][slot 0][slot 1]...[slot N-1]
uint32_t payload_addr(HalProgrammableCoreType t) { return unreserved_region(t).base; }
uint32_t slot_addr(HalProgrammableCoreType t, uint32_t transfer_bytes) {
    return unreserved_region(t).base + ((transfer_bytes + 1023) / 1024) * 1024;
}

std::string core_desc(IDevice* device, const CoreCoord& logical, CoreType type) {
    const auto& cluster = MetalContext::instance().get_cluster();
    const CoreCoord virt = device->virtual_core_from_logical_core(logical, type);
    const CoreCoord phys = cluster.get_physical_coordinate_from_logical_coordinates(device->id(), logical, type, true);
    return fmt::format("log({},{}) virt({},{}) phys({},{})", logical.x, logical.y, virt.x, virt.y, phys.x, phys.y);
}

std::string line_desc(IDevice* device, const Line& line) {
    std::string s = fmt::format("{} [{} cores]:", line.name, line.cores.size());
    for (const auto& c : line.cores) {
        s += " " + core_desc(device, c, line.host_type);
    }
    return s;
}

std::vector<CoreCoord> sorted(const std::unordered_set<CoreCoord>& cores) {
    std::vector<CoreCoord> v(cores.begin(), cores.end());
    std::sort(
        v.begin(), v.end(), [](const CoreCoord& a, const CoreCoord& b) { return a.y != b.y ? a.y < b.y : a.x < b.x; });
    return v;
}

std::vector<Line> tensix_rows(IDevice* device) {
    std::vector<Line> lines;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    for (uint32_t y = 0; y < grid.y; y++) {
        Line l{LineKind::TensixRow, fmt::format("row{}", y), {}, HalProgrammableCoreType::TENSIX, CoreType::WORKER};
        for (uint32_t x = 0; x < grid.x; x++) {
            l.cores.emplace_back(x, y);
        }
        lines.push_back(std::move(l));
    }
    return lines;
}

std::vector<Line> tensix_cols(IDevice* device) {
    std::vector<Line> lines;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    for (uint32_t x = 0; x < grid.x; x++) {
        Line l{LineKind::TensixCol, fmt::format("col{}", x), {}, HalProgrammableCoreType::TENSIX, CoreType::WORKER};
        for (uint32_t y = 0; y < grid.y; y++) {
            l.cores.emplace_back(x, y);
        }
        lines.push_back(std::move(l));
    }
    return lines;
}

std::vector<Line> eth_lines(IDevice* device) {
    std::vector<Line> lines;
    auto active = sorted(device->get_active_ethernet_cores(true));
    if (!active.empty()) {
        lines.push_back(
            {LineKind::ActiveEth, "active-eth", std::move(active), HalProgrammableCoreType::ACTIVE_ETH, CoreType::ETH});
    }
    auto idle = sorted(device->get_inactive_ethernet_cores());
    if (!idle.empty()) {
        lines.push_back(
            {LineKind::IdleEth, "idle-eth", std::move(idle), HalProgrammableCoreType::IDLE_ETH, CoreType::ETH});
    }
    return lines;
}

// Which (RISC, NOC) combinations can run a kernel on a line of this kind.
std::vector<RiscNoc> risc_noc_combos(const Line& line, const std::vector<NOC>& nocs) {
    const auto& hal = MetalContext::instance().hal();
    std::vector<RiscNoc> combos;
    if (line.kind == LineKind::ActiveEth && !hal.get_eth_fw_is_cooperative()) {
        // Base firmware owns ERISC0/NOC0 on this architecture. With two ERISCs exposed, keep noc index ==
        // processor index (same rule as eth_test_common::set_arch_specific_eth_config); with one, only NOC1.
        if (hal.get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH) > 1) {
            for (NOC n : nocs) {
                combos.push_back(
                    {n == NOC::NOC_0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1, n});
            }
        } else if (std::find(nocs.begin(), nocs.end(), NOC::NOC_1) != nocs.end()) {
            combos.push_back({DataMovementProcessor::RISCV_0, NOC::NOC_1});
        }
        return combos;
    }
    for (NOC n : nocs) {
        combos.push_back({DataMovementProcessor::RISCV_0, n});
    }
    return combos;
}

bool can_run_kernel(const Line& line, bool slow_dispatch, tt::ARCH arch) {
    if (!line.is_eth()) {
        return true;
    }
    if (!slow_dispatch) {
        return false;  // fast dispatch to Ethernet cores is not supported
    }
    // Idle-ERISC kernels on Wormhole are blocked on an instruction-cache flush issue (#14640).
    if (line.kind == LineKind::IdleEth && arch == tt::ARCH::WORMHOLE_B0) {
        return false;
    }
    return true;
}

std::string pair_signature(const PairResult& r) {
    if (r.undelivered) {
        return "UNDELIVERED (slot still poison)";
    }
    std::string flits;
    for (uint32_t w = 0; w < r.flit_word_hist.size(); w++) {
        if (r.flit_word_hist[w]) {
            flits += fmt::format("{}{}:{}", flits.empty() ? "" : ",", w, r.flit_word_hist[w]);
        }
    }
    const char* dir = (r.zero_to_one && !r.one_to_zero)   ? "0->1 only"
                      : (!r.zero_to_one && r.one_to_zero) ? "1->0 only"
                                                          : "mixed";
    return fmt::format(
        "{}/{} bad words, xor_or=0x{:08x}, {}, flit-word{{idx:count}}={{{}}}, first bad @+0x{:x} exp=0x{:08x} "
        "got=0x{:08x}",
        r.bad_words,
        r.total_words,
        r.xor_or,
        dir,
        flits,
        r.first_bad_off,
        r.first_exp,
        r.first_got);
}

PairResult compare_slot(
    const CoreCoord& src, const CoreCoord& dst, const uint32_t* got, const std::vector<uint32_t>& expected) {
    PairResult r;
    r.src = src;
    r.dst = dst;
    r.total_words = expected.size();
    bool all_poison = true;
    for (uint32_t i = 0; i < expected.size(); i++) {
        const uint32_t e = expected[i];
        const uint32_t g = got[i];
        all_poison &= (g == kPoison);
        if (e == g) {
            continue;
        }
        if (r.bad_words == 0) {
            r.first_bad_off = i * 4;
            r.first_exp = e;
            r.first_got = g;
        }
        r.bad_words++;
        const uint32_t x = e ^ g;
        r.xor_or |= x;
        if (g & ~e) {
            r.zero_to_one++;
        }
        if (e & ~g) {
            r.one_to_zero++;
        }
        r.flit_word_hist[((i * 4) % kFlitBytes) / 4]++;
    }
    r.undelivered = r.bad_words > 0 && all_poison;
    return r;
}

class NocLineSweepFixture : public MeshDispatchFixture {
protected:
    SweepConfig cfg_;
    std::vector<std::string> suite_failures_;
    std::unordered_set<std::string> warned_caps_;  // (source line -> sink line) pairs already warned about
    uint32_t steps_run_ = 0;
    uint32_t pairs_checked_ = 0;
    // Time spent in each phase of run_step since the last reset, so a slow platform can be diagnosed from the
    // sweep summary line alone: host writes (payload + poison), program build + launch + wait, readback + compare.
    double t_write_ = 0, t_run_ = 0, t_read_ = 0;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        cfg_ = config_from_env();
        std::string noc_list;
        for (NOC n : cfg_.nocs) {
            noc_list += fmt::format("{}{}", noc_list.empty() ? "" : ",", static_cast<int>(n));
        }
        log_info(
            tt::LogTest,
            "NOC line sweep config: transfer={}B chunk={}B iters={} seed=0x{:x} nocs={} dispatch={} verbose={} "
            "inject_xor=0x{:x} inject_flit_word={}",
            cfg_.transfer_bytes,
            cfg_.chunk_bytes,
            cfg_.num_iters,
            cfg_.seed,
            noc_list,
            slow_dispatch_ ? "slow" : "fast",
            cfg_.verbose ? 1 : 0,
            cfg_.inject_xor,
            cfg_.inject_flit_word);
        if (cfg_.inject_xor != 0) {
            log_warning(tt::LogTest, "FAULT INJECTION ENABLED: every pair is expected to be reported bad");
        }
    }

    void TearDown() override {
        log_info(
            tt::LogTest,
            "NOC line sweep: {} steps, {} pairs checked, {} bad pairs",
            steps_run_,
            pairs_checked_,
            suite_failures_.size());
        if (!suite_failures_.empty()) {
            log_error(tt::LogTest, "===== NOC line sweep: bad (source -> destination) pairs =====");
            uint32_t printed = 0;
            for (const auto& f : suite_failures_) {
                if (printed++ == cfg_.max_report) {
                    log_error(
                        tt::LogTest,
                        "... and {} more (raise TT_NOC_SWEEP_MAX_REPORT to see them)",
                        suite_failures_.size() - cfg_.max_report);
                    break;
                }
                log_error(tt::LogTest, "{}", f);
            }
        }
        MeshDispatchFixture::TearDown();
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> selected_devices() const {
        std::vector<std::shared_ptr<distributed::MeshDevice>> out;
        for (const auto& md : devices_) {
            const ChipId id = md->get_device_ids().front();
            if (cfg_.devices.empty() || std::find(cfg_.devices.begin(), cfg_.devices.end(), id) != cfg_.devices.end()) {
                out.push_back(md);
            }
        }
        return out;
    }

    // Cap the transfer so that every core of the sink line can hold one slot per source core.
    uint32_t effective_transfer(const Line& sources, const Line& sinks) const {
        uint32_t t = cfg_.transfer_bytes;
        for (const Line* l : {&sources, &sinks}) {
            const Region r = unreserved_region(l->prog_type);
            // payload + num_sources slots must fit; payload area is rounded up to 1 KB.
            const uint32_t need_slots = l == &sinks ? static_cast<uint32_t>(sources.cores.size()) : 0;
            const uint32_t max_t = (r.size - 1024) / (need_slots + 1);
            t = std::min(t, max_t - (max_t % kFlitBytes));
        }
        return std::max<uint32_t>(t, kFlitBytes);
    }

    // Runs one step and returns the number of bad pairs. `pull` selects read (active reads from passive) instead
    // of write (active writes to passive).
    uint32_t run_step(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        const Line& active,
        const Line& passive,
        RiscNoc rn,
        bool pull,
        uint32_t step_index) {
        IDevice* device = mesh_device->get_device(distributed::MeshCoordinate(0, 0));
        const ChipId dev_id = device->id();
        const Line& sources = pull ? passive : active;
        const Line& sinks = pull ? active : passive;
        const uint32_t transfer = effective_transfer(sources, sinks);
        if (transfer != cfg_.transfer_bytes && warned_caps_.insert(sources.name + "->" + sinks.name).second) {
            log_warning(
                tt::LogTest,
                "[dev {}] {} -> {}: transfer capped to {} B so that {} slots fit in the {} unreserved region",
                dev_id,
                sources.name,
                sinks.name,
                transfer,
                sources.cores.size(),
                sinks.name);
        }
        const uint32_t words = transfer / 4;
        const uint32_t src_payload = payload_addr(sources.prog_type);
        const uint32_t sink_slots = slot_addr(sinks.prog_type, transfer);

        auto phase_t = std::chrono::steady_clock::now();
        auto lap = [&phase_t]() {
            const auto now = std::chrono::steady_clock::now();
            const double d = std::chrono::duration<double>(now - phase_t).count();
            phase_t = now;
            return d;
        };

        // 1. Host-generated payload per source core, written into that core's L1.
        const uint32_t step_seed =
            cfg_.seed ^ (static_cast<uint32_t>(dev_id) * 0x9E3779B9u) ^ (step_index * 0x85EBCA6Bu);
        std::vector<std::vector<uint32_t>> payloads(sources.cores.size());
        for (uint32_t s = 0; s < sources.cores.size(); s++) {
            std::mt19937 rng(step_seed + s * 0x27D4EB2Fu);
            payloads[s].resize(words);
            for (auto& w : payloads[s]) {
                w = rng();
            }
            if (cfg_.inject_xor == 0) {
                detail::WriteToDeviceL1(device, sources.cores[s], src_payload, payloads[s], sources.host_type);
            } else {
                // Checker self-test: the device gets a deliberately corrupted copy; the comparison still expects the
                // pristine payload, so every pair must be reported with exactly the injected signature.
                std::vector<uint32_t> corrupted = payloads[s];
                for (uint32_t i = cfg_.inject_flit_word; i < corrupted.size(); i += kFlitBytes / 4) {
                    corrupted[i] |= cfg_.inject_xor;
                }
                detail::WriteToDeviceL1(device, sources.cores[s], src_payload, corrupted, sources.host_type);
            }
        }
        // 2. Poison every slot on every sink core.
        std::vector<uint32_t> poison(words * sources.cores.size(), kPoison);
        for (const auto& c : sinks.cores) {
            detail::WriteToDeviceL1(device, c, sink_slots, poison, sinks.host_type);
        }

        t_write_ += lap();

        // 3. One kernel on every active core.
        Program program = CreateProgram();
        std::set<CoreRange> ranges;
        for (const auto& c : active.cores) {
            ranges.insert(CoreRange(c, c));
        }
        const CoreRangeSet active_set(ranges);
        KernelHandle kernel;
        if (active.is_eth()) {
            kernel = CreateKernel(
                program,
                kKernelPath,
                active_set,
                EthernetConfig{
                    .eth_mode = active.kind == LineKind::IdleEth ? Eth::IDLE : Eth::SENDER,
                    .noc = rn.noc,
                    .processor = rn.proc});
        } else {
            kernel =
                CreateKernel(program, kKernelPath, active_set, DataMovementConfig{.processor = rn.proc, .noc = rn.noc});
        }
        std::vector<uint32_t> peer_args;
        for (const auto& c : passive.cores) {
            const CoreCoord v = device->virtual_core_from_logical_core(c, passive.host_type);
            peer_args.push_back(v.x);
            peer_args.push_back(v.y);
        }
        for (uint32_t i = 0; i < active.cores.size(); i++) {
            std::vector<uint32_t> rt;
            if (pull) {
                // read every passive payload into my slot p
                rt = {
                    1u,
                    sink_slots,
                    src_payload,
                    transfer,
                    cfg_.chunk_bytes,
                    cfg_.num_iters,
                    static_cast<uint32_t>(passive.cores.size())};
            } else {
                // write my payload into slot i on every passive core
                rt = {
                    0u,
                    src_payload,
                    sink_slots + i * transfer,
                    transfer,
                    cfg_.chunk_bytes,
                    cfg_.num_iters,
                    static_cast<uint32_t>(passive.cores.size())};
            }
            rt.insert(rt.end(), peer_args.begin(), peer_args.end());
            SetRuntimeArgs(program, kernel, active.cores[i], rt);
        }
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        RunProgram(mesh_device, workload);
        t_run_ += lap();

        // 4. Read every slot back and compare.
        uint32_t bad_pairs = 0;
        uint32_t reported = 0;
        std::vector<uint32_t> readback;
        for (const auto& sink : sinks.cores) {
            detail::ReadFromDeviceL1(device, sink, sink_slots, poison.size() * 4, readback, sinks.host_type);
            for (uint32_t s = 0; s < sources.cores.size(); s++) {
                pairs_checked_++;
                PairResult r = compare_slot(sources.cores[s], sink, readback.data() + s * words, payloads[s]);
                if (r.bad_words == 0) {
                    continue;
                }
                bad_pairs++;
                const std::string msg = fmt::format(
                    "[dev {}] {} {} -> {} {} over noc{} ({}, {}): {}",
                    dev_id,
                    sources.name,
                    core_desc(device, r.src, sources.host_type),
                    sinks.name,
                    core_desc(device, r.dst, sinks.host_type),
                    static_cast<int>(rn.noc),
                    pull ? "PULL/read" : "PUSH/write",
                    active.is_eth() ? "erisc kernel" : "tensix kernel",
                    pair_signature(r));
                suite_failures_.push_back(msg);
                if (reported++ < cfg_.max_report) {
                    log_error(tt::LogTest, "{}", msg);
                }
            }
        }
        t_read_ += lap();
        steps_run_++;
        if (cfg_.verbose || bad_pairs != 0) {
            log_info(
                tt::LogTest,
                "[dev {}] step {}: {} {} {} over noc{} (risc{}): {} pairs x {} B -> {} bad",
                dev_id,
                step_index,
                active.name,
                pull ? "<-" : "->",
                passive.name,
                static_cast<int>(rn.noc),
                static_cast<int>(rn.proc),
                sources.cores.size() * sinks.cores.size(),
                transfer,
                bad_pairs);
        }
        return bad_pairs;
    }

    // Full sweep: every active line against every passive line (skipping identical lines), every NOC combo.
    void run_sweep(
        const std::string& sweep_name,
        const std::function<std::vector<Line>(IDevice*)>& active_lines_fn,
        const std::function<std::vector<Line>(IDevice*)>& passive_lines_fn,
        bool pull) {
        auto devices = selected_devices();
        ASSERT_FALSE(devices.empty()) << "no devices selected";
        uint32_t total_bad = 0;
        uint32_t steps = 0;
        for (const auto& mesh_device : devices) {
            IDevice* device = mesh_device->get_device(distributed::MeshCoordinate(0, 0));
            auto active_lines = active_lines_fn(device);
            auto passive_lines = passive_lines_fn(device);
            if (!slow_dispatch_) {
                // Under fast dispatch some Ethernet cores host dispatch kernels whose buffers live in the region
                // this test uses, and kernels cannot be launched on Ethernet cores at all. Keep to tensix.
                auto drop_eth = [](std::vector<Line>& lines) {
                    lines.erase(
                        std::remove_if(lines.begin(), lines.end(), [](const Line& l) { return l.is_eth(); }),
                        lines.end());
                };
                drop_eth(active_lines);
                drop_eth(passive_lines);
                log_warning(
                    tt::LogTest,
                    "[dev {}] fast dispatch: Ethernet lines excluded; run with TT_METAL_SLOW_DISPATCH_MODE=1 for "
                    "full coverage",
                    device->id());
            }
            // Enumerate the steps first so progress can be reported as "step k/n".
            struct Step {
                const Line* active;
                const Line* passive;
                RiscNoc rn;
            };
            std::vector<Step> plan;
            for (const auto& active : active_lines) {
                if (!can_run_kernel(active, slow_dispatch_, arch_)) {
                    log_warning(
                        tt::LogTest,
                        "[dev {}] skipping active line {} (kernels on this core type need slow dispatch / are not "
                        "supported on this arch)",
                        device->id(),
                        active.name);
                    continue;
                }
                for (const auto& passive : passive_lines) {
                    if (passive.kind == active.kind && passive.name == active.name) {
                        continue;
                    }
                    for (const RiscNoc& rn : risc_noc_combos(active, cfg_.nocs)) {
                        plan.push_back({&active, &passive, rn});
                    }
                }
            }
            log_info(
                tt::LogTest,
                "[dev {}] sweep '{}': {} active x {} passive lines, {} steps, grid {}x{}, harvesting mask 0x{:x}",
                device->id(),
                sweep_name,
                active_lines.size(),
                passive_lines.size(),
                plan.size(),
                device->compute_with_storage_grid_size().x,
                device->compute_with_storage_grid_size().y,
                MetalContext::instance().get_cluster().get_harvesting_mask(device->id()));
            if (cfg_.verbose) {
                for (const auto& l : active_lines) {
                    log_info(tt::LogTest, "[dev {}] active {}", device->id(), line_desc(device, l));
                }
                for (const auto& l : passive_lines) {
                    log_info(tt::LogTest, "[dev {}] passive {}", device->id(), line_desc(device, l));
                }
            }

            const auto t0 = std::chrono::steady_clock::now();
            auto last_report = t0;
            t_write_ = t_run_ = t_read_ = 0;
            uint32_t dev_bad = 0;
            const uint32_t pairs_before = pairs_checked_;
            for (uint32_t i = 0; i < plan.size(); i++) {
                const Step& st = plan[i];
                dev_bad += run_step(mesh_device, *st.active, *st.passive, st.rn, pull, i);
                const auto now = std::chrono::steady_clock::now();
                if (!cfg_.verbose && i + 1 < plan.size() && now - last_report >= kProgressInterval) {
                    log_info(
                        tt::LogTest,
                        "[dev {}] sweep '{}': step {}/{} ({} {} {} noc{}), {} bad so far, {:.1f} s elapsed",
                        device->id(),
                        sweep_name,
                        i + 1,
                        plan.size(),
                        st.active->name,
                        pull ? "<-" : "->",
                        st.passive->name,
                        static_cast<int>(st.rn.noc),
                        dev_bad,
                        std::chrono::duration<double>(now - t0).count());
                    last_report = now;
                }
            }
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            log_info(
                tt::LogTest,
                "[dev {}] sweep '{}': {} steps, {} pairs, {} bad, {:.1f} s ({:.1f} ms/step: host-write {:.1f}, "
                "build+run {:.1f}, readback+check {:.1f})",
                device->id(),
                sweep_name,
                plan.size(),
                pairs_checked_ - pairs_before,
                dev_bad,
                secs,
                plan.empty() ? 0.0 : 1e3 * secs / plan.size(),
                plan.empty() ? 0.0 : 1e3 * t_write_ / plan.size(),
                plan.empty() ? 0.0 : 1e3 * t_run_ / plan.size(),
                plan.empty() ? 0.0 : 1e3 * t_read_ / plan.size());
            total_bad += dev_bad;
            steps += plan.size();
        }
        if (steps == 0) {
            GTEST_SKIP() << "sweep '" << sweep_name << "' had no runnable steps on this platform/dispatch mode";
        }
        EXPECT_EQ(total_bad, 0u) << "sweep '" << sweep_name << "': " << total_bad
                                 << " (source -> destination) pairs delivered wrong data; see log for signatures";
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Tensix <-> tensix

// Every tensix row writes into every other tensix row (row-major sweep).
TEST_F(NocLineSweepFixture, TensixRowMajorPush) { run_sweep("tensix row-major push", tensix_rows, tensix_rows, false); }

// Every tensix column writes into every other tensix column (column-major sweep).
TEST_F(NocLineSweepFixture, TensixColMajorPush) { run_sweep("tensix col-major push", tensix_cols, tensix_cols, false); }

// Same sweeps with NOC reads instead of writes (exercises the read request / read response path).
TEST_F(NocLineSweepFixture, TensixRowMajorPull) { run_sweep("tensix row-major pull", tensix_rows, tensix_rows, true); }
TEST_F(NocLineSweepFixture, TensixColMajorPull) { run_sweep("tensix col-major pull", tensix_cols, tensix_cols, true); }

// ---------------------------------------------------------------------------------------------------------------
// Ethernet <-> tensix. This is the path the fabric router uses: an ERISC kernel writing into tensix L1.

// Every Ethernet core (active and idle groups) writes into every tensix row and every tensix column.
TEST_F(NocLineSweepFixture, EthToTensixPush) {
    run_sweep(
        "eth -> tensix push",
        eth_lines,
        [](IDevice* d) {
            auto lines = tensix_rows(d);
            auto cols = tensix_cols(d);
            lines.insert(lines.end(), cols.begin(), cols.end());
            return lines;
        },
        false);
}

// Every Ethernet core reads from every tensix row and column (the inbound direction into the Ethernet core's L1).
TEST_F(NocLineSweepFixture, EthFromTensixPull) {
    run_sweep(
        "eth <- tensix pull",
        eth_lines,
        [](IDevice* d) {
            auto lines = tensix_rows(d);
            auto cols = tensix_cols(d);
            lines.insert(lines.end(), cols.begin(), cols.end());
            return lines;
        },
        true);
}

// Every tensix row and column writes into the Ethernet cores (no kernel on the Ethernet core, so this also works
// under fast dispatch).
TEST_F(NocLineSweepFixture, TensixToEthPush) {
    run_sweep(
        "tensix -> eth push",
        [](IDevice* d) {
            auto lines = tensix_rows(d);
            auto cols = tensix_cols(d);
            lines.insert(lines.end(), cols.begin(), cols.end());
            return lines;
        },
        eth_lines,
        false);
}

// Active Ethernet cores <-> idle Ethernet cores.
TEST_F(NocLineSweepFixture, EthToEthPush) { run_sweep("eth -> eth push", eth_lines, eth_lines, false); }

}  // namespace
}  // namespace tt::tt_metal
