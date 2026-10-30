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
//   * A data line stuck at 0 or 1, or inverted, on one NOC path: shows up as the same XOR mask on every bad word,
//     and is classified as stuck-at-0 / stuck-at-1 / inverted / intermittent. Every payload is built from random
//     flits each followed by its bitwise complement, so every bit at every flit position is sent as both 0 and 1
//     and the classification is exact rather than statistical.
//   * A fault tied to a position inside a NOC flit: shows up as bad words concentrated on one flit-word index.
//   * A fault that depends on the source, the destination, or the NOC: shows up as a bad (source, destination,
//     NOC) subset while the same cores are clean in every other combination.
//   * A write or read that never completes: shows up as a slot that still holds the poison pattern.
//
// Structure:
//   * A "line" is an ordered set of NOC endpoints: a logical tensix row, a logical tensix column, the
//     active-Ethernet cores, the idle-Ethernet cores, or the DRAM banks of a chip.
//   * One "step" takes an active line and a passive line. In PUSH mode every core of the active line runs the
//     kernel and writes its own payload into a dedicated slot on its destination cores in the passive line. In
//     PULL mode every core of the active line reads its source cores' payloads into dedicated local slots. Which
//     (source, destination) pairs a step uses is defined by step_pairs below. DRAM cannot run a kernel, so it is
//     only ever the passive line.
//   * A sweep iterates: for each transfer size, for each active line, for each passive line (itself included), for
//     each NOC (and RISC) combination.
//   * After every step the host reads every slot back and compares it to the expected payload. Nothing on the
//     device ever waits for data from another core, so corruption cannot turn into a hang.
//
// Environment knobs (all optional):
//   TT_NOC_SWEEP_TRANSFER_BYTES  comma list of bytes per (source, destination) pair, each a multiple of 128;
//                                every step of every sweep runs once per size   (default 1024,2048,4096,8192)
//   TT_NOC_SWEEP_CHUNK_BYTES     bytes per NOC transaction, 0 = whole transfer per call   (default 0)
//   TT_NOC_SWEEP_ITERS           how many times each step re-sends the same data, to keep the path busy
//                                longer. Only what the last repetition left behind is checked. (default 1)
//   TT_NOC_SWEEP_SEED            payload RNG seed                                        (default 0xC0FFEE)
//   TT_NOC_SWEEP_NOCS            comma list of NOCs to test, e.g. "1" or "0,1"            (default "0,1")
//   TT_NOC_SWEEP_DEVICES         comma list of chip ids to test, empty = all              (default all)
//   TT_NOC_SWEEP_MAX_REPORT      max bad pairs logged per step                            (default 16)
//   TT_NOC_SWEEP_VERBOSE         1 = log every step and the cores of every line; 0 = one summary line per
//                                sweep per chip plus a progress line every few seconds   (default 0)
//   TT_NOC_SWEEP_INJECT_XOR      checker self-test: apply this bit mask to the payload the host writes to every
//                                source core while the expected data is left untouched, so every pair must be
//                                reported bad with exactly this signature. 0 = off (default).
//   TT_NOC_SWEEP_INJECT_MODE     how the self-test mask is applied: "set" (stuck-at-1), "clear" (stuck-at-0) or
//                                "flip" (inverted)                                        (default "set")
//   TT_NOC_SWEEP_INJECT_FLIT_WORD  which 32-bit word of every 64-byte flit receives the injected mask
//                                (default 15).
//   TT_NOC_SWEEP_SKIP_ETH_CHANS  comma list of logical Ethernet channels left out of every Ethernet line, or
//                                "none". Default: 10,11 on Blackhole (their SerDes is shared with PCIe and only
//                                one of the two can be in use), none elsewhere.
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

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>

#include "env_lib.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/tt_cluster.hpp"
#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal {
namespace {

constexpr uint32_t kPoison = 0xDEADBEEF;
constexpr uint32_t kFlitBytes = 64;
constexpr uint32_t kFlitWords = kFlitBytes / 4;
// Payloads are built from pairs of flits: a random flit followed by its bitwise complement, so every bit at
// every flit-word position is sent as both 0 and 1 within each pair. Transfers are multiples of this.
constexpr uint32_t kFlitPairBytes = 2 * kFlitBytes;
// The payload area of every region is rounded up to this, so the slots that follow it start aligned.
constexpr uint32_t kPayloadAlign = 1024;
constexpr const char* kKernelPath = "tests/tt_metal/tt_metal/test_kernels/dataflow/noc_line_sweep.cpp";

enum class LineKind { TensixRow, TensixCol, ActiveEth, IdleEth, Dram };

struct Line {
    LineKind kind;
    std::string name;
    std::vector<CoreCoord> cores;       // logical coordinates; for the DRAM line (bank id, 0)
    HalProgrammableCoreType prog_type;  // unused for the DRAM line
    CoreType host_type;

    bool is_eth() const { return kind == LineKind::ActiveEth || kind == LineKind::IdleEth; }
    bool is_dram() const { return kind == LineKind::Dram; }
};

struct RiscNoc {
    DataMovementProcessor proc;
    NOC noc;
};

// How often a progress line is logged while a sweep runs in non-verbose mode.
constexpr auto kProgressInterval = std::chrono::seconds(10);

struct SweepConfig {
    std::vector<uint32_t> transfer_sizes = {1024, 2048, 4096, 8192};
    uint32_t chunk_bytes = 0;
    uint32_t num_iters = 1;
    uint32_t seed = 0xC0FFEE;
    std::vector<NOC> nocs = {NOC::NOC_0, NOC::NOC_1};
    std::vector<ChipId> devices;  // empty = all
    uint32_t max_report = 16;
    bool verbose = false;
    uint32_t inject_xor = 0;  // checker self-test, see header
    std::string inject_mode = "set";
    uint32_t inject_flit_word = 15;
    std::vector<uint32_t> skip_eth_chans;  // resolved per architecture in SetUp unless given by the environment
    bool skip_eth_chans_from_env = false;
};

// Per (source core, destination core) verification result.
struct PairResult {
    CoreCoord src;  // logical
    CoreCoord dst;  // logical
    uint32_t bad_words = 0;
    uint32_t total_words = 0;
    uint32_t xor_or = 0;                                    // OR of (expected ^ got) over all bad words
    std::array<uint32_t, kFlitBytes / 4> flit_word_hist{};  // histogram of (byte offset % 64) / 4
    // Fault classification. Over the bits of xor_or at the flit-word positions that had any bad word: how many
    // times a 0 was sent (chance for a 0->1 flip) and how many of those arrived as 1, and the same for 1->0.
    uint32_t chances_0to1 = 0;
    uint32_t flips_0to1 = 0;
    uint32_t chances_1to0 = 0;
    uint32_t flips_1to0 = 0;
    bool undelivered = false;  // entire slot still holds the poison pattern
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
    const std::string sizes = tt::parse_env<std::string>("TT_NOC_SWEEP_TRANSFER_BYTES", "");
    if (!sizes.empty()) {
        cfg.transfer_sizes = parse_list(sizes);
        TT_FATAL(!cfg.transfer_sizes.empty(), "TT_NOC_SWEEP_TRANSFER_BYTES must list at least one size");
    }
    for (auto& t : cfg.transfer_sizes) {
        t = std::max<uint32_t>(kFlitPairBytes, t - (t % kFlitPairBytes));
    }
    cfg.chunk_bytes = tt::parse_env<uint32_t>("TT_NOC_SWEEP_CHUNK_BYTES", cfg.chunk_bytes);
    cfg.chunk_bytes -= cfg.chunk_bytes % kFlitBytes;
    cfg.num_iters = std::max<uint32_t>(1, tt::parse_env<uint32_t>("TT_NOC_SWEEP_ITERS", cfg.num_iters));
    cfg.seed = tt::parse_env<uint32_t>("TT_NOC_SWEEP_SEED", cfg.seed);
    cfg.max_report = tt::parse_env<uint32_t>("TT_NOC_SWEEP_MAX_REPORT", cfg.max_report);
    cfg.verbose = tt::parse_env<uint32_t>("TT_NOC_SWEEP_VERBOSE", 0) != 0;
    cfg.inject_xor = tt::parse_env<uint32_t>("TT_NOC_SWEEP_INJECT_XOR", cfg.inject_xor);
    cfg.inject_mode = tt::parse_env<std::string>("TT_NOC_SWEEP_INJECT_MODE", cfg.inject_mode);
    TT_FATAL(
        cfg.inject_mode == "set" || cfg.inject_mode == "clear" || cfg.inject_mode == "flip",
        "TT_NOC_SWEEP_INJECT_MODE must be set, clear or flip");
    cfg.inject_flit_word = tt::parse_env<uint32_t>("TT_NOC_SWEEP_INJECT_FLIT_WORD", cfg.inject_flit_word) % kFlitWords;
    const std::string nocs = tt::parse_env<std::string>("TT_NOC_SWEEP_NOCS", "0,1");
    cfg.nocs.clear();
    for (uint32_t n : parse_list(nocs)) {
        TT_FATAL(n <= 1, "TT_NOC_SWEEP_NOCS: NOC {} does not exist (0 or 1)", n);
        cfg.nocs.push_back(n == 0 ? NOC::NOC_0 : NOC::NOC_1);
    }
    for (uint32_t d : parse_list(tt::parse_env<std::string>("TT_NOC_SWEEP_DEVICES", ""))) {
        cfg.devices.push_back(static_cast<ChipId>(d));
    }
    const std::string skip = tt::parse_env<std::string>("TT_NOC_SWEEP_SKIP_ETH_CHANS", "default");
    if (skip != "default") {
        cfg.skip_eth_chans_from_env = true;
        cfg.skip_eth_chans = skip == "none" ? std::vector<uint32_t>{} : parse_list(skip);
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

// Layout of every endpoint's region: [payload][slot 0][slot 1]...[slot N-1], payload area rounded up to 1 KB.
// For a core the region is its unreserved L1; for a DRAM bank it is that bank's page of a buffer allocated per
// step.
uint32_t payload_area_bytes(uint32_t transfer_bytes) {
    return ((transfer_bytes + kPayloadAlign - 1) / kPayloadAlign) * kPayloadAlign;
}

// DRAM view (what the allocator calls a DRAM channel) behind an allocator bank id.
uint32_t dram_view_of_bank(IDevice* device, uint32_t bank) {
    return device->allocator()->impl().get_dram_channel_from_bank_id(bank);
}

// NOC endpoint of a DRAM bank for one NOC, for reporting only: the view's preferred worker endpoint, which is the
// endpoint an interleaved-buffer access from a kernel takes on that NOC. The kernel addresses a bank by bank id
// through the same tables, so this is never used for addressing. Translated (NOC0-style) coordinates.
CoreCoord dram_endpoint(IDevice* device, uint32_t bank, NOC noc) {
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    return soc.get_preferred_worker_core_for_dram_view(dram_view_of_bank(device, bank), static_cast<uint8_t>(noc));
}

std::string core_desc(IDevice* device, const Line& line, uint32_t i, NOC noc) {
    const auto& cluster = MetalContext::instance().get_cluster();
    if (line.is_dram()) {
        const auto& soc = cluster.get_soc_desc(device->id());
        const CoreCoord ep = dram_endpoint(device, i, noc);
        return fmt::format(
            "bank{} ch{} endpoint tr({},{})",
            i,
            soc.get_channel_for_dram_view(dram_view_of_bank(device, i)),
            ep.x,
            ep.y);
    }
    const CoreCoord logical = line.cores[i];
    const CoreCoord virt = device->virtual_core_from_logical_core(logical, line.host_type);
    const CoreCoord phys =
        cluster.get_physical_coordinate_from_logical_coordinates(device->id(), logical, line.host_type, true);
    return fmt::format("log({},{}) virt({},{}) phys({},{})", logical.x, logical.y, virt.x, virt.y, phys.x, phys.y);
}

std::string line_desc(IDevice* device, const Line& line) {
    std::string s = fmt::format("{} [{} endpoints]:", line.name, line.cores.size());
    for (uint32_t i = 0; i < line.cores.size(); i++) {
        if (line.is_dram()) {
            const CoreCoord e0 = dram_endpoint(device, i, NOC::NOC_0);
            const CoreCoord e1 = dram_endpoint(device, i, NOC::NOC_1);
            s += fmt::format(" bank{} noc0({},{}) noc1({},{})", i, e0.x, e0.y, e1.x, e1.y);
        } else {
            s += " " + core_desc(device, line, i, NOC::NOC_0);
        }
    }
    return s;
}

std::vector<CoreCoord> sorted(const std::unordered_set<CoreCoord>& cores) {
    std::vector<CoreCoord> v(cores.begin(), cores.end());
    std::sort(
        v.begin(), v.end(), [](const CoreCoord& a, const CoreCoord& b) { return a.y != b.y ? a.y < b.y : a.x < b.x; });
    return v;
}

std::vector<Line> tensix_rows(IDevice* device, const SweepConfig& /*cfg*/) {
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

std::vector<Line> tensix_cols(IDevice* device, const SweepConfig& /*cfg*/) {
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

// The Ethernet cores whose logical channel is in cfg.skip_eth_chans, with their channel, sorted by channel.
std::vector<std::pair<uint32_t, CoreCoord>> skipped_eth_cores(IDevice* device, const SweepConfig& cfg) {
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    std::vector<std::pair<uint32_t, CoreCoord>> out;
    for (const auto& [logical, chan] : soc.logical_eth_core_to_chan_map) {
        const auto c = static_cast<uint32_t>(chan);
        if (std::find(cfg.skip_eth_chans.begin(), cfg.skip_eth_chans.end(), c) != cfg.skip_eth_chans.end()) {
            out.emplace_back(c, logical);
        }
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
}

// The active-Ethernet cores and the idle-Ethernet cores as two lines, minus the skipped channels.
std::vector<Line> eth_lines(IDevice* device, const SweepConfig& cfg) {
    std::unordered_set<CoreCoord> skipped;
    for (const auto& [chan, logical] : skipped_eth_cores(device, cfg)) {
        skipped.insert(logical);
    }
    auto without_skipped = [&](std::unordered_set<CoreCoord> cores) {
        for (const auto& c : skipped) {
            cores.erase(c);
        }
        return sorted(cores);
    };
    std::vector<Line> lines;
    auto active = without_skipped(device->get_active_ethernet_cores(true));
    if (!active.empty()) {
        lines.push_back(
            {LineKind::ActiveEth, "active-eth", std::move(active), HalProgrammableCoreType::ACTIVE_ETH, CoreType::ETH});
    }
    auto idle = without_skipped(device->get_inactive_ethernet_cores());
    if (!idle.empty()) {
        lines.push_back(
            {LineKind::IdleEth, "idle-eth", std::move(idle), HalProgrammableCoreType::IDLE_ETH, CoreType::ETH});
    }
    return lines;
}

// Every DRAM bank of the chip as one line. Only ever a passive line (no kernel can run there).
std::vector<Line> dram_line(IDevice* device, const SweepConfig& /*cfg*/) {
    Line l{LineKind::Dram, "dram", {}, HalProgrammableCoreType::TENSIX, CoreType::DRAM};
    const uint32_t num_banks = device->allocator()->get_num_banks(BufferType::DRAM);
    for (uint32_t b = 0; b < num_banks; b++) {
        l.cores.emplace_back(b, 0);
    }
    return {std::move(l)};
}

std::vector<Line> tensix_rows_and_cols(IDevice* device, const SweepConfig& cfg) {
    auto lines = tensix_rows(device, cfg);
    auto cols = tensix_cols(device, cfg);
    lines.insert(lines.end(), cols.begin(), cols.end());
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

// Human-readable fault class from the chance / flip counts.
std::string fault_class(const PairResult& r) {
    const bool all_0to1 = r.flips_0to1 == r.chances_0to1;
    const bool all_1to0 = r.flips_1to0 == r.chances_1to0;
    const char* cls = "intermittent mixed";
    if (all_0to1 && all_1to0) {
        cls = "inverted";
    } else if (all_0to1 && r.flips_1to0 == 0) {
        cls = "stuck-at-1";
    } else if (all_1to0 && r.flips_0to1 == 0) {
        cls = "stuck-at-0";
    } else if (r.flips_0to1 != 0 && r.flips_1to0 == 0) {
        cls = "intermittent 0->1";
    } else if (r.flips_0to1 == 0 && r.flips_1to0 != 0) {
        cls = "intermittent 1->0";
    }
    return fmt::format(
        "{} [0->1: {}/{}, 1->0: {}/{}]", cls, r.flips_0to1, r.chances_0to1, r.flips_1to0, r.chances_1to0);
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
    return fmt::format(
        "{}/{} bad words, xor_or=0x{:08x}, fault={}, flit-word{{idx:count}}={{{}}}, first bad @+0x{:x} "
        "exp=0x{:08x} got=0x{:08x}",
        r.bad_words,
        r.total_words,
        r.xor_or,
        fault_class(r),
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
        r.flit_word_hist[((i * 4) % kFlitBytes) / 4]++;
    }
    r.undelivered = r.bad_words > 0 && all_poison;
    if (r.bad_words == 0 || r.undelivered) {
        return r;
    }
    // Second pass: at every affected flit-word position, count how often each bit of the mask was sent as 0 / 1
    // and how often it arrived flipped. A stuck-at takes every chance in one direction and none in the other; an
    // inverter takes every chance in both; anything less is intermittent.
    for (uint32_t i = 0; i < expected.size(); i++) {
        if (r.flit_word_hist[i % kFlitWords] == 0) {
            continue;
        }
        const uint32_t e = expected[i] & r.xor_or;
        const uint32_t g = got[i] & r.xor_or;
        r.chances_0to1 += __builtin_popcount(r.xor_or & ~e);
        r.chances_1to0 += __builtin_popcount(e);
        r.flips_0to1 += __builtin_popcount(g & ~e);
        r.flips_1to0 += __builtin_popcount(e & ~g);
    }
    return r;
}

// Host-only checks of the checker itself, runnable without hardware: build a complement-flit payload, corrupt a
// copy the way a given fault would, and confirm compare_slot / fault_class report exactly that fault.
namespace checker_test {

std::vector<uint32_t> payload(uint32_t words, uint32_t seed = 1) {
    std::mt19937 rng(seed);
    std::vector<uint32_t> p(words);
    for (uint32_t i = 0; i < words; i++) {
        p[i] = (i / kFlitWords) % 2 == 1 ? ~p[i - kFlitWords] : rng();
    }
    return p;
}

// Apply `mask` to flit word `w` of every flit: mode 's' sets, 'c' clears, 'f' flips.
std::vector<uint32_t> corrupt(std::vector<uint32_t> p, uint32_t w, uint32_t mask, char mode) {
    for (uint32_t i = w; i < p.size(); i += kFlitWords) {
        p[i] = mode == 's' ? (p[i] | mask) : mode == 'c' ? (p[i] & ~mask) : (p[i] ^ mask);
    }
    return p;
}

constexpr uint32_t kWords = 256;  // 1 KB: 16 flits, 8 random + 8 complements
constexpr uint32_t kBit = 0x800;
constexpr uint32_t kWord = 15;

}  // namespace checker_test

TEST(NocLineSweepChecker, CleanSlotIsNotReported) {
    const auto exp = checker_test::payload(checker_test::kWords);
    const PairResult r = compare_slot({0, 0}, {1, 0}, exp.data(), exp);
    EXPECT_EQ(r.bad_words, 0u);
    EXPECT_FALSE(r.undelivered);
}

TEST(NocLineSweepChecker, StuckAtOneOnOneFlitWord) {
    using namespace checker_test;
    const auto exp = payload(kWords);
    const auto got = corrupt(exp, kWord, kBit, 's');
    const PairResult r = compare_slot({0, 0}, {1, 0}, got.data(), exp);
    // Complement flits guarantee exactly half the 16 samples of word 15 carry a 0 at bit 11.
    EXPECT_EQ(r.bad_words, 8u);
    EXPECT_EQ(r.xor_or, kBit);
    EXPECT_EQ(fault_class(r), "stuck-at-1 [0->1: 8/8, 1->0: 0/8]");
    for (uint32_t w = 0; w < kFlitWords; w++) {
        EXPECT_EQ(r.flit_word_hist[w], w == kWord ? 8u : 0u) << "flit word " << w;
    }
    EXPECT_FALSE(r.undelivered);
}

TEST(NocLineSweepChecker, StuckAtZeroOnOneFlitWord) {
    using namespace checker_test;
    const auto exp = payload(kWords);
    const auto got = corrupt(exp, kWord, kBit, 'c');
    const PairResult r = compare_slot({0, 0}, {1, 0}, got.data(), exp);
    EXPECT_EQ(r.bad_words, 8u);
    EXPECT_EQ(fault_class(r), "stuck-at-0 [0->1: 0/8, 1->0: 8/8]");
}

TEST(NocLineSweepChecker, InvertedBitOnOneFlitWord) {
    using namespace checker_test;
    const auto exp = payload(kWords);
    const auto got = corrupt(exp, kWord, kBit, 'f');
    const PairResult r = compare_slot({0, 0}, {1, 0}, got.data(), exp);
    EXPECT_EQ(r.bad_words, 16u);
    EXPECT_EQ(fault_class(r), "inverted [0->1: 8/8, 1->0: 8/8]");
}

TEST(NocLineSweepChecker, IntermittentFlipIsNotAStuckAt) {
    using namespace checker_test;
    const auto exp = payload(kWords);
    auto got = exp;
    // One flit's word 15 loses bit 11 where it was 1; the other 15 flits are intact.
    for (uint32_t i = kWord; i < got.size(); i += kFlitWords) {
        if (exp[i] & kBit) {
            got[i] &= ~kBit;
            break;
        }
    }
    const PairResult r = compare_slot({0, 0}, {1, 0}, got.data(), exp);
    EXPECT_EQ(r.bad_words, 1u);
    EXPECT_EQ(fault_class(r), "intermittent 1->0 [0->1: 0/8, 1->0: 1/8]");
}

TEST(NocLineSweepChecker, PoisonedSlotIsUndelivered) {
    using namespace checker_test;
    const auto exp = payload(kWords);
    const std::vector<uint32_t> got(kWords, kPoison);
    const PairResult r = compare_slot({0, 0}, {1, 0}, got.data(), exp);
    EXPECT_TRUE(r.undelivered);
    EXPECT_EQ(pair_signature(r), "UNDELIVERED (slot still poison)");
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
    // Devices whose skipped Ethernet channels have been logged.
    std::unordered_set<ChipId> reported_skips_;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        cfg_ = config_from_env();
        if (!cfg_.skip_eth_chans_from_env && arch_ == tt::ARCH::BLACKHOLE) {
            cfg_.skip_eth_chans = {10, 11};
        }
        std::string skip_list;
        for (uint32_t c : cfg_.skip_eth_chans) {
            skip_list += fmt::format("{}{}", skip_list.empty() ? "" : ",", c);
        }
        std::string noc_list;
        for (NOC n : cfg_.nocs) {
            noc_list += fmt::format("{}{}", noc_list.empty() ? "" : ",", static_cast<int>(n));
        }
        std::string size_list;
        for (uint32_t t : cfg_.transfer_sizes) {
            size_list += fmt::format("{}{}", size_list.empty() ? "" : ",", t);
        }
        log_info(
            tt::LogTest,
            "NOC line sweep config: transfer={}B chunk={}B iters={} seed=0x{:x} nocs={} dispatch={} verbose={} "
            "inject_xor=0x{:x} inject_flit_word={} skip_eth_chans={}",
            size_list,
            cfg_.chunk_bytes,
            cfg_.num_iters,
            cfg_.seed,
            noc_list,
            slow_dispatch_ ? "slow" : "fast",
            cfg_.verbose ? 1 : 0,
            cfg_.inject_xor,
            cfg_.inject_flit_word,
            skip_list.empty() ? "none" : skip_list);
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

    // Cap the transfer so that every core of the sink line can hold `num_slots` slots next to its payload. The DRAM
    // line is not capped: its buffer is sized to the step.
    uint32_t effective_transfer(const Line& sources, const Line& sinks, uint32_t num_slots, uint32_t requested) const {
        uint32_t t = requested;
        for (const Line* l : {&sources, &sinks}) {
            if (l->is_dram()) {
                continue;
            }
            const Region r = unreserved_region(l->prog_type);
            TT_FATAL(
                r.size > kPayloadAlign, "{}: unreserved region of {} B is too small for a payload", l->name, r.size);
            const uint32_t need_slots = l == &sinks ? num_slots : 0;
            const uint32_t max_t = (r.size - kPayloadAlign) / (need_slots + 1);
            t = std::min(t, max_t - (max_t % kFlitPairBytes));
        }
        return std::max<uint32_t>(t, kFlitPairBytes);
    }

    // One transfer of a step: the payload of source `src` lands in slot `slot` of sink `dst` (indices into the
    // source and sink lines).
    struct Pair {
        uint32_t src;
        uint32_t dst;
        uint32_t slot;
    };

    // The (source, sink) pairs a step exercises:
    //  * A line against itself: every core against every other core of the line, never against itself. All the
    //    traffic stays inside the line.
    //  * Two tensix lines of the same kind (row -> row, column -> column): only the cores with the same index. Every
    //    packet then travels perpendicular to both lines, so neither line carries traffic along itself.
    //  * Everything else (an Ethernet group or the DRAM line against anything, or anything against them): every
    //    core against every core, since those lines have no geometry to align with.
    static std::vector<Pair> step_pairs(const Line& sources, const Line& sinks) {
        std::vector<Pair> pairs;
        const bool same_line = sources.kind == sinks.kind && sources.name == sinks.name;
        const bool aligned = !same_line && sources.kind == sinks.kind && !sources.is_eth() && !sources.is_dram() &&
                             sources.cores.size() == sinks.cores.size();
        if (aligned) {
            for (uint32_t i = 0; i < sources.cores.size(); i++) {
                pairs.push_back({i, i, 0});
            }
            return pairs;
        }
        for (uint32_t s = 0; s < sources.cores.size(); s++) {
            for (uint32_t d = 0; d < sinks.cores.size(); d++) {
                if (same_line && s == d) {
                    continue;
                }
                pairs.push_back({s, d, s});
            }
        }
        return pairs;
    }

    // Runs one step and returns the number of bad pairs. `pull` selects read (active reads from passive) instead
    // of write (active writes to passive).
    uint32_t run_step(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        const Line& active,
        const Line& passive,
        RiscNoc rn,
        bool pull,
        uint32_t requested_transfer,
        uint32_t step_index) {
        IDevice* device = mesh_device->get_device(0, 0);
        const ChipId dev_id = device->id();
        TT_FATAL(!active.is_dram(), "the DRAM line cannot be the active line");
        const Line& sources = pull ? passive : active;
        const Line& sinks = pull ? active : passive;
        const std::vector<Pair> pairs = step_pairs(sources, sinks);
        uint32_t num_slots = 0;
        for (const Pair& p : pairs) {
            num_slots = std::max(num_slots, p.slot + 1);
        }
        const uint32_t transfer = effective_transfer(sources, sinks, num_slots, requested_transfer);
        if (transfer != requested_transfer &&
            warned_caps_.insert(fmt::format("{}->{}@{}", sources.name, sinks.name, requested_transfer)).second) {
            log_warning(
                tt::LogTest,
                "[dev {}] {} -> {}: transfer capped to {} B so that {} slots fit in the {} unreserved region",
                dev_id,
                sources.name,
                sinks.name,
                transfer,
                num_slots,
                sinks.name);
        }
        const uint32_t words = transfer / 4;
        const uint32_t payload_area = payload_area_bytes(transfer);
        const uint32_t slots_bytes = num_slots * transfer;

        // The DRAM line lives in a buffer allocated for this step: one page per bank, laid out like a core's
        // region, with page b on bank b. The host keeps a full image of it and moves it in one write and one read.
        const bool dram_passive = passive.is_dram();
        std::shared_ptr<distributed::MeshBuffer> dram_buf;
        std::vector<uint32_t> dram_image;
        uint32_t page_words = 0;
        if (dram_passive) {
            const uint32_t page_bytes = payload_area + (pull ? 0 : slots_bytes);
            page_words = page_bytes / 4;
            distributed::DeviceLocalBufferConfig local{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
            distributed::ReplicatedBufferConfig global{
                .size = static_cast<DeviceAddr>(page_bytes) * passive.cores.size()};
            dram_buf = distributed::MeshBuffer::create(global, local, mesh_device.get());
            dram_image.assign(page_words * passive.cores.size(), kPoison);
        }
        // Base of every endpoint's region in `line`, as the kernel addresses it: the unreserved L1 base, or for the
        // DRAM line the buffer address (the kernel's bank-id lookup adds each bank's own offset).
        auto region_base = [&](const Line& line) -> uint32_t {
            if (line.is_dram()) {
                return static_cast<uint32_t>(dram_buf->address());
            }
            return unreserved_region(line.prog_type).base;
        };

        auto phase_t = std::chrono::steady_clock::now();
        auto lap = [&phase_t]() {
            const auto now = std::chrono::steady_clock::now();
            const double d = std::chrono::duration<double>(now - phase_t).count();
            phase_t = now;
            return d;
        };

        // 1. Host-generated payload per source, written into that source's region.
        const uint32_t step_seed =
            cfg_.seed ^ (static_cast<uint32_t>(dev_id) * 0x9E3779B9u) ^ (step_index * 0x85EBCA6Bu);
        std::vector<std::vector<uint32_t>> payloads(sources.cores.size());
        for (uint32_t s = 0; s < sources.cores.size(); s++) {
            std::mt19937 rng(step_seed + s * 0x27D4EB2Fu);
            payloads[s].resize(words);
            for (uint32_t i = 0; i < words; i++) {
                // Even flits are random; each odd flit is the complement of the flit before it.
                const bool complement_flit = (i / kFlitWords) % 2 == 1;
                payloads[s][i] = complement_flit ? ~payloads[s][i - kFlitWords] : rng();
            }
            std::vector<uint32_t> to_device = payloads[s];
            if (cfg_.inject_xor != 0) {
                // Checker self-test: the device gets a deliberately corrupted copy; the comparison still expects the
                // pristine payload, so every pair must be reported with exactly the injected signature.
                for (uint32_t i = cfg_.inject_flit_word; i < to_device.size(); i += kFlitWords) {
                    if (cfg_.inject_mode == "set") {
                        to_device[i] |= cfg_.inject_xor;
                    } else if (cfg_.inject_mode == "clear") {
                        to_device[i] &= ~cfg_.inject_xor;
                    } else {
                        to_device[i] ^= cfg_.inject_xor;
                    }
                }
            }
            if (sources.is_dram()) {
                std::copy(to_device.begin(), to_device.end(), dram_image.begin() + s * page_words);
            } else {
                detail::WriteToDeviceL1(device, sources.cores[s], region_base(sources), to_device, sources.host_type);
            }
        }
        // 2. Poison every slot on every sink (the DRAM image starts out as poison).
        std::vector<uint32_t> poison(words * num_slots, kPoison);
        if (!sinks.is_dram()) {
            for (uint32_t d = 0; d < sinks.cores.size(); d++) {
                detail::WriteToDeviceL1(
                    device, sinks.cores[d], region_base(sinks) + payload_area, poison, sinks.host_type);
            }
        }
        if (dram_passive) {
            detail::WriteToBuffer(*dram_buf->get_device_buffer(distributed::MeshCoordinate(0, 0)), dram_image);
        }

        t_write_ += lap();

        // 3. One kernel on every active core, with its own peer list (kind, a, b, remote_base, slot).
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
        std::vector<std::vector<uint32_t>> peer_args(active.cores.size());
        for (const Pair& p : pairs) {
            const uint32_t a = pull ? p.dst : p.src;  // index into the active line
            const uint32_t q = pull ? p.src : p.dst;  // index into the passive line
            // PUSH: the peer's slot region. PULL: the peer's payload.
            const uint32_t remote_base = region_base(passive) + (pull ? 0 : payload_area);
            if (passive.is_dram()) {
                peer_args[a].insert(peer_args[a].end(), {1u, q, 0u, remote_base, p.slot});
            } else {
                const CoreCoord v = device->virtual_core_from_logical_core(passive.cores[q], passive.host_type);
                peer_args[a].insert(
                    peer_args[a].end(),
                    {0u, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), remote_base, p.slot});
            }
        }
        for (uint32_t i = 0; i < active.cores.size(); i++) {
            // PUSH: my payload. PULL: my slot region.
            const uint32_t local = region_base(active) + (pull ? payload_area : 0);
            std::vector<uint32_t> rt = {
                pull ? 1u : 0u,
                local,
                transfer,
                cfg_.chunk_bytes,
                cfg_.num_iters,
                static_cast<uint32_t>(peer_args[i].size() / 5)};
            rt.insert(rt.end(), peer_args[i].begin(), peer_args[i].end());
            SetRuntimeArgs(program, kernel, active.cores[i], rt);
        }
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        RunProgram(mesh_device, workload);
        t_run_ += lap();

        // 4. Read the slots of every sink back and compare each pair's slot.
        std::vector<std::vector<uint32_t>> readback(sinks.cores.size());
        if (sinks.is_dram()) {
            detail::ReadFromBuffer(
                *dram_buf->get_device_buffer(distributed::MeshCoordinate(0, 0)),
                reinterpret_cast<uint8_t*>(dram_image.data()));
            for (uint32_t d = 0; d < sinks.cores.size(); d++) {
                const auto first = dram_image.begin() + d * page_words + payload_area / 4;
                readback[d].assign(first, first + slots_bytes / 4);
            }
        } else {
            for (uint32_t d = 0; d < sinks.cores.size(); d++) {
                detail::ReadFromDeviceL1(
                    device,
                    sinks.cores[d],
                    region_base(sinks) + payload_area,
                    slots_bytes,
                    readback[d],
                    sinks.host_type);
            }
        }
        uint32_t bad_pairs = 0;
        uint32_t reported = 0;
        for (const Pair& p : pairs) {
            pairs_checked_++;
            PairResult r = compare_slot(
                sources.cores[p.src], sinks.cores[p.dst], readback[p.dst].data() + p.slot * words, payloads[p.src]);
            if (r.bad_words == 0) {
                continue;
            }
            bad_pairs++;
            const std::string msg = fmt::format(
                "[dev {}] {} {} -> {} {} over noc{} ({}, {}, {} B): {}",
                dev_id,
                sources.name,
                core_desc(device, sources, p.src, rn.noc),
                sinks.name,
                core_desc(device, sinks, p.dst, rn.noc),
                static_cast<int>(rn.noc),
                pull ? "PULL/read" : "PUSH/write",
                active.is_eth() ? "erisc kernel" : "tensix kernel",
                transfer,
                pair_signature(r));
            suite_failures_.push_back(msg);
            if (reported++ < cfg_.max_report) {
                log_error(tt::LogTest, "{}", msg);
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
                pairs.size(),
                transfer,
                bad_pairs);
        }
        return bad_pairs;
    }

    // Full sweep: every active line against every passive line, itself included, every NOC combo.
    void run_sweep(
        const std::string& sweep_name,
        const std::function<std::vector<Line>(IDevice*, const SweepConfig&)>& active_lines_fn,
        const std::function<std::vector<Line>(IDevice*, const SweepConfig&)>& passive_lines_fn,
        bool pull) {
        auto devices = selected_devices();
        ASSERT_FALSE(devices.empty()) << "no devices selected";
        uint32_t total_bad = 0;
        uint32_t steps = 0;
        for (const auto& mesh_device : devices) {
            IDevice* device = mesh_device->get_device(0, 0);
            auto active_lines = active_lines_fn(device, cfg_);
            auto passive_lines = passive_lines_fn(device, cfg_);
            if (reported_skips_.insert(device->id()).second) {
                std::string skipped;
                for (const auto& [chan, logical] : skipped_eth_cores(device, cfg_)) {
                    skipped += fmt::format(" ch{}=log({},{})", chan, logical.x, logical.y);
                }
                if (!skipped.empty()) {
                    log_info(
                        tt::LogTest,
                        "[dev {}] Ethernet channels left out of every Ethernet line (TT_NOC_SWEEP_SKIP_ETH_CHANS):{}",
                        device->id(),
                        skipped);
                }
            }
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
                uint32_t transfer;
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
                }
            }
            // Every (active, passive, risc/noc) combination is run once per transfer size, smallest size first so
            // the whole chip gets a quick first pass before the larger packets.
            for (uint32_t transfer : cfg_.transfer_sizes) {
                for (const auto& active : active_lines) {
                    if (!can_run_kernel(active, slow_dispatch_, arch_)) {
                        continue;
                    }
                    for (const auto& passive : passive_lines) {
                        for (const RiscNoc& rn : risc_noc_combos(active, cfg_.nocs)) {
                            plan.push_back({&active, &passive, rn, transfer});
                        }
                    }
                }
            }
            log_info(
                tt::LogTest,
                "[dev {}] sweep '{}': {} active x {} passive lines x {} sizes, {} steps, grid {}x{}, harvesting mask "
                "0x{:x}",
                device->id(),
                sweep_name,
                active_lines.size(),
                passive_lines.size(),
                cfg_.transfer_sizes.size(),
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
                dev_bad += run_step(mesh_device, *st.active, *st.passive, st.rn, pull, st.transfer, i);
                const auto now = std::chrono::steady_clock::now();
                if (!cfg_.verbose && i + 1 < plan.size() && now - last_report >= kProgressInterval) {
                    log_info(
                        tt::LogTest,
                        "[dev {}] sweep '{}': step {}/{} ({} {} {} noc{} {} B), {} bad so far, {:.1f} s elapsed",
                        device->id(),
                        sweep_name,
                        i + 1,
                        plan.size(),
                        st.active->name,
                        pull ? "<-" : "->",
                        st.passive->name,
                        static_cast<int>(st.rn.noc),
                        st.transfer,
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

// Every tensix row writes within itself (every core to every other core of the row) and into every other row
// (each core to the core in the same column), so each step carries either row-only or column-only traffic.
TEST_F(NocLineSweepFixture, TensixRowMajorPush) { run_sweep("tensix row-major push", tensix_rows, tensix_rows, false); }

// Same for columns: within itself, and into every other column one row at a time.
TEST_F(NocLineSweepFixture, TensixColMajorPush) { run_sweep("tensix col-major push", tensix_cols, tensix_cols, false); }

// Same sweeps with NOC reads instead of writes (exercises the read request / read response path).
TEST_F(NocLineSweepFixture, TensixRowMajorPull) { run_sweep("tensix row-major pull", tensix_rows, tensix_rows, true); }
TEST_F(NocLineSweepFixture, TensixColMajorPull) { run_sweep("tensix col-major pull", tensix_cols, tensix_cols, true); }

// ---------------------------------------------------------------------------------------------------------------
// Ethernet <-> tensix. This is the path the fabric router uses: an ERISC kernel writing into tensix L1.

// Every Ethernet core (active and idle groups) writes into every tensix row and every tensix column.
TEST_F(NocLineSweepFixture, EthToTensixPush) {
    run_sweep("eth -> tensix push", eth_lines, tensix_rows_and_cols, false);
}

// Every Ethernet core reads from every tensix row and column (the inbound direction into the Ethernet core's L1).
TEST_F(NocLineSweepFixture, EthFromTensixPull) {
    run_sweep("eth <- tensix pull", eth_lines, tensix_rows_and_cols, true);
}

// Every tensix row and column writes into the Ethernet cores. Like every Ethernet test this needs slow dispatch:
// under fast dispatch the Ethernet lines are dropped on both sides, since dispatch kernels may own the region.
TEST_F(NocLineSweepFixture, TensixToEthPush) {
    run_sweep("tensix -> eth push", tensix_rows_and_cols, eth_lines, false);
}

// Every Ethernet core writes into every other Ethernet core, within its own group and across groups.
TEST_F(NocLineSweepFixture, EthToEthPush) { run_sweep("eth -> eth push", eth_lines, eth_lines, false); }

// ---------------------------------------------------------------------------------------------------------------
// Tensix <-> DRAM. Every bank is reached through its preferred endpoint for the NOC in use, the endpoint an
// interleaved-buffer access from a kernel takes. DRAM never runs a kernel, so it is only ever the passive line.

// Every tensix row and column writes into every DRAM bank.
TEST_F(NocLineSweepFixture, TensixToDramPush) {
    run_sweep("tensix -> dram push", tensix_rows_and_cols, dram_line, false);
}

// Every tensix row and column reads from every DRAM bank.
TEST_F(NocLineSweepFixture, TensixFromDramPull) {
    run_sweep("tensix <- dram pull", tensix_rows_and_cols, dram_line, true);
}

}  // namespace
}  // namespace tt::tt_metal
