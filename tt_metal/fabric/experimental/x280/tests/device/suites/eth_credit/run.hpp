// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/eth_credit/run.hpp -- the host sequence for the eth_credit benchmark
// across one direct Ethernet link between two chips.
//
// One run: find a live link, check both tiles' Ethernet queues are idle, lay
// out and zero every peer-written region, launch E0 on both tiles and E1
// where a direction arrives through Metal, launch both chips' forwarders
// through the X280 runtime, wait on all of them, and harvest every result
// block. The stage selects the bring-up modes E0 and E1 are built with.

#ifndef X280_SUITES_ETH_CREDIT_RUN_HPP
#define X280_SUITES_ETH_CREDIT_RUN_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "tests/device/suites/eth_credit/regions.hpp"
#include "harness/regions/l1_window.hpp"
#include "harness/regions/metal_l1_region.hpp"
#include "harness/regions/regions.hpp"
#include "harness/session/session.hpp"
#include "harness/metal_context.hpp"
#include "tests/device/suites/eth_credit/records.hpp"

#include "eth_credit_abi.h"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

#include "internal/tt-1xx/blackhole/dev_mem_map.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace x280::suites::eth_credit {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hmeas = x280::harness::measurement;
namespace hout = x280::harness::output;
namespace hreg = x280::harness::regions;
namespace ec = x280::harness::regions::eth_credit;

/* Physical chips of the pair: direct Ethernet neighbours on this host
 * (Cluster::get_ethernet_connections, chip 0 channel 8 <-> chip 1 channel 5). */
inline const std::vector<int> kChipPair = {0, 1};

// The NoCs E0 and E1 issue on; the X280 forwarders use output::kX280Noc.
constexpr ttm::NOC kE0Noc = ttm::NOC::NOC_0;
constexpr ttm::NOC kE1Noc = ttm::NOC::NOC_1;

/* Poll iterations, not ticks, that each E0/E1 barrier wait may take before
 * giving up (core_sync_transport_metal.hpp, iteration_cap). */
constexpr uint64_t kBarrierPollCap = 1000000000ull;

constexpr uint32_t kTxqBase = 0xFFB90000u;
constexpr uint32_t kRxqBase = 0xFFB94000u;
constexpr uint32_t kQueueStride = 0x1000u;
constexpr uint32_t kTxqStatus = 0x08u;
constexpr uint32_t kTxqMaxPktSize = 0x0Cu;
constexpr uint32_t kTxqCmdOngoingBit = 16u;
constexpr uint32_t kRxqOutstandingWrCnt = 0x50u;

/** The bring-up modes E0 and E1 are built with; all off is the benchmark. */
struct Stage {
    bool e0_sink_local = false;
    bool e0_source_self = false;
    bool e1_sink_local = false;
    bool tail_marker = false;
    uint32_t counter_first_delay = 0;
    /** Launch E1 on the receiving tile; false leaves the link's far end idle. */
    bool launch_e1 = true;
    /** Launch E0 on chip B; false withholds one side of the handshake. */
    bool launch_e0_b = true;
    /** Misalign chip A's E0 peer_rx_base, an argument E0 must refuse. */
    bool corrupt_e0_peer_rx = false;
    /** Nonzero: an EC_JIT_SALT define no kernel reads, which gives E0 and E1
     * a JIT cache key no earlier build had, so this launch compiles them. */
    uint32_t jit_salt = 0;
    /** Dirty this chip's ERISC control area (or LIM arena) between its zero
     * write and its read-back, a setup failure run_case must stop at; -1 is
     * none. */
    int dirty_erisc_chip = -1;
    int dirty_lim_chip = -1;
};

/** One run's request. */
struct Case {
    const char* name = "";
    ec_shape_t shape{1u, 8u, 8u, EC_HEADER_BYTES, 1u};
    ec::Direction direction = ec::Direction::AToB;
    std::array<uint32_t, EC_MAX_P> packets{};
    Stage stage{};
    uint64_t stall_budget_ms = 2000;
    /** Whether both X280 sessions are expected to reach ResultReady. The host
     * X280 wait is derived from stall_budget_ms (ec::admit's host wait floor). */
    bool expect_x280_ready = true;
};

/* The x280 tree the E0/E1 sources are compiled from: the tree that holds
 * `kernel_dir` (tests/device/kernels), or X280_ETH_CREDIT_KERNEL_ROOT, which a
 * mutation check points at an edited copy so the tree itself is never
 * changed. */
inline std::filesystem::path kernel_root(const std::filesystem::path& kernel_dir) {
    const std::string override_root = hart_::env_str("X280_ETH_CREDIT_KERNEL_ROOT");
    if (!override_root.empty()) {
        return override_root;
    }
    return hart_::x280_root_of(kernel_dir);
}

/** One kernel's harvested result block, as the kernel wrote it. */
struct KernelResult {
    bool launched = false;
    ec_result_t block{};
};

struct ChipResult {
    int chip = -1;
    ttm::CoreCoord eth_logical{};
    ttm::CoreCoord eth_translated{};
    /** The Ethernet tile's NoC0 coordinate. */
    std::array<uint32_t, 2> eth_noc0{};
    x280::TileId tile{};
    KernelResult e0;
    KernelResult e1;
    std::array<KernelResult, EC_MAX_P> fwd{};
    hexec::X280Outcome x280{};
    uint32_t txq_max_pkt_bytes = 0;
    /** The base firmware's link retrain count on this ERISC, read before
     * launch and after Finish; a difference means the link retrained during
     * the run. */
    uint32_t retrain_before = 0;
    uint32_t retrain_after = 0;
    /** CMFW's programmed AICLK for this chip, in Hz. */
    uint64_t aiclk_hz = 0;
    /** The ERISC wall clock sampled around the run and assessed against
     * aiclk_hz; conversion_hz is nonzero only when that rate held. */
    hmeas::AiclkRunEvidence erisc_clock{};
};

/** cr's Ethernet tile as its records place it, issuing on `noc` when given. */
inline hout::CorePlacement erisc_placement(const ChipResult& cr, std::optional<uint32_t> noc) {
    return {
        .logical =
            std::array<uint32_t, 2>{static_cast<uint32_t>(cr.eth_logical.x), static_cast<uint32_t>(cr.eth_logical.y)},
        .virt =
            std::array<uint32_t, 2>{
                static_cast<uint32_t>(cr.eth_translated.x), static_cast<uint32_t>(cr.eth_translated.y)},
        .noc0 = cr.eth_noc0,
        .noc = noc,
    };
}

/** cr's X280 tile as its records place it. */
inline hout::CorePlacement x280_placement(const ChipResult& cr) {
    return {.noc0 = std::array<uint32_t, 2>{cr.tile.noc_x, cr.tile.noc_y}, .noc = hout::kX280Noc};
}

/** The Metal call that threw during a run, if any. */
enum class MetalFailure : uint32_t { None, Enqueue, Finish };

inline const char* metal_failure_name(MetalFailure m) {
    switch (m) {
        case MetalFailure::None: return "none";
        case MetalFailure::Enqueue: return "enqueue";
        case MetalFailure::Finish: return "finish";
    }
    return "?";
}

/**
 * The host's attribution of a run (SPEC §Failure and termination). No kernel
 * tells its neighbours it failed, so they exit on their own stall budgets: the
 * origin is every kernel whose error is not STALL_TIMEOUT, and without one
 * the host reports the timed-out kernels' stall sites. A launched kernel that
 * wrote no result is listed as missing, the prime suspect.
 */
struct Attribution {
    std::vector<Culprit> origins;
    std::vector<Culprit> stalled;
    std::vector<Culprit> missing;

    bool failed() const { return !origins.empty() || !stalled.empty() || !missing.empty(); }

    /** One log line: the origins, else the stall sites, plus any missing kernel. */
    std::string describe() const {
        std::ostringstream o;
        const auto name = [&o](const Culprit& k) {
            const uint32_t role = ec_kernel_role(k.kernel_id);
            o << "chip " << k.chip << " "
              << (role == EC_ROLE_E0   ? "E0"
                  : role == EC_ROLE_E1 ? "E1"
                                       : "FWD")
              << (role == EC_ROLE_X280_FWD ? std::to_string(ec_kernel_index(k.kernel_id)) : "");
        };
        if (!failed()) {
            return "no failure";
        }
        if (!origins.empty()) {
            o << "origin";
            for (const Culprit& k : origins) {
                o << " [";
                name(k);
                o << " " << error_name(k.code) << " detail " << k.detail << "]";
            }
        } else if (!stalled.empty()) {
            o << "no origin; stalled";
            for (const Culprit& k : stalled) {
                o << " [";
                name(k);
                o << " at site " << k.detail << "]";
            }
        }
        if (!missing.empty()) {
            o << (origins.empty() && stalled.empty() ? "" : ";") << " no result from";
            for (const Culprit& k : missing) {
                o << " [";
                name(k);
                o << "]";
            }
        }
        return o.str();
    }
};

/**
 * Why a finished run may have left TT-link traffic in flight. Queue readiness
 * cannot rule that out: a TXQ keeps resending unacknowledged packets after
 * CMD_ONGOING clears (EthernetTxRx.md, ETH_TXQ_CMD and TX Queues), and no
 * Blackhole register counts them. A planned negative control, whose kernels
 * all end with a result and whose X280 deadline the case expected, is None.
 */
enum class Undrained : uint32_t {
    None,
    /** A Metal call threw, so kernels may still be running. */
    MetalFailure,
    /** An X280 wait expired where the case expected ResultReady. */
    UnplannedDeadline,
    /** A launched E0 or E1 left no result, or one still RUNNING, after Finish. */
    KernelUnfinished,
    /** A kernel ended LINK_DOWN, or the link retrained during the run. */
    LinkDisturbed,
};

inline const char* undrained_name(Undrained u) {
    switch (u) {
        case Undrained::None: return "none";
        case Undrained::MetalFailure: return "metal_failure";
        case Undrained::UnplannedDeadline: return "unplanned_deadline";
        case Undrained::KernelUnfinished: return "kernel_unfinished";
        case Undrained::LinkDisturbed: return "link_disturbed";
    }
    return "?";
}

struct RunResult {
    bool launched = false;
    /** Set when this run may have left traffic in flight; see Undrained. */
    Undrained undrained = Undrained::None;
    /** The host's reading of the run's failure, if any; see Attribution. */
    Attribution attribution{};
    /** EnqueueMeshWorkload was called. */
    bool metal_enqueued = false;
    MetalFailure metal_failure = MetalFailure::None;
    ec::Admission admission{};
    std::array<ChipResult, ec::kChips> chips{};
};

/** A live link from chip `a` to chip `b` and the core at each end. */
struct Link {
    ttm::CoreCoord a_logical{};
    ttm::CoreCoord b_logical{};
};

/**
 * Finds an active, non-tunnel Ethernet core on `a` whose link is up and whose
 * far end is on `b`, and checks the reverse mapping names it back.
 */
inline std::optional<Link> find_link(ttm::IDevice& a, ttm::IDevice& b) {
    auto& cluster = x280::harness::cluster();
    for (const auto& core : a.get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true)) {
        if (!cluster.is_ethernet_link_up(a.id(), core)) {
            continue;
        }
        const auto [rchip, rcore] = cluster.get_connected_ethernet_core({a.id(), core});
        if (rchip != b.id() || !cluster.is_ethernet_link_up(b.id(), rcore)) {
            continue;
        }
        const auto [back_chip, back_core] = cluster.get_connected_ethernet_core({b.id(), rcore});
        if (back_chip == a.id() && back_core == core) {
            return Link{core, rcore};
        }
    }
    return std::nullopt;
}

/* Host sequence step 4: a TXQ still accepting a command or an RXQ with
 * writes in flight belongs to an aborted run whose resends could land after
 * this run's reset. Adds a gtest failure for each busy queue and returns
 * whether all were idle; the caller must not launch otherwise. */
inline bool queues_idle(int chip, const ttm::CoreCoord& eth, uint32_t* txq_max_pkt) {
    auto& cluster = x280::harness::cluster();
    const tt_cxy_pair core(chip, eth);
    bool idle = true;
    for (uint32_t q = 0; q < 2u; ++q) {
        uint32_t status = 0xFFFFFFFFu;
        uint32_t outstanding = 0xFFFFFFFFu;
        cluster.read_reg(&status, core, kTxqBase + q * kQueueStride + kTxqStatus);
        cluster.read_reg(&outstanding, core, kRxqBase + q * kQueueStride + kRxqOutstandingWrCnt);
        if (((status >> kTxqCmdOngoingBit) & 1u) != 0u) {
            ADD_FAILURE() << "chip " << chip << " TXQ " << q << " ETH_TXQ_STATUS 0x" << std::hex << status
                          << ": a command is still being accepted";
            idle = false;
        }
        if (outstanding != 0u) {
            ADD_FAILURE() << "chip " << chip << " RXQ " << q << " ETH_RXQ_OUTSTANDING_WR_CNT " << outstanding;
            idle = false;
        }
    }
    cluster.read_reg(txq_max_pkt, core, kTxqBase + kTxqMaxPktSize);
    return idle;
}

/** The defines E0 and E1 are JIT-built with: the shape and the stage's modes. */
inline std::map<std::string, std::string> kernel_defines(const ec_shape_t& s, const Stage& st) {
    std::map<std::string, std::string> d = {
        {"EC_P", std::to_string(s.P)},
        {"EC_N_LINK", std::to_string(s.n_link)},
        {"EC_N_DELIVER", std::to_string(s.n_deliver)},
        {"EC_SLOT_SIZE", std::to_string(s.slot_size)},
        {"EC_BATCH", std::to_string(s.batch)},
        {"EC_E0_SINK_LOCAL", st.e0_sink_local ? "1" : "0"},
        {"EC_E0_SOURCE_SELF", st.e0_source_self ? "1" : "0"},
        {"EC_E1_SINK_LOCAL", st.e1_sink_local ? "1" : "0"},
        {"EC_TAIL_MARKER", st.tail_marker ? "1" : "0"},
        {"EC_E0_COUNTER_FIRST_DELAY", std::to_string(st.counter_first_delay)},
    };
    if (st.jit_salt != 0u) {
        d.emplace("EC_JIT_SALT", std::to_string(st.jit_salt));
    }
    return d;
}

/* Writes zeros over [addr, addr + bytes) of an ERISC's L1 and reads them back.
 * `dirty` writes one nonzero byte in between, for the setup-failure test.
 * Adds a gtest failure and returns false when the read-back is not all zero;
 * the caller must not launch. */
inline bool zero_erisc(int chip, const ttm::CoreCoord& eth, uint32_t addr, uint32_t bytes, bool dirty) {
    auto& cluster = x280::harness::cluster();
    const tt_cxy_pair core(chip, eth);
    const std::vector<uint8_t> zeros(bytes, 0u);
    cluster.write_core(zeros.data(), bytes, core, addr);
    if (dirty) {
        const uint8_t junk = 0xA5u;
        cluster.write_core(&junk, 1u, core, addr);
    }
    std::vector<uint8_t> back(bytes, 0xFFu);
    cluster.read_core(back.data(), bytes, core, addr);
    if (back != zeros) {
        ADD_FAILURE() << "chip " << chip << " ERISC L1 [0x" << std::hex << addr << ", +0x" << bytes
                      << ") did not read back zero";
        return false;
    }
    return true;
}

/* The base firmware's link retrain count, its RETRAIN_COUNT mailbox word
 * (eth_live_status_t::retrain_count), read as llrt.cpp's link dump reads it.
 * The HAL's RETRAIN_COUNT L1 address names the same word (0x7CC00 + 128 +
 * 256 + 128 = MEM_RETRAIN_COUNT_ADDR, dev_mem_map.h), which run_case checks. */
inline uint32_t read_retrain_count(int chip, const ttm::CoreCoord& eth) {
    const auto& hal = ttm::MetalContext::instance().hal();
    uint32_t v = 0;
    x280::harness::cluster().read_reg(
        &v, tt_cxy_pair(chip, eth), hal.get_eth_fw_mailbox_val(ttm::FWMailboxMsg::RETRAIN_COUNT));
    return v;
}

inline ec_result_t read_erisc_result(int chip, const ttm::CoreCoord& eth, uint32_t addr) {
    ec_result_t r{};
    x280::harness::cluster().read_core(&r, sizeof(r), tt_cxy_pair(chip, eth), addr);
    return r;
}

/** Sorts every launched kernel of `r` into origins, stalled and missing (see Attribution). */
inline Attribution attribute(const Case& c, const RunResult& r) {
    Attribution a;
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        const ChipResult& cr = r.chips[i];
        const auto add = [&](const KernelResult& k, uint32_t kernel_id) {
            if (!k.launched) {
                return;
            }
            const ec_result_t& b = k.block;
            const Culprit culprit{cr.chip, kernel_id, b.status.code, b.status.detail};
            if (b.magic != EC_RESULT_MAGIC) {
                a.missing.push_back(culprit);
            } else if (b.status.state == EC_STATE_FAILED) {
                (b.status.code == EC_ERROR_STALL_TIMEOUT ? a.stalled : a.origins).push_back(culprit);
            }
        };
        add(cr.e0, EC_KERNEL_ID(EC_ROLE_E0, 0u, i));
        add(cr.e1, EC_KERNEL_ID(EC_ROLE_E1, 0u, i));
        for (uint32_t p = 0; p < c.shape.P; ++p) {
            add(cr.fwd[p], EC_KERNEL_ID(EC_ROLE_X280_FWD, p, i));
        }
    }
    return a;
}

/* The first run in this process that ended undrained, and its case name.
 * Once set, no later run_case launches until the chips are reset (docs/benchmarks/eth_credit.md,
 * "Running it"). */
inline Undrained g_undrained = Undrained::None;
inline std::string g_undrained_case;

/** Test-only: clears the undrained record after a case that made it on
 * purpose, with no traffic actually left in flight. */
inline void forget_undrained_run() {
    g_undrained = Undrained::None;
    g_undrained_case.clear();
}

/* Classifies a finished run; see Undrained. */
inline Undrained classify_drain(const Case& c, const RunResult& r) {
    if (r.metal_failure != MetalFailure::None) {
        return Undrained::MetalFailure;
    }
    for (const ChipResult& cr : r.chips) {
        if (c.expect_x280_ready && cr.x280.enqueued() && cr.x280.wait.outcome == x280::WaitOutcome::TimedOut) {
            return Undrained::UnplannedDeadline;
        }
    }
    for (const ChipResult& cr : r.chips) {
        for (const KernelResult* k : {&cr.e0, &cr.e1}) {
            if (k->launched && (k->block.magic != EC_RESULT_MAGIC || k->block.status.state == EC_STATE_RUNNING)) {
                return Undrained::KernelUnfinished;
            }
        }
    }
    for (const ChipResult& cr : r.chips) {
        if (cr.retrain_after != cr.retrain_before) {
            return Undrained::LinkDisturbed;
        }
        for (const KernelResult* k : {&cr.e0, &cr.e1}) {
            if (k->launched && k->block.status.code == EC_ERROR_LINK_DOWN) {
                return Undrained::LinkDisturbed;
            }
        }
    }
    return Undrained::None;
}

/**
 * Writes one `eth_credit_kernel` JSONL record for a launched kernel's result
 * block: its identity and terminal status, packets per p, every kept stall
 * and cost site, its own-clock window, and failure detail. A kernel that was
 * not launched writes nothing; one that left no result writes its magic.
 */
inline void write_kernel_jsonl(
    const x280::harness::RunSpec& spec, const Case& c, int chip, const char* role, const KernelResult& k) {
    if (!k.launched) {
        return;
    }
    hout::write_jsonl(spec.suite_label, hout::record(kernel_record(c.name, chip, role, k.block)));
}

/**
 * Runs one case end to end on the fixture's two-chip mesh and harvests every
 * participant. Adds gtest failures for anything that is wrong regardless of
 * the case's expectation: a refused admission, a queue left busy, a region
 * that did not zero, or an X280 session that did not complete.
 */
inline RunResult run_case(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, const Case& c) {
    RunResult out;
    const auto& hal = ttm::MetalContext::instance().hal();
    if (g_undrained != Undrained::None) {
        ADD_FAILURE() << "refusing to launch " << c.name << ": run " << g_undrained_case
                      << " in this process ended undrained (" << undrained_name(g_undrained)
                      << "); reset both chips and re-prepare the X280s (docs/benchmarks/eth_credit.md, Running it)";
        return out;
    }

    std::array<ttm::IDevice*, ec::kChips> dev{};
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        dev[i] = f.mesh->get_device(ttmd::MeshCoordinate(0, i));
        if (dev[i] == nullptr) {
            ADD_FAILURE() << "mesh has no device at (0, " << i << ")";
            return out;
        }
        out.chips[i].chip = dev[i]->id();
    }
    const auto link = find_link(*dev[0], *dev[1]);
    if (!link) {
        ADD_FAILURE() << "no live Ethernet link between chip " << dev[0]->id() << " and chip " << dev[1]->id();
        return out;
    }
    out.chips[0].eth_logical = link->a_logical;
    out.chips[1].eth_logical = link->b_logical;

    ec::Config cfg{};
    cfg.shape = c.shape;
    cfg.direction = c.direction;
    cfg.packets = c.packets;
    cfg.stall_budget_ms = c.stall_budget_ms;
    cfg.run_nonce = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 1u;
    cfg.barrier_cap = kBarrierPollCap;

    std::array<x280::X280Tile*, ec::kChips> tiles{};
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        ChipResult& cr = out.chips[i];
        ec::ChipPlacement& pl = cfg.chips[i];
        // The X280's window reaches the Ethernet tile on NoC 0.
        pl.eth = x280::X280Runtime::virtual_core_from_logical_core(
            *f.mesh, ttmd::MeshCoordinate(0, i), cr.eth_logical, tt::CoreType::ETH, ttm::NOC::NOC_0);
        cr.eth_translated = ttm::CoreCoord{pl.eth->x(), pl.eth->y()};
        cr.eth_noc0 = hexec::noc0_at(x280::harness::cluster(), cr.chip, cr.eth_translated);
        cr.aiclk_hz = hexec::programmed_aiclk_hz(x280::harness::cluster(), *dev[i]);
        const x280::TileId* tile = f.tile_for_chip(cr.chip);
        if (tile == nullptr) {
            ADD_FAILURE() << "chip " << cr.chip << " has no session tile";
            return out;
        }
        cr.tile = *tile;
        tiles[i] = &f.runtime->tile(cr.tile);
        pl.x280_noc0 = f.runtime->virtual_core_from_tile(cr.tile, ttm::NOC::NOC_0);
        pl.x280_noc1 = f.runtime->virtual_core_from_tile(cr.tile, ttm::NOC::NOC_1);
        for (uint32_t p = 0; p < EC_MAX_P; ++p) {
            pl.stream_ids[p] = 1u + p;
        }
        if (!queues_idle(cr.chip, cr.eth_translated, &cr.txq_max_pkt_bytes)) {
            return out;
        }
    }

    const hreg::L1Window hal_eth_window = hreg::query_unreserved_window(hal, ttm::HalProgrammableCoreType::ACTIVE_ETH);
    /* E0/E1 refuse any address outside kBounds (metal_common.hpp), built from
     * the same dev_mem_map.h macros; a HAL window past them would be refused
     * on device instead of admitted here. */
    if (hal_eth_window.local.base < MEM_AERISC_MAP_END + MEM_ERISC_KERNEL_CONFIG_SIZE ||
        hal_eth_window.local.base + hal_eth_window.local.size > MEM_ERISC_MAX_SIZE) {
        ADD_FAILURE() << "HAL ACTIVE_ETH window [0x" << std::hex << hal_eth_window.local.base << ", +0x"
                      << hal_eth_window.local.size << ") lies outside the kernels' L1 bounds";
        return out;
    }
    const auto lim0 = tiles[0]->available_lim();
    const auto lim1 = tiles[1]->available_lim();
    /* One layout places both chips' arenas, so a differing base would put
     * chip 1's at chip 0's addresses. */
    if (lim0.begin != lim1.begin) {
        ADD_FAILURE() << "LIM window base differs: chip " << out.chips[0].chip << " 0x" << std::hex << lim0.begin
                      << ", chip " << out.chips[1].chip << " 0x" << lim1.begin;
        return out;
    }
    const ec::Capacity cap{
        /* compute_layout aligns the control area within it. */
        .eth = hal_eth_window.local,
        .lim = hreg::Window{.base = lim0.begin, .size = std::min(lim0.size(), lim1.size())},
        .x280_hz = hmeas::kX280MtimeHz,
        /* The faster chip's rate, so neither chip's budget is shorter than
         * stall_budget_ms. */
        .erisc_hz = std::max(out.chips[0].aiclk_hz, out.chips[1].aiclk_hz),
    };
    out.admission = ec::admit(cfg, cap);
    if (!out.admission.ok()) {
        ADD_FAILURE() << "admission refused: " << out.admission.describe();
        return out;
    }
    ec::Arguments args = ec::derive_arguments(cfg, out.admission);

    /* Bring-up modes move the source or the sink off the forwarders. */
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        for (uint32_t p = 0; p < c.shape.P; ++p) {
            if (c.stage.e0_source_self) {
                args.chips[i].fwd[p].send_count = 0u;
            }
            if (c.stage.e1_sink_local || c.stage.e0_sink_local || !c.stage.launch_e1) {
                args.chips[i].fwd[p].recv_count = 0u;
            }
            /* A withheld E1 is no barrier participant, so the director must
             * not wait for it. */
            if (!c.stage.launch_e1) {
                args.chips[i].fwd[p].flags &= ~EC_FWD_FLAG_INBOUND_ACTIVE;
            }
        }
    }

    if (c.stage.corrupt_e0_peer_rx) {
        args.chips[0].e0.peer_rx_base += 8u;
    }

    const auto root = kernel_root(spec.open.kernel_dir);
    const auto e0_src = root / "kernels" / "eth_credit" / "e0_kernel.cpp";
    const auto e1_src = root / "kernels" / "eth_credit" / "e1_kernel.cpp";
    const auto defines = kernel_defines(c.shape, c.stage);

    ttmd::MeshWorkload workload;
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        ChipResult& cr = out.chips[i];
        const ec::ChipLayout& l = out.admission.layout.chips[i];
        /* Host sequence step 6: peer-written storage reads back zero before
         * anything is enqueued, or the run stops here. */
        if (!zero_erisc(
                cr.chip,
                cr.eth_translated,
                l.eth_ctrl_base,
                l.eth.total_bytes,
                c.stage.dirty_erisc_chip == static_cast<int>(i))) {
            return out;
        }
        const hreg::Region arena{l.lim_base, hreg::round_up(l.x280.total_bytes, hreg::kCacheLineBytes)};
        hreg::zero_region(*tiles[i], arena);
        if (c.stage.dirty_lim_chip == static_cast<int>(i)) {
            hreg::write_args(*tiles[i], hreg::Region{arena.addr, hreg::kCacheLineBytes}, uint32_t{0xA5A5A5A5u});
        }
        const auto back = tiles[i]->read_available_lim(arena.addr, arena.bytes);
        if (!std::all_of(back.begin(), back.end(), [](uint8_t b) { return b == 0u; })) {
            ADD_FAILURE() << "chip " << cr.chip << " LIM arena did not read back zero";
            return out;
        }
        hreg::write_args(
            *tiles[i], hreg::Region{l.lim_base + l.x280.args_offset, l.x280.args_bytes}, args.chips[i].fwd);

        ttm::Program program = ttm::CreateProgram();
        const ttm::CoreRangeSet core(ttm::CoreRange(cr.eth_logical, cr.eth_logical));
        /* tt-metal fails creation of a dual-ERISC kernel in dedicated NoC mode unless
         * ERISC0 uses NoC 0 and ERISC1 NoC 1 (tt_metal/impl/host_api/tt_metal.cpp,
         * "In dual ERISC mode, ERISC0 in dedicated mode must use NOC0"); the
         * placement's x280_noc0 and x280_noc1 are resolved for those NoCs. */
        if (i == 0u || c.stage.launch_e0_b) {
            const auto e0 = ttm::CreateKernel(
                program,
                e0_src.string(),
                core,
                ttm::EthernetConfig{
                    .eth_mode = ttm::Eth::SENDER,
                    .noc = kE0Noc,
                    .processor = ttm::DataMovementProcessor::RISCV_0,
                    .defines = defines});
            ttm::SetRuntimeArgs(program, e0, core, ec::pack_runtime_args(args.chips[i].e0));
            cr.e0.launched = true;
        }
        if (args.chips[i].has_e1 && c.stage.launch_e1) {
            const auto e1 = ttm::CreateKernel(
                program,
                e1_src.string(),
                core,
                ttm::EthernetConfig{
                    .eth_mode = ttm::Eth::SENDER,
                    .noc = kE1Noc,
                    .processor = ttm::DataMovementProcessor::RISCV_1,
                    .defines = defines});
            ttm::SetRuntimeArgs(program, e1, core, ec::pack_runtime_args(args.chips[i].e1));
            cr.e1.launched = true;
        }
        /* Slow dispatch otherwise JIT-builds each chip's program inside its
         * own launch, so a build only chip B needs (its E1, in one direction)
         * delays chip B's E0 behind chip A's past the handshake budget. */
        program.impl().compile(dev[i]);
        workload.add_program(ttmd::MeshCoordinateRange(ttmd::MeshCoordinate(0, i)), std::move(program));
        for (uint32_t p = 0; p < c.shape.P; ++p) {
            cr.fwd[p].launched = true;
        }
    }

    std::array<hexec::X280Launch, ec::kChips> launches{};
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        const ec::ChipLayout& l = out.admission.layout.chips[i];
        launches[i] = hexec::X280Launch{
            .image_path = hart_::image_path(spec.open.image_env_var, spec.open.image_file),
            .run_value = (static_cast<uint64_t>(0xEC00u + i) << 32) | cfg.run_nonce,
            .args_addr = l.lim_base + l.x280.args_offset,
            .args_bytes = l.x280.args_bytes,
            .cleanup_timeout = std::chrono::seconds{10},
            .wait_timeout =
                std::chrono::ceil<std::chrono::seconds>(std::chrono::milliseconds(out.admission.host_wait_ms)),
        };
    }

    EXPECT_EQ(
        hal.get_eth_fw_mailbox_val(ttm::FWMailboxMsg::RETRAIN_COUNT),
        hal.get_dev_addr(ttm::HalProgrammableCoreType::ACTIVE_ETH, ttm::HalL1MemAddrType::RETRAIN_COUNT))
        << "the two RETRAIN_COUNT addresses no longer name one word";
    std::array<hmeas::AiclkEndpoint, ec::kChips> clock_start{};
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        clock_start[i] = hexec::read_aiclk_start(x280::harness::cluster(), *dev[i], out.chips[i].eth_translated);
        out.chips[i].retrain_before = read_retrain_count(out.chips[i].chip, out.chips[i].eth_translated);
    }
    /* A Metal throw still falls through to the harvest: the kernels' result
     * blocks in ERISC L1 and LIM outlive it and carry the attribution. */
    try {
        out.metal_enqueued = true;
        hexec::enqueue_tensix(*f.mesh, workload);
    } catch (const std::exception& e) {
        out.metal_failure = MetalFailure::Enqueue;
        ADD_FAILURE() << "EnqueueMeshWorkload threw: " << e.what();
        for (auto& cr : out.chips) {
            for (auto& fwd : cr.fwd) {
                fwd.launched = false;
            }
        }
    }
    for (uint32_t i = 0; i < ec::kChips && out.metal_failure == MetalFailure::None; ++i) {
        out.chips[i].x280 = hexec::enqueue_x280(f.handle_for(out.chips[i].tile), launches[i]);
    }
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        if (out.chips[i].x280.enqueued()) {
            hexec::wait_x280(f.handle_for(out.chips[i].tile), launches[i], out.chips[i].x280);
        }
    }
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        if (out.chips[i].x280.enqueued()) {
            hexec::cleanup_x280(f.handle_for(out.chips[i].tile), launches[i], out.chips[i].x280);
        }
    }
    if (out.metal_failure == MetalFailure::None) {
        try {
            hexec::finish_tensix(*f.mesh);
        } catch (const std::exception& e) {
            out.metal_failure = MetalFailure::Finish;
            ADD_FAILURE() << "Finish threw: " << e.what();
        }
    }
    out.launched = true;
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        ChipResult& cr = out.chips[i];
        cr.retrain_after = read_retrain_count(cr.chip, cr.eth_translated);
        const auto clock_end = hexec::read_aiclk_end(x280::harness::cluster(), *dev[i], cr.eth_translated);
        cr.erisc_clock = hmeas::assess_aiclk_run(cr.aiclk_hz, clock_start[i], clock_end);
    }

    for (uint32_t i = 0; i < ec::kChips; ++i) {
        ChipResult& cr = out.chips[i];
        const ec::ChipLayout& l = out.admission.layout.chips[i];
        EXPECT_TRUE(cr.x280.enqueued()) << "chip " << cr.chip << " X280 enqueue rejected";
        if (c.expect_x280_ready) {
            EXPECT_EQ(cr.x280.wait.outcome, x280::WaitOutcome::ResultReady)
                << "chip " << cr.chip << ": " << x280::describe(cr.x280.wait);
        }
        EXPECT_TRUE(cr.x280.cleaned_up()) << "chip " << cr.chip << ": " << x280::describe(cr.x280.cleanup);
        cr.e0.block = read_erisc_result(cr.chip, cr.eth_translated, l.eth_ctrl_base + l.eth.e0_result);
        if (cr.e1.launched) {
            cr.e1.block = read_erisc_result(cr.chip, cr.eth_translated, l.eth_ctrl_base + l.eth.e1_result);
        }
        for (uint32_t p = 0; p < c.shape.P; ++p) {
            cr.fwd[p].block =
                hreg::read_struct<ec_result_t>(*tiles[i], hreg::Region{l.fwd_area(p) + l.x280.result, EC_RESULT_BYTES});
            for (uint32_t w = 0; w < X280_TLB_ETH_CREDIT_PER_HART; ++w) {
                hexec::expect_window_cleared(*tiles[i], ec::fwd_tlb_base(p) + w);
            }
        }
    }
    out.attribution = attribute(c, out);
    if (out.attribution.failed()) {
        std::printf("[eth_credit %s] attribution: %s\n", c.name, out.attribution.describe().c_str());
    }
    hout::write_jsonl(
        spec.suite_label,
        hout::record(
            attribution_record(c.name, out.attribution.origins, out.attribution.stalled, out.attribution.missing)));
    out.undrained = classify_drain(c, out);
    if (out.undrained != Undrained::None) {
        ADD_FAILURE() << c.name << " ended undrained (" << undrained_name(out.undrained)
                      << "); later runs in this process are refused";
        g_undrained = out.undrained;
        g_undrained_case = c.name;
    }
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        const ChipResult& cr = out.chips[i];
        hout::write_jsonl(
            spec.suite_label,
            hout::record(link_record({
                .case_name = c.name,
                .chip = cr.chip,
                .erisc = erisc_placement(cr, std::nullopt),
                .retrain_count_before = cr.retrain_before,
                .retrain_count_after = cr.retrain_after,
                .metal_failure = metal_failure_name(out.metal_failure),
                .undrained = undrained_name(out.undrained),
                .e0_status = cr.e0.block.status,
                .l2cpu = static_cast<uint32_t>(cr.tile.hw_index),
                .txq0_max_pkt_bytes = cr.txq_max_pkt_bytes,
                .aiclk_hz = cr.aiclk_hz,
                .erisc_clock = cr.e0.launched || cr.e1.launched ? &cr.erisc_clock : nullptr,
            })));
        write_kernel_jsonl(spec, c, cr.chip, "E0", cr.e0);
        write_kernel_jsonl(spec, c, cr.chip, "E1", cr.e1);
        for (uint32_t p = 0; p < c.shape.P; ++p) {
            write_kernel_jsonl(spec, c, cr.chip, "FWD", cr.fwd[p]);
        }
    }
    return out;
}

/** The entry of `r` for `site`, or nullptr when the kernel does not keep it. */
inline const ec_stall_t* find_stall(const ec_result_t& r, uint32_t site) {
    for (const auto& st : r.stalls) {
        if (st.site == site) {
            return &st;
        }
    }
    return nullptr;
}

/** Ticks `r` spent at its stall sites, cost sites excluded: the figure the
 * "stalls least" comparison ranks kernels by. */
inline uint64_t stall_ticks(const ec_result_t& r) {
    uint64_t total = 0;
    for (const auto& st : r.stalls) {
        if (st.site != EC_SITE_NONE && !ec_site_is_cost(st.site)) {
            total += st.ticks;
        }
    }
    return total;
}

inline std::string describe(const char* who, const KernelResult& k) {
    std::ostringstream o;
    const ec_result_t& r = k.block;
    o << who << ": ";
    if (!k.launched) {
        o << "not launched";
        return o.str();
    }
    if (r.magic != EC_RESULT_MAGIC) {
        o << "no result (magic 0x" << std::hex << r.magic << std::dec << ")";
        return o.str();
    }
    o << "state " << r.status.state << " code " << error_name(r.status.code) << " detail " << r.status.detail
      << " origin 0x" << std::hex << r.status.origin << std::dec << " in/out";
    for (uint32_t p = 0; p < EC_MAX_P; ++p) {
        o << " " << r.packets_in[p] << "/" << r.packets_out[p];
    }
    o << " stalls";
    for (const auto& st : r.stalls) {
        if (st.site != EC_SITE_NONE) {
            o << " [" << (ec_site_is_cost(st.site) ? "cost" : "stall") << " site " << st.site << ": " << st.count
              << " x, " << st.ticks << " ticks]";
        }
    }
    o << " stall ticks " << stall_ticks(r);
    o << " ts " << r.first_ts << ".." << r.last_ts;
    if (r.status.code != EC_ERROR_NONE) {
        o << " failure{p " << r.failure.p << " exp " << r.failure.expected << " seen " << r.failure.seen << " site "
          << r.failure.site << " local " << r.failure.local_ptr << " shadow " << r.failure.shadow_ptr << "}";
    }
    return o.str();
}

/** The ERISC clock evidence in one line: the measured rate bounds, then either
 * the conversion rate or the reasons it was withheld. */
inline std::string describe_clock(const hmeas::AiclkRunEvidence& e) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(0) << "elapsed rate [" << e.elapsed_rate.lower_hz << ", "
      << e.elapsed_rate.upper_hz << "] Hz, ";
    if (e.conversion_hz != 0u) {
        o << "converts at " << e.conversion_hz << " Hz";
    } else {
        o << "no conversion:";
        for (const auto err : e.errors) {
            o << " " << hmeas::clock_error_name(err);
        }
    }
    return o.str();
}

/**
 * Prints a failed run's detail after run_case's attribution line: the shape,
 * then each chip's link, ERISC clock and kernel results. A clean run prints
 * nothing; the eth_credit_link and eth_credit_kernel records hold every run.
 */
inline void print_run(const Case& c, const RunResult& r) {
    if (!r.attribution.failed() && r.metal_failure == MetalFailure::None && r.undrained == Undrained::None) {
        return;
    }
    std::printf(
        "[eth_credit %s] P=%u N_LINK=%u N_DELIVER=%u SLOT=%u BATCH=%u, Metal failure %s\n",
        c.name,
        c.shape.P,
        c.shape.n_link,
        c.shape.n_deliver,
        c.shape.slot_size,
        c.shape.batch,
        metal_failure_name(r.metal_failure));
    for (const auto& cr : r.chips) {
        std::printf(
            "  chip %d: ERISC logical (%zu, %zu) translated (%zu, %zu), L2CPU %d, TXQ0 max pkt %u B\n",
            cr.chip,
            cr.eth_logical.x,
            cr.eth_logical.y,
            cr.eth_translated.x,
            cr.eth_translated.y,
            cr.tile.hw_index,
            cr.txq_max_pkt_bytes);
        std::printf("    retrain count %u -> %u\n", cr.retrain_before, cr.retrain_after);
        std::printf(
            "    AICLK %llu Hz; ERISC wall clock %s\n",
            static_cast<unsigned long long>(cr.aiclk_hz),
            describe_clock(cr.erisc_clock).c_str());
        std::printf("    %s\n", describe("E0", cr.e0).c_str());
        std::printf("    %s\n", describe("E1", cr.e1).c_str());
        for (uint32_t p = 0; p < c.shape.P; ++p) {
            const std::string who = "FWD" + std::to_string(p);
            std::printf("    %s\n", describe(who.c_str(), cr.fwd[p]).c_str());
        }
    }
}

/** Requires every launched kernel's result to be present, for its own kernel id. */
inline void expect_results_present(const RunResult& r) {
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        const ChipResult& cr = r.chips[i];
        const auto check = [&](const KernelResult& k, uint32_t role, uint32_t index, const char* who) {
            if (!k.launched) {
                return;
            }
            EXPECT_EQ(k.block.magic, EC_RESULT_MAGIC) << "chip " << cr.chip << " " << who << " wrote no result";
            EXPECT_EQ(k.block.kernel_id, EC_KERNEL_ID(role, index, i)) << "chip " << cr.chip << " " << who;
        };
        check(cr.e0, EC_ROLE_E0, 0u, "E0");
        check(cr.e1, EC_ROLE_E1, 0u, "E1");
        for (uint32_t p = 0; p < EC_MAX_P; ++p) {
            check(cr.fwd[p], EC_ROLE_X280_FWD, p, "forwarder");
        }
    }
}

/** Requires every launched kernel to have finished DONE with no error and valid stall totals. */
inline void expect_all_done(const RunResult& r) {
    expect_results_present(r);
    for (const auto& cr : r.chips) {
        const auto check = [&](const KernelResult& k, const char* who) {
            if (!k.launched) {
                return;
            }
            EXPECT_EQ(k.block.status.state, static_cast<uint32_t>(EC_STATE_DONE))
                << "chip " << cr.chip << " " << describe(who, k);
            EXPECT_EQ(k.block.status.code, static_cast<uint32_t>(EC_ERROR_NONE))
                << "chip " << cr.chip << " " << describe(who, k);
            for (const auto& st : k.block.stalls) {
                EXPECT_NE(st.ticks, EC_STALL_TICKS_INVALID)
                    << "chip " << cr.chip << " " << who << " site " << st.site << ": a stall interval was refused";
            }
        };
        check(cr.e0, "E0");
        check(cr.e1, "E1");
        for (uint32_t p = 0; p < EC_MAX_P; ++p) {
            check(cr.fwd[p], ("FWD" + std::to_string(p)).c_str());
        }
    }
}

/**
 * Per-hop count conservation for one direction, sender chip `s`: packets the
 * forwarders sent, E0 forwarded, E1 received and the forwarders consumed all
 * equal the requested count for every p. A hop the stage moved off the
 * forwarders is checked at the kernel that took it over.
 */
inline void expect_conserved(const Case& c, const RunResult& r, uint32_t s) {
    const ChipResult& tx = r.chips[s];
    const ChipResult& rx = r.chips[1u - s];
    for (uint32_t p = 0; p < c.shape.P; ++p) {
        const uint32_t n = c.packets[p];
        if (!c.stage.e0_source_self) {
            EXPECT_EQ(tx.fwd[p].block.packets_out[p], n) << "chip " << tx.chip << " forwarder " << p << " sent";
        }
        EXPECT_EQ(tx.e0.block.packets_out[p], n) << "chip " << tx.chip << " E0 p " << p;
        if (rx.e1.launched) {
            EXPECT_EQ(rx.e1.block.packets_in[p], n) << "chip " << rx.chip << " E1 p " << p;
            if (!c.stage.e1_sink_local) {
                EXPECT_EQ(rx.fwd[p].block.packets_in[p], n) << "chip " << rx.chip << " forwarder " << p << " consumed";
            }
        }
    }
}

/** What `st` leaves of the path between the two forwarders. */
inline Path measured_path(const Stage& st) {
    if (st.e1_sink_local) {
        return kEthernetHop;
    }
    if (st.e0_sink_local) {
        return kNocHop;
    }
    return kFullPath;
}

/**
 * Writes one JSONL record per p for a completed direction, of the path
 * measured_path() names: the receiving forwarders' packet counts over their
 * own mtime window, with `c.stage.e0_sink_local` the sending forwarders' sent
 * counts over theirs, or with `c.stage.e1_sink_local` E1's over its ERISC
 * wall-clock window (P must be 1, a fatal failure otherwise); both chips, both
 * ERISC coordinates, the X280 tile, the shape, the X280 PLL, and both chips'
 * ERISC clock evidence.
 *
 * The rate the tabulator derives is n - k1 packets over [first_ts, last_ts]:
 * the window's kernel stamps once per batch, after the batch completes, so the
 * window covers the packets after a first batch of k1. The record carries
 * `packets`, n, `first_batch_packets`, k1, and the window; the tabulator
 * derives the n - k1. `slot_bytes_per_request` is what crossed the
 * Ethernet link, timed at the consumer; the forwarder itself receives only
 * the EC_HEADER_BYTES header of each slot, the record's useful bytes.
 */
inline void record_throughput(
    const x280::harness::RunSpec& spec,
    const x280::harness::session::Fixture& f,
    const Case& c,
    const RunResult& r,
    uint32_t s) {
    const ChipResult& tx = r.chips[s];
    const ChipResult& rx = r.chips[1u - s];
    const Path path = measured_path(c.stage);
    const bool at_e1 = path.receiver == Receiver::E1;
    const bool at_sender = path.receiver == Receiver::Sender;
    ASSERT_TRUE(!at_e1 || c.shape.P == 1u) << "E1's receive window covers every p, so it reports only P = 1";
    for (uint32_t p = 0; p < c.shape.P; ++p) {
        const ec_result_t& b = at_e1 ? rx.e1.block : at_sender ? tx.fwd[p].block : rx.fwd[p].block;
        const ReceiveWindow window{
            .case_name = c.name,
            .p = p,
            .tx_chip = tx.chip,
            .rx_chip = rx.chip,
            .x280_hw_index = static_cast<uint32_t>((at_sender ? tx : rx).tile.hw_index),
            .tx_erisc = erisc_placement(tx, static_cast<uint32_t>(kE0Noc)),
            .rx_erisc = erisc_placement(rx, static_cast<uint32_t>(kE1Noc)),
            .tx_x280 = x280_placement(tx),
            .rx_x280 = x280_placement(rx),
            .x280_pll_mhz = f.pll_mhz,
            .shape = c.shape,
            .bidirectional = c.direction == ec::Direction::Both,
            .path = path,
        };
        hout::write_jsonl(spec.suite_label, hout::record(throughput_record(window, b, tx.erisc_clock, rx.erisc_clock)));
        std::printf(
            "  p %u: %u packets over %llu %s ticks\n",
            p,
            at_sender ? b.packets_out[p] : b.packets_in[p],
            static_cast<unsigned long long>(b.last_ts - b.first_ts),
            at_e1 ? "E1 wall" : "mtime");
    }
}

}  // namespace x280::suites::eth_credit

#endif  // X280_SUITES_ETH_CREDIT_RUN_HPP
