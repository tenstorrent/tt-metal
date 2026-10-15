// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/packet_injection/regions.hpp -- pure host admission, layout computation
// and ERISC writer selection for the packet injection characterization.
//
// Defines pure host types and validation functions without hardware dependencies,
// allowing host-only unit tests to run without silicon or device drivers.

#ifndef X280_SUITES_PACKET_INJECTION_REGIONS_HPP
#define X280_SUITES_PACKET_INJECTION_REGIONS_HPP

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "x280/device/x280_state.hpp"

#include "kernels/packet_injection/packet_injection_abi.h"
#include "rt/x280_noc_map.h"
#include "harness/artifacts/artifacts.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/release_group.hpp"

namespace x280::harness::regions::packet_injection {

/* The host deadline's floor: kHostTimeoutPhases phase budgets plus
 * kHostTimeoutSlackMs (measurement::host_wait_floor_ms). */
inline constexpr uint64_t kHostTimeoutPhases = 9;
inline constexpr uint64_t kHostTimeoutSlackMs = 5000;

/** LIM backing the packet arena, 14 reserved 128-KiB ways. */
inline constexpr uint64_t kRequiredLimBackedBytes = x280::kPreparedLimWays * x280::kLimWayBytes;

/** Producers per tile: BRISC alone, BRISC and NCRISC, or one idle ERISC per
 * Ethernet tile. */
enum class WorkerFamily : uint32_t { BriscOnly = 1u, Paired = 2u, Ethernet = 3u };

struct LogicalCoord {
    uint32_t x = 0;
    uint32_t y = 0;

    bool operator==(const LogicalCoord& other) const { return x == other.x && y == other.y; }
};

/**
 * Measured device state that admission checks a Config against.
 *
 * Every field must hold a real reading: admission rejects a zero clock or an
 * unusable L1 window rather than skipping the check. `producer_timer_hz` is
 * the frequency of the producer's native wall counter, which its per-phase
 * `timeout_ticks` budget is counted in (AICLK on Tensix), and `mtime_hz` is
 * measurement::kX280MtimeHz.
 *
 * `[l1_window_base, l1_span_limit)` is the producer target's core-local L1
 * window, in bytes: on Tensix the allocator's L1 base and `l1_size_per_core()`.
 * Admission requires a tile span to be no larger than the window; that the
 * allocated span ends inside it is `allocate_tile_l1_buffer()`'s check in
 * `harness/regions/tensix_l1.hpp`.
 */
struct Capacity {
    uint64_t lim_backed_bytes = 0;
    uint32_t l1_span_limit = 0;
    uint32_t l1_window_base = 0;
    uint64_t producer_timer_hz = 0;
    uint64_t mtime_hz = 0;
};

struct Config {
    uint32_t mode = PI_MODE_LATENCY;
    uint32_t include_creation = 0;
    WorkerFamily family = WorkerFamily::BriscOnly;
    uint32_t tile_count = 1;
    uint32_t S = 16;
    uint32_t B = 1;
    uint32_t C = 1;
    uint32_t K = 1;
    uint32_t inflight = 1;  // workers one consumer pass visits together (pi_x280_args_t::inflight)
    uint64_t W = 1000;
    uint64_t N = 4096;
    std::vector<LogicalCoord> ordered_tiles;
    uint16_t dst_chip = 0;
    uint16_t dst_mesh = 0;
    uint32_t poll_cadence = PI_DEFAULT_POLL_CADENCE;
    uint64_t phase_timeout_ms = 60000;
    uint64_t host_timeout_ms = 600000;
};

/**
 * Sets cfg.inflight from PACKET_INJECTION_INFLIGHT when that variable is set
 * and non-empty; throws std::invalid_argument for a value that is not a
 * decimal fitting uint32_t. Whether the value is admissible is checked at
 * admission (AdmissionCheck::Inflight).
 */
inline void read_inflight_env(Config& cfg) {
    const std::string v = x280::harness::artifacts::env_str("PACKET_INJECTION_INFLIGHT");
    if (!v.empty()) {
        cfg.inflight = static_cast<uint32_t>(
            x280::harness::artifacts::parse_u64_strict(v, "PACKET_INJECTION_INFLIGHT", UINT32_MAX));
    }
}

/**
 * Workers in a run: one per tile for BriscOnly and Ethernet, two for Paired,
 * and zero for any other family, which every caller then rejects.
 *
 * Returns 64-bit so an unvalidated `tile_count` cannot overflow before the
 * PI_MAX_WORKERS bound is applied.
 */
inline uint64_t worker_count(const Config& cfg) {
    const uint64_t tiles = static_cast<uint64_t>(cfg.tile_count);
    switch (cfg.family) {
        case WorkerFamily::BriscOnly:
        case WorkerFamily::Ethernet: return tiles;
        case WorkerFamily::Paired: return 2u * tiles;
    }
    return 0u;
}

/**
 * One run's layout, exactly as pi_compute_layout() derived it.
 *
 * `abi` carries every LIM and L1 offset and extent; this header reproduces none
 * of those formulas. In a Layout that compute_layout() produced, `abi` is
 * zeroed exactly when !ok(), and `check` then names the class of rejection with
 * the offending value and the bound it broke.
 *
 * Only ever read one that compute_layout() or validate_config() returned: a
 * default-constructed Layout is indistinguishable from a successful one,
 * because PI_LAYOUT_OK is zero, so "never computed" and "computed, no problem"
 * share a bit pattern that ok() cannot separate.
 */
struct Layout {
    pi_layout_t abi{};
    pi_layout_check_t check{};

    bool ok() const { return check.code == PI_LAYOUT_OK; }

    std::string describe() const {
        return std::string(pi_layout_code_name(check.code)) + ": value " + std::to_string(check.value) +
               " against bound " + std::to_string(check.bound);
    }
};

/**
 * How many exported latency samples belong to one worker, and the two values of
 * the check that rejected the count when one did.
 *
 * `observed` and `bound` are the compared values in the units of the failing
 * check, so a caller can assert or branch on them; describe() renders them only
 * when someone asks for text.
 */
struct SamplePrefix {
    uint64_t count = 0;
    pi_error_t error = PI_ERROR_NONE;
    uint32_t worker_index = 0;
    uint64_t observed = 0;
    uint64_t bound = 0;

    bool ok() const { return error == PI_ERROR_NONE; }

    std::string describe() const {
        const std::string w = "worker " + std::to_string(worker_index);
        switch (error) {
            case PI_ERROR_NONE: return w + " latency prefix " + std::to_string(count) + " samples";
            case PI_ERROR_COUNTER_DISTANCE:
                return w + " retired " + std::to_string(observed) + " latency samples but X280 stored only " +
                       std::to_string(bound);
            case PI_ERROR_PHASE_OVERRUN:
                return w + " serviced " + std::to_string(observed) + " requests, beyond its export share of " +
                       std::to_string(bound) + " samples";
            case PI_ERROR_CAPACITY:
                return w + " has no export share: the region holds " + std::to_string(bound) + " worker shares";
            default:
                return w + " latency prefix rejected with error " + std::to_string(static_cast<uint32_t>(error)) +
                       ": observed " + std::to_string(observed) + " against bound " + std::to_string(bound);
        }
    }
};

/**
 * Decides how many exported latency samples belong to `worker_index`.
 *
 * X280 writes a sample when it services a request and counts that in
 * pi_consumer_slot_t::phase_done, while the producer's `latency_count` counts
 * requests it has seen credited, which lags. The consumer count is therefore
 * the prefix length and the producer count is retirement evidence only: a run
 * that fails between the store and the credit leaves samples that exist and are
 * valid, and truncating by `latency_count` would discard them.
 *
 * Three checks, all against the layout so no extent is restated here. A
 * producer claiming more retirements than X280 serviced holds by construction --
 * it only retires on the credit X280 returns after servicing -- so that one is a
 * corruption check with no reachable trigger, kept for that reason. The
 * per-worker share and the region containment are the same bound the export was
 * sized with, which is why a share check against a caller-supplied N is not a
 * separate condition.
 *
 * @return `count` samples to read from the worker's slice of the export, or
 *         !ok() with `observed` and `bound` from the rejecting check. Outside
 *         PI_MODE_LATENCY the layout reserves no export and the count is zero.
 */
inline SamplePrefix latency_sample_prefix(
    const pi_consumer_slot_t& slot,
    const pi_producer_result_t& result,
    const pi_layout_t& layout,
    uint32_t worker_index) {
    SamplePrefix prefix{};
    prefix.worker_index = worker_index;

    const uint64_t share = layout.latency_sample_stride / sizeof(uint64_t);
    if (share == 0u) {
        return prefix;
    }
    if (result.latency_count > slot.phase_done) {
        prefix.error = PI_ERROR_COUNTER_DISTANCE;
        prefix.observed = result.latency_count;
        prefix.bound = slot.phase_done;
        return prefix;
    }
    if (slot.phase_done > share) {
        prefix.error = PI_ERROR_PHASE_OVERRUN;
        prefix.observed = slot.phase_done;
        prefix.bound = share;
        return prefix;
    }
    // Division, not worker_index * share: a product taken in 32 bits wraps, and
    // at a share of 4 an index of 1<<30 truncates to zero and reads as
    // in-bounds while addressing far outside the export region.
    const uint64_t shares = (layout.latency_export_bytes / sizeof(uint64_t)) / share;
    if (worker_index >= shares) {
        prefix.error = PI_ERROR_CAPACITY;
        prefix.observed = worker_index;
        prefix.bound = shares;
        return prefix;
    }

    prefix.count = slot.phase_done;
    return prefix;
}

/**
 * True when both of X280's aggregate intervals in `run_res`, `mtime` and
 * `rdcycle`, are non-zero and inside the counter half-range, which also
 * rejects an inverted or unwritten pair.
 */
inline bool aggregate_interval_valid(const pi_run_result_t& run_res) {
    return x280::harness::output::throughput_interval_valid(
               run_res.aggregate_start_mtime, run_res.aggregate_end_mtime) &&
           x280::harness::output::throughput_interval_valid(
               run_res.aggregate_start_rdcycle, run_res.aggregate_end_rdcycle);
}

// Throws, naming `record` and both values, unless worker count `P` and tile
// count `T` fit the ABI's 16-bit count fields.
inline void require_u16_counts(const char* record, uint32_t P, uint32_t T) {
    if (P > UINT16_MAX || T > UINT16_MAX) {
        throw std::runtime_error(
            std::string(record) + " count does not fit 16 bits: P " + std::to_string(P) + " T " + std::to_string(T));
    }
}

/**
 * The X280 consumer args for one admitted run. `P`/`T` are the admitted worker
 * and tile counts, `layout` the placed layout, `x280_timeout_ticks` the
 * admission's budget in mtime ticks, and `release` the release selector's
 * group for the tile table, or none for one unicast per tile. Throws, naming
 * the value, if `P` or `T` does not fit the ABI's 16-bit count field.
 */
inline pi_x280_args_t make_x280_args(
    const Config& cfg,
    uint32_t P,
    uint32_t T,
    const pi_layout_t& layout,
    uint32_t abi_tag,
    uint64_t x280_timeout_ticks,
    uint32_t nonce,
    const std::optional<ReleaseGroup>& release) {
    require_u16_counts("X280 args", P, T);
    pi_x280_args_t xargs{};
    xargs.abi_tag = abi_tag;
    xargs.worker_count = static_cast<uint16_t>(P);
    xargs.tile_count = static_cast<uint16_t>(T);
    xargs.S = cfg.S;
    xargs.B = cfg.B;
    xargs.C = cfg.C;
    xargs.K = cfg.K;
    xargs.inflight = cfg.inflight;
    xargs.poll_cadence = cfg.poll_cadence;
    xargs.l1_tile_base = layout.l1_tile_base;
    xargs.W = cfg.W;
    xargs.N = cfg.N;
    xargs.x280_timeout_ticks = x280_timeout_ticks;
    xargs.run_nonce = nonce;
    if (release) {
        xargs.release_num_dests = release->num_dests;
        xargs.release_rect = release->rect();
    }
    return xargs;
}

/**
 * Worker `worker_index`'s producer args for one admitted run. `P`, `T`,
 * `layout`, `abi_tag` and `nonce` are as for make_x280_args(); `peer` the X280
 * tile resolved for the producer's NoC; `timeout_ticks` the admission's per-phase
 * budget in the producer's native wall ticks. Throws, naming the value, if `P`
 * or `T` does not fit the ABI's 16-bit count field.
 */
inline pi_producer_args_t make_producer_args(
    const Config& cfg,
    uint32_t P,
    uint32_t T,
    const pi_layout_t& layout,
    uint32_t abi_tag,
    uint32_t nonce,
    uint32_t worker_index,
    const TranslatedCoord& peer,
    uint64_t timeout_ticks) {
    require_u16_counts("producer args", P, T);
    pi_producer_args_t args{};
    args.abi_tag = abi_tag;
    args.run_nonce = nonce;
    args.worker_index = worker_index;
    args.worker_count = static_cast<uint16_t>(P);
    args.tile_count = static_cast<uint16_t>(T);
    args.S = cfg.S;
    args.B = cfg.B;
    args.poll_cadence = cfg.poll_cadence;
    args.dst_chip = cfg.dst_chip;
    args.dst_mesh = cfg.dst_mesh;
    args.peer = peer.noc();
    args.l1_tile_base = layout.l1_tile_base;
    args.W = cfg.W;
    args.N = cfg.N;
    args.timeout_ticks = timeout_ticks;
    return args;
}

/**
 * Derives the layout of `cfg` with its Tensix L1 pages at `l1_tile_base`.
 *
 * @param l1_tile_base device-local L1 byte address of the per-tile pages, from
 *                     allocate_tile_l1_buffer(). Zero is valid for sizing
 *                     before that allocation exists: only `l1_tile_base`
 *                     itself and the 32-bit extent bound depend on it.
 * @return `check` as pi_compute_layout() reported it, so ok() and describe()
 *         answer whether and why. A worker count outside [1, PI_MAX_WORKERS] is
 *         rejected here rather than narrowed into the call, and reports the
 *         count against its limit.
 */
inline Layout compute_layout(const Config& cfg, uint32_t l1_tile_base = 0) {
    Layout out{};
    const uint64_t P = worker_count(cfg);
    if (P == 0u || P > PI_MAX_WORKERS) {
        out.check = pi_layout_check(PI_LAYOUT_BAD_PARAM, P, PI_MAX_WORKERS);
        return out;
    }
    out.check = pi_compute_layout(
        static_cast<uint32_t>(P), cfg.tile_count, cfg.S, cfg.B, cfg.N, cfg.mode, l1_tile_base, &out.abi);
    return out;
}

/**
 * The L2CPU TLB slots a packet_injection X280 consumer opens for a run of
 * `tile_count` tiles in `mode`, and so must leave cleared: one credit slot per
 * tile, and the timer slot in latency mode alone (the consumer's
 * ensure_timer_window). A slot outside this set is never written by the run.
 * Throws std::invalid_argument naming `tile_count` when it exceeds
 * X280_TLB_PACKET_INJECTION_CREDIT_COUNT, past which a credit slot would alias
 * the timer slot.
 */
inline std::vector<uint32_t> consumer_window_slots(uint32_t mode, uint32_t tile_count) {
    if (tile_count > X280_TLB_PACKET_INJECTION_CREDIT_COUNT) {
        throw std::invalid_argument(
            "consumer_window_slots: tile_count " + std::to_string(tile_count) + " exceeds the " +
            std::to_string(X280_TLB_PACKET_INJECTION_CREDIT_COUNT) + " credit slots");
    }
    std::vector<uint32_t> slots;
    for (uint32_t t = 0; t < tile_count; ++t) {
        slots.push_back(X280_TLB_PACKET_INJECTION_CREDIT_BASE + t);
    }
    if (mode == PI_MODE_LATENCY) {
        slots.push_back(X280_TLB_PACKET_INJECTION_TIMER);
    }
    return slots;
}

/** Which admission check refused a Config; Admission::value and bound are its operands. */
enum class AdmissionCheck : uint32_t {
    None,                ///< admitted
    Mode,                ///< value: mode
    IncludeCreation,     ///< value: include_creation, bound: 1
    ThroughputCreation,  ///< value: include_creation, bound: 0 (throughput creates no packets)
    Family,              ///< value: family
    TileCount,           ///< value: tile_count, bound: PI_MAX_TILES
    WorkerCount,         ///< value: P, bound: PI_MAX_WORKERS
    OrderedTiles,        ///< value: ordered_tiles entries, bound: tile_count
    DuplicateTile,       ///< value: the later index, bound: the earlier index of the same tile (Admission::tile)
    DestinationChip,     ///< value: dst_chip, bound: FS_MAX_MESH_SIZE
    DestinationMesh,     ///< value: dst_mesh, bound: FS_MAX_NUM_MESHES
    S,                   ///< value: S, bound: 2^31 (a nonzero power of 2 below it)
    B,                   ///< value: B, bound: 1
    K,                   ///< value: K, bound: 1
    C,                   ///< value: C, bound: K
    Inflight,            ///< value: inflight, bound: 4 (1, 2 or 4; 1 in latency mode)
    N,                   ///< value: N, bound: 1
    PollCadence,         ///< value: poll_cadence, bound: 2^20 (a power of 2 up to it)
    PhaseTimeout,        ///< value: phase_timeout_ms, bound: 600000
    HostTimeout,         ///< value: host_timeout_ms, bound: the floor it must reach
    ProducerTimerHz,     ///< value: producer_timer_hz (zero)
    MtimeHz,             ///< value: mtime_hz (zero)
    X280Ticks,           ///< value: phase_timeout_ms, bound: mtime_hz
    ProducerTicks,       ///< value: phase_timeout_ms, bound: producer_timer_hz
    WPlusN,              ///< value: W, bound: N (their sum overflows)
    PTimesW,             ///< value: P, bound: W (their product overflows)
    PTimesN,             ///< value: P, bound: N (their product overflows)
    LimBacked,           ///< value: bytes required, bound: LIM bytes backed
    Layout,              ///< value and bound: Admission::layout.check's, for Admission::layout_inputs
    L1Window,            ///< value: l1_window_base, bound: l1_span_limit
    L1Tile,              ///< value: tile L1 bytes required, bound: bytes available
};

/**
 * The verdict on one Config against one Capacity.
 *
 * `layout` and the two tick budgets are meaningful only when ok(), and are the
 * values the launch is meant to transmit: admission derives them to prove they
 * hold, so no caller needs to re-derive them. `required_bytes` is then the
 * arena the run needs and `available_bytes` the arena limit. A rejection sets
 * `error`, the `check` that refused, and that check's `value` and `bound`;
 * a DuplicateTile rejection also names the `tile`, and a Layout rejection the
 * `layout_inputs` it was computed from. describe() renders them on demand.
 */
struct Admission {
    /** The terms compute_layout() was given: worker and tile counts, S, B, N, mode, L1 base. */
    struct LayoutInputs {
        uint64_t P = 0;
        uint32_t T = 0;
        uint32_t S = 0;
        uint32_t B = 0;
        uint64_t N = 0;
        uint32_t mode = 0;
        uint32_t l1_tile_base = 0;
    };

    pi_error_t error = PI_ERROR_NONE;
    AdmissionCheck check = AdmissionCheck::None;
    uint64_t value = 0;
    uint64_t bound = 0;
    LogicalCoord tile{};
    LayoutInputs layout_inputs{};
    uint64_t required_bytes = 0;
    uint64_t available_bytes = 0;
    Layout layout{};
    uint64_t x280_timeout_ticks = 0;
    uint64_t producer_timeout_ticks = 0;

    bool ok() const { return error == PI_ERROR_NONE; }

    std::string describe() const {
        const std::string v = std::to_string(value);
        const std::string b = std::to_string(bound);
        switch (check) {
            case AdmissionCheck::None: return "admitted";
            case AdmissionCheck::Mode: return "unknown packet injection mode " + v;
            case AdmissionCheck::IncludeCreation: return "include_creation must be 0 or 1, got " + v;
            case AdmissionCheck::ThroughputCreation: return "throughput mode requires include_creation=0, got " + v;
            case AdmissionCheck::Family: return "worker family must be 1 (brisc), 2 (paired) or 3 (ethernet), got " + v;
            case AdmissionCheck::TileCount: return "tile_count must be in [1, " + b + "], got " + v;
            case AdmissionCheck::WorkerCount: return "worker_count P (" + v + ") exceeds max " + b;
            case AdmissionCheck::OrderedTiles:
                return "ordered_tiles has fewer entries (" + v + ") than tile_count (" + b + ")";
            case AdmissionCheck::DuplicateTile:
                return "duplicate tile (" + std::to_string(tile.x) + "," + std::to_string(tile.y) + ") at indices " +
                       b + " and " + v;
            case AdmissionCheck::DestinationChip: return "destination chip " + v + " >= " + b;
            case AdmissionCheck::DestinationMesh: return "destination mesh " + v + " >= " + b;
            case AdmissionCheck::S: return "S must be a nonzero power of 2 < " + b + ", got " + v;
            case AdmissionCheck::B: return "B must be >= " + b + ", got " + v;
            case AdmissionCheck::K: return "K must be >= " + b + ", got " + v;
            case AdmissionCheck::C: return "C must be in [1, K], got C=" + v + " K=" + b;
            case AdmissionCheck::Inflight: return "inflight must be 1, 2 or 4, and 1 in latency mode, got " + v;
            case AdmissionCheck::N: return "N must be >= " + b + ", got " + v;
            case AdmissionCheck::PollCadence: return "poll_cadence must be a power of 2 in [1, " + b + "], got " + v;
            case AdmissionCheck::PhaseTimeout: return "phase_timeout_ms must be in [1, " + b + "], got " + v;
            case AdmissionCheck::HostTimeout:
                return "host_timeout_ms (" + v + ") must be >= " + std::to_string(kHostTimeoutPhases) +
                       "*phase_timeout_ms+" + std::to_string(kHostTimeoutSlackMs) + " (" + b + ")";
            case AdmissionCheck::ProducerTimerHz: return "producer_timer_hz must be nonzero, got " + v;
            case AdmissionCheck::MtimeHz: return "mtime_hz must be nonzero, got " + v;
            case AdmissionCheck::X280Ticks:
                return "phase_timeout_ms (" + v + ") does not convert to a half-range tick budget at mtime_hz " + b;
            case AdmissionCheck::ProducerTicks:
                return "phase_timeout_ms (" + v +
                       ") does not convert to a half-range tick budget at producer_timer_hz " + b;
            case AdmissionCheck::WPlusN: return "W (" + v + ") + N (" + b + ") overflows uint64";
            case AdmissionCheck::PTimesW: return "P (" + v + ") * W (" + b + ") overflows uint64";
            case AdmissionCheck::PTimesN: return "P (" + v + ") * N (" + b + ") overflows uint64";
            case AdmissionCheck::LimBacked:
                return "LIM backed bytes (" + b + ") less than required " + v + " (14 ways)";
            case AdmissionCheck::Layout: {
                const LayoutInputs& in = layout_inputs;
                return "checked layout rejected P=" + std::to_string(in.P) + " T=" + std::to_string(in.T) +
                       " S=" + std::to_string(in.S) + " B=" + std::to_string(in.B) + " N=" + std::to_string(in.N) +
                       " mode=" + std::to_string(in.mode) + " l1_tile_base=" + std::to_string(in.l1_tile_base) + ": " +
                       layout.describe();
            }
            case AdmissionCheck::L1Window:
                return "L1 span limit (" + b + ") does not exceed L1 window base (" + v + ")";
            case AdmissionCheck::L1Tile: return "tile L1 requirement (" + v + ") exceeds available L1 (" + b + ")";
        }
        return "unknown admission check " + std::to_string(static_cast<uint32_t>(check));
    }
};

/**
 * Validates external configuration and placement against measured device
 * state, and returns the layout the run will use.
 *
 * Checks the mode and creation variant, the family and the P/T relationship,
 * the ordered tile list for shortfall and duplicates, the destination, the
 * S/B/C/K/N domains, poll cadence, the phase and host timeouts against both
 * clocks, host-side counting overflow, LIM and L1 capacity, and finally the
 * checked layout itself. Nothing is derived from a rejected Config.
 *
 * @param l1_tile_base as for compute_layout(); pass the allocated base to
 *                     re-prove alignment and the 32-bit extent bound once the
 *                     L1 pages exist.
 */
inline Admission validate_config(const Config& cfg, const Capacity& cap, uint32_t l1_tile_base = 0) {
    Admission adm{};
    const auto reject = [&adm](pi_error_t error, AdmissionCheck check, uint64_t value, uint64_t bound) {
        adm.error = error;
        adm.check = check;
        adm.value = value;
        adm.bound = bound;
        return adm;
    };

    // Mode check
    if (cfg.mode != PI_MODE_LATENCY && cfg.mode != PI_MODE_THROUGHPUT && cfg.mode != PI_MODE_BACKPRESSURE) {
        return reject(PI_ERROR_MODE_MISMATCH, AdmissionCheck::Mode, cfg.mode, 0u);
    }

    // include_creation flag check
    if (cfg.include_creation > 1u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::IncludeCreation, cfg.include_creation, 1u);
    }
    if (cfg.mode == PI_MODE_THROUGHPUT && cfg.include_creation != 0u) {
        return reject(PI_ERROR_MODE_MISMATCH, AdmissionCheck::ThroughputCreation, cfg.include_creation, 0u);
    }

    // Worker family, which fixes P/T
    if (cfg.family != WorkerFamily::BriscOnly && cfg.family != WorkerFamily::Paired &&
        cfg.family != WorkerFamily::Ethernet) {
        return reject(PI_ERROR_BAD_PLACEMENT, AdmissionCheck::Family, static_cast<uint32_t>(cfg.family), 0u);
    }

    // Tile count bounds
    if (cfg.tile_count == 0u || cfg.tile_count > PI_MAX_TILES) {
        return reject(PI_ERROR_BAD_PLACEMENT, AdmissionCheck::TileCount, cfg.tile_count, PI_MAX_TILES);
    }

    // Workers count P
    const uint64_t P = worker_count(cfg);
    if (P > PI_MAX_WORKERS) {
        return reject(PI_ERROR_BAD_PLACEMENT, AdmissionCheck::WorkerCount, P, PI_MAX_WORKERS);
    }

    // Ordered tiles list
    if (cfg.ordered_tiles.size() < cfg.tile_count) {
        return reject(PI_ERROR_BAD_PLACEMENT, AdmissionCheck::OrderedTiles, cfg.ordered_tiles.size(), cfg.tile_count);
    }
    for (size_t i = 0; i < cfg.tile_count; ++i) {
        for (size_t j = i + 1; j < cfg.tile_count; ++j) {
            if (cfg.ordered_tiles[i] == cfg.ordered_tiles[j]) {
                adm.tile = cfg.ordered_tiles[i];
                return reject(PI_ERROR_BAD_PLACEMENT, AdmissionCheck::DuplicateTile, j, i);
            }
        }
    }

    // Destination checks (FS_MAX_MESH_SIZE = 256, FS_MAX_NUM_MESHES = 1024)
    if (cfg.dst_chip >= 256u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::DestinationChip, cfg.dst_chip, 256u);
    }
    if (cfg.dst_mesh >= 1024u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::DestinationMesh, cfg.dst_mesh, 1024u);
    }

    // S check: nonzero power of 2, S < 2^31
    if (cfg.S == 0u || (cfg.S & (cfg.S - 1u)) != 0u || cfg.S >= 0x80000000u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::S, cfg.S, 0x80000000u);
    }

    // B and K check: B >= 1, K >= 1
    if (cfg.B == 0u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::B, cfg.B, 1u);
    }
    if (cfg.K == 0u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::K, cfg.K, 1u);
    }

    // C check: 1 <= C <= K
    if (cfg.C == 0u || cfg.C > cfg.K) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::C, cfg.C, cfg.K);
    }

    if ((cfg.inflight != 1u && cfg.inflight != 2u && cfg.inflight != 4u) ||
        (cfg.mode == PI_MODE_LATENCY && cfg.inflight != 1u)) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::Inflight, cfg.inflight, 4u);
    }

    // N check: N must be > 0
    if (cfg.N == 0u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::N, cfg.N, 1u);
    }

    // poll_cadence check: power of 2 in [1, 2^20]
    if (cfg.poll_cadence == 0u || (cfg.poll_cadence & (cfg.poll_cadence - 1u)) != 0u || cfg.poll_cadence > (1u << 20)) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::PollCadence, cfg.poll_cadence, 1u << 20);
    }

    // Phase and host timeouts
    if (cfg.phase_timeout_ms == 0u || cfg.phase_timeout_ms > 600000u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::PhaseTimeout, cfg.phase_timeout_ms, 600000u);
    }
    /* The phase cap above keeps the floor far inside 64 bits. */
    uint64_t min_host_timeout = 0;
    if (!measurement::host_wait_floor_ms(
            kHostTimeoutPhases, cfg.phase_timeout_ms, kHostTimeoutSlackMs, &min_host_timeout) ||
        cfg.host_timeout_ms < min_host_timeout) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::HostTimeout, cfg.host_timeout_ms, min_host_timeout);
    }

    // Clocks: both are reported and both convert the phase budget into ticks
    uint64_t x280_ticks = 0;
    uint64_t producer_ticks = 0;
    if (cap.producer_timer_hz == 0u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::ProducerTimerHz, cap.producer_timer_hz, 0u);
    }
    if (cap.mtime_hz == 0u) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::MtimeHz, cap.mtime_hz, 0u);
    }
    if (!measurement::phase_timeout_ticks(cfg.phase_timeout_ms, cap.mtime_hz, &x280_ticks)) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::X280Ticks, cfg.phase_timeout_ms, cap.mtime_hz);
    }
    if (!measurement::phase_timeout_ticks(cfg.phase_timeout_ms, cap.producer_timer_hz, &producer_ticks)) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::ProducerTicks, cfg.phase_timeout_ms, cap.producer_timer_hz);
    }

    // Checked arithmetic for the host's own per-run counting
    uint64_t count = 0;
    if (__builtin_add_overflow(cfg.W, cfg.N, &count)) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::WPlusN, cfg.W, cfg.N);
    }
    if (__builtin_mul_overflow(P, cfg.W, &count)) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::PTimesW, P, cfg.W);
    }
    if (__builtin_mul_overflow(P, cfg.N, &count)) {
        return reject(PI_ERROR_BAD_ARGS, AdmissionCheck::PTimesN, P, cfg.N);
    }

    // LIM backing the arena
    if (cap.lim_backed_bytes < kRequiredLimBackedBytes) {
        return reject(PI_ERROR_CAPACITY, AdmissionCheck::LimBacked, kRequiredLimBackedBytes, cap.lim_backed_bytes);
    }

    // Assigned before the first path that can return, so every rejection from
    // here on carries the check that explains it. PI_LAYOUT_OK is zero, so a
    // default-constructed Layout reports ok() -- "never computed" and
    // "computed, no problem" are the same bit pattern, and a rejection that
    // left this member default would claim a layout it never derived.
    const Layout layout = compute_layout(cfg, l1_tile_base);
    adm.layout = layout;
    if (!layout.ok()) {
        adm.layout_inputs = {P, cfg.tile_count, cfg.S, cfg.B, cfg.N, cfg.mode, l1_tile_base};
        return reject(
            pi_layout_code_to_error(layout.check.code), AdmissionCheck::Layout, layout.check.value, layout.check.bound);
    }

    // Target L1 bounds, against the window the target can actually hand out
    if (cap.l1_span_limit <= cap.l1_window_base) {
        return reject(PI_ERROR_CAPACITY, AdmissionCheck::L1Window, cap.l1_window_base, cap.l1_span_limit);
    }
    const uint64_t avail_l1 = (uint64_t)(cap.l1_span_limit - cap.l1_window_base);
    if (layout.abi.l1_tile_bytes > avail_l1) {
        return reject(PI_ERROR_CAPACITY, AdmissionCheck::L1Tile, layout.abi.l1_tile_bytes, avail_l1);
    }

    adm.error = PI_ERROR_NONE;
    adm.required_bytes = layout.abi.arena_bytes;
    adm.available_bytes = PI_ARENA_BYTES;
    adm.x280_timeout_ticks = x280_ticks;
    adm.producer_timeout_ticks = producer_ticks;
    return adm;
}

/** Ethernet tiles on one Blackhole device, and so the most ERISC writers a run
 * selects (tt-isa-documentation BlackholeA0/EthernetTile). */
inline constexpr uint32_t kMaxEthernetWriters = 14u;

/**
 * The ERISC writers of one run, chosen by select_ethernet_writers().
 *
 * `requested` is the writer count the run needs, with an all-available request
 * already resolved; `available` the distinct inactive Ethernet cores offered.
 * `writers` holds exactly `requested` cores only when ok().
 */
struct WriterSelection {
    enum class Outcome : uint32_t {
        Selected,   ///< `writers` is the first `requested` cores in order
        Shortfall,  ///< fewer than `requested` cores are available
        BeyondCap,  ///< `requested` exceeds kMaxEthernetWriters
        Duplicate   ///< `duplicate` was offered twice
    };

    Outcome outcome = Outcome::Shortfall;
    std::vector<LogicalCoord> writers;
    uint32_t requested = 0;
    uint32_t available = 0;
    LogicalCoord duplicate{};

    bool ok() const { return outcome == Outcome::Selected; }

    std::string describe() const {
        switch (outcome) {
            case Outcome::Selected:
                return "selected " + std::to_string(requested) + " of " + std::to_string(available) +
                       " inactive Ethernet cores";
            case Outcome::Shortfall:
                if (requested == 0u) {
                    return "requested every available ERISC writer, but the device has no "
                           "inactive Ethernet core";
                }
                return "requested " + std::to_string(requested) + " ERISC writers but the device has " +
                       std::to_string(available) + " inactive Ethernet cores";
            case Outcome::BeyondCap:
                return "requested " + std::to_string(requested) + " ERISC writers, above the " +
                       std::to_string(kMaxEthernetWriters) + " Ethernet tiles of one device";
            case Outcome::Duplicate:
                return "inactive Ethernet core (" + std::to_string(duplicate.x) + "," + std::to_string(duplicate.y) +
                       ") was reported twice";
        }
        return "unknown writer selection outcome";
    }
};

/**
 * Chooses the writers of one ERISC run from the device's inactive Ethernet
 * cores, in any order.
 *
 * The cores are sorted by logical (x, y), so every writer count is a prefix of
 * every larger one and the one-writer core is present in every point.
 * `requested` zero asks for every available core, capped at
 * kMaxEthernetWriters. A shortfall is reported, never narrowed to a smaller
 * count.
 */
inline WriterSelection select_ethernet_writers(std::vector<LogicalCoord> inactive, uint32_t requested) {
    WriterSelection sel{};
    std::sort(inactive.begin(), inactive.end(), [](const LogicalCoord& a, const LogicalCoord& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });
    for (size_t i = 1; i < inactive.size(); ++i) {
        if (inactive[i] == inactive[i - 1]) {
            sel.outcome = WriterSelection::Outcome::Duplicate;
            sel.duplicate = inactive[i];
            return sel;
        }
    }
    sel.available = static_cast<uint32_t>(inactive.size());
    sel.requested = requested != 0u ? requested : std::min(sel.available, kMaxEthernetWriters);
    if (sel.requested > kMaxEthernetWriters) {
        sel.outcome = WriterSelection::Outcome::BeyondCap;
        return sel;
    }
    if (sel.requested == 0u || sel.requested > sel.available) {
        sel.outcome = WriterSelection::Outcome::Shortfall;
        return sel;
    }
    sel.writers.assign(inactive.begin(), inactive.begin() + sel.requested);
    sel.outcome = WriterSelection::Outcome::Selected;
    return sel;
}

}  // namespace x280::harness::regions::packet_injection

#endif  // X280_SUITES_PACKET_INJECTION_REGIONS_HPP
