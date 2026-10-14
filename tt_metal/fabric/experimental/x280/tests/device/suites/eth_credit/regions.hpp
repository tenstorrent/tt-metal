// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/eth_credit/regions.hpp -- pure host admission, layout, and argument
// derivation for the eth_credit benchmark across a direct Ethernet link.
//
// No hardware dependency, so the host-only tests run without silicon.

#ifndef X280_SUITES_ETH_CREDIT_REGIONS_HPP
#define X280_SUITES_ETH_CREDIT_REGIONS_HPP

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kernels/eth_credit/eth_credit_abi.h"
#include "harness/regions/l1_window.hpp"
#include "harness/measurement/measurement.hpp"
#include "rt/x280_noc_map.h"
#include "x280/device/translated_coord.hpp"

namespace x280::harness::regions::eth_credit {

/** Chips of the pair; chip 0 is A and chip 1 is B. */
inline constexpr uint32_t kChips = 2u;

enum class Direction : uint32_t { AToB = 1u, BToA = 2u, Both = 3u };

/** Whether chip `c` sends in direction `d`. */
inline bool outbound(Direction d, uint32_t c) { return (static_cast<uint32_t>(d) & (c == 0u ? 1u : 2u)) != 0u; }

/** Whether chip `c` receives in direction `d`. */
inline bool inbound(Direction d, uint32_t c) { return outbound(d, c == 0u ? 1u : 0u); }

/**
 * One chip's placement, in translated (virtual) coordinates from the X280
 * runtime resolvers: `x280_noc0` and `x280_noc1` are the X280 tile as E0 (on
 * NoC 0) and E1 (on NoC 1) address it with get_noc_addr(), and `eth` the
 * Ethernet tile as the X280's TLB window takes it, as the p2p suite's
 * metal_peer_runner aims its window at an Ethernet peer. admit() refuses a
 * placement missing any of the three as EC_CHECK_COORD whose value has bit 0,
 * 1 or 2 set for each. `stream_ids[p]` is this tile's pkts[p]
 * overlay stream.
 */
struct ChipPlacement {
    std::optional<TranslatedCoord> x280_noc0;
    std::optional<TranslatedCoord> x280_noc1;
    std::optional<TranslatedCoord> eth;
    std::array<uint32_t, EC_MAX_P> stream_ids{};
};

/**
 * Measured device state. Each window is the unreserved region the HAL reports
 * for that core type (the X280's is its available LIM); clocks are the rates
 * each kernel's stall-budget clock ticks at.
 */
struct Capacity {
    Window eth{};
    Window lim{};
    uint64_t x280_hz = 0;
    uint64_t erisc_hz = 0;
};

struct Config {
    ec_shape_t shape{1u, 8u, 8u, EC_HEADER_BYTES, 1u};
    Direction direction = Direction::AToB;
    /** Packets per p in each active direction; zero is a run with no traffic,
     * and entries at p >= P must be 0. */
    std::array<uint32_t, EC_MAX_P> packets{};
    uint32_t poll_cadence = EC_DEFAULT_POLL_CADENCE;
    uint64_t stall_budget_ms = 1000;
    /** Forwarder hart that directs each chip's barrier. */
    uint32_t director_p = 0;
    uint16_t dst_mesh = 0;
    uint32_t run_nonce = 1;
    /** Iteration cap on each Metal-side barrier poll. */
    uint64_t barrier_cap = 100000000ull;
    /** The host's X280 wait in ms; 0 takes the floor for stall_budget_ms,
     * a no-progress floor plus slack, not a total-run deadline. A case whose
     * traffic can exceed kHostWaitSlackMs passes its wait explicitly. */
    uint64_t host_wait_ms = 0;
    std::array<ChipPlacement, kChips> chips{};
};

/* Stall budgets a failing run can spend in sequence before every kernel has
 * exited: E0's handshake, the timed loop, E0's final credit drain, and one more
 * for a neighbour that notices the failure only on its own budget. */
inline constexpr uint64_t kHostWaitPhases = 4;
/* Launch and a healthy run's traffic, which a budget reset on progress does
 * not bound (as packet_injection's 9 x phase + 5000 ms floor). */
inline constexpr uint64_t kHostWaitSlackMs = 5000;

/** One chip's layout at its absolute bases. */
struct ChipLayout {
    bool outbound = false;
    bool inbound = false;
    ec_eth_layout_t eth{};
    uint32_t eth_ctrl_base = 0;
    ec_x280_layout_t x280{};
    uint64_t lim_base = 0;
    /** End of the LIM window the layout was placed in, which every role takes
     * as its LIM bound. */
    uint64_t lim_end = 0;

    uint64_t fwd_area(uint32_t p) const {
        return lim_base + x280.fwd_offset + static_cast<uint64_t>(p) * x280.fwd_stride;
    }
    uint32_t tx_slot(uint32_t p, uint32_t k) const {
        return eth_ctrl_base + eth.tx_offset + p * eth.fifo_stride + k * slot_size;
    }
    uint32_t rx_slot(uint32_t p, uint32_t k) const {
        return eth_ctrl_base + eth.rx_offset + p * eth.fifo_stride + k * slot_size;
    }

    uint32_t slot_size = EC_HEADER_BYTES;
};

/**
 * Both chips' layouts, or the first check that refused them. `chip` names the
 * chip the failing check was evaluated on. Only read one that compute_layout()
 * or admit() returned: a default Layout reads as ok().
 */
struct Layout {
    std::array<ChipLayout, kChips> chips{};
    ec_check_t check{};
    uint32_t chip = 0;

    bool ok() const { return check.code == EC_CHECK_OK; }
};

/** Name of an ec_constraint_t, for describe(). Never null. */
inline const char* constraint_name(uint32_t c) {
    switch (static_cast<ec_constraint_t>(c)) {
        case EC_CONSTRAINT_NONE: return "none";
        case EC_CONSTRAINT_P: return "P";
        case EC_CONSTRAINT_N_LINK: return "N_LINK";
        case EC_CONSTRAINT_N_DELIVER: return "N_DELIVER";
        case EC_CONSTRAINT_SLOT_SIZE: return "SLOT_SIZE";
        case EC_CONSTRAINT_BATCH: return "BATCH";
        case EC_CONSTRAINT_ETH_WINDOW: return "Ethernet L1 window";
        case EC_CONSTRAINT_LIM_WINDOW: return "X280 LIM window";
        case EC_CONSTRAINT_ETH_BASE: return "Ethernet control base";
        case EC_CONSTRAINT_LIM_BASE: return "X280 LIM base";
        case EC_CONSTRAINT_DIRECTION: return "direction";
        case EC_CONSTRAINT_ARGS: return "argument record";
        case EC_CONSTRAINT_PEER_BASE: return "peer tile region";
        case EC_CONSTRAINT_FWD_AREA: return "forwarder area";
        case EC_CONSTRAINT_STREAM: return "stream id";
        case EC_CONSTRAINT_READY: return "barrier ready table";
        case EC_CONSTRAINT_HOST_WAIT: return "host X280 wait";
    }
    return "unknown constraint";
}

/** Name of an ec_check_code_t, for describe(). Never null. */
inline const char* check_code_name(uint32_t c) {
    switch (static_cast<ec_check_code_t>(c)) {
        case EC_CHECK_OK: return "ok";
        case EC_CHECK_BAD_PARAM: return "parameter out of domain";
        case EC_CHECK_ALIGNMENT: return "misaligned";
        case EC_CHECK_CAPACITY: return "does not fit";
        case EC_CHECK_CONGRUENCE: return "not congruent mod 16";
        case EC_CHECK_STREAM_ID: return "stream id";
        case EC_CHECK_ARG_WORDS: return "runtime-argument words";
        case EC_CHECK_COORD: return "NoC coordinate";
    }
    return "unknown check";
}

/** Renders a check for a human; the fields stay the record. */
inline std::string describe(const ec_check_t& c, uint32_t chip) {
    std::string s = std::string(check_code_name(c.code)) + " (" + constraint_name(c.constraint) + ") on chip " +
                    std::to_string(chip) + ": value " + std::to_string(c.value) + " against bound " +
                    std::to_string(c.bound);
    if (c.code == EC_CHECK_CAPACITY && c.value > c.bound) {
        s += ", " + std::to_string(c.value - c.bound) + " bytes over";
    }
    return s;
}

/**
 * Derives both chips' layouts at the windows in `cap`. Each Ethernet control
 * area sits at the window base rounded up to 512, and the LIM area at its base
 * rounded up to 64. Every placed extent must fit its window, and the
 * handshake unit must share an address on both tiles because the handshake
 * writes the peer's copy at its own address.
 */
inline Layout compute_layout(const Config& cfg, const Capacity& cap) {
    Layout out{};
    const ec_check_t shape = ec_validate_shape(&cfg.shape);
    if (shape.code != EC_CHECK_OK) {
        out.check = shape;
        return out;
    }
    const uint32_t dir = static_cast<uint32_t>(cfg.direction);
    if (dir < 1u || dir > 3u) {
        out.check = ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_DIRECTION, dir, 3u);
        return out;
    }

    const auto place = [](const Window& w,
                          uint64_t align,
                          uint64_t need,
                          uint32_t constraint,
                          uint32_t base_constraint,
                          uint64_t* base) -> ec_check_t {
        /* Every placement ends at or below 2^32, and bounding the base first keeps
         * the alignment and extent sums far from wrapping. */
        if (w.base > (1ull << 32)) {
            return ec_check(EC_CHECK_CAPACITY, base_constraint, w.base, 1ull << 32);
        }
        const uint64_t aligned = (w.base + align - 1u) & ~(align - 1u);
        const uint64_t required = (aligned - w.base) + need;
        if (w.size < required) {
            return ec_check(EC_CHECK_CAPACITY, constraint, required, w.size);
        }
        if (aligned + need > (1ull << 32)) {
            return ec_check(EC_CHECK_CAPACITY, base_constraint, aligned + need, 1ull << 32);
        }
        *base = aligned;
        return ec_check_ok();
    };

    for (uint32_t c = 0; c < kChips; ++c) {
        ChipLayout& l = out.chips[c];
        l.outbound = outbound(cfg.direction, c);
        l.inbound = inbound(cfg.direction, c);
        l.slot_size = cfg.shape.slot_size;
        out.chip = c;

        ec_check_t k = ec_compute_eth_layout(&cfg.shape, l.outbound ? 1 : 0, l.inbound ? 1 : 0, &l.eth);
        if (k.code != EC_CHECK_OK) {
            out.check = k;
            return out;
        }
        uint64_t base = 0;
        k = place(
            cap.eth, EC_ETH_CONTROL_ALIGN, l.eth.total_bytes, EC_CONSTRAINT_ETH_WINDOW, EC_CONSTRAINT_ETH_BASE, &base);
        if (k.code != EC_CHECK_OK) {
            out.check = k;
            return out;
        }
        l.eth_ctrl_base = static_cast<uint32_t>(base);

        k = ec_compute_x280_layout(&cfg.shape, &l.x280);
        if (k.code != EC_CHECK_OK) {
            out.check = k;
            return out;
        }
        k = place(
            cap.lim, EC_LIM_ALIGN, l.x280.total_bytes, EC_CONSTRAINT_LIM_WINDOW, EC_CONSTRAINT_LIM_BASE, &l.lim_base);
        if (k.code != EC_CHECK_OK) {
            out.check = k;
            return out;
        }
        l.lim_end = cap.lim.base + cap.lim.size;
    }

    const uint32_t hs0 = out.chips[0].eth_ctrl_base + out.chips[0].eth.handshake;
    const uint32_t hs1 = out.chips[1].eth_ctrl_base + out.chips[1].eth.handshake;
    if (hs0 != hs1) {
        out.chip = 1;
        out.check = ec_check(EC_CHECK_ALIGNMENT, EC_CONSTRAINT_ETH_BASE, hs1, hs0);
        return out;
    }
    out.chip = 0;
    out.check = ec_check_ok();
    return out;
}

/**
 * Every base and stride of `layout` is a multiple of 16. Blackhole silently
 * drops a NoC write whose source and destination differ mod 16, and every
 * transfer address is a base plus multiples of these strides and of layout
 * offsets that are whole 16-byte units by construction, so this is the whole
 * congruence requirement.
 *
 * @return OK, or EC_CHECK_CONGRUENCE with the offending base or stride in
 *         `value`, what it is in `constraint` and 16 in `bound`; `*chip`
 *         names its chip.
 */
inline ec_check_t check_congruence(const Layout& layout, uint32_t* chip) {
    for (uint32_t c = 0; c < kChips; ++c) {
        const ChipLayout& l = layout.chips[c];
        const std::array<std::pair<uint64_t, uint32_t>, 5> units = {{
            {l.eth_ctrl_base, EC_CONSTRAINT_ETH_BASE},
            {l.lim_base, EC_CONSTRAINT_LIM_BASE},
            {l.eth.fifo_stride, EC_CONSTRAINT_ETH_WINDOW},
            {l.x280.fwd_stride, EC_CONSTRAINT_LIM_WINDOW},
            {l.slot_size, EC_CONSTRAINT_SLOT_SIZE},
        }};
        for (const auto& [value, constraint] : units) {
            if ((value % EC_UNIT_BYTES) != 0u) {
                *chip = c;
                return ec_check(EC_CHECK_CONGRUENCE, constraint, value, EC_UNIT_BYTES);
            }
        }
    }
    return ec_check_ok();
}

/**
 * The verdict on one Config against one Capacity. `layout` and the two tick
 * budgets are meaningful only when ok(), and are what derive_arguments()
 * transmits. `chip` names the chip a failing check was evaluated on.
 */
struct Admission {
    ec_check_t check{};
    uint32_t chip = 0;
    Layout layout{};
    uint64_t x280_stall_ticks = 0;
    uint64_t erisc_stall_ticks = 0;
    /** The host's X280 wait: Config::host_wait_ms, or its floor when that is 0. */
    uint64_t host_wait_ms = 0;

    bool ok() const { return check.code == EC_CHECK_OK; }
    std::string describe() const { return eth_credit::describe(check, chip); }
};

/**
 * Validates `cfg` against `cap` and returns the layout the run will use.
 *
 * Checks the shape, direction, per-p packet counts, poll cadence, stall budget
 * against both clocks, the host X280 wait against its floor, director hart, stream ids (1..31, distinct per tile),
 * that every placement coordinate is present, fit per window, and that every base and stride is a
 * multiple of 16 (check_congruence).
 * Nothing is derived from a rejected Config.
 */
inline Admission admit(const Config& cfg, const Capacity& cap) {
    Admission adm{};
    const auto reject = [&adm](ec_check_t c, uint32_t chip) {
        adm.check = c;
        adm.chip = chip;
        return adm;
    };

    ec_check_t k = ec_validate_shape(&cfg.shape);
    if (k.code != EC_CHECK_OK) {
        return reject(k, 0u);
    }
    /* A zero count at p < P is a run with no traffic, which is how the barrier
     * and handshake are brought up alone. */
    for (uint32_t p = cfg.shape.P; p < EC_MAX_P; ++p) {
        if (cfg.packets[p] != 0u) {
            return reject(ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_P, p, cfg.packets[p]), 0u);
        }
    }
    if (!ec_is_pow2(cfg.poll_cadence) || cfg.poll_cadence > EC_MAX_POLL_CADENCE) {
        return reject(ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_NONE, cfg.poll_cadence, EC_MAX_POLL_CADENCE), 0u);
    }
    if (cfg.director_p >= cfg.shape.P) {
        return reject(ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_P, cfg.director_p, cfg.shape.P), 0u);
    }
    if (cfg.run_nonce == 0u || cfg.barrier_cap == 0u) {
        return reject(ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_NONE, cfg.run_nonce, cfg.barrier_cap), 0u);
    }
    if (!measurement::phase_timeout_ticks(cfg.stall_budget_ms, cap.x280_hz, &adm.x280_stall_ticks) ||
        !measurement::phase_timeout_ticks(cfg.stall_budget_ms, cap.erisc_hz, &adm.erisc_stall_ticks)) {
        return reject(ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_NONE, cfg.stall_budget_ms, 0u), 0u);
    }
    uint64_t wait_floor = 0;
    if (!measurement::host_wait_floor_ms(kHostWaitPhases, cfg.stall_budget_ms, kHostWaitSlackMs, &wait_floor)) {
        return reject(ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_HOST_WAIT, cfg.stall_budget_ms, 0u), 0u);
    }
    if (cfg.host_wait_ms != 0u && cfg.host_wait_ms < wait_floor) {
        return reject(ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_HOST_WAIT, cfg.host_wait_ms, wait_floor), 0u);
    }
    adm.host_wait_ms = cfg.host_wait_ms != 0u ? cfg.host_wait_ms : wait_floor;

    for (uint32_t c = 0; c < kChips; ++c) {
        const ChipPlacement& chip = cfg.chips[c];
        for (uint32_t p = 0; p < cfg.shape.P; ++p) {
            const uint32_t id = chip.stream_ids[p];
            if (id < EC_STREAM_ID_MIN || id > EC_STREAM_ID_MAX) {
                return reject(ec_check(EC_CHECK_STREAM_ID, EC_CONSTRAINT_P, id, EC_STREAM_ID_MAX), c);
            }
            for (uint32_t q = 0; q < p; ++q) {
                if (chip.stream_ids[q] == id) {
                    return reject(ec_check(EC_CHECK_STREAM_ID, EC_CONSTRAINT_P, id, q), c);
                }
            }
        }
        const uint32_t missing = (chip.x280_noc0 ? 0u : 1u) | (chip.x280_noc1 ? 0u : 2u) | (chip.eth ? 0u : 4u);
        if (missing != 0u) {
            return reject(ec_check(EC_CHECK_COORD, EC_CONSTRAINT_NONE, missing, 0u), c);
        }
    }

    adm.layout = compute_layout(cfg, cap);
    if (!adm.layout.ok()) {
        return reject(adm.layout.check, adm.layout.chip);
    }
    uint32_t failing_chip = 0;
    k = check_congruence(adm.layout, &failing_chip);
    if (k.code != EC_CHECK_OK) {
        return reject(k, failing_chip);
    }
    adm.check = ec_check_ok();
    return adm;
}

/** One chip's derived arguments. Forwarder p runs on hart p. */
struct ChipArguments {
    std::array<ec_x280_fwd_args_t, EC_X280_HARTS> fwd{};
    ec_e0_args_t e0{};
    bool has_e1 = false;
    ec_e1_args_t e1{};
};

struct Arguments {
    std::array<ChipArguments, kChips> chips{};
};

/** The TLB window forwarder hart `p` owns onto its Ethernet tile. */
inline uint32_t fwd_tlb_base(uint32_t p) { return X280_TLB_ETH_CREDIT_BASE + p * X280_TLB_ETH_CREDIT_PER_HART; }

/**
 * Derives every role's arguments on both chips from one admission. `adm` must
 * be ok() and come from admit(cfg, ...); nothing here re-validates.
 */
inline Arguments derive_arguments(const Config& cfg, const Admission& adm) {
    Arguments out{};
    const ec_shape_t& s = cfg.shape;
    const uint64_t eb = adm.erisc_stall_ticks;
    for (uint32_t c = 0; c < kChips; ++c) {
        const uint32_t o = 1u - c;
        const ChipLayout& l = adm.layout.chips[c];
        const ChipLayout& peer = adm.layout.chips[o];
        const ChipPlacement& pl = cfg.chips[c];
        ChipArguments& a = out.chips[c];
        const uint32_t ready_base = static_cast<uint32_t>(l.lim_base + l.x280.ready_offset);

        for (uint32_t p = 0; p < s.P; ++p) {
            ec_x280_fwd_args_t& f = a.fwd[p];
            f.magic = EC_ABI_MAGIC;
            f.kernel_id = EC_KERNEL_ID(EC_ROLE_X280_FWD, p, c);
            f.flags = (p == cfg.director_p ? EC_FWD_FLAG_DIRECTOR : 0u) |
                      (l.outbound ? EC_FWD_FLAG_OUTBOUND_ACTIVE : 0u) | (l.inbound ? EC_FWD_FLAG_INBOUND_ACTIVE : 0u);
            f.shape = s;
            f.p = p;
            f.director_p = cfg.director_p;
            f.run_nonce = cfg.run_nonce;
            f.poll_cadence = cfg.poll_cadence;
            f.send_count = l.outbound ? cfg.packets[p] : 0u;
            f.recv_count = l.inbound ? cfg.packets[p] : 0u;
            f.dst_chip = static_cast<uint16_t>(o);
            f.dst_mesh = cfg.dst_mesh;
            f.stall_budget_ticks = adm.x280_stall_ticks;
            f.lim_base = l.lim_base;
            f.lim_end = l.lim_end;
            f.eth = pl.eth->noc();
            f.eth_ctrl_base = l.eth_ctrl_base;
            f.tlb_base = fwd_tlb_base(p);
        }

        /* E0 and E1 share their record prefix, so one writer fills both. */
        const auto common = [&](ec_erisc_args_t& e, uint32_t kernel_id, const TranslatedCoord& fwd) {
            e.magic = EC_ABI_MAGIC;
            e.kernel_id = kernel_id;
            e.shape = s;
            e.run_nonce = cfg.run_nonce;
            e.poll_cadence = cfg.poll_cadence;
            e.stall_budget_lo = static_cast<uint32_t>(eb);
            e.stall_budget_hi = static_cast<uint32_t>(eb >> 32);
            e.ctrl_base = l.eth_ctrl_base;
            e.fwd = fwd.noc();
            e.barrier_ready_base = ready_base;
            e.barrier_cap_lo = static_cast<uint32_t>(cfg.barrier_cap);
            e.barrier_cap_hi = static_cast<uint32_t>(cfg.barrier_cap >> 32);
            for (uint32_t p = 0; p < s.P; ++p) {
                e.fwd_area[p] = static_cast<uint32_t>(l.fwd_area(p));
            }
            e.lim_end_lo = static_cast<uint32_t>(l.lim_end);
            e.lim_end_hi = static_cast<uint32_t>(l.lim_end >> 32);
        };

        common(a.e0.erisc, EC_KERNEL_ID(EC_ROLE_E0, 0u, c), *pl.x280_noc0);
        a.e0.erisc.flags =
            (c == 0u ? EC_E0_FLAG_HANDSHAKE_SENDER : 0u) | (l.outbound ? EC_E0_FLAG_OUTBOUND_ACTIVE : 0u);
        a.e0.peer_ctrl_base = peer.eth_ctrl_base;
        a.e0.peer_rx_base = peer.eth_ctrl_base + peer.eth.rx_offset;
        for (uint32_t p = 0; p < s.P; ++p) {
            a.e0.erisc.count[p] = l.outbound ? cfg.packets[p] : 0u;
            a.e0.erisc.stream_id[p] = cfg.chips[o].stream_ids[p];
            a.e0.local_stream_id[p] = pl.stream_ids[p];
        }

        a.has_e1 = l.inbound;
        if (l.inbound) {
            common(a.e1.erisc, EC_KERNEL_ID(EC_ROLE_E1, 0u, c), *pl.x280_noc1);
            a.e1.rx_base = l.eth_ctrl_base + l.eth.rx_offset;
            a.e1.peer_ctrl_base = peer.eth_ctrl_base;
            for (uint32_t p = 0; p < s.P; ++p) {
                a.e1.erisc.count[p] = cfg.packets[p];
                a.e1.erisc.stream_id[p] = pl.stream_ids[p];
            }
        }
    }
    return out;
}

/**
 * Serializes an E0 or E1 argument record into its runtime-argument words:
 * word i is the record's i-th uint32_t (eth_credit_abi.h, "Metal roles take
 * their arguments as runtime words"). It chooses nothing; derive_arguments()
 * did.
 */
template <typename Args>
inline std::vector<uint32_t> pack_runtime_args(const Args& a) {
    static_assert(
        std::is_same_v<Args, ec_e0_args_t> || std::is_same_v<Args, ec_e1_args_t>,
        "only the Metal roles' records are runtime words");
    std::vector<uint32_t> w(sizeof(Args) / sizeof(uint32_t));
    std::memcpy(w.data(), &a, sizeof(Args));
    return w;
}

}  // namespace x280::harness::regions::eth_credit

#endif  // X280_SUITES_ETH_CREDIT_REGIONS_HPP
