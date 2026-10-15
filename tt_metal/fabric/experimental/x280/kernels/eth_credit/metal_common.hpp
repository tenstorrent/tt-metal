// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/eth_credit/metal_common.hpp
//
// What E0 and E1 share: the compile-time shape and its
// runtime check, the result block, bounded NoC waits, the
// core_sync worker, and the stall-budget clock. Metal JIT builds only.

#ifndef X280_KERNELS_ETH_CREDIT_METAL_COMMON_HPP
#define X280_KERNELS_ETH_CREDIT_METAL_COMMON_HPP

#include <stddef.h>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "eth_credit_abi.h"
#include "../../include/x280_address_map.h"
#include "../shared/core_sync_barrier.hpp"
#include "../shared/core_sync_transport_metal.hpp"
#include "../shared/metal_bounded_wait.hpp"

#if defined(COMPILE_FOR_ERISC)
#include "internal/ethernet/tt_eth_api.h"
#include "internal/ethernet/tunneling.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"
#endif

#if !defined(EC_P) || !defined(EC_N_LINK) || !defined(EC_N_DELIVER) || !defined(EC_SLOT_SIZE) || !defined(EC_BATCH)
#error \
    "eth_credit Metal kernels are built per shape: EC_P, EC_N_LINK, EC_N_DELIVER, EC_SLOT_SIZE and EC_BATCH must be defined"
#endif

static_assert(EC_P >= 1 && EC_P <= EC_MAX_P, "EC_P must be in [1, EC_MAX_P]");
static_assert(
    EC_N_LINK >= 1 && EC_N_LINK <= EC_MAX_DEPTH && (EC_N_LINK & (EC_N_LINK - 1)) == 0,
    "EC_N_LINK must be a power of two no larger than EC_MAX_DEPTH");
static_assert(
    EC_N_DELIVER >= 1 && EC_N_DELIVER <= EC_MAX_DEPTH && (EC_N_DELIVER & (EC_N_DELIVER - 1)) == 0,
    "EC_N_DELIVER must be a power of two no larger than EC_MAX_DEPTH");
static_assert(
    EC_SLOT_SIZE >= EC_HEADER_BYTES && EC_SLOT_SIZE <= EC_MAX_SLOT_BYTES &&
        (EC_SLOT_SIZE - EC_HEADER_BYTES) % EC_UNIT_BYTES == 0,
    "EC_SLOT_SIZE must be 96 + 16k");
static_assert(
    EC_BATCH >= 1 && EC_BATCH <= EC_N_LINK && EC_BATCH <= EC_N_DELIVER,
    "EC_BATCH must be in [1, min(N_LINK, N_DELIVER)]");

namespace eth_credit::metal {

inline constexpr ec_shape_t kShape{EC_P, EC_N_LINK, EC_N_DELIVER, EC_SLOT_SIZE, EC_BATCH};
inline constexpr uint32_t kUnit = EC_UNIT_BYTES;

/* What an argument may name: the ERISC L1 above the firmware and kernel
 * config regions, up to the fabric router's reserved base (dev_mem_map.h),
 * and the X280 LIM, whose end ec_narrow_lim lowers to the record's lim_end. */
inline constexpr ec_bounds_t kBounds{
    MEM_AERISC_MAP_END + MEM_ERISC_KERNEL_CONFIG_SIZE, MEM_ERISC_MAX_SIZE, X280_LIM_BASE, X280_LIM_END};

/* This core's stall-budget and timestamp clock, in its own raw ticks. */
using metal_wait::now;

/**
 * Reads an E0 or E1 argument record from its runtime-argument words: word i
 * is the record's i-th uint32_t (eth_credit_abi.h, "Metal roles take their
 * arguments as runtime words").
 */
template <typename Args>
inline void read_args(Args& a) {
    static_assert(sizeof(Args) % sizeof(uint32_t) == 0u, "an argument record is whole words");
    uint32_t* w = reinterpret_cast<uint32_t*>(&a);
    for (uint32_t i = 0; i < sizeof(Args) / sizeof(uint32_t); ++i) {
        w[i] = get_arg_val<uint32_t>(i);
    }
}

inline uint32_t local_word(uint32_t addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(addr);
}

inline void set_local_word(uint32_t addr, uint32_t v) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = v; }

/** A kernel's result block in its own L1, written only by that kernel. */
class Result {
public:
    /** Zeroes the block and stamps its identity and the kernel's four sites. */
    void init(uint32_t addr, uint32_t kernel_id, const uint32_t (&sites)[EC_MAX_STALL_SITES]) {
        r_ = reinterpret_cast<volatile tt_l1_ptr ec_result_t*>(addr);
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t i = 0; i < EC_RESULT_BYTES / sizeof(uint32_t); ++i) {
            w[i] = 0u;
        }
        r_->magic = EC_RESULT_MAGIC;
        r_->kernel_id = kernel_id;
        for (uint32_t i = 0; i < EC_MAX_STALL_SITES; ++i) {
            r_->stalls[i].site = sites[i];
        }
    }

    /** Counts one pass at stall site `index` that took `ticks`, as ec_stall_sum. */
    void stall(uint32_t index, uint64_t ticks) {
        r_->stalls[index].count += 1u;
        r_->stalls[index].ticks = ec_stall_sum(r_->stalls[index].ticks, ticks);
    }

    /** Counts one pass at stall site `index` spanning [start, end] of now(), as ec_stall_add. */
    void stall(uint32_t index, uint64_t start, uint64_t end) {
        r_->stalls[index].count += 1u;
        r_->stalls[index].ticks = ec_stall_add(r_->stalls[index].ticks, start, end);
    }

    void stamp(uint64_t t) {
        if (r_->first_ts == 0u) {
            r_->first_ts = t;
        }
        r_->last_ts = t;
    }

    void packets_in(uint32_t p, uint32_t n) { r_->packets_in[p] = n; }
    void packets_out(uint32_t p, uint32_t n) { r_->packets_out[p] = n; }

    void failure(uint32_t p, uint32_t expected, uint32_t seen, uint32_t site, uint32_t local, uint32_t shadow) {
        r_->failure.p = p;
        r_->failure.expected = expected;
        r_->failure.seen = seen;
        r_->failure.site = site;
        r_->failure.local_ptr = local;
        r_->failure.shadow_ptr = shadow;
    }

    /** Records a refused argument: the constraint in `p`, the offending
     * value and its bound in `expected` and `seen`. */
    void bad_args(const ec_check_t& c) {
        failure(c.constraint, static_cast<uint32_t>(c.value), static_cast<uint32_t>(c.bound), EC_SITE_NONE, 0u, 0u);
    }

    /** Commits the terminal status, state last. */
    void finish(const ec_status_t& s) {
        r_->status.origin = s.origin;
        r_->status.code = s.code;
        r_->status.detail = s.detail;
        r_->status.state = s.state;
    }

private:
    volatile tt_l1_ptr ec_result_t* r_ = nullptr;
};

template <ProgrammableCoreType CoreType>
struct WorkerSync {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = false;
    static inline uint32_t kLocalReleaseAddr = 0u;
    static inline core_sync::MetalWordArrivalTarget kArrivalTarget{};
};

/**
 * Joins this chip's barrier as a word-arrival worker: the director is an X280
 * hart, which cannot host a counter. Each poll is bounded by `cap` iterations.
 * `cap` reaches the transport as sync()'s run_value (zero selects
 * MetalWaitContext's default). No host STOP reaches this path, so
 * Outcome::StopRequested means the cap expired.
 */
template <ProgrammableCoreType CoreType>
inline core_sync::Outcome barrier(
    x280_noc_coord_t fwd,
    uint32_t ready_base,
    uint32_t index,
    uint32_t release_addr,
    uint64_t cap,
    uint32_t run_nonce) {
    using Transport = core_sync::MetalTransport<CoreType, core_sync::ArrivalKind::Word>;
    WorkerSync<CoreType>::kLocalReleaseAddr = release_addr;
    WorkerSync<CoreType>::kArrivalTarget = core_sync::MetalWordArrivalTarget{fwd, ready_base, index};
    return core_sync::core_sync_barrier<WorkerSync<CoreType>, Transport>::sync(0u, cap, run_nonce);
}

#if defined(COMPILE_FOR_ERISC)

/**
 * Whether the Ethernet PCS reports link up, read directly. is_link_up() is not
 * used in a loop: on ERISC0 a down link makes it wait about 5.4e9 cycles
 * before answering (eth_fw_api.h:283-294), which would overrun the stall
 * budget the check enforces. ERISC1's arm of it is this same register read.
 */
inline bool link_up() {
    return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(ETH_CORE_A_ETH_CTRL_A_PCS_STATUS_REG_ADDR) == 1u;
}

/**
 * Waits until TXQ `q` is idle, or `budget` ticks pass. A TXQ command is
 * accepted once the queue is idle; on Blackhole ETH_TXQ_CMD reads zero, so the
 * status bit is the only readiness.
 * @return false on timeout; otherwise true, with the ticks waited added to
 *         `*stalled` as ec_stall_add.
 */
__attribute__((always_inline)) inline bool bounded_txq_idle(uint32_t q, uint64_t budget, uint64_t* stalled) {
    const uint64_t t0 = now();
    while (internal_::eth_txq_is_busy(q)) {
        if (now() - t0 >= budget) {
            return false;
        }
    }
    *stalled = ec_stall_add(*stalled, t0, now());
    return true;
}

static_assert(
    sizeof(erisc::datamover::handshake::handshake_info_t) == EC_HANDSHAKE_BYTES,
    "the control area reserves one handshake_info_t");

/**
 * erisc::datamover::handshake's sender and receiver sides with every wait
 * bounded by `budget` ticks: the same handshake_info_t at `addr` on both
 * tiles, the same 16-byte scratch-to-local_value TT-link write on TXQ 0, and
 * the same magic value. The sender resends until its own local_value holds the
 * magic; the receiver waits for it, then answers once.
 *
 * The unbounded original hands ERISC0 to base firmware (run_routing) after a
 * long wait; this one fails instead, within a budget far shorter than that.
 *
 * @return false on expiry, which the caller reports as
 *         STALL_TIMEOUT(EC_SITE_HANDSHAKE).
 */
inline bool bounded_handshake(uint32_t addr, bool sender, uint64_t budget) {
    namespace hs = erisc::datamover::handshake;
    constexpr uint32_t kTxq = 0u;
    volatile tt_l1_ptr hs::handshake_info_t* info = hs::init_handshake_info(addr, 0u, 0u);
    const uint32_t local_value = reinterpret_cast<uint32_t>(&info->local_value);
    const uint32_t scratch = reinterpret_cast<uint32_t>(&info->scratch);
    const uint64_t t0 = now();
    const auto send = [&]() {
        while (internal_::eth_txq_is_busy(kTxq)) {
            if (now() - t0 >= budget) {
                return false;
            }
        }
        internal_::eth_send_packet_bytes_unsafe(kTxq, scratch, local_value, kUnit);
        return true;
    };
    while (info->local_value != hs::MAGIC_HANDSHAKE_VALUE) {
        if (now() - t0 >= budget) {
            return false;
        }
        if (sender && !send()) {
            return false;
        }
        invalidate_l1_cache();
    }
    return sender || send();
}

#endif  // COMPILE_FOR_ERISC

/** The ec_error_t and site a barrier outcome other than Success maps to. */
inline uint32_t barrier_error(core_sync::Outcome o) {
    return o == core_sync::Outcome::StopRequested ? EC_ERROR_STALL_TIMEOUT : EC_ERROR_BAD_ARGS;
}

}  // namespace eth_credit::metal

#endif  // X280_KERNELS_ETH_CREDIT_METAL_COMMON_HPP
