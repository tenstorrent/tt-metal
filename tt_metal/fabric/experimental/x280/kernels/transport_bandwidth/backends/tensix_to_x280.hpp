// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/transport_bandwidth/backends/tensix_to_x280.hpp -- Tensix backend
// for the Tensix -> X280 transport_bandwidth member.
//
// This header is included ONLY by the Tensix data-movement kernel (the .cpp
// under tests/device/kernels). It uses Metal's dataflow API and must not
// leak into an X280 image compilation, which is why it lives beside the
// Tensix kernel path rather than under kernels/shared/.
//
// Tensix is the initiator; X280 is a passive addressable target -- no X280
// image is submitted for this member. Every issue is one noc_async_write
// of `transfer_size` bytes from a host-seeded L1 staging region into an
// X280 LIM destination the host allocated and prepared.
//
// Timing: paired native live-timer and per-RISC cycle CSR reads. Failed reads
// publish an invalid delta for host rejection.
//
// See docs/benchmarks/design.md § Transport bandwidth.

#ifndef X280_TRANSPORT_BW_BACKEND_TENSIX_TO_X280_HPP
#define X280_TRANSPORT_BW_BACKEND_TENSIX_TO_X280_HPP

#include <cstdint>
#include "../../../include/rt/measurement.h"
#include "../../../include/x280_noc_coord.h"
#include "../../shared/tensix_live_timer.h"
#include "../../shared/ubench_units.h"

#include "api/dataflow/dataflow_api.h"

#include "tensix_to_x280.h"
#include "transport_bandwidth_common.h"
#include "transport_bandwidth_frontend_cxx.hpp"

namespace transport_bandwidth::backend_tensix_to_x280 {

// Path-verification magic. Different from the X280 -> Tensix member's magic
// so a mis-launched kernel can never accidentally pass; chosen not-zero and
// not equal to the nonce for any small nonce so a stale word from an
// earlier run cannot answer.
constexpr uint32_t kVerifyMagic = 0xB4A5C0DEu;

// Admitted transfer sizes: multiples of the 64 B NoC word up to
// NOC_MAX_BURST_SIZE, so every issue is one NoC packet.
using sizes = size_list<64u, 128u, 256u, 320u, 384u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u>;
static_assert(sizes::max() <= NOC_MAX_BURST_SIZE, "an issue must be one NoC packet");
static_assert(
    sizes::max() == TB_TX2X_STAGING_MAX_BYTES,
    "the L1 staging region and X280 LIM destination must hold the largest transfer");

// ---------------------------------------------------------------------------
// Backend types
// ---------------------------------------------------------------------------

struct timestamp_t {
    uint64_t ticks;
    uint64_t cpu;
};

// Primary is the tile wall timer; secondary is the native per-RISC cycle CSR.
struct counter_delta_t {
    uint64_t primary;
    uint64_t secondary;  // raw native CPU cycles
};

struct state_t {
    // Parsed protocol args (transfer_size, warmup_count, transfer_count,
    // boundary, run_nonce). The frontend consults these through args().
    transport_bw_protocol_args_t args;

    // Own regions on L1 (u32 addresses; the API takes u32 for source).
    uint32_t own_staging_addr;  // source of every issue
    uint32_t own_scratch_addr;  // 4B slot for the verify read-back

    // L1 pointers for the verify path.
    volatile tt_l1_ptr uint32_t* staging32;
    volatile tt_l1_ptr uint32_t* scratch32;

    // Peer info: 48-bit LIM addresses packed as u64. Only the low half
    // lands in the NIU register but the host may pack a wider address.
    uint64_t peer_dst;
    uint64_t peer_verify;
    x280_noc_coord_t peer_noc;  // TRANSLATED coordinates

    // Result region on L1.
    transport_bw_result_t* r;
};

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------
// Parses the runtime-arg schema, sanity-checks it, and populates the result
// region's static fields plus the per-run staging pattern. Returns false on
// any layout or option mismatch -- a layout cookie other than
// TB_TX2X_LAYOUT_COOKIE before any other slot is read; the kernel returns
// without touching the transport, and the host observes DONE-not-set as
// ARG_LAYOUT_MISMATCH.
inline bool initialize(state_t& state) {
    if (!ubench_layout_is_current(get_arg_val<uint32_t>(TB_TX2X_ARG_LAYOUT_COOKIE), TB_TX2X_LAYOUT_COOKIE)) {
        return false;
    }
    state.args.transfer_size = get_arg_val<uint32_t>(TB_TX2X_ARG_TRANSFER_SIZE);
    state.args.warmup_count = get_arg_val<uint32_t>(TB_TX2X_ARG_WARMUP_COUNT);
    state.args.transfer_count = get_arg_val<uint32_t>(TB_TX2X_ARG_TRANSFER_COUNT);
    state.args.boundary = get_arg_val<uint32_t>(TB_TX2X_ARG_BOUNDARY);
    state.args.run_nonce = get_arg_val<uint32_t>(TB_TX2X_ARG_RUN_NONCE);
    state.args.reserved = 0u;

    state.own_staging_addr = get_arg_val<uint32_t>(TB_TX2X_ARG_OWN_STAGING_ADDR);
    state.own_scratch_addr = get_arg_val<uint32_t>(TB_TX2X_ARG_OWN_SCRATCH_ADDR);
    const uint32_t own_result = get_arg_val<uint32_t>(TB_TX2X_ARG_OWN_RESULT_ADDR);
    const uint32_t peer_dst_lo = get_arg_val<uint32_t>(TB_TX2X_ARG_PEER_DST_ADDR_LO);
    const uint32_t peer_dst_hi = get_arg_val<uint32_t>(TB_TX2X_ARG_PEER_DST_ADDR_HI);
    const uint32_t peer_ver_lo = get_arg_val<uint32_t>(TB_TX2X_ARG_PEER_VERIFY_ADDR_LO);
    const uint32_t peer_ver_hi = get_arg_val<uint32_t>(TB_TX2X_ARG_PEER_VERIFY_ADDR_HI);
    state.peer_noc = {get_arg_val<uint32_t>(TB_TX2X_ARG_PEER_NOC_X), get_arg_val<uint32_t>(TB_TX2X_ARG_PEER_NOC_Y)};

    state.peer_dst = (static_cast<uint64_t>(peer_dst_hi) << 32) | static_cast<uint64_t>(peer_dst_lo);
    state.peer_verify = (static_cast<uint64_t>(peer_ver_hi) << 32) | static_cast<uint64_t>(peer_ver_lo);

    // Refuse invalid arg combinations before touching the transport. A
    // device that trusts its args silently makes a defective host packing
    // look like a measurement.
    if (!sizes::contains(state.args.transfer_size)) {
        return false;
    }
    if (state.args.transfer_count == 0u) {
        return false;
    }
    if (state.own_staging_addr == 0u || state.own_scratch_addr == 0u || own_result == 0u) {
        return false;
    }
    if (state.peer_dst == 0u || state.peer_verify == 0u) {
        return false;
    }
    if (!x280_noc_coord_valid(state.peer_noc)) {
        return false;
    }
    if ((state.own_staging_addr % UBENCH_NOC_UNIT_BYTES) != 0u) {
        return false;
    }
    if ((state.peer_dst % UBENCH_NOC_UNIT_BYTES) != 0u) {
        return false;
    }
    if (state.args.boundary != TRANSPORT_BW_BOUNDARY_DRAINED &&
        state.args.boundary != TRANSPORT_BW_BOUNDARY_INJECTION) {
        return false;
    }

    state.staging32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state.own_staging_addr);
    state.scratch32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state.own_scratch_addr);
    state.r = reinterpret_cast<transport_bw_result_t*>(own_result);

    // Initialise the result region. Preserves the invariant that the host
    // reads DONE only after every field is populated.
    state.r->backend_id = TRANSPORT_BW_BACKEND_ID_TENSIX_TO_X280;
    state.r->boundary = state.args.boundary;
    state.r->terminal_state = static_cast<uint32_t>(TRANSPORT_BW_STATE_INIT);
    state.r->transfer_size = state.args.transfer_size;
    state.r->transfer_count = state.args.transfer_count;
    state.r->warmup_completed = 0u;
    state.r->logical_bytes = 0ull;
    state.r->raw_batch_delta = 0ull;
    state.r->secondary_batch_delta = 0ull;
    state.r->path_sentinel = 0u;
    state.r->reserved = 0u;

    // Fill the staging region with a distinguishable pattern so a receiver
    // that ever needs to be inspected has non-zero bytes to look at. Only
    // the header 4 bytes come from the sentinel; the rest use run_nonce so
    // two runs cannot leave identical residue.
    for (uint32_t off = 0u; off < state.args.transfer_size; off += sizeof(uint32_t)) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state.own_staging_addr + off);
        *p = state.args.run_nonce + (off / sizeof(uint32_t));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Frontend adapter
// ---------------------------------------------------------------------------
// verify_path and issue are always_inline because an out-of-line call that
// receives `s` lets it escape, and every NoC counter store in the timed loop
// then forces a reload of the transfer operands. issue is also flatten, so
// noc_async_write is never called out of line with the loop's registers
// spilled around the call.
struct Backend {
    using state_t = ::transport_bandwidth::backend_tensix_to_x280::state_t;
    using timestamp_t = ::transport_bandwidth::backend_tensix_to_x280::timestamp_t;
    using counter_delta_t = ::transport_bandwidth::backend_tensix_to_x280::counter_delta_t;
    using sizes = ::transport_bandwidth::backend_tensix_to_x280::sizes;

    static const transport_bw_protocol_args_t& args(state_t& s) { return s.args; }
    static transport_bw_result_t& result(state_t& s) { return *s.r; }

    // Verify path: one untimed write to the sentinel address, drain,
    // read-back via NOC read into L1 scratch, invalidate cache, compare.
    // Blackhole L1 caches every address and a peer's NOC write does NOT
    // invalidate the cache, so the invalidate before the load is
    // mandatory. Publishes r.path_sentinel as an observation.
    static inline __attribute__((always_inline)) bool verify_path(state_t& s) {
        const uint32_t sentinel = s.args.run_nonce ^ kVerifyMagic;
        *s.staging32 = sentinel;
        noc_async_write(s.own_staging_addr, NOC_XY_ADDR(s.peer_noc.x, s.peer_noc.y, s.peer_verify), sizeof(uint32_t));
        noc_async_write_barrier();

        *s.scratch32 = 0u;
        noc_async_read(NOC_XY_ADDR(s.peer_noc.x, s.peer_noc.y, s.peer_verify), s.own_scratch_addr, sizeof(uint32_t));
        noc_async_read_barrier();

        invalidate_l1_cache();
        const uint32_t observed = *s.scratch32;
        s.r->path_sentinel = observed;
        return observed == sentinel;
    }

    // Issue: one noc_async_write of TransferSize bytes. The size must reach
    // noc_async_write as the compile-time constant, not args.transfer_size:
    // with a runtime size the API cannot fold its `size <= NOC_MAX_BURST_SIZE`
    // test, so it emits the multi-packet split path and re-tests the bound on
    // every iteration of the timed loop. Both are dead weight here -- arg
    // validation admits only `sizes`, so the two are always equal. The
    // static_assert prevents a specialisation the args validation would
    // not have admitted.
    template <uint32_t TransferSize>
    static inline __attribute__((always_inline, flatten)) void issue(state_t& s) {
        static_assert(sizes::contains(TransferSize), "TransferSize is not in sizes");
        noc_async_write(s.own_staging_addr, NOC_XY_ADDR(s.peer_noc.x, s.peer_noc.y, s.peer_dst), TransferSize);
    }

    // Drain: barrier on outstanding writes.
    static void drain(state_t&) { noc_async_write_barrier(); }

    // Read wall then CPU outside the issued transfer loop. Each reader
    // preserves compiler ordering; the counters have independent origins.
    static timestamp_t measurement_now(state_t&) {
        timestamp_t stamp{UINT64_MAX, UINT64_MAX};
        if (!tensix_live_timer_read(TENSIX_TIMER_LOCAL_LOW_ADDR, 8, &stamp.ticks) ||
            !measurement_cpu_cycles(&stamp.cpu)) {
            return {UINT64_MAX, UINT64_MAX};
        }
        return stamp;
    }

    static counter_delta_t delta(state_t&, const timestamp_t& t0, const timestamp_t& t1) {
        counter_delta_t d;
        if (t0.ticks == UINT64_MAX || t1.ticks == UINT64_MAX || !measurement_delta(t0.ticks, t1.ticks, &d.primary)) {
            d.primary = UINT64_MAX;
        }
        if (t0.cpu == UINT64_MAX || t1.cpu == UINT64_MAX || !measurement_delta(t0.cpu, t1.cpu, &d.secondary)) {
            d.secondary = UINT64_MAX;
        }
        return d;
    }

    static void publish_delta(state_t& s, const counter_delta_t& d) {
        s.r->raw_batch_delta = d.primary;
        s.r->secondary_batch_delta = d.secondary;
    }

    // Blackhole L1 is write-through; program-order stores are visible to
    // the host after Finish. No fence needed.
    static void publish_barrier(state_t&) {}

    static bool publish_footer(state_t&) { return true; }

    static constexpr uint32_t BACKEND_ID = TRANSPORT_BW_BACKEND_ID_TENSIX_TO_X280;
};

}  // namespace transport_bandwidth::backend_tensix_to_x280

#endif  // X280_TRANSPORT_BW_BACKEND_TENSIX_TO_X280_HPP
