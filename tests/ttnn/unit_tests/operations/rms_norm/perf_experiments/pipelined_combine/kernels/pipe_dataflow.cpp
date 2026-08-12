// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH — idea I9: make the rms_norm cross-core combine's RENDEZVOUS
// cheaper and pipelined. Not production code; never dispatched by the op.
//
// Every core of a reduction rectangle starts with `rows_t` bf16 stat tiles resident in
// L1 (a pinned HEIGHT shard). The group must land rsqrt(sum + eps) on every member.
// Only the two-level gather + the root's one return multicast are reconstructed; the
// statistics pipeline, the reader and the apply pass are out of frame.
//
// MODES
//   0 baseline   the op's CURRENT rendezvous: per-contributor
//                  noc_async_write(data) ; noc_async_write_barrier() ; semaphore.up
//                and an ALL-OR-NOTHING wait on the receiving side
//                  sem.wait_min(fan_in - 1) ; cb_push_back(all slots)
//                at BOTH levels, then root finalize + one return multicast.
//   5 flag       CHEAPER DELIVERY ONLY (same all-or-nothing barrier):
//                  set_async_write_state once per destination, hoisted out of the
//                  block loop, + async_write_with_state per tile, and the arrival
//                  SIGNAL is a monotone inline_dw_write of (b+1) into a per-sender
//                  flag word instead of a NoC atomic increment. One fewer NoC
//                  transaction class per contributor and no per-transfer command
//                  re-programming; the receiver polls S flag words instead of one
//                  semaphore.
//   6 incr       DELIVERY (as 5) + PIPELINED ACCUMULATION: the receiver no longer
//                waits for the whole fan-in. It pushes each contributor's slot into
//                cb_stat_gather the moment that contributor's flag is up, and the
//                compute chain consumes with WaitPolicy::Cumulative, i.e. it starts
//                adding partial k while partials k+1.. are still in flight. Same at
//                level 2. Requires rows_t == 1 (see the host guard): the gather CB is
//                laid out (r * fan_in + slot), so a per-slot incremental push is only
//                front-contiguous when there is one tile-row.
//
// SKEW. In the real op contributors do NOT arrive together — the round-1 zone data
// shows cp_wait_rstd min 735 / max 4371 ns, a ~3.6 us arrival spread caused by unequal
// pre-combine work. A bench where every core starts at once would make ANY pipelining
// idea look like a null for the wrong reason, so each core busy-waits a deterministic
// per-core `skew_iters` before contributing. The pattern is identical in every variant.
//
// RAW-NoC SUBSTITUTIONS AND WHY (same justification as the round-1 combine bench)
//   * The gather legs are raw NoC writes + a hand-rolled signal rather than
//     mcast_pipe's SenderPipe/ReceiverPipe: mcast_pipe is a one-to-many broadcast of
//     ONE buffer whose stated precondition is "single sender per receiver, dst_l1
//     identical on all receivers" (mcast_pipe.hpp:44-45). A gather is many senders into
//     DISJOINT slots of one receiver — the opposite direction — so the helper cannot
//     express it at all.
//   * The arrival signal is a monotone flag WORD (single writer, host-zeroed, compared
//     against b+1) rather than `Semaphore::up`. Semaphore<> only exposes a counter, and
//     a counter cannot say WHICH contributors have landed — which is exactly what
//     incremental accumulation needs. The transfer itself still goes through the `Noc`
//     object (inline_dw_write / async_write_with_state); only the handshake is
//     hand-rolled.
//   * The return multicast is unchanged mcast_pipe in every mode.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

namespace {
constexpr uint32_t cb_zero_tile = 4;     // pinned, host-zeroed: page 0 = identity tile, page 1 = flags
constexpr uint32_t cb_stat_partial = 7;  // pinned input shard
constexpr uint32_t cb_stat_gather = 8;   // level-1 landing (row leader)
constexpr uint32_t cb_stat_sum = 9;
constexpr uint32_t cb_rstd_send = 10;     // the root's multicast source
constexpr uint32_t cb_stat_gather2 = 15;  // level-2 landing (root)
constexpr uint32_t cb_out = 16;           // pinned output shard
constexpr uint32_t cb_branch_sum = 18;    // a leader's level-1 result

constexpr uint32_t FLAG_L1 = 0;    // flag words [0, S1)   — level-1 arrivals, indexed by slot
constexpr uint32_t FLAG_L2 = 128;  // flag words [128, ..) — level-2 arrivals, indexed by grid row
}  // namespace

using namespace dataflow_kernel_lib;

namespace {
FORCE_INLINE void wait_flag(volatile tt_l1_ptr uint32_t* w, uint32_t target) {
    while (true) {
        invalidate_l1_cache();
        if (*w >= target) {
            return;
        }
    }
}
// Arrival bit j of the semaphore word (MODE 7): every contributor atomically adds
// 1 << its_slot, so the single word carries per-contributor identity.
FORCE_INLINE void wait_bit(volatile tt_l1_ptr uint32_t* w, uint32_t j) {
    const uint32_t bit = 1u << j;
    while (true) {
        invalidate_l1_cache();
        if ((*w) & bit) {
            return;
        }
    }
}
FORCE_INLINE void wait_flags(volatile tt_l1_ptr uint32_t* flags, uint32_t n, uint32_t target) {
    for (uint32_t j = 0; j < n; ++j) {
        wait_flag(flags + j, target);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t S1 = get_compile_time_arg_val(1);  // level-1 fan-in
    constexpr uint32_t S2 = get_compile_time_arg_val(2);  // level-2 fan-in; 1 == flat tree
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(4);
    constexpr uint32_t SEM_GATHER2 = get_compile_time_arg_val(5);
    constexpr uint32_t MCAST_CT_BASE = 7;
    constexpr uint32_t MCAST_RT_BASE = 15;
    constexpr auto mc = McastArgs<MCAST_CT_BASE, MCAST_RT_BASE>();
    constexpr bool TWO_STAGE = S2 > 1;
    constexpr bool FLAGS = (MODE == 5) || (MODE == 6);  // flag-word delivery
    constexpr bool BITSET = MODE == 7;                  // atomic bitmask delivery
    constexpr bool INCR = (MODE == 6) || (MODE == 7);   // pipelined accumulation
    static_assert(!BITSET || (S1 <= 32 && S2 <= 32), "the arrival bitmask holds 32 slots");

    const uint32_t rows_t = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t my_slot = get_arg_val<uint32_t>(2);
    const uint32_t is_leader = get_arg_val<uint32_t>(3);
    const uint32_t is_root = get_arg_val<uint32_t>(4);
    const uint32_t my_row_slot = get_arg_val<uint32_t>(5);
    const uint32_t skew_iters = get_arg_val<uint32_t>(6);
    const uint32_t leader_x = get_arg_val<uint32_t>(7);
    const uint32_t leader_y = get_arg_val<uint32_t>(8);
    const uint32_t root_x = get_arg_val<uint32_t>(9);
    const uint32_t root_y = get_arg_val<uint32_t>(10);
    const uint32_t box_x0 = get_arg_val<uint32_t>(11);
    const uint32_t box_y0 = get_arg_val<uint32_t>(12);
    const uint32_t box_x1 = get_arg_val<uint32_t>(13);
    const uint32_t box_y1 = get_arg_val<uint32_t>(14);

    Noc noc;
    [[maybe_unused]] Semaphore<> gather_sem(SEM_GATHER);
    [[maybe_unused]] Semaphore<> gather2_sem(SEM_GATHER2);

    const uint32_t tb = get_tile_size(cb_stat_partial);
    const uint32_t partial_base = get_read_ptr(cb_stat_partial);
    const uint32_t gather_base = get_write_ptr(cb_stat_gather);
    [[maybe_unused]] const uint32_t gather2_base = get_write_ptr(cb_stat_gather2);
    const uint32_t flags_base = get_read_ptr(cb_zero_tile) + tb;
    volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags_base);

    cb_push_back(cb_zero_tile, 1);  // the combine's identity operand (host-zeroed page 0)

    [[maybe_unused]] auto sender = mc.sender(noc);
    [[maybe_unused]] auto receiver = mc.receiver(noc);

    [[maybe_unused]] UnicastEndpoint uni;
    // Level-1 destination: my slot in the leader's gather CB. Under FLAGS the write
    // command is programmed ONCE here and replayed per tile / per block.
    // programmed once per block (the level-2 write and the return multicast reprogram
    // the write command buffer, so the state cannot be hoisted out of the block loop).
    const uint32_t l1_dst = gather_base + my_slot * tb;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t src = partial_base + b * rows_t * tb;

        // Per-core arrival SKEW — models the op's unequal pre-combine work. Identical
        // in every variant; it is what makes a pipelining idea measurable at all.
        for (volatile uint32_t i = 0; i < skew_iters; ++i) {
        }

        // ---- level 1: every member -> its grid row's leader --------------------
        if constexpr (S1 > 1) {
            if constexpr (FLAGS) {
                noc.set_async_write_state(uni, tb, {leader_x, leader_y, l1_dst});
                for (uint32_t r = 0; r < rows_t; ++r) {
                    noc.async_write_with_state(
                        uni, uni, tb, {0, 0, src + r * tb}, {leader_x, leader_y, l1_dst + r * S1 * tb});
                }
                noc.async_write_barrier();
                if (is_leader) {
                    flags[FLAG_L1 + my_slot] = b + 1;
                } else {
                    noc.inline_dw_write<NocOptions::INLINE_L1>(
                        uni, b + 1, {leader_x, leader_y, flags_base + 4 * (FLAG_L1 + my_slot)});
                }
            } else {
                for (uint32_t r = 0; r < rows_t; ++r) {
                    const uint32_t dst = gather_base + (r * S1 + my_slot) * tb;
                    noc_async_write(src + r * tb, get_noc_addr(leader_x, leader_y, dst), tb);
                }
                noc_async_write_barrier();
                if constexpr (BITSET) {
                    // The op's SAME atomic-increment signal, but the increment is
                    // 1 << my_slot instead of 1: the receiver's one semaphore word then
                    // says WHICH contributors have landed, not merely how many — which
                    // is what incremental release needs. Delivery cost is unchanged
                    // (one NoC atomic, exactly as the baseline).
                    // Loopback atomic for the leader too: a LOCAL read-modify-write
                    // could lose a concurrent peer's remote atomic.
                    gather_sem.up(noc, leader_x, leader_y, 1u << my_slot);
                } else if (!is_leader) {
                    gather_sem.up(noc, leader_x, leader_y, 1);
                }
            }
        } else {
            // Degenerate fan-in: the "gather" is a local copy into my own slot.
            for (uint32_t r = 0; r < rows_t; ++r) {
                noc_async_write(src + r * tb, get_noc_addr(leader_x, leader_y, gather_base + r * tb), tb);
            }
            noc_async_write_barrier();
        }

        if (is_leader) {
            cb_reserve_back(cb_stat_gather, rows_t * S1);
            if constexpr (S1 > 1) {
                if constexpr (BITSET) {
                    // PIPELINED over the atomic bitmask: release slot j the moment its
                    // bit is set. Reset for the next block is safe here — a peer cannot
                    // start block b+1 before the return multicast, which is still ahead.
                    volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore<ProgrammableCoreType::TENSIX>(SEM_GATHER));
                    for (uint32_t j = 0; j < S1; ++j) {
                        wait_bit(mask, j);
                        cb_push_back(cb_stat_gather, rows_t);
                    }
                    gather_sem.set(0);
                } else if constexpr (INCR) {
                    // PIPELINED: release each contributor's slot as it lands, so the
                    // combine chain's adds overlap the remaining arrivals.
                    for (uint32_t j = 0; j < S1; ++j) {
                        wait_flag(flags + FLAG_L1 + j, b + 1);
                        cb_push_back(cb_stat_gather, rows_t);
                    }
                } else if constexpr (FLAGS) {
                    wait_flags(flags + FLAG_L1, S1, b + 1);
                    cb_push_back(cb_stat_gather, rows_t * S1);
                } else {
                    gather_sem.wait_min((b + 1) * (S1 - 1));
                    cb_push_back(cb_stat_gather, rows_t * S1);
                }
            } else {
                cb_push_back(cb_stat_gather, rows_t * S1);
            }
        }

        // ---- level 2: every leader -> the root --------------------------------
        if constexpr (TWO_STAGE) {
            if (is_leader) {
                cb_wait_front(cb_branch_sum, rows_t);
                const uint32_t bsrc = get_read_ptr(cb_branch_sum);
                for (uint32_t r = 0; r < rows_t; ++r) {
                    const uint32_t dst = gather2_base + (r * S2 + my_row_slot) * tb;
                    noc_async_write(bsrc + r * tb, get_noc_addr(root_x, root_y, dst), tb);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_branch_sum, rows_t);
                if constexpr (FLAGS) {
                    if (is_root) {
                        flags[FLAG_L2 + my_row_slot] = b + 1;
                    } else {
                        noc.inline_dw_write<NocOptions::INLINE_L1>(
                            uni, b + 1, {root_x, root_y, flags_base + 4 * (FLAG_L2 + my_row_slot)});
                    }
                } else if constexpr (BITSET) {
                    gather2_sem.up(noc, root_x, root_y, 1u << my_row_slot);
                } else if (!is_root) {
                    gather2_sem.up(noc, root_x, root_y, 1);
                }
                if (is_root) {
                    cb_reserve_back(cb_stat_gather2, rows_t * S2);
                    if constexpr (BITSET) {
                        volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            get_semaphore<ProgrammableCoreType::TENSIX>(SEM_GATHER2));
                        for (uint32_t j = 0; j < S2; ++j) {
                            wait_bit(mask, j);
                            cb_push_back(cb_stat_gather2, rows_t);
                        }
                        gather2_sem.set(0);
                    } else if constexpr (INCR) {
                        for (uint32_t j = 0; j < S2; ++j) {
                            wait_flag(flags + FLAG_L2 + j, b + 1);
                            cb_push_back(cb_stat_gather2, rows_t);
                        }
                    } else if constexpr (FLAGS) {
                        wait_flags(flags + FLAG_L2, S2, b + 1);
                        cb_push_back(cb_stat_gather2, rows_t * S2);
                    } else {
                        gather2_sem.wait_min((b + 1) * (S2 - 1));
                        cb_push_back(cb_stat_gather2, rows_t * S2);
                    }
                }
            }
        }

        // ---- the root's ONE return broadcast of the finished rstd -------------
        cb_reserve_back(cb_out, rows_t);
        const uint32_t land_dst = get_write_ptr(cb_out);
        if (is_root) {
            cb_wait_front(cb_rstd_send, rows_t);
            sender.send(get_read_ptr(cb_rstd_send), land_dst, rows_t * tb);
            cb_pop_front(cb_rstd_send, rows_t);
        } else {
            receiver.receive();
        }
        cb_push_back(cb_out, rows_t);

        cb_wait_front(cb_out, rows_t);
        cb_pop_front(cb_out, rows_t);
    }
}
