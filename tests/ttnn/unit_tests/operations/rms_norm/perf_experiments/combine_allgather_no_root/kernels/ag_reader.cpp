// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (NCRISC / NoC0) — op_design.md §4.1 and §4.2.
//
// Per core: a disjoint range of tile-rows AND (under the cross-core W-split) a
// disjoint W-slice of them, looped in row-blocks of HT_BLOCK tile-rows.  Each
// row-block is read as NW chunks of WT_CHUNK W-tiles; the number of reader
// passes over a row-block is the ONE structural difference between the two
// residency regimes:
//
//   X_RESIDENT     -> 1 pass  (whole HT_BLOCK x Wt strip stays in L1)
//   streaming       -> 2 passes (pass A feeds the reduce, pass B feeds the scale)
//
// Every count below is a function of the block knobs (HT_BLOCK / WT_CHUNK / NW)
// — never of a whole-op dimension.
//
// TILE input is read with whole-tile pages through a TensorAccessor and one
// barrier per chunk (batched, coalescing the chunk into a single NoC burst
// train).  ROW_MAJOR input goes through
// dataflow_kernel_lib::read_sticks_for_tilize, whose byte_offset_within_page
// argument IS the WT_CHUNK knob on the read side.
//
// SHARDED input (SHARDED_IN): there is NO read at all.  cb_input_tiles is
// placed directly on the core's own L1 shard by the host
// (ttnn.cb_descriptor_from_sharded_tensor), so the reader only has to make the
// already-resident data visible to compute — one reserve/push of the whole
// shard — and zero any trailing W-padding tiles the shard grid over-covers.
//
// CROSS-CORE W-SPLIT (W_SPLIT): this kernel owns the *receive* half of the
// combine.  Per row-block a LEADER waits for all CW1 partial-sum tiles of its
// stage-1 fan-in to land in cb_group_partials (the writers push them; see
// rms_norm_writer.cpp) and hands them to compute.  With CW2 == 1 the only leader
// is the group root and that single gather IS the combine (the flat topology).
// With CW2 > 1 the combine is TWO-STAGE: each grid row's leader folds its row,
// and the root then waits for the CW2 row-sums in cb_group_partials2 — the
// serial fan-in is CW1 + CW2 instead of CW1 * CW2.  The root finally multicasts
// compute's combined mean(x^2) back over the group rectangle; every other core
// waits for that broadcast.  The broadcast goes through
// kernel_lib/mcast_pipe.hpp (SenderPipe / ReceiverPipe) — the multicast, the
// readiness handshake and the per-NoC rectangle corner ordering are the
// helper's, never hand-rolled here.
//
// PER-STAGE INSTRUMENTATION (permanent, Perf 1). Stage boundaries carry a
// MaybeDeviceZoneScope: rdr_scaler / rdr_gamma_resident / rdr_shard_publish /
// rdr_read_a / rdr_gather_wait / rdr_mcast / rdr_read_b. Free when the profiler
// is off (perf_instrumentation.hpp's durability contract) — never remove one,
// and extend the set to any new predicate-guarded path.
//
// Helper substitution note: the TILE path uses TensorAccessor +
// noc_async_read_page directly because no dataflow helper can express a
// tile-page read of an already-tiled tensor (read_sticks_for_tilize is
// stick-indexed and feeds the tilize helper). op_design.md §7 mandates exactly
// this.  The gather leg likewise uses raw noc_async_write + noc_semaphore_inc:
// mcast_pipe covers one-to-many broadcast, and there is no many-to-one helper
// (references/cross_core_reduction_design.md §1 "Reach for the raw primitives
// only for the gather leg, which the pipe does not cover").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_input_rm = 3;
constexpr uint32_t cb_gamma_rm = 4;
constexpr uint32_t cb_ones = 5;
constexpr uint32_t cb_group_partials = 6;
constexpr uint32_t cb_rms_mean = 7;
// LAB: allgather_pull consumes cb_partial_out directly (the writer's gather_hop
// is skipped entirely for this variant -- see rms_norm_writer.cpp), so this
// kernel now needs the index the baseline reader never touched.
constexpr uint32_t cb_partial_out = 8;
constexpr uint32_t cb_group_partials2 = 9;
constexpr uint32_t cb_rms_sum = 26;
// LAB: new semaphore id for the all-to-all readiness counter (see
// lab_descriptor.py's SEM_ALLGATHER -- ids 0..9 are all spoken for by the
// op's own combine + gamma broadcast, so 10 is free).
constexpr uint32_t SEM_ALLGATHER = 10;
}  // namespace

namespace dkl = dataflow_kernel_lib;

void kernel_main() {
    // ---- regime flags (§5.2) ----
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(1) != 0;
    constexpr bool IS_RM_GAMMA = get_compile_time_arg_val(2) != 0;
    constexpr bool X_RESIDENT = get_compile_time_arg_val(3) != 0;
    constexpr bool GAMMA_RESIDENT = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(5) != 0;
    // ---- block knobs (§1.2) ----
    constexpr uint32_t WT = get_compile_time_arg_val(6);  // W-tiles THIS core owns
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t WT_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t NW = get_compile_time_arg_val(9);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(10);
    // W-chunks coalesced into ONE reserve/barrier/push on the resident TILE
    // path. 1 = pipeline read with compute (latency-bound, few cores);
    // NW = one transfer per row-block (throughput-bound, grid full). Host-side
    // rationale + measurements: rms_norm_program_descriptor._x_read_chunks.
    constexpr uint32_t X_READ_CHUNKS = get_compile_time_arg_val(11);
    // ---- geometry ----
    constexpr uint32_t W_VALID_LAST = get_compile_time_arg_val(12);
    constexpr uint32_t CHUNK_ROW_BYTES = get_compile_time_arg_val(13);
    constexpr uint32_t LAST_ROW_BYTES = get_compile_time_arg_val(14);
    constexpr uint32_t G_CHUNK_ROW_BYTES = get_compile_time_arg_val(15);
    constexpr uint32_t G_LAST_ROW_BYTES = get_compile_time_arg_val(16);
    constexpr uint32_t TOTAL_STICKS = get_compile_time_arg_val(17);
    // ---- cross-core W-split (§4.2) ----
    constexpr bool W_SPLIT = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t CW = get_compile_time_arg_val(19);         // cores per combine group
    constexpr uint32_t WT_STRIDE = get_compile_time_arg_val(20);  // whole-tensor Wt
    constexpr bool SHARDED_IN = get_compile_time_arg_val(21) != 0;
    constexpr bool SHARDED_OUT = get_compile_time_arg_val(22) != 0;
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(23);
    // Combine topology: CW1 * CW2 == CW. CW2 == 1 -> flat root gather (the only
    // leader is the root); CW2 > 1 -> two-stage (row leaders, then the root).
    constexpr uint32_t CW1 = get_compile_time_arg_val(24);
    constexpr uint32_t CW2 = get_compile_time_arg_val(25);
    constexpr uint32_t SEM_GATHER2 = get_compile_time_arg_val(26);
    constexpr bool TWO_STAGE = CW2 > 1;
    static_assert(CW1 * CW2 == CW, "combine stages must tile CW");
    (void)SHARDED_OUT;

    // ONE multicast family per virtually-contiguous column run of the group. A
    // NoC multicast addresses a VIRTUAL rectangle, and the logical compute grid
    // is not virtually contiguous (blackhole_p150b: logical x 0..6 -> virtual
    // 1..7, logical x 7..10 -> virtual 10..13), so a group that spans the seam
    // must broadcast as two rectangles rather than one bounding box that would
    // target non-worker endpoints. Family B is inactive (`active == 0`) whenever
    // the group fits in one run, and then compiles away entirely.
    constexpr auto mc_a = dkl::McastArgs</*CT=*/27, /*RT=*/11>();
    constexpr auto mc_b = dkl::McastArgs<mc_a.next_compile_time_args_offset(), mc_a.next_runtime_args_offset()>();

    // ---- Perf 1: the reuse-shared gamma broadcast (op_design Lamp L2) -------
    // Flag word (bit0) plus one multicast family per virtually-contiguous column
    // run. The host owns the predicate — see _gamma_mcast_plan; when it refuses,
    // GAMMA_MCAST is 0 and every leg below compiles away, leaving the per-core
    // TensorAccessor read byte-identical to the pre-Perf-1 spelling.
    constexpr uint32_t GAMMA_FLAGS = get_compile_time_arg_val(mc_b.next_compile_time_args_offset());
    constexpr bool GAMMA_MCAST = (GAMMA_FLAGS & 0x1u) != 0;
    constexpr auto gmc_a =
        dkl::McastArgs<mc_b.next_compile_time_args_offset() + 1, mc_b.next_runtime_args_offset() + 2>();
    constexpr auto gmc_b = dkl::McastArgs<gmc_a.next_compile_time_args_offset(), gmc_a.next_runtime_args_offset()>();
    constexpr auto in_args = TensorAccessorArgs<gmc_b.next_compile_time_args_offset()>();
    static_assert(gmc_b.next_compile_time_args_offset() == 48, "shared dataflow CT block must be 48 words");
    constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    // ---- LAB: combine_allgather_no_root -----------------------------------
    // COMBINE_VARIANT selects the combine's transport: 0 = the op's own
    // gather-to-root + Mcast2D-back (byte-identical baseline); 1 =
    // allgather_pull (this idea) -- delete the round trip instead of shrinking
    // it. Appended AFTER every TensorAccessorArgs block so their own
    // next_compile_time_args_offset() chaining is untouched.
    constexpr uint32_t COMBINE_VARIANT = get_compile_time_arg_val(gamma_args.next_compile_time_args_offset());
    constexpr bool ALLGATHER_PULL = COMBINE_VARIANT == 1;
    // Fixed RT layout emitted by the host (create_program_descriptor): 11 base
    // args + mcast_rt(4*MAX_MCAST_FAMILIES) + [g_inject, g_family] +
    // gamma_rt(4*MAX_MCAST_FAMILIES) = 29 words, THEN the peer list (2*CW
    // words, only when W_SPLIT). See lab_descriptor.py's reader_rt assembly.
    constexpr uint32_t PEER_RT_BASE = 29;

    static_assert(WT_LAST == WT_CHUNK, "reader assumes uniform chunk widths");
    static_assert(X_READ_CHUNKS >= 1 && NW % X_READ_CHUNKS == 0, "read batch must tile NW");

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(3);
    const uint32_t wt_start = get_arg_val<uint32_t>(4);
    // LAB: unused under allgather_pull (no root/leader distinction, no mcast
    // receive), hence [[maybe_unused]].
    [[maybe_unused]] const uint32_t is_root = get_arg_val<uint32_t>(6);
    const uint32_t is_last_w_core = get_arg_val<uint32_t>(7);
    const uint32_t wt_real = get_arg_val<uint32_t>(8);
    [[maybe_unused]] const uint32_t mcast_family = get_arg_val<uint32_t>(9);
    [[maybe_unused]] const uint32_t is_leader = get_arg_val<uint32_t>(10);
    // Perf 1: gamma-broadcast role. `gamma_inject` is a per-family bitmask ("I
    // read gamma from DRAM and broadcast it to that family"); `gamma_family` is
    // the rectangle this core receives in.
    [[maybe_unused]] const uint32_t gamma_inject = get_arg_val<uint32_t>(mc_b.next_runtime_args_offset() + 0);
    [[maybe_unused]] const uint32_t gamma_family = get_arg_val<uint32_t>(mc_b.next_runtime_args_offset() + 1);

    // Filler core: inside the multicast rectangle so the broadcast lands
    // somewhere legal, but it owns no data and takes no part in the combine.
    if (num_tile_rows == 0) {
        return;
    }

    // ---- LAB: allgather_pull's peer list (virtual coords, slot order) ------
    // A flat all-to-all list, not a tree: every real core in the group reads
    // this SAME list (built host-side from the group's "slots"), including its
    // own slot (a harmless same-core NoC loopback -- gather_hop already relies
    // on the identical trick when a worker's leader is itself).
    uint32_t peer_vx[CW];
    uint32_t peer_vy[CW];
    if constexpr (W_SPLIT && ALLGATHER_PULL) {
        for (uint32_t i = 0; i < CW; ++i) {
            peer_vx[i] = get_arg_val<uint32_t>(PEER_RT_BASE + 2 * i);
            peer_vy[i] = get_arg_val<uint32_t>(PEER_RT_BASE + 2 * i + 1);
        }
    }

    // ---- 1. scaler / partial-W mask: one tile, pushed once, never popped ----
    // Under a W-split only the core whose slice ends on the tensor's last
    // W-tile actually consumes the mask; the others push it and ignore it
    // (AccumulateViaAdd leaves the scaler CB unused without a partial).
    {
        MaybeDeviceZoneScope("rdr_scaler");
        if constexpr (HAS_PARTIAL_W) {
            dkl::prepare_reduce_mask<cb_scaler, ckernel::ReduceDim::REDUCE_ROW>(W_VALID_LAST);
        } else {
            dkl::calculate_and_prepare_reduce_scaler<
                cb_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW>();
        }
        if constexpr (W_SPLIT) {
            // Float32 unit scaler for the root's combine reduce over the CW
            // gathered partials (its input CB is Float32, so its scaler must be).
            dkl::calculate_and_prepare_reduce_scaler<cb_ones, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
        }
    }

    const auto in_acc = TensorAccessor(in_args, src_addr);
    [[maybe_unused]] const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);
    const uint32_t in_tile_bytes = get_tile_size(cb_input_tiles);

    // ---- 2. gamma: reuse-shared operand, read once when it fits L1 (§1.1) ----
    // Under a W-split each core owns a DISJOINT gamma slice [wt_start, +WT), so
    // gamma is read exactly once across the whole grid. Page indices are clamped
    // to the tensor's last tile so a shard that over-covers W never reads OOB.
    // wt_start is always a whole number of chunks (a core's slice is NW*WT_CHUNK
    // wide), so a chunk's byte offset into the full-W stick is just a multiple
    // of the chunk's row bytes.
    const uint32_t chunk0 = wt_start / WT_CHUNK;
    auto gamma_page = [&](uint32_t t) {
        const uint32_t p = wt_start + t;
        return (p < WT_STRIDE) ? p : (WT_STRIDE - 1);
    };
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT) {
        MaybeDeviceZoneScope("rdr_gamma_resident");
        if constexpr (IS_RM_GAMMA) {
            for (uint32_t wc = 0; wc < NW; ++wc) {
                const uint32_t rb = (is_last_w_core && wc + 1 == NW) ? G_LAST_ROW_BYTES : G_CHUNK_ROW_BYTES;
                dkl::read_sticks_for_tilize<cb_gamma_rm, dkl::TilizeGranularity::ROW>(
                    gamma_acc, /*total_num_rows=*/1, rb, /*start_page=*/0, (chunk0 + wc) * G_CHUNK_ROW_BYTES);
            }
        } else {
            const uint32_t gt = get_tile_size(cb_gamma);
            cb_reserve_back(cb_gamma, WT);
            const uint32_t base = get_write_ptr(cb_gamma);
            if constexpr (GAMMA_MCAST) {
                // Perf 1 — the reuse-shared broadcast. `cb_gamma` is declared at
                // the identical size on every core of the rectangle, so its write
                // pointer is the SAME L1 address everywhere: the injector sends
                // `src == dst` (its own copy is already in place, so the pipe
                // picks EXCLUDE-source) and every receiver's landing address is
                // right by construction. Delivery stays in the PROLOGUE: measured
                // 1.09x SLOWER when moved after the first pass-A read, because the
                // prologue is the one window where the injector's own DRAM read is
                // uncontended. The rectangle, the corner ordering and the
                // readiness handshake are mcast_pipe's, exactly as for 1/rms.
                Noc gnoc;
                if (gamma_inject) {
                    uint32_t addr = base;
                    for (uint32_t t = 0; t < WT; ++t) {
                        noc_async_read_page(gamma_page(t), gamma_acc, addr);
                        addr += gt;
                    }
                    noc_async_read_barrier();
                    if constexpr (gmc_a.active) {
                        if (gamma_inject & 0x1u) {
                            auto pipe = gmc_a.sender(gnoc);
                            pipe.send(base, base, WT * gt);
                        }
                    }
                    if constexpr (gmc_b.active) {
                        if (gamma_inject & 0x2u) {
                            auto pipe = gmc_b.sender(gnoc);
                            pipe.send(base, base, WT * gt);
                        }
                    }
                } else if (gamma_family == 0) {
                    auto pipe = gmc_a.receiver(gnoc);
                    pipe.receive();
                } else {
                    auto pipe = gmc_b.receiver(gnoc);
                    pipe.receive();
                }
            } else {
                uint32_t addr = base;
                for (uint32_t t = 0; t < WT; ++t) {
                    noc_async_read_page(gamma_page(t), gamma_acc, addr);
                    addr += gt;
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_gamma, WT);
        }
    }

    // ---- per-chunk readers -------------------------------------------------

    // TILE: `batch` W-chunks of ht x WT_CHUNK whole tile pages, one barrier for
    // the whole batch. `batch > 1` requires HT_BLOCK == 1 (guaranteed by R7
    // whenever NW > 1), so the tiles land in flat column order and the resident
    // strip stays row-major.
    auto read_input_batch_tile = [&](uint32_t wc0, uint32_t batch, uint32_t row0, uint32_t ht) {
        const uint32_t n = ht * WT_CHUNK * batch;
        cb_reserve_back(cb_input_tiles, n);
        uint32_t addr = get_write_ptr(cb_input_tiles);
        for (uint32_t k = 0; k < batch; ++k) {
            for (uint32_t h = 0; h < ht; ++h) {
                const uint32_t base_tile = (row0 + h) * WT_STRIDE + wt_start + (wc0 + k) * WT_CHUNK;
                for (uint32_t t = 0; t < WT_CHUNK; ++t) {
                    noc_async_read_page(base_tile + t, in_acc, addr);
                    addr += in_tile_bytes;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, n);
    };

    // ROW_MAJOR: one row page per stick; `valid_rows` clamps the read to the
    // sticks that actually exist (non-tile-aligned H), and the missing rows of
    // the tile-row block are pushed unread so the tilize helper always consumes
    // whole 32-row blocks (their stale content lands in H-padding rows the
    // writer never writes back).
    auto read_input_chunk_rm = [&](uint32_t wc, uint32_t row0, uint32_t ht, uint32_t valid_rows) {
        const uint32_t rb = (is_last_w_core && wc + 1 == NW) ? LAST_ROW_BYTES : CHUNK_ROW_BYTES;
        dkl::read_sticks_for_tilize<cb_input_rm, dkl::TilizeGranularity::ROW>(
            in_acc, valid_rows, rb, row0 * 32u, (chunk0 + wc) * CHUNK_ROW_BYTES);
        const uint32_t pad_rows = ht * 32u - valid_rows;
        if (pad_rows != 0) {
            cb_reserve_back(cb_input_rm, pad_rows);
            cb_push_back(cb_input_rm, pad_rows);
        }
    };

    // One pass over a row-block's whole W. On the resident TILE path the reads
    // are grouped X_READ_CHUNKS at a time; every other path is per-chunk (the
    // streaming input CB holds only X_DEPTH*B pages, and the RM path's
    // cb_input_tiles is produced by compute's tilize, not by the reader).
    auto read_input_pass = [&](uint32_t row0, uint32_t ht, uint32_t valid_rows) {
        if constexpr (IS_RM) {
            for (uint32_t wc = 0; wc < NW; ++wc) {
                read_input_chunk_rm(wc, row0, ht, valid_rows);
            }
        } else {
            for (uint32_t wc = 0; wc < NW; wc += X_READ_CHUNKS) {
                read_input_batch_tile(wc, X_READ_CHUNKS, row0, ht);
            }
        }
    };

    auto read_gamma_chunk = [&](uint32_t wc) {
        if constexpr (IS_RM_GAMMA) {
            const uint32_t rb = (is_last_w_core && wc + 1 == NW) ? G_LAST_ROW_BYTES : G_CHUNK_ROW_BYTES;
            dkl::read_sticks_for_tilize<cb_gamma_rm, dkl::TilizeGranularity::ROW>(
                gamma_acc, /*total_num_rows=*/1, rb, /*start_page=*/0, (chunk0 + wc) * G_CHUNK_ROW_BYTES);
        } else {
            const uint32_t gt = get_tile_size(cb_gamma);
            cb_reserve_back(cb_gamma, WT_CHUNK);
            uint32_t addr = get_write_ptr(cb_gamma);
            for (uint32_t t = 0; t < WT_CHUNK; ++t) {
                noc_async_read_page(gamma_page(wc * WT_CHUNK + t), gamma_acc, addr);
                addr += gt;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, WT_CHUNK);
        }
    };

    // ---- 2b. sharded input: the shard IS the block, already in L1 ----------
    if constexpr (SHARDED_IN) {
        MaybeDeviceZoneScope("rdr_shard_publish");
        const uint32_t shard_tiles = num_tile_rows * WT;
        cb_reserve_back(cb_input_tiles, shard_tiles);
        // A shard grid may over-cover W (auto_shard_config pads the last core).
        // Those trailing tiles are uninitialized L1, so zero them before they
        // reach the reduce — n_reduced stays the true element count W.
        if (wt_real < WT) {
            const uint32_t base = get_write_ptr(cb_input_tiles);
            const uint32_t words = in_tile_bytes >> 2;
            for (uint32_t h = 0; h < num_tile_rows; ++h) {
                for (uint32_t t = wt_real; t < WT; ++t) {
                    volatile tt_l1_ptr uint32_t* p =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + (h * WT + t) * in_tile_bytes);
                    for (uint32_t i = 0; i < words; ++i) {
                        p[i] = 0;
                    }
                }
            }
        }
        cb_push_back(cb_input_tiles, shard_tiles);
    }

    // ---- 2c. cross-core combine: the receive half -------------------------
    [[maybe_unused]] const uint32_t fp32_tile_bytes = get_tile_size(cb_rms_sum);
    [[maybe_unused]] Noc noc;
    [[maybe_unused]] Semaphore<> gather_sem(SEM_GATHER);
    [[maybe_unused]] Semaphore<> gather2_sem(SEM_GATHER2);

    // ---- 3. row-block loop -------------------------------------------------
    const uint32_t num_row_blocks = (num_tile_rows + HT_BLOCK - 1) / HT_BLOCK;
    for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
        const uint32_t row0 = start_tile_row + hb * HT_BLOCK;
        uint32_t ht = num_tile_rows - hb * HT_BLOCK;
        if (ht > HT_BLOCK) {
            ht = HT_BLOCK;
        }

        uint32_t valid_rows = ht * 32u;
        if constexpr (IS_RM) {
            const uint32_t remaining = TOTAL_STICKS - row0 * 32u;
            if (remaining < valid_rows) {
                valid_rows = remaining;
            }
        }

        // pass A — feeds square -> chunked SUM.
        if constexpr (!SHARDED_IN) {
            MaybeDeviceZoneScope("rdr_read_a");
            read_input_pass(row0, ht, valid_rows);
        }

        // ---- combine: gather (root) -> finalize (compute) -> broadcast -----
        // This sits BETWEEN the two reader passes on purpose: compute's pass B
        // cannot start before the broadcast lands, so a reader that queued its
        // pass-B reads first would fill cb_input_tiles and deadlock.
        if constexpr (W_SPLIT && ALLGATHER_PULL) {
            // ---- LAB IDEA: one-hop, no-root all-gather -----------------------
            // Every core (not just the root) pulls all CW peers' raw partial
            // tiles directly out of their L1 -- no gather target, no mcast-back.
            MaybeDeviceZoneScope("rdr_allgather");
            cb_wait_front(cb_partial_out, ht);
            const uint32_t my_partial_addr = get_read_ptr(cb_partial_out);
            const uint32_t partial_tile_bytes = get_tile_size(cb_partial_out);
            Semaphore<> ag_sem(SEM_ALLGATHER);
            // Announce: bump every peer's copy of the shared readiness counter
            // (including our own, via a same-core NoC loopback). MONOTONIC --
            // never reset -- so a fast peer's NEXT round-block announce landing
            // early can never be misattributed to THIS round: wait_min on an
            // ever-growing counter is immune to interleaving order, unlike a
            // reset-to-0-then-wait(CW) counter would be (see lab_descriptor.py).
            for (uint32_t i = 0; i < CW; ++i) {
                ag_sem.up(noc, peer_vx[i], peer_vy[i], 1);
            }
            ag_sem.wait_min((hb + 1) * CW);
            // Pull: read every peer's `ht` raw tiles into OUR OWN cb_group_partials,
            // landing h-major (tile h*CW + slot) -- the SAME layout the fold below
            // (and the baseline's root fold) already expects.
            cb_reserve_back(cb_group_partials, HT_BLOCK * CW);
            const uint32_t base = get_write_ptr(cb_group_partials);
            for (uint32_t i = 0; i < CW; ++i) {
                const uint64_t src = get_noc_addr(peer_vx[i], peer_vy[i], my_partial_addr);
                for (uint32_t h = 0; h < ht; ++h) {
                    const uint32_t dst = base + (h * CW + i) * partial_tile_bytes;
                    noc_async_read(src + h * partial_tile_bytes, dst, partial_tile_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_group_partials, HT_BLOCK * CW);
            cb_pop_front(cb_partial_out, ht);
            // NO broadcast-back: compute folds cb_group_partials straight into
            // cb_rms_sum itself (see rms_norm_compute.cpp's cmp_combine).
        } else if constexpr (W_SPLIT) {
            // stage 1 — every leader folds its own fan-in (CW1 cores). On the
            // flat topology the root is the only leader and CW1 == CW.
            {
                MaybeDeviceZoneScope("rdr_gather_wait");
                if (is_leader) {
                    cb_reserve_back(cb_group_partials, HT_BLOCK * CW1);
                    gather_sem.wait(CW1);
                    gather_sem.set(0);
                    cb_push_back(cb_group_partials, HT_BLOCK * CW1);
                }
                // stage 2 — the root folds the CW2 row-sums the leaders shipped.
                if constexpr (TWO_STAGE) {
                    if (is_root) {
                        cb_reserve_back(cb_group_partials2, HT_BLOCK * CW2);
                        gather2_sem.wait(CW2);
                        gather2_sem.set(0);
                        cb_push_back(cb_group_partials2, HT_BLOCK * CW2);
                    }
                }
            }
            {
                MaybeDeviceZoneScope("rdr_mcast");
                cb_reserve_back(cb_rms_sum, HT_BLOCK);
                const uint32_t dst = get_write_ptr(cb_rms_sum);
                if (is_root) {
                    cb_wait_front(cb_rms_mean, ht);
                    const uint32_t src = get_read_ptr(cb_rms_mean);
                    const uint32_t bytes = ht * fp32_tile_bytes;
                    // The root is inside exactly one family's rectangle, so that
                    // send() loops the data back into its OWN cb_rms_sum; the other
                    // family reaches the cores across the virtual seam.
                    if constexpr (mc_a.active) {
                        auto pipe = mc_a.sender(noc);
                        pipe.send(src, dst, bytes);
                    }
                    if constexpr (mc_b.active) {
                        auto pipe = mc_b.sender(noc);
                        pipe.send(src, dst, bytes);
                    }
                    cb_pop_front(cb_rms_mean, ht);
                } else if (mcast_family == 0) {
                    auto pipe = mc_a.receiver(noc);
                    pipe.receive();
                } else {
                    auto pipe = mc_b.receiver(noc);
                    pipe.receive();
                }
                cb_push_back(cb_rms_sum, HT_BLOCK);
            }
        }

        // pass B — re-read x only when it is not resident; stream gamma when it
        // is not resident.
        if constexpr (!X_RESIDENT || (HAS_GAMMA && !GAMMA_RESIDENT)) {
            MaybeDeviceZoneScope("rdr_read_b");
            for (uint32_t wc = 0; wc < NW; ++wc) {
                if constexpr (!X_RESIDENT) {
                    if constexpr (IS_RM) {
                        read_input_chunk_rm(wc, row0, ht, valid_rows);
                    } else {
                        read_input_batch_tile(wc, 1, row0, ht);
                    }
                }
                if constexpr (HAS_GAMMA && !GAMMA_RESIDENT) {
                    read_gamma_chunk(wc);
                }
            }
        }
    }

    // ReceiverPipe's readiness ack is a NON-POSTED remote atomic; drain it
    // before the kernel exits so the core's NoC transaction counters balance
    // (an outstanding atomic at exit stalls dispatch completion).
    if constexpr (W_SPLIT || GAMMA_MCAST) {
        noc_async_atomic_barrier();
    }
}
