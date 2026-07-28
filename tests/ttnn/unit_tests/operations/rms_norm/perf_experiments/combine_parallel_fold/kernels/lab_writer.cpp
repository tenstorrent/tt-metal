// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer (BRISC / NoC1) — op_design.md §4.1 and §4.2.
//
// Mirrors the reader's row-block/chunk nest exactly: per core, per row-block of
// HT_BLOCK tile-rows, drain NW chunks of WT_CHUNK W-tiles of this core's W
// slice. The TILE path writes whole tile pages with one barrier per chunk (the
// batched twin of the reader's batched chunk read); the ROW_MAJOR path goes
// through dataflow_kernel_lib::write_sticks_after_untilize, which skips the L1
// W-padding and writes only the valid sticks of a short last tile-row — that is
// what makes non-aligned H/W native with no host-side slice.
//
// SHARDED output (SHARDED_OUT): there is NO write at all. cb_output_tiles is
// placed directly on the core's own L1 output shard, so compute's pack lands in
// the output tensor in place — zero-copy, no NoC traffic.
//
// CROSS-CORE W-SPLIT (W_SPLIT): this kernel owns the *gather* half of the
// combine. Per row-block it ships this core's raw sum(x^2) tiles into slot
// `s1_slot` of its LEADER's cb_group_partials and bumps that leader's stage-1
// counter; the leader's reader waits for all CW1 of them (see
// rms_norm_reader.cpp). With CW2 == 1 the leader is the group root and that is
// the whole combine. With CW2 > 1 the combine is TWO-STAGE, and a leader ships a
// SECOND tile — the row sum compute folded out of its stage-1 gather — into slot
// `s2_slot` of the root's cb_group_partials2, bumping the stage-2 counter. Both
// pushes ride the SAME cb_partial_out (one producer, one consumer, two
// sequential pushes), so the staged topology adds no CB.
// Many-to-one has no multicast form and no helper covers it, so each leg is a
// plain noc_async_write + barrier + noc_semaphore_inc
// (references/cross_core_reduction_design.md §1/§7).
//
// PER-STAGE INSTRUMENTATION (permanent, Perf 1). Stage boundaries carry a
// MaybeDeviceZoneScope: wtr_gather_hop / wtr_write. Free when the profiler is
// off (perf_instrumentation.hpp's durability contract) — never remove one, and
// extend the set to any new predicate-guarded path.
//
// Helper substitution note: the TILE path uses TensorAccessor +
// noc_async_write_page directly because write_sticks_after_untilize writes
// *sticks* produced by the untilize helper, and the TILE path never untilizes —
// its input contract cannot be met. op_design.md §7 mandates exactly this.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_group_partials = 6;
constexpr uint32_t cb_rms_mean = 7;
constexpr uint32_t cb_partial_out = 8;
constexpr uint32_t cb_group_partials2 = 9;
// row_rotate / pack_rotate reuse slot 9 as the OWNER-output CB (two_stage's
// stage-2 gather CB is unused there), so the fold's push granularity gets its
// own buffer.
constexpr uint32_t cb_owner_out = 9;
// PERF-2 (combine_parallel_fold): column-pack/select scalers, reused verbatim
// from gather_payload_shrink's mechanism.
constexpr uint32_t cb_packsel = 10;
constexpr uint32_t cb_colsel = 11;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
}  // namespace

namespace dkl = dataflow_kernel_lib;

void kernel_main() {
    // ---- regime flags (§5.2) ----
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    // args 1..5 are the reader/compute regime flags; kept in one shared layout.
    // ---- block knobs (§1.2) ----
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t WT_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t NW = get_compile_time_arg_val(9);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(10);
    // arg 11 (X_READ_CHUNKS) is a read-side knob; the writer's granularity is
    // set by the compute output CB, which always publishes one chunk at a time.
    // ---- geometry ----
    constexpr uint32_t CHUNK_ROW_BYTES = get_compile_time_arg_val(13);
    constexpr uint32_t LAST_ROW_BYTES = get_compile_time_arg_val(14);
    constexpr uint32_t TOTAL_STICKS = get_compile_time_arg_val(17);
    // ---- cross-core W-split (§4.2) ----
    constexpr bool W_SPLIT = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t CW = get_compile_time_arg_val(19);
    constexpr uint32_t WT_STRIDE = get_compile_time_arg_val(20);
    constexpr bool SHARDED_OUT = get_compile_time_arg_val(22) != 0;
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(23);
    constexpr uint32_t CW1 = get_compile_time_arg_val(24);
    constexpr uint32_t CW2 = get_compile_time_arg_val(25);
    constexpr uint32_t SEM_GATHER2 = get_compile_time_arg_val(26);
    // ---- LAB: SCHEME 2 (per-tile-row root rotation) — see lab_reader.cpp ----
    constexpr bool ROW_ROTATE = get_compile_time_arg_val(27) != 0;
    constexpr uint32_t GATHER_PAGES = get_compile_time_arg_val(28);
    // ---- PERF-2 IDEA (combine_parallel_fold): the K-way pack/rotate hybrid ----
    constexpr bool PACK_ROTATE = get_compile_time_arg_val(29) != 0;
    constexpr uint32_t PACK_K = get_compile_time_arg_val(30);
    constexpr bool TWO_STAGE = CW2 > 1;
    static_assert(CW1 * CW2 == CW, "combine stages must tile CW");
    static_assert(!ROW_ROTATE || CW2 == 1, "row_rotate has no leader tree");
    static_assert(!PACK_ROTATE || CW2 == 1, "pack_rotate has no leader tree");
    static_assert(!(ROW_ROTATE && PACK_ROTATE), "row_rotate and pack_rotate are mutually exclusive");
    (void)GATHER_PAGES;

    // 31..40 are the two multicast-family CT blocks (reader-side; the writer
    // shares the layout so a knob cannot drift between them).
    constexpr auto out_args = TensorAccessorArgs<41>();

    static_assert(WT_LAST == WT_CHUNK, "writer assumes uniform chunk widths");

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t wt_start = get_arg_val<uint32_t>(3);
    const uint32_t s1_slot = get_arg_val<uint32_t>(4);
    const uint32_t leader_x = get_arg_val<uint32_t>(5);
    const uint32_t leader_y = get_arg_val<uint32_t>(6);
    const uint32_t is_last_w_core = get_arg_val<uint32_t>(7);
    const uint32_t root_x = get_arg_val<uint32_t>(8);
    const uint32_t root_y = get_arg_val<uint32_t>(9);
    const uint32_t s2_slot = get_arg_val<uint32_t>(10);
    const uint32_t is_leader = get_arg_val<uint32_t>(11);

    // Filler core (inside the multicast rectangle, owns no data).
    if (num_tile_rows == 0) {
        return;
    }

    const auto out_acc = TensorAccessor(out_args, dst_addr);
    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
    const uint32_t chunk0 = wt_start / WT_CHUNK;

    // Gather targets: the leader's cb_group_partials (stage 1) and the root's
    // cb_group_partials2 (stage 2). Every core declares both CBs at the same
    // size, so each sits at the same L1 offset everywhere and this core's own
    // write pointer IS the remote base address.
    [[maybe_unused]] const uint32_t partial_tile_bytes = get_tile_size(cb_partial_out);
    [[maybe_unused]] const uint32_t gather_base = get_write_ptr(cb_group_partials);
    [[maybe_unused]] const uint32_t gather2_base = get_write_ptr(cb_group_partials2);
    [[maybe_unused]] const uint32_t gather_sem_addr = get_semaphore<ProgrammableCoreType::TENSIX>(SEM_GATHER);
    [[maybe_unused]] const uint32_t gather2_sem_addr = get_semaphore<ProgrammableCoreType::TENSIX>(SEM_GATHER2);

    // SCHEME 2 / PERF-2: the destination of tile-row h (row_rotate) or pack-lane
    // p (pack_rotate) is a group slot, so this core needs every slot's virtual
    // coords (RT 12.., 2 words per slot) and the grid-uniform base of
    // cb_rms_mean (the root's collect buffer).
    constexpr bool NEEDS_OWNER_COORDS = ROW_ROTATE || PACK_ROTATE;
    [[maybe_unused]] const uint32_t rms_mean_base = get_write_ptr(cb_rms_mean);
    uint32_t owner_vx[NEEDS_OWNER_COORDS ? CW : 1];
    uint32_t owner_vy[NEEDS_OWNER_COORDS ? CW : 1];
    if constexpr (NEEDS_OWNER_COORDS) {
        for (uint32_t j = 0; j < CW; ++j) {
            owner_vx[j] = get_arg_val<uint32_t>(12 + 2 * j);
            owner_vy[j] = get_arg_val<uint32_t>(12 + 2 * j + 1);
        }
    } else {
        owner_vx[0] = 0;
        owner_vy[0] = 0;
    }
    // PERF-2 (combine_parallel_fold): owner_rt is 2*CW words wide unconditionally
    // (see rms_norm host descriptor), so N_REDUCED — appended LAST, after it —
    // sits at a compile-time-known offset.
    [[maybe_unused]] const uint32_t n_reduced_arg = get_arg_val<uint32_t>(12 + 2 * CW);

    // ==== PERF-2 (combine_parallel_fold): the column-pack's two scaler BANKS ==
    //
    // RAW L1 FILL — deliberate helper bypass, reusing gather_payload_shrink's
    // measured mechanism verbatim (its probe_mechanism.py established this on
    // real silicon): dataflow_kernel_lib's reduce-scaler helpers emit only the
    // CANONICAL layout (face-row 0 of every face), because that is the only
    // layout a plain REDUCE_ROW needs. This idea's mechanism is two NON-canonical
    // layouts:
    //
    //   cb_packsel[m]  1.0 across FACE-ROW m of every face
    //                  -> reduce_tile writes the row-sum into COLUMN m, so
    //                     `ho = HT_BLOCK/PACK_K` reduce_tiles accumulating into
    //                     ONE dest tile column-PACK a pack-lane's row-sums.
    //   cb_colsel[m]   1/W at (face-row 0, col m) of faces 0 and 2
    //                  -> reduce_tile SELECTS input column m into column 0 and
    //                     applies the 1/N in the same op.
    //
    // Sized HT_BLOCK unconditionally (a superset of the `ho <= HT_BLOCK` columns
    // any PACK_K actually uses), so PACK_K can vary without a second CB size.
    // Runs on the WRITER (idle until its first gather hop), not the reader —
    // gather_payload_shrink measured +20.5 us / a false REGRESSION from doing
    // this by hand on the reader, in front of the shard publish compute's first
    // pass waits on; the NoC memset (`async_write_zeros`) is what makes it cheap.
    if constexpr (W_SPLIT && PACK_ROTATE) {
        MaybeDeviceZoneScope("wtr_selectors");
        constexpr uint32_t FACE_W = 16;
        constexpr uint32_t FACE_WORDS = 256;  // fp32 16x16 face
        Noc zero_noc;
        auto put = [](uint32_t addr, uint32_t face, uint32_t r, uint32_t c, float v) {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
            p[face * FACE_WORDS + r * FACE_W + c] = __builtin_bit_cast(uint32_t, v);
        };

        DataflowBuffer psel(cb_packsel);
        psel.reserve_back(HT_BLOCK);
        const uint32_t pa = psel.get_write_ptr();
        const uint32_t sel_bytes = get_tile_size(cb_packsel);
        zero_noc.async_write_zeros(psel, HT_BLOCK * sel_bytes);
        zero_noc.write_zeros_l1_barrier();
        for (uint32_t m = 0; m < HT_BLOCK; ++m) {
            for (uint32_t f = 0; f < 4; ++f) {
                for (uint32_t c = 0; c < FACE_W; ++c) {
                    put(pa + m * sel_bytes, f, m, c, 1.0f);
                }
            }
        }
        psel.push_back(HT_BLOCK);

        DataflowBuffer csel(cb_colsel);
        csel.reserve_back(HT_BLOCK);
        const uint32_t ca = csel.get_write_ptr();
        const uint32_t csel_bytes = get_tile_size(cb_colsel);
        zero_noc.async_write_zeros(csel, HT_BLOCK * csel_bytes);
        zero_noc.write_zeros_l1_barrier();
        const float inv_n = 1.0f / static_cast<float>(n_reduced_arg);
        for (uint32_t m = 0; m < HT_BLOCK; ++m) {
            put(ca + m * csel_bytes, 0, 0, m, inv_n);
            put(ca + m * csel_bytes, 2, 0, m, inv_n);
        }
        csel.push_back(HT_BLOCK);
    }

    // One gather hop: ship `ht` settled cb_partial_out tiles into `slot` of the
    // destination core's gather CB (laid out h-major, tile h*fan_in + slot, so
    // the folding core reads them as a contiguous (ht x fan_in) block), then let
    // it know they LANDED.
    [[maybe_unused]] auto gather_hop = [&](uint32_t ht,
                                           uint32_t dst_x,
                                           uint32_t dst_y,
                                           uint32_t base,
                                           uint32_t fan_in,
                                           uint32_t slot,
                                           uint32_t sem_addr) {
        cb_wait_front(cb_partial_out, ht);
        uint32_t src = get_read_ptr(cb_partial_out);
        for (uint32_t h = 0; h < ht; ++h) {
            const uint64_t dst = get_noc_addr(dst_x, dst_y, base + (h * fan_in + slot) * partial_tile_bytes);
            noc_async_write(src, dst, partial_tile_bytes);
            src += partial_tile_bytes;
        }
        // The data must have LANDED before the destination sees the counter move.
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(dst_x, dst_y, sem_addr), 1);
        cb_pop_front(cb_partial_out, ht);
    };

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

        // ---- gather legs: this core's raw partial sum, then (leaders only)
        // the row sum compute folded out of the stage-1 gather.
        if constexpr (W_SPLIT) {
            MaybeDeviceZoneScope("wtr_gather_hop");
            if constexpr (ROW_ROTATE) {
                // ---- leg 1: SPREAD. Tile-row h lands on the core that OWNS it
                // (group slot h % CW), so no single core's inbound NoC absorbs
                // the whole group's ht x CW payload — each takes CW tiles.
                // ONE barrier covers all ht writes; then one atomic per distinct
                // destination (data-before-signal, per destination).
                cb_wait_front(cb_partial_out, ht);
                uint32_t src = get_read_ptr(cb_partial_out);
                for (uint32_t h = 0; h < ht; ++h) {
                    const uint32_t j = h % CW;
                    const uint32_t k = h / CW;
                    const uint64_t dst =
                        get_noc_addr(owner_vx[j], owner_vy[j], gather_base + (k * CW + s1_slot) * partial_tile_bytes);
                    noc_async_write(src, dst, partial_tile_bytes);
                    src += partial_tile_bytes;
                }
                noc_async_write_barrier();
                const uint32_t owners = (ht < CW) ? ht : CW;
                for (uint32_t j = 0; j < owners; ++j) {
                    noc_semaphore_inc(get_noc_addr(owner_vx[j], owner_vy[j], gather_sem_addr), 1);
                }
                cb_pop_front(cb_partial_out, ht);

                // ---- leg 2: the owner's FINALIZED mean(x^2) tiles go straight
                // into slot h of the root's cb_rms_mean, so the broadcast leg is
                // byte-for-byte the flat topology's. ht tiles reach the root here,
                // not ht * CW.
                if (s1_slot < ht) {
                    const uint32_t ho = (ht - s1_slot + CW - 1) / CW;
                    // cb_owner_out, NOT cb_partial_out: mixing an `ht`-page push
                    // (pass A) with an `ho`-page push on one CB straddles
                    // fifo_limit on the next row-block (measured hang).
                    cb_wait_front(cb_owner_out, ho);
                    uint32_t osrc = get_read_ptr(cb_owner_out);
                    for (uint32_t k = 0; k < ho; ++k) {
                        const uint32_t h = k * CW + s1_slot;
                        const uint64_t dst = get_noc_addr(root_x, root_y, rms_mean_base + h * partial_tile_bytes);
                        noc_async_write(osrc, dst, partial_tile_bytes);
                        osrc += partial_tile_bytes;
                    }
                    noc_async_write_barrier();
                    noc_semaphore_inc(get_noc_addr(root_x, root_y, gather2_sem_addr), 1);
                    cb_pop_front(cb_owner_out, ho);
                }
            } else if constexpr (PACK_ROTATE) {
                // PERF-2 (combine_parallel_fold). ---- leg 1: SHIP the PACK_K
                // column-packed tiles compute produced this row-block, one per
                // pack-lane p, to the FIXED owner of lane p (group slot p, same
                // core every row-block — unlike row_rotate's per-tile-row target,
                // which core owns which LANE never changes). Every owner still
                // receives CW tiles total (one per group member, at THIS core's
                // own slot s1_slot), because every worker computes all PACK_K
                // lanes regardless of who owns the fold.
                //
                // The host's uniform-block invariant (rows_core_max % HT_BLOCK
                // == 0, asserted in create_program_descriptor) guarantees ht ==
                // HT_BLOCK on every row-block this kernel ever runs, so PACK_K
                // packed tiles are unconditionally in cb_partial_out here.
                cb_wait_front(cb_partial_out, PACK_K);
                uint32_t src = get_read_ptr(cb_partial_out);
                for (uint32_t p = 0; p < PACK_K; ++p) {
                    const uint64_t dst =
                        get_noc_addr(owner_vx[p], owner_vy[p], gather_base + s1_slot * partial_tile_bytes);
                    noc_async_write(src, dst, partial_tile_bytes);
                    src += partial_tile_bytes;
                }
                noc_async_write_barrier();
                for (uint32_t p = 0; p < PACK_K; ++p) {
                    noc_semaphore_inc(get_noc_addr(owner_vx[p], owner_vy[p], gather_sem_addr), 1);
                }
                cb_pop_front(cb_partial_out, PACK_K);

                // ---- leg 2: the owner's `ho = HT_BLOCK/PACK_K` FINALIZED
                // mean(x^2) tiles (rows p, p+PACK_K, p+2*PACK_K, ...) go straight
                // into the root's cb_rms_mean, exactly as row_rotate's leg 2 does.
                if (s1_slot < PACK_K) {
                    const uint32_t ho_real = HT_BLOCK / PACK_K;
                    cb_wait_front(cb_owner_out, ho_real);
                    uint32_t osrc = get_read_ptr(cb_owner_out);
                    for (uint32_t m = 0; m < ho_real; ++m) {
                        const uint32_t h = m * PACK_K + s1_slot;
                        const uint64_t dst = get_noc_addr(root_x, root_y, rms_mean_base + h * partial_tile_bytes);
                        noc_async_write(osrc, dst, partial_tile_bytes);
                        osrc += partial_tile_bytes;
                    }
                    noc_async_write_barrier();
                    noc_semaphore_inc(get_noc_addr(root_x, root_y, gather2_sem_addr), 1);
                    cb_pop_front(cb_owner_out, ho_real);
                }
            } else {
                gather_hop(ht, leader_x, leader_y, gather_base, CW1, s1_slot, gather_sem_addr);
                if constexpr (TWO_STAGE) {
                    if (is_leader) {
                        gather_hop(ht, root_x, root_y, gather2_base, CW2, s2_slot, gather2_sem_addr);
                    }
                }
            }
        }

        for (uint32_t wc = 0; wc < NW; ++wc) {
            MaybeDeviceZoneScope("wtr_write");
            if constexpr (SHARDED_OUT) {
                // Zero-copy: compute packed straight into the output shard.
                continue;
            } else if constexpr (IS_RM) {
                const uint32_t rb = (is_last_w_core && wc + 1 == NW) ? LAST_ROW_BYTES : CHUNK_ROW_BYTES;
                dkl::write_sticks_after_untilize<cb_output_rm>(
                    out_acc, valid_rows, rb, row0 * 32u, (chunk0 + wc) * CHUNK_ROW_BYTES);
            } else {
                const uint32_t n = ht * WT_CHUNK;
                cb_wait_front(cb_output_tiles, n);
                uint32_t addr = get_read_ptr(cb_output_tiles);
                for (uint32_t h = 0; h < ht; ++h) {
                    const uint32_t base_tile = (row0 + h) * WT_STRIDE + wt_start + wc * WT_CHUNK;
                    for (uint32_t t = 0; t < WT_CHUNK; ++t) {
                        noc_async_write_page(base_tile + t, out_acc, addr);
                        addr += out_tile_bytes;
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_tiles, n);
            }
        }
    }

    // noc_semaphore_inc issues a NON-POSTED atomic; leaving one outstanding at
    // kernel exit leaves the core's NoC transaction counters unbalanced, which
    // the dispatcher never sees drain (measured as a device hang on the sharded
    // cells, whose writer has no trailing write barrier to absorb it).
    if constexpr (W_SPLIT) {
        noc_async_atomic_barrier();
    }
}
