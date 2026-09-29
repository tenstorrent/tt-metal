// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE post program, reader (one output column tile per core): the routing rows (top_k uint16 indices and bf16
// scores per token), the device's expert owner row, the top_k score tiles the fused weighted reduce's reader builds
// (column 0, row j = the bf16 score when this device owns the slot's expert, else +0.0; padding rows +0.0), the top_k
// activation tiles of this column (every page starts as the constant zero tile; the owned (row, slot) 64-byte
// ROW_MAJOR fragments of moe_compute's local combine buffer then land in the tile's two faces), the shared partial
// tile and, when the shared expert arrives ungated, its column-broadcast sigmoid tile.  Pages of slots this device
// does not own are never read, so the fill the composed chain needs is gone.  The zero seeds (the activation
// tiles, the score tiles) are the NoC's zero write from the firmware's zero block (no DRAM traffic: 80 cores
// reading one DRAM tile page cost 30 us).  Routing rows that are one shard on one core (moe_compute's drain-core
// L1 shard, the served form) come in one read per tensor at the shard's page pitch: 2 x rows page reads from one
// core's L1 by all 80 cores were the served step's hot spot (measured 2026-09-25).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t pages_addr = get_arg_val<uint32_t>(0);
    const uint32_t scores_addr = get_arg_val<uint32_t>(1);
    const uint32_t indices_addr = get_arg_val<uint32_t>(2);
    const uint32_t owner_addr = get_arg_val<uint32_t>(3);
    const uint32_t shared_addr = get_arg_val<uint32_t>(4);
    const uint32_t sig_addr = get_arg_val<uint32_t>(5);
    const uint32_t tile_col = get_arg_val<uint32_t>(6);
    const uint32_t rows = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_act = get_named_compile_time_arg_val("cb_act");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_owner = get_named_compile_time_arg_val("cb_owner");
    constexpr uint32_t cb_route = get_named_compile_time_arg_val("cb_route");
    constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");
    constexpr uint32_t cb_shared = get_named_compile_time_arg_val("cb_shared");
    constexpr uint32_t cb_sig = get_named_compile_time_arg_val("cb_sig");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t has_sig = get_named_compile_time_arg_val("has_sig");
    constexpr uint32_t stage_pages = get_named_compile_time_arg_val("stage_pages");
    constexpr uint32_t owner_bytes = get_named_compile_time_arg_val("owner_bytes");
    constexpr uint32_t route_contiguous = get_named_compile_time_arg_val("route_contiguous");
    constexpr uint32_t route_bytes = top_k * 2;  // one ROW_MAJOR routing row
    constexpr uint32_t route_pitch = 64;         // staged routing row pitch: a DRAM page lands at its own 64-byte phase
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t fragment_bytes = 64;  // 32 bf16 of one token row = one tile column
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t face_row_bytes = 32;  // 16 bf16 of one face row
    static_assert(top_k <= 32, "one staged routing row holds 32 slots");

    constexpr auto pages_args = TensorAccessorArgs<0, 0>();
    constexpr auto scores_args =
        TensorAccessorArgs<pages_args.next_compile_time_args_offset(), pages_args.next_common_runtime_args_offset()>();
    constexpr auto indices_args = TensorAccessorArgs<
        scores_args.next_compile_time_args_offset(),
        scores_args.next_common_runtime_args_offset()>();
    constexpr auto owner_args = TensorAccessorArgs<
        indices_args.next_compile_time_args_offset(),
        indices_args.next_common_runtime_args_offset()>();
    constexpr auto shared_args =
        TensorAccessorArgs<owner_args.next_compile_time_args_offset(), owner_args.next_common_runtime_args_offset()>();
    constexpr auto sig_args = TensorAccessorArgs<
        shared_args.next_compile_time_args_offset(),
        shared_args.next_common_runtime_args_offset()>();
    const auto pages = TensorAccessor(pages_args, pages_addr);
    const auto scores = TensorAccessor(scores_args, scores_addr);
    const auto indices = TensorAccessor(indices_args, indices_addr);
    const auto owner = TensorAccessor(owner_args, owner_addr);
    const auto shared = TensorAccessor(shared_args, shared_addr);
    const auto sig = TensorAccessor(sig_args, sig_addr);

    Noc noc;
    DataflowBuffer act(cb_act);
    DataflowBuffer score_tiles(cb_scores);
    DataflowBuffer owner_row(cb_owner);
    DataflowBuffer route(cb_route);
    DataflowBuffer stage(cb_stage);
    DataflowBuffer shared_tile(cb_shared);
    DataflowBuffer sig_tile(cb_sig);

    uint32_t act_base, owner_base, idx_base, sc_base, stile_base, stage_base, n;
    uint32_t route_words = route_pitch / 2;  // uint16 per staged routing row
    uint32_t owned[32];
    {
        FUSED_ZONE("fz_mp_r_setup");
        // the shared partial tile (and the sigmoid tile) first: the reads overlap the routing work
        shared_tile.reserve_back(1);
        noc.async_read(
            shared, CoreLocalMem<uint32_t>(shared_tile.get_write_ptr()), tile_bytes, {.page_id = tile_col}, {});
        if constexpr (has_sig) {
            sig_tile.reserve_back(1);
            noc.async_read(sig, CoreLocalMem<uint32_t>(sig_tile.get_write_ptr()), tile_bytes, {.page_id = 0}, {});
        }
        // every activation page starts as zeros (the composed chain's fill and tilize padding)
        act.reserve_back(top_k);
        act_base = act.get_write_ptr();
        noc.async_write_zeros(CoreLocalMem<uint32_t>(act_base), top_k * tile_bytes, {});
        // and every score tile: the loop below writes only the owned (row, slot) scores
        score_tiles.reserve_back(top_k);
        stile_base = score_tiles.get_write_ptr();
        noc.async_write_zeros(CoreLocalMem<uint32_t>(stile_base), top_k * tile_bytes, {});
        owner_row.reserve_back(1);
        owner_base = owner_row.get_write_ptr();
        noc.async_read(owner, CoreLocalMem<uint32_t>(owner_base), owner_bytes, {.page_id = 0}, {});
        route.reserve_back(2);
        idx_base = route.get_write_ptr();
        sc_base = idx_base + 32 * route_pitch;
        if constexpr (route_contiguous) {
            // one shard on one core: the rows lie at the shard's page pitch, one read per tensor
            route_words = indices.get_aligned_page_size() / 2;
            noc_async_read(indices.get_noc_addr(0), idx_base, rows * 2 * route_words);
            noc_async_read(scores.get_noc_addr(0), sc_base, rows * 2 * route_words);
        } else {
            for (uint32_t j = 0; j < rows; ++j) {
                noc.async_read(
                    indices, CoreLocalMem<uint32_t>(idx_base + j * route_pitch), route_bytes, {.page_id = j}, {});
                noc.async_read(
                    scores, CoreLocalMem<uint32_t>(sc_base + j * route_pitch), route_bytes, {.page_id = j}, {});
            }
        }
        noc.async_read_barrier();
        shared_tile.push_back(1);
        if constexpr (has_sig) {
            sig_tile.push_back(1);
        }
    }

    {
        FUSED_ZONE("fz_mp_r_scores");
        // deepseek_moe_fast_reduce_nc_fused_reader.cpp: tile k, column 0, row j = score[j][k] when the expert is on
        // this device's axis, else bf16 +0.0 (the zero seed; rows past the tokens stay +0.0); the other columns are
        // never read (COL broadcast)
        volatile tt_l1_ptr uint16_t* owner_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(owner_base);
        volatile tt_l1_ptr uint16_t* idx_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_base);
        volatile tt_l1_ptr uint16_t* sc_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc_base);
        volatile tt_l1_ptr uint16_t* stile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stile_base);
        for (uint32_t j = 0; j < 32; ++j) {
            owned[j] = 0;
        }
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t col0 = (j < 16) ? j * 16 : 512 + (j - 16) * 16;
            uint32_t mask = 0;
            for (uint32_t k = 0; k < top_k; ++k) {
                if (owner_u16[idx_u16[j * route_words + k]] != 0) {
                    stile[k * 1024 + col0] = sc_u16[j * route_words + k];
                    mask |= 1u << k;
                }
            }
            owned[j] = mask;
        }
        score_tiles.push_back(top_k);
    }

    {
        FUSED_ZONE("fz_mp_r_frag_dram");
        // owned fragments: 64-byte reads (DRAM read alignment) into a 64-byte-aligned staging area, then each 32-byte
        // half into face (j / 16) * 2 and (j / 16) * 2 + 1, row j % 16, of the slot's tile
        stage.reserve_back(stage_pages);
        stage_base = (stage.get_write_ptr() + fragment_bytes - 1) & ~(fragment_bytes - 1);
        n = 0;
        for (uint32_t j = 0; j < rows; ++j) {
            for (uint32_t k = 0; k < top_k; ++k) {
                if (owned[j] & (1u << k)) {
                    noc.async_read(
                        pages,
                        CoreLocalMem<uint32_t>(stage_base + n * fragment_bytes),
                        fragment_bytes,
                        {.page_id = k * rows + j, .offset_bytes = tile_col * fragment_bytes},
                        {});
                    ++n;
                }
            }
        }
        noc.async_read_barrier();  // the zero pages and the fragments have landed
    }
    {
        FUSED_ZONE("fz_mp_r_frag_place");
        n = 0;
        for (uint32_t j = 0; j < rows; ++j) {
            for (uint32_t k = 0; k < top_k; ++k) {
                if (owned[j] & (1u << k)) {
                    const uint32_t src = stage_base + n * fragment_bytes;
                    const uint32_t dst =
                        act_base + k * tile_bytes + (j >> 4) * 2 * face_bytes + (j & 15) * face_row_bytes;
                    noc_async_read(get_noc_addr(src), dst, face_row_bytes);
                    noc_async_read(get_noc_addr(src + face_row_bytes), dst + face_bytes, face_row_bytes);
                    ++n;
                }
            }
        }
        noc_async_read_barrier();
    }
    act.push_back(top_k);
    stage.push_back(stage_pages);
    owner_row.push_back(1);
    route.push_back(2);
}
