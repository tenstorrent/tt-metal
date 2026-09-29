// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE combine program, reader.  Work unit = (row tile r, column group g): 32 token rows by `cols` column tiles of the
// output; this core takes units unit_start .. unit_start + unit_count - 1 in order (units are r-major, so the score
// tiles of a row tile serve every unit of that row tile in a row).
//
// Per row tile (when it changes): the 32 routing rows (top_k uint16 indices and bf16 scores each, one 20-byte DRAM page
// per row, staged at a 64-byte pitch: a DRAM page lands at its own 64-byte phase), then the top_k score tiles the
// fused weighted reduce's reader builds (deepseek_moe_fast_reduce_nc_fused_reader.cpp): tile k, column 0, row j =
// score[j][k] when this device owns the slot's expert (the owner row read once), else bf16 +0.0; the other columns are
// never read (BroadcastType::COL), as in that reader.
//
// Per unit, slot by slot in the MAC's order: one ROW_MAJOR block of 32 rows by cols x 64 bytes into cb_rm (the tilize
// op's reader pattern, reader_unary_stick_layout_split_rows_multicore.cpp: row j at j x (cols x 64) bytes), each row a
// contiguous read of the page (slot e, token 32 r + j) at byte offset g x cols x 64.  With read_all off the rows of the
// slots this device does not own are zero-filled by the NoC instead of read (the chain multiplies them by +0.0).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t combine_addr = get_arg_val<uint32_t>(0);
    const uint32_t scores_addr = get_arg_val<uint32_t>(1);
    const uint32_t indices_addr = get_arg_val<uint32_t>(2);
    const uint32_t owner_addr = get_arg_val<uint32_t>(3);
    const uint32_t unit_start = get_arg_val<uint32_t>(4);
    const uint32_t unit_count = get_arg_val<uint32_t>(5);
    const uint32_t rows = get_arg_val<uint32_t>(6);  // the page's token rows (runtime: one program per slab size)
    if (unit_count == 0) {
        return;  // fp.split_work never hands out an empty range; no read may be left in flight if it ever did
    }

    constexpr uint32_t cb_rm = get_named_compile_time_arg_val("cb_rm");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_owner = get_named_compile_time_arg_val("cb_owner");
    constexpr uint32_t cb_route = get_named_compile_time_arg_val("cb_route");
    constexpr uint32_t cols = get_named_compile_time_arg_val("cols");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t groups = get_named_compile_time_arg_val("groups");
    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t owner_bytes = get_named_compile_time_arg_val("owner_bytes");
    constexpr uint32_t read_all = get_named_compile_time_arg_val("read_all");
    constexpr uint32_t route_bytes = top_k * 2;  // one ROW_MAJOR routing row
    constexpr uint32_t route_pitch = 64;         // staged routing row pitch
    constexpr uint32_t route_words = route_pitch / 2;
    constexpr uint32_t frag_bytes = cols * 64;  // one token row's slice of the unit's columns (cols x 32 bf16)
    constexpr uint32_t tile_u16 = 1024;         // uint16 per bf16 tile
    static_assert(top_k <= 32, "one staged routing row holds 32 slots; the owned mask is 32 bits");
    static_assert(frag_bytes % 64 == 0, "a row fragment is whole 64-byte DRAM read units");

    constexpr auto combine_args = TensorAccessorArgs<0, 0>();
    constexpr auto scores_args = TensorAccessorArgs<
        combine_args.next_compile_time_args_offset(),
        combine_args.next_common_runtime_args_offset()>();
    constexpr auto indices_args = TensorAccessorArgs<
        scores_args.next_compile_time_args_offset(),
        scores_args.next_common_runtime_args_offset()>();
    constexpr auto owner_args = TensorAccessorArgs<
        indices_args.next_compile_time_args_offset(),
        indices_args.next_common_runtime_args_offset()>();
    const auto combine = TensorAccessor(combine_args, combine_addr);
    const auto scores = TensorAccessor(scores_args, scores_addr);
    const auto indices = TensorAccessor(indices_args, indices_addr);
    const auto owner = TensorAccessor(owner_args, owner_addr);

    Noc noc;
    DataflowBuffer rm(cb_rm);
    DataflowBuffer score_tiles(cb_scores);
    DataflowBuffer owner_row(cb_owner);
    DataflowBuffer route(cb_route);

    // scratch (this kernel alone reads it): the owner row and the two staged routing tables, reserved once
    owner_row.reserve_back(1);
    const uint32_t owner_base = owner_row.get_write_ptr();
    noc.async_read(owner, CoreLocalMem<uint32_t>(owner_base), owner_bytes, {.page_id = 0}, {});
    route.reserve_back(2);
    const uint32_t idx_base = route.get_write_ptr();
    const uint32_t sc_base = idx_base + 32 * route_pitch;
    volatile tt_l1_ptr uint16_t* owner_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(owner_base);
    volatile tt_l1_ptr uint16_t* idx_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_base);
    volatile tt_l1_ptr uint16_t* sc_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc_base);

    uint32_t owned[32];
    uint32_t last_r = 0xffffffffu;
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t r = u / groups;
        const uint32_t g = u % groups;
        if (r != last_r) {
            {
                FUSED_ZONE("fz_mc_r_route");
                for (uint32_t j = 0; j < 32; ++j) {
                    const uint32_t t = r * 32 + j;
                    noc.async_read(
                        indices, CoreLocalMem<uint32_t>(idx_base + j * route_pitch), route_bytes, {.page_id = t}, {});
                    noc.async_read(
                        scores, CoreLocalMem<uint32_t>(sc_base + j * route_pitch), route_bytes, {.page_id = t}, {});
                }
                noc.async_read_barrier();  // the owner row too, the first time
            }
            {
                FUSED_ZONE("fz_mc_r_scores");
                // deepseek_moe_fast_reduce_nc_fused_reader.cpp: tile k, column 0, row j = score[j][k] when the slot's
                // expert is on this device, else bf16 +0.0 (bit pattern 0x0000); column 0 of row j is face 0 (j < 16)
                // or face 2 (j >= 16) at uint16 index j * 16 or 512 + (j - 16) * 16
                score_tiles.reserve_back(top_k);
                volatile tt_l1_ptr uint16_t* stile =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(score_tiles.get_write_ptr());
                for (uint32_t j = 0; j < 32; ++j) {
                    const uint32_t col0 = (j < 16) ? j * 16 : 512 + (j - 16) * 16;
                    uint32_t mask = 0;
                    for (uint32_t k = 0; k < top_k; ++k) {
                        const uint32_t expert = idx_u16[j * route_words + k];
                        const bool own = expert < experts && owner_u16[expert] != 0;
                        stile[k * tile_u16 + col0] = own ? sc_u16[j * route_words + k] : static_cast<uint16_t>(0);
                        if (own) {
                            mask |= 1u << k;
                        }
                    }
                    owned[j] = mask;
                }
                score_tiles.push_back(top_k);
            }
            last_r = r;
        }
        {
            FUSED_ZONE("fz_mc_r_blocks");
            for (uint32_t e = 0; e < top_k; ++e) {
                rm.reserve_back(cols);
                const uint32_t base = rm.get_write_ptr();
                bool zeroed = false;
                for (uint32_t j = 0; j < 32; ++j) {
                    const uint32_t dst = base + j * frag_bytes;
                    if (read_all || ((owned[j] >> e) & 1u)) {
                        noc.async_read(
                            combine,
                            CoreLocalMem<uint32_t>(dst),
                            frag_bytes,
                            {.page_id = e * rows + r * 32 + j, .offset_bytes = g * frag_bytes},
                            {});
                    } else {
                        noc.async_write_zeros(CoreLocalMem<uint32_t>(dst), frag_bytes, {});
                        zeroed = true;
                    }
                }
                noc.async_read_barrier();
                if (zeroed) {
                    noc.write_zeros_l1_barrier();
                }
                rm.push_back(cols);
            }
        }
    }
}
