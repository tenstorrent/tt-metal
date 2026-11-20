// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sparse-MLA row converter (compute, one per S-block): packed scaled-FP8 KV rows -> BF16 K tiles.
//
// A slab is 32 packed rows at the DRAM page stride, [LATENT x FP8 e4m3 | 4 x FP32 scale | ROPE x BF16 | pad].
// The latent field is tilized straight out of the slab (the unpacker is given the page stride as its row width,
// the trick sparse_sdpa's compute kernel uses), multiplied per 128-wide block by the row's scale (FP32 tiles with
// the scale in column 0, built by the reader) and packed as BF16; the RoPE field is tilized from its byte offset.
// Every value is exact: an e4m3 value has four significant bits, the scale is a power of two, BF16 holds eight.
// Output per slab: DHt = (LATENT + ROPE) / 32 tiles, latent first, pushed to cb_kout (this core's own FIFO of
// converted slabs; the sender on this core forwards them).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

#include "sm_common.hpp"

// Tilize `field_ct` tiles of a 32-row field that starts `offset_16b` 16-byte units into each packed row, into
// `output_cb` at tile `out_index`. `physical_ct` is the row stride in tiles of the input format.
template <uint32_t input_cb, uint32_t output_cb, uint32_t physical_ct, uint32_t field_ct, uint32_t offset_16b>
ALWI void tilize_packed_field(uint32_t out_index) {
    reconfig_data_format_srca(input_cb);
    pack_reconfig_data_format(output_cb);
    tilize_init(input_cb, physical_ct, output_cb);

    uint32_t saved_rd_ptr = 0;
    UNPACK({
        auto& iface = get_local_cb_interface(input_cb);
        saved_rd_ptr = iface.fifo_rd_ptr;
        iface.fifo_rd_ptr = saved_rd_ptr + offset_16b;
    });
    tilize_block(input_cb, field_ct, output_cb, 0, out_index);
    UNPACK(get_local_cb_interface(input_cb).fifo_rd_ptr = saved_rd_ptr;);
    tilize_uninit(input_cb, output_cb);
}

void kernel_main() {
#ifdef SM_NOP
    return;
#endif
    constexpr uint32_t cb_rows = get_named_compile_time_arg_val("cb_rows");
    constexpr uint32_t cb_rope_rm = get_named_compile_time_arg_val("cb_rope_rm");
    constexpr uint32_t cb_scale = get_named_compile_time_arg_val("cb_scale");
    constexpr uint32_t cb_lat = get_named_compile_time_arg_val("cb_lat");
    constexpr uint32_t cb_k = get_named_compile_time_arg_val("cb_kout");
    constexpr uint32_t cb_ctrl = get_named_compile_time_arg_val("cb_ctrl");
    constexpr uint32_t STRIDE = get_named_compile_time_arg_val("row_stride");
    constexpr uint32_t LATENT = get_named_compile_time_arg_val("latent");
    constexpr uint32_t ROPE = get_named_compile_time_arg_val("rope");
    constexpr uint32_t SCALE_BLOCKS = get_named_compile_time_arg_val("scale_blocks");

    constexpr uint32_t LATt = LATENT / 32;
    constexpr uint32_t ROPEt = ROPE / 32;
    constexpr uint32_t DHt = LATt + ROPEt;
    constexpr uint32_t TILES_PER_BLOCK = LATt / SCALE_BLOCKS;
    constexpr uint32_t lat_physical_ct = STRIDE / 32;   // FP8: one byte per value
    constexpr uint32_t rope_physical_ct = STRIDE / 64;  // BF16: two bytes per value
    constexpr uint32_t rope_offset_16b = (LATENT + SCALE_BLOCKS * 4) / 16;
    static_assert(STRIDE % 64 == 0 && (LATENT + SCALE_BLOCKS * 4) % 16 == 0 && LATt % SCALE_BLOCKS == 0);

    if (sm::role_from_position<get_named_compile_time_arg_val("x0_left"), get_named_compile_time_arg_val("x0_right")>()
            .slot >= get_named_compile_time_arg_val("converters")) {
        return;  // a core of the reader columns that holds no converter
    }
    CircularBuffer rows(cb_rows), scale(cb_scale), lat(cb_lat), k(cb_k), ctrl(cb_ctrl);

    compute_kernel_hw_startup(cb_rows, cb_lat);

    while (true) {
        ctrl.wait_front(1);
        const uint32_t slabs = ckernel::read_tile_value(cb_ctrl, 0, 0);
        ctrl.pop_front(1);
        if (slabs == sm::CTRL_END) {
            break;
        }
        for (uint32_t slab = 0; slab < slabs; ++slab) {
            rows.wait_front(sm::TILE_ROWS);
            scale.wait_front(SCALE_BLOCKS);
            k.reserve_back(DHt);

            // FP8 latent -> BF16 tiles (unscaled).
            lat.reserve_back(LATt);
            tilize_packed_field<cb_rows, cb_lat, lat_physical_ct, LATt, 0>(0);
            lat.push_back(LATt);

            // x scale, one FP32 scale tile per 128-wide block, broadcast along the columns.
            lat.wait_front(LATt);
            reconfig_data_format(cb_lat, cb_scale);
            pack_reconfig_data_format(cb_k);
            mul_bcast_cols_init(cb_lat, cb_scale);
            for (uint32_t t = 0; t < LATt; ++t) {
                tile_regs_acquire();
                mul_tiles_bcast_cols(cb_lat, cb_scale, t, t / TILES_PER_BLOCK, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_k, t);
                tile_regs_release();
            }
            lat.pop_front(LATt);

            // RoPE: already BF16; cb_rope_rm is a BF16 view of the same slab memory.
            UNPACK(get_local_cb_interface(cb_rope_rm).fifo_rd_ptr = get_local_cb_interface(cb_rows).fifo_rd_ptr;);
            tilize_packed_field<cb_rope_rm, cb_k, rope_physical_ct, ROPEt, rope_offset_16b>(LATt);

            k.push_back(DHt);
            scale.pop_front(SCALE_BLOCKS);
            rows.pop_front(sm::TILE_ROWS);
        }
    }
}
