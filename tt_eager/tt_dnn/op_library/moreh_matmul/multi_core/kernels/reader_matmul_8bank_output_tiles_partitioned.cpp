// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implemented based on reader_bmm_8bank_output_tiles_partitioned.cpp
#include <stdint.h>

#include "dataflow_api.h"
#include "debug_print.h"

void mask_tile(uint32_t l1_addr, uint32_t mask_w = 32, uint32_t mask_h = 32) {
    union {
        float f;
        uint32_t u;
    } zero;
    zero.f = 0.0f;
    auto ptr = reinterpret_cast<uint16_t *>(l1_addr);
    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }
        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }
        // sub tile 2
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }
        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }
}

inline bool is_in0_last_row(uint32_t itileA, uint32_t Mt, uint32_t Kt, uint32_t MtKt) {
    bool in0_last_row = false;
    if ((itileA % MtKt) / Kt == (Mt - 1)) {
        in0_last_row = true;
    }
    return in0_last_row;
}

inline bool is_in1_last_col(uint32_t itileB, uint32_t Nt, uint32_t Kt, uint32_t KtNt, uint32_t b_transpose) {
    bool in1_last_col = false;
    if (b_transpose) {
        if ((itileB % KtNt) / Kt == (Nt - 1)) {
            in1_last_col = true;
        }
    } else {
        if ((itileB % KtNt) == (Nt - 1)) {
            in1_last_col = true;
        }
    }
    return in1_last_col;
}

inline void mask_in0_tile(
    uint32_t l1_write_addr_in0,
    uint32_t in0_last_row,
    uint32_t kt,
    uint32_t Kt,
    uint32_t in0_mask_h,
    uint32_t in0_mask_w) {
    if (in0_last_row) {
        if (kt != Kt - 1) {
            if (in0_mask_h != 32)
                mask_tile(l1_write_addr_in0, 32, in0_mask_h);
        } else {
            if (in0_mask_h != 32 || in0_mask_w != 32)
                mask_tile(l1_write_addr_in0, in0_mask_w, in0_mask_h);
        }
    } else if (kt == Kt - 1) {
        if (in0_mask_w != 32)
            mask_tile(l1_write_addr_in0, in0_mask_w);
    }
}

inline void mask_in1_tile(
    uint32_t l1_write_addr_in1,
    uint32_t in1_last_col,
    uint32_t kt,
    uint32_t Kt,
    uint32_t in1_mask_h,
    uint32_t in1_mask_w,
    uint32_t b_transpose) {
    if (in1_last_col) {
        if (kt != Kt - 1) {
            if (b_transpose) {
                if (in1_mask_h != 32)
                    mask_tile(l1_write_addr_in1, 32, in1_mask_h);

            } else {
                if (in1_mask_w != 32)
                    mask_tile(l1_write_addr_in1, in1_mask_w, 32);
            }
        } else {
            if (in1_mask_w != 32 || in1_mask_h != 32)
                mask_tile(l1_write_addr_in1, in1_mask_w, in1_mask_h);
        }
    }
    // last row
    else if (kt == Kt - 1) {
        if (b_transpose) {
            if (in1_mask_w != 32)
                mask_tile(l1_write_addr_in1, in1_mask_w, 32);

        } else {
            if (in1_mask_h != 32)
                mask_tile(l1_write_addr_in1, 32, in1_mask_h);
        }
    }
}

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);
    uint32_t KtNt = get_arg_val<uint32_t>(6);
    uint32_t a_bcast_B = get_arg_val<uint32_t>(7);
    uint32_t b_bcast_B = get_arg_val<uint32_t>(8);
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(9);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(10);
    uint32_t MtNt = get_arg_val<uint32_t>(11);
    uint32_t a_transpose = get_arg_val<uint32_t>(12);
    uint32_t b_transpose = get_arg_val<uint32_t>(13);
    uint32_t a_start_tile_id = get_arg_val<uint32_t>(14);
    uint32_t b_start_tile_id = get_arg_val<uint32_t>(15);
    uint32_t in0_mask_h = get_arg_val<uint32_t>(16);
    uint32_t in0_mask_w = get_arg_val<uint32_t>(17);
    uint32_t in1_mask_h = get_arg_val<uint32_t>(18);
    uint32_t in1_mask_w = get_arg_val<uint32_t>(19);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;

    // DPRINT << "Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt << " MtKt=" << MtKt << "KtNt=" << KtNt << ENDL();
    // DPRINT << "src0=" << src0_addr << " src1=" << src1_addr << ENDL();
    // DPRINT << "bcast " << a_bcast_B << " " << b_bcast_B << ENDL();
    // DPRINT << "transpose " << a_transpose << " " << b_transpose << ENDL();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t onetile = 1;
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = in0_tile_bytes, .data_format = in0_data_format};

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = in1_tile_bytes, .data_format = in1_data_format};

    uint32_t output_tile_id = output_tile_start_id;
    for (uint32_t n = 0; n < num_output_tiles; n++) {
        // get tile index of a an b
        uint32_t itileA = output_tile_id / Nt * Kt;
        if (a_bcast_B) {
            itileA %= MtKt;
        }
        itileA += a_start_tile_id;

        uint32_t itileB = output_tile_id / MtNt * KtNt;
        if (b_transpose) {
            itileB += (output_tile_id % Nt) * Kt;
        } else {
            itileB += (output_tile_id % Nt);
        }

        if (b_bcast_B) {
            if (b_transpose) {
                itileB %= KtNt;
            } else {
                itileB %= Nt;
            }
        }
        itileB += b_start_tile_id;

        // get last row or last col for mask
        bool in0_last_row = is_in0_last_row(itileA, Mt, Kt, MtKt);
        bool in1_last_col = is_in1_last_col(itileB, Nt, Kt, KtNt, b_transpose);

        for (uint32_t kt = 0; kt < Kt; kt++) {
            {  // Read A's tile at (mt, kt)
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(itileA, s0, l1_write_addr_in0);
                noc_async_read_barrier();

                // mask in in0 tile
                mask_in0_tile(l1_write_addr_in0, in0_last_row, kt, Kt, in0_mask_h, in0_mask_w);
                cb_push_back(cb_id_in0, onetile);
            }

            {  // Read B's tile at (kt, nt)
                cb_reserve_back(cb_id_in1, onetile);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(itileB, s1, l1_write_addr_in1);
                noc_async_read_barrier();

                // mask in in1 tile
                mask_in1_tile(l1_write_addr_in1, in1_last_col, kt, Kt, in1_mask_h, in1_mask_w, b_transpose);
                cb_push_back(cb_id_in1, onetile);
            }

            itileA += 1;  // A is MK
            if (b_transpose == 1) {
                itileB += 1;
            } else {
                itileB += Nt;  // B is KN, so to get k++ we stride by Nt
            }
        }  // Kt loop
        output_tile_id++;
    }
}
