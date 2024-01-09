// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

FORCE_INLINE void generate_bcast_scaler() {
    constexpr uint32_t cb_in_2 = 2;
    uint32_t scaler = get_arg_val<uint32_t>(10);
    //DPRINT << "basic Scaler = " << F32(u.f) << ENDL();
    cb_reserve_back(cb_in_2, 1);
    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_in_2);
    // Fill tile with zeros
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_in_2));
    uint32_t idx = 0;
    for (uint32_t k = 0; k < 4; ++k) {
        uint32_t curr_idx = idx;
        for (uint32_t j = 0; j < 8; ++j) {
            ptr[curr_idx] = scaler;
            curr_idx++;
        }
        idx += 128;
    }
    cb_push_back(cb_in_2, 1);
}

// HW-bcast scale for fused scale-attn-softmax
FORCE_INLINE void generate_inv_sqrt_hw_bcast_tile() {
    constexpr uint32_t scale_cb_id = 3;
    uint32_t u = get_arg_val<uint32_t>(2);
    cb_reserve_back(scale_cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(scale_cb_id));
    // for (int j = 0; j < 1024; j++)
    //     ptr[j] = uint16_t(0);
    ptr[0] = u>>16;
    cb_push_back(scale_cb_id, 1);
}

void kernel_main() {

    const uint32_t src_addr  = get_arg_val<uint32_t>(0);
    const uint32_t blk  = get_arg_val<uint32_t>(1);
    const uint32_t num_blks = get_arg_val<uint32_t>(3); // same arg index as in reader_unary and in reader_unary_transpose_wh_8bank
    const uint32_t tile_offset = get_arg_val<uint32_t>(4);
    const uint32_t Wt = get_arg_val<uint32_t>(5);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_in0 = 0, cb_id_in1 = 1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);

    #if FUSED_SCALE_MASK
    uint32_t Ht = get_arg_val<uint32_t>(6);
    uint32_t mask_addr = get_arg_val<uint32_t>(7);
    uint32_t start_ht = get_arg_val<uint32_t>(8);
    uint32_t start_mask_id = get_arg_val<uint32_t>(9);
    constexpr bool mask_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t cb_id_attn = 4;
    uint32_t mask_tile_bytes = get_tile_size(cb_id_attn);
    const DataFormat mask_data_format = get_dataformat(cb_id_attn);

    const InterleavedAddrGenFast<mask_is_dram> addr_mask = {
        .bank_base_address = mask_addr,
        .page_size = mask_tile_bytes,
        .data_format = mask_data_format
    };

    #if CAUSAL_MASK
    constexpr uint32_t num_tiles_causal_mask = get_compile_time_arg_val(2);
    uint32_t mask_start_ht = get_arg_val<uint32_t>(11);
    uint32_t mask_offset = get_arg_val<uint32_t>(12);

    uint32_t mask_id_offset = mask_offset;
    uint32_t mask_ht = mask_start_ht;
    #endif

    uint32_t ht = start_ht;
    uint32_t mask_id = start_mask_id;
    bool read_mask = true;
    generate_inv_sqrt_hw_bcast_tile();
    #endif

    const InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };


    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t i_tile = 0;
    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i<num_blks; ++i) {
        for (uint32_t j = 0; j < Wt; j += blk) {
            uint32_t rem = blk; // (i + blk > num_tiles) ? num_tiles - i : blk;
            cb_reserve_back(cb_id_in0, rem);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

            for (uint32_t r = 0; r<rem; ++r) {
                noc_async_read_tile(curr_tile, src_a, l1_write_addr); // TODO(AP): data type size
                curr_tile++;
                l1_write_addr += src0_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, rem);
        }

        #if FUSED_SCALE_MASK
        // Recall that the total attention tensor size in tiles is NC,1,Wt
        // For fused scale-mask softmax we write Wt attention tiles for every partHt*Wt
        // of slice of tensor that was assigned to our core, then we skip to next batch
        #if CAUSAL_MASK
        for (uint32_t j = 0; j < Wt; j += blk) {
            cb_reserve_back(cb_id_attn, blk);
            uint32_t l1_write_addr = get_write_ptr(cb_id_attn);
            for (uint32_t wb = 0; wb<blk; ++wb) {
                noc_async_read_tile(mask_id, addr_mask, l1_write_addr);
                l1_write_addr += mask_tile_bytes;
                ++mask_id;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_attn, blk);
        }
        ++ht;
        ++mask_ht;
        if (ht == Ht) {
            ht = 0;
            mask_ht = 0;
            mask_id_offset += num_tiles_causal_mask;
        } else if (mask_ht == Wt) {
            mask_ht = 0;
            mask_id = mask_id_offset;
        }
        #else
        if (read_mask) {
            for (uint32_t j = 0; j < Wt; j += blk) {
                // This is only executed every blk wts
                cb_reserve_back(cb_id_attn, blk);
                uint32_t l1_write_addr = get_write_ptr(cb_id_attn);
                for (uint32_t wb = 0; wb<blk; ++wb) {
                    noc_async_read_tile(mask_id, addr_mask, l1_write_addr);
                    l1_write_addr += mask_tile_bytes;
                    ++mask_id;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_attn, blk);
            }
            read_mask = false;
        }
        ++ht;
        if (ht == Ht) {
            ht = 0;
            read_mask = true;
        }
        #endif // CAUSAL_MASK

        #endif // FUSED_SCALE_MASK
    }
}
