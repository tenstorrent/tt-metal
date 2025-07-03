// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// Zero out all tiles for a given circular buffer.
template <uint32_t cb_id>
FORCE_INLINE void zero_out_tiles() {
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    static_assert(
        tile_size % MEM_ZEROS_SIZE == 0, "Tile size must be a multiple of MEM_ZEROS_BASE for zeroing out tiles");
    const uint32_t num_tiles = get_local_cb_interface(cb_id).fifo_num_pages;
    const uint32_t num_zeros_reads = (tile_size / MEM_ZEROS_SIZE) * num_tiles;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_write_barrier();
}

template <
    uint32_t dilation_w,
    uint32_t coalesced_read_bytes,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t stride_w_bytes,
    uint32_t weight_size_w>
FORCE_INLINE void read_sticks(
    uint32_t act_block_h_datums_read_curr,
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr,
    uint32_t reader_offset,
    uint32_t& l1_write_addr_act,
    uint32_t& reader_idx) {
    for (uint32_t bhd = 0; bhd < act_block_h_datums_read_curr; bhd++) {
        // local read from reader_index + reader_offset;
        uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];
        uint32_t reader_idx_1 = two_reader_indices & 0xffff;
        uint32_t reader_idx_2 = two_reader_indices >> 16;

        if constexpr (dilation_w == 1) {
            uint32_t act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
            noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
            l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);

            act_l1_offset = reader_offset + (reader_idx_2 * conv_act_c_read_bytes);
            noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
            l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
        } else {
            uint32_t act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
            for (uint32_t inner = 0; inner < weight_size_w; inner++) {
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                l1_write_addr_act += conv_act_c_read_bytes;
                act_l1_offset += stride_w_bytes;
            }
            l1_write_addr_act += act_block_w_extra_align_bytes;

            act_l1_offset = reader_offset + (reader_idx_2 * conv_act_c_read_bytes);
            for (uint32_t inner = 0; inner < weight_size_w; inner++) {
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                l1_write_addr_act += conv_act_c_read_bytes;
                act_l1_offset += stride_w_bytes;
            }
            l1_write_addr_act += act_block_w_extra_align_bytes;
        }
        reader_idx++;
    }
}

void kernel_main() {
    constexpr uint32_t dilation_h = get_compile_time_arg_val(0);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(1);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(2);
    // need to have these as compile-time, they are inner loop bouds / unroll loops / constexpr conditionals based on
    // them
    constexpr uint32_t window_outer = get_compile_time_arg_val(3);
    constexpr uint32_t window_inner = get_compile_time_arg_val(4);
    constexpr uint32_t act_block_h_datums = get_compile_time_arg_val(5);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(8);
    constexpr uint32_t conv_act_size_w_padded = get_compile_time_arg_val(9);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(11);
    constexpr uint32_t act_block_h_datums_last_block = get_compile_time_arg_val(20);

    constexpr uint32_t act_block_h_datums_read_last_block =
        act_block_h_datums_last_block > act_block_h_datums ? act_block_h_datums / 2 : act_block_h_datums_last_block / 2;
    constexpr uint32_t act_block_h_datums_second_reader = get_compile_time_arg_val(21);
    constexpr uint32_t act_block_h_datums_second_reader_read = act_block_h_datums_second_reader / 2;
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(22) == 1;
    constexpr uint32_t cb_id_act = get_compile_time_arg_val(23);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(24);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(25);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(31);
    constexpr uint32_t transaction_size_bytes = get_compile_time_arg_val(32);
    constexpr uint32_t test_id = get_compile_time_arg_val(33);

    uint32_t i = 0;
    uint32_t noop = get_arg_val<uint32_t>(i);
    i += 1;

    if (noop) {
        return;
    }

    // if constexpr (needs_act_block_zero_out) {
    //     zero_out_tiles<cb_id_act>();
    // }

    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;

    constexpr uint32_t act_block_h_datums_read = act_block_h_datums / 2;  // Extra /2 because of packed uint16 reads

    // LOOP TO FILL READER INDICES
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(26));

    uint32_t reader_idx = 0;

    // TODO: need to make the read coalescing optimization cleaner
    // pass coalesce_window_inner_reads as a compile time arg and num_coalesced_reads so we can constexpr the if
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side we check if window_inner == weight_size_w to make sure coalescing is legal along full window_inner
    // so the loop can be removed
    constexpr bool coalesce_window_inner_reads = true;
    constexpr uint32_t num_coalesced_reads = weight_size_w;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? num_coalesced_reads * conv_act_c_read_bytes : conv_act_c_read_bytes);
    // the conditional selecting between coalescing and no-colescing must be constexpr to that compiler can optimized
    // the other path away this has shown to be a big perf win

    // coalesce reads along weight_size_w
    uint32_t act_l1_read_addr = get_compile_time_arg_val(27);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV1");
        static_assert(coalesced_read_bytes <= NOC_MAX_BURST_SIZE);
        // set_state uses just x/y from the get_noc_addr, addr is ignored
        noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);

        constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
        uint32_t start_reader_idx = 0;
        for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
            uint32_t reader_offset = act_l1_read_addr;
            for (uint32_t outer = 0; outer < window_outer; outer++) {
                // Reset reader_idx to finish act_block_h_datums
                reader_idx = start_reader_idx;

                uint32_t l1_write_addr_act = get_compile_time_arg_val(29);

                uint32_t act_block_h_datums_read_curr =
                    bh == act_num_blocks_h - 1 ? act_block_h_datums_read_last_block : act_block_h_datums_read;

                read_sticks<
                    dilation_w,
                    coalesced_read_bytes,
                    conv_act_c_read_bytes,
                    act_block_w_extra_align_bytes,
                    stride_w_bytes,
                    weight_size_w>(
                    act_block_h_datums_read_curr,
                    packed_reader_indices_ptr,
                    reader_offset,
                    l1_write_addr_act,
                    reader_idx);

                noc_async_read_barrier();

                reader_offset += window_outer_offset;
            }

            start_reader_idx = reader_idx;
#ifdef SPLIT_READER
            start_reader_idx += act_block_h_datums_second_reader_read;
#endif
        }
        noc_async_write_barrier();
    }
}
