// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;

    uint32_t has_work = get_arg_val<uint32_t>(i++);
    const bool has_work_bool = has_work == 1;

    uint32_t src0_addr            = get_arg_val<uint32_t>(i++);
    uint32_t src1_addr            = get_arg_val<uint32_t>(i++);
    uint32_t Mt                   = get_arg_val<uint32_t>(i++);
    uint32_t Kt                   = get_arg_val<uint32_t>(i++);
    uint32_t Nt                   = get_arg_val<uint32_t>(i++);
    uint32_t MtKt                 = get_arg_val<uint32_t>(i++);
    uint32_t num_kv_heads         = get_arg_val<uint32_t>(i++); // in1[1] (ie. in1 C)
    uint32_t in1_KtNt             = get_arg_val<uint32_t>(i++);
    uint32_t in1_CKtNt_skip       = get_arg_val<uint32_t>(i++); // 0 if in0 and in1 Kt are the same
    uint32_t in1_CKtNt_mul_32     = get_arg_val<uint32_t>(i++);
    uint32_t blocks               = get_arg_val<uint32_t>(i++);
    uint32_t in0_start_id         = get_arg_val<uint32_t>(i++);
    uint32_t in1_start_id         = get_arg_val<uint32_t>(i++);
    uint32_t kv_heads_addr_offset = get_arg_val<uint32_t>(i++);

    uint32_t in1_mcast_dest_noc_start_x                  = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_dest_noc_start_y                  = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_dest_noc_end_x                    = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_dest_noc_end_y                    = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_num_dests                         = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_num_cores                         = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_sender_semaphore_addr             = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_receiver_semaphore_addr           = get_arg_val<uint32_t>(i++);

    uint32_t in1_mcast_sender_size_bytes                 = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_sender_id                         = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_sender_num_x                      = get_arg_val<uint32_t>(i++);
    uint32_t in1_mcast_sender_num_y                      = get_arg_val<uint32_t>(i++);
    volatile tt_l1_ptr uint32_t *in1_mcast_sender_noc_x  = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(i)); i+=in1_mcast_sender_num_x;
    volatile tt_l1_ptr uint32_t *in1_mcast_sender_noc_y  = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(i)); i+=in1_mcast_sender_num_y;


    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    #define transpose_hw_bool get_compile_time_arg_val(2) == 1
    constexpr bool row_major = (bool) get_compile_time_arg_val(3);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1; // copy single KV heads for Q heads
    constexpr uint32_t cb_id_in2 = 2; // mcast receiver
    constexpr uint32_t cb_id_in3 = 3; // all interleaved or sharded KV heads for one user batch
    constexpr uint32_t cb_id_intermed0 = 24;
    constexpr uint32_t cb_id_intermed1 = 25;
    constexpr uint32_t cb_id_intermed2 = 26;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_one_tile = 32;
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    #ifdef IN0_SHARDED
    if (has_work_bool) {
        cb_reserve_back(cb_id_in0, blocks * MtKt);
        cb_push_back(cb_id_in0, blocks * MtKt);
    }
    #else
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr,
        .page_size = in0_tile_bytes,
        .data_format = in0_data_format
    };
    #endif

    #ifndef IN1_SHARDED
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr,
        .page_size = in1_tile_bytes,
        .data_format = in1_data_format
    };
    #endif

    // Mcast setup
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);
    noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in1_mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_sender_semaphore_addr);

    uint64_t in1_mcast_sender_semaphore_noc_addr_vec[num_rows_in_one_tile];
    if constexpr(row_major) {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_rows_in_one_tile; ++i) {
            in1_mcast_sender_semaphore_noc_addr_vec[i] = get_noc_addr(in1_mcast_sender_noc_x[x], in1_mcast_sender_noc_y[y], in1_mcast_sender_semaphore_addr);
            ++x;
            if (x == in1_mcast_sender_num_x) {
                x = 0;
                ++y;
            }
        }
    } else {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_rows_in_one_tile; ++i) {
            in1_mcast_sender_semaphore_noc_addr_vec[i] = get_noc_addr(in1_mcast_sender_noc_x[x], in1_mcast_sender_noc_y[y], in1_mcast_sender_semaphore_addr);
            ++y;
            if (y == in1_mcast_sender_num_y) {
                y = 0;
                ++x;
            }
        }
    }

    uint64_t in1_multicast_noc_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x,
        in1_mcast_dest_noc_start_y,
        in1_mcast_dest_noc_end_x,
        in1_mcast_dest_noc_end_y,
        0
    );

    uint64_t in1_mcast_receiver_semaphore_noc_addr = in1_multicast_noc_addr | in1_mcast_receiver_semaphore_addr;


    // CB write ptr; no pop/push for cb 2 and 3 so write/read ptr's never change
    uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
    uint32_t l1_write_addr_in3 = get_write_ptr(cb_id_in3);
    uint64_t in1_multicast_data_addr = in1_multicast_noc_addr | l1_write_addr_in2;
    uint64_t noc_l1_read_addr_for_kv_heads = get_noc_addr(l1_write_addr_in2 + kv_heads_addr_offset);

    // TODO: Clean this up; don't think this will work if we double buffer intermed 1/2
    uint32_t cb_intermed1_addr_initial = get_read_ptr(cb_id_intermed1);
    uint32_t cb_intermed2_addr_initial = get_write_ptr(cb_id_intermed2);
    uint32_t cb_intermed1_addr;
    uint32_t cb_intermed2_addr;
    constexpr uint32_t bfloat16_row_bytes = 64;

    // Only used for interleaved
    uint32_t in0_batch = in0_start_id;
    uint32_t in1_batch;
    uint32_t in0_Mt;
    uint32_t in1_Nt;
    uint32_t in0_tensor_id;
    uint32_t in1_tensor_id;

    // Only used for sharded
    // Don't need to track batch because user batch must be 32 (ie. Mt must be 1)
    uint64_t in1_sharded_cb_noc_addr_Nt = get_noc_addr(l1_write_addr_in3);  // Read/write ptr should be the same
    uint64_t in1_sharded_cb_noc_addr;
    uint32_t Nt_bytes = Nt * in1_tile_bytes;
    uint32_t in1_KtNt_bytes = in1_KtNt * in1_tile_bytes;
    uint32_t in1_CKtNt_skip_bytes = in1_CKtNt_skip * in1_tile_bytes;
    for (uint32_t b = 0; b < blocks; b++) {
        in0_Mt = in0_batch;
        in1_batch = in1_start_id;

        for (uint32_t m = 0; m < Mt; m++) {
            in1_Nt = in1_batch;

            #ifndef IN0_SHARDED
            if (has_work_bool) {
                in0_tensor_id = in0_Mt;
                cb_reserve_back(cb_id_in0, Kt);
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    // Read in0 tile at (mt, kt)
                    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                    noc_async_read_tile(in0_tensor_id, s0, l1_write_addr_in0);
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in0, onetile);

                    in0_tensor_id++; // in0 is MK
                }
            }
            #endif

            for (uint32_t n = 0; n < Nt; n++) {
                cb_intermed1_addr = cb_intermed1_addr_initial;
                cb_intermed2_addr = cb_intermed2_addr_initial;
                in1_tensor_id = in1_Nt;
                in1_sharded_cb_noc_addr = in1_sharded_cb_noc_addr_Nt;

                if (has_work_bool) {
                    cb_reserve_back(cb_id_intermed2, 1);
                }
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
                    for (uint32_t kt = 0; kt < Kt; kt++) {
                        // Read in1 tile at (kt, nt)
                        if (tile_row_id == in1_mcast_sender_id) {
                            // MCAST SENDER: send all kv_heads in one user batch
                            #ifdef IN1_SHARDED
                            // Copy to cb_id_in2 to mcast
                            uint64_t in1_sharded_cb_current_noc_addr = in1_sharded_cb_noc_addr;
                            uint32_t in2_current_l1_write_addr = l1_write_addr_in2;
                            for (uint32_t kv_heads_id = 0; kv_heads_id < num_kv_heads; kv_heads_id++) {
                                noc_async_read(in1_sharded_cb_current_noc_addr, in2_current_l1_write_addr, in1_tile_bytes);
                                in1_sharded_cb_current_noc_addr += in1_KtNt_bytes; // Increment by Nt to get to next kv_heads
                                in2_current_l1_write_addr += in1_tile_bytes;
                            }
                            // These indices are local to each core, so don't modify when looping num_rows_in_one_tile
                            in1_sharded_cb_noc_addr += Nt_bytes; // Kt is in in1[2], so stride is Nt
                            noc_async_read_barrier();
                            #else
                            uint32_t in1_tensor_current_id = in1_tensor_id;
                            uint32_t in2_current_l1_write_addr = l1_write_addr_in2;
                            for (uint32_t kv_heads_id = 0; kv_heads_id < num_kv_heads; kv_heads_id++) {
                                noc_async_read_tile(in1_tensor_current_id, s1, in2_current_l1_write_addr);

                                in1_tensor_current_id += in1_KtNt; // Increment by KtNt to get to next kv_heads
                                in2_current_l1_write_addr += in1_tile_bytes;
                            }
                            noc_async_read_barrier();
                            #endif

                            // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value should be in1_mcast_num_dests), then reset
                            // the semaphore_addr value back to zero for the next block
                            noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
                            noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

                            // Now we have the block in the CB address, we can mcast to dests!
                            // num_dests will source, since we are copying to a different local CB as well
                            noc_async_write_multicast_loopback_src(l1_write_addr_in2, in1_multicast_data_addr, in1_mcast_sender_size_bytes, in1_mcast_num_cores + 1);

                            // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same cmd_buf
                            // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

                            // We should also multicast VALID flag to destinations for receiver semaphore
                            noc_semaphore_set_multicast(in1_mcast_receiver_semaphore_addr, in1_mcast_receiver_semaphore_noc_addr, in1_mcast_num_cores);

                            // Write barrier needed since we mcast to self, and also needed to finish sending mcast flag before we modify locally
                            noc_async_write_barrier();
                        } else {
                            // MCAST RECEIVER: receive all kv_heads in one user batch
                            // Set in1 semaphore value to INVALID
                            noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

                            // Atomic increment source core counter
                            uint64_t in1_mcast_sender_semaphore_noc_addr = in1_mcast_sender_semaphore_noc_addr_vec[tile_row_id];
                            noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

                            // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
                            noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);
                        }
                        if (has_work_bool) {
                            // Choose matching kv_heads for q_heads
                            cb_reserve_back(cb_id_in1, onetile);
                            noc_async_read(noc_l1_read_addr_for_kv_heads, get_write_ptr(cb_id_in1), in1_tile_bytes);
                            noc_async_read_barrier();
                            cb_push_back(cb_id_in1, onetile);
                        }

                        #if (transpose_hw_bool)
                        in1_tensor_id++; // Kt is in in1[3], so it is contiguous in memory
                        #else
                        in1_tensor_id += Nt; // Kt is in in1[2], so stride is Nt
                        #endif
                    } // Kt loop

                    if (has_work_bool) {
                        // Read 32 untilized tiles and select correct rows to reconstruct single correct tile
                        cb_wait_front(cb_id_intermed1, 1);
                        noc_async_read(get_noc_addr(cb_intermed1_addr), cb_intermed2_addr, bfloat16_row_bytes);
                        noc_async_read_barrier();
                        cb_pop_front(cb_id_intermed1, 1);
                        cb_intermed1_addr += bfloat16_row_bytes;
                        cb_intermed2_addr += bfloat16_row_bytes;
                    }

                    in1_tensor_id += in1_CKtNt_skip; // different depending on transpose_hw
                } // 32 tiles loop

                if (has_work_bool) {
                    cb_push_back(cb_id_intermed2, 1);
                }

                // Next tile in Nt
                #if (transpose_hw_bool)
                in1_Nt += Kt; // next tile in Nt is in in1[2], so stride is Kt
                #else
                in1_Nt++;
                #endif

                in1_sharded_cb_noc_addr_Nt += in1_tile_bytes;
            } // Nt loop

            in0_Mt += Kt;
            // here, KtNt is the stride of the full in1 tensor (ie. max cache length is incorporated in one of Kt or Nt depending on transpose_hw)
            in1_batch += in1_CKtNt_mul_32; // different depending on transpose_hw
        } // Mt loop
        in0_batch += MtKt;
    } // B loop
}
