// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
// #include "debug/dprint.h"


FORCE_INLINE void generate_zero_mask_fill_partial_rows(const uint32_t cb_id, const uint32_t num_rows, const uint32_t num_cols) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));

    uint32_t num_rows_to_fill = num_rows;

    for (uint32_t k = 0; k < 2; ++k) {
        uint32_t face_offset = k << 9;
        uint32_t row = 0;
        for (uint32_t i = 0; i < 256; i += 16) {
            if (row < num_rows_to_fill) {
                if (num_cols <= 16) {
                    for (uint32_t j = 0; j < num_cols; ++j) {
                        ptr[face_offset + i + j] = 0x3f80;
                    }
                    for (uint32_t j = num_cols; j < 16; ++j) {
                        ptr[face_offset + i + j] = 0;
                    }
                    for (uint32_t j = 0; j < 16; ++j) {
                        ptr[face_offset + 256 + i + j] = 0;
                    }
                } else {
                    for (uint32_t j = 0; j < 16; ++j) {
                        ptr[face_offset + i + j] = 0x3f80;
                    }

                    for (uint32_t j = 0; j < num_cols - 16; ++j) {
                        ptr[face_offset + 256 + i + j] = 0x3f80;
                    }
                    for (uint32_t j = num_cols - 16; j < 16; ++j) {
                        ptr[face_offset + 256 + i + j] = 0;
                    }
                }
            } else {
                for (uint32_t j = 0; j < 16; ++j) {
                    ptr[face_offset + i + j] = 0;
                }

                for (uint32_t j = 0; j < 16; ++j) {
                    ptr[face_offset + 256 + i + j] = 0;
                }
            }
            row++;
        }
        num_rows_to_fill -= 16;
    }

    cb_push_back(cb_id, 1);
}


FORCE_INLINE void generate_zero_mask_fill_full_rows(const uint32_t cb_id, const uint32_t num_rows) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));

    uint32_t num_rows_to_fill = num_rows;

    for (uint32_t k = 0; k < 2; ++k) {
        uint32_t face_offset = k << 9;
        uint32_t row = 0;
        for (uint32_t i = 0; i < 256; i += 16) {
            if (row < num_rows_to_fill) {
                for (uint32_t j = 0; j < 16; ++j) {
                    ptr[face_offset + i + j] = 0x3f80;
                }

                for (uint32_t j = 0; j < 16; ++j) {
                    ptr[face_offset + 256 + i + j] = 0x3f80;
                }
            } else {
                for (uint32_t j = 0; j < 16; ++j) {
                    ptr[face_offset + i + j] = 0;
                }

                for (uint32_t j = 0; j < 16; ++j) {
                    ptr[face_offset + 256 + i + j] = 0;
                }
            }
            row++;
        }
        num_rows_to_fill -= 16;
    }

    cb_push_back(cb_id, 1);
}


FORCE_INLINE void generate_zero_mask(const uint32_t cb_id, const uint32_t num_cols) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));

    if (num_cols <= 16) {
        for (uint32_t j = 0; j < num_cols; ++j) {
            ptr[j] = 0x3f80;
        }

        for (uint32_t j = num_cols; j < 16; ++j) {
            ptr[j] = 0;
        }

        for (uint32_t j = 0; j < 16; ++j) {
            ptr[256 + j] = 0;
        }
    } else {
        for (uint32_t j = 0; j < 16; ++j) {
            ptr[j] = 0x3f80;
        }

        for (uint32_t j = 0; j < num_cols - 16; ++j) {
            ptr[256 + j] = 0x3f80;
        }

        for (uint32_t j = num_cols - 16; j < 16; ++j) {
            ptr[256 + j] = 0;
        }
    }

    cb_push_back(cb_id, 1);
}


void kernel_main() {
    constexpr bool is_mcast_sender                  = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma                       = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta                        = get_compile_time_arg_val(2) == 1;
    constexpr bool gamma_is_dram                    = get_compile_time_arg_val(3) == 1;
    constexpr bool beta_is_dram                     = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_cols_tile_gamma_beta                      = get_compile_time_arg_val(5);
    constexpr uint32_t per_core_N                     = get_compile_time_arg_val(6);
    constexpr uint32_t is_num_channel_div_by_tile     = get_compile_time_arg_val(7);
    constexpr uint32_t num_cols_per_group         = get_compile_time_arg_val(8);
    constexpr uint32_t group_offset         = get_compile_time_arg_val(9);
    constexpr uint32_t num_nz_rows_per_tile         = get_compile_time_arg_val(10);
    constexpr uint32_t batch_offset       = get_compile_time_arg_val(11);
    constexpr uint32_t num_group                     = get_compile_time_arg_val(12);
    constexpr uint32_t num_batch                    = get_compile_time_arg_val(13);
    constexpr uint32_t block_h                        = get_compile_time_arg_val(14);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(15);
    constexpr uint32_t block_h_offset                        = get_compile_time_arg_val(16);
    constexpr uint32_t block_w_offset                        = get_compile_time_arg_val(17);

    // DPRINT << "num_cols_tile_gamma_beta " <<num_cols_tile_gamma_beta<<ENDL();
    // DPRINT << "is_num_channel_div_by_tile " <<is_num_channel_div_by_tile<<ENDL();
    // DPRINT << "num_cols_per_group " <<num_cols_per_group<<ENDL();
    // DPRINT << "num_nz_rows_per_tile " <<num_nz_rows_per_tile<<ENDL();
    // DPRINT << "num_group " <<num_group<<ENDL();
    // DPRINT << "num_batch " <<num_batch<<ENDL();
    // DPRINT << "group_offset " <<group_offset<<ENDL();
    // DPRINT << "batch_offset " <<batch_offset<<ENDL();

    // DPRINT << "block_h " <<block_h<<ENDL();
    // DPRINT << "block_w " <<block_w<<ENDL();
    // DPRINT << "block_h_offset " <<block_h_offset<<ENDL();
    // DPRINT << "block_w_offset " <<block_w_offset<<ENDL();

    const uint32_t gamma_addr                     = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr                      = get_arg_val<uint32_t>(4);
    const uint32_t gamma_tile_start_id            = get_arg_val<uint32_t>(5);
    const uint32_t beta_tile_start_id             = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_gamma = tt::CB::c_in5;
    constexpr uint32_t cb_beta = tt::CB::c_in6;
    constexpr uint32_t cb_im_out = tt::CB::c_intermed2;
    constexpr uint32_t cb_out = tt::CB::c_out0;
    constexpr uint32_t cb_zero_mask = tt::CB::c_intermed4;
    constexpr uint32_t cb_zero_mask_full_tile = tt::CB::c_intermed6;

    // constexpr uint32_t block_w = 4;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_gamma);

    {
        constexpr uint32_t cb_in_2 = tt::CB::c_in2;
        const uint32_t scalar_w = get_arg_val<uint32_t>(1);
        generate_reduce_scaler(cb_in_2, scalar_w);
    }
    if constexpr(is_mcast_sender) {
        constexpr uint32_t cb_in_4 = tt::CB::c_in4;
        const uint32_t scalar_c = get_arg_val<uint32_t>(0);
        generate_reduce_scaler(cb_in_4, scalar_c);
    }
    {
        constexpr uint32_t eps_cb_id = 3;
        const uint32_t eps = get_arg_val<uint32_t>(2);
        generate_bcast_col_scalar(eps_cb_id, eps);
    }
    if constexpr (num_nz_rows_per_tile < 32) { // less than tile height
        generate_zero_mask_fill_full_rows(cb_zero_mask_full_tile, num_nz_rows_per_tile);
        if constexpr (num_cols_per_group != 0) {
            generate_zero_mask_fill_partial_rows(cb_zero_mask, num_nz_rows_per_tile, num_cols_per_group);
        }
    } else if (num_cols_per_group != 0){
        generate_zero_mask(cb_zero_mask, num_cols_per_group);
    }

    #define stick_size_is_pow2 get_compile_time_arg_val(18) == 1
    #if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(19);
    #else
    constexpr uint32_t page_size = get_compile_time_arg_val(19);
    #endif

    if constexpr(fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        #if (stick_size_is_pow2)
        const InterleavedPow2AddrGen<gamma_is_dram> gamma = {
            .bank_base_address = gamma_addr,
            .log_base_2_of_page_size = log_base_2_of_page_size
        };
        #else
        const InterleavedAddrGen<gamma_is_dram> gamma = {
            .bank_base_address = gamma_addr,
            .page_size = page_size
        };
        #endif

        cb_reserve_back(cb_gamma, num_cols_tile_gamma_beta);
        uint32_t l1_write_addr_gamma = get_write_ptr(cb_gamma);
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            uint64_t gamma_noc_addr = get_noc_addr(tile_id, gamma);
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma, 32);
            gamma_noc_addr += 32;
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma + 512, 32);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, num_cols_tile_gamma_beta);
    }

    if constexpr(fuse_beta) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
        #if (stick_size_is_pow2)
        const InterleavedPow2AddrGen<beta_is_dram> beta = {
            .bank_base_address = beta_addr,
            .log_base_2_of_page_size = log_base_2_of_page_size
        };
        #else
        const InterleavedAddrGen<beta_is_dram> beta = {
            .bank_base_address = beta_addr,
            .page_size = page_size
        };
        #endif

        uint32_t l1_write_addr_beta = get_write_ptr(cb_beta);
        cb_reserve_back(cb_beta, num_cols_tile_gamma_beta);
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            uint64_t beta_noc_addr = get_noc_addr(tile_id, beta);
            noc_async_read(beta_noc_addr, l1_write_addr_beta, 32);
            beta_noc_addr += 32;
            noc_async_read(beta_noc_addr, l1_write_addr_beta + 512, 32);
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, num_cols_tile_gamma_beta);
    }

    volatile tt_l1_ptr uint16_t* wptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_out));
    uint32_t out_l1_write_addr = get_write_ptr(cb_out);
    // pick values from sharded input cb to cb
    uint32_t batch_index = 0;
    for (uint32_t i=0; i < num_batch; ++i) {
        uint32_t group_index = 0;
        for (uint32_t j=0; j < num_group; ++j) {
            uint32_t h_index = 0;
            for (uint32_t h=0; h < block_h; ++h) {
                uint32_t w_index = 0;
                for (uint32_t w=0; w < block_w; ++w) {
                    cb_wait_front(cb_im_out, 1);
                    uint32_t group_batch_index_offset = batch_index + group_index + h_index + w_index;

                    if (w == block_w - 1 and not is_num_channel_div_by_tile) {
                        volatile tt_l1_ptr uint16_t* rptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_im_out));
                        uint32_t write_l1_index = group_batch_index_offset;
                        uint32_t read_l1_index = 0;
                        for (uint32_t t=0; t < num_nz_rows_per_tile; ++t) {
                            for (uint32_t c=0; c < num_cols_per_group; ++c) {
                                wptr[c + write_l1_index] = rptr[c + read_l1_index];
                            }
                            read_l1_index += 32;
                            write_l1_index += per_core_N;
                        }
                    } else {
                        volatile tt_l1_ptr uint16_t* rptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_im_out));
                        uint32_t write_l1_index = group_batch_index_offset;
                        uint32_t read_l1_index = 0;
                        for (uint32_t t=0; t < num_nz_rows_per_tile; ++t) {
                            for (uint32_t c=0; c < 32; ++c) {
                                wptr[c + write_l1_index] = rptr[c + read_l1_index];
                            }
                            read_l1_index += 32;
                            write_l1_index += per_core_N;
                        }
                    }
                    cb_pop_front(cb_im_out, 1);
                    w_index += block_w_offset;
                }
                h_index += block_h_offset;
            }
            group_index += group_offset;
        }
        batch_index += batch_offset;
    }
}
