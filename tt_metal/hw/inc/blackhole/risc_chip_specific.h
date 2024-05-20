// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "noc_parameters.h"
#include "noc_nonblocking_api.h"
#include "risc.h"
#include "unpack_pack_stream_intf.h"
#include "dram_stream_intf.h"
#include "risc_common.h"
#include "epoch.h"

////

const uint32_t PTR_UPDATE_TYPE_WR_PTR_UPDATE = 1 << 23;
const uint32_t PTR_UPDATE_TYPE_EPOCH_W_STRIDE = 1 << 23;
const uint32_t PTR_UPDATE_TYPE_EPOCH = 1 << 22;
const uint32_t PTR_UPDATE_TYPE_STRIDE = 1 << 21;
const uint32_t PTR_UPDATE_TYPE_DRAM_OUTPUT_STREAM_STATE = 1 << 23;

const uint32_t PTR_UPDATE_REG_WR_PTR_UPDATE = 1;
const uint32_t PTR_UPDATE_REG_TYPE = 2;
const uint32_t PTR_UPDATE_REG_STRIDE_WRAP = 3;
const uint32_t PTR_UPDATE_REG_STRIDE = 4;
const uint32_t PTR_UPDATE_REG_DRAM_OUTPUT_STREAM_STATE = 5;

const uint32_t CYCLES_SINCE_LAST_STREAM_DRAM_WRITE_THRESH = 650;

const uint32_t DRAM_HEADER_LAST = 7; // last byte of the header
const uint32_t PACKET_END_MARKER = 0xabcd1234;

const uint32_t DRAM_STREAM_1 = 8;
const uint32_t DRAM_STREAM_2 = 9;

void init_tile_clear();
void wait_till_tile_clear_done(uint32_t stream_id);
void process_tile_clearing(kernel_input_stream_state_t* input_stream_state, uint32_t streams_to_clear);

int get_epoch_table_x(int my_x, int my_y) __attribute__((const));
int get_epoch_table_y(int my_x, int my_y) __attribute__((const));
int get_epoch_index_x(int my_x) __attribute__((const));
int get_epoch_index_y(int my_y) __attribute__((const));

inline __attribute__((always_inline)) uint16_t op_pack_tiles_ptr_add(uint16_t a, uint16_t b) {
//#ifdef RISC_B0_HW // FIXME: This cahnge isnt supported in kernels yet, reenable when supported by kernels
//  return (a + b) & 0x3FF;
//#else
  return a + b;
//#endif
}

inline __attribute__((always_inline)) uint16_t op_pack_tiles_ptr_sub(uint16_t a, uint16_t b) {
//#ifdef RISC_B0_HW // FIXME: This cahnge isnt supported in kernels yet, reenable when supported by kernels
//  return (a - b) & 0x3FF;
//#else
  return a - b;
//#endif
}

inline __attribute__((always_inline)) bool addr_is_pcie(uint64_t dram_ptr_addr) {
  uint32_t x = NOC_UNICAST_ADDR_X(dram_ptr_addr);
  uint32_t y = NOC_UNICAST_ADDR_Y(dram_ptr_addr);
  return x == 0 && y == 3;
}

inline void set_noc_trans_table(uint32_t noc, uint8_t& noc_trans_table_en, uint8_t& my_logical_x, uint8_t& my_logical_y) {
  noc_trans_table_en = (NOC_CFG_READ_REG(noc, NIU_CFG_0) >> NIU_CFG_0_NOC_ID_TRANSLATE_EN) & 0x1;

  uint32_t noc_id_logical_reg = NOC_CFG_READ_REG(noc, NOC_ID_LOGICAL);
  my_logical_x = noc_id_logical_reg & NOC_NODE_ID_MASK;
  my_logical_y = (noc_id_logical_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
}

inline __attribute__((always_inline)) bool check_packet_end_marker(uint32_t l1_addr) {
  return false;
}

inline __attribute__((always_inline)) void set_packet_end_marker(uint32_t l1_addr) {
}

inline __attribute__((always_inline)) bool header_reads_flushed(uint32_t noc, uint32_t transaction_id, volatile uint32_t tt_l1_ptr * l1_ptr_addr) {
  return (ncrisc_noc_reads_flushed(noc, transaction_id) || check_packet_end_marker((uint32_t)(&(l1_ptr_addr[DRAM_HEADER_LAST]))));
}

inline __attribute__((always_inline)) void dram_input_stream_issue_scatter_read_init(uint32_t data_rec_chunk_size_tiles, uint32_t dram_io_scatter_chunk_size_tiles, uint32_t dram_io_scatter_chunk_size_bytes, uint32_t stream_dest_addr, uint32_t& transaction_id) {
  if (transaction_id == NCRISC_RD_END_TRID) {
    transaction_id = NCRISC_RD_START_TRID;
  } else {
    transaction_id += 1;
  }
}

inline __attribute__((always_inline)) bool dram_input_stream_check_next_chunk_flushed(uint32_t input_noc, uint32_t chunk_pending_start_addr, uint32_t chunk_size_bytes, uint32_t scatter_chunk_size_bytes, uint32_t& transaction_id) {
  uint32_t transaction_id_temp = transaction_id;
  if (transaction_id_temp == NCRISC_RD_END_TRID) {
    transaction_id_temp = NCRISC_RD_START_TRID;
  } else {
    transaction_id_temp += 1;
  }
  bool reads_flushed = ncrisc_noc_reads_flushed(input_noc, transaction_id_temp);
  if (reads_flushed) {
    transaction_id = transaction_id_temp;
  }
  return reads_flushed;
}

inline __attribute__((always_inline)) uint32_t get_total_in_flight_tiles(dram_output_stream_state_t* curr_dram_output_stream_state) {
#ifdef RISC_B0_HW
  uint32_t total_in_flight_tiles = 0;
  if (curr_dram_output_stream_state->moves_raw_data) {
    total_in_flight_tiles = curr_dram_output_stream_state->in_flight_tiles;
  } else {
    total_in_flight_tiles = curr_dram_output_stream_state->in_flight_tiles + curr_dram_output_stream_state->in_flight_tiles_2;
  }
#else
  uint32_t total_in_flight_tiles = curr_dram_output_stream_state->in_flight_tiles;
#endif

  return total_in_flight_tiles;
}

void risc_wait_for_cmd_buf(uint32_t noc, uint32_t cmd_buf);
void risc_dram_write_init(uint32_t dram_stream);
void risc_dram_write (uint32_t dram_writes_with_cmd_buf, uint32_t dram_stream, uint32_t noc, uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes, uint32_t len_tiles, uint32_t vc, uint32_t stream_msg_info_buf_addr, uint32_t transaction_id);
bool risc_dram_write_ok(uint32_t dram_writes_with_cmd_buf, uint32_t dram_stream, uint32_t output_noc);
bool risc_dram_writes_sent(uint32_t dram_writes_with_cmd_buf, uint32_t dram_stream);
void replicate(uint32_t noc_id, uint32_t src_addr, uint64_t dest_addr, uint32_t chunk_size_bytes, uint32_t times_to_replicate, uint32_t transaction_id);
void __attribute__((section("code_l1"))) replicate_l1(uint32_t noc_id, uint32_t src_addr, uint64_t dest_addr, uint32_t chunk_size_bytes, uint32_t times_to_replicate, uint32_t transaction_id);
bool has_pending_dram_write_ptrs(uint32_t dram_stream);
void write_pending_dram_write_ptrs(uint32_t dram_stream, dram_output_stream_state_t *dram_output_stream_state_base);
void set_pending_dram_write_ptrs(uint32_t dram_stream, uint32_t dram_writes_with_cmd_buf, bool is_ram, bool is_strided_write, uint32_t write_stride, uint32_t total_write_strides, uint32_t dram_wrptr_q_slots, uint32_t output_noc, uint32_t output_vc,
                                 uint64_t dram_buf_addr, dram_output_stream_state_t* curr_dram_output_stream_state, uint32_t curr_dram_output_stream_state_idx, volatile dram_io_state_t tt_l1_ptr * l1_ptrs, uint32_t curr_stride_wrap, uint32_t next_stride_wrap);
void process_dram_write(
  uint32_t &num_dram_output_streams, dram_output_stream_state_t *dram_output_stream_state, uint32_t &dram_ptr_update_cnt, uint32_t &total_tiles_to_clear
);
void process_dram_write_clear(uint32_t &num_dram_output_streams, dram_output_stream_state_t *dram_output_stream_state, uint32_t& total_tiles_to_clear);
void __attribute__((section("code_l1"))) __attribute__((noinline)) process_dram_write_moves_raw_data_l1(dram_output_stream_state_t* curr_dram_output_stream_state, dram_q_state_t tt_l1_ptr * next_dram_q_issue, uint32_t stream_id,
                                                                                                     uint16_t data_send_chunk_size_tiles, uint32_t output_vc, uint32_t data_send_chunk_size_bytes, uint64_t dram_buf_addr,
                                                                                                     uint32_t& stream_rd_ptr_byte, uint32_t dram_buf_size_bytes, bool& full_q_slot_sent);
