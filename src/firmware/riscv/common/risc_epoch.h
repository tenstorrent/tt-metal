#ifndef _RISC_EPOCH_H_
#define _RISC_EPOCH_H_

#include <stdint.h>

#include "risc_common.h"
#include "noc_parameters.h"
#include "noc_overlay_parameters.h"
#include "noc_wrappers.h"
#include "epoch.h"
#include "risc.h"
#include "stream_interface.h"
#include "risc_chip_specific.h"
#include "tensix.h"
#include "epoch_q.h"
#include "dram_stream_intf.h"

#define SCRATCH_PAD_DRAM_READ_IDX 0
#define SCRATCH_PAD_DRAM_WRITE_IDX 16

#define EPOCH_EMPTY_CHECK_MASK ((0x1 << 12) - 1)

#define MAX_DRAM_QUEUES_TO_UPDATE 256

inline void risc_epoch_q_rdptr_update(uint32_t rd_ptr, volatile uint32_t *noc_read_scratch_buf, uint64_t &my_q_table_offset) {
  uint32_t noc_write_dest_buf_offset = my_q_table_offset % (NOC_WORD_BYTES);
  uint32_t noc_write_dest_buf_addr = (uint32_t)(&(noc_read_scratch_buf[SCRATCH_PAD_DRAM_WRITE_IDX]));
  uint32_t noc_write_dest_buf_ptr_addr = noc_write_dest_buf_addr+noc_write_dest_buf_offset;
  volatile uint32_t *noc_write_dest_buf_ptr = (volatile uint32_t *)(noc_write_dest_buf_ptr_addr + epoch_queue::EPOCH_Q_RDPTR_OFFSET); 
  *noc_write_dest_buf_ptr = rd_ptr;
  uint64_t q_rd_ptr_addr = my_q_table_offset + epoch_queue::EPOCH_Q_RDPTR_OFFSET;
  RISC_POST_DEBUG(0x10000009);
  RISC_POST_DEBUG(rd_ptr);
  RISC_POST_DEBUG(noc_write_dest_buf_ptr_addr);
  RISC_POST_DEBUG(q_rd_ptr_addr >> 32);
  RISC_POST_DEBUG(q_rd_ptr_addr & 0xFFFFFFFF);
  // Reg poll loop, flushed immediately
  while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_REG_CMD_BUF));
  ncrisc_noc_fast_write(loading_noc, NCRISC_WR_REG_CMD_BUF, noc_write_dest_buf_ptr_addr, q_rd_ptr_addr, 4,
                        DRAM_PTR_UPDATE_VC, false, false, 1);
}

inline uint64_t risc_get_epoch_q_dram_ptr(uint32_t my_x, uint32_t my_y) {
  
  const uint64_t INITIAL_EPOCH_VECTOR_TABLE_ADDR = NOC_XY_ADDR(NOC_X(get_epoch_table_x(my_x, my_y)), NOC_Y(get_epoch_table_y(my_x, my_y)), epoch_queue::EPOCH_TABLE_DRAM_ADDR);
  uint64_t q_table_offset = INITIAL_EPOCH_VECTOR_TABLE_ADDR + ((my_y * epoch_queue::GridSizeCol) + my_x)*epoch_queue::EPOCH_TABLE_ENTRY_SIZE_BYTES;
  return q_table_offset;
}

inline void risc_epoch_q_get_ptr(volatile uint32_t *noc_read_scratch_buf, uint64_t &my_q_table_offset, uint32_t &my_q_rd_ptr, uint32_t &my_q_wr_ptr) {

  uint32_t noc_read_dest_buf_offset = my_q_table_offset % (NOC_WORD_BYTES);
  uint32_t noc_read_dest_buf_addr = (uint32_t)(noc_read_scratch_buf);
  uint32_t noc_read_dest_buf_ptr_addr = noc_read_dest_buf_addr+noc_read_dest_buf_offset;
  ncrisc_noc_fast_read_any_len(loading_noc, NCRISC_RD_CMD_BUF,
                               my_q_table_offset,
                               noc_read_dest_buf_ptr_addr,
                               8);
  while (!ncrisc_noc_reads_flushed(loading_noc));
  volatile uint32_t *noc_read_dest_buf_ptr = (volatile uint32_t *)(noc_read_dest_buf_ptr_addr); 
  my_q_rd_ptr = noc_read_dest_buf_ptr[0];
  my_q_wr_ptr = noc_read_dest_buf_ptr[1];
}

inline bool risc_is_epoch_q_empty(volatile uint32_t *noc_read_scratch_buf, uint64_t &my_q_table_offset, uint32_t &my_q_rd_ptr, uint32_t &my_q_wr_ptr, uint32_t &epoch_empty_check_cnt) {

  bool is_empty;

  if (dram_io_empty(my_q_rd_ptr, my_q_wr_ptr)) {
    if (epoch_empty_check_cnt == 0) {
      uint32_t noc_read_dest_buf_offset = my_q_table_offset % (NOC_WORD_BYTES);
      uint32_t noc_read_dest_buf_addr = (uint32_t)(noc_read_scratch_buf);
      uint32_t noc_read_dest_buf_ptr_addr = noc_read_dest_buf_addr+noc_read_dest_buf_offset;
      ncrisc_noc_fast_read_any_len(loading_noc, NCRISC_RD_CMD_BUF,
                                  my_q_table_offset,
                                  noc_read_dest_buf_ptr_addr,
                                  8);
      while (!ncrisc_noc_reads_flushed(loading_noc));
      volatile uint32_t *noc_read_dest_buf_ptr = (volatile uint32_t *)(noc_read_dest_buf_ptr_addr); 
      my_q_wr_ptr = noc_read_dest_buf_ptr[1];
      is_empty = dram_io_empty(my_q_rd_ptr, my_q_wr_ptr);
    } else {
      is_empty = true;
    }
  } else {
    is_empty = false;
  }

  epoch_empty_check_cnt = (epoch_empty_check_cnt + 1) & EPOCH_EMPTY_CHECK_MASK;

  return is_empty;
}

inline void risc_get_noc_addr_from_dram_ptr(volatile uint32_t *noc_read_dest_buf_ptr, uint64_t& dram_addr_offset, uint32_t& dram_coord_x, uint32_t& dram_coord_y) {
  uint64_t dram_addr_offset_lo = noc_read_dest_buf_ptr[0];
  uint64_t dram_addr_offset_hi = noc_read_dest_buf_ptr[1] & 0xFFFF;
  dram_addr_offset = dram_addr_offset_lo | (dram_addr_offset_hi << 32);
  dram_coord_x = (noc_read_dest_buf_ptr[1] >> 16) & 0x3F;
  dram_coord_y = (noc_read_dest_buf_ptr[1] >> 22) & 0x3F;
}

inline __attribute__((section("code_l1"))) void risc_get_noc_addr_from_dram_ptr_l1(volatile uint32_t *noc_read_dest_buf_ptr, uint64_t& dram_addr_offset, uint32_t& dram_coord_x, uint32_t& dram_coord_y) {
  uint64_t dram_addr_offset_lo = noc_read_dest_buf_ptr[0];
  uint64_t dram_addr_offset_hi = noc_read_dest_buf_ptr[1] & 0xFFFF;
  dram_addr_offset = dram_addr_offset_lo | (dram_addr_offset_hi << 32);
  dram_coord_x = (noc_read_dest_buf_ptr[1] >> 16) & 0x3F;
  dram_coord_y = (noc_read_dest_buf_ptr[1] >> 22) & 0x3F;
}

inline uint64_t risc_get_epoch_dram_ptr(uint32_t &epoch_command, volatile uint32_t *noc_read_scratch_buf, uint64_t &my_q_table_offset, uint32_t &my_q_rd_ptr, uint32_t &epoch_empty_check_cnt) {

  uint64_t my_q_entry_offset = (my_q_rd_ptr & (epoch_queue::EPOCH_Q_NUM_SLOTS-1)) * epoch_queue::EPOCH_Q_SLOT_SIZE + epoch_queue::EPOCH_Q_SLOTS_OFFSET + my_q_table_offset;
  uint32_t noc_read_dest_buf_offset = my_q_entry_offset % (NOC_WORD_BYTES);
  uint32_t noc_read_dest_buf_addr = (uint32_t)(noc_read_scratch_buf);
  uint32_t noc_read_dest_buf_ptr_addr = noc_read_dest_buf_addr+noc_read_dest_buf_offset;
  ncrisc_noc_fast_read_any_len(loading_noc, NCRISC_RD_CMD_BUF,
                               my_q_entry_offset,
                               noc_read_dest_buf_ptr_addr,
                               32);
  while (!ncrisc_noc_reads_flushed(loading_noc));
  volatile uint32_t *noc_read_dest_buf_ptr = (uint32_t *)(noc_read_dest_buf_ptr_addr); 
  uint64_t dram_addr_offset;
  uint32_t dram_coord_x;
  uint32_t dram_coord_y;
  risc_get_noc_addr_from_dram_ptr(noc_read_dest_buf_ptr, dram_addr_offset, dram_coord_x, dram_coord_y);
  epoch_command = (noc_read_dest_buf_ptr[1] >> 28) & 0xF;
  return NOC_XY_ADDR(NOC_X(dram_coord_x), NOC_Y(dram_coord_y), dram_addr_offset);
}

inline __attribute__((section("code_l1"))) void risc_get_epoch_update_info(epoch_queue::IOQueueUpdateCmdInfo &queue_update_info, volatile uint32_t *noc_read_scratch_buf, uint64_t &my_q_table_offset, uint32_t &my_q_rd_ptr, uint64_t dram_next_epoch_ptr) {

  volatile epoch_queue::IOQueueUpdateCmdInfo* update_info;

  uint64_t my_q_entry_offset = (my_q_rd_ptr & (epoch_queue::EPOCH_Q_NUM_SLOTS-1)) * epoch_queue::EPOCH_Q_SLOT_SIZE + epoch_queue::EPOCH_Q_SLOTS_OFFSET + my_q_table_offset;
  uint32_t noc_read_dest_buf_offset = my_q_entry_offset % (NOC_WORD_BYTES);
  uint32_t noc_read_dest_buf_addr = (uint32_t)(noc_read_scratch_buf);
  uint32_t noc_read_dest_buf_ptr_addr = noc_read_dest_buf_addr+noc_read_dest_buf_offset;

  update_info = (volatile epoch_queue::IOQueueUpdateCmdInfo*)noc_read_dest_buf_ptr_addr;

  queue_update_info.queue_header_addr = update_info->queue_header_addr;
  queue_update_info.num_buffers = update_info->num_buffers;
  queue_update_info.reader_index = update_info->reader_index;
  queue_update_info.num_readers = update_info->num_readers;
  queue_update_info.update_mask = update_info->update_mask;
  queue_update_info.header[0] = update_info->header[0];
  queue_update_info.header[1] = update_info->header[1];
  queue_update_info.header[2] = update_info->header[2];
  queue_update_info.header[3] = update_info->header[3];
  queue_update_info.header[4] = update_info->header[4];
}

void run_epoch(
    void (*risc_epoch_load)(uint64_t), void (*risc_kernels_load)(uint64_t), void (*init_ncrisc_streams)(),
    bool skip_initial_epoch_dram_load, uint64_t dram_next_epoch_ptr, bool& skip_kernels, uint32_t& epoch_empty_check_cnt,
#ifdef RISC_GSYNC_ENABLED
    volatile uint32_t &gsync_epoch, volatile uint32_t &epochs_in_progress,
#endif
    uint32_t &num_dram_input_streams, uint32_t &num_dram_output_streams, uint32_t &num_active_streams, uint32_t &num_active_dram_queues, uint32_t &num_dram_prefetch_streams,
    dram_q_state_t *dram_q_state, dram_input_stream_state_t *dram_input_stream_state, dram_output_stream_state_t *dram_output_stream_state, active_stream_info_t *active_stream_info,
    volatile epoch_stream_info_t* *dram_prefetch_epoch_stream_info, volatile active_stream_info_t* *dram_prefetch_active_stream_info
);
void __attribute__((section("code_l1"))) run_dram_queue_update(
    void * pFunction, volatile uint32_t *noc_read_scratch_buf, uint64_t& my_q_table_offset, uint32_t& my_q_rd_ptr, uint64_t& dram_next_epoch_ptr, uint8_t& loading_noc
);

#endif

