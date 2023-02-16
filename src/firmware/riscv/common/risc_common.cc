
#include "risc_common.h"
#include "noc_nonblocking_api.h"
#include "epoch.h"
#include "stream_interface.h"

#ifdef RISC_GSYNC_ENABLED
void global_sync_init(volatile uint32_t &gsync_epoch, volatile uint32_t &epochs_in_progress) {

  gsync_epoch = 0xFFFFFFFF;
  epochs_in_progress = 0;

}

// For this to work it is assumed that core 1-1 will do all epochs
void global_sync(volatile uint32_t &gsync_epoch, volatile uint32_t &epochs_in_progress) {
  if (my_x[0] == 1 && my_y[0] == 1) {
 
    while (epochs_in_progress);

    gsync_epoch = RISC_EPOCH_INFO_PTR->active_streams[0]->epoch_start_phase >> 10;

    uint64_t dest_addr = NOC_MULTICAST_ADDR(0, 0, noc_size_x-1, noc_size_y-1, ((uint32_t)(&gsync_epoch))); // NOC id conversion isnt necessary since we target entire grid
    while (!ncrisc_noc_fast_write_ok_l1(loading_noc, NCRISC_WR_REG_CMD_BUF));
    ncrisc_noc_fast_write_l1(loading_noc, NCRISC_WR_REG_CMD_BUF, ((uint32_t)(&gsync_epoch)), dest_addr, 4,
                          4, true, false, NUM_TENSIXES-1);

  } else {

    while (gsync_epoch == 0xFFFFFFFF);
    while (gsync_epoch < (RISC_EPOCH_INFO_PTR->active_streams[0]->epoch_start_phase >> 10));

    uint64_t dest_addr = NOC_XY_ADDR(NOC_X(1), NOC_Y(1), ((uint32_t)(&epochs_in_progress)));
    while (!ncrisc_noc_fast_write_ok_l1(loading_noc, NCRISC_AT_CMD_BUF));
    noc_fast_atomic_increment_l1(loading_noc, NCRISC_AT_CMD_BUF, dest_addr, 1, 31, false); // Inc by 1

  }
}

void global_sync_update(volatile uint32_t &gsync_epoch, volatile uint32_t &epochs_in_progress) {
  if (!(my_x[0] == 1 && my_y[0] == 1)) {

    uint64_t dest_addr = NOC_XY_ADDR(NOC_X(1), NOC_Y(1), ((uint32_t)(&epochs_in_progress)));
    while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_AT_CMD_BUF));
    noc_fast_atomic_increment(loading_noc, NCRISC_AT_CMD_BUF, dest_addr, 0xffffffff, 31, false); // dec by 1

  }
}
#endif

void risc_reset_check()
{
  volatile uint32_t *risc_reset_req = (volatile uint32_t *)l1_mem::address_map::NCRISC_L1_CONTEXT_BASE;
  if (risc_reset_req[0] == 1)
  {
    risc_reset_req[0] = 0;
    // Assert NCRISC reset
    uint32_t temp = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
    temp |= 0x40000;
    WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, temp);
    set_risc_reset_vector();

    // Deassert NCRISC reset
    temp = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
    temp &= 0xFFFBFFFF;
    WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, temp);
  }
}

void risc_init() {
  for (uint32_t n = 0; n < NUM_NOCS; n++) {
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
    my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
    my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    if (n == 0) {
      noc_size_x = (noc_id_reg >> (NOC_ADDR_NODE_ID_BITS+NOC_ADDR_NODE_ID_BITS)) & ((((uint64_t)0x1) << (NOC_ADDR_NODE_ID_BITS+1)) - 1);
      noc_size_y = (noc_id_reg >> (NOC_ADDR_NODE_ID_BITS+NOC_ADDR_NODE_ID_BITS+(NOC_ADDR_NODE_ID_BITS+1))) & ((((uint64_t)0x1) << (NOC_ADDR_NODE_ID_BITS+1)) - 1);
    }
  }
}

void replicate(uint32_t noc_id, uint32_t src_addr, uint64_t dest_addr, uint32_t chunk_size_bytes, uint32_t times_to_replicate) {
  const uint32_t REPLICATE_VC = 0;
  for (uint32_t j = 0; j < times_to_replicate; j++) {
    while (!ncrisc_noc_fast_write_ok(noc_id, NCRISC_WR_CMD_BUF));
    ncrisc_noc_fast_write(noc_id, NCRISC_WR_CMD_BUF,
                          src_addr,
                          dest_addr,
                          chunk_size_bytes,
                          REPLICATE_VC, false, false, 1);
    dest_addr += chunk_size_bytes;
  }
}

void replicate_l1(uint32_t noc_id, uint32_t src_addr, uint64_t dest_addr, uint32_t chunk_size_bytes, uint32_t times_to_replicate) {
  const uint32_t REPLICATE_VC = 0;
  for (uint32_t j = 0; j < times_to_replicate; j++) {
    while (!ncrisc_noc_fast_write_ok_l1(noc_id, NCRISC_WR_CMD_BUF));
    ncrisc_noc_fast_write_l1(noc_id, NCRISC_WR_CMD_BUF,
                          src_addr,
                          dest_addr,
                          chunk_size_bytes,
                          REPLICATE_VC, false, false, 1);
    dest_addr += chunk_size_bytes;
  }
}

/*
void __attribute__((section("code_l1"))) tile_header_buffer_init() {
  const uint32_t TILE_HEADER_BUF_INIT_CHUNK_WORDS = 64;
  const uint32_t TILE_HEADER_BUF_INIT_CHUNK_BYTES = TILE_HEADER_BUF_INIT_CHUNK_WORDS*16;
  const uint32_t NUM_TILE_HEADER_BUF_CHUNKS = MAX_TILES_PER_PHASE / TILE_HEADER_BUF_INIT_CHUNK_WORDS;
  uint32_t header_buf_init_noc = NUM_NOCS-1-loading_noc;
  volatile uint32_t* l1_tile_size_words_ptr = &(RISC_EPOCH_INFO_PTR->tile_size_words[0]);
  volatile uint32_t* l1_tile_size_header_buf_addr_ptr = &(RISC_EPOCH_INFO_PTR->tile_size_header_buf_addr[0]);
  // L1 reads flushed by immediate usage
  uint32_t num_tile_sizes = RISC_EPOCH_INFO_PTR->num_tile_sizes;
  for (uint32_t i = 0; i < num_tile_sizes; i++) {
    uint32_t tile_size_words = *l1_tile_size_words_ptr;
    uint32_t tile_size_header_buf_addr = *l1_tile_size_header_buf_addr_ptr;
    l1_tile_size_words_ptr++;
    l1_tile_size_header_buf_addr_ptr++;
    volatile uint32_t* l1_header_buf_tile_size_ptr = (volatile uint32_t*)tile_size_header_buf_addr;
    for (uint32_t j = 0; j < TILE_HEADER_BUF_INIT_CHUNK_WORDS; j++) {
      l1_header_buf_tile_size_ptr[0] = tile_size_words;
      l1_header_buf_tile_size_ptr += 4;
    }

    uint64_t tile_size_header_buf_curr_dest_addr = NOC_XY_ADDR(my_x[header_buf_init_noc], my_y[header_buf_init_noc], (tile_size_header_buf_addr + TILE_HEADER_BUF_INIT_CHUNK_BYTES));
    replicate_l1(header_buf_init_noc, tile_size_header_buf_addr, tile_size_header_buf_curr_dest_addr, TILE_HEADER_BUF_INIT_CHUNK_BYTES, NUM_TILE_HEADER_BUF_CHUNKS-1);
  }
  while (!ncrisc_noc_nonposted_writes_flushed_l1(header_buf_init_noc));
}
*/

void risc_get_next_epoch() {
  while (!RISC_EPOCH_INFO_PTR->all_streams_ready && !RISC_EPOCH_INFO_PTR->end_program)
  {
    risc_reset_check();
  }

  // Detect case when core is not used for this epoch
  if (RISC_EPOCH_INFO_PTR->all_streams_ready == 0xba) {
    volatile uint32_t* test_mailbox_ptr = (volatile uint32_t*)(l1_mem::address_map::FIRMWARE_BASE + TEST_MAILBOX_ADDRESS);
    test_mailbox_ptr[0] = 0xabcd1234;
    while(true) {
      risc_reset_check();
    }
  }
}


void risc_signal_epoch_done() {
  RISC_EPOCH_INFO_PTR->all_streams_and_kernels_done = 1;

  // Wait for ncrisc to "accept" kernel completion
  while (RISC_EPOCH_INFO_PTR->all_streams_ready == 1)
  {
    risc_reset_check();
  }
}

