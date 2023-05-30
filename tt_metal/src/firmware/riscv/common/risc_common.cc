
#include "risc_common.h"
#include "noc_nonblocking_api.h"
#include "stream_interface.h"

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
