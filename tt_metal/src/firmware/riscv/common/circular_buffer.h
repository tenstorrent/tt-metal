#pragma once


struct CQReadInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit;
    uint32_t fifo_page_size;
    uint32_t fifo_rd_ptr;
    uint32_t fifo_rd_toggle;
};

struct CBReadInterface {
   uint32_t fifo_size;
   uint32_t fifo_limit;
   uint32_t fifo_page_size;
   uint32_t fifo_rd_ptr;

   // local copy, used only by unpacker
   uint16_t tiles_acked;
};
struct CBWriteInterface {
   uint32_t fifo_size;
   uint32_t fifo_limit;
   uint32_t fifo_num_pages;
   uint32_t fifo_page_size;
   uint32_t fifo_wr_ptr;

   // local copy, used only by packer
   uint16_t tiles_received;
   // used by packer for in-order packing
   uint32_t fifo_wr_tile_ptr;
};
