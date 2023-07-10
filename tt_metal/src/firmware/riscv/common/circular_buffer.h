#pragma once

// TODO: we probably don't need any of these to be int16
// All of this is in local memory only, not in L1
struct CBReadInterface {
   uint32_t fifo_size;
   uint32_t fifo_limit;
   uint32_t fifo_rd_ptr;

   // local copy, used only by unpacker
   uint16_t tiles_acked;
};

struct CQReadInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit;
    uint32_t fifo_rd_ptr;
    uint32_t fifo_rd_toggle;
};

struct CBWriteInterface {
   uint32_t fifo_size;
   uint32_t fifo_limit;
   uint32_t fifo_size_tiles;
   uint32_t fifo_wr_ptr;

   // local copy, used only by packer
   uint16_t tiles_received;
   // used by packer for in-order packing
   uint32_t fifo_wr_tile_ptr;
};
