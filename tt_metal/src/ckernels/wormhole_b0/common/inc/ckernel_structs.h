#pragma once

#include "circular_buffer.h"
#include "hostdevcommon/kernel_structs.h"

namespace ckernel
{

// Semaphores mapping and trisc space -> tensix space conversion
struct semaphore
{
    constexpr static uint32_t MATH_PACK = 1;   // math <-> pack sync on dest register
    constexpr static uint32_t UNPACK_PACK = 2; // pack <-> unpack sync on scratch buffer
    constexpr static uint32_t UNPACK_OPERAND_SYNC = 3; // unpack <-> pack, math sync on operand get/release
    constexpr static uint32_t PACK_DONE = 4; // Wait for beinning and end of each pack-iteration. For recording perf events.
    constexpr static uint32_t UNPACK_SYNC = 5; // trisc <-> unpack sync on hw kernel
    constexpr static uint32_t PACK_SYNC = 6; // thcon <-> pack sync on tile write to l1

    constexpr static uint16_t t6_sem(const uint8_t sem_index)
    {
        return (1 << sem_index);
    }
};

struct mutex
{
    constexpr static uint32_t REG_RMW = 0;   // used for atomic register read-modify-write from different threads
};

constexpr uint8_t PC_BUF_SEMAPHORE_BASE = 8; // base address for semaphores in PC buffer
constexpr uint8_t MATH_HALF_DEST_SIZE = 32;  // arch specific 1/2 dest registers size in 16x16 faces
constexpr uint8_t MAX_CONFIG_STATES = 2;

// Firmware messages to ckernels
enum firmware_msg_e
{
    FLIP_STATE_ID = 1,
    RUN_INSTRUCTIONS = 2,
    RESET_DEST_OFFSET_ID = 3,
    SET_PERF_SCRATCH = 4
};

constexpr uint8_t OPERAND_BASE_REG = 16; // base register used for operand storage
constexpr uint8_t OUTPUT_BASE_REG = 16; // base register used for output storage

typedef struct {
   uint32_t fifo_rd_ptr;
   uint32_t fifo_limit;
   uint16_t tiles_acked;
   uint16_t reserved_1;
   uint32_t words_acked;
   uint32_t fifo_size;
   uint16_t blocks_per_iter; // total number of ublocks popped from interm buffer per input
   uint16_t curr_block; // current number of ublocks popped per input
   uint16_t num_iter;  // total number of passes through the interm buffer per input
   uint16_t curr_iter;  // current numer of passes through the interm buffer per input
   uint32_t fifo_rd_base_ptr;
} operand_t;

static_assert(sizeof(operand_t) == (sizeof(uint32_t) * 8));

typedef union {
   operand_t f;
   uint32_t val[8];
} operand_u;

typedef struct {
   uint32_t fifo_wr_ptr;
   uint32_t fifo_limit;
   uint32_t fifo_size;
   uint32_t fifo_num_pages;
   uint32_t fifo_wr_base_ptr;
   uint16_t fifo_wr_tile_ptr;
   uint16_t tiles_received;
   bool     legacy_pack;
   uint8_t  reserved_0;
   uint8_t  fork;
   uint8_t  num_fork_streams;
   bool     shared_buffer;  // interm buffer is shared with output
   uint8_t  shared_buffer_operand; //shared buffer output operand
   uint8_t  reserved_1[2];
   uint8_t  fork_stream_ids[16];
   union {
      uint16_t ublock_ct;       //ublock ct dim in tiles
      uint16_t out_tile_dim;   //output block dim in tiles
   };
   union {
      uint16_t ublock_tile_dim; //number of tiles in ublock for untilized output
      uint16_t blocks_per_iter; //total number of ublocks pushed to interm buffer per input
   };
   union {
      uint16_t row_tile_dim;    //one row of tiles
   };
   union {
      uint16_t block_tile_dim;  //one row of ublocks for untilized output
      uint16_t num_iter; //total number of passes through the interm buffer per input
   };
   union {
      uint16_t ublock_tile_cnt;
      uint16_t curr_block;  //current number of ublocks pushed to interm buffer per input
   };
   union {
      uint16_t block_tile_cnt;  //current number of packed tiles for untilized output
      uint16_t curr_iter;  // current numer of passes through the interm buffer per input
   };
} output_t;

static_assert(sizeof(output_t) == (sizeof(uint32_t) * 15));

typedef union {
   output_t f;
   uint32_t val[15];
} output_u;


} // namespace ckelimitrnel
