#ifndef _EPOCH_H_
#define _EPOCH_H_

#include <cstdint>
#include "l1_address_map.h"
#include "eth_l1_address_map.h"
#include "noc_overlay_parameters.h"
#include "stream_io_map.h"

const uint32_t EPOCH_INFO_ADDR = l1_mem::address_map::OVERLAY_BLOB_BASE;
const uint32_t BLOB_START_ADDR = (l1_mem::address_map::OVERLAY_BLOB_BASE + 0x1000);

const uint32_t ETH_EPOCH_INFO_ADDR = eth_l1_mem::address_map::OVERLAY_BLOB_BASE;


#define EPOCH_INFO_PTR ((volatile epoch_t*)EPOCH_INFO_ADDR)
#define ETH_EPOCH_INFO_PTR ((volatile epoch_t*)ETH_EPOCH_INFO_ADDR)

// Kernel parameters are subsumed under inputs.
// Kernel intermediates are inputs/outputs that map to the same stream.
const uint32_t EPOCH_MAX_INPUTS = 24;
const uint32_t EPOCH_MAX_OUTPUTS = 16;
const uint32_t EPOCH_MAX_OUTPUT_FORKS = 16;
const uint32_t EPOCH_MAX_NUM_TILE_SIZES = 8;

const uint32_t PERF_NUM_THREADS = 4;

enum stream_state_t {
  STREAM_STATE_START = 0,
  STREAM_STATE_ACTIVE,
  STREAM_STATE_EPOCH_DONE
};

#define STREAM_INPUT_PARK     (((uint32_t)0x1) << 0)
#define STREAM_OUTPUT_PARK    (((uint32_t)0x1) << 7)
#define STREAM_DRAM_IO        (((uint32_t)0x1) << 1)
#define STREAM_DRAM_STREAMING (((uint32_t)0x1) << 5)
#define STREAM_DRAM_INPUT     (((uint32_t)0x1) << 2)
#define STREAM_DRAM_OUTPUT    (((uint32_t)0x1) << 3)
#define STREAM_INTERMEDIATE   (((uint32_t)0x1) << 4)
#define STREAM_MOVES_RAW_DATA (((uint32_t)0x1) << 6)
#define STREAM_IS_FORK        (((uint32_t)0x1) << 8)
#define STREAM_DRAM_RAM       (((uint32_t)0x1) << 9)
#define STREAM_BRISC_PACK     (((uint32_t)0x1) << 10)

#pragma pack(push)
#pragma pack(4)

#define DRAM_IO_STATE_RD_SEC 0
#define DRAM_IO_STATE_WR_SEC 8

struct dram_io_scatter_state_t {
  uint32_t unused1;
  uint32_t scatter_offsets_size;
  uint32_t scatter_chunk_size_bytes;
  uint32_t q_slot_size_bytes;
  uint32_t scatter_chunk_size_tiles;
  uint32_t unused2;
  uint32_t unused3;
  uint32_t* scatter_offsets;
} ;

typedef struct dram_io_scatter_state_t dram_io_scatter_state_t;

static_assert(sizeof(dram_io_scatter_state_t) == (1*32));
static_assert(sizeof(dram_io_scatter_state_t*) == 4);

struct dram_io_state_t {
  // Section for dram to l1 traffic
  uint32_t rd_dram_rdptr;
  uint32_t rd_dram_wrptr;
  uint16_t rd_dram_local_rdptr;
  uint16_t rd_epoch_id_tag;
  uint16_t rd_stride;
  uint16_t rd_flags;
  uint8_t  rd_grd_ptr_autoinc;
  uint8_t  rd_gwr_ptr_autoinc;
  uint8_t  rd_lrd_ptr_autoinc;
  uint8_t  unused5;
  uint16_t rd_queue_update_stride;
  uint16_t unused0;
  uint32_t unused1;
  uint32_t unused2;

  // Section for l1 to dram traffic
  uint32_t wr_dram_rdptr;
  uint32_t wr_dram_wrptr;
  uint16_t wr_dram_local_rdptr;
  uint16_t wr_epoch_id_tag;
  uint16_t wr_stride;
  uint16_t wr_flags;
  uint8_t  wr_grd_ptr_autoinc;
  uint8_t  wr_gwr_ptr_autoinc;
  uint8_t  wr_lrd_ptr_autoinc;
  uint8_t  unused6;
  uint16_t wr_queue_update_stride;
  uint16_t unused3;
  uint32_t data_chunk_size_bytes;
  uint16_t data_chunk_size_tiles;
  uint16_t unused4;

  
  // Temp variables
  uint32_t dram_buf_size_bytes;
  uint32_t dram_buf_size_q_slots;
  uint64_t dram_buf_addr;
  uint32_t dram_q_slot_size_tiles;
  uint8_t reader_index;
  uint8_t total_readers;
  uint8_t unused7;
  uint8_t stride_wrap;
  struct dram_io_scatter_state_t* dram_io_scatter_state;
  struct dram_io_state_t* next;
} ;

typedef struct dram_io_state_t dram_io_state_t;

static_assert(sizeof(dram_io_state_t) == (3 * 32));
static_assert(sizeof(dram_io_state_t*) == 4);


typedef struct {
  uint32_t dram_q_slot_size_bytes;
  uint8_t input_noc;
  uint8_t output_noc;
  uint8_t unused5;
  uint8_t unused6;
  uint32_t unused0;
  uint32_t unused1;
  uint32_t unused2;
  uint32_t epoch_q_slots_remaining;
  uint32_t unused3;
  dram_io_state_t* dram_io_state;
} epoch_stream_dram_io_info_t;

static_assert(sizeof(epoch_stream_dram_io_info_t) == (1 * 32));
static_assert(sizeof(epoch_stream_dram_io_info_t*) == 4);


typedef struct {

  // Other stream state/info can be read from the stream registers directly,
  // no need to store it into this structure.
  uint16_t stream_id;
  uint16_t producer_epoch_id;
  uint32_t epoch_start_phase;
  uint32_t epoch_num_tiles;
  uint32_t tile_size_words;
  uint32_t buf_size_tiles;
  uint32_t buf_full_size_bytes;
  uint32_t buf_base_addr;
  uint16_t num_msgs_in_block;
  uint8_t start_phase_num_cfg_regs;
  uint8_t packer_operand;
  uint32_t msg_info_buf_start;
  uint32_t blob_start_addr;
  uint32_t blob_size;
  uint32_t num_iters_in_epoch;
  uint32_t epoch_iters_remaining;
  uint32_t num_scatter_inner_loop;
  uint8_t legacy_pack;
  uint8_t log_num_fork_streams_with_operand;
  uint16_t num_mblock_buffering;

  uint32_t flags;
  uint16_t stride;
  uint16_t total_strides;
  uint32_t stride_offset_size_bytes;
  uint32_t skip_col_bytes;
  uint32_t skip_col_tile_row_bytes;
  uint32_t skip_col_row_bytes;
  uint32_t skip_zcol_bytes;
  uint32_t skip_col_zrow_bytes;
  uint16_t c_dim_size;
  uint16_t r_dim_size;
  uint16_t zc_dim_size;
  uint16_t zr_dim_size;

  uint32_t num_dram_io_bufs;
  epoch_stream_dram_io_info_t* dram_io_info;
  uint16_t num_fork_streams;
  uint16_t scatter_order_size;
  uint8_t fork_idxs[EPOCH_MAX_OUTPUT_FORKS];
  
} epoch_stream_info_t;


static_assert(sizeof(epoch_stream_info_t) == (4 * 32));
static_assert(sizeof(epoch_stream_info_t*) == 4);


typedef struct {
  
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t num_active_streams;

  uint8_t  epoch_valid;
  uint8_t  all_streams_ready;
  uint8_t  all_streams_and_kernels_done;
  uint8_t  end_program;

  uint32_t num_tile_sizes;
  uint32_t tile_size_words[EPOCH_MAX_NUM_TILE_SIZES];
  uint32_t tile_size_header_buf_addr[EPOCH_MAX_NUM_TILE_SIZES];

  epoch_stream_info_t* inputs[EPOCH_MAX_INPUTS];
  epoch_stream_info_t* outputs[EPOCH_MAX_OUTPUTS];
  epoch_stream_info_t* active_streams[NOC_NUM_STREAMS];

  uint32_t perf_dram_copy_req[PERF_NUM_THREADS];
  uint32_t perf_dram_copy_ack[PERF_NUM_THREADS];
  uint64_t perf_dram_addr[PERF_NUM_THREADS];
  uint16_t perf_req_max[PERF_NUM_THREADS];

  uint16_t ublock_rt;
  uint16_t ublock_ct;
  uint16_t mblock_m;
  uint16_t mblock_n;
  uint16_t mblock_k;
  uint16_t unused0;

  uint16_t overlay_valid; // signifies if core has valid overlay that it needs to load/run
  uint16_t skip_kernels; // if true, don't load/run trisc binaries - there are none for this epoch on this core.
  uint32_t padding32b[1];
  uint32_t dummy_phase_tile_header_and_data[4]; // Needed to make dummy phases work, always set to 0x1, must always be at the end of epoch_t

} epoch_t;

static_assert(sizeof(epoch_t) == (((2*EPOCH_MAX_NUM_TILE_SIZES) + EPOCH_MAX_INPUTS + EPOCH_MAX_OUTPUTS + NOC_NUM_STREAMS + (4*PERF_NUM_THREADS) + 16) * 4));
static_assert((sizeof(epoch_t) % 32) == 0);

#pragma pack(pop)


#endif // ndef _EPOCH_H_
