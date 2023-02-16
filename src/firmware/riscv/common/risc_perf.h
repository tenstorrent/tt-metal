#pragma once

#include <stdint.h>
#include "epoch.h"
#include "noc_nonblocking_api.h"

#ifndef INTERMED_DUMP
#define INTERMED_DUMP 0
#endif

#ifndef PERF_DUMP_LEVEL
#define PERF_DUMP_LEVEL 0
#endif

extern volatile uint32_t local_mem_barrier;

namespace risc
{
const uint32_t PERF_SPILL_CHECK_MASK = (0x1 << 4) - 1;
const uint32_t PERF_DUMP_VC = 2;
const uint32_t PERF_DUMP_NOC = 0;
constexpr uint32_t PERF_DUMP_END_SIGNAL = 0xbeeff00d;
constexpr uint32_t PERF_DUMP_PADDING = 0xdeadbead;


constexpr uint32_t EPOCH_START_OFFSET = 1;
constexpr uint32_t EPOCH_END_OFFSET = 4;
constexpr uint32_t STREAM_HANDLER_START_OFFSET = 7;
constexpr uint32_t STREAM_HANDLER_END_OFFSET = 10;
constexpr uint32_t EPILOGUE_START_OFFSET = 13;
constexpr uint32_t EPILOGUE_END_OFFSET = 16;
constexpr uint32_t PROLOGUE_START_OFFSET = 19;
constexpr uint32_t PROLOGUE_END_OFFSET = 22;
constexpr uint32_t PERF_START_OFFSET = 25;

extern uint32_t perf_end;
extern volatile uint32_t *perf_double_buf_base[2];
extern volatile uint32_t *perf_buf_base;
extern uint32_t perf_index;
extern volatile uint32_t epoch_perf_scratch[PERF_START_OFFSET];


struct perf_event {
    constexpr static uint32_t EPOCH = 1 << 24;
    constexpr static uint32_t STREAM_HANDLER_LOOP = 2 << 24;
    constexpr static uint32_t EPOCH_EPILOGUE = 3 << 24;
    constexpr static uint32_t STREAM_HANDLER_INIT = 4 << 24;
    constexpr static uint32_t EPOCH_Q_SLOT_COMPLETE = 5 << 24;
    constexpr static uint32_t WALL_CLOCK_TOP_32B = 6 << 24;
    constexpr static uint32_t DRAM_READ_ISSUED = 7 << 24;
    constexpr static uint32_t DRAM_READ_TILE_FLUSHED = 8 << 24;
    constexpr static uint32_t DRAM_WRITE_SENT = 9 << 24;
    constexpr static uint32_t DRAM_WRITE_TILES_CLEARED = 10 << 24;
    constexpr static uint32_t DRAM_IO_Q_STATUS = 11 << 24;
    constexpr static uint32_t STREAM_RESTART = 12 << 24;
    constexpr static uint32_t STREAM_INFO = 13 << 24;
    constexpr static uint32_t STREAM_BUF_STATUS = 14 << 24;
    constexpr static uint32_t EPOCH_Q_EMPTY = 15 << 24;
};

#if (PERF_DUMP_LEVEL == 0)
#define TRISC_PERF_BUF_SIZE l1_mem::address_map::TRISC_PERF_BUF_SIZE_LEVEL_0
#define NCRISC_PERF_BUF_SIZE l1_mem::address_map::NCRISC_PERF_BUF_SIZE_LEVEL_0
#else
#define TRISC_PERF_BUF_SIZE l1_mem::address_map::TRISC_PERF_BUF_SIZE_LEVEL_1
#define NCRISC_PERF_BUF_SIZE l1_mem::address_map::NCRISC_PERF_BUF_SIZE_LEVEL_1
#endif

#define T0_PERF_L l1_mem::address_map::PERF_BUF_BASE_ADDR
#define T0_PERF_H (T0_PERF_L + TRISC_PERF_BUF_SIZE/2)
#define T1_PERF_L (T0_PERF_L + TRISC_PERF_BUF_SIZE)
#define T1_PERF_H (T1_PERF_L + TRISC_PERF_BUF_SIZE/2)
#define T2_PERF_L (T1_PERF_L + TRISC_PERF_BUF_SIZE)
#define T2_PERF_H (T2_PERF_L + TRISC_PERF_BUF_SIZE/2)
#define T3_PERF_L l1_mem::address_map::NCRISC_L1_PERF_BUF_BASE
#define T3_PERF_H (T3_PERF_L + NCRISC_PERF_BUF_SIZE/2)


inline __attribute__((always_inline)) void record_info(uint32_t event_id, uint32_t epoch_iters_remaining, uint32_t epoch_q_slots_remaining, uint32_t q_slot_size_tiles, uint32_t data_chunk_size_tiles, uint32_t data_chunk_size_bytes, uint32_t flags) {
    // stream information
    perf_buf_base[perf_index] = event_id;
    perf_buf_base[perf_index + 1] = flags;
    perf_buf_base[perf_index + 2] = epoch_iters_remaining;
    perf_buf_base[perf_index + 3] = epoch_q_slots_remaining;
    perf_buf_base[perf_index + 4] = q_slot_size_tiles;
    perf_buf_base[perf_index + 5] = data_chunk_size_tiles;
    perf_buf_base[perf_index + 6] = data_chunk_size_bytes;
    perf_index += 7;
}

inline void record_perf_value(uint32_t event_id, uint32_t event_value) {
    // Record a single event, and a timestamp
    perf_buf_base[perf_index] = event_id;
    perf_buf_base[perf_index + 1] = event_value;
    perf_index += 2;
}

inline __attribute__((section("code_l1"))) void record_perf_value_l1(uint32_t event_id, uint32_t event_value) {
    // Record a single event, and a timestamp
    perf_buf_base[perf_index] = event_id;
    perf_buf_base[perf_index + 1] = event_value;
    perf_index += 2;
}

inline __attribute__((always_inline)) void record_perf_value_at_offset(uint32_t event_id, uint32_t event_value1, uint32_t event_value2, uint32_t offset) {
    // Record a single value, and a timestamp at given offset
    if constexpr (INTERMED_DUMP && (PERF_DUMP_LEVEL != 0)) {
        epoch_perf_scratch[offset] = event_id;
        epoch_perf_scratch[offset + 1] = event_value1;
        epoch_perf_scratch[offset + 2] = event_value2;
    } else {
        perf_buf_base[offset] = event_id;
        perf_buf_base[offset + 1] = event_value1;
        perf_buf_base[offset + 2] = event_value2;
    }
}

// This function gets called when half-perf-buffer is full and need to switch.
inline void switch_perf_buffers_and_record_event(uint32_t event_id, uint32_t event_value) {
    if constexpr (INTERMED_DUMP) {
        EPOCH_INFO_PTR->perf_dram_copy_req[3]++;
        perf_buf_base = perf_buf_base == perf_double_buf_base[0] ? perf_double_buf_base[1] : perf_double_buf_base[0];
        //add padding to last two locations incase events cannot be written to them becuase of overlap 
        //with last location that is supposed to hold perf dump end signal.
        //This is done so that post processor does not read any stale event as a valid event.
        perf_buf_base[perf_end-1] = PERF_DUMP_PADDING;
        perf_buf_base[perf_end-2] = PERF_DUMP_PADDING;
        perf_index = 0;
        record_perf_value(event_id, event_value);
    }
}

inline void record_perf_value_and_check_overflow(uint32_t event_id, uint32_t event_value, uint32_t leave_space = 0) {
    // Record a single value, and a timestamp
    if constexpr (INTERMED_DUMP) {
        if (perf_index + 1 < perf_end - 1) {
            record_perf_value(event_id, event_value);
        } else {
            switch_perf_buffers_and_record_event(event_id, event_value);
        }
    } else {
        if (perf_index + 1 < perf_end - 1) {
            record_perf_value(event_id, event_value);
        }
    }
}

inline __attribute__((section("code_l1"))) void spill_risc_epoch_perf_scratch() {
    //EPOCH_INFO_PTR->perf_dram_addr[3] == 0 signals that dram buffer has been filled up and that
    //for current epoch, we do not have dram buffer space available for perf dump.
    if (EPOCH_INFO_PTR->perf_dram_addr[3]) {
        ncrisc_noc_fast_write_any_len_l1(PERF_DUMP_NOC, NCRISC_WR_CMD_BUF, (uint32_t)epoch_perf_scratch, EPOCH_INFO_PTR->perf_dram_addr[3], sizeof(epoch_perf_scratch), PERF_DUMP_VC, false, false, 1);
        while (!ncrisc_noc_nonposted_writes_sent_l1(PERF_DUMP_NOC)){};
    }
}

void init_perf_dram_state()__attribute__((section("code_l1")));
void allocate_perf_buffer()__attribute__((section("code_l1")));
void record_perf_dump_end()__attribute__((section("code_l1")));
void record_timestamp(uint32_t event_id);
void record_timestamp_at_offset(uint32_t event_id, uint32_t offset);
void record_timestamp_at_offset_l1(uint32_t event_id, uint32_t offset)__attribute__((section("code_l1")));
void check_dram_spill_requests()__attribute__((section("code_l1")));

}
