// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <climits>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC)
#include "risc_common.h"
#include "dataflow_api_addrgen.h"
#else
#include "ckernel.h"
#endif

#include "hostdevcommon/profiler_common.h"
#include "risc_attribs.h"

#include <dev_msgs.h>

#define DO_PRAGMA(x) _Pragma(#x)

#define Stringize(L) #L
#define MakeString(M, L) M(L)
#define $Line MakeString(Stringize, __LINE__)

#define PROFILER_MSG __FILE__ "," $Line ",KERNEL_PROFILER"
#define PROFILER_MSG_NAME(name) name "," PROFILER_MSG

#define SrcLocNameToHash(name)                   \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name))); \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));

#if defined(PROFILE_KERNEL)

namespace kernel_profiler {

extern uint32_t wIndex;
extern uint32_t time_out;
extern uint32_t stackSize;

extern uint32_t sums[SUM_COUNT];
extern uint32_t sumIDs[SUM_COUNT];

constexpr uint32_t PROFILER_PUSH_TIME_OUT = 100000;
constexpr uint32_t NOC_PUSH_MARKER_COUNT = 2;
constexpr int WALL_CLOCK_HIGH_INDEX = 1;
constexpr int WALL_CLOCK_LOW_INDEX = 0;
constexpr uint32_t ALL_DATA_SENT = PROFILER_L1_VECTOR_SIZE + 1;

#if !defined(COMPILE_FOR_TRISC)
extern uint32_t core_flat_id;
extern uint32_t profiler_core_count_per_dram;
extern uint32_t dram_buffer_page_size;
#endif

volatile tt_l1_ptr uint32_t* profiler_control_buffer =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));

volatile tt_l1_ptr uint32_t (*profiler_data_buffer)[kernel_profiler::PROFILER_L1_VECTOR_SIZE] =
    reinterpret_cast<volatile tt_l1_ptr uint32_t (*)[kernel_profiler::PROFILER_L1_VECTOR_SIZE]>(
        GET_MAILBOX_ADDRESS_DEV(profiler.buffer));

#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t myRiscID = 0;
#elif defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
constexpr uint32_t myRiscID = 0;
#elif defined(COMPILE_FOR_NCRISC)
constexpr uint32_t myRiscID = 1;
#elif defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
constexpr uint32_t myRiscID = 2;
#elif defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 1
constexpr uint32_t myRiscID = 3;
#elif defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 2
constexpr uint32_t myRiscID = 4;
#endif

constexpr uint32_t Hash32_CT(const char* str, size_t n, uint32_t basis = UINT32_C(2166136261)) {
    return n == 0 ? basis : Hash32_CT(str + 1, n - 1, (basis ^ str[0]) * UINT32_C(16777619));
}

template <size_t N>
constexpr uint32_t Hash16_CT(const char (&s)[N]) {
    auto res = Hash32_CT(s, N - 1);
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC)

#if (defined(DISPATCH_KERNEL))

// Saves several NoC register states that may be setup by dispatch kernels and restores them
// when the NocDestinationStateSaver is destroyed
struct NocDestinationStateSaver {
    uint32_t noc_ctrl_state;
    uint32_t noc_ret_addr_coord_state;
#ifdef ARCH_BLACKHOLE
    uint32_t noc_ret_addr_mid_state;
#endif

    inline __attribute__((always_inline)) NocDestinationStateSaver() {
        noc_ctrl_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_CTRL);
        noc_ret_addr_coord_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE);
#ifdef ARCH_BLACKHOLE
        noc_ret_addr_mid_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_MID);
#endif
    }

    inline __attribute__((always_inline)) ~NocDestinationStateSaver() {
        while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_CTRL, noc_ctrl_state);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE, noc_ret_addr_coord_state);
#ifdef ARCH_BLACKHOLE
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_MID, noc_ret_addr_mid_state);
#endif
    }
};

#else

struct NocDestinationStateSaver {};

#endif

inline void __attribute__((always_inline)) profiler_noc_async_write_posted(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index) {
    WAYPOINT("NAWW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
    ncrisc_noc_fast_write_any_len<noc_mode>(
        noc, write_cmd_buf, src_local_l1_addr, dst_noc_addr, size, NOC_UNICAST_WRITE_VC, false, false, 1, true, true);
    WAYPOINT("NAWD");
}

FORCE_INLINE
void profiler_noc_async_flush_posted_write(uint8_t noc = noc_index) {
    WAYPOINT("NPPW");
    while (!ncrisc_noc_posted_writes_sent(noc));
    WAYPOINT("NPPD");
}

#endif

__attribute__((noinline)) void init_profiler(
    uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0) {
    wIndex = CUSTOM_MARKERS;
    profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + myRiscID] = 0;
    stackSize = 0;

#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_BRISC)
    profiler_control_buffer[PROFILER_DONE] = 0;

    for (uint32_t riscID = 0; riscID < PROFILER_RISC_COUNT; riscID++) {
        for (uint32_t i = ID_HH; i < GUARANTEED_MARKER_1_H; i++) {
            profiler_data_buffer[riscID][i] = 0;
        }
    }

    profiler_control_buffer[NOC_X] = my_x[0];
    profiler_control_buffer[NOC_Y] = my_y[0];

    for (uint32_t riscID = 0; riscID < PROFILER_RISC_COUNT; riscID++) {
        for (uint32_t i = GUARANTEED_MARKER_1_H; i < CUSTOM_MARKERS; i++) {
            // TODO(MO): Clean up magic numbers
            profiler_data_buffer[riscID][i] = 0x80000000;
        }
    }

    while (!profiler_control_buffer[DRAM_PROFILER_ADDRESS]);

    core_flat_id = profiler_control_buffer[FLAT_ID];
    profiler_core_count_per_dram = profiler_control_buffer[CORE_COUNT_PER_DRAM];
    dram_buffer_page_size = PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * MAX_RISCV_PER_CORE * profiler_core_count_per_dram;

    profiler_data_buffer[myRiscID][ID_LH] = ((core_flat_id & 0xFF) << 3) | myRiscID;
#endif
}

constexpr uint32_t get_const_id(uint32_t id, PacketTypes type) { return ((id & 0xFFFF) | ((type << 16) & 0x7FFFF)); }

inline __attribute__((always_inline)) uint32_t get_id(uint32_t id, PacketTypes type) {
    return ((id & 0xFFFF) | ((type << 16) & 0x7FFFF));
}

inline __attribute__((always_inline)) bool bufferHasRoom() {
    return wIndex < (PROFILER_L1_VECTOR_SIZE - stackSize - (NOC_PUSH_MARKER_COUNT * PROFILER_L1_MARKER_UINT32_SIZE));
}

inline __attribute__((always_inline)) void mark_time_at_index_inlined(uint32_t index, uint32_t timer_id) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    profiler_data_buffer[myRiscID][index] =
        0x80000000 | ((timer_id & 0x7FFFF) << 12) | (p_reg[WALL_CLOCK_HIGH_INDEX] & 0xFFF);
    profiler_data_buffer[myRiscID][index + 1] = p_reg[WALL_CLOCK_LOW_INDEX];
}

inline __attribute__((always_inline)) void mark_padding() {
    if (wIndex < PROFILER_L1_VECTOR_SIZE) {
        profiler_data_buffer[myRiscID][wIndex] = 0x80000000;
        profiler_data_buffer[myRiscID][wIndex + 1] = 0;
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
    }
}

inline __attribute__((always_inline)) void mark_dropped_timestamps(uint32_t index) {
    uint32_t curr = profiler_control_buffer[DROPPED_ZONES];
    profiler_control_buffer[DROPPED_ZONES] = (1 << index) | curr;
}

inline __attribute__((always_inline)) void set_host_counter(uint32_t counterValue) {
    for (uint32_t riscID = 0; riscID < PROFILER_RISC_COUNT; riscID++) {
        profiler_data_buffer[riscID][ID_LL] = counterValue;
    }
}

inline __attribute__((always_inline)) void set_profiler_zone_valid(bool condition) {
    profiler_control_buffer[PROFILER_DONE] = !condition;
}

inline __attribute__((always_inline)) void risc_finished_profiling() {
    for (uint32_t i = 0; i < (wIndex % NOC_ALIGNMENT_FACTOR); i++) {
        mark_padding();
    }
    profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + myRiscID] = wIndex;
}

template <bool RECORD_ZONE = true>
__attribute__((noinline)) void finish_profiler() {
    if constexpr (RECORD_ZONE) {
        SrcLocNameToHash("PROFILER-DRAM-PUSH");
        mark_time_at_index_inlined(wIndex, hash);
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;

        mark_time_at_index_inlined(wIndex, get_const_id(hash, ZONE_END));
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
    }
    risc_finished_profiling();
    profiler_control_buffer[PROFILER_DONE] = 1;
#if defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_NCRISC)
#else

// Dispatch core might get to finish
#if defined(DISPATCH_KERNEL)
    while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
    core_flat_id = profiler_control_buffer[FLAT_ID];
    profiler_core_count_per_dram = profiler_control_buffer[CORE_COUNT_PER_DRAM];
    dram_buffer_page_size = PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * MAX_RISCV_PER_CORE * profiler_core_count_per_dram;

    profiler_data_buffer[myRiscID][ID_LH] = ((core_flat_id & 0xFF) << 3) | myRiscID;
#endif
    NocDestinationStateSaver noc_state;

    constexpr uint32_t startRisc = myRiscID;
#if defined(COMPILE_FOR_BRISC)
    // Send all riscs
    constexpr uint32_t endRisc = PROFILER_RISC_COUNT;
#else
    constexpr uint32_t endRisc = myRiscID + 1;
#endif

    for (uint32_t riscID = startRisc; riscID < endRisc; riscID++) {
        int hostIndex = riscID;
        int deviceIndex = DEVICE_BUFFER_END_INDEX_BR_ER + riscID;
        if (profiler_control_buffer[deviceIndex]) {
            uint32_t currEndIndex = profiler_control_buffer[deviceIndex] + profiler_control_buffer[hostIndex];

            bool do_noc = false;
            uint32_t dram_offset = 0;
            uint32_t send_size = 0;
            if (currEndIndex <= PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC) {
                dram_offset = (core_flat_id % profiler_core_count_per_dram) * MAX_RISCV_PER_CORE *
                                  PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                              hostIndex * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                              profiler_control_buffer[hostIndex] * sizeof(uint32_t);

                send_size = profiler_control_buffer[deviceIndex] * sizeof(uint32_t);

                do_noc = true;
                profiler_control_buffer[hostIndex] = currEndIndex;
            } else {
                mark_dropped_timestamps(hostIndex);
            }

            if (do_noc) {
                const InterleavedAddrGen<true> s = {
                    .bank_base_address = profiler_control_buffer[DRAM_PROFILER_ADDRESS],
                    .page_size = dram_buffer_page_size};

                uint64_t dram_bank_dst_noc_addr =
                    s.get_noc_addr(core_flat_id / profiler_core_count_per_dram, dram_offset);

                profiler_noc_async_write_posted(
                    reinterpret_cast<uint32_t>(profiler_data_buffer[hostIndex]), dram_bank_dst_noc_addr, send_size);
            }
            profiler_control_buffer[deviceIndex] = ALL_DATA_SENT;
        }
    }

    profiler_noc_async_flush_posted_write();
#endif
    profiler_control_buffer[PROFILER_DONE] = 0;
    wIndex = CUSTOM_MARKERS;
}

void push_time_out() {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_BRISC)
    if (time_out++ > PROFILER_PUSH_TIME_OUT && wIndex > CUSTOM_MARKERS + PROFILER_L1_MARKER_UINT32_SIZE) {
        time_out = 0;
        finish_profiler();
    }
#endif
}

struct scopePush {
    inline __attribute__((always_inline)) scopePush() {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_BRISC)
#if !defined(DISPATCH_KERNEL)
        uint32_t runCounter = profiler_control_buffer[RUN_COUNTER];
        profiler_data_buffer[myRiscID][wIndex] = (runCounter & 0xFFFF) |
                                                 ((((core_flat_id & 0xFF) << 3) | myRiscID) << 16) |
                                                 ((runCounter & 0xF) << 27) | (0x1 << 31);
#endif
#else
        if (wIndex >= (PROFILER_L1_VECTOR_SIZE - (NOC_PUSH_MARKER_COUNT * PROFILER_L1_MARKER_UINT32_SIZE))) {
            // This risc did request a noc push so wait until its data is pushed
            while (profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + myRiscID] != ALL_DATA_SENT) {
            }
            wIndex = CUSTOM_MARKERS;
        } else if (profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + myRiscID] == ALL_DATA_SENT) {
            // This risc did not request a noc push but its data might have been sent so flush the buffer
            wIndex = CUSTOM_MARKERS;
        }
#endif
    }

    inline __attribute__((always_inline)) ~scopePush() {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_BRISC)
        if (profiler_control_buffer[PROFILER_DONE] ||
            wIndex >= (PROFILER_L1_VECTOR_SIZE - (NOC_PUSH_MARKER_COUNT * PROFILER_L1_MARKER_UINT32_SIZE))) {
            // If any other risc has its buffer full or we are full
            finish_profiler();
        }
#else
        // Set control buffer index, in case B/NCRISC attempt to do a noc push
        profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + myRiscID] = wIndex;
        if (wIndex >= (PROFILER_L1_VECTOR_SIZE - (NOC_PUSH_MARKER_COUNT * PROFILER_L1_MARKER_UINT32_SIZE))) {
            finish_profiler();
        }
#endif
    }
};

template <uint32_t timer_id>
struct profileScope {
    bool start_marked = false;
    inline __attribute__((always_inline)) profileScope() {
        if (bufferHasRoom()) {
            stackSize += PROFILER_L1_MARKER_UINT32_SIZE;
            start_marked = true;
            mark_time_at_index_inlined(wIndex, timer_id);
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }

    inline __attribute__((always_inline)) ~profileScope() {
        if (start_marked) {
            mark_time_at_index_inlined(wIndex, get_const_id(timer_id, ZONE_END));
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
            start_marked = false;
            stackSize -= PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
};

template <uint32_t data_id>
inline __attribute__((always_inline)) void timeStampedData(uint64_t data) {
    if (bufferHasRoom()) {
        mark_time_at_index_inlined(wIndex, get_const_id(data_id, TS_DATA));
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
        profiler_data_buffer[myRiscID][wIndex++] = data >> 32;
        profiler_data_buffer[myRiscID][wIndex++] = (data << 32) >> 32;
    }
}

inline __attribute__((always_inline)) void recordEvent(uint16_t event_id) {
    if (bufferHasRoom()) {
        mark_time_at_index_inlined(wIndex, get_id(event_id, TS_EVENT));
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
    }
}
}  // namespace kernel_profiler

#define DeviceTimestampedData(data_id, data)

#define DeviceRecordEvent(event_id)

#define DeviceValidateProfiler(condition) ;

#define DeviceZoneScopedMainN(name)                                            \
    kernel_profiler::scopePush scope_push = kernel_profiler::scopePush();      \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

#define DeviceZoneScopedMainChildN(name)                                       \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

#define DeviceZoneScopedN(name)                                                \
    kernel_profiler::scopePush scope_push = kernel_profiler::scopePush();      \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

#define DeviceZoneSetCounter(counter) ;

#define DeviceZonesTimeoutPush() kernel_profiler::push_time_out();

#define DeviceZoneScopedPush() kernel_profiler::scopePush scope_push = kernel_profiler::scopePush();

#define DeviceProfilerInit() kernel_profiler::init_profiler();

#else

#define DeviceValidateProfiler(condition)

#define DeviceZoneScopedMainN(name)

#define DeviceZoneScopedPush()

#define DeviceZoneScopedMainChildN(name)

#define DeviceZoneScopedN(name)

#define DeviceZoneSetCounter(counter)

#define DeviceTimestampedData(data_id, data)

#define DeviceRecordEvent(event_id)

#define DeviceZonesTimeoutPush()

#define DeviceProfilerInit()

#endif

#define DeviceZoneScopedSumN1(name)

#define DeviceZoneScopedSumN2(name)

#define RECORD_NOC_EVENT_WITH_ADDR(type, noc_addr, num_bytes, vc)
#define RECORD_NOC_EVENT_WITH_ID(type, noc_id, num_bytes, vc)
#define RECORD_NOC_EVENT(type)
