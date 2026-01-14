// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <climits>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_AERISC)
#include "risc_common.h"
#include "internal/dataflow/dataflow_api_addrgen.h"
#include "api/tensor/tensor_accessor.h"
#else
#include "ckernel.h"
#endif

#include "hostdevcommon/profiler_common.h"
#include "internal/risc_attribs.h"

#include "hostdev/dev_msgs.h"

#include "internal/ethernet/erisc.h"

#define DO_PRAGMA(x) _Pragma(#x)

#define Stringize(L) #L
#define MakeString(M, L) M(L)
#define $Line MakeString(Stringize, __LINE__)

#define PROFILER_MSG __FILE__ "," $Line ",KERNEL_PROFILER"
#define PROFILER_MSG_NAME(name) name "," PROFILER_MSG

#define SrcLocNameToHash(name)                   \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name))); \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));

#if defined(PROFILE_KERNEL) && \
    (!defined(DISPATCH_KERNEL) || (defined(DISPATCH_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_DISPATCH_CORES)))
namespace kernel_profiler {

extern uint32_t wIndex;
extern uint32_t stackSize;
extern uint32_t traceCount;

extern uint32_t sums[SUM_COUNT];
extern uint32_t sumIDs[SUM_COUNT];

constexpr uint32_t PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC = PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC / sizeof(uint32_t);
constexpr uint32_t NOC_ALIGNMENT_FACTOR = 4;
constexpr uint32_t QUICK_PUSH_MARKER_COUNT = 2;
constexpr uint32_t DISPATCH_META_DATA_COUNT = 2;
constexpr uint32_t DISPATCH_META_DATA_UINT32_SIZE = 4;
constexpr uint32_t DISPATCH_PARENT_ZONE_MARKER_COUNT = 2;

#if (PROFILE_KERNEL & PROFILER_OPT_DO_TRACE_ONLY) && !(defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC))
constexpr bool TRACE_ON_TENSIX = true;
#else
constexpr bool TRACE_ON_TENSIX = false;
#endif
#if (PROFILE_KERNEL & PROFILER_OPT_DO_SUM)
constexpr bool DO_SUM = true;
#else
constexpr bool DO_SUM = false;
#endif
constexpr uint32_t TRACE_MARK_FW_START = (1 << 31);
constexpr uint32_t TRACE_MARK_KERNEL_START = (1 << 30);
constexpr uint32_t TRACE_MARK_ALL_ENDS = (1 << 29);
// Space has to be left in the buffer in order to guarantee
// that the next dispatch command can make it fully populated
// with its meta data (op id + command type)
constexpr uint32_t DISPATCH_HEADROOM_SIZE =
    PROFILER_L1_MARKER_UINT32_SIZE * (DISPATCH_PARENT_ZONE_MARKER_COUNT + QUICK_PUSH_MARKER_COUNT) +
    DISPATCH_META_DATA_UINT32_SIZE * DISPATCH_META_DATA_COUNT;

constexpr int WALL_CLOCK_HIGH_INDEX = 1;
constexpr int WALL_CLOCK_LOW_INDEX = 0;

volatile tt_l1_ptr uint32_t* profiler_control_buffer =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));

volatile tt_l1_ptr profiler_msg_buffer_t* profiler_data_buffer =
    reinterpret_cast<volatile tt_l1_ptr profiler_msg_buffer_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.buffer));

#if (PROFILE_KERNEL & PROFILER_OPT_DO_TRACE_ONLY)
constexpr uint32_t myRiscID = 0;
#else
constexpr uint32_t myRiscID = PROCESSOR_INDEX;
#endif

#if defined(DEVICE_DEBUG_DUMP)
// Each risc has their own DRAM profiler address index
constexpr bool NON_DROPPING = true;
constexpr uint32_t DRAM_PROFILER_ADDRESS = DRAM_PROFILER_ADDRESS_BR_ER_0 + myRiscID;
#else
constexpr bool NON_DROPPING = false;
constexpr uint32_t DRAM_PROFILER_ADDRESS = DRAM_PROFILER_ADDRESS_DEFAULT;
#endif

constexpr uint32_t HOST_BUFFER_END_INDEX = HOST_BUFFER_END_INDEX_BR_ER + myRiscID;

constexpr uint32_t Hash32_CT(const char* str, size_t n, uint32_t basis = UINT32_C(2166136261)) {
    return n == 0 ? basis : Hash32_CT(str + 1, n - 1, (basis ^ str[0]) * UINT32_C(16777619));
}

template <size_t N>
constexpr uint32_t Hash16_CT(const char (&s)[N]) {
    auto res = Hash32_CT(s, N - 1);
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}

enum class DoingDispatch { DISPATCH, DISPATCH_META, NOT_DISPATCH };

__attribute__((noinline)) void init_profiler(
    uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0) {
    wIndex = CUSTOM_MARKERS;
    stackSize = 0;

    for (int i = 0; i < SUM_COUNT; i++) {
        sumIDs[i] = 0;
        sums[i] = 0;
    }

#if defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || \
    defined(COMPILE_FOR_BRISC)
    uint32_t runCounter = profiler_control_buffer[RUN_COUNTER];
    profiler_control_buffer[PROFILER_DONE] = 0;
    if constexpr (NON_DROPPING) {
        profiler_control_buffer[DROPPED_ZONES] = 0;
    }
    if (runCounter == 0) {
        for (uint32_t riscID = 0; riscID < PROCESSOR_COUNT; riscID++) {
            for (uint32_t i = ID_HH; i < GUARANTEED_MARKER_1_H; i++) {
                profiler_data_buffer[riscID].data[i] = 0;
            }
#if !defined(COMPILE_FOR_IDLE_ERISC)
            // Update every risc's trace ID
            profiler_data_buffer[riscID].data[ID_LH] =
                (traceCount & 0xFFFF) << 11 | ((profiler_data_buffer[riscID].data[ID_LH] & 0x7FF));
#endif
        }
        profiler_control_buffer[NOC_X] = my_x[0];
        profiler_control_buffer[NOC_Y] = my_y[0];
    }

    for (uint32_t riscID = 0; riscID < PROCESSOR_COUNT; riscID++) {
        for (uint32_t i = GUARANTEED_MARKER_1_H; i < CUSTOM_MARKERS; i++) {
            // TODO(MO): Clean up magic numbers
            profiler_data_buffer[riscID].data[i] = 0x80000000;
        }
    }
#endif
}

constexpr uint32_t get_const_id(uint32_t id, PacketTypes type) { return ((id & 0xFFFF) | ((type << 16) & 0x7FFFF)); }

inline __attribute__((always_inline)) uint32_t get_id(uint32_t id, PacketTypes type) {
    return ((id & 0xFFFF) | ((type << 16) & 0x7FFFF));
}

template <DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
inline __attribute__((always_inline)) bool bufferHasRoom(uint32_t additional_slots = 0) {
    bool bufferHasRoom = false;
    if constexpr (dispatch == DoingDispatch::DISPATCH) {
        bufferHasRoom = wIndex + additional_slots < (PROFILER_L1_VECTOR_SIZE - stackSize - DISPATCH_HEADROOM_SIZE);
    } else if constexpr (dispatch == DoingDispatch::DISPATCH_META) {
        bufferHasRoom = wIndex + additional_slots < (PROFILER_L1_VECTOR_SIZE - stackSize -
                                                     (QUICK_PUSH_MARKER_COUNT * PROFILER_L1_MARKER_UINT32_SIZE));
    } else {
        bufferHasRoom = wIndex + additional_slots < (PROFILER_L1_VECTOR_SIZE - stackSize);
    }
    return bufferHasRoom;
}

inline __attribute__((always_inline)) void mark_time_at_index_inlined(uint32_t index, uint32_t timer_id) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    profiler_data_buffer[myRiscID].data[index] =
        0x80000000 | ((timer_id & 0x7FFFF) << 12) | (p_reg[WALL_CLOCK_HIGH_INDEX] & 0xFFF);
    profiler_data_buffer[myRiscID].data[index + 1] = p_reg[WALL_CLOCK_LOW_INDEX];
}

inline __attribute__((always_inline)) void mark_padding() {
    if (wIndex < PROFILER_L1_VECTOR_SIZE) {
        profiler_data_buffer[myRiscID].data[wIndex] = 0x80000000;
        profiler_data_buffer[myRiscID].data[wIndex + 1] = 0;
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
    }
}

inline __attribute__((always_inline)) void mark_dropped_timestamps(uint32_t index) {
    uint32_t curr = profiler_control_buffer[DROPPED_ZONES];
    profiler_control_buffer[DROPPED_ZONES] = (1 << index) | curr;
}

inline __attribute__((always_inline)) bool get_dropped_timestamps(uint32_t index) {
    uint32_t curr = profiler_control_buffer[DROPPED_ZONES];
    return ((curr >> index) & 0x1);
}

inline __attribute__((always_inline)) void set_host_counter(uint32_t counterValue) {
    for (uint32_t riscID = 0; riscID < PROCESSOR_COUNT; riscID++) {
        profiler_data_buffer[riscID].data[ID_LL] = counterValue;
    }
}

inline __attribute__((always_inline)) void set_profiler_zone_valid(bool condition) {
    profiler_control_buffer[PROFILER_DONE] = !condition;
}

inline __attribute__((always_inline)) void risc_finished_profiling() {
    for (int i = 0; i < SUM_COUNT; i++) {
        if (sums[i] > 0) {
            if (wIndex < PROFILER_L1_VECTOR_SIZE) {
                profiler_data_buffer[myRiscID].data[wIndex] =
                    0x80000000 | ((get_id(sumIDs[i], ZONE_TOTAL) & 0x7FFFF) << 12);
                profiler_data_buffer[myRiscID].data[wIndex + 1] = sums[i];
                wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
            }
        }
    }

    for (uint32_t i = 0; i < (wIndex % NOC_ALIGNMENT_FACTOR); i++) {
        mark_padding();
    }
    profiler_control_buffer[kernel_profiler::DEVICE_BUFFER_END_INDEX_BR_ER + myRiscID] = wIndex;
}

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0))

// Saves several NoC register states restores them
// when the NocRegisterStateSave is destroyed
struct NocRegisterStateSave {
    uint32_t noc_ctrl_state;
    uint32_t noc_ret_addr_coord_state;
    uint32_t noc_targ_addr_lo_state;
    uint32_t noc_ret_addr_lo_state;
    uint32_t noc_at_len_be_state;
    uint32_t noc_targ_addr_coordinate_state;
    uint32_t noc_targ_addr_mid_state;
    uint32_t noc_packet_tag_state;
    uint32_t noc_at_data_state;

#ifdef ARCH_BLACKHOLE
    uint32_t noc_ret_addr_mid_state;
#endif

    inline __attribute__((always_inline)) NocRegisterStateSave() {
        noc_ctrl_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_CTRL);

        // https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/NoC/MemoryMap.md#noc_ctrl
        constexpr uint32_t reserved_bit_mask = ((1u << 27) - (1u << 18)) | (1u << 31);
        noc_ctrl_state &= ~reserved_bit_mask;

        noc_ret_addr_coord_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE);
        noc_targ_addr_lo_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_LO);
        noc_ret_addr_lo_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_LO);
        noc_at_len_be_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_AT_LEN_BE);
        noc_targ_addr_coordinate_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_COORDINATE);
        noc_targ_addr_mid_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_MID);

        noc_packet_tag_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_PACKET_TAG);
        // reset the counter to zero before the push
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_PACKET_TAG, 0);

        noc_at_data_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_AT_DATA);
#ifdef ARCH_BLACKHOLE
        noc_ret_addr_mid_state = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_MID);
#endif
    }

    inline __attribute__((always_inline)) ~NocRegisterStateSave() {
        while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_CTRL, noc_ctrl_state);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE, noc_ret_addr_coord_state);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_LO, noc_targ_addr_lo_state);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_LO, noc_ret_addr_lo_state);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_AT_LEN_BE, noc_at_len_be_state);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_COORDINATE, noc_targ_addr_coordinate_state);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_MID, noc_targ_addr_mid_state);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_PACKET_TAG, noc_packet_tag_state);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_AT_DATA, noc_at_data_state);
#ifdef ARCH_BLACKHOLE
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_MID, noc_ret_addr_mid_state);
#endif
    }
};

inline void __attribute__((always_inline)) profiler_noc_async_write_posted(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index) {
    WAYPOINT("NAWW");
#if !defined(KERNEL_BUILD)
    constexpr uint8_t noc_mode = DM_DEDICATED_NOC;
#endif
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

// Signal the host that this RISC's destination DRAM buffer is full and wait for a new DRAM profiler address
__attribute__((noinline)) void signal_host_buffer_full(uint32_t control_buffer_index_for_dram = DRAM_PROFILER_ADDRESS) {
    profiler_control_buffer[control_buffer_index_for_dram] = DRAM_PROFILER_ADDRESS_STALLED;

    // Wait for host to give new profiler address
    do {
        invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
        internal_::risc_context_switch(false);
#endif
    } while (profiler_control_buffer[control_buffer_index_for_dram] == DRAM_PROFILER_ADDRESS_STALLED);
}

__attribute__((noinline)) void finish_profiler() {
    risc_finished_profiling();
#if defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || \
    defined(COMPILE_FOR_BRISC)
    if (profiler_control_buffer[PROFILER_DONE] == 1) {
        return;
    }
    uint32_t core_flat_id = profiler_control_buffer[FLAT_ID];
    uint32_t profiler_core_count_per_dram = profiler_control_buffer[CORE_COUNT_PER_DRAM];
    bool is_dram_set = profiler_control_buffer[DRAM_PROFILER_ADDRESS] != 0;
    int dramProfilerAddressIndex = DRAM_PROFILER_ADDRESS;

    uint32_t pageSize =
        PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * MaxProcessorsPerCoreType * profiler_core_count_per_dram;

    NocRegisterStateSave noc_state;
    for (uint32_t riscID = 0; riscID < PROCESSOR_COUNT; riscID++) {
        bool do_noc = true;

        if constexpr (NON_DROPPING) {
            dramProfilerAddressIndex = kernel_profiler::DRAM_PROFILER_ADDRESS_BR_ER_0 + riscID;
            is_dram_set = profiler_control_buffer[dramProfilerAddressIndex] != 0;
        }

#if defined(COMPILE_FOR_IDLE_ERISC)
        profiler_data_buffer[riscID].data[ID_LH] = ((core_flat_id & 0xFF) << 3) | riscID;
#else
        // Need to preserve the upper bits of ID_LH which contain the trace ID
        profiler_data_buffer[riscID].data[ID_LH] =
            (profiler_data_buffer[riscID].data[ID_LH] & 0x7FFF800) | (((core_flat_id & 0xFF) << 3) | riscID);
#endif
        int hostIndex = kernel_profiler::HOST_BUFFER_END_INDEX_BR_ER + riscID;
        int deviceIndex = kernel_profiler::DEVICE_BUFFER_END_INDEX_BR_ER + riscID;
        if (profiler_control_buffer[deviceIndex]) {
            uint32_t currEndIndexAll = profiler_control_buffer[deviceIndex] + profiler_control_buffer[hostIndex];
            uint32_t currEndIndexGuaranteed = CUSTOM_MARKERS + profiler_control_buffer[hostIndex];
            uint32_t send_size = 0;
            uint32_t dram_offset = (core_flat_id % profiler_core_count_per_dram) * MaxProcessorsPerCoreType *
                                       PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                                   hostIndex * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                                   profiler_control_buffer[hostIndex] * sizeof(uint32_t);

            if constexpr (NON_DROPPING) {
                // Send everything
                if (currEndIndexAll > PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC) {
                    signal_host_buffer_full(dramProfilerAddressIndex);
                    // Host index is reset because we got a new DRAM buffer
                    profiler_control_buffer[hostIndex] = 0;
                    currEndIndexAll = profiler_control_buffer[deviceIndex] + profiler_control_buffer[hostIndex];
                }
            }

            if (currEndIndexAll <= PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC) {
                send_size = profiler_control_buffer[deviceIndex] * sizeof(uint32_t);
                profiler_control_buffer[hostIndex] = currEndIndexAll;
            } else if (currEndIndexGuaranteed <= PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC) {
                // At least send the guaranteed markers
                send_size = CUSTOM_MARKERS * sizeof(uint32_t);
                profiler_control_buffer[hostIndex] = currEndIndexGuaranteed;
                mark_dropped_timestamps(hostIndex);
            } else {
                // If we get here, host will trigger TT_FATAL on missing data
                do_noc = false;
                mark_dropped_timestamps(hostIndex);
            }

            if (do_noc && is_dram_set) {
                const auto s = TensorAccessor(
                    tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(),
                    profiler_control_buffer[dramProfilerAddressIndex],
                    pageSize);

                uint64_t dram_bank_dst_noc_addr =
                    s.get_noc_addr(core_flat_id / profiler_core_count_per_dram, dram_offset);

                profiler_noc_async_write_posted(
                    reinterpret_cast<uint32_t>(profiler_data_buffer[hostIndex].data),
                    dram_bank_dst_noc_addr,
                    send_size);
            }
        }
    }

    profiler_noc_async_flush_posted_write();
    profiler_control_buffer[RUN_COUNTER]++;
    profiler_control_buffer[PROFILER_DONE] = 1;
#endif
}

__attribute__((noinline)) void quick_push() {
#if (                                                                                               \
    defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_IDLE_ERISC) || \
    (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)))

    // tt-metal/issues/22578 - forbid quick_push if any cmd buffer has NOC_CMD_VC_LINKED bit set
    auto linked_bit_is_set = [](const uint32_t reg_val) { return reg_val & NOC_CMD_VC_LINKED; };
    uint32_t read_buf_reg = NOC_CMD_BUF_READ_REG(noc_index, read_cmd_buf, NOC_CTRL);
    uint32_t write_buf_reg = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_CTRL);
    uint32_t write_reg_buf_reg = NOC_CMD_BUF_READ_REG(noc_index, write_reg_cmd_buf, NOC_CTRL);
    uint32_t write_at_buf_reg = NOC_CMD_BUF_READ_REG(noc_index, write_at_cmd_buf, NOC_CTRL);
    if (linked_bit_is_set(read_buf_reg) || linked_bit_is_set(write_buf_reg) || linked_bit_is_set(write_reg_buf_reg) ||
        linked_bit_is_set(write_at_buf_reg)) {
        return;
    }
    if (!profiler_control_buffer[DRAM_PROFILER_ADDRESS] || get_dropped_timestamps(myRiscID)) {
        return;
    }

    SrcLocNameToHash("PROFILER-NOC-QUICK-PUSH");
    mark_time_at_index_inlined(wIndex, hash);
    wIndex += PROFILER_L1_MARKER_UINT32_SIZE;

    uint32_t core_flat_id = profiler_control_buffer[FLAT_ID];
    uint32_t profiler_core_count_per_dram = profiler_control_buffer[CORE_COUNT_PER_DRAM];

    profiler_data_buffer[myRiscID].data[ID_LH] =
        (profiler_data_buffer[myRiscID].data[ID_LH] & 0x7FFF800) | (((core_flat_id & 0xFF) << 3) | myRiscID);

    uint32_t currEndIndex = profiler_control_buffer[HOST_BUFFER_END_INDEX] + wIndex;

    mark_time_at_index_inlined(wIndex, get_const_id(hash, ZONE_END));
    wIndex += PROFILER_L1_MARKER_UINT32_SIZE;

    if constexpr (NON_DROPPING) {
        if (currEndIndex > PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC) {
            signal_host_buffer_full();
            // Host index is reset because we got a new DRAM buffer
            profiler_control_buffer[HOST_BUFFER_END_INDEX] = 0;
            currEndIndex = wIndex;
        }
    }

    uint32_t dram_offset = (core_flat_id % profiler_core_count_per_dram) * MaxProcessorsPerCoreType *
                               PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                           HOST_BUFFER_END_INDEX * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                           profiler_control_buffer[HOST_BUFFER_END_INDEX] * sizeof(uint32_t);

    const auto s = TensorAccessor(
        tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(),
        profiler_control_buffer[DRAM_PROFILER_ADDRESS],
        PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * MaxProcessorsPerCoreType * profiler_core_count_per_dram);

    uint64_t dram_bank_dst_noc_addr = s.get_noc_addr(core_flat_id / profiler_core_count_per_dram, dram_offset);

    for (uint32_t i = 0; i < (wIndex % NOC_ALIGNMENT_FACTOR); i++) {
        mark_padding();
    }

    currEndIndex = profiler_control_buffer[HOST_BUFFER_END_INDEX] + wIndex;

    // If sending all optional markers still leaves room for the two guaranteed end markers, send everything
    if (currEndIndex <= PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC -
                            (PROFILER_L1_GUARANTEED_MARKER_COUNT / 2) * PROFILER_L1_MARKER_UINT32_SIZE) {
        NocRegisterStateSave noc_state;
        profiler_noc_async_write_posted(
            reinterpret_cast<uint32_t>(profiler_data_buffer[myRiscID].data),
            dram_bank_dst_noc_addr,
            wIndex * sizeof(uint32_t));

        profiler_noc_async_flush_posted_write();
        profiler_control_buffer[HOST_BUFFER_END_INDEX] = currEndIndex;
    } else {
        mark_dropped_timestamps(HOST_BUFFER_END_INDEX);
    }

    wIndex = CUSTOM_MARKERS;
#endif
}

// Initiates a quick_push() if the specified cmd buf is NOT currently in linked
// state, and linked arg is set to true. Useful for preemptively flushing to
// DRAM in the event that a long series of linked multicast will prevent
// flushing and cause dropped events.
void quick_push_if_linked(uint32_t cmd_buf, bool linked) {
#if (                                                                                               \
    defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_IDLE_ERISC) || \
    (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)))
    if (!linked) {
        return;
    }
    uint32_t cmd_buf_reg_val = NOC_CMD_BUF_READ_REG(noc_index, cmd_buf, NOC_CTRL);
    bool cmd_buf_currently_linked = cmd_buf_reg_val & NOC_CMD_VC_LINKED;
    if (!cmd_buf_currently_linked) {
        kernel_profiler::quick_push();
    }
#endif
}

template <uint32_t timer_id, DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
struct profileScope {
    bool start_marked = false;
    inline __attribute__((always_inline)) profileScope() {
        if (bufferHasRoom<dispatch>()) {
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
            if constexpr (dispatch == DoingDispatch::DISPATCH) {
                if (wIndex >= (PROFILER_L1_VECTOR_SIZE - DISPATCH_HEADROOM_SIZE)) {
                    quick_push();
                }
            }
        }
    }
};

template <uint32_t timer_id, uint32_t index>
struct profileScopeGuaranteed {
    static constexpr uint32_t start_index = (2 * index * PROFILER_L1_MARKER_UINT32_SIZE) + GUARANTEED_MARKER_1_H;
    static constexpr uint32_t end_index = (2 * index * PROFILER_L1_MARKER_UINT32_SIZE) + GUARANTEED_MARKER_2_H;

    static_assert(start_index < CUSTOM_MARKERS);
    static_assert(end_index < CUSTOM_MARKERS);
    inline __attribute__((always_inline)) profileScopeGuaranteed() {
        if constexpr (TRACE_ON_TENSIX) {
            uint32_t trace_replay_status = profiler_control_buffer[TRACE_REPLAY_STATUS];
            if constexpr (index == 0) {
#if !defined(COMPILE_FOR_TRISC)
                if (trace_replay_status & TRACE_MARK_FW_START) {
                    mark_time_at_index_inlined(start_index, get_const_id(timer_id, ZONE_START));
                    profiler_control_buffer[TRACE_REPLAY_STATUS] = TRACE_MARK_KERNEL_START;
                }
#endif
            } else {
                if (trace_replay_status & TRACE_MARK_KERNEL_START) {
                    mark_time_at_index_inlined(start_index, get_const_id(timer_id, ZONE_START));
                    profiler_control_buffer[TRACE_REPLAY_STATUS] = TRACE_MARK_ALL_ENDS;
                }
            }
        } else {
            if constexpr (index == 0) {
                init_profiler();
            }
            mark_time_at_index_inlined(start_index, get_const_id(timer_id, ZONE_START));
        }
    }
    inline __attribute__((always_inline)) ~profileScopeGuaranteed() {
        if constexpr (TRACE_ON_TENSIX) {
            if (profiler_control_buffer[TRACE_REPLAY_STATUS] == TRACE_MARK_ALL_ENDS) {
                mark_time_at_index_inlined(end_index, get_const_id(timer_id, ZONE_END));
#if defined(COMPILE_FOR_BRISC)
                // Validate profiler_data_buffer in L1
                profiler_data_buffer[myRiscID].data[ID_HH] = 0x0;
#endif
            }
            if constexpr (index == 0) {
                profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER] = wIndex;
            }
        } else {
            mark_time_at_index_inlined(end_index, get_const_id(timer_id, ZONE_END));
            if constexpr (index == 0) {
                finish_profiler();
            }
        }
    }
};

template <uint32_t timer_id, uint32_t index>
struct profileScopeAccumulate {
    uint64_t start_time = 0;
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);

    inline __attribute__((always_inline)) profileScopeAccumulate() {
        if constexpr (kernel_profiler::DO_SUM) {
            start_time = ((uint64_t)p_reg[WALL_CLOCK_HIGH_INDEX] << 32) | p_reg[WALL_CLOCK_LOW_INDEX];
        }
    }
    inline __attribute__((always_inline)) ~profileScopeAccumulate() {
        if constexpr (kernel_profiler::DO_SUM) {
            sumIDs[index] = timer_id;
            sums[index] += (((uint64_t)p_reg[WALL_CLOCK_HIGH_INDEX] << 32) | p_reg[WALL_CLOCK_LOW_INDEX]) - start_time;
        }
    }
};

// performs quick push to DRAM if buffers appear full
template <DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
inline __attribute__((always_inline)) void flush_to_dram_if_full(uint32_t additional_slots = 0) {
    if (not bufferHasRoom<dispatch>(additional_slots)) {
        quick_push();
    }
}

template <
    uint32_t data_id,
    DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH,
    PacketTypes packet_type = kernel_profiler::PacketTypes::TS_DATA,
    typename... Args>
inline __attribute__((always_inline)) void timeStampedData(uint64_t data, Args... trailers) {
    constexpr uint32_t total_data_count = 1 + sizeof...(trailers);
    constexpr uint32_t expected_size = kernel_profiler::TimestampedDataSize<packet_type>::size;

    static_assert(
        expected_size == 0 || total_data_count == expected_size,
        "Number of arguments does not match expected size for this PacketType");

    constexpr uint32_t additional_slots = sizeof...(trailers);

    if (bufferHasRoom<dispatch>(additional_slots)) {
        mark_time_at_index_inlined(wIndex, get_const_id(data_id, packet_type));
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;

        profiler_data_buffer[myRiscID].data[wIndex++] = data >> 32;
        profiler_data_buffer[myRiscID].data[wIndex++] = (data << 32) >> 32;

        ((profiler_data_buffer[myRiscID].data[wIndex++] = trailers >> 32,
          profiler_data_buffer[myRiscID].data[wIndex++] = (trailers << 32) >> 32),
         ...);
    }
}

template <DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
inline __attribute__((always_inline)) void recordEvent(uint16_t event_id) {
    if (bufferHasRoom<dispatch>()) {
        mark_time_at_index_inlined(wIndex, get_id(event_id, TS_EVENT));
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
    }
}

inline __attribute__((always_inline)) void increment_trace_count() {
    if constexpr (!TRACE_ON_TENSIX) {
        traceCount++;
        for (uint32_t riscID = 0; riscID < PROCESSOR_COUNT; riscID++) {
#if !defined(COMPILE_FOR_IDLE_ERISC)
            // Update every risc's trace ID
            profiler_data_buffer[riscID].data[ID_LH] =
                (traceCount & 0xFFFF) << 11 | ((profiler_data_buffer[riscID].data[ID_LH] & 0x7FF));
#endif
        }
    }
}

__attribute__((noinline)) void trace_only_init() {
    if constexpr (TRACE_ON_TENSIX) {
        if (traceCount > 0) {
            quick_push();
        }
        traceCount++;
        set_host_counter(traceCount);
        profiler_control_buffer[TRACE_REPLAY_STATUS] = TRACE_MARK_FW_START;
        // Invalidate profiler_data_buffer in L1
        // As the start of profiler buffer ID_HH = 0x0
        // indicates valid profiler data
        profiler_data_buffer[myRiscID].data[ID_HH] = 0x80000000;
    }
}

}  // namespace kernel_profiler

#include "noc_event_profiler.hpp"
#include "perf_counters.hpp"

// Not dispatch
#if (!defined(DISPATCH_KERNEL))

#define DeviceZoneScopedN(name)                                                \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

#define DeviceTimestampedData(name, data)                                          \
    {                                                                              \
        DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
        auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
        kernel_profiler::timeStampedData<hash>(data);                              \
    }

#define DeviceRecordEvent(event_id) kernel_profiler::recordEvent(event_id);

// Dispatch and enabled
#elif (defined(DISPATCH_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_DISPATCH_CORES))

#define DeviceZoneScopedN(name)                                                         \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                         \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));           \
    kernel_profiler::profileScope<hash, kernel_profiler::DoingDispatch::DISPATCH> zone = \
        kernel_profiler::profileScope<hash, kernel_profiler::DoingDispatch::DISPATCH>();

#define DeviceTimestampedData(name, data)                                                            \
    {                                                                                                \
        DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                                 \
        auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));                   \
        kernel_profiler::timeStampedData<hash, kernel_profiler::DoingDispatch::DISPATCH_META>(data); \
    }

#define DeviceRecordEvent(event_id) kernel_profiler::recordEvent<kernel_profiler::DoingDispatch::DISPATCH>(event_id);

// Dispatch but disabled
#else

#define DeviceZoneScopedN(name)

#define DeviceTimestampedData(data_id, data)

#define DeviceRecordEvent(event_id)

#endif

#define DeviceValidateProfiler(condition) kernel_profiler::set_profiler_zone_valid(condition);

#define DeviceZoneScopedMainN(name)                                            \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeGuaranteed<hash, 0> zone = kernel_profiler::profileScopeGuaranteed<hash, 0>();

#define DeviceZoneScopedMainChildN(name)                                       \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeGuaranteed<hash, 1> zone = kernel_profiler::profileScopeGuaranteed<hash, 1>();

#define DeviceZoneScopedSumN1(name)                                            \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeAccumulate<hash, 0> zone = kernel_profiler::profileScopeAccumulate<hash, 0>();

#define DeviceZoneScopedSumN2(name)                                            \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeAccumulate<hash, 1> zone = kernel_profiler::profileScopeAccumulate<hash, 1>();

#define DeviceZoneSetCounter(counter)                  \
    if constexpr (!kernel_profiler::TRACE_ON_TENSIX) { \
        kernel_profiler::set_host_counter(counter);    \
    }

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_AERISC)
#define DeviceProfilerInit()                          \
    if constexpr (kernel_profiler::TRACE_ON_TENSIX) { \
        kernel_profiler::init_profiler();             \
    }                                                 \
    kernel_profiler::traceCount = 0;
#else
#define DeviceProfilerInit()                          \
    if constexpr (kernel_profiler::TRACE_ON_TENSIX) { \
        kernel_profiler::init_profiler();             \
    }
#endif

#define DeviceTraceOnlyProfilerInit() kernel_profiler::trace_only_init();

#define DeviceIncrementTraceCount() kernel_profiler::increment_trace_count();

#else

#define DeviceValidateProfiler(condition)

#define DeviceZoneScopedMainN(name)

#define DeviceZoneScopedMainChildN(name)

#define DeviceZoneScopedN(name)

#define DeviceZoneScopedSumN1(name)

#define DeviceZoneScopedSumN2(name)

#define DeviceTraceOnlyProfilerInit()

#define DeviceZoneSetCounter(counter)

#define DeviceTimestampedData(data_id, data)

#define DeviceRecordEvent(event_id)

#define DeviceProfilerInit()

#define DeviceIncrementTraceCount()

// null macros when noc tracing is disabled
#define RECORD_NOC_EVENT_WITH_ADDR(type, noc_addr, num_bytes, vc)
#define RECORD_NOC_EVENT_WITH_ID(type, noc_id, addrgen, offset, num_bytes, vc)
#define RECORD_NOC_EVENT(type)
#define NOC_TRACE_QUICK_PUSH_IF_LINKED(cmd_buf, linked)

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define RecordPerfCounters()

#endif
