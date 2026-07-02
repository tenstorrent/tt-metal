// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SPSC variant of the device kernel profiler (X280-drained).
//
// Same public macro API as the original (now kept verbatim in
// kernel_profiler_push.hpp), but a different backend: instead of filling an L1
// buffer and pushing it to DRAM on quick_push()/finish(), each RISC streams its
// markers into a per-RISC single-producer/single-consumer (SPSC) ring in L1. The
// **X280** is the consumer that continuously drains those rings. The producing
// RISC **blocks** (spins on the consumer head) when its ring is full — so the
// stream is lossless and flow-controlled, with **no DRAM traffic** on
// quick_push() or finish().
//
//   Per RISC `r` (Tensix: BRISC/NCRISC/TRISC0-2):
//     ring storage : profiler_data_buffer[r].data[0 .. PROFILER_L1_VECTOR_SIZE-1]
//     tail (prod.) : profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + r]
//     head (cons.) : profiler_control_buffer[HOST_BUFFER_END_INDEX_BR_ER  + r]
//   tail/head are MONOTONIC word counts; storage index = count % CAPACITY.
//   Append blocks while (tail - head) > CAPACITY - need, then writes + publishes
//   tail. The X280 advances head as it drains.
//
// NOTE: with this backend a profiled run REQUIRES the X280 consumer to be draining
// — if a ring fills and nothing drains it, the producing RISC blocks (by design).
// Tensix-focused; ETH cores are not a target here.

#pragma once

#include <climits>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_AERISC) || defined(COMPILE_FOR_DM)
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

extern uint32_t wIndex;  // producer tail (monotonic word count for this RISC's ring)
extern uint32_t stackSize;
extern uint32_t traceCount;

extern uint32_t sums[SUM_COUNT];
extern uint32_t sumIDs[SUM_COUNT];

constexpr uint32_t NOC_ALIGNMENT_FACTOR = 4;

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

// SPSC backend never drops — it blocks on a full ring — so dropping is always off.
// (Kept because noc_event_profiler.hpp references kernel_profiler::NON_DROPPING.)
constexpr bool NON_DROPPING = false;

constexpr uint32_t TRACE_MARK_FW_START = (1 << 31);
constexpr uint32_t TRACE_MARK_KERNEL_START = (1 << 30);
constexpr uint32_t TRACE_MARK_ALL_ENDS = (1 << 29);

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

// SPSC ring geometry for this RISC.
constexpr uint32_t RING_CAPACITY = PROFILER_L1_VECTOR_SIZE;                // words (= data[] length)
constexpr uint32_t TAIL_INDEX = DEVICE_BUFFER_END_INDEX_BR_ER + myRiscID;  // producer (this RISC)
constexpr uint32_t HEAD_INDEX = HOST_BUFFER_END_INDEX_BR_ER + myRiscID;    // consumer (X280)

constexpr uint32_t Hash32_CT(const char* str, size_t n, uint32_t basis = UINT32_C(2166136261)) {
    return n == 0 ? basis : Hash32_CT(str + 1, n - 1, (basis ^ str[0]) * UINT32_C(16777619));
}

template <size_t N>
constexpr uint32_t Hash16_CT(const char (&s)[N]) {
    auto res = Hash32_CT(s, N - 1);
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}

enum class DoingDispatch { DISPATCH, DISPATCH_META, NOT_DISPATCH };

constexpr uint32_t get_const_id(uint32_t id, PacketTypes type) { return ((id & 0xFFFF) | ((type << 16) & 0x7FFFF)); }

inline __attribute__((always_inline)) uint32_t get_id(uint32_t id, PacketTypes type) {
    return ((id & 0xFFFF) | ((type << 16) & 0x7FFFF));
}

// ---- SPSC ring primitives -------------------------------------------------

// Block until the X280 consumer has drained enough that `nwords` more fit.
inline __attribute__((always_inline)) void ring_ensure_room(uint32_t nwords) {
    // outstanding = tail - head (monotonic, unsigned). Room iff outstanding <= CAP - nwords.
    while ((wIndex - profiler_control_buffer[HEAD_INDEX]) > (RING_CAPACITY - nwords)) {
        invalidate_l1_cache();  // re-read the X280-updated head (and the terminate flag)
        // Terminate phase (device teardown / X280 consumer stopping): stop blocking so this RISC
        // can reach "done". We drop the marker instead of stalling forever on a ring nobody drains.
        if (profiler_control_buffer[PROFILER_TERMINATE]) {
            return;
        }
    }
}

inline __attribute__((always_inline)) void ring_write_word(uint32_t v) {
    profiler_data_buffer[myRiscID].data[wIndex % RING_CAPACITY] = v;
    wIndex++;
}

inline __attribute__((always_inline)) void publish_tail() { profiler_control_buffer[TAIL_INDEX] = wIndex; }

// Append one 2-word timing marker (timer_id + wall clock), blocking if full.
inline __attribute__((always_inline)) void mark_time(uint32_t timer_id) {
    ring_ensure_room(PROFILER_L1_MARKER_UINT32_SIZE);
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    ring_write_word(0x80000000 | ((timer_id & 0x7FFFF) << 12) | (p_reg[WALL_CLOCK_HIGH_INDEX] & 0xFFF));
    ring_write_word(p_reg[WALL_CLOCK_LOW_INDEX]);
    publish_tail();
}

// Fixed-index write retained only for the trace-only build mode (writes directly
// into the ring storage region; not used by the default SPSC path).
inline __attribute__((always_inline)) void mark_time_at_index_inlined(uint32_t index, uint32_t timer_id) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    profiler_data_buffer[myRiscID].data[index] =
        0x80000000 | ((timer_id & 0x7FFFF) << 12) | (p_reg[WALL_CLOCK_HIGH_INDEX] & 0xFFF);
    profiler_data_buffer[myRiscID].data[index + 1] = p_reg[WALL_CLOCK_LOW_INDEX];
}

// ---- dropped-timestamp bookkeeping (no drops in blocking mode; kept for API) --

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

__attribute__((noinline)) void init_profiler(
    uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0) {
    stackSize = 0;
    for (int i = 0; i < SUM_COUNT; i++) {
        sumIDs[i] = 0;
        sums[i] = 0;
    }

#if defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || \
    defined(COMPILE_FOR_BRISC)
    uint32_t runCounter = profiler_control_buffer[RUN_COUNTER];
    profiler_control_buffer[PROFILER_DONE] = 0;
    if (runCounter == 0) {
        // First launch: empty every RISC's ring (head = tail = 0) and stamp coords.
        for (uint32_t riscID = 0; riscID < PROCESSOR_COUNT; riscID++) {
            profiler_control_buffer[HOST_BUFFER_END_INDEX_BR_ER + riscID] = 0;
            profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + riscID] = 0;
            profiler_data_buffer[riscID].data[ID_HH] = 0;
        }
        profiler_control_buffer[NOC_X] = my_x[0];
        profiler_control_buffer[NOC_Y] = my_y[0];
    }
#endif
    // Resume this RISC's monotonic tail from L1 (rings persist across launches).
    wIndex = profiler_control_buffer[TAIL_INDEX];
}

// Append accumulated SUM zones (if any) and publish the tail. No DRAM.
inline __attribute__((always_inline)) void risc_finished_profiling() {
    for (int i = 0; i < SUM_COUNT; i++) {
        if (sums[i] > 0) {
            ring_ensure_room(PROFILER_L1_MARKER_UINT32_SIZE);
            ring_write_word(0x80000000 | ((get_id(sumIDs[i], ZONE_TOTAL) & 0x7FFFF) << 12));
            ring_write_word(sums[i]);
        }
    }
    publish_tail();
}

__attribute__((noinline)) void finish_profiler() {
    risc_finished_profiling();
#if defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || \
    defined(COMPILE_FOR_BRISC)
    profiler_control_buffer[RUN_COUNTER]++;
    profiler_control_buffer[PROFILER_DONE] = 1;
#endif
}

// No DRAM push in the SPSC backend — the X280 drains the ring continuously. These
// are kept (as tail publishes / no-ops) so the existing call sites still compile.
__attribute__((noinline)) void quick_push() { publish_tail(); }

inline __attribute__((always_inline)) void quick_push_if_linked(uint32_t cmd_buf, bool linked) {
    (void)cmd_buf;
    (void)linked;
}

template <DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
inline __attribute__((always_inline)) void flush_to_dram_if_full(uint32_t additional_slots = 0) {
    (void)additional_slots;
}

template <uint32_t timer_id, DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
struct profileScope {
    inline __attribute__((always_inline)) profileScope() { mark_time(timer_id); }
    inline __attribute__((always_inline)) ~profileScope() { mark_time(get_const_id(timer_id, ZONE_END)); }
};

template <uint32_t timer_id, uint32_t index>
struct profileScopeGuaranteed {
    inline __attribute__((always_inline)) profileScopeGuaranteed() {
        if constexpr (TRACE_ON_TENSIX) {
            uint32_t trace_replay_status = profiler_control_buffer[TRACE_REPLAY_STATUS];
            if constexpr (index == 0) {
#if !defined(COMPILE_FOR_TRISC)
                if (trace_replay_status & TRACE_MARK_FW_START) {
                    mark_time(get_const_id(timer_id, ZONE_START));
                    profiler_control_buffer[TRACE_REPLAY_STATUS] = TRACE_MARK_KERNEL_START;
                }
#endif
            } else {
                if (trace_replay_status & TRACE_MARK_KERNEL_START) {
                    mark_time(get_const_id(timer_id, ZONE_START));
                    profiler_control_buffer[TRACE_REPLAY_STATUS] = TRACE_MARK_ALL_ENDS;
                }
            }
        } else {
            if constexpr (index == 0) {
                init_profiler();
            }
            mark_time(get_const_id(timer_id, ZONE_START));
        }
    }
    inline __attribute__((always_inline)) ~profileScopeGuaranteed() {
        if constexpr (TRACE_ON_TENSIX) {
            if (profiler_control_buffer[TRACE_REPLAY_STATUS] == TRACE_MARK_ALL_ENDS) {
                mark_time(get_const_id(timer_id, ZONE_END));
            }
        } else {
            mark_time(get_const_id(timer_id, ZONE_END));
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

    // 1 timing marker (2 words) + 2 words per 64-bit datum.
    ring_ensure_room(PROFILER_L1_MARKER_UINT32_SIZE + 2 * total_data_count);

    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t marker_id = get_const_id(data_id, packet_type);
    ring_write_word(0x80000000 | ((marker_id & 0x7FFFF) << 12) | (p_reg[WALL_CLOCK_HIGH_INDEX] & 0xFFF));
    ring_write_word(p_reg[WALL_CLOCK_LOW_INDEX]);

    ring_write_word(data >> 32);
    ring_write_word((data << 32) >> 32);
    ((ring_write_word(trailers >> 32), ring_write_word((trailers << 32) >> 32)), ...);
    publish_tail();
}

template <DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
inline __attribute__((always_inline)) void recordEvent(uint16_t event_id) {
    mark_time(get_id(event_id, TS_EVENT));
}

inline __attribute__((always_inline)) void increment_trace_count() {
    if constexpr (!TRACE_ON_TENSIX) {
        traceCount++;
    }
}

__attribute__((noinline)) void trace_only_init() {
    if constexpr (TRACE_ON_TENSIX) {
        traceCount++;
        set_host_counter(traceCount);
        profiler_control_buffer[TRACE_REPLAY_STATUS] = TRACE_MARK_FW_START;
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

#define DeviceZoneScopedN(name)                                                          \
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

#define DeviceZoneScopedN(name) (void(sizeof(name)))

#define DeviceTimestampedData(data_id, data) (void(sizeof(data_id) + sizeof(data)))

#define DeviceRecordEvent(event_id) (void(sizeof(event_id)))

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

// The void(sizeof(FOO)) idiom (a) ensures FOO is syntactically and
// semantically sane and (b) means that we avoid 'var-set-but-unused'
// diagnostics, if the only use of a particular var is here.  The
// sizeof argument is processed in a non-evaluating context -- no code
// is generated.
#define DeviceValidateProfiler(condition) (void(sizeof(condition)))

#define DeviceZoneScopedMainN(name) (void(name))

#define DeviceZoneScopedMainChildN(name) (void(name))

#define DeviceZoneScopedN(name) (void(name))

#define DeviceZoneScopedSumN1(name) (void(name))

#define DeviceZoneScopedSumN2(name) (void(name))

#define DeviceTraceOnlyProfilerInit()

#define DeviceZoneSetCounter(counter) (void(sizeof(counter)))

#define DeviceTimestampedData(data_id, data) (void(sizeof(data_id) + sizeof(data)))

#define DeviceRecordEvent(event_id) (void(sizeof(event_id)))

#define DeviceProfilerInit()

#define DeviceIncrementTraceCount()

// null macros when noc tracing is disabled
#define RECORD_NOC_EVENT_WITH_ADDR(type, local_addr, noc_addr, num_bytes, vc, posted, noc)
#define RECORD_NOC_EVENT_WITH_ID(type, local_addr, noc_id, addrgen, offset, num_bytes, vc, posted, noc)
#define RECORD_NOC_EVENT(type, posted, noc)
#define NOC_TRACE_QUICK_PUSH_IF_LINKED(cmd_buf, linked)

// null macros when noc debugging is disabled
#define RECORD_SCOPED_LOCK_EVENT(event_type, locked_address_base, num_bytes)

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define RecordPerfCounters()

#endif
