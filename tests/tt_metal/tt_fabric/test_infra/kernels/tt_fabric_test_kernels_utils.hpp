// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_fabric_test_pattern.hpp"

namespace tt::tt_fabric {
namespace fabric_tests {

// Fixed per-core connection storage shared by all kernel configurations.
#ifdef ARCH_BLACKHOLE
static constexpr uint8_t MAX_NUM_FABRIC_CONNECTIONS = 5;
#else
static constexpr uint8_t MAX_NUM_FABRIC_CONNECTIONS = 4;
#endif

struct LocalArgsBuffer {
    uint32_t base_address = 0;
    uint32_t buffer_size = 0;

    void init(uint32_t base_addr, uint32_t buf_size) {
        base_address = base_addr;
        buffer_size = buf_size;
    }

    template <typename T>
    FORCE_INLINE T get_arg_val(size_t arg_idx) {
        static_assert("Error: only 4B args are supported" && sizeof(T) == 4);

        uint32_t current_offset = arg_idx * sizeof(T);
        ASSERT(current_offset + sizeof(T) <= buffer_size);

        tt_l1_ptr T* local_args_ptr = reinterpret_cast<tt_l1_ptr T*>(base_address);
        return local_args_ptr[arg_idx];
    }
};

// Global instance of the local args buffer manager
static LocalArgsBuffer local_args_buffer;

inline void init_local_args(uint32_t base_address, uint32_t buffer_size) {
    local_args_buffer.init(base_address, buffer_size);
}

template <typename T>
FORCE_INLINE T get_local_arg_val(size_t arg_idx) {
    return local_args_buffer.get_arg_val<T>(arg_idx);
}


// Helper functions for writing test results
inline void write_test_status(uint32_t result_buffer_base, uint32_t status) {
    auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base);
    result_buffer[TT_FABRIC_STATUS_INDEX] = status;
}

inline void write_test_cycles(uint32_t result_buffer_base, uint64_t cycles) {
    auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base);
    result_buffer[TT_FABRIC_CYCLES_INDEX] = static_cast<uint32_t>(cycles);
    result_buffer[TT_FABRIC_CYCLES_INDEX + 1] = static_cast<uint32_t>(cycles >> 32);
}

inline void write_test_packets(uint32_t result_buffer_base, uint64_t packets) {
    auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base);
    result_buffer[TT_FABRIC_WORD_CNT_INDEX] = static_cast<uint32_t>(packets);
    result_buffer[TT_FABRIC_WORD_CNT_INDEX + 1] = static_cast<uint32_t>(packets >> 32);
}

inline void clear_test_results(uint32_t result_buffer_base, uint32_t result_buffer_size) {
    auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base);
    uint32_t num_words = result_buffer_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; i++) {
        result_buffer[i] = 0;
    }
}

// Opt-in cost breakdown (host sets FABRIC_TEST_PERF_BREAKDOWN when TT_FABRIC_TEST_PERF_BREAKDOWN=1).
// Eight u64 slots written at word PERF_BREAKDOWN_BASE_WORD_INDEX of the result buffer; slot meanings
// are defined in tt_fabric_test_perf_breakdown.hpp. Compiles to nothing when the define is absent.
static constexpr uint32_t PERF_BREAKDOWN_BASE_WORD_INDEX = 16;
static constexpr uint32_t PERF_BREAKDOWN_NUM_SLOTS = 8;

#ifdef FABRIC_TEST_PERF_BREAKDOWN
// Deltas use the low 32 bits of the wall clock: one register read, and unsigned subtraction stays correct
// across a wrap. Any single measured interval must stay under 2^32 cycles (~3 s at 1.35 GHz).
inline uint64_t perf_breakdown_slots[PERF_BREAKDOWN_NUM_SLOTS] = {};
#define PERF_BD_NOW() get_timestamp_32b()
#define PERF_BD_ADD_SINCE(slot, t0) (perf_breakdown_slots[(slot)] += static_cast<uint32_t>(get_timestamp_32b() - (t0)))
#define PERF_BD_SET_SINCE(slot, t0) (perf_breakdown_slots[(slot)] = static_cast<uint32_t>(get_timestamp_32b() - (t0)))
#define PERF_BD_INC(slot) (perf_breakdown_slots[(slot)]++)
// Full 64-bit variants for whole-loop spans that can exceed 2^32 cycles.
#define PERF_BD_NOW64() get_timestamp()
#define PERF_BD_SET_SINCE64(slot, t0) (perf_breakdown_slots[(slot)] = get_timestamp() - (t0))

inline void write_perf_breakdown(uint32_t result_buffer_base) {
    auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base);
    for (uint32_t i = 0; i < PERF_BREAKDOWN_NUM_SLOTS; i++) {
        result_buffer[PERF_BREAKDOWN_BASE_WORD_INDEX + 2 * i] = static_cast<uint32_t>(perf_breakdown_slots[i]);
        result_buffer[PERF_BREAKDOWN_BASE_WORD_INDEX + 2 * i + 1] = static_cast<uint32_t>(perf_breakdown_slots[i] >> 32);
    }
}
#else
#define PERF_BD_NOW() uint64_t{0}
#define PERF_BD_ADD_SINCE(slot, t0) ((void)(t0))
#define PERF_BD_SET_SINCE(slot, t0) ((void)(t0))
#define PERF_BD_INC(slot) ((void)0)
#define PERF_BD_NOW64() uint64_t{0}
#define PERF_BD_SET_SINCE64(slot, t0) ((void)(t0))

inline void write_perf_breakdown(uint32_t) {}
#endif

// FABRIC_TEST_PERF_SEND_DETAIL (with FABRIC_TEST_PERF_BREAKDOWN) times each step of the sender's per-packet path in
// slots 0, 5, 6 and 7, which otherwise hold setup, credit-blocked count, close and mux termination.
#if defined(FABRIC_TEST_PERF_BREAKDOWN) && defined(FABRIC_TEST_PERF_SEND_DETAIL)
#define PERF_BD_DETAIL_NOW() PERF_BD_NOW()
#define PERF_BD_DETAIL_ADD_SINCE(slot, t0) PERF_BD_ADD_SINCE(slot, t0)
#define PERF_BD_SUMMARY_SET_SINCE(slot, t0) ((void)(t0))
#define PERF_BD_SUMMARY_INC(slot) ((void)0)
#else
#define PERF_BD_DETAIL_NOW() uint64_t{0}
#define PERF_BD_DETAIL_ADD_SINCE(slot, t0) ((void)(t0))
#define PERF_BD_SUMMARY_SET_SINCE(slot, t0) PERF_BD_SET_SINCE(slot, t0)
#define PERF_BD_SUMMARY_INC(slot) PERF_BD_INC(slot)
#endif

// FABRIC_TEST_PERF_SCHED (with FABRIC_TEST_PERF_BREAKDOWN): the hot-state sender records how its send loop spent its
// time and when each config sent its last packet, from word PERF_SCHED_BASE_WORD_INDEX of the result buffer:
// PERF_SCHED_NUM_COUNTERS u64 counters (SenderSchedCounter), then per config its finish cycle (from loop start) and
// its credit misses. Layout must match tt_fabric_test_perf_breakdown.hpp. PERF_SCHED(...) compiles its statements
// only in this mode.
static constexpr uint32_t PERF_SCHED_BASE_WORD_INDEX = 512;
static constexpr uint32_t PERF_SCHED_NUM_COUNTERS = 8;
static constexpr uint32_t PERF_SCHED_MAX_CONFIGS = 160;

enum SenderSchedCounter : uint32_t {
    SC_PACKETS = 0,        // packets sent
    SC_IDLE_CYCLES = 1,    // loop passes that sent nothing, plus waits on one connection in the rotation loop
    SC_RELOADS = 2,        // free-slot re-reads of a connection
    SC_RELOADS_FULL = 3,   // re-reads that found the connection full
    SC_CREDIT_MISSES = 4,  // send attempts without receiver credits
    SC_PASSES = 5,         // passes over all configs / connections
    SC_IDLE_PASSES = 6,    // passes that sent nothing
    SC_BUDGET_SUM = 7,     // free slots found by the re-reads
};

#if defined(FABRIC_TEST_PERF_BREAKDOWN) && defined(FABRIC_TEST_PERF_SCHED)
#define PERF_SCHED(...) __VA_ARGS__

inline void write_sender_sched(
    uint32_t result_buffer_base,
    const uint64_t* counters,
    const uint32_t* finish_cycles,
    const uint32_t* credit_misses,
    uint32_t num_configs) {
    auto* words = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base) + PERF_SCHED_BASE_WORD_INDEX;
    for (uint32_t i = 0; i < PERF_SCHED_NUM_COUNTERS; i++) {
        words[2 * i] = static_cast<uint32_t>(counters[i]);
        words[2 * i + 1] = static_cast<uint32_t>(counters[i] >> 32);
    }
    words += 2 * PERF_SCHED_NUM_COUNTERS;
    for (uint32_t i = 0; i < num_configs && i < PERF_SCHED_MAX_CONFIGS; i++) {
        words[i] = finish_cycles[i];
        words[PERF_SCHED_MAX_CONFIGS + i] = credit_misses[i];
    }
}
#else
#define PERF_SCHED(...)
#endif

// FABRIC_TEST_PERF_HOT_DETAIL=<period> (with FABRIC_TEST_PERF_BREAKDOWN): the hot-state sender times the steps of one
// send in every <period> (a power of two), one timestamp per step boundary, and the sends it does not sample run
// untimed (the summary's per-packet fill / slot wait / flush timers are off). Recorded from word
// PERF_HOT_DETAIL_BASE_WORD_INDEX of the result buffer as PERF_HOT_NUM_STEPS u64 (SenderHotStep), cycles except the
// counts. Layout must match tt_fabric_test_perf_breakdown.hpp. PERF_HOT(...) compiles its statements only in this mode.
static constexpr uint32_t PERF_HOT_DETAIL_BASE_WORD_INDEX = 864;
static constexpr uint32_t PERF_HOT_CALIBRATION_STEPS = 64;

enum SenderHotStep : uint32_t {
    HS_SAMPLES = 0,      // count: sampled sends that sent a packet
    HS_CALIBRATION = 1,  // PERF_HOT_CALIBRATION_STEPS empty steps (the cost of one step boundary)
    HS_GAPS = 2,         // count: sampled sends followed by another send
    HS_GAP = 3,          // send loop between a sampled send's return and the next send
    HS_CREDIT = 4,       // receiver credit check
    HS_SLOT = 5,         // free-slot poll of the fabric/mux connection, including any wait
    HS_FILL = 6,         // SFPU page wait (or RISC pattern fill)
    HS_PAYLOAD = 7,      // payload NOC write
    HS_HEADER = 8,       // header NOC write + slot handoff to the fabric/mux
    HS_FLUSH = 9,        // noc_async_writes_flushed
    HS_POST = 10,        // page pop, header advance, seed, counters
    HS_CALLS = 11,       // count: all hot sends, sampled or not
    PERF_HOT_NUM_STEPS = 12,
};

#if defined(FABRIC_TEST_PERF_BREAKDOWN) && defined(FABRIC_TEST_PERF_HOT_DETAIL)
#define PERF_HOT(...) __VA_ARGS__
static_assert((FABRIC_TEST_PERF_HOT_DETAIL & (FABRIC_TEST_PERF_HOT_DETAIL - 1)) == 0);

inline uint64_t perf_hot_steps[PERF_HOT_NUM_STEPS] = {};
inline uint32_t perf_hot_calls = 0;
inline uint32_t perf_hot_t = 0;
inline bool perf_hot_sampled = false;

// Ends the current step of a sampled send.
FORCE_INLINE void perf_hot_step(SenderHotStep step) {
    if (perf_hot_sampled) {
        const uint32_t now = get_timestamp_32b();
        perf_hot_steps[step] += now - perf_hot_t;
        perf_hot_t = now;
    }
}

// Starts a send: ends the loop gap after a sampled send and decides whether this send is sampled.
FORCE_INLINE void perf_hot_begin() {
    const bool after_sample = perf_hot_sampled;
    perf_hot_sampled = (perf_hot_calls++ & (FABRIC_TEST_PERF_HOT_DETAIL - 1)) == 0;
    if (after_sample || perf_hot_sampled) {
        const uint32_t now = get_timestamp_32b();
        if (after_sample) {
            perf_hot_steps[HS_GAP] += now - perf_hot_t;
            perf_hot_steps[HS_GAPS]++;
        }
        perf_hot_t = now;
    }
}

// Ends a send that sent its packet.
FORCE_INLINE void perf_hot_end() {
    perf_hot_step(HS_POST);
    perf_hot_steps[HS_SAMPLES] += perf_hot_sampled;
}

inline void perf_hot_calibrate() {
    perf_hot_sampled = true;
    perf_hot_t = get_timestamp_32b();
    for (uint32_t i = 0; i < PERF_HOT_CALIBRATION_STEPS; i++) {
        perf_hot_step(HS_CALIBRATION);
    }
    perf_hot_sampled = false;
}

inline void write_perf_hot_detail(uint32_t result_buffer_base) {
    perf_hot_steps[HS_CALLS] = perf_hot_calls;
    auto* words = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base) + PERF_HOT_DETAIL_BASE_WORD_INDEX;
    for (uint32_t i = 0; i < PERF_HOT_NUM_STEPS; i++) {
        words[2 * i] = static_cast<uint32_t>(perf_hot_steps[i]);
        words[2 * i + 1] = static_cast<uint32_t>(perf_hot_steps[i] >> 32);
    }
}

// The hot sender's per-packet summary timers, off while the steps are sampled.
#define PERF_BD_HOT_NOW() uint64_t{0}
#define PERF_BD_HOT_ADD_SINCE(slot, t0) ((void)(t0))
#else
#define PERF_HOT(...)
#define PERF_BD_HOT_NOW() PERF_BD_NOW()
#define PERF_BD_HOT_ADD_SINCE(slot, t0) PERF_BD_ADD_SINCE(slot, t0)
#endif

// Diagnostics: whether the router flushes its receive channel's outstanding NOC writes before applying the atomic
// increments the infra sends, for the atomic_inc ntype's packets (FABRIC_TEST_ATOMIC_INC_NO_FLUSH turns it off) and
// for receiver credit returns (FABRIC_TEST_CREDIT_RETURN_NO_FLUSH). Both default to the fabric header's flush.
#ifdef FABRIC_TEST_ATOMIC_INC_NO_FLUSH
constexpr bool ATOMIC_INC_FLUSH = false;
#else
constexpr bool ATOMIC_INC_FLUSH = true;
#endif
#ifdef FABRIC_TEST_CREDIT_RETURN_NO_FLUSH
constexpr bool CREDIT_RETURN_FLUSH = false;
#else
constexpr bool CREDIT_RETURN_FLUSH = true;
#endif

// Per-config result entry for per-endpoint progress tracking.
// Layout must match the host-side definition in tt_fabric_test_memory_map.hpp exactly.
struct PerConfigResult {
    uint32_t packets_low;
    uint32_t packets_high;
};

// Word index in the result buffer where per-config results begin
static constexpr uint32_t PER_CONFIG_RESULT_BASE_WORD_INDEX = 32;

inline tt_l1_ptr PerConfigResult* get_per_config_results(uint32_t result_buffer_base) {
    return reinterpret_cast<tt_l1_ptr PerConfigResult*>(
        result_buffer_base + PER_CONFIG_RESULT_BASE_WORD_INDEX * sizeof(uint32_t));
}

inline void write_per_config_result(tt_l1_ptr PerConfigResult* entry, uint64_t packets) {
    entry->packets_low = static_cast<uint32_t>(packets);
    entry->packets_high = static_cast<uint32_t>(packets >> 32);
}

struct SequentialDataPattern {
    static constexpr uint32_t WORD_SIZE = sizeof(uint32_t);

    static void fill(uint32_t buffer_address, uint32_t payload_size, uint32_t start_value) {
        auto* buffer_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(buffer_address);
        uint32_t num_words = payload_size / WORD_SIZE;
        uint32_t i = 0;
#ifdef FABRIC_TEST_FAST_PATTERN
        // Unrolled so each word costs one store plus one add, instead of also paying the pointer
        // increment and loop branch per word.
        for (; i + 8 <= num_words; i += 8) {
            const uint32_t v = start_value + i;
            buffer_ptr[i + 0] = v + 0;
            buffer_ptr[i + 1] = v + 1;
            buffer_ptr[i + 2] = v + 2;
            buffer_ptr[i + 3] = v + 3;
            buffer_ptr[i + 4] = v + 4;
            buffer_ptr[i + 5] = v + 5;
            buffer_ptr[i + 6] = v + 6;
            buffer_ptr[i + 7] = v + 7;
        }
#endif
        for (; i < num_words; i++) {
            buffer_ptr[i] = start_value + i;
        }
    }

    static bool poll(uint32_t buffer_address, uint32_t payload_size, uint32_t start_value) {
        auto* buffer_ptr =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(buffer_address + payload_size - WORD_SIZE);
        uint32_t expected_value = start_value + payload_size / WORD_SIZE - 1;
        return *buffer_ptr == expected_value;
    }

    static bool validate(uint32_t buffer_address, uint32_t payload_size, uint32_t start_value) {
        auto* buffer_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(buffer_address);
        uint32_t num_words = payload_size / WORD_SIZE;
        uint32_t i = 0;
#ifdef FABRIC_TEST_FAST_PATTERN
        // Issue eight independent loads before using any of them so their L1 latency overlaps, and
        // branch once per block on the OR of the differences. Checks exactly the same words.
        for (; i + 8 <= num_words; i += 8) {
            const uint32_t w0 = buffer_ptr[i + 0];
            const uint32_t w1 = buffer_ptr[i + 1];
            const uint32_t w2 = buffer_ptr[i + 2];
            const uint32_t w3 = buffer_ptr[i + 3];
            const uint32_t w4 = buffer_ptr[i + 4];
            const uint32_t w5 = buffer_ptr[i + 5];
            const uint32_t w6 = buffer_ptr[i + 6];
            const uint32_t w7 = buffer_ptr[i + 7];
            const uint32_t v = start_value + i;
            const uint32_t diff = (w0 ^ (v + 0)) | (w1 ^ (v + 1)) | (w2 ^ (v + 2)) | (w3 ^ (v + 3)) |
                                  (w4 ^ (v + 4)) | (w5 ^ (v + 5)) | (w6 ^ (v + 6)) | (w7 ^ (v + 7));
            if (diff != 0) {
                return false;
            }
        }
#endif
        for (; i < num_words; i++) {
            if (buffer_ptr[i] != (start_value + i)) {
                return false;
            }
        }
        return true;
    }
};

class StreamingBuffer {
public:
    StreamingBuffer(uint32_t base_address, uint32_t total_size, uint32_t payload_size) :
        base_address_(base_address), total_size_(total_size), payload_size_(payload_size) {
        ASSERT(total_size > 0);
        ASSERT(payload_size > 0);
        ASSERT(payload_size <= total_size);
        reset();
    }

    uint32_t get_current_offset() const { return current_offset_; }
    uint32_t get_total_size() const { return total_size_; }
    constexpr bool has_wrapped() const { return has_wrapped_; }

    void advance() {
        current_offset_ += payload_size_;
        // need to check if we have enough space in the buffer for another payload without wrapping
        if (current_offset_ + payload_size_ > total_size_) {
            current_offset_ = 0;
            has_wrapped_ = true;
        }
    }

    void reset() {
        current_offset_ = 0;
        has_wrapped_ = false;
    }

protected:
    uint32_t base_address_;
    uint32_t total_size_;
    uint32_t payload_size_;
    uint32_t current_offset_;
    bool has_wrapped_ = false;
};

class SenderPayloadBuffer : public StreamingBuffer {
public:
    SenderPayloadBuffer(uint32_t physical_base_address, uint32_t virtual_total_size, uint32_t payload_size) :
        StreamingBuffer(physical_base_address, virtual_total_size, payload_size) {}

    uint32_t get_physical_address() const { return base_address_; }

    void fill_data(uint32_t start_value) {
        SequentialDataPattern::fill(get_physical_address(), this->payload_size_, start_value);
    }
};

class ReceiverPayloadBuffer : public StreamingBuffer {
public:
    ReceiverPayloadBuffer(uint32_t base_address, uint32_t total_size, uint32_t payload_size) :
        StreamingBuffer(base_address, total_size, payload_size) {}

    uint32_t get_physical_address() const { return base_address_ + this->get_current_offset(); }

    bool poll_for_data(uint32_t start_value) {
        return SequentialDataPattern::poll(get_physical_address(), this->payload_size_, start_value);
    }

    bool validate_data(uint32_t start_value) const {
        return SequentialDataPattern::validate(get_physical_address(), this->payload_size_, start_value);
    }
};

struct SenderTrafficConfigMetadata {
    static SenderTrafficConfigMetadata build_from_args(size_t& arg_idx) { return SenderTrafficConfigMetadata(arg_idx); }

    SenderTrafficConfigMetadata(const SenderTrafficConfigMetadata& other) :
        num_packets(other.num_packets),
        seed(other.seed),
        payload_buffer_size(other.payload_buffer_size),
        sfpu_payload(other.sfpu_payload) {}

    uint32_t num_packets = 0;
    uint32_t seed = 0;
    uint32_t payload_buffer_size = 0;
    // The config's payloads come from the core's compute engine (tt_fabric_test_payload_gen.cpp), as pages of the
    // SFPU_PAYLOAD_CB_INDEX CB in strict round-robin order over such configs, instead of being stored by this kernel.
    bool sfpu_payload = false;

private:
    SenderTrafficConfigMetadata(size_t& arg_idx) {
        this->num_packets = get_local_arg_val<uint32_t>(arg_idx++);
        this->seed = get_local_arg_val<uint32_t>(arg_idx++);
        this->payload_buffer_size = get_local_arg_val<uint32_t>(arg_idx++);
        this->sfpu_payload = get_local_arg_val<uint32_t>(arg_idx++) != 0;
    }
};

struct ChipUnicastFields1D {
    static ChipUnicastFields1D build_from_args(size_t& arg_idx) {
        uint32_t num_hops = get_local_arg_val<uint32_t>(arg_idx++);
        return ChipUnicastFields1D(num_hops);
    }

    ChipUnicastFields1D(uint32_t num_hops) : num_hops(num_hops) {}

    uint32_t num_hops;
};

struct ChipUnicastFields2D {
    static ChipUnicastFields2D build_from_args(size_t& arg_idx) {
        uint16_t dst_device_id = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t dst_mesh_id = get_local_arg_val<uint32_t>(arg_idx++);
        return ChipUnicastFields2D(dst_device_id, dst_mesh_id);
    }

    ChipUnicastFields2D(uint16_t dst_device_id, uint16_t dst_mesh_id) :
        dst_device_id(dst_device_id), dst_mesh_id(dst_mesh_id) {}

    uint16_t dst_device_id;
    uint16_t dst_mesh_id;
};

struct ChipMulticastFields1D {
    static ChipMulticastFields1D build_from_args(size_t& arg_idx) {
        uint32_t mcast_start_hops = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t num_hops = get_local_arg_val<uint32_t>(arg_idx++);
        return ChipMulticastFields1D(mcast_start_hops, num_hops);
    }

    ChipMulticastFields1D(uint32_t mcast_start_hops, uint32_t num_hops) :
        mcast_start_hops(mcast_start_hops), num_hops(num_hops) {}

    uint32_t mcast_start_hops;
    uint32_t num_hops;
};

struct ChipMulticastFields2D {
    static ChipMulticastFields2D build_from_args(size_t& arg_idx) {
        uint16_t dst_device_id = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t dst_mesh_id = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_n = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_s = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_e = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_w = get_local_arg_val<uint32_t>(arg_idx++);
        return ChipMulticastFields2D(dst_device_id, dst_mesh_id, num_hops_n, num_hops_s, num_hops_e, num_hops_w);
    }

    ChipMulticastFields2D(
        uint16_t dst_device_id,
        uint16_t dst_mesh_id,
        uint16_t num_hops_n,
        uint16_t num_hops_s,
        uint16_t num_hops_e,
        uint16_t num_hops_w) :
        dst_device_id(dst_device_id),
        dst_mesh_id(dst_mesh_id),
        num_hops_n(num_hops_n),
        num_hops_s(num_hops_s),
        num_hops_e(num_hops_e),
        num_hops_w(num_hops_w) {}

    uint16_t dst_device_id;
    uint16_t dst_mesh_id;
    uint16_t num_hops_n;
    uint16_t num_hops_s;
    uint16_t num_hops_e;
    uint16_t num_hops_w;
};

struct NocUnicastWriteFields {
    template <bool IS_SOURCE>
    static NocUnicastWriteFields build_from_args(size_t& arg_idx) {
        uint32_t payload_size_bytes = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_address = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_noc_encoding = 0;
        if constexpr (IS_SOURCE) {
            dst_noc_encoding = get_local_arg_val<uint32_t>(arg_idx++);
        }
        return NocUnicastWriteFields(payload_size_bytes, dst_address, dst_noc_encoding);
    }

    NocUnicastWriteFields(uint32_t payload_size_bytes, uint32_t dst_address, uint32_t dst_noc_encoding) :
        payload_size_bytes(payload_size_bytes), dst_address(dst_address), dst_noc_encoding(dst_noc_encoding) {}

    uint32_t payload_size_bytes;
    uint32_t dst_address;
    uint32_t dst_noc_encoding;
};

struct NocUnicastAtomicIncFields {
    template <bool IS_SOURCE>
    static NocUnicastAtomicIncFields build_from_args(size_t& arg_idx) {
        uint32_t atomic_inc_val = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_address = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_noc_encoding = 0;
        if constexpr (IS_SOURCE) {
            dst_noc_encoding = get_local_arg_val<uint32_t>(arg_idx++);
        }
        return NocUnicastAtomicIncFields(atomic_inc_val, dst_address, dst_noc_encoding);
    }

    NocUnicastAtomicIncFields(uint32_t atomic_inc_val, uint32_t dst_address, uint32_t dst_noc_encoding) :
        atomic_inc_val(atomic_inc_val), dst_address(dst_address), dst_noc_encoding(dst_noc_encoding) {}

    uint32_t atomic_inc_val;
    uint32_t dst_address;
    uint32_t dst_noc_encoding;
};

struct NocUnicastWriteAtomicIncFields {
    template <bool IS_SOURCE>
    static NocUnicastWriteAtomicIncFields build_from_args(size_t& arg_idx) {
        const auto write_fields = NocUnicastWriteFields::build_from_args<IS_SOURCE>(arg_idx);
        const auto atomic_inc_fields = NocUnicastAtomicIncFields::build_from_args<IS_SOURCE>(arg_idx);
        auto fields = NocUnicastWriteAtomicIncFields(write_fields, atomic_inc_fields);
        if constexpr (IS_SOURCE) {
            fields.flush = get_local_arg_val<uint32_t>(arg_idx++) != 0;
        }
        return fields;
    }

    NocUnicastWriteAtomicIncFields(NocUnicastWriteFields write_fields, NocUnicastAtomicIncFields atomic_inc_fields) :
        write_fields(write_fields), atomic_inc_fields(atomic_inc_fields) {}

    NocUnicastWriteFields write_fields;
    NocUnicastAtomicIncFields atomic_inc_fields;
    bool flush = false;  // Flush the write before the increment (sender only)
};

struct NocUnicastScatterWriteFields {
    static constexpr uint32_t MAX_CHUNKS = NOC_SCATTER_WRITE_MAX_CHUNKS;

    template <bool IS_SOURCE>
    static NocUnicastScatterWriteFields build_from_args(size_t& arg_idx) {
        uint32_t payload_size_bytes = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t chunk_count = get_local_arg_val<uint32_t>(arg_idx++);
        ASSERT(chunk_count >= NOC_SCATTER_WRITE_MIN_CHUNKS && chunk_count <= MAX_CHUNKS);

        std::array<uint32_t, MAX_CHUNKS> dst_addresses{};
        for (uint32_t i = 0; i < chunk_count; i++) {
            dst_addresses[i] = get_local_arg_val<uint32_t>(arg_idx++);
        }

        uint32_t dst_noc_encoding = 0;
        if constexpr (IS_SOURCE) {
            dst_noc_encoding = get_local_arg_val<uint32_t>(arg_idx++);
        }

        std::array<uint16_t, MAX_CHUNKS - 1> chunk_sizes{};
        for (uint32_t i = 0; i < (chunk_count - 1); i++) {
            chunk_sizes[i] = static_cast<uint16_t>(get_local_arg_val<uint32_t>(arg_idx++));
        }

        auto fields = NocUnicastScatterWriteFields(
            payload_size_bytes, static_cast<uint8_t>(chunk_count), dst_addresses, chunk_sizes, dst_noc_encoding);

        // Optional trailing semaphore increment (ntype option atomic_inc)
        fields.has_atomic_inc = get_local_arg_val<uint32_t>(arg_idx++) != 0;
        if (fields.has_atomic_inc) {
            const auto atomic_inc_fields = NocUnicastAtomicIncFields::build_from_args<IS_SOURCE>(arg_idx);
            ASSERT(!IS_SOURCE || atomic_inc_fields.dst_noc_encoding == dst_noc_encoding);
            fields.atomic_inc_val = atomic_inc_fields.atomic_inc_val;
            fields.atomic_inc_dst_address = atomic_inc_fields.dst_address;
            if constexpr (IS_SOURCE) {
                fields.atomic_inc_flush = get_local_arg_val<uint32_t>(arg_idx++) != 0;
            }
        }
        return fields;
    }

    NocUnicastScatterWriteFields(
        uint32_t payload_size_bytes,
        uint8_t chunk_count,
        const std::array<uint32_t, MAX_CHUNKS>& dst_addresses,
        const std::array<uint16_t, MAX_CHUNKS - 1>& chunk_sizes,
        uint32_t dst_noc_encoding) :
        payload_size_bytes(payload_size_bytes),
        chunk_count(chunk_count),
        dst_addresses(dst_addresses),
        chunk_sizes(chunk_sizes),
        dst_noc_encoding(dst_noc_encoding) {}

    uint32_t payload_size_bytes;
    uint8_t chunk_count;
    std::array<uint32_t, MAX_CHUNKS> dst_addresses;
    std::array<uint16_t, MAX_CHUNKS - 1> chunk_sizes;
    uint32_t dst_noc_encoding;
    bool has_atomic_inc = false;
    uint32_t atomic_inc_val = 0;
    uint32_t atomic_inc_dst_address = 0;
    bool atomic_inc_flush = false;  // Flush the scatter writes before the increment (sender only)
};

template <typename T>
void setup_2d_mcast_route(uint32_t packet_header_address, const ChipMulticastFields2D& mcast_fields) {
    (void)fabric_set_2d_mcast_route(
        (T*)packet_header_address,
        mcast_fields.dst_device_id,
        mcast_fields.dst_mesh_id,
        mcast_fields.num_hops_e,
        mcast_fields.num_hops_w,
        mcast_fields.num_hops_n,
        mcast_fields.num_hops_s);
}

/**
 * Template-based dispatch system for chip send type handling.
 * Specialized for:
 * - 1D vs 2D fabric routing
 * - Unicast vs multicast transmission
 */
template <ChipSendType chip_type, bool IS_2D_FABRIC>
struct ChipSendTypeHandler {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header);
};

// 1D Unicast specialization
template <>
struct ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, false> {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
        const auto unicast_fields = ChipUnicastFields1D::build_from_args(arg_idx);
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)packet_header, unicast_fields.num_hops);
    }
};

// 2D Unicast specialization
template <>
struct ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, true> {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
        const auto unicast_fields = ChipUnicastFields2D::build_from_args(arg_idx);
        fabric_set_unicast_route(
            (HybridMeshPacketHeader*)packet_header_address, unicast_fields.dst_device_id, unicast_fields.dst_mesh_id);
    }
};

// 1D Multicast specialization
template <>
struct ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, false> {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
        const auto mcast_fields = ChipMulticastFields1D::build_from_args(arg_idx);
        packet_header->to_chip_multicast(MulticastRoutingCommandHeader{
            static_cast<uint8_t>(mcast_fields.mcast_start_hops), static_cast<uint8_t>(mcast_fields.num_hops)});
    }
};

// 2D Multicast specialization
template <>
struct ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, true> {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
        const auto mcast_fields = ChipMulticastFields2D::build_from_args(arg_idx);
        setup_2d_mcast_route<HybridMeshPacketHeader>(packet_header_address, mcast_fields);
    }
};

template <typename EdmSenderT = WorkerToFabricEdmSender>
struct SenderKernelTrafficConfig;

// NOC op structs and function pointer types are all templated on EdmSenderT.
// VC_ID is a compile-time arg so each binary contains exactly one EdmSenderT
// instantiation — function pointer types match call sites exactly, no casting needed.
template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocOperationTypes {
    using ParseSetupFunc = void (*)(SenderKernelTrafficConfig<EdmSenderT>*, size_t&);
    using UpdateHeaderFunc = void (*)(SenderKernelTrafficConfig<EdmSenderT>*);

    struct Operations {
        ParseSetupFunc parse_and_setup;
        UpdateHeaderFunc update_header;
    };
};

// NOC Operation Class Declarations (implementations after SenderKernelTrafficConfig)
template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocWriteSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config);
};

template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocAtomicSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config);
};

template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocFusedSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config);
};

template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocScatterWriteSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config);
    static void write_header(SenderKernelTrafficConfig<EdmSenderT>* config, uint32_t buffer_offset);
};

/* ****************************************************************************
 * MuxCachedInfo
 * *****************************************************************************/
struct MuxCachedInfo {
    uint8_t mux_x = 0;
    uint8_t mux_y = 0;
    size_t mux_status_address = 0;
    size_t local_mux_status_address = 0;
};

/* ****************************************************************************
 * FabricConnectionArray: Unified connection management for kernel
 *
 * Provides type-erased storage for both WorkerToFabricEdmSender and
 * WorkerToFabricMuxSender connections with runtime dispatch.
 * *****************************************************************************/
template <typename EdmSenderT = WorkerToFabricEdmSender>
struct FabricConnectionArray {
    // TODO: get the num buffers more systematically
    static constexpr uint8_t NUM_BUFFERS = 8;

    using MuxConnectionType = tt::tt_fabric::WorkerToFabricMuxSender<NUM_BUFFERS>;
    static constexpr size_t MAX_CONNECTION_SIZE = std::max(sizeof(EdmSenderT), sizeof(MuxConnectionType));

    // Type-erased storage for connections (sized for maximum).
    alignas(std::max(alignof(EdmSenderT), alignof(MuxConnectionType)))
        std::array<char, MAX_NUM_FABRIC_CONNECTIONS * MAX_CONNECTION_SIZE> storage;
    std::array<bool, MAX_NUM_FABRIC_CONNECTIONS> is_mux;

    // Cached mux info for wait_for_fabric_endpoint_ready
    std::array<MuxCachedInfo, MAX_NUM_FABRIC_CONNECTIONS> mux_cached_info;

    // Actual number of connections in use (set at initialization, bounds-checked in kernel)
    uint8_t num_connections = 0;

    // Accessors with proper type casting
    FORCE_INLINE EdmSenderT& get_fabric_connection(uint8_t idx) {
        return *reinterpret_cast<EdmSenderT*>(storage.data() + idx * MAX_CONNECTION_SIZE);
    }

    FORCE_INLINE MuxConnectionType& get_mux_connection(uint8_t idx) {
        return *reinterpret_cast<MuxConnectionType*>(storage.data() + idx * MAX_CONNECTION_SIZE);
    }

    // Parse connections from runtime args
    // Memory map is required for allocating local semaphore addresses for mux connections
    template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX, typename MemoryMapType>
    void parse_from_args(size_t& rt_args_idx, MemoryMapType& memory_map) {
        for (uint8_t i = 0; i < num_connections; i++) {
            // Parse connection type flag
            is_mux[i] = get_arg_val<uint32_t>(rt_args_idx++) != 0;

            if (is_mux[i]) {
                // Initialize mux connection using placement new
                mux_cached_info[i].mux_x = get_arg_val<uint32_t>(rt_args_idx++);
                mux_cached_info[i].mux_y = get_arg_val<uint32_t>(rt_args_idx++);
                uint8_t worker_stream_id = get_arg_val<uint32_t>(rt_args_idx++);
                uint8_t mux_num_buffers_per_channel = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_channel_buffer_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_channel_base_address = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_connection_info_address = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_connection_handshake_address = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_flow_control_address = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_buffer_index_address = get_arg_val<uint32_t>(rt_args_idx++);
                mux_cached_info[i].mux_status_address = get_arg_val<uint32_t>(rt_args_idx++);

                // Allocate local semaphore addresses for this mux connection (cursor-based)
                const auto mux_local_addrs = memory_map.get_mux_local_addresses_for_connection();
                mux_cached_info[i].local_mux_status_address = mux_local_addrs.status_buffer_address;

                auto conn = build_connection_to_fabric_endpoint<NUM_BUFFERS>(
                    mux_cached_info[i].mux_x,
                    mux_cached_info[i].mux_y,
                    worker_stream_id,
                    mux_num_buffers_per_channel,
                    mux_channel_buffer_size_bytes,
                    mux_channel_base_address,
                    mux_connection_info_address,
                    mux_connection_handshake_address,
                    mux_flow_control_address,
                    mux_buffer_index_address,
                    mux_local_addrs.flow_control_address,
                    mux_local_addrs.teardown_address,
                    mux_local_addrs.buffer_index_address);
                new (&get_mux_connection(i)) MuxConnectionType(conn);
            } else {
                // Initialize fabric connection using placement new
                auto conn = EdmSenderT::template build_from_args<core_type>(rt_args_idx);
                new (&get_fabric_connection(i)) EdmSenderT(conn);
            }
        }
    }

    // Lifecycle management
    FORCE_INLINE void open_all() {
        for (uint8_t i = 0; i < num_connections; i++) {
            if (is_mux[i]) {
                // Wait for mux to be ready before connecting
                const auto& info = mux_cached_info[i];
                tt::tt_fabric::wait_for_fabric_endpoint_ready(
                    info.mux_x, info.mux_y, info.mux_status_address, info.local_mux_status_address);
                get_mux_connection(i).open();
            } else {
                get_fabric_connection(i).open();
            }
        }
    }

    FORCE_INLINE void close_all() {
        for (uint8_t i = 0; i < num_connections; i++) {
            if (is_mux[i]) {
                get_mux_connection(i).close();
            } else {
                get_fabric_connection(i).close();
            }
        }
    }

    // Unified send operations (dispatch hidden from callers)

    // Wait for connection to have space
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void wait_for_empty_write_slot(void* conn_ptr, uint8_t idx) {
        if constexpr (BENCHMARK_MODE) {
            // Fast path: no runtime check, direct cast
            static_cast<EdmSenderT*>(conn_ptr)->wait_for_empty_write_slot();
        } else {
            // Normal path: runtime dispatch using cached is_mux array
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->wait_for_empty_write_slot();
            } else {
                static_cast<EdmSenderT*>(conn_ptr)->wait_for_empty_write_slot();
            }
        }
    }

    // Send header only (used for credit returns)
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_header_non_blocking(void* conn_ptr, uint8_t idx, uint32_t header_addr) {
        if constexpr (BENCHMARK_MODE) {
            static_cast<EdmSenderT*>(conn_ptr)->send_payload_flush_non_blocking_from_address(
                header_addr, sizeof(PACKET_HEADER_TYPE));
        } else {
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->send_payload_flush_non_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            } else {
                static_cast<EdmSenderT*>(conn_ptr)->send_payload_flush_non_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }

    // Send payload without header (used for multi-part sends)
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_payload_without_header(void* conn_ptr, uint8_t idx, uint32_t payload_addr, size_t size) {
        if constexpr (BENCHMARK_MODE) {
            static_cast<EdmSenderT*>(conn_ptr)->send_payload_without_header_non_blocking_from_address(
                payload_addr, size);
        } else {
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->send_payload_without_header_non_blocking_from_address(
                    payload_addr, size);
            } else {
                static_cast<EdmSenderT*>(conn_ptr)->send_payload_without_header_non_blocking_from_address(
                    payload_addr, size);
            }
        }
    }

    // Send header with flush (used for completing multi-part sends)
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_header_flush_blocking(void* conn_ptr, uint8_t idx, uint32_t header_addr) {
        if constexpr (BENCHMARK_MODE) {
            static_cast<EdmSenderT*>(conn_ptr)->send_payload_flush_blocking_from_address(
                header_addr, sizeof(PACKET_HEADER_TYPE));
        } else {
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->send_payload_flush_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            } else {
                static_cast<EdmSenderT*>(conn_ptr)->send_payload_flush_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }

    // Combined: send payload + header
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_payload_with_header(
        void* conn_ptr, uint8_t idx, uint32_t payload_addr, size_t payload_size, uint32_t header_addr) {
        if constexpr (BENCHMARK_MODE) {
            auto* conn = static_cast<EdmSenderT*>(conn_ptr);
            if (payload_size > 0) {
                conn->send_payload_without_header_non_blocking_from_address(payload_addr, payload_size);
            }
            conn->send_payload_flush_non_blocking_from_address(header_addr, sizeof(PACKET_HEADER_TYPE));
        } else {
            if (is_mux[idx]) {
                auto* conn = static_cast<MuxConnectionType*>(conn_ptr);
                if (payload_size > 0) {
                    conn->send_payload_without_header_non_blocking_from_address(payload_addr, payload_size);
                }
                conn->send_payload_flush_non_blocking_from_address(header_addr, sizeof(PACKET_HEADER_TYPE));
            } else {
                auto* conn = static_cast<EdmSenderT*>(conn_ptr);
                if (payload_size > 0) {
                    conn->send_payload_without_header_non_blocking_from_address(payload_addr, payload_size);
                }
                conn->send_payload_flush_non_blocking_from_address(header_addr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }
};

// Maximum root outputs a 2D multicast config can inject into.
constexpr uint8_t MAX_MCAST_INJECTIONS = MAX_NUM_FABRIC_CONNECTIONS;

template <typename EdmSenderT = WorkerToFabricEdmSender>
struct LineSyncConfig {
    LineSyncConfig(
        FabricConnectionArray<EdmSenderT>* connection_array,
        const uint8_t* connection_indices,
        uint8_t num_connections,
        const uint32_t packet_header_address,
        const uint32_t line_sync_val) :
        connection_manager_(connection_array), num_connections_(num_connections), line_sync_val(line_sync_val) {
        packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);

        ASSERT(num_connections_ > 0 && num_connections_ <= MAX_MCAST_INJECTIONS);
        for (uint8_t i = 0; i < num_connections_; i++) {
            const uint8_t idx = connection_indices[i];
            connection_indices_[i] = idx;
            if (connection_manager_->is_mux[idx]) {
                connection_ptrs_[i] = &connection_manager_->get_mux_connection(idx);
            } else {
                connection_ptrs_[i] = &connection_manager_->get_fabric_connection(idx);
            }
        }
    }

    template <bool IS_2D_FABRIC, ChipSendType CHIP_SEND_TYPE>
    void setup_packet_header(size_t& arg_idx, uint32_t packet_header_address) {
        ChipSendTypeHandler<CHIP_SEND_TYPE, IS_2D_FABRIC>::parse_and_setup(
            arg_idx, packet_header_address, packet_header);

        auto fields = NocUnicastAtomicIncFields::build_from_args<true>(arg_idx);
        line_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fields.dst_address);

        uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
        packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{noc_addr, fields.atomic_inc_val});
    }

    void global_sync_start() {
        for (uint8_t i = 0; i < num_connections_; i++) {
            connection_manager_->template wait_for_empty_write_slot<false>(connection_ptrs_[i], connection_indices_[i]);
            connection_manager_->template send_header_non_blocking<false>(
                connection_ptrs_[i], connection_indices_[i], (uint32_t)packet_header);
        }
    }

    void global_sync_finish(uint8_t sync_iter) {
        // sync wait
        noc_semaphore_wait_min(line_sync_ptr, line_sync_val * (sync_iter + 1));
    }

private:
    FabricConnectionArray<EdmSenderT>* connection_manager_;

    // Every connection this sync config injects into. Length 1 unless this is an express multicast
    // root whose canonical action names more than one output edge.
    void* connection_ptrs_[MAX_MCAST_INJECTIONS];
    uint8_t connection_indices_[MAX_MCAST_INJECTIONS];
    uint8_t num_connections_;

    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header;
    volatile tt_l1_ptr uint32_t* line_sync_ptr;
    uint32_t line_sync_val;
};

template <bool IS_MASTER_CORE, uint8_t NUM_LOCAL_CORES>
struct LocalSyncConfig {
    LocalSyncConfig(const uint32_t sync_address, const uint32_t sync_val) :
        sync_address(sync_address), sync_val(sync_val) {
        sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_address);
    }

    void setup_core_coordinates(size_t& arg_idx) {
        // Get core coordinates from runtime args
        for (uint8_t i = 0; i < NUM_LOCAL_CORES; i++) {
            sync_core_xy_encoding_[i] = get_local_arg_val<uint32_t>(arg_idx++);
        }
    }

    void local_sync(uint8_t sync_iter) {
        if constexpr (IS_MASTER_CORE) {
            // Master core: signal all local cores
            for (uint8_t i = 0; i < NUM_LOCAL_CORES; i++) {
                auto dest_noc_addr = get_noc_addr_helper(sync_core_xy_encoding_[i], sync_address);
                noc_semaphore_inc(dest_noc_addr, 1);
            }
            // Wait for all local cores to acknowledge
            uint32_t expected_val = NUM_LOCAL_CORES * (sync_iter + 1);
            noc_semaphore_wait(sync_ptr, expected_val);
        } else {
            uint32_t expected_val = sync_iter + 1;
            noc_semaphore_wait(sync_ptr, expected_val);
            // send ack back to master sender
            auto master_sender_noc_addr = get_noc_addr_helper(sync_core_xy_encoding_[0], sync_address);
            noc_semaphore_inc(master_sender_noc_addr, 1);
        }
    }

private:
    std::array<uint32_t, NUM_LOCAL_CORES> sync_core_xy_encoding_;
    uint32_t sync_address;
    volatile tt_l1_ptr uint32_t* sync_ptr;
    uint32_t sync_val;
};

struct SenderCreditInfo {
    SenderCreditInfo() = default;

    static SenderCreditInfo build_from_args(size_t& arg_idx) { return SenderCreditInfo(arg_idx); }

    uint32_t expected_receiver_count = 0;
    uint32_t credit_reception_address_base = 0;  // Base address of credit chunk (for mcast)
    uint32_t initial_credits = 0;

private:
    SenderCreditInfo(size_t& arg_idx) {
        this->expected_receiver_count = get_local_arg_val<uint32_t>(arg_idx++);
        this->credit_reception_address_base = get_local_arg_val<uint32_t>(arg_idx++);
        this->initial_credits = get_local_arg_val<uint32_t>(arg_idx++);
    }
};

// Helper class to manage sender-side credit consumption
// Encapsulates all credit checking and consumption logic in one place
struct SenderCreditManager {
    SenderCreditManager() = default;

    // Initialize from args
    void init(size_t& arg_idx, uint32_t total_credits) {
        enabled_ = get_local_arg_val<uint32_t>(arg_idx++) != 0;
        if (!enabled_) {
            return;
        }

        sender_credit_info_ = SenderCreditInfo::build_from_args(arg_idx);
        credit_semaphores_base_ptr_ =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_credit_info_.credit_reception_address_base);
        credit_semaphores_base_noc_addr_ = get_noc_addr(0) + sender_credit_info_.credit_reception_address_base;

        num_receivers_ = sender_credit_info_.expected_receiver_count;
        initial_credits_ = sender_credit_info_.initial_credits;
        estimated_available_credits_ = initial_credits_;
        prev_processed_credits_ = 0;
        total_credits_ = total_credits;

        ASSERT(num_receivers_ > 0);
        ASSERT(credit_semaphores_base_ptr_ != nullptr);
    }

    // Initialize credit semaphores
    void initialize() {
        if (!enabled_) {
            return;
        }

        for (uint32_t i = 0; i < num_receivers_; i++) {
            credit_semaphores_base_ptr_[i * CREDIT_STRIDE_WORDS] = 0;
        }
    }

    // Check if credits available (non-blocking, called before send)
    FORCE_INLINE bool has_credits_available(uint32_t num_packets_processed) const {
        if (!enabled_) {
            return true;  // Always available when disabled
        }

        // Fast path: if we think we have credits, return true
        if (estimated_available_credits_ > 0) {
            return true;
        }

        return const_cast<SenderCreditManager*>(this)->update_available_credits();
    }

    // Update available credits by checking all receivers (called when blocked)
    bool update_available_credits() {
        if (!enabled_) {
            return true;
        }

        invalidate_l1_cache();

        // Find minimum credits across all receivers (slowest receiver determines limit)
        uint32_t min_credits = credit_semaphores_base_ptr_[0];

        for (uint32_t i = 1; i < num_receivers_; i++) {
            uint32_t recv_credits = credit_semaphores_base_ptr_[i * CREDIT_STRIDE_WORDS];
            if (recv_credits < min_credits) {
                min_credits = recv_credits;
            }
        }

        int32_t new_credits = min_credits - prev_processed_credits_;
        if (new_credits <= 0) {
            return false;  // No new credits available
        }

        estimated_available_credits_ = new_credits;
        prev_processed_credits_ += new_credits;
        return true;
    }

    // Hands the currently available credits to a caller that counts them down itself (the sender's local-memory
    // hot state) and returns how many; 0 when none have come back yet.
    uint32_t take_available_credits() {
        if (estimated_available_credits_ == 0 && !update_available_credits()) {
            return 0;
        }
        const uint32_t credits = estimated_available_credits_;
        estimated_available_credits_ = 0;
        return credits;
    }

    // Consume one credit (called after successful send - decrements ALL receivers for mcast)
    FORCE_INLINE void consume_credit() {
        if (!enabled_) {
            return;
        }

        ASSERT(estimated_available_credits_ > 0);
        estimated_available_credits_--;
    }

    // Wait for all credits back (called at connection close)
    bool got_all_credits_back() {
        if (!enabled_) {
            return true;
        }

        if (!got_all_credits_back_) {
            invalidate_l1_cache();
            got_all_credits_back_ = true;

            for (uint32_t i = 0; i < num_receivers_; i++) {
                if (credit_semaphores_base_ptr_[i * CREDIT_STRIDE_WORDS] < total_credits_) {
                    got_all_credits_back_ = false;
                    break;
                }
            }
        }

        return got_all_credits_back_;
    }

    bool is_enabled() const { return enabled_; }

private:
    bool enabled_ = false;
    SenderCreditInfo sender_credit_info_;

    // Per-receiver credit tracking
    volatile tt_l1_ptr uint32_t* credit_semaphores_base_ptr_ = nullptr;
    uint64_t credit_semaphores_base_noc_addr_ = 0;
    uint32_t num_receivers_ = 0;
    uint32_t initial_credits_ = 0;
    uint32_t total_credits_ = 0;
    uint32_t estimated_available_credits_ = 0;
    uint32_t prev_processed_credits_ = 0;
    bool got_all_credits_back_ = false;

    static constexpr uint32_t CREDIT_ADDRESS_STRIDE = 16;
    static constexpr uint32_t CREDIT_STRIDE_WORDS = CREDIT_ADDRESS_STRIDE / sizeof(uint32_t);
};

// A sender config's per-packet state, copied into the sender's local memory (its stack) for the send loop. The
// config itself lives in L1, where every field read costs an L1 load (and the fabric adapters invalidate the L1 data
// cache on each slot check); reading the hot fields from local memory keeps those loads off the per-packet path.
// HOT_FALLBACK configs (multi-injection mcast) send through the L1 config as before.
#ifdef FABRIC_TEST_CORRUPT_PAYLOAD
// Verifier canary: one flipped bit in packet 7 of every config with a payload, whichever engine produced it, must fail
// validation. FABRIC_TEST_CORRUPT_PAYLOAD 1 flips a word mid-payload, 2 the second-to-last word and 3 the third-to-last
// (the last block's checks, on both word parities of a payload in 16 B steps). The last word itself is the receiver's
// sign that the packet has landed (poll), so corrupting it hangs the receiver instead of failing it.
FORCE_INLINE void corrupt_payload_canary(uint32_t payload_address, uint32_t payload_size_bytes, uint32_t packet_idx) {
    if (packet_idx == 7) {
        const uint32_t words = payload_size_bytes / 4;
        const uint32_t word = FABRIC_TEST_CORRUPT_PAYLOAD == 1 ? words / 2 : words - FABRIC_TEST_CORRUPT_PAYLOAD;
        reinterpret_cast<tt_l1_ptr uint32_t*>(payload_address)[word] ^= 0x10;
    }
}
#endif

struct SenderHotState {
    enum Mode : uint8_t { HOT_FIXED_HEADER = 0, HOT_WRITE_ADDRESS = 1, HOT_FALLBACK = 2, HOT_SCATTER = 3 };

    uint64_t header_noc_base;  // HOT_WRITE_ADDRESS: the header's write NOC address at ring offset 0
    void* conn;                // the config's connection (in the local-memory connection array)
    uint32_t header_addr;
    uint32_t packets_left;
    uint32_t credits;  // flow-control credits on hand
    uint32_t ring_offset;
    uint32_t ring_wrap;  // ring offsets past this wrap to 0
    uint32_t payload_size;
    uint32_t seed;
    uint8_t is_mux;
    uint8_t conn_idx;  // index of conn in the connection array (keys the per-connection slot cache)
    uint8_t mode;
    uint8_t flow_control;
    uint8_t scatter_chunks;  // HOT_SCATTER: how many chunk NOC addresses the header carries
    uint8_t header_stale;    // FABRIC_TEST_SENDER_BATCH: the header still describes the last packet sent
    uint8_t sent_epoch;      // FABRIC_TEST_SENDER_BATCH: sender_batch.flush_epoch at the last send
    uint8_t sfpu_payload;    // payloads come from the core's compute engine (SenderTrafficConfigMetadata)
};

enum class HotSendResult : uint8_t { SENT, NO_CREDITS, CONNECTION_FULL };

#ifdef FABRIC_TEST_SENDER_BATCH
// FABRIC_TEST_SENDER_BATCH: the hot sender flushes its NOC writes only when it must, instead of after every packet:
// before a config's header is rewritten (or its payload refilled) while that config's last send may still be reading
// it, and with SFPU payloads once per FABRIC_TEST_SENDER_BATCH_PAGES pages, when the pages sent since the last flush
// go back to the generator in one CB update. Pages come from a count of pages known to be ready, so the CB is read
// only when that count runs out. Headers are advanced at the config's next send, not right after a send.
struct SenderBatchState {
    uint32_t page_addr;    // next SFPU page to send
    uint32_t pages_ready;  // pages generated and not yet sent
    uint32_t pages_sent;   // pages sent since the last flush, not yet returned to the generator
    uint32_t pages_acked;  // the CB's acked count (only this kernel writes it)
    uint8_t flush_epoch;   // flushes so far, mod 256
};
inline SenderBatchState sender_batch{};

// Picks up the payload CB's state, at the start and after the fallback path has used the CB directly. Every sender
// core has the CB (a placeholder where no config takes SFPU pages), so this is harmless on cores that never use it.
inline void sender_batch_sync() {
    sender_batch.page_addr = get_read_ptr(SFPU_PAYLOAD_CB_INDEX);
    sender_batch.pages_acked = *get_cb_tiles_acked_ptr(SFPU_PAYLOAD_CB_INDEX);
    sender_batch.pages_ready = 0;
}

FORCE_INLINE void sender_batch_flush() {
    noc_async_writes_flushed();
    if (sender_batch.pages_sent > 0) {
        sender_batch.pages_acked += sender_batch.pages_sent;
        sender_batch.pages_sent = 0;
        *get_cb_tiles_acked_ptr(SFPU_PAYLOAD_CB_INDEX) = sender_batch.pages_acked;
        // Where cb_pop_front would leave the read pointer, for the fallback path.
        get_local_cb_interface(SFPU_PAYLOAD_CB_INDEX).fifo_rd_ptr = sender_batch.page_addr;
    }
    sender_batch.flush_epoch++;
}

// The next generated page (strict generation order). A flush returns at least one page per
// FABRIC_TEST_SENDER_BATCH_PAGES (at most half the CB) sent, so the generator always has room while this waits.
FORCE_INLINE uint32_t sender_batch_take_page() {
    if (sender_batch.pages_ready == 0) {
        const uintptr_t received_ptr = reinterpret_cast<uintptr_t>(get_cb_tiles_received_ptr(SFPU_PAYLOAD_CB_INDEX));
        const uint16_t taken = static_cast<uint16_t>(sender_batch.pages_acked + sender_batch.pages_sent);
        uint16_t ready;
        do {
            ready = static_cast<uint16_t>(static_cast<uint16_t>(reg_read(received_ptr)) - taken);
        } while (ready == 0);
        sender_batch.pages_ready = ready;
    }
    sender_batch.pages_ready--;
    sender_batch.pages_sent++;
    const auto& cb = get_local_cb_interface(SFPU_PAYLOAD_CB_INDEX);
    const uint32_t page = sender_batch.page_addr;
    sender_batch.page_addr += cb.fifo_page_size;
    if (sender_batch.page_addr == cb.fifo_limit) {
        sender_batch.page_addr -= cb.fifo_size;
    }
    return page;
}
#endif

#ifdef FABRIC_TEST_SENDER_CMDBUFS
// FABRIC_TEST_SENDER_CMDBUFS: the hot sender writes a packet's payload, header and slot credit on three command
// buffers (write, write-reg and atomic) whose control registers are set once (sender_cmdbuf_init). Moving to another
// connection rewrites the data buffers' destination coordinates and reprograms the credit buffer; a send on the same
// connection as the last one sets only its source and slot addresses (and the payload length when it changes).
// FABRIC_TEST_SENDER_CREDIT_BATCH also holds back the slot credits of consecutive sends on one connection and returns
// them in one write (sender_credits_flush) before the sender moves to another connection or waits on anything.
// Anything else that uses these command buffers on this core must call sender_cmdbuf_init() afterwards.
#if defined(FABRIC_TEST_SENDER_CREDIT_BATCH) && !defined(FABRIC_TEST_SENDER_SLOT_CACHE)
#error "FABRIC_TEST_SENDER_CREDIT_BATCH needs FABRIC_TEST_SENDER_SLOT_CACHE (credits are returned before a slot re-read)"
#endif
#if defined(FABRIC_TEST_SENDER_CREDIT_BATCH) && defined(FABRIC_TEST_SENDER_SKIP)
#error "FABRIC_TEST_SENDER_CREDIT_BATCH is not wired into the rotating (FABRIC_TEST_SENDER_SKIP) loop"
#endif
struct SenderCmdBufState {
    uint8_t conn;              // the connection the command buffers point at (UINT8_MAX: none)
    uint32_t payload_len;      // the length in the payload command buffer
    uint32_t credits_pending;  // FABRIC_TEST_SENDER_CREDIT_BATCH: slot credits not yet returned to conn
};
inline SenderCmdBufState sender_cmdbuf{};

inline void sender_cmdbuf_init() {
    static_assert(noc_mode == DM_DEDICATED_NOC, "write and write-reg command buffers alias in dynamic NOC mode");
    constexpr uint8_t noc = get_fabric_worker_noc();
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    noc_write_init_state<write_cmd_buf>(noc, NOC_UNICAST_WRITE_VC);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, 0);
    while (!noc_cmd_buf_ready(noc, write_reg_cmd_buf));
    noc_write_init_state<write_reg_cmd_buf>(noc, NOC_UNICAST_WRITE_VC);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_AT_LEN_BE, sizeof(PACKET_HEADER_TYPE));
    sender_cmdbuf.conn = UINT8_MAX;
    sender_cmdbuf.payload_len = 0;
    sender_cmdbuf.credits_pending = 0;
}

// Returns the held-back slot credits of the current connection.
FORCE_INLINE void sender_credits_flush() {
#ifdef FABRIC_TEST_SENDER_CREDIT_BATCH
    if (sender_cmdbuf.credits_pending > 0) {
        noc_inline_dw_write_with_state<
            /*update_addr_lo=*/false,
            /*update_counter=*/true,
            /*posted=*/false,
            /*update_addr_hi=*/false,
            /*update_val=*/true,
            InlineWriteDst::REG>(
            pack_value_for_inc_on_write_stream_reg_write(-static_cast<int32_t>(sender_cmdbuf.credits_pending)),
            0,
            write_at_cmd_buf,
            get_fabric_worker_noc());
        sender_cmdbuf.credits_pending = 0;
    }
#endif
}
#endif

template <typename EdmSenderT>
struct SenderKernelTrafficConfig {
    SenderKernelTrafficConfig(
        FabricConnectionArray<EdmSenderT>* connection_array,
        const uint8_t* connection_indices,
        uint8_t num_connections,
        const SenderTrafficConfigMetadata& metadata,
        const uint32_t packet_header_address) :
        connection_manager_(connection_array),
        num_connections_(num_connections),
        metadata(metadata),
        noc_send_type_(static_cast<NocSendType>(0)),
        payload_buffer_(nullptr) {
        packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);

        ASSERT(num_connections_ > 0 && num_connections_ <= MAX_MCAST_INJECTIONS);
        for (uint8_t i = 0; i < num_connections_; i++) {
            const uint8_t idx = connection_indices[i];
            connection_indices_[i] = idx;
            if (connection_manager_->is_mux[idx]) {
                connection_ptrs_[i] = &connection_manager_->get_mux_connection(idx);
            } else {
                connection_ptrs_[i] = &connection_manager_->get_fabric_connection(idx);
            }
        }
        connection_idx_ = connection_indices_[0];
        connection_ptr_ = connection_ptrs_[0];

        // Initialize function pointers to null (will be set in parse_and_setup_noc_send_type)
        noc_ops_.parse_and_setup = nullptr;
        noc_ops_.update_header = nullptr;
    }

    template <bool IS_2D_FABRIC>
    void parse_and_setup_chip_send_type(size_t& arg_idx, uint32_t packet_header_address) {
        ChipSendType chip_send_type = static_cast<ChipSendType>(get_local_arg_val<uint32_t>(arg_idx++));

        if (chip_send_type == ChipSendType::CHIP_UNICAST) {
            ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, IS_2D_FABRIC>::parse_and_setup(
                arg_idx, packet_header_address, packet_header);
        } else if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
            ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, IS_2D_FABRIC>::parse_and_setup(
                arg_idx, packet_header_address, packet_header);
        } else {
            ASSERT(false);
        }
    }

    void parse_and_setup_noc_send_type(size_t& arg_idx) {
        uint32_t noc_type_raw = get_local_arg_val<uint32_t>(arg_idx++);
        noc_send_type_ = static_cast<NocSendType>(noc_type_raw);

        switch (noc_send_type_) {
            case NocSendType::NOC_UNICAST_WRITE:
                noc_ops_.parse_and_setup = NocWriteSenderOperations<EdmSenderT>::parse_and_setup_impl;
                noc_ops_.update_header = NocWriteSenderOperations<EdmSenderT>::update_header_impl;
                break;
            case NocSendType::NOC_UNICAST_ATOMIC_INC:
                noc_ops_.parse_and_setup = NocAtomicSenderOperations<EdmSenderT>::parse_and_setup_impl;
                noc_ops_.update_header = NocAtomicSenderOperations<EdmSenderT>::update_header_impl;
                break;
            case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC:
                noc_ops_.parse_and_setup = NocFusedSenderOperations<EdmSenderT>::parse_and_setup_impl;
                noc_ops_.update_header = NocFusedSenderOperations<EdmSenderT>::update_header_impl;
                break;
            case NocSendType::NOC_UNICAST_SCATTER_WRITE:
                noc_ops_.parse_and_setup = NocScatterWriteSenderOperations<EdmSenderT>::parse_and_setup_impl;
                noc_ops_.update_header = NocScatterWriteSenderOperations<EdmSenderT>::update_header_impl;
                break;
            default: ASSERT(false); break;
        }

        ASSERT(noc_ops_.parse_and_setup != nullptr);
        ASSERT(noc_ops_.update_header != nullptr);

        noc_ops_.parse_and_setup(this, arg_idx);
    }

    void setup_payload_buffer(uint32_t payload_buffer_address, uint32_t payload_buffer_size) {
        if (payload_size_bytes > 0) {
            payload_buffer_ = new (payload_buffer_storage.data())
                SenderPayloadBuffer(payload_buffer_address, payload_buffer_size, this->payload_size_bytes);
        } else {
            payload_buffer_ = nullptr;
        }
    }

    bool has_packets_to_send() const { return num_packets_processed < metadata.num_packets; }

    FORCE_INLINE void setup_credit_update_noc_state(const EdmSenderT& adapter, uint8_t noc) {
        auto packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
        const uint64_t noc_sem_addr =
            get_noc_addr(adapter.edm_noc_x, adapter.edm_noc_y, adapter.edm_buffer_remote_free_slots_update_addr, noc);
        noc_inline_dw_write_set_state<false /*posted*/, true /*set_val*/>(
            noc_sem_addr, packed_val, 0xf, adapter.sync_noc_cmd_buf, noc);
    }

    template <bool BENCHMARK_MODE>
    FORCE_INLINE void send_packets_stateful(const uint32_t num_packets, const uint32_t num_warmup) {
        ASSERT(connection_ptr_ != nullptr);
        // The stateful path holds NOC state for one connection, so it cannot fan a packet out to
        // several. Only benchmark flows take it, and those are unicast.
        ASSERT(num_connections_ == 1);
        auto* conn = static_cast<EdmSenderT*>(connection_ptr_);

        // Perform stateful noc send by filling buffers with headers, first, then performing credit-only NOC sends
        // Phase 1: Warmup — send actual headers to fill all buffer slots
        const uint32_t warmup_end = (num_packets < num_warmup) ? num_packets : num_warmup;
        for (uint32_t pkt = 0; pkt < warmup_end; pkt++) {
            this->template send_one_packet<BENCHMARK_MODE, false>();
        }

        setup_credit_update_noc_state(*conn, get_fabric_worker_noc());

        // Phase 2: Steady state — credit-only sends with stateful NOC
        for (uint32_t pkt = warmup_end; pkt < num_packets; pkt++) {
            this->template send_one_packet<BENCHMARK_MODE, true>();
        }
    }

    // Send exactly one packet per call (round-robin scheduling)
    // Returns: true if packet was sent, false if blocked (no credits)
    template <bool BENCHMARK_MODE, bool STATEFUL_NOC = false>
    bool send_one_packet() {
        // STEP 1: Check credits BEFORE sending (non-benchmark mode only)
        if constexpr (!BENCHMARK_MODE) {
            [[maybe_unused]] uint64_t credit_t0 = PERF_BD_DETAIL_NOW();
            const bool has_credits = credit_manager_.has_credits_available(num_packets_processed);
            PERF_BD_DETAIL_ADD_SINCE(0, credit_t0);  // SD_CREDIT_CHECK
            if (!has_credits) {
                PERF_BD_SUMMARY_INC(5);  // S_CREDIT_BLOCKED
                return false;            // No credits available - blocked
            }
        }

        // STEP 2: Wait for space
        if constexpr (BENCHMARK_MODE) {
            connection_manager_->template wait_for_empty_write_slot<BENCHMARK_MODE>(connection_ptr_, connection_idx_);
            // STEP 3: Send packet
            auto* conn = static_cast<EdmSenderT*>(connection_ptr_);
            if (num_packets_processed < conn->num_buffers_per_channel) {
                conn->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
            } else {
                fabric_detail::update_credits_and_slots<STATEFUL_NOC>(conn);
            }
        } else {
            const bool has_payload = payload_size_bytes > 0 && payload_buffer_;
            uint32_t payload_address = 0;
            if (has_payload) {
                [[maybe_unused]] uint64_t fill_t0 = PERF_BD_NOW();
                if (metadata.sfpu_payload) {
                    // The core's compute kernel generates this packet's payload (same seed + i words) into the next
                    // page of the payload CB, in the same strict round-robin order the sender consumes them.
                    cb_wait_front(SFPU_PAYLOAD_CB_INDEX, 1);
                    payload_address = get_read_ptr(SFPU_PAYLOAD_CB_INDEX);
                } else {
                    payload_buffer_->fill_data(metadata.seed);
                    payload_address = payload_buffer_->get_physical_address();
                }
#ifdef FABRIC_TEST_CORRUPT_PAYLOAD
                corrupt_payload_canary(payload_address, payload_size_bytes, num_packets_processed);
#endif
                PERF_BD_ADD_SINCE(2, fill_t0);  // S_FILL (S_FILL = waiting for the generated page in SFPU mode)
            }
            // STEP 3: Send packet, once per connection.
            for (uint8_t i = 0; i < num_connections_; i++) {
                [[maybe_unused]] uint64_t slot_t0 = PERF_BD_NOW();
                connection_manager_->template wait_for_empty_write_slot<BENCHMARK_MODE>(
                    connection_ptrs_[i], connection_indices_[i]);
                PERF_BD_ADD_SINCE(3, slot_t0);  // S_SLOT_WAIT
                if (has_payload) {
                    [[maybe_unused]] uint64_t payload_t0 = PERF_BD_DETAIL_NOW();
                    // Send payload without header
                    connection_manager_->template send_payload_without_header<BENCHMARK_MODE>(
                        connection_ptrs_[i],
                        connection_indices_[i],
                        payload_address,
                        payload_size_bytes);
                    PERF_BD_DETAIL_ADD_SINCE(5, payload_t0);  // SD_PAYLOAD_SEND
                }
                [[maybe_unused]] uint64_t header_t0 = PERF_BD_DETAIL_NOW();
                // Send header
                connection_manager_->template send_header_non_blocking<BENCHMARK_MODE>(
                    connection_ptrs_[i], connection_indices_[i], (uint32_t)packet_header);
                PERF_BD_DETAIL_ADD_SINCE(6, header_t0);  // SD_HEADER_SEND
            }
        }

        // STEP 4: Update state (after successful send)
        if constexpr (!BENCHMARK_MODE) {
            // avoid race condition where we update the ptrs but fabric write is not done yet.
            [[maybe_unused]] uint64_t flush_t0 = PERF_BD_NOW();
            noc_async_writes_flushed();
            PERF_BD_ADD_SINCE(4, flush_t0);  // S_FLUSH

            [[maybe_unused]] uint64_t post_t0 = PERF_BD_DETAIL_NOW();
            if (payload_size_bytes > 0 && payload_buffer_) {
                if (metadata.sfpu_payload) {
                    // The flush above means the page has left L1, so the generator may reuse it.
                    cb_pop_front(SFPU_PAYLOAD_CB_INDEX, 1);
                }
                payload_buffer_->advance();
                update_header_for_next_packet();
                metadata.seed = prng_next(metadata.seed);
            }

            // STEP 5: Consume credit AFTER successful send
            credit_manager_.consume_credit();
            PERF_BD_DETAIL_ADD_SINCE(7, post_t0);  // SD_POST
        }

        num_packets_processed += 1;  // Always increment by 1

        return true;  // Packet sent successfully
    }

    void advance_dst_address() {
        if (payload_buffer_) {
            payload_buffer_->advance();
            update_header_for_next_packet();
        }
    }

    void reset_dst_address() {
        if (payload_buffer_) {
            payload_buffer_->reset();
            update_header_for_next_packet();
        }
    }

    bool has_wrapped() const { return payload_buffer_ ? payload_buffer_->has_wrapped() : false; }

    // Points this config at another copy of the connection array (the sender's local-memory copy, and back).
    void rebind_connections(FabricConnectionArray<EdmSenderT>* connection_array) {
        connection_manager_ = connection_array;
        for (uint8_t i = 0; i < num_connections_; i++) {
            const uint8_t idx = connection_indices_[i];
            connection_ptrs_[i] = connection_array->is_mux[idx]
                                      ? static_cast<void*>(&connection_array->get_mux_connection(idx))
                                      : static_cast<void*>(&connection_array->get_fabric_connection(idx));
        }
        connection_ptr_ = connection_ptrs_[0];
    }

    // Call after rebind_connections() so the hot state carries the local-memory connection.
    SenderHotState make_hot_state() {
        SenderHotState h{};
        h.conn = connection_ptr_;
        h.conn_idx = connection_idx_;
        h.is_mux = connection_manager_->is_mux[connection_idx_];
        h.header_addr = reinterpret_cast<uint32_t>(packet_header);
        h.packets_left = metadata.num_packets - num_packets_processed;
        h.flow_control = credit_manager_.is_enabled();
        h.credits = 0;
        h.seed = metadata.seed;
        h.payload_size = payload_buffer_ ? payload_size_bytes : 0;
        h.sfpu_payload = h.payload_size > 0 && metadata.sfpu_payload;
        h.ring_offset = payload_buffer_ ? payload_buffer_->get_current_offset() : 0;
        h.ring_wrap = payload_buffer_ ? payload_buffer_->get_total_size() - payload_size_bytes : 0;
        if (num_connections_ != 1) {
            h.mode = SenderHotState::HOT_FALLBACK;
        } else if (noc_send_type_ == NocSendType::NOC_UNICAST_ATOMIC_INC) {
            h.mode = SenderHotState::HOT_FIXED_HEADER;
        } else if (
            noc_send_type_ == NocSendType::NOC_UNICAST_WRITE ||
            noc_send_type_ == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
            // Both headers keep the write address as the first word of the command fields.
            static_assert(offsetof(NocUnicastAtomicIncFusedCommandHeader, noc_address) == 0);
            static_assert(offsetof(NocUnicastCommandHeader, noc_address) == 0);
            h.mode = SenderHotState::HOT_WRITE_ADDRESS;
            h.header_noc_base = packet_header->command_fields.unicast_write.noc_address - h.ring_offset;
        } else if (noc_send_type_ == NocSendType::NOC_UNICAST_SCATTER_WRITE) {
            // Both scatter headers (with or without the fused increment) start with the chunks' NOC addresses.
            static_assert(offsetof(NocUnicastScatterCommandHeader, noc_address) == 0);
            static_assert(offsetof(NocUnicastScatterAtomicIncFusedCommandHeader, noc_address) == 0);
            h.mode = SenderHotState::HOT_SCATTER;
            h.scatter_chunks = noc_fields_.scatter_write_fields.chunk_count;
        } else {
            h.mode = SenderHotState::HOT_FALLBACK;
        }
        return h;
    }

    // Whether this config may be passed over while its connection is full (FABRIC_TEST_SENDER_SKIP). Configs that
    // take SFPU-generated pages may not: the pages arrive in strict round-robin order.
    FORCE_INLINE static bool can_skip_when_full([[maybe_unused]] const SenderHotState& h) {
#ifdef FABRIC_TEST_SENDER_SKIP
        return !h.sfpu_payload;
#else
        return false;
#endif
    }

    // Free slots on h's connection right now (one read of the fabric/mux credit state).
    FORCE_INLINE static uint32_t connection_free_slots(const SenderHotState& h) {
        return h.is_mux ? static_cast<typename FabricConnectionArray<EdmSenderT>::MuxConnectionType*>(h.conn)
                              ->get_num_free_write_slots()
                        : static_cast<EdmSenderT*>(h.conn)->get_num_free_write_slots();
    }

    // send_one_packet() on local-memory state. slot_budget is the sender's cached count of free slots on this config's
    // connection (FABRIC_TEST_SENDER_SLOT_CACHE): sends count it down and the connection is only re-read once it is
    // used up. Returns CONNECTION_FULL only for configs that may be skipped.
    FORCE_INLINE HotSendResult send_one_packet_hot(SenderHotState& h, [[maybe_unused]] uint32_t& slot_budget) {
        PERF_HOT(perf_hot_begin();)
        if (h.flow_control && h.credits == 0) {
            h.credits = credit_manager_.take_available_credits();
            if (h.credits == 0) {
#ifdef FABRIC_TEST_SENDER_CMDBUFS
                sender_credits_flush();  // the receivers may be waiting on the held-back packets
#endif
                PERF_BD_SUMMARY_INC(5);  // S_CREDIT_BLOCKED
                PERF_HOT(perf_hot_step(HS_CREDIT);)
                return HotSendResult::NO_CREDITS;
            }
        }
        PERF_HOT(perf_hot_step(HS_CREDIT);)
#ifdef FABRIC_TEST_SENDER_BATCH
        // Advance the header past the last packet, once that packet has left L1.
        if (h.header_stale) {
            if (h.sent_epoch == sender_batch.flush_epoch) {
                sender_batch_flush();
                PERF_HOT(perf_hot_step(HS_FLUSH);)
            }
            advance_hot_header(h);
            h.header_stale = 0;
            PERF_HOT(perf_hot_step(HS_POST);)
        }
#endif
#ifdef FABRIC_TEST_SENDER_SLOT_CACHE
        if (slot_budget == 0) {
#ifdef FABRIC_TEST_SENDER_CMDBUFS
            sender_credits_flush();  // the connection frees no slot it has not been told about
#endif
            slot_budget = connection_free_slots(h);
            if (slot_budget == 0) {
                if (can_skip_when_full(h)) {
                    PERF_BD_SUMMARY_INC(5);  // counts skips with the credit-blocked attempts
                    PERF_HOT(perf_hot_step(HS_SLOT);)
                    return HotSendResult::CONNECTION_FULL;
                }
                [[maybe_unused]] uint64_t slot_t0 = PERF_BD_HOT_NOW();
                do {
                    slot_budget = connection_free_slots(h);
                } while (slot_budget == 0);
                PERF_BD_HOT_ADD_SINCE(3, slot_t0);  // S_SLOT_WAIT
            }
        }
        slot_budget--;
        PERF_HOT(perf_hot_step(HS_SLOT);)
#endif

        uint32_t payload_address = 0;
        if (h.payload_size > 0) {
            [[maybe_unused]] uint64_t fill_t0 = PERF_BD_HOT_NOW();
            if (h.sfpu_payload) {
#ifdef FABRIC_TEST_SENDER_BATCH
                payload_address = sender_batch_take_page();
#else
                cb_wait_front(SFPU_PAYLOAD_CB_INDEX, 1);
                payload_address = get_read_ptr(SFPU_PAYLOAD_CB_INDEX);
#endif
            } else {
                payload_buffer_->fill_data(h.seed);
                payload_address = payload_buffer_->get_physical_address();
            }
#ifdef FABRIC_TEST_CORRUPT_PAYLOAD
            corrupt_payload_canary(payload_address, h.payload_size, metadata.num_packets - h.packets_left);
#endif
            PERF_BD_HOT_ADD_SINCE(2, fill_t0);  // S_FILL
            PERF_HOT(perf_hot_step(HS_FILL);)
        }

#ifdef FABRIC_TEST_SENDER_SLOT_CACHE
        constexpr bool WAIT_FOR_SLOT = false;  // the slot was taken from slot_budget above
#else
        constexpr bool WAIT_FOR_SLOT = true;
#endif
        if (h.is_mux) {
            send_on_connection_hot<WAIT_FOR_SLOT>(
                static_cast<typename FabricConnectionArray<EdmSenderT>::MuxConnectionType*>(h.conn),
                payload_address,
                h);
        } else {
            send_on_connection_hot<WAIT_FOR_SLOT>(static_cast<EdmSenderT*>(h.conn), payload_address, h);
        }

#ifdef FABRIC_TEST_SENDER_BATCH
        // Only a config whose header or payload source changes per packet needs its writes flushed before reuse.
        if (h.payload_size > 0) {
            h.header_stale = 1;
            h.sent_epoch = sender_batch.flush_epoch;
            if (sender_batch.pages_sent >= FABRIC_TEST_SENDER_BATCH_PAGES) {
                sender_batch_flush();
            }
        }
        PERF_HOT(perf_hot_step(HS_FLUSH);)
#else
        // The header and payload must leave L1 before the header is rewritten and the page is reused.
        [[maybe_unused]] uint64_t flush_t0 = PERF_BD_HOT_NOW();
        noc_async_writes_flushed();
        PERF_BD_HOT_ADD_SINCE(4, flush_t0);  // S_FLUSH
        PERF_HOT(perf_hot_step(HS_FLUSH);)

        if (h.payload_size > 0) {
            if (h.sfpu_payload) {
                cb_pop_front(SFPU_PAYLOAD_CB_INDEX, 1);
            }
            advance_hot_header(h);
        }
#endif
        h.credits--;  // unused (and harmless) without flow control
        h.packets_left--;
        PERF_HOT(perf_hot_end();)
        return HotSendResult::SENT;
    }

    // Moves a payload config's header (destination address) and seed on to its next packet.
    FORCE_INLINE static void advance_hot_header(SenderHotState& h) {
        const uint32_t prev_ring_offset = h.ring_offset;
        h.ring_offset += h.payload_size;
        if (h.ring_offset > h.ring_wrap) {
            h.ring_offset = 0;
        }
        auto* header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(h.header_addr);
        if (h.mode == SenderHotState::HOT_WRITE_ADDRESS) {
            header->command_fields.unicast_write.noc_address = h.header_noc_base + h.ring_offset;
        } else if (h.mode == SenderHotState::HOT_SCATTER) {
            // Every chunk keeps its place in the slot: move each chunk address with the slot.
            const uint64_t step = static_cast<uint64_t>(h.ring_offset) - prev_ring_offset;
            for (uint8_t i = 0; i < h.scatter_chunks; i++) {
                header->command_fields.unicast_scatter_write.noc_address[i] += step;
            }
        }
        h.seed = prng_next(h.seed);
    }

    // Writes the hot state's progress back to this L1 config (packet count; the rest is not read after the loop).
    void write_back_hot_state(const SenderHotState& h) {
        if (h.mode != SenderHotState::HOT_FALLBACK) {
            num_packets_processed = metadata.num_packets - h.packets_left;
        }
    }

    friend struct NocWriteSenderOperations<EdmSenderT>;
    friend struct NocAtomicSenderOperations<EdmSenderT>;
    friend struct NocFusedSenderOperations<EdmSenderT>;
    friend struct NocScatterWriteSenderOperations<EdmSenderT>;

private:
    template <bool WAIT_FOR_SLOT, typename ConnT>
    FORCE_INLINE void send_on_connection_hot(ConnT* conn, uint32_t payload_address, const SenderHotState& h) {
        if constexpr (WAIT_FOR_SLOT) {
            [[maybe_unused]] uint64_t slot_t0 = PERF_BD_HOT_NOW();
            conn->wait_for_empty_write_slot();
            PERF_BD_HOT_ADD_SINCE(3, slot_t0);  // S_SLOT_WAIT
            PERF_HOT(perf_hot_step(HS_SLOT);)
        }
#ifdef FABRIC_TEST_SENDER_CMDBUFS
        send_on_connection_cmdbufs(conn, payload_address, h);
#else
        if (h.payload_size > 0) {
            conn->send_payload_without_header_non_blocking_from_address(payload_address, h.payload_size);
            PERF_HOT(perf_hot_step(HS_PAYLOAD);)
        }
        conn->send_payload_flush_non_blocking_from_address(h.header_addr, sizeof(PACKET_HEADER_TYPE));
        PERF_HOT(perf_hot_step(HS_HEADER);)
#endif
    }

#ifdef FABRIC_TEST_SENDER_CMDBUFS
    // The adapter's send_payload_without_header / send_payload_flush pair plus its slot credit and slot advance
    // (advance_buffer_slot_write_index), on this file's command buffers (see SenderCmdBufState).
    template <typename ConnT>
    FORCE_INLINE void send_on_connection_cmdbufs(ConnT* conn, uint32_t payload_address, const SenderHotState& h) {
        static_assert(ConnT::IS_WORKER, "only a worker connection returns its slot credit with one inline write");
        constexpr uint32_t HEADER_SIZE = sizeof(PACKET_HEADER_TYPE);
        constexpr uint8_t noc = get_fabric_worker_noc();
        const bool switched = sender_cmdbuf.conn != h.conn_idx;
        if (switched) {
            sender_credits_flush();
            noc_inline_dw_write_set_state</*posted=*/false, /*set_val=*/true>(
                get_noc_addr(conn->edm_noc_x, conn->edm_noc_y, conn->edm_buffer_remote_free_slots_update_addr, noc),
                pack_value_for_inc_on_write_stream_reg_write(-1),
                0xF,
                write_at_cmd_buf,
                noc,
                NOC_UNICAST_WRITE_VC);
            sender_cmdbuf.conn = h.conn_idx;
        }
        uint32_t slot;
        if constexpr (ConnT::USER_DEFINED_NUM_BUFFER_SLOTS) {
            slot = conn->edm_buffer_slot_addrs[conn->buffer_slot_index.get()];
        } else {
            slot = conn->edm_buffer_addr;
        }
        const uint64_t slot_noc_addr = get_noc_addr(conn->edm_noc_x, conn->edm_noc_y, slot, noc);
        if (h.payload_size > 0) {
            const bool new_len = h.payload_size != sender_cmdbuf.payload_len;
            sender_cmdbuf.payload_len = h.payload_size;
            const uint64_t dst = slot_noc_addr + HEADER_SIZE;
            if (switched) {
                if (new_len) {
                    noc_write_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL>(noc, payload_address, dst, h.payload_size);
                } else {
                    noc_write_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDl>(noc, payload_address, dst);
                }
            } else {
                if (new_len) {
                    noc_write_with_state<noc_mode, write_cmd_buf, CQ_NOC_SnDL>(noc, payload_address, dst, h.payload_size);
                } else {
                    noc_write_with_state<noc_mode, write_cmd_buf, CQ_NOC_SnDl>(noc, payload_address, dst);
                }
            }
            PERF_HOT(perf_hot_step(HS_PAYLOAD);)
        }
        if (switched) {
            noc_write_with_state<noc_mode, write_reg_cmd_buf, CQ_NOC_SNDl>(noc, h.header_addr, slot_noc_addr);
        } else {
            noc_write_with_state<noc_mode, write_reg_cmd_buf, CQ_NOC_SnDl>(noc, h.header_addr, slot_noc_addr);
        }
#ifdef FABRIC_TEST_SENDER_CREDIT_BATCH
        sender_cmdbuf.credits_pending++;
#else
        noc_inline_dw_write_with_state<
            /*update_addr_lo=*/false,
            /*update_counter=*/true,
            /*posted=*/false,
            /*update_addr_hi=*/false,
            /*update_val=*/false,
            InlineWriteDst::REG>(0, 0, write_at_cmd_buf, noc);
#endif
        // The adapter's slot advance (advance_buffer_slot_write_index) for a worker connection.
        conn->buffer_slot_write_counter.counter++;
        if constexpr (ConnT::USER_DEFINED_NUM_BUFFER_SLOTS) {
            constexpr size_t NUM_SLOTS = std::tuple_size_v<decltype(ConnT::edm_buffer_slot_addrs)>;
            conn->buffer_slot_index = BufferIndex{wrap_increment<NUM_SLOTS>(conn->buffer_slot_index.get())};
        } else {
            conn->buffer_slot_index =
                BufferIndex{wrap_increment(conn->buffer_slot_index.get(), conn->num_buffers_per_channel)};
            conn->edm_buffer_addr =
                conn->edm_buffer_base_addr + conn->buffer_slot_index.get() * conn->buffer_size_bytes;
        }
        PERF_HOT(perf_hot_step(HS_HEADER);)
    }
#endif

    void update_header_for_next_packet() {
        if (payload_buffer_) {
            noc_ops_.update_header(this);
        }
    }

public:
    FabricConnectionArray<EdmSenderT>* connection_manager_;
    void* connection_ptr_;    // Cached connection pointer (== connection_ptrs_[0])
    uint8_t connection_idx_;  // Index into the connection array (== connection_indices_[0])

    // Every connection this config injects into. Length 1 unless this is an express multicast root
    void* connection_ptrs_[MAX_MCAST_INJECTIONS];
    uint8_t connection_indices_[MAX_MCAST_INJECTIONS];
    uint8_t num_connections_;

    SenderTrafficConfigMetadata metadata;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header;
    uint32_t payload_size_bytes = 0;
    uint32_t num_packets_processed = 0;
    uint64_t elapsed_cycles = 0;

    SenderCreditManager credit_manager_;

private:
    NocSendType noc_send_type_;
    typename NocOperationTypes<EdmSenderT>::Operations noc_ops_;

    union NocFields {
        NocUnicastWriteFields write_fields;
        NocUnicastAtomicIncFields atomic_inc_fields;
        NocUnicastWriteAtomicIncFields write_atomic_inc_fields;
        NocUnicastScatterWriteFields scatter_write_fields;

        // Constructor needed because member types have user-defined constructors
        NocFields() {}  // Will be properly initialized later based on NOC type
    } noc_fields_;

    alignas(SenderPayloadBuffer) std::array<char, sizeof(SenderPayloadBuffer)> payload_buffer_storage;
    SenderPayloadBuffer* payload_buffer_;
};

// NOC Operation Implementations (now that SenderKernelTrafficConfig is fully defined)
template <typename EdmSenderT>
inline void NocWriteSenderOperations<EdmSenderT>::parse_and_setup_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx) {
    auto fields = NocUnicastWriteFields::build_from_args<true>(arg_idx);

    uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
    config->packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_addr}, fields.payload_size_bytes);

    config->noc_fields_.write_fields = fields;
    config->payload_size_bytes = fields.payload_size_bytes;
}

template <typename EdmSenderT>
inline void NocWriteSenderOperations<EdmSenderT>::update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config) {
    const auto& fields = config->noc_fields_.write_fields;
    uint32_t buffer_offset = config->payload_buffer_->get_current_offset();
    uint32_t dest_address = fields.dst_address + buffer_offset;
    uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, dest_address);
    config->packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_addr}, fields.payload_size_bytes);
}

template <typename EdmSenderT>
inline void NocAtomicSenderOperations<EdmSenderT>::parse_and_setup_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx) {
    auto fields = NocUnicastAtomicIncFields::build_from_args<true>(arg_idx);

    uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
    config->packet_header->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{noc_addr, fields.atomic_inc_val, ATOMIC_INC_FLUSH});

    config->noc_fields_.atomic_inc_fields = fields;
    config->payload_size_bytes = 0;
}

template <typename EdmSenderT>
inline void NocAtomicSenderOperations<EdmSenderT>::update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config) {
    // No-op - atomic operations use fixed addresses
}

template <typename EdmSenderT>
inline void NocFusedSenderOperations<EdmSenderT>::parse_and_setup_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx) {
    auto fields = NocUnicastWriteAtomicIncFields::build_from_args<true>(arg_idx);

    // Test infra allocates the payload and atomic counter on the same core, so NOC ordering preserves
    // write-before-atomic without flushing the write pipeline (the ntype option `flush` flushes anyway).
    ASSERT(fields.write_fields.dst_noc_encoding == fields.atomic_inc_fields.dst_noc_encoding);

    uint64_t write_noc_addr =
        get_noc_addr_helper(fields.write_fields.dst_noc_encoding, fields.write_fields.dst_address);
    uint64_t atomic_noc_addr =
        get_noc_addr_helper(fields.atomic_inc_fields.dst_noc_encoding, fields.atomic_inc_fields.dst_address);

    config->packet_header->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{
            write_noc_addr, atomic_noc_addr, fields.atomic_inc_fields.atomic_inc_val, fields.flush},
        fields.write_fields.payload_size_bytes);

    config->noc_fields_.write_atomic_inc_fields = fields;
    config->payload_size_bytes = fields.write_fields.payload_size_bytes;
}

template <typename EdmSenderT>
inline void NocFusedSenderOperations<EdmSenderT>::update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config) {
    const auto& fields = config->noc_fields_.write_atomic_inc_fields;
    uint32_t buffer_offset = config->payload_buffer_->get_current_offset();
    uint32_t write_dest_address = fields.write_fields.dst_address + buffer_offset;
    uint64_t write_noc_addr = get_noc_addr_helper(fields.write_fields.dst_noc_encoding, write_dest_address);
    uint64_t atomic_noc_addr =
        get_noc_addr_helper(fields.atomic_inc_fields.dst_noc_encoding, fields.atomic_inc_fields.dst_address);

    config->packet_header->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{
            write_noc_addr, atomic_noc_addr, fields.atomic_inc_fields.atomic_inc_val, fields.flush},
        fields.write_fields.payload_size_bytes);
}

template <typename EdmSenderT>
inline void NocScatterWriteSenderOperations<EdmSenderT>::write_header(
    SenderKernelTrafficConfig<EdmSenderT>* config, uint32_t buffer_offset) {
    const auto& fields = config->noc_fields_.scatter_write_fields;
    std::array<uint64_t, NocUnicastScatterWriteFields::MAX_CHUNKS> chunk_noc_addrs{};
    for (uint32_t i = 0; i < fields.chunk_count; i++) {
        chunk_noc_addrs[i] = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_addresses[i] + buffer_offset);
    }

    if (fields.has_atomic_inc) {
        ASSERT(fields.chunk_count == NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS);
        const uint64_t atomic_noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.atomic_inc_dst_address);
        config->packet_header->to_noc_fused_unicast_scatter_write_atomic_inc(
            NocUnicastScatterAtomicIncFusedCommandHeader(
                {chunk_noc_addrs[0], chunk_noc_addrs[1]},
                atomic_noc_addr,
                {fields.chunk_sizes[0]},
                static_cast<uint16_t>(fields.atomic_inc_val),
                fields.atomic_inc_flush),
            fields.payload_size_bytes);
    } else {
        config->packet_header->to_noc_unicast_scatter_write(
            NocUnicastScatterCommandHeader(chunk_noc_addrs.data(), fields.chunk_sizes.data(), fields.chunk_count),
            fields.payload_size_bytes);
    }
}

template <typename EdmSenderT>
inline void NocScatterWriteSenderOperations<EdmSenderT>::parse_and_setup_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx) {
    auto fields = NocUnicastScatterWriteFields::build_from_args<true>(arg_idx);
    config->noc_fields_.scatter_write_fields = fields;
    config->payload_size_bytes = fields.payload_size_bytes;
    write_header(config, 0);
}

template <typename EdmSenderT>
inline void NocScatterWriteSenderOperations<EdmSenderT>::update_header_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config) {
    write_header(config, config->payload_buffer_->get_current_offset());
}

struct CommonMemoryMap {
    CommonMemoryMap() = default;
    static CommonMemoryMap build_from_args(size_t& arg_idx) { return CommonMemoryMap(arg_idx); }

    uint32_t local_args_base;
    uint32_t local_args_size;
    uint32_t result_buffer_base;
    uint32_t result_buffer_size;
    uint32_t kernel_config_base;
    uint32_t kernel_config_size;
    uint32_t mux_local_addresses_base;
    uint32_t mux_local_addresses_size;
    uint32_t mux_termination_sync_address;

private:
    CommonMemoryMap(size_t& arg_idx) {
        // Extract and initialize local args system first
        local_args_base = get_arg_val<uint32_t>(arg_idx++);
        local_args_size = get_arg_val<uint32_t>(arg_idx++);
        init_local_args(local_args_base, local_args_size);

        // Then parse the rest
        result_buffer_base = get_arg_val<uint32_t>(arg_idx++);
        result_buffer_size = get_arg_val<uint32_t>(arg_idx++);
        kernel_config_base = get_arg_val<uint32_t>(arg_idx++);
        kernel_config_size = get_arg_val<uint32_t>(arg_idx++);
        mux_local_addresses_base = get_arg_val<uint32_t>(arg_idx++);
        mux_local_addresses_size = get_arg_val<uint32_t>(arg_idx++);
        mux_termination_sync_address = get_arg_val<uint32_t>(arg_idx++);
    }
};

/* ****************************************************************************
 * MuxLocalAddresses: Standalone struct for mux connection local semaphores
 *
 * Used by both sender and receiver memory maps to allocate local L1 addresses
 * for mux connection flow control.
 * ****************************************************************************/
struct MuxLocalAddresses {
    uint32_t flow_control_address;
    uint32_t teardown_address;
    uint32_t buffer_index_address;
    uint32_t status_buffer_address;
    uint32_t sync_address;

    static MuxLocalAddresses allocate_from_base(uint32_t base_address, uint32_t address_padding_bytes) {
        uint32_t current_addr = base_address;
        uint32_t flow_control_address = current_addr;
        current_addr += address_padding_bytes;
        uint32_t teardown_address = current_addr;
        current_addr += address_padding_bytes;
        uint32_t buffer_index_address = current_addr;
        current_addr += address_padding_bytes;
        uint32_t status_buffer_address = current_addr;
        current_addr += address_padding_bytes;
        uint32_t sync_address = current_addr;

        // zero initialize all addresses
        auto* base_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(base_address);
        for (uint32_t i = 0; i < (current_addr - base_address) / sizeof(uint32_t); i++) {
            base_ptr[i] = 0;
        }

        return MuxLocalAddresses{
            flow_control_address, teardown_address, buffer_index_address, status_buffer_address, sync_address};
    }

    // Helper to calculate total size needed for one connection
    static constexpr uint32_t size_per_connection(uint32_t address_padding_bytes) {
        constexpr uint32_t num_addresses = sizeof(MuxLocalAddresses) / sizeof(uint32_t);
        return num_addresses * address_padding_bytes;
    }
};

/* ****************************************************************************
 * MuxTerminationManager: Template-based mux termination handler
 *
 * Specializations:
 * - HAS_MUX_CONNECTIONS=false: No-op (not a mux client)
 * - HAS_MUX_CONNECTIONS=true: Runtime master/subordinate role with NUM_MUXES template param
 * ****************************************************************************/
template <bool HAS_MUX_CONNECTIONS, uint8_t NUM_MUXES = 0>
struct MuxTerminationManager;

// Specialization: No mux connections
template <uint8_t NUM_MUXES>
struct MuxTerminationManager<false, NUM_MUXES> {
    MuxTerminationManager(size_t& local_args_idx, uint32_t sync_address) {
        // No args to parse
    }

    FORCE_INLINE void terminate_muxes() {
        // No-op
    }
};

// Specialization: Has mux connections (runtime determines master vs subordinate)
template <uint8_t NUM_MUXES>
struct MuxTerminationManager<true, NUM_MUXES> {
    MuxTerminationManager(size_t& local_args_idx, uint32_t sync_address) {
        is_master_ = get_local_arg_val<uint32_t>(local_args_idx++) != 0;
        total_mux_clients_ = get_local_arg_val<uint32_t>(local_args_idx++);
        uint32_t master_noc_encoding = get_local_arg_val<uint32_t>(local_args_idx++);

        if (is_master_) {
            // Master: setup sync semaphore (should be cleared by host)
            termination_sync_ptr_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_address);

            num_muxes_to_terminate_ = get_local_arg_val<uint32_t>(local_args_idx++);
            ASSERT(num_muxes_to_terminate_ <= NUM_MUXES);

            // Parse mux list (x, y, signal_addr triples)
            for (uint8_t i = 0; i < num_muxes_to_terminate_; i++) {
                mux_x_[i] = get_local_arg_val<uint32_t>(local_args_idx++);
                mux_y_[i] = get_local_arg_val<uint32_t>(local_args_idx++);
                mux_signal_addrs_[i] = get_local_arg_val<uint32_t>(local_args_idx++);
            }
        } else {
            // Subordinate: setup NOC address to master's sync semaphore
            master_noc_addr_ = get_noc_addr_helper(master_noc_encoding, sync_address);
        }
    }

    FORCE_INLINE void terminate_muxes() {
        if (is_master_) {
            // Wait for all subordinates (total_clients - 1, excluding self)
            noc_semaphore_wait(termination_sync_ptr_, total_mux_clients_ - 1);

            // Terminate all muxes in sequence
            for (uint8_t i = 0; i < num_muxes_to_terminate_; i++) {
                tt::tt_fabric::fabric_endpoint_terminate(mux_x_[i], mux_y_[i], mux_signal_addrs_[i]);
            }
        } else {
            // Signal the master
            noc_semaphore_inc(master_noc_addr_, 1);
            noc_async_atomic_barrier();
        }
    }

private:
    bool is_master_ = false;
    uint32_t total_mux_clients_ = 0;

    // Master members:
    volatile tt_l1_ptr uint32_t* termination_sync_ptr_ = nullptr;
    uint8_t num_muxes_to_terminate_ = 0;
    uint8_t mux_x_[NUM_MUXES];
    uint8_t mux_y_[NUM_MUXES];
    uint32_t mux_signal_addrs_[NUM_MUXES];

    // Subordinate members:
    uint64_t master_noc_addr_ = 0;
};

struct SenderKernelMemoryMap {
    static constexpr uint32_t address_padding_bytes = 16;
    // Encapsulated common memory map
    CommonMemoryMap common;

    SenderKernelMemoryMap() {}

    static SenderKernelMemoryMap build_from_args(const CommonMemoryMap& common_map, size_t& rt_args_idx) {
        return SenderKernelMemoryMap(common_map, rt_args_idx);
    }

    uint32_t get_packet_header_address() {
        uint32_t addr = curr_packet_header_address_;
        ASSERT(addr + sizeof(PACKET_HEADER_TYPE) < payload_buffer_region_base_);
        curr_packet_header_address_ += sizeof(PACKET_HEADER_TYPE);
        return addr;
    }

    uint32_t get_payload_buffer_address(uint32_t size) {
        uint32_t addr = curr_payload_buffer_address_;
        ASSERT(addr + size < highest_usable_address_);

        // TODO: ensure noc alignment
        curr_payload_buffer_address_ += size;
        return addr;
    }

    // Mux local address allocation (allocates from cursor, then advances it)
    MuxLocalAddresses get_mux_local_addresses_for_connection() {
        auto addrs = MuxLocalAddresses::allocate_from_base(curr_mux_local_address_, address_padding_bytes);
        curr_mux_local_address_ += MuxLocalAddresses::size_per_connection(address_padding_bytes);
        return addrs;
    }

private:
    SenderKernelMemoryMap(const CommonMemoryMap& common_map, size_t& rt_args_idx) {
        // Use pre-parsed common memory map and parse only sender-specific args
        common = common_map;
        packet_header_region_base_ = get_arg_val<uint32_t>(rt_args_idx++);
        payload_buffer_region_base_ = get_arg_val<uint32_t>(rt_args_idx++);
        highest_usable_address_ = get_arg_val<uint32_t>(rt_args_idx++);

        // set the current addresses to the base
        curr_packet_header_address_ = packet_header_region_base_;
        curr_payload_buffer_address_ = payload_buffer_region_base_;
        curr_mux_local_address_ = common.mux_local_addresses_base;
    }

    uint32_t packet_header_region_base_;
    uint32_t payload_buffer_region_base_;
    uint32_t highest_usable_address_;
    uint32_t curr_packet_header_address_;
    uint32_t curr_payload_buffer_address_;
    uint32_t curr_mux_local_address_;
};

// Receiver kernel memory map - for allocating credit return packet headers and mux local addresses
struct ReceiverKernelMemoryMap {
    static constexpr uint32_t address_padding_bytes = 16;

    // Encapsulated common memory map
    CommonMemoryMap common;

    ReceiverKernelMemoryMap() {}

    static ReceiverKernelMemoryMap build_from_args(const CommonMemoryMap& common_map, size_t& rt_args_idx) {
        return ReceiverKernelMemoryMap(common_map, rt_args_idx);
    }

    uint32_t get_credit_header_address() {
        uint32_t addr = curr_credit_header_address_;
        ASSERT(addr + sizeof(PACKET_HEADER_TYPE) <= credit_header_region_end_);
        curr_credit_header_address_ += sizeof(PACKET_HEADER_TYPE);
        return addr;
    }

    // Mux local address allocation (allocates from cursor, then advances it)
    MuxLocalAddresses get_mux_local_addresses_for_connection() {
        auto addrs = MuxLocalAddresses::allocate_from_base(curr_mux_local_address_, address_padding_bytes);
        curr_mux_local_address_ += MuxLocalAddresses::size_per_connection(address_padding_bytes);
        return addrs;
    }

private:
    ReceiverKernelMemoryMap(const CommonMemoryMap& common_map, size_t& rt_args_idx) {
        // Use pre-parsed common memory map and parse only receiver-specific args
        common = common_map;
        credit_header_region_base_ = get_arg_val<uint32_t>(rt_args_idx++);
        credit_header_region_end_ = get_arg_val<uint32_t>(rt_args_idx++);

        // Set the current address to the base
        curr_credit_header_address_ = credit_header_region_base_;
        curr_mux_local_address_ = common.mux_local_addresses_base;
    }

    uint32_t credit_header_region_base_;   // Start of credit header allocation region
    uint32_t credit_header_region_end_;    // End of credit header allocation region
    uint32_t curr_credit_header_address_;  // Current allocation pointer
    uint32_t curr_mux_local_address_;      // Cursor for allocating mux local addresses
};

/* Layout for the run time args for sender
1. Memory map args (unified: common + sender-specific args parsed together)
2. Fabric connection args
3. Traffic config args
3.1. TrafficConfigCommonFields
3.2. Chip send type fields
3.3. Noc send type fields
*/
template <
    uint8_t NUM_TRAFFIC_CONFIGS,
    bool IS_2D_FABRIC,
    bool LINE_SYNC,
    uint8_t NUM_LOCAL_SYNC_CORES,
    typename EdmSenderT = WorkerToFabricEdmSender>
struct SenderKernelConfig {
    static constexpr bool MASTER_SYNC_CORE = false;

    static SenderKernelConfig build_from_args(
        const CommonMemoryMap& common_map,
        size_t& rt_args_idx,
        size_t& local_args_idx,
        uint8_t num_fabric_connections) {
        return SenderKernelConfig(common_map, rt_args_idx, local_args_idx, num_fabric_connections);
    }

    void open_connections() {
        connections.open_all();
        // Initialize credit management for all traffic configs
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_ptrs[i]->credit_manager_.initialize();
        }
    }

    void local_sync(uint8_t sync_iter) {
        if constexpr (LINE_SYNC) {
            local_sync_config().local_sync(sync_iter);
        }
    }

    void close_connections() {
        // Wait for all credits to be returned before closing
        bool got_all_credits_back = false;
        while (!got_all_credits_back) {
            got_all_credits_back = true;
            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                got_all_credits_back &= traffic_config_ptrs[i]->credit_manager_.got_all_credits_back();
            }
        }
        connections.close_all();
    }

    SenderKernelMemoryMap memory_map;

    FabricConnectionArray<EdmSenderT> connections;

    alignas(LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>)
        std::array<char, sizeof(LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>)> local_sync_config_storage;
    // Per-config connection indices use fixed MAX_MCAST_INJECTIONS strides.
    std::array<uint8_t, NUM_TRAFFIC_CONFIGS> traffic_config_connection_counts;
    std::array<uint8_t, NUM_TRAFFIC_CONFIGS * MAX_MCAST_INJECTIONS> traffic_config_to_fabric_connection_map;

    using TrafficConfigType = SenderKernelTrafficConfig<EdmSenderT>;

    alignas(
        TrafficConfigType) std::array<char, NUM_TRAFFIC_CONFIGS * sizeof(TrafficConfigType)> traffic_configs_storage;
    std::array<TrafficConfigType*, NUM_TRAFFIC_CONFIGS> traffic_config_ptrs;

    // Helper accessors
    LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>& local_sync_config() {
        return *reinterpret_cast<LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>*>(
            local_sync_config_storage.data());
    }

    TrafficConfigType* traffic_configs(uint8_t idx) {
        return reinterpret_cast<TrafficConfigType*>(traffic_configs_storage.data() + idx * sizeof(TrafficConfigType));
    }

    const std::array<TrafficConfigType*, NUM_TRAFFIC_CONFIGS>& traffic_config_ptrs_array() const {
        return traffic_config_ptrs;
    }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return memory_map.common.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return memory_map.common.result_buffer_size; }

private:
    SenderKernelConfig(
        const CommonMemoryMap& common_map,
        size_t& rt_args_idx,
        size_t& local_args_idx,
        uint8_t num_fabric_connections) {
        // Parse memory map args from runtime args using pre-parsed common map
        this->memory_map = SenderKernelMemoryMap::build_from_args(common_map, rt_args_idx);

        // Parse all fabric connections using unified array (memory map needed for mux local addresses)
        connections.num_connections = num_fabric_connections;
        connections.template parse_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx, this->memory_map);

        // add line sync initializations here, for each fabric connection, ex, forward and backward connection, run line
        // sync for all.
        if constexpr (LINE_SYNC) {
            uint32_t sync_address = get_local_arg_val<uint32_t>(local_args_idx++);
            uint32_t sync_val = get_local_arg_val<uint32_t>(local_args_idx++);
            new (&local_sync_config()) LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>(sync_address, sync_val);

            // setup core coordinates
            local_sync_config().setup_core_coordinates(local_args_idx);
        }
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            const uint8_t num_conns = get_local_arg_val<uint32_t>(local_args_idx++);
            ASSERT(num_conns > 0 && num_conns <= MAX_MCAST_INJECTIONS);
            traffic_config_connection_counts[i] = num_conns;
            for (uint8_t j = 0; j < num_conns; j++) {
                traffic_config_to_fabric_connection_map[i * MAX_MCAST_INJECTIONS + j] =
                    get_local_arg_val<uint32_t>(local_args_idx++);
            }
        }

        // Initialize traffic config pointers
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_ptrs[i] = nullptr;
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto metadata = SenderTrafficConfigMetadata::build_from_args(local_args_idx);
            const uint8_t num_conns = traffic_config_connection_counts[i];
            const uint8_t* fabric_connection_indices =
                &traffic_config_to_fabric_connection_map[i * MAX_MCAST_INJECTIONS];
            for (uint8_t j = 0; j < num_conns; j++) {
                ASSERT(fabric_connection_indices[j] < connections.num_connections);
            }

            uint32_t packet_header_address = this->memory_map.get_packet_header_address();

            // Get pointer to pre-allocated storage and initialize with placement new
            TrafficConfigType* config_ptr = traffic_configs(i);
            traffic_config_ptrs[i] = config_ptr;

            // Initialize traffic config with connection array pointer and indices
            new (config_ptr)
                TrafficConfigType(&connections, fabric_connection_indices, num_conns, metadata, packet_header_address);

            traffic_config_ptrs[i]->template parse_and_setup_chip_send_type<IS_2D_FABRIC>(
                local_args_idx, packet_header_address);

            traffic_config_ptrs[i]->parse_and_setup_noc_send_type(local_args_idx);

            // Initialize credit manager (parses credit_management_enabled + SenderCreditInfo)
            traffic_config_ptrs[i]->credit_manager_.init(local_args_idx, metadata.num_packets);

            // the payload buffer size here is the virtual size of the buffer, not the physical size
            // this virtual size is used to keep track of the physical buffer on the receiver side
            // on the sender side, the physical buffer will only be the size of the payload
            uint32_t payload_buffer_size = metadata.payload_buffer_size;
            uint32_t payload_buffer_address =
                this->memory_map.get_payload_buffer_address(traffic_config_ptrs[i]->payload_size_bytes);
            traffic_config_ptrs[i]->setup_payload_buffer(payload_buffer_address, payload_buffer_size);
        }
    };
};

// Helper class to manage credit accumulation and return
// Encapsulates all credit batching logic in one place
// Works with FabricConnectionArray (supports both direct and mux connections)
struct ReceiverCreditManager {
    ReceiverCreditManager() : credit_fields_(0, 0, 0) {}

    template <bool IS_2D_FABRIC>
    void setup_packet_header(size_t& arg_idx, uint32_t packet_header_address) {
        ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, IS_2D_FABRIC>::parse_and_setup(
            arg_idx, packet_header_address, packet_header_);

        credit_fields_ = NocUnicastAtomicIncFields::build_from_args<true>(arg_idx);
        uint64_t noc_addr = get_noc_addr_helper(credit_fields_.dst_noc_encoding, credit_fields_.dst_address);
        packet_header_->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{noc_addr, credit_fields_.atomic_inc_val, CREDIT_RETURN_FLUSH});
    }

    // Initialize with credit info and fabric connection array
    template <bool IS_2D_FABRIC>
    void init(
        size_t& arg_idx, FabricConnectionArray<>* connections, uint8_t connection_idx, uint32_t credit_header_address) {
        connection_manager_ = connections;
        connection_idx_ = connection_idx;
        accumulated_credits_ = 0;
        enabled_ = true;

        // Cache connection pointer during initialization
        if (connection_manager_->is_mux[connection_idx_]) {
            connection_ptr_ = &connection_manager_->get_mux_connection(connection_idx_);
        } else {
            connection_ptr_ = &connection_manager_->get_fabric_connection(connection_idx_);
        }

        packet_header_ = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(credit_header_address);
        setup_packet_header<IS_2D_FABRIC>(arg_idx, credit_header_address);
    }

    // Called after each packet is processed
    FORCE_INLINE void accumulate_and_maybe_send() {
        if (!enabled_) {
            return;
        }

        accumulated_credits_++;

        // Send credits in batches for efficiency
        if (accumulated_credits_ >= credit_fields_.atomic_inc_val) {
            send_credits();
            accumulated_credits_ = 0;
        }
    }

    // Called at end to flush remaining credits
    FORCE_INLINE void flush_remaining() {
        if (enabled_ && accumulated_credits_ > 0) {
            send_credits(accumulated_credits_);
            accumulated_credits_ = 0;
        }
    }

private:
    FORCE_INLINE void send_credits() {
        connection_manager_->wait_for_empty_write_slot<false>(connection_ptr_, connection_idx_);
        connection_manager_->send_header_non_blocking<false>(
            connection_ptr_, connection_idx_, (uint32_t)packet_header_);
    }

    FORCE_INLINE void send_credits(uint32_t num_credits) {
        // flush writes before updating the header to avoid race conditions
        noc_async_writes_flushed();

        uint64_t noc_addr = get_noc_addr_helper(credit_fields_.dst_noc_encoding, credit_fields_.dst_address);
        packet_header_->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{noc_addr, num_credits, CREDIT_RETURN_FLUSH});

        connection_manager_->wait_for_empty_write_slot<false>(connection_ptr_, connection_idx_);
        connection_manager_->send_header_flush_blocking<false>(
            connection_ptr_, connection_idx_, (uint32_t)packet_header_);
    }

    FabricConnectionArray<>* connection_manager_ = nullptr;
    void* connection_ptr_ = nullptr;  // Cached connection pointer
    uint8_t connection_idx_ = 0;
    uint32_t accumulated_credits_ = 0;
    bool enabled_ = false;
    NocUnicastAtomicIncFields credit_fields_;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_;
};

struct ReceiverTrafficConfigMetadata {
    static ReceiverTrafficConfigMetadata build_from_args(size_t& arg_idx) {
        return ReceiverTrafficConfigMetadata(arg_idx);
    }

    ReceiverTrafficConfigMetadata(const ReceiverTrafficConfigMetadata& other) :
        num_packets(other.num_packets),
        seed(other.seed),
        payload_buffer_size(other.payload_buffer_size),
        sfpu_verify(other.sfpu_verify) {}

    uint32_t num_packets = 0;
    uint32_t seed = 0;
    uint32_t payload_buffer_size = 0;
    // The config's payloads are checked on the core's compute engine (tt_fabric_test_payload_verify.cpp) instead of
    // by this kernel.
    bool sfpu_verify = false;

private:
    ReceiverTrafficConfigMetadata(size_t& arg_idx) {
        this->num_packets = get_local_arg_val<uint32_t>(arg_idx++);
        this->seed = get_local_arg_val<uint32_t>(arg_idx++);
        this->payload_buffer_size = get_local_arg_val<uint32_t>(arg_idx++);
        this->sfpu_verify = get_local_arg_val<uint32_t>(arg_idx++) != 0;
    }
};

/*
Semantics for data validation: poll() -> validate() -> advance()
*/
struct TrafficValidationConfigBase {
    using PollFunc = bool (*)(TrafficValidationConfigBase*);
    using ValidateFunc = bool (*)(TrafficValidationConfigBase*);
    using UpdateFunc = void (*)(TrafficValidationConfigBase*);

    struct ValidationOps {
        PollFunc poll;
        ValidateFunc validate;
        UpdateFunc update;
    };

    TrafficValidationConfigBase(const ReceiverTrafficConfigMetadata& metadata) : metadata(metadata) {
        // Function pointers will be set by derived classes
        ops.poll = nullptr;
        ops.validate = nullptr;
        ops.update = nullptr;
    }

    bool has_packets_to_validate() const { return num_packets_processed < metadata.num_packets; }

    bool poll() { return ops.poll(this); }

    bool validate() { return ops.validate(this); }

    void advance() {
        advance_cursor();
        retire();
    }

    // The two halves of advance(), for receivers that verify packets asynchronously: advance_cursor() moves on to
    // the next packet's slot and expected values (so it can be polled), retire() counts a packet and returns its
    // credit once it has been checked. Both happen in packet order per config.
    void advance_cursor() {
        ops.update(this);
        num_packets_issued++;
    }

    void retire() {
        num_packets_processed++;

        // Automatically handle credit return after processing packet
        if (credit_manager_ != nullptr) {
            static_cast<ReceiverCreditManager*>(credit_manager_)->accumulate_and_maybe_send();
        }
    }

    bool has_packets_to_issue() const { return num_packets_issued < metadata.num_packets; }

    ReceiverTrafficConfigMetadata metadata;
    uint32_t num_packets_processed = 0;
    uint32_t num_packets_issued = 0;  // packets whose slot the cursor has moved past (>= num_packets_processed)
    ValidationOps ops;
    // The payload at the current buffer slot as the core's compute engine verifies it (metadata.sfpu_verify): its
    // words (0 when the config has no payload), and the segments its expected words shift at
    // (SFPU_VERIFY_REQ_SEGMENTS: segment s ends at slot word payload_segment_end[s] and expects word w to be
    // seed + w + payload_segment_delta[s]).
    uint32_t payload_words = 0;
    uint32_t num_payload_segments = 0;
    uint32_t payload_segment_end[SFPU_VERIFY_MAX_SEGMENTS] = {};
    uint32_t payload_segment_delta[SFPU_VERIFY_MAX_SEGMENTS] = {};

    void set_contiguous_payload(uint32_t words) {
        payload_words = words;
        num_payload_segments = 1;
        payload_segment_end[0] = words;
        payload_segment_delta[0] = 0;
    }

    // Pointer to credit manager (set by ReceiverKernelConfig during initialization)
    void* credit_manager_ = nullptr;  // Type-erased pointer to ReceiverCreditManager
};

struct AtomicIncValidationConfig : public TrafficValidationConfigBase {
    AtomicIncValidationConfig(
        const NocUnicastAtomicIncFields& atomic_inc_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata) {
        // Set up function pointers
        ops.poll = poll_impl;
        ops.validate = validate_impl;
        ops.update = update_impl;

        poll_address = reinterpret_cast<tt_l1_ptr uint32_t*>(atomic_inc_fields.dst_address);
        value_step_size = atomic_inc_fields.atomic_inc_val;

        // set the initial expected value equal to the step size
        expected_value = value_step_size;
    }

    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<AtomicIncValidationConfig*>(base_config);
        uint32_t current_value = *config->poll_address;
        return current_value >= config->expected_value;
    }

    static bool validate_impl(TrafficValidationConfigBase* base_config) {
        return true;  // no-op for atomic incs
    }

    static void update_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<AtomicIncValidationConfig*>(base_config);
        config->expected_value += config->value_step_size;
    }

    volatile tt_l1_ptr uint32_t* poll_address;
    uint32_t expected_value;
    uint32_t value_step_size;
};

struct WriteValidationConfig : public TrafficValidationConfigBase {
    WriteValidationConfig(const NocUnicastWriteFields& write_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata) {
        // Set up function pointers
        ops.poll = poll_impl;
        ops.validate = validate_impl;
        ops.update = update_impl;

        payload_buffer_ = new (payload_buffer_storage.data()) ReceiverPayloadBuffer(
            write_fields.dst_address, metadata.payload_buffer_size, write_fields.payload_size_bytes);
        set_contiguous_payload(write_fields.payload_size_bytes / sizeof(uint32_t));
    }

    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteValidationConfig*>(base_config);
        return config->payload_buffer_->poll_for_data(config->metadata.seed);
    }

    static bool validate_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteValidationConfig*>(base_config);
        return config->payload_buffer_->validate_data(config->metadata.seed);
    }

    static void update_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteValidationConfig*>(base_config);
        config->metadata.seed = prng_next(config->metadata.seed);
        config->payload_buffer_->advance();
    }

    alignas(ReceiverPayloadBuffer) std::array<char, sizeof(ReceiverPayloadBuffer)> payload_buffer_storage;
    ReceiverPayloadBuffer* payload_buffer_;
};

struct WriteAtomicIncValidationConfig : public TrafficValidationConfigBase {
    WriteAtomicIncValidationConfig(
        const NocUnicastWriteAtomicIncFields& write_atomic_inc_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata) {
        // Set up function pointers
        ops.poll = poll_impl;
        ops.validate = validate_impl;
        ops.update = update_impl;

        const auto& write_fields = write_atomic_inc_fields.write_fields;
        const auto& atomic_fields = write_atomic_inc_fields.atomic_inc_fields;

        payload_buffer_ = new (payload_buffer_storage.data()) ReceiverPayloadBuffer(
            write_fields.dst_address, metadata.payload_buffer_size, write_fields.payload_size_bytes);
        set_contiguous_payload(write_fields.payload_size_bytes / sizeof(uint32_t));

        atomic_inc_address = reinterpret_cast<tt_l1_ptr uint32_t*>(atomic_fields.dst_address);
        atomic_inc_val = atomic_fields.atomic_inc_val;
        expected_atomic_value = atomic_inc_val;
    }

    // Strict: poll() only looks at the semaphore and never waits for the payload. Once the increment for a packet
    // is visible its write must already have landed, so validate() checks it in one pass and an increment that
    // overtakes its write fails as a data mismatch instead of being waited out.
    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteAtomicIncValidationConfig*>(base_config);
        // Also drops any cached payload lines from a previous lap, so validate() reads what is in L1 now.
        invalidate_l1_cache();
        return *config->atomic_inc_address >= config->expected_atomic_value;
    }

    static bool validate_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteAtomicIncValidationConfig*>(base_config);
        return config->payload_buffer_->validate_data(config->metadata.seed);
    }

    static void update_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteAtomicIncValidationConfig*>(base_config);
        config->metadata.seed = prng_next(config->metadata.seed);

        config->expected_atomic_value += config->atomic_inc_val;

        config->payload_buffer_->advance();
    }

    alignas(ReceiverPayloadBuffer) std::array<char, sizeof(ReceiverPayloadBuffer)> payload_buffer_storage;
    ReceiverPayloadBuffer* payload_buffer_;
    volatile tt_l1_ptr uint32_t* atomic_inc_address;
    uint32_t atomic_inc_val;
    uint32_t expected_atomic_value;
};

struct ScatterWriteValidationConfig : public TrafficValidationConfigBase {
    ScatterWriteValidationConfig(
        const NocUnicastScatterWriteFields& scatter_write_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata) {
        // Set up function pointers
        ops.poll = poll_impl;
        ops.validate = validate_impl;
        ops.update = update_impl;

        // Store base addresses and chunk sizes; the last chunk size is implicit (remaining payload)
        chunk_count = scatter_write_fields.chunk_count;
        uint32_t explicit_chunk_bytes = 0;
        for (uint32_t i = 0; i < chunk_count; i++) {
            base_dst_addresses[i] = scatter_write_fields.dst_addresses[i];
            dst_addresses[i] = scatter_write_fields.dst_addresses[i];
            if (i + 1 < chunk_count) {
                chunk_sizes[i] = scatter_write_fields.chunk_sizes[i];
                explicit_chunk_bytes += chunk_sizes[i];
            }
        }
        chunk_sizes[chunk_count - 1] = scatter_write_fields.payload_size_bytes - explicit_chunk_bytes;

        payload_size_bytes = scatter_write_fields.payload_size_bytes;
        current_offset = 0;

        // For the compute verifier: the chunks tile the slot (the host places them in reverse payload order), and
        // each one carries the pattern from its offset in the payload. Segments go in slot order.
        static_assert(NocUnicastScatterWriteFields::MAX_CHUNKS <= SFPU_VERIFY_MAX_SEGMENTS);
        std::array<uint32_t, NocUnicastScatterWriteFields::MAX_CHUNKS> payload_offsets{};
        std::array<uint8_t, NocUnicastScatterWriteFields::MAX_CHUNKS> slot_order{};
        for (uint32_t i = 0; i < chunk_count; i++) {
            payload_offsets[i] = i == 0 ? 0 : payload_offsets[i - 1] + chunk_sizes[i - 1];
            uint32_t j = i;  // insertion sort by slot address
            for (; j > 0 && base_dst_addresses[slot_order[j - 1]] > base_dst_addresses[i]; j--) {
                slot_order[j] = slot_order[j - 1];
            }
            slot_order[j] = static_cast<uint8_t>(i);
        }
        const uint32_t slot_base = base_dst_addresses[slot_order[0]];
        payload_words = payload_size_bytes / sizeof(uint32_t);
        num_payload_segments = chunk_count;
        for (uint32_t s = 0; s < chunk_count; s++) {
            const uint32_t i = slot_order[s];
            const uint32_t slot_offset = base_dst_addresses[i] - slot_base;
            ASSERT(slot_offset == (s == 0 ? 0 : payload_segment_end[s - 1] * sizeof(uint32_t)));
            payload_segment_end[s] = (slot_offset + chunk_sizes[i]) / sizeof(uint32_t);
            payload_segment_delta[s] = payload_offsets[i] / sizeof(uint32_t) - slot_offset / sizeof(uint32_t);
        }
    }

    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<ScatterWriteValidationConfig*>(base_config);

        // Check if all chunks have been written by polling the last word of each chunk
        uint32_t offset = 0;
        for (uint32_t i = 0; i < config->chunk_count; i++) {
            const uint32_t chunk_size = config->chunk_sizes[i];
            if (!SequentialDataPattern::poll(
                    config->dst_addresses[i], chunk_size, config->metadata.seed + offset / sizeof(uint32_t))) {
                return false;
            }
            offset += chunk_size;
        }
        return true;
    }

    static bool validate_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<ScatterWriteValidationConfig*>(base_config);

        // Each chunk carries the pattern from its offset in the payload, wherever it was placed
        uint32_t offset = 0;
        for (uint32_t i = 0; i < config->chunk_count; i++) {
            const uint32_t chunk_size = config->chunk_sizes[i];
            if (!SequentialDataPattern::validate(
                    config->dst_addresses[i], chunk_size, config->metadata.seed + offset / sizeof(uint32_t))) {
                return false;
            }
            offset += chunk_size;
        }
        return true;
    }

    static void update_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<ScatterWriteValidationConfig*>(base_config);
        config->metadata.seed = prng_next(config->metadata.seed);

        // Advance buffer offset (similar to ReceiverPayloadBuffer::advance())
        // Need to check if we have enough space in the buffer for the next payload
        config->current_offset += config->payload_size_bytes;
        if (config->current_offset + config->payload_size_bytes > config->metadata.payload_buffer_size) {
            config->current_offset = 0;  // Wrap around
        }

        // Update all destination addresses based on new offset
        for (uint32_t i = 0; i < config->chunk_count; i++) {
            config->dst_addresses[i] = config->base_dst_addresses[i] + config->current_offset;
        }
    }

    uint32_t chunk_count;
    std::array<uint32_t, NocUnicastScatterWriteFields::MAX_CHUNKS> base_dst_addresses;
    std::array<uint32_t, NocUnicastScatterWriteFields::MAX_CHUNKS> dst_addresses;
    std::array<uint32_t, NocUnicastScatterWriteFields::MAX_CHUNKS> chunk_sizes;  // All chunks, in payload order
    uint32_t payload_size_bytes;
    uint32_t current_offset;
};

// Strict validation for a scatter write whose last chunk is a semaphore increment, same as
// WriteAtomicIncValidationConfig: poll() only looks at the semaphore, and validate() checks every chunk in one pass.
struct ScatterWriteAtomicIncValidationConfig : public ScatterWriteValidationConfig {
    ScatterWriteAtomicIncValidationConfig(
        const NocUnicastScatterWriteFields& scatter_write_fields, const ReceiverTrafficConfigMetadata& metadata) :
        ScatterWriteValidationConfig(scatter_write_fields, metadata) {
        ops.poll = poll_impl;
        ops.update = update_impl;

        atomic_inc_address = reinterpret_cast<tt_l1_ptr uint32_t*>(scatter_write_fields.atomic_inc_dst_address);
        atomic_inc_val = scatter_write_fields.atomic_inc_val;
        expected_atomic_value = atomic_inc_val;
    }

    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<ScatterWriteAtomicIncValidationConfig*>(base_config);
        // Also drops any cached payload lines from a previous lap, so validate() reads what is in L1 now.
        invalidate_l1_cache();
        return *config->atomic_inc_address >= config->expected_atomic_value;
    }

    static void update_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<ScatterWriteAtomicIncValidationConfig*>(base_config);
        ScatterWriteValidationConfig::update_impl(base_config);
        config->expected_atomic_value += config->atomic_inc_val;
    }

    volatile tt_l1_ptr uint32_t* atomic_inc_address;
    uint32_t atomic_inc_val;
    uint32_t expected_atomic_value;
};

/* Layout for the run time args for receiver
1. Memory map args (unified: result buffer only, as receivers don't allocate memory)
2. Traffic config args
2.1. TrafficConfigCommonFields
2.2. Noc send type fields
*/
template <uint8_t NUM_TRAFFIC_CONFIGS, uint8_t NUM_CREDIT_CONNECTIONS, bool IS_2D_FABRIC>
struct ReceiverKernelConfig {
    static ReceiverKernelConfig build_from_args(
        const CommonMemoryMap& common_map, size_t& rt_args_idx, size_t& local_args_idx) {
        return ReceiverKernelConfig(common_map, rt_args_idx, local_args_idx);
    }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return memory_map.common.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return memory_map.common.result_buffer_size; }

    // Traffic config accessor
    TrafficValidationConfigBase** traffic_configs() { return traffic_configs_.data(); }

    // Credit connection lifecycle methods
    void open_credit_connections() { credit_connections.open_all(); }

    void close_credit_connections() {
        // Automatically flush any remaining credits before closing
        flush_remaining_credits();
        credit_connections.close_all();
    }

private:
    // Flush any remaining accumulated credits (called automatically by close_credit_connections)
    void flush_remaining_credits() {
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            credit_managers_[i].flush_remaining();
        }
    }

    ReceiverKernelMemoryMap memory_map;
    FabricConnectionArray<> credit_connections;
    std::array<uint8_t, NUM_TRAFFIC_CONFIGS> traffic_config_to_credit_connection_map;

    // Credit managers - one per traffic config
    std::array<ReceiverCreditManager, NUM_TRAFFIC_CONFIGS> credit_managers_;

    constexpr static size_t MAX_VALIDATION_CONFIG_SIZE = std::max(
        {sizeof(WriteValidationConfig),
         sizeof(AtomicIncValidationConfig),
         sizeof(WriteAtomicIncValidationConfig),
         sizeof(ScatterWriteValidationConfig),
         sizeof(ScatterWriteAtomicIncValidationConfig)});

    alignas(TrafficValidationConfigBase)
        std::array<char, NUM_TRAFFIC_CONFIGS * MAX_VALIDATION_CONFIG_SIZE> validation_configs_storage;
    std::array<TrafficValidationConfigBase*, NUM_TRAFFIC_CONFIGS> traffic_configs_;

private:
    ReceiverKernelConfig(const CommonMemoryMap& common_map, size_t& rt_args_idx, size_t& local_args_idx) {
        // Parse receiver-specific memory map (includes credit header region)
        this->memory_map = ReceiverKernelMemoryMap::build_from_args(common_map, rt_args_idx);

        // Parse credit connections from runtime args (memory map needed for mux local addresses)
        credit_connections.num_connections = NUM_CREDIT_CONNECTIONS;
        credit_connections.parse_from_args(rt_args_idx, this->memory_map);

        // Parse traffic config to credit connection mapping
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_to_credit_connection_map[i] = get_arg_val<uint32_t>(rt_args_idx++);
        }

        // Parse traffic configs from local args (local_args_idx passed from caller)
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_configs_[i] = nullptr;
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            const auto metadata = ReceiverTrafficConfigMetadata::build_from_args(local_args_idx);
            NocSendType noc_send_type = static_cast<NocSendType>(get_local_arg_val<uint32_t>(local_args_idx++));

            // Get pointer to pre-allocated storage for this config
            char* config_storage = validation_configs_storage.data() + i * MAX_VALIDATION_CONFIG_SIZE;

            if (noc_send_type == NocSendType::NOC_UNICAST_WRITE) {
                const auto write_fields = NocUnicastWriteFields::build_from_args<false>(local_args_idx);
                traffic_configs_[i] = new (config_storage) WriteValidationConfig(write_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_UNICAST_ATOMIC_INC) {
                const auto atomic_inc_fields = NocUnicastAtomicIncFields::build_from_args<false>(local_args_idx);
                traffic_configs_[i] = new (config_storage) AtomicIncValidationConfig(atomic_inc_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
                const auto write_atomic_inc_fields =
                    NocUnicastWriteAtomicIncFields::build_from_args<false>(local_args_idx);
                traffic_configs_[i] =
                    new (config_storage) WriteAtomicIncValidationConfig(write_atomic_inc_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_UNICAST_SCATTER_WRITE) {
                const auto scatter_write_fields = NocUnicastScatterWriteFields::build_from_args<false>(local_args_idx);
                if (scatter_write_fields.has_atomic_inc) {
                    traffic_configs_[i] =
                        new (config_storage) ScatterWriteAtomicIncValidationConfig(scatter_write_fields, metadata);
                } else {
                    traffic_configs_[i] =
                        new (config_storage) ScatterWriteValidationConfig(scatter_write_fields, metadata);
                }
            } else {
                ASSERT(false);
            }

            // First parse the presence flag, then conditionally parse the data
            bool has_credit_info = get_local_arg_val<uint32_t>(local_args_idx++) != 0;

            // Initialize credit manager for this traffic config
            if (has_credit_info) {
                // Allocate space for pre-built credit return header using memory map
                const uint32_t credit_header_address = this->memory_map.get_credit_header_address();
                const uint8_t connection_idx = traffic_config_to_credit_connection_map[i];
                credit_managers_[i].template init<IS_2D_FABRIC>(
                    local_args_idx, &credit_connections, connection_idx, credit_header_address);

                // Link the credit manager to this traffic config so advance() can call it automatically
                traffic_configs_[i]->credit_manager_ = &credit_managers_[i];
            }
        }
    }
};

/* ********************
 * SyncKernelConfig   *
 **********************/
template <
    uint8_t NUM_SYNC_FABRIC_CONNECTIONS,
    bool IS_2D_FABRIC,
    uint8_t NUM_LOCAL_SYNC_CORES,
    bool USE_UNICAST_SYNC_PACKETS,
    uint8_t NUM_SYNC_CONFIGS>
struct SyncKernelConfig {
    static SyncKernelConfig build_from_args(
        const CommonMemoryMap& common_map, size_t& rt_args_idx, size_t& local_args_idx) {
        return SyncKernelConfig(common_map, rt_args_idx, local_args_idx);
    }

    void global_sync(uint8_t sync_iter) {
        // Open all sync connections
        sync_connections.open_all();

        // Send sync start packets
        for (uint8_t i = 0; i < NUM_SYNC_CONFIGS; i++) {
            line_sync_configs()[i].global_sync_start();
        }

        // Wait for acks (only need one config to check)
        line_sync_configs()[0].global_sync_finish(sync_iter);

        // Close all sync connections
        sync_connections.close_all();
    }

    void local_sync(uint8_t sync_iter) { local_sync_config().local_sync(sync_iter); }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return memory_map.common.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return memory_map.common.result_buffer_size; }

    SenderKernelMemoryMap memory_map;

    FabricConnectionArray<> sync_connections;

    using LineSyncConfigType = LineSyncConfig<>;
    alignas(
        LineSyncConfigType) std::array<char, NUM_SYNC_CONFIGS * sizeof(LineSyncConfigType)> line_sync_configs_storage;
    alignas(LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>)
        std::array<char, sizeof(LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>)> local_sync_config_storage;

    // Per-config connection indices use fixed MAX_MCAST_INJECTIONS strides.
    std::array<uint8_t, NUM_SYNC_CONFIGS> sync_config_connection_counts;
    std::array<uint8_t, NUM_SYNC_CONFIGS * MAX_MCAST_INJECTIONS> sync_config_to_fabric_connection_map;

    // Helper accessors
    LineSyncConfigType* line_sync_configs() {
        return reinterpret_cast<LineSyncConfigType*>(line_sync_configs_storage.data());
    }
    LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>& local_sync_config() {
        return *reinterpret_cast<LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>*>(local_sync_config_storage.data());
    }

private:
    SyncKernelConfig(const CommonMemoryMap& common_map, size_t& rt_args_idx, size_t& local_args_idx) {
        // Parse memory map args from runtime args using pre-parsed common map
        this->memory_map = SenderKernelMemoryMap::build_from_args(common_map, rt_args_idx);

        // Parse all sync connections using unified array (memory map needed for mux local addresses)
        sync_connections.num_connections = NUM_SYNC_FABRIC_CONNECTIONS;
        sync_connections.parse_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx, this->memory_map);

        // Initialize line sync configurations with connection array
        uint32_t line_sync_val = get_local_arg_val<uint32_t>(local_args_idx++);

        // Parse sync config to fabric connection mapping (same pattern as sender traffic configs)
        for (uint8_t i = 0; i < NUM_SYNC_CONFIGS; i++) {
            const uint8_t num_conns = get_local_arg_val<uint32_t>(local_args_idx++);
            ASSERT(num_conns > 0 && num_conns <= MAX_MCAST_INJECTIONS);
            sync_config_connection_counts[i] = num_conns;
            for (uint8_t j = 0; j < num_conns; j++) {
                sync_config_to_fabric_connection_map[i * MAX_MCAST_INJECTIONS + j] =
                    get_local_arg_val<uint32_t>(local_args_idx++);
            }
        }

        for (uint8_t i = 0; i < NUM_SYNC_CONFIGS; i++) {
            uint32_t packet_header_address = this->memory_map.get_packet_header_address();
            const uint8_t num_conns = sync_config_connection_counts[i];
            const uint8_t* connection_indices = &sync_config_to_fabric_connection_map[i * MAX_MCAST_INJECTIONS];
            for (uint8_t j = 0; j < num_conns; j++) {
                ASSERT(connection_indices[j] < sync_connections.num_connections);
            }
            new (&line_sync_configs()[i]) LineSyncConfigType(
                &sync_connections, connection_indices, num_conns, packet_header_address, line_sync_val);

            // setup packet header fields
            constexpr ChipSendType CHIP_SEND_TYPE =
                USE_UNICAST_SYNC_PACKETS ? ChipSendType::CHIP_UNICAST : ChipSendType::CHIP_MULTICAST;
            line_sync_configs()[i].template setup_packet_header<IS_2D_FABRIC, CHIP_SEND_TYPE>(
                local_args_idx, packet_header_address);
        }

        // Initialize local sync config
        uint32_t sync_address = get_local_arg_val<uint32_t>(local_args_idx++);
        uint32_t sync_val = get_local_arg_val<uint32_t>(local_args_idx++);
        new (&local_sync_config()) LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>(sync_address, sync_val);

        // setup core coordinates
        local_sync_config().setup_core_coordinates(local_args_idx);
    }
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
