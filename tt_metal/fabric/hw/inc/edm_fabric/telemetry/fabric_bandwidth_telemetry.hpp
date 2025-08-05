// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/ethernet/tt_eth_api.h"
#include "tt_metal/hw/inc/risc_attribs.h"

#include <cstdint>
#include <cstddef>

/**
 * @brief Timestamp structure with 64-bit value accessible as full or split into high/low 32-bit parts
 */
struct RiscTimestamp {
    union {
        uint64_t full;
        struct {
            uint32_t lo;
            uint32_t hi;
        };
    };
};

/**
 * @brief Enumeration of performance telemetry recorder types
 */
enum class PerfTelemetryRecorderType : uint8_t {
    NONE = 0,
    LOW_RESOLUTION_BANDWIDTH = 1,
};

/**
 * @brief Low resolution bandwidth telemetry data structure
 *
 * This is expected to be dumped after completing the inner loop
 * and is expected to be stored in local memory.
 */
struct LowResolutionBandwidthTelemetry {
    RiscTimestamp timestamp_start;
    RiscTimestamp timestamp_end;
    uint32_t num_words_sent;
    uint32_t num_packets_sent;
};

/**
 * @brief L1 buffer wrapper for storing telemetry data
 */
struct L1PerfTelemetrySingleBuffer {
    volatile tt_l1_ptr uint32_t* buffer_ptr;

    /**
     * @brief Constructs a telemetry buffer with the given pointer
     * @param ptr Pointer to the L1 memory buffer, must be 16-byte aligned
     */
    L1PerfTelemetrySingleBuffer(uint32_t* ptr) : buffer_ptr(ptr) {}

    /**
     * @brief Returns the buffer pointer
     * @return Pointer to the L1 telemetry buffer
     */
    FORCE_INLINE volatile tt_l1_ptr uint32_t* get_ptr() const { return buffer_ptr; }
};

/**
 * @brief Default template function to check if an event was captured in the last capture window
 *        This helper is useful for filtering out idle time from reported data.
 * @param perf_telemetry_collector Reference to the telemetry collector
 * @return Always returns false for default case
 */
template <typename PERF_TELEMETRY_COLLECTOR>
bool captured_an_event(PERF_TELEMETRY_COLLECTOR& perf_telemetry_collector) {
    return false;
}

/**
 * @brief Specialized function to check if a bandwidth telemetry event was captured
 * @param perf_telemetry_collector Reference to the bandwidth telemetry collector
 * @return True if packets were sent, false otherwise
 */
template <>
bool captured_an_event<LowResolutionBandwidthTelemetry>(LowResolutionBandwidthTelemetry& perf_telemetry_collector) {
    return perf_telemetry_collector.num_packets_sent > 0;
}

/**
 * @brief Default template function to open a performance recording window.
 *        Does nothing.
 * @param perf_telemetry_collector Reference to the telemetry collector
 */
template <typename PERF_TELEMETRY_COLLECTOR>
FORCE_INLINE void open_perf_recording_window(PERF_TELEMETRY_COLLECTOR& perf_telemetry_collector) {
    // do nothing as default behaviour
}

/**
 * @brief Specialized function to open a bandwidth telemetry recording window
 * @param perf_telemetry_collector Reference to the bandwidth telemetry collector
 */
template <>
FORCE_INLINE void open_perf_recording_window<LowResolutionBandwidthTelemetry>(
    LowResolutionBandwidthTelemetry& perf_telemetry_collector) {
    perf_telemetry_collector.timestamp_start.full = eth_read_wall_clock();
    perf_telemetry_collector.num_words_sent = 0;
    perf_telemetry_collector.num_packets_sent = 0;
}

/**
 * @brief Default template function to close a performance recording window
 * @param perf_telemetry_collector Reference to the telemetry collector
 */
template <typename PERF_TELEMETRY_COLLECTOR>
FORCE_INLINE void close_perf_recording_window(PERF_TELEMETRY_COLLECTOR& perf_telemetry_collector) {
    // do nothing as default behaviour
}

/**
 * @brief Specialized function to close a bandwidth telemetry recording window
 * @param perf_telemetry_collector Reference to the bandwidth telemetry collector
 */
template <>
FORCE_INLINE void close_perf_recording_window<LowResolutionBandwidthTelemetry>(
    LowResolutionBandwidthTelemetry& perf_telemetry_collector) {
    perf_telemetry_collector.timestamp_end.full = eth_read_wall_clock();
}

/**
 * @brief Default template function to clear a telemetry buffer
 * @param buffer Reference to the telemetry buffer
 */
template <typename BUFFER_TYPE>
FORCE_INLINE void clear_telemetry_buffer(BUFFER_TYPE& buffer) {
    // do nothing as default behaviour
}

/**
 * @brief Specialized function to clear an L1 telemetry buffer
 * @param buffer Reference to the L1 telemetry buffer
 */
template <>
FORCE_INLINE void clear_telemetry_buffer<L1PerfTelemetrySingleBuffer>(L1PerfTelemetrySingleBuffer& buffer) {
    volatile tt_l1_ptr auto* buffer_ptr = buffer.get_ptr();
    constexpr size_t num_words_to_clear = sizeof(LowResolutionBandwidthTelemetry) / sizeof(size_t);
    static_assert(sizeof(LowResolutionBandwidthTelemetry) > sizeof(size_t), "L1PerfTelemetrySingleBuffer is too small");
    for (size_t i = 0; i < num_words_to_clear; i++) {
        buffer_ptr[i] = 0;
    }
}

/**
 * @brief Default template function to write performance recording window results
 * @param perf_telemetry_collector Reference to the telemetry collector
 * @param buffer Reference to the telemetry buffer
 */
template <typename PERF_TELEMETRY_COLLECTOR, typename BUFFER_TYPE>
FORCE_INLINE void write_perf_recording_window_results(
    PERF_TELEMETRY_COLLECTOR& perf_telemetry_collector, BUFFER_TYPE& buffer) {
    // do nothing as default behaviour
}

/**
 * @brief Specialized function to write bandwidth telemetry results to L1 buffer
 * @param perf_telemetry_collector Reference to the bandwidth telemetry collector
 * @param buffer Reference to the L1 telemetry buffer
 */
template <>
FORCE_INLINE void write_perf_recording_window_results<LowResolutionBandwidthTelemetry, L1PerfTelemetrySingleBuffer>(
    LowResolutionBandwidthTelemetry& perf_telemetry_collector, L1PerfTelemetrySingleBuffer& buffer) {
    volatile tt_l1_ptr auto* buffer_ptr = buffer.get_ptr();
    auto num_words_sent_l1_copy = buffer_ptr[4];
    auto num_packets_sent_l1_copy = buffer_ptr[5];
    uint64_t total_cycles = (static_cast<uint64_t>(buffer_ptr[1]) << 32) | buffer_ptr[0];
    auto added_cycles =
        total_cycles + (perf_telemetry_collector.timestamp_end.full - perf_telemetry_collector.timestamp_start.full);

    // For the time being, this telemetry mode records elapsed elapsed (busy) cycles, not start/end, because
    // we want to exclude large idle times from the bandwidth calculation
    buffer_ptr[0] = added_cycles & 0xFFFFFFFF;
    buffer_ptr[1] = added_cycles >> 32;
    // Skip end timstamp
    buffer_ptr[4] = num_words_sent_l1_copy + perf_telemetry_collector.num_words_sent;
    buffer_ptr[5] = num_packets_sent_l1_copy + perf_telemetry_collector.num_packets_sent;
}

/**
 * @brief Default template function to record a packet send event
 * @param perf_telemetry_collector Reference to the telemetry collector
 * @param channel_idx Sender channel index of the channel (unused in this implementation)
 * @param packet_size_bytes Size of the packet in bytes
 */
template <typename PERF_TELEMETRY_COLLECTOR>
FORCE_INLINE void record_packet_send(
    PERF_TELEMETRY_COLLECTOR& perf_telemetry_collector, size_t channel_idx, size_t packet_size_bytes) {
    // do nothing as default behaviour
}

/**
 * @brief Specialized function to record a packet send event for bandwidth telemetry
 * @param perf_telemetry_collector Reference to the bandwidth telemetry collector
 * @param channel_idx Sender channel index of the channel (unused in this implementation)
 * @param packet_size_bytes Size of the packet in bytes
 */
template <>
FORCE_INLINE void record_packet_send<LowResolutionBandwidthTelemetry>(
    LowResolutionBandwidthTelemetry& perf_telemetry_collector, size_t channel_idx, size_t packet_size_bytes) {
    // do nothing as default behaviour
    perf_telemetry_collector.num_packets_sent++;
    perf_telemetry_collector.num_words_sent += bytes_to_eth_words_truncated(packet_size_bytes);
}

/**
 * @brief Default template function to record packet receiver forward event
 * @param perf_telemetry_collector Reference to the telemetry collector
 * @param channel_idx receiver channel index the event is captured from
 * @param packet_size_bytes Size of the packet in bytes
 */
template <typename PERF_TELEMETRY_COLLECTOR>
FORCE_INLINE void record_packet_receiver_forward(
    PERF_TELEMETRY_COLLECTOR& perf_telemetry_collector, size_t channel_idx, size_t packet_size_bytes) {}

/**
 * @brief Default template function to record packet receiver write to NOC event
 * @param perf_telemetry_collector Reference to the telemetry collector
 * @param channel_idx receiver channel index the event is captured from
 * @param packet_size_bytes Size of the packet in bytes
 */
template <typename PERF_TELEMETRY_COLLECTOR>
FORCE_INLINE void record_packet_receiver_write_to_noc(
    PERF_TELEMETRY_COLLECTOR& perf_telemetry_collector, size_t channel_idx, size_t packet_size_bytes) {}

/**
 * @brief Template function to build a performance telemetry recorder for NONE type
 * @return Always returns false for NONE type
 */
template <PerfTelemetryRecorderType PERF_TELEMETRY_RECORDER_TYPE>
std::enable_if_t<PERF_TELEMETRY_RECORDER_TYPE == PerfTelemetryRecorderType::NONE, bool>
build_perf_telemetry_recorder() {
    return false;
}

/**
 * @brief Template function to build a low resolution bandwidth telemetry recorder
 * @return Initialized LowResolutionBandwidthTelemetry structure
 */
template <PerfTelemetryRecorderType PERF_TELEMETRY_RECORDER_TYPE>
std::enable_if_t<
    PERF_TELEMETRY_RECORDER_TYPE == PerfTelemetryRecorderType::LOW_RESOLUTION_BANDWIDTH,
    LowResolutionBandwidthTelemetry>
build_perf_telemetry_recorder() {
    auto local_perf_telemetry_recorder = LowResolutionBandwidthTelemetry();
    local_perf_telemetry_recorder.timestamp_start.full = 0;
    local_perf_telemetry_recorder.timestamp_end.full = 0;
    local_perf_telemetry_recorder.num_words_sent = 0;
    local_perf_telemetry_recorder.num_packets_sent = 0;
    return local_perf_telemetry_recorder;
}

/**
 * @brief Builds a performance telemetry buffer and initializes it
 * @param perf_telemetry_buffer_addr Pointer to the telemetry buffer address
 * @return Initialized L1PerfTelemetrySingleBuffer with cleared contents
 */
L1PerfTelemetrySingleBuffer build_perf_telemetry_buffer(uint32_t* perf_telemetry_buffer_addr) {
    auto local_perf_telemetry_buffer = L1PerfTelemetrySingleBuffer(perf_telemetry_buffer_addr);
    clear_telemetry_buffer(local_perf_telemetry_buffer);
    return local_perf_telemetry_buffer;
}
