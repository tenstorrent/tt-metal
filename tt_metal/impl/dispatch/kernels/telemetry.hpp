// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "internal/risc_attribs.h"
#include "api/debug/assert.h"

#include "hostdevcommon/dispatch_telemetry_types.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

template <typename T>
FORCE_INLINE volatile tt_l1_ptr T* write_to_l1(uint32_t dst_addr, const T& src_object);

template <typename Telemetry, uint32_t telemetry_addr, bool enabled>
FORCE_INLINE void init_telemetry() {
    Telemetry telemetry{};
    if constexpr (!enabled) {
        // If disabled, invalidate the signature. This prevents reading stale telemetry values.
        telemetry.signature = tt::tt_metal::INVALID_TELEMETRY_SIGNATURE;
    }
    write_to_l1(telemetry_addr, telemetry);
}

template <
    uint32_t blocked_count_store_addr,
    uint32_t unblocked_count_store_addr,
    uint32_t* local_blocked_counter,
    bool enabled>
class TelemetryBlockGuard;

// Note: Uses single local block counter to keep L1 block and unblock counters in lockstep.
template <uint32_t blocked_count_store_addr, uint32_t unblocked_count_store_addr, uint32_t* local_blocked_counter>
class TelemetryBlockGuard<blocked_count_store_addr, unblocked_count_store_addr, local_blocked_counter, true> {
public:
    FORCE_INLINE explicit TelemetryBlockGuard() {
        if (local_blocked_counter == nullptr) {
            ASSERT(0);
            return;
        }
        auto* blocked_count_addr_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blocked_count_store_addr);
        *blocked_count_addr_ = ++(*local_blocked_counter);
    }

    FORCE_INLINE ~TelemetryBlockGuard() {
        if (local_blocked_counter == nullptr) {
            ASSERT(0);
            return;
        }
        auto* unblocked_count_addr_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(unblocked_count_store_addr);
        *unblocked_count_addr_ = *local_blocked_counter;
    }

    TelemetryBlockGuard(const TelemetryBlockGuard&) = delete;
    TelemetryBlockGuard& operator=(const TelemetryBlockGuard&) = delete;
    TelemetryBlockGuard(TelemetryBlockGuard&& other) = delete;
    TelemetryBlockGuard& operator=(TelemetryBlockGuard&& other) = delete;
};

// Designed to be no-op if telemetry is disabled
template <uint32_t blocked_count_store_addr, uint32_t unblocked_count_store_addr, uint32_t* local_blocked_counter>
class TelemetryBlockGuard<blocked_count_store_addr, unblocked_count_store_addr, local_blocked_counter, false> {
public:
    FORCE_INLINE explicit TelemetryBlockGuard() = default;
};

using NoTelemetryBlockGuard = TelemetryBlockGuard<0, 0, nullptr, false>;

// Allows compile time checks when passing as template parameter
template <typename T>
struct is_telemetry_block_guard : std::false_type {};
template <uint32_t a, uint32_t b, uint32_t* c, bool d>
struct is_telemetry_block_guard<TelemetryBlockGuard<a, b, c, d>> : std::true_type {};
