// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "internal/risc_attribs.h"
#include "tt_metal/api/tt-metalium/experimental/dispatch_telemetry_types.hpp"

#include <cstddef>
#include <cstdint>

template <typename Telemetry, uint32_t telemetry_addr>
FORCE_INLINE volatile tt_l1_ptr Telemetry* get_telemetry_ptr() {
    return reinterpret_cast<volatile tt_l1_ptr Telemetry*>(telemetry_addr);
}

template <typename Telemetry, uint32_t telemetry_addr, bool enabled>
FORCE_INLINE void init_telemetry() {
    Telemetry telemetry{};
    if constexpr (!enabled) {
        // If disabled, invalidate the signature. This prevents reading stale telemetry values.
        telemetry.signature = 0;
    }
    write_to_l1(telemetry_addr, telemetry);
}

template <typename Telemetry, uint32_t telemetry_addr, bool enabled>
class TelemetryBlockGuardImpl;

template <typename Telemetry, uint32_t telemetry_addr>
class TelemetryBlockGuardImpl<Telemetry, telemetry_addr, true> {
public:
    FORCE_INLINE explicit TelemetryBlockGuardImpl() :
        telemetry_(get_telemetry_ptr<Telemetry, telemetry_addr>()) {
            telemetry_->blocked_by_host_count++;
        }

    FORCE_INLINE ~TelemetryBlockGuardImpl() {
        telemetry_->unblocked_by_host_count++;
    }

    TelemetryBlockGuardImpl(const TelemetryBlockGuardImpl&) = delete;
    TelemetryBlockGuardImpl& operator=(const TelemetryBlockGuardImpl&) = delete;
    TelemetryBlockGuardImpl(TelemetryBlockGuardImpl&& other) = delete;
    TelemetryBlockGuardImpl& operator=(TelemetryBlockGuardImpl&& other) = delete;

private:
    volatile tt_l1_ptr Telemetry* telemetry_;
};

// Designed to be no-op if telemetry is disabled
template <typename Telemetry, uint32_t telemetry_addr>
class TelemetryBlockGuardImpl<Telemetry, telemetry_addr, false> {
public:
    FORCE_INLINE explicit TelemetryBlockGuardImpl() {}
    FORCE_INLINE ~TelemetryBlockGuardImpl() = default;
};

template <typename Telemetry, uint32_t telemetry_addr, bool enabled>
using TelemetryBlockGuard = TelemetryBlockGuardImpl<Telemetry, telemetry_addr, enabled>;
