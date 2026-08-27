// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compile_time_args.h"

// Fabric-only diagnostic ("debug") router build.
//
// When fabric_debug is false, none of the diagnostic code is compiled in, so the router binary and its
// L1 layout are identical to a normal build (apples-to-apples). The named args are emitted only for the
// router kernel (see FabricEriscDatamoverBuilder::get_telemetry_compile_time_args); these reads are kept
// out of the shared fabric_erisc_router_ct_args.hpp so other kernels that include it (e.g. the mux) are
// not forced to supply the args.
//
// The debug buffer is allocated LAST, off the top of erisc L1 (see erisc_datamover_builder.cpp), so
// every other structure keeps an identical address whether debug is on or off.

namespace tt::tt_fabric {

constexpr bool fabric_debug = static_cast<bool>(get_named_compile_time_arg_val("ENABLE_FABRIC_DEBUG"));
constexpr uint32_t fabric_debug_buffer_addr =
    static_cast<uint32_t>(get_named_compile_time_arg_val("FABRIC_DEBUG_BUFFER_ADDR"));

// Written to debug_buffer[0] at router startup so the host can confirm the diagnostic router is active.
constexpr uint32_t FABRIC_DEBUG_ACTIVE_SENTINEL = 0xDB600000;

}  // namespace tt::tt_fabric
