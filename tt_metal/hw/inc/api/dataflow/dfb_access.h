// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Access pattern of one side of a DFB. Shared by the host (DataflowBufferConfig), the Quasar
// device code and the generated DFBBindingToken on every arch. UNKNOWN = built from a raw id, no token.
namespace dfb {
enum AccessPattern : uint8_t {
    STRIDED,
    ALL,
    BLOCKED,
    UNKNOWN,
};
}  // namespace dfb

// Quasar: DataflowBuffer is a template on the pattern pair. WH/BH: a plain class (patterns
// change nothing there). The macros let one class body serve both.
#ifdef ARCH_QUASAR
template <dfb::AccessPattern Pap = dfb::AccessPattern::UNKNOWN, dfb::AccessPattern Cap = dfb::AccessPattern::UNKNOWN>
class DataflowBuffer;
#define DFB_TEMPLATE_DECL template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
#define DFB_CLASS DataflowBuffer<Pap, Cap>
using DataflowBufferAnyPattern = DataflowBuffer<>;
#else
class DataflowBuffer;
#define DFB_TEMPLATE_DECL
#define DFB_CLASS DataflowBuffer
using DataflowBufferAnyPattern = DataflowBuffer;
#endif

// "Is T a DataflowBuffer?" for any pattern pair, on any arch.
template <typename T>
inline constexpr bool is_dataflow_buffer_v = false;
#ifdef ARCH_QUASAR
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline constexpr bool is_dataflow_buffer_v<DataflowBuffer<Pap, Cap>> = true;
#else
template <>
inline constexpr bool is_dataflow_buffer_v<DataflowBuffer> = true;
#endif
