// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Access pattern of one side of a DFB, carried by the generated DFBBindingToken. Same numbering
// as dfb::AccessPattern. UNKNOWN = built from a raw id, no token.
enum class DFBAccess : uint8_t { STRIDED = 0, ALL = 1, BLOCKED = 2, UNKNOWN = 3 };

// Quasar: DataflowBuffer is a template on the pattern pair. WH/BH: a plain class (patterns
// change nothing there). The macros let one class body serve both.
#ifdef ARCH_QUASAR
template <DFBAccess Pap = DFBAccess::UNKNOWN, DFBAccess Cap = DFBAccess::UNKNOWN>
class DataflowBuffer;
#define DFB_TEMPLATE_DECL template <DFBAccess Pap, DFBAccess Cap>
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
template <DFBAccess Pap, DFBAccess Cap>
inline constexpr bool is_dataflow_buffer_v<DataflowBuffer<Pap, Cap>> = true;
#else
template <>
inline constexpr bool is_dataflow_buffer_v<DataflowBuffer> = true;
#endif
