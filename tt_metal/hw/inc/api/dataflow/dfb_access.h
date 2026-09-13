// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// A binding's producer / consumer access pattern, baked into DFBBindingToken at compile time by
// the generated bindings header. Same numbering as dfb::AccessPattern on Quasar; arch-neutral here
// so Gen1 (WH/BH) kernels compile against it too. UNKNOWN is the pattern of a DataflowBuffer built
// from a raw id (legacy constructor) rather than a binding token.
enum class DFBAccess : uint8_t { STRIDED = 0, ALL = 1, BLOCKED = 2, UNKNOWN = 3 };

// On Quasar the DataflowBuffer class is specialized on the pattern pair at compile time. On
// tt-1xx it stays a plain class: there a DFB is a circular buffer and the patterns change nothing.
// The two macros let one class body serve both shapes.
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

// True for a DataflowBuffer of any pattern pair -- the arch-neutral way to ask "is this a DFB?"
// now that the Quasar type carries template arguments.
template <typename T>
inline constexpr bool is_dataflow_buffer_v = false;
#ifdef ARCH_QUASAR
template <DFBAccess Pap, DFBAccess Cap>
inline constexpr bool is_dataflow_buffer_v<DataflowBuffer<Pap, Cap>> = true;
#else
template <>
inline constexpr bool is_dataflow_buffer_v<DataflowBuffer> = true;
#endif
