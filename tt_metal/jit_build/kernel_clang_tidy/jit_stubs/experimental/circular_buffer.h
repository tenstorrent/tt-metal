// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// clang-tidy stub for experimental/circular_buffer.h
//
// The real circular_buffer.h gates:
//   • The CircularBuffer method bodies (reserve_back, push_back, etc.) and the
//     noc_traits_t<CircularBuffer> specialisation behind #ifndef COMPILE_FOR_TRISC.
//   • LLK-backed implementations (PACK/UNPACK macros) behind #ifdef COMPILE_FOR_TRISC.
//
// Because the kernel_clang_tidy build defines COMPILE_FOR_TRISC globally (to
// satisfy compute kernel includes), the noc_traits_t<CircularBuffer>
// specialisation is stripped out for ALL translation units — including dataflow
// kernel writer_unary.cpp which calls noc.async_write(cb, ...) and legitimately
// needs it.  Without the specialisation the primary template fires:
//
//   static_assert(sizeof(T) == 0, "NoC transactions are not supported for this type");
//
// The fix must be context-aware:
//
//   DATAFLOW kernel TUs (e.g. writer_unary.cpp):
//     Include api/dataflow/dataflow_api.h BEFORE this header.  Our
//     jit_stubs/api/dataflow/dataflow_api.h stub sets KCT_REG_READ_SHIM_DEFINED
//     as a sentinel.  When that sentinel is present we temporarily undefine
//     COMPILE_FOR_TRISC so the real header takes the !COMPILE_FOR_TRISC branch
//     and emits the noc_traits_t<CircularBuffer> specialisation.
//     api/dataflow/dataflow_api.h is already #pragma once-locked at this point,
//     so re-including it through experimental/noc.h is a no-op — no duplicate
//     definitions occur.
//
//   COMPUTE (TRISC) kernel TUs (e.g. eltwise_binary.cpp):
//     Do NOT include api/dataflow/dataflow_api.h before this header.
//     KCT_REG_READ_SHIM_DEFINED is therefore NOT defined when we arrive here.
//     We include the real header with COMPILE_FOR_TRISC intact so it takes the
//     LLK code path — no NOC/dataflow headers are pulled in, avoiding duplicate
//     definitions with api/compute/common.h and api/compute/cb_api.h.

#pragma once

#ifdef KCT_REG_READ_SHIM_DEFINED
// ── Dataflow kernel context ─────────────────────────────────────────────────
// jit_stubs/api/dataflow/dataflow_api.h was already included (and is now
// #pragma once-locked), confirming we are in a dataflow kernel TU.
// Temporarily undefine COMPILE_FOR_TRISC to let the real header emit the
// noc_traits_t<CircularBuffer> specialisation and the NOC-backed method bodies.
#pragma push_macro("COMPILE_FOR_TRISC")
#undef COMPILE_FOR_TRISC
#include_next "experimental/circular_buffer.h"
#pragma pop_macro("COMPILE_FOR_TRISC")
#else
// ── Compute (TRISC) kernel context ──────────────────────────────────────────
// api/dataflow/dataflow_api.h has not been included yet.  Keep COMPILE_FOR_TRISC
// defined so the real header takes the LLK path and does not pull in noc.h /
// dataflow_api.h — those would conflict with api/compute/common.h and friends.
// The noc_traits_t<CircularBuffer> specialisation is not needed in compute
// kernels (they never call noc.async_write(cb, ...)).
#include_next "experimental/circular_buffer.h"
#endif
