// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// KCT stub for experimental/core_local_mem.h
//
// Problem: the real header guards noc_traits_t<CoreLocalMem<T, AddressType>>
// with #if !defined(COMPILE_FOR_TRISC).  KCT compiles ALL kernels with
// COMPILE_FOR_TRISC (needed by risc_common.h), so the specialisation is
// always stripped.  Dataflow kernels that pass CoreLocalMem as a Noc src/dst
// then hit "no type named 'src_args_type'" substitution failures.
//
// Fix: temporarily undefine COMPILE_FOR_TRISC while processing the real
// header so the noc_traits_t specialisation gets instantiated.  Also suppress
// static_assert so that the unconditionally-false assert in dst_addr_mcast
// does not fire during template parsing.
//
// The noc.h #pragma-once lock means the re-inclusion inside core_local_mem.h
// (triggered when COMPILE_FOR_TRISC is absent) is a no-op.

#pragma once

#pragma push_macro("COMPILE_FOR_TRISC")
#undef COMPILE_FOR_TRISC

#pragma push_macro("static_assert")
#define static_assert(...)  // KCT: suppress CoreLocalMem dst_addr_mcast assert

#include_next "experimental/core_local_mem.h"

#pragma pop_macro("static_assert")
#pragma pop_macro("COMPILE_FOR_TRISC")
