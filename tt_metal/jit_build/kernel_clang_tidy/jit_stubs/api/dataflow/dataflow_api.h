// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Wrapper for api/dataflow/dataflow_api.h (clang-tidy analysis only).
//
// Audit (April 2026): no compute kernel (TRISC_MATH/TRISC_PACK/TRISC_UNPACK)
// transitively includes dataflow_api.h, so the function renaming machinery
// originally added to avoid get_arg_addr / get_arg_val duplicate definitions
// is not needed and is dropped here.
//
// What we DO need: reg_read() is called unqualified inside dataflow_api.h
// (cb page tracking helpers at lines ~367, ~401, ~438, ~475).  On device,
// non-TRISC firmware provides a global reg_read() via risc_common.h.  In our
// clang-tidy setup all TUs are compiled with -DCOMPILE_FOR_TRISC (one
// consistent flag set), which gates out risc_common.h's global reg_read().
// We inject a thin global shim before the real header so the unqualified
// calls resolve.
//
// The shim guard (KCT_REG_READ_SHIM_DEFINED) prevents duplicate definitions
// when this header is included multiple times transitively.

#pragma once

#ifndef KCT_REG_READ_SHIM_DEFINED
#define KCT_REG_READ_SHIM_DEFINED
#include <cstdint>
// Provide the global reg_read() that dataflow_api.h calls unqualified.
// risc_common.h normally defines this for non-TRISC TUs; we replicate it
// here unconditionally so that dataflow kernels compiled with COMPILE_FOR_TRISC
// (for consistency) also see it.
inline __attribute__((always_inline)) uint32_t reg_read(uint32_t addr) {
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(addr));
    return *p;
}
#endif  // KCT_REG_READ_SHIM_DEFINED

#include_next "api/dataflow/dataflow_api.h"

// ── Fallback get_noc_addr for TensorAccessor<DSpec> (KCT stub) ──────────────
//
// The primary get_noc_addr template in dataflow_api_addrgen.h uses:
//   decltype(addrgen.get_noc_addr())   ← SFINAE resolver, 0 args
// TensorAccessor<DSpec>::get_noc_addr(page_id, offset=0, noc=noc_index)
// requires at least page_id (no default), so the resolver is ill-formed
// and the template is SFINAE-eliminated — leaving "no matching function".
//
// On-device production builds work because InterleavedAddrGen<DRAM> has a
// specific deprecated overload, and sharded TensorAccessor types are handled
// at a higher level.  In the KCT build (KERNEL_COMPILE_TIME_ARGS=1,...) the
// is_sharded bit is 1, so TensorAccessor<DSpec> becomes the sharded variant
// which doesn't inherit from InterleavedAddrGen — and there is no overload.
//
// Fix: add a C++20 constrained fallback that activates exactly when:
//  • the type has get_noc_addr(uint32, uint32, uint8)  [has_get_noc_addr_v]
//  • the type does NOT have get_noc_addr()             [the broken resolver]
// This correctly handles TensorAccessor<DSpec> without conflicting with the
// specific deprecated overloads for InterleavedAddrGen<DRAM> et al.

template <typename AddrGen>
    requires(has_get_noc_addr_v<std::decay_t<AddrGen>> &&
             !requires(const AddrGen& a) { a.get_noc_addr(); })
FORCE_INLINE uint64_t get_noc_addr(
    const uint32_t id, const AddrGen& addrgen, uint32_t offset = 0, uint8_t noc = noc_index) {
    return addrgen.get_noc_addr(id, offset, noc);
}
