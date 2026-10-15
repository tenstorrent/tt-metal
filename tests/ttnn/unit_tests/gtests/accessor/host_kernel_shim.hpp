// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host stubs that let device-side headers (api/tensor/tensor_accessor.h, and kernel helpers written
// against it such as the all_gather gather_walk.hpp) compile into a host gtest.
//
// Usage:
//   #include "host_kernel_shim.hpp"
//     Defines the stubs, includes api/tensor/tensor_accessor.h, then #undefs the kernel-style macros
//     again so they do not leak into the rest of the test TU. Drop-in for the stub block that used to
//     live inline in test_tensor_accessor.cpp.
//
//   #define HOST_KERNEL_SHIM_KEEP_MACROS
//   #include "host_kernel_shim.hpp"
//     Same, but FORCE_INLINE / ASSERT / ... stay defined, for a test that goes on to include more
//     kernel headers (which need them) or copies kernel code verbatim.
//
// ASSERT throws std::out_of_range, so a test can EXPECT_THROW on a violated kernel precondition.
//
// Include order: this shim must come before any host-safe kernel header (e.g. gather_walk.hpp,
// gather_packer.hpp). Those headers do not define fallback FORCE_INLINE / ASSERT on the host; they #error if
// the shim (or an equivalent) has not defined them first.
//
// NOC_MAX_BURST_SIZE: a kernel build gets it from the arch's noc_parameters.h. Here it defaults to the
// Blackhole value; define HOST_KERNEL_SHIM_NOC_MAX_BURST_SIZE (or NOC_MAX_BURST_SIZE itself) before the
// include to model another arch. The per-arch figures are also exposed as constexprs below so one TU can
// cover both.

#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace host_kernel_shim {
// NOC_MAX_BURST_WORDS (256) * NOC_WORD_BYTES, from tt_metal/hw/inc/internal/tt-1xx/<arch>/noc/noc_parameters.h.
inline constexpr uint32_t kNocMaxBurstSizeBlackhole = 16384;  // 512-bit NoC
inline constexpr uint32_t kNocMaxBurstSizeWormhole = 8192;    // 256-bit NoC
}  // namespace host_kernel_shim

#ifndef HOST_KERNEL_SHIM_NOC_MAX_BURST_SIZE
#define HOST_KERNEL_SHIM_NOC_MAX_BURST_SIZE 16384
#define HOST_KERNEL_SHIM_DEFINED_BURST_KNOB
#endif
#ifndef NOC_MAX_BURST_SIZE
#define NOC_MAX_BURST_SIZE HOST_KERNEL_SHIM_NOC_MAX_BURST_SIZE
#define HOST_KERNEL_SHIM_DEFINED_NOC_MAX_BURST_SIZE
#endif

// NOLINTBEGIN(bugprone-macro-parentheses)

// Defines to include api/tensor/tensor_accessor.h but won't need these
#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

template <int N>
constexpr auto get_ct_arg();
#define get_compile_time_arg_val(arg_idx) get_ct_arg<arg_idx>()

namespace tensor_accessor {
// Declared only: a host test must not reach get_noc_addr() on a DRAM accessor.
uint64_t get_dram_bank_base_offset(uint32_t bank_id, uint8_t noc);
}  // namespace tensor_accessor

[[maybe_unused]] static uint32_t my_x[1] = {0};
[[maybe_unused]] static uint32_t my_y[1] = {0};

[[maybe_unused]] static void host_assert(bool condition, const char* expression) {
    if (!condition) {
        throw std::out_of_range(expression);
    }
}

#define noc_index 0
#define ASSERT(condition, ...) host_assert(condition, #condition)
#define FORCE_INLINE inline __attribute__((always_inline))
#define DPRINT std::cout
#define ENDL() std::endl
#define DPRINT_DATA0(x) x
#define DPRINT_DATA1(x) x
#define DPRINT_MATH(x) x
#define NOC_UNICAST_ADDR_X(addr) addr
#define NOC_UNICAST_ADDR_Y(addr) addr
// Stubs for L1 NOC address macros (used in TensorAccessor::get_noc_addr for !is_dram path).
// Host-side tests only check page_id(), not actual NOC addresses, so the values don't matter.
#define DYNAMIC_NOC_X(noc, x) (x)
#define DYNAMIC_NOC_Y(noc, y) (y)
#define NOC_XY_ADDR(x, y, addr) \
    (static_cast<uint64_t>(x) << 32 | static_cast<uint64_t>(y) << 16 | static_cast<uint64_t>(addr))
// Host mock of the device NoC address backend (see internal/dataflow/noc_address_backend_xy.h),
// mirroring the macro stubs above. Required even when no test calls these paths: qualified
// names in TensorAccessor's member bodies are looked up when the template is parsed.
namespace noc_address_backend {
[[maybe_unused]] static bool is_local(uint64_t noc_addr, uint8_t noc) {
    return NOC_UNICAST_ADDR_X(noc_addr) == my_x[noc] && NOC_UNICAST_ADDR_Y(noc_addr) == my_y[noc];
}
[[maybe_unused]] static uint64_t worker_address(uint32_t x, uint32_t y, uint32_t addr, [[maybe_unused]] uint8_t noc) {
    return NOC_XY_ADDR(DYNAMIC_NOC_X(noc, x), DYNAMIC_NOC_Y(noc, y), addr);
}
}  // namespace noc_address_backend
#endif

#include "api/tensor/tensor_accessor.h"

#ifndef HOST_KERNEL_SHIM_KEEP_MACROS
#undef get_compile_time_arg_val
#undef noc_index
#undef ASSERT
#undef DPRINT
#undef ENDL
#undef DPRINT_DATA0
#undef DPRINT_DATA1
#undef DPRINT_MATH
#undef FORCE_INLINE
#undef DYNAMIC_NOC_X
#undef DYNAMIC_NOC_Y
#undef NOC_XY_ADDR
#undef NOC_UNICAST_ADDR_X
#undef NOC_UNICAST_ADDR_Y
// Only what this shim defined; an includer's own definition is left alone.
#ifdef HOST_KERNEL_SHIM_DEFINED_NOC_MAX_BURST_SIZE
#undef NOC_MAX_BURST_SIZE
#undef HOST_KERNEL_SHIM_DEFINED_NOC_MAX_BURST_SIZE
#endif
#ifdef HOST_KERNEL_SHIM_DEFINED_BURST_KNOB
#undef HOST_KERNEL_SHIM_NOC_MAX_BURST_SIZE
#undef HOST_KERNEL_SHIM_DEFINED_BURST_KNOB
#endif
#endif

// NOLINTEND(bugprone-macro-parentheses)
