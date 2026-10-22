// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Compiles the device-side api/tensor/tensor_accessor.h on host. Shared by the host tests that run kernel
// headers (test_tensor_accessor.cpp, test_all_gather_kernel_lib.cpp). The macros stay defined, so an includer
// can bring in more kernel headers before undefining them.

#include <cstdint>
#include <iostream>
#include <stdexcept>

// NOLINTBEGIN(bugprone-macro-parentheses)

// Defines to include api/tensor/tensor_accessor.h but won't need these
#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

template <int N>
constexpr auto get_ct_arg();
#define get_compile_time_arg_val(arg_idx) get_ct_arg<arg_idx>()

namespace tensor_accessor {
uint64_t get_dram_bank_base_offset(uint32_t bank_id, uint8_t noc);
}

[[maybe_unused]] static uint32_t my_x[1] = {0};
[[maybe_unused]] static uint32_t my_y[1] = {0};

static void host_assert(bool condition, const char* expression) {
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

// NOLINTEND(bugprone-macro-parentheses)
