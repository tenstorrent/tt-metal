// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ALWI inline __attribute__((always_inline))

#include "ckernel.h"
#include "internal/firmware_common.h"
#include "ckernel_include.h"
#include "hostdevcommon/kernel_structs.h"

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#define MATH(...) __VA_ARGS__
#define MAIN math_main()
#else
#define MATH(...)
#endif

#ifdef TRISC_PACK
#define PACK(...) __VA_ARGS__
#define MAIN pack_main()
#else
#define PACK(...)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#define UNPACK(...) __VA_ARGS__
#define MAIN unpack_main()
#else
#define UNPACK(...)
#endif

namespace ckernel {

// Quasar's DataFormat has no UInt32, Bfp8_b or Bfp4_b. API format checks name those formats through
// these predicates so one check compiles on every arch; each is false where the format does not exist.
constexpr bool is_uint32_format([[maybe_unused]] DataFormat format) {
#ifdef ARCH_QUASAR
    return false;
#else
    return format == DataFormat::UInt32;
#endif
}

constexpr bool is_bfp8_b_format([[maybe_unused]] DataFormat format) {
#ifdef ARCH_QUASAR
    return false;
#else
    return format == DataFormat::Bfp8_b;
#endif
}

constexpr bool is_bfp4_b_format([[maybe_unused]] DataFormat format) {
#ifdef ARCH_QUASAR
    return false;
#else
    return format == DataFormat::Bfp4_b;
#endif
}

}  // namespace ckernel
