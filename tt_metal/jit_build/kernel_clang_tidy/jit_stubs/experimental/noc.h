// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KCT stub for experimental/noc.h (tt_metal/hw/inc/experimental/noc.h)
//
// The real header defines a primary template:
//
//   template <typename T>
//   struct noc_traits_t {
//       static_assert(sizeof(T) == 0, "NoC transactions not supported for this type");
//   };
//
// With KERNEL_COMPILE_TIME_ARGS=1,1,...,1 some kernels instantiate
// Noc::async_read()/async_write_multicast() with TensorAccessor<> types whose
// noc_traits_t specialisations are not in scope (because the sharded
// distribution spec from CT args=1 doesn't match any known specialisation).
// The primary template's static_assert then fires:
//
//   error: static assertion failed due to requirement
//     'sizeof(experimental::CoreLocalMem<unsigned int, unsigned int>) == 0':
//     NoC transactions are not supported for this type
//
// For KCT analysis we just need the code to parse; suppress the static_assert
// for the duration of the real header include.

#pragma once

#pragma push_macro("static_assert")
#define static_assert(...)  // KCT: suppressed to allow dummy CT arg types
#include_next "experimental/noc.h"
#pragma pop_macro("static_assert")
