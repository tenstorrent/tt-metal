// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What a kernel in this directory takes from tt-metal's build for its buffers' formats. The Compute API (llk2/inc)
// does not name tt-metal; the kernels do, here.
//
//   LLK2_CB_FORMAT("cb")   the data format of the kernel's buffer "cb": the host passes it as the named compile-time
//                          arg "cb_format", the same on all three TRISCs (the Tensors' specs, and through them the
//                          Resolver, need it at compile time on every thread). A macro because the name is pasted
//                          into a string literal. It expands to get_named_compile_time_arg_val, which tt-metal
//                          generates for every kernel build (the tt-llk harness: tests/helpers/tt_metal_kernel.h)

#pragma once

#include "tensix_types.h" // DataFormat

#define LLK2_CB_FORMAT(name) static_cast<DataFormat>(get_named_compile_time_arg_val(name "_format"))
