// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// A name identifying a TensorParameter within a ProgramSpec.
//
// CONVENTION: define names as `constexpr const char*` constants, e.g.:
//   constexpr const char* INPUT_TENSOR = "input_tensor";
//   TensorParameter{.unique_id = INPUT_TENSOR, ...};
// Reusing a single constant helps catch typos and errors at compile time.
using TensorParameterName = std::string;

// A TensorParameter is used to declare that a Program operates on a MeshTensor
// with a particular (single-device) tensor layout.
// The actual MeshTensor is supplied at execution time, via ProgramRunParams.
struct TensorParameter {
    // Tensor identifier: used to reference this Tensor within the ProgramSpec
    TensorParameterName unique_id;

    // Single-device tensor layout
    tt::tt_metal::TensorSpec spec;

    // Dynamic-shape opt-in (ADVANCED).
    //
    // By default, a TensorParameter declares an EXACT tensor layout. The MeshTensor
    // bound at execution time must match `spec` exactly: same logical shape, dtype,
    // page config, memory config, alignment.
    //
    // Setting `dynamic_tensor_shape = true` loosens that match along exactly one
    // axis: the bound tensor's `logical_shape()` may differ from `spec.logical_shape()`.
    // Everything else (dtype, page config, memory config including any shard_spec,
    // alignment) must still match exactly.
    //
    // Use case: kernels whose code is genuinely shape-agnostic (e.g., eltwise). The
    // same compiled kernel can then be bound to tensors of varying shape without
    // forcing a JIT recompile per shape.
    //
    // Mechanics by buffer type:
    //  - Interleaved: zero device-side change. The accessor's compile-time payload
    //    never depended on logical shape; this is purely a host-side validation
    //    loosening so a different-shape tensor can be bound.
    //  - Sharded: the `tensor_shape_in_pages` words move from compile-time args to
    //    common runtime args. The CTA payload becomes stable across shape variations
    //    (so the JIT cache hits across shapes), and the actual tensor's shape is
    //    written into CRTAs at enqueue time. The shard_spec (grid + shard_shape)
    //    is part of the layout match and so is fixed across binds; only logical_shape
    //    varies, and the bound tensor must still fit on the same shard grid.
    //
    // Default to false. The JIT compiler gets strictly more information with a
    // fixed shape; enable this only for kernels that genuinely don't care about
    // tensor shape at compile time.
    bool dynamic_tensor_shape = false;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
