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

    ///////////////////////////////////////////////////////////////////
    // Advanced options
    ///////////////////////////////////////////////////////////////////

    // By default, the MeshTensor argument provided at execution time must
    // EXACTLY match the TensorParameter's declared TensorSpec. The advanced
    // options below relax this match requirement in particular ways.
    //
    // NOTE: These options are UNSAFE if set to true; most kernels will not function
    // correctly if the tensor argument's spec deviates from the declared spec.
    // Use with caution and ensure that your kernel logic is compatible.

    // Permit tensor arguments whose logical_shape differs from the declared shape.
    // The argument's padded_shape must still match exactly.
    // Effects:
    //  - Validation checks are relaxed
    //  - TensorAccessor configuration is completely unchanged
    bool match_padded_shape_only = false;

    // Permit tensor arguments with dynamic logical shape.
    // The argument's logical_shape AND padded_shape may differ from the declared shape.
    // Effects:
    //  - Validation checks are relaxed
    //  - For an interleaved tensor, TensorAccessor configuration is unchanged
    //  - For a sharded tensor, the TensorAccessor configuration dynamically reflects the
    //    argument's actual shape. (Shape becomes an implicit runtime argument.)
    bool dynamic_tensor_shape = false;

    // Additional relaxation options will be added in the future.
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
