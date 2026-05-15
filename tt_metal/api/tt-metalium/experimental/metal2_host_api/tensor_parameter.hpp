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
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
