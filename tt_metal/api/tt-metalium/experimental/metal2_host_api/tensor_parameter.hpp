// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  TensorParameter API
// ============================================================================
//
// A TensorParameter is used to declare that a Program operates on a MeshTensor
// with a particular (single-device) tensor layout.
//
// The TensorParameter describes the required properties of the tensor argument
// (MeshTensor object) that will be supplied at Program execution time.
// The MeshTensor argument is supplied via ProgramRunArgs. If its properties
// do not match the declared TensorParameter, a runtime error is triggered.
//
// Unlike Program-local resources (like DFBs and semaphores), a MeshTensor is
// a user-managed memory resource. Its lifetime is not bound to the Program.
//
// ============================================================================

// A name identifying a TensorParameter within a ProgramSpec.
using TensorParamName = ttsl::StrongType<std::string, struct TensorParamNameTag>;

struct TensorParameter {
    // Tensor identifier: used to reference this Tensor within the ProgramSpec
    TensorParamName unique_id;

    // Single-device tensor layout
    tt::tt_metal::TensorSpec spec;

    ////////////////////////////////////////////////
    // Advanced options (see advanced_options.hpp)
    ////////////////////////////////////////////////
    TensorParameterAdvancedOptions advanced_options;
};

}  // namespace tt::tt_metal::experimental
