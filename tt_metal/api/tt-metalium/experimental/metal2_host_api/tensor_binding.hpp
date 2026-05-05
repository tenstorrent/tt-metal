// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// A name identifying a TensorBinding within a ProgramSpec.
//
// CONVENTION: define names as `constexpr const char*` constants, e.g.:
//   constexpr const char* INPUT_TENSOR = "input_tensor";
//   TensorBinding{.unique_id = INPUT_TENSOR, ...};
// Reusing a single constant helps catch typos and errors at compile time.
using TensorBindingName = std::string;

// A TensorBinding declares that a Program operates on a Tensor with a particular layout.
//
// This is categorically different from DataflowBufferSpec / SemaphoreSpec:
//  - DFBs and Semaphores are PROGRAM-MANAGED: ephemeral resources allocated and freed by the runtime.
//  - Tensors are USER-MANAGED: the user owns the underlying device memory and its lifetime.
// TensorBinding is grouped separately in ProgramSpec to make this distinction visible.
//
// A TensorBinding captures the static layout (TensorSpec) the program expects. The actual
// MeshTensor — which carries the per-enqueue base address — is supplied at execution time
// via TensorRunParams (see program_run_params.hpp).
//
// Multi-kernel sharing: One TensorBinding can be referenced by many TensorAccessorBindings
// across different KernelSpecs. The single program-level declaration is the single source
// of truth, eliminating the possibility of binding-skew when the same tensor is read by
// multiple kernels.
struct TensorBinding {
    // Tensor identifier: used to reference this Tensor within the ProgramSpec
    TensorBindingName unique_id;

    // Single-device tensor layout (logical shape, layout, memory config).
    // TensorSpec is single-device by design; the same spec describes the layout on every
    // device the program runs on. Validated against the supplied MeshTensor at enqueue.
    tt::tt_metal::TensorSpec spec;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
