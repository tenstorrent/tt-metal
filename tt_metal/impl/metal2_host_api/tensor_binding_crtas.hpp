// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal::experimental {

// Emit the CRTA words for a single tensor binding, in order:
//   [base_address_word, optional runtime_field_words...]
// invoking emit(uint32_t) once per word. Total = 1 + handle.num_runtime_field_crta_words.
//
// The base address always lives in CRTAs (per-enqueue, since the bound MeshTensor's
// address can change between binds). Additional runtime field words appear immediately
// after, when the TensorParameter opts into a dynamic accessor field. Two kinds exist,
// mutually exclusive per binding (discriminated by handle.runtime_field_is_page_size):
//   - tensor_shape_in_pages, for sharded TensorParameters with dynamic_tensor_shape=true
//     (`rank` shape words); or
//   - the aligned page size, for interleaved row-major TensorParameters with
//     dynamic_tensor_shape=true (one word).
//
// Allocation-free by design: this runs once per binding on every enqueue, so callers
// emit straight into their destination (push_back onto the assembled CRTA vector on the
// full path, or write into the kernel's existing CRTA buffer slot on the partial paths)
// rather than through a per-binding temporary vector.
//
// `emit` is taken by const-ref (not a forwarding reference): it is invoked multiple times
// here, so it must not be forwarded/moved-from.
template <typename Emit>
void EmitBindingCrtaValues(const TensorBindingHandle& handle, const MeshTensor& tensor, const Emit& emit) {
    const auto address = tensor.address();
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(),
        "Tensor argument for TensorParameter '{}' base address {} exceeds uint32_t max",
        handle.tensor_parameter_name,
        address);
    emit(static_cast<uint32_t>(address));

    if (handle.num_runtime_field_crta_words == 0) {
        return;
    }

    // Both runtime-field kinds source their values from the bound MeshTensor's reference buffer.
    const tt::tt_metal::Buffer* buffer = tensor.mesh_buffer().get_reference_buffer();
    TT_FATAL(
        buffer != nullptr,
        "Tensor argument for TensorParameter '{}' has runtime accessor field CRTA words but no backing Buffer to "
        "source them from.",
        handle.tensor_parameter_name);

    if (handle.runtime_field_is_page_size) {
        // Page-size runtime field (interleaved row-major): emit the buffer's aligned page size, re-derived
        // each dispatch so it tracks a width-varying tensor across program-cache hits. Exactly one
        // word by construction (ResolveTensorParameterStaticCTAs reserves a single slot). No
        // BufferDistributionSpec is involved -- interleaved tensors have none, which is exactly why
        // the field-kind discriminator exists (the shape path below would FATAL on a missing BDS).
        emit(static_cast<uint32_t>(buffer->aligned_page_size()));
        return;
    }

    // dynamic_tensor_shape (sharded): the runtime tensor's shape-in-pages, one word per dim.
    const auto& bds_opt = buffer->buffer_distribution_spec();
    TT_FATAL(
        bds_opt.has_value(),
        "Tensor argument for TensorParameter '{}' has no BufferDistributionSpec.",
        handle.tensor_parameter_name);
    const auto& tensor_shape = bds_opt->tensor_shape_in_pages();
    TT_FATAL(
        tensor_shape.rank() == handle.num_runtime_field_crta_words,
        "Tensor argument for TensorParameter '{}' supplied a MeshTensor whose sharded distribution rank ({}) differs "
        "from the rank ({}) reserved at ProgramSpec resolution time. This is the shard-layout rank -- the dim count "
        "of the shape-in-pages after squeezing -- NOT the tensor's logical rank, so relax_logical_rank does not "
        "permit it. A permitted shape change can alter it, when the shard shape tiles the tensor differently. Not "
        "supported; see TensorSpecRelaxations.",
        handle.tensor_parameter_name,
        tensor_shape.rank(),
        handle.num_runtime_field_crta_words);
    for (size_t i = 0; i < tensor_shape.rank(); ++i) {
        emit(static_cast<uint32_t>(tensor_shape[i]));
    }
}

}  // namespace tt::tt_metal::experimental
