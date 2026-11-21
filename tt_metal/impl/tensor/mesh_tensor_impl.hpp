// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <utility>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/assert.hpp>

#include "spec/layout/tensor_layout_impl.hpp"

/**
 * Internal implementation header for MeshTensor.
 *
 * This header exposes MeshTensorImpl so that translation units within tt_metal (e.g. tensor_apis.cpp)
 * can operate directly on the implementation without going through the public MeshTensor compatibility accessors.
 * It is private to tt_metal and is not part of the installed public API.
 */

namespace tt::tt_metal {

class MeshTensorImpl {
public:
    MeshTensorImpl(std::shared_ptr<distributed::MeshBuffer> mesh_buffer, TensorSpec spec, TensorTopology topology) :
        mesh_buffer_(std::move(mesh_buffer)), spec_(std::move(spec)), topology_(std::move(topology)) {
        TT_FATAL(mesh_buffer_ != nullptr, "MeshBuffer cannot be nullptr.");
        TT_FATAL(mesh_buffer_->is_allocated(), "MeshBuffer must be allocated.");
        validate_buffer_for_spec(*mesh_buffer_, spec_);
    }

    const distributed::MeshBuffer& mesh_buffer() const { return *raw_mesh_buffer(); }
    const std::shared_ptr<distributed::MeshBuffer>& raw_mesh_buffer() const { return mesh_buffer_; }
    const TensorSpec& spec() const { return spec_; }
    const TensorTopology& topology() const { return topology_; }
    void update_topology(TensorTopology topology) { topology_ = std::move(topology); }
    void update_spec(TensorSpec spec) {
        validate_buffer_for_spec(*mesh_buffer_, spec);
        spec_ = std::move(spec);
    }

private:
    // The buffer must hold the tensor in the memory type and memory layout the spec describes. A view's page can be
    // larger than the spec's: a narrowed ROW_MAJOR view keeps the source's row as the page stride.
    static void validate_buffer_for_spec(const distributed::MeshBuffer& mesh_buffer, const TensorSpec& spec) {
        TT_FATAL(
            mesh_buffer.size() >= spec.compute_packed_buffer_size_bytes(),
            "MeshBuffer must be large enough to hold the tensor.");
        const auto& config = mesh_buffer.device_local_config();
        TT_FATAL(
            config.buffer_type == spec.memory_config().buffer_type(),
            "MeshBuffer type {} differs from the tensor's {}",
            config.buffer_type,
            spec.memory_config().buffer_type());
        TT_FATAL(
            is_sharded(config.sharding_args.buffer_layout()) == spec.memory_config().is_sharded(),
            "MeshBuffer layout {} doesn't match the tensor's {}",
            config.sharding_args.buffer_layout(),
            spec.memory_config().memory_layout());
        // From the cached physical shape: this runs for every tensor an op creates.
        const auto& layout = spec.tensor_layout().impl();
        const size_t page_size = layout.compute_page_size_bytes(layout.compute_page_shape(spec.physical_shape()));
        TT_FATAL(
            config.page_size >= page_size,
            "MeshBuffer page size {} is smaller than the tensor's page size {}",
            config.page_size,
            page_size);
    }

    // Invariant:
    // 1. Cannot be nullptr and must be allocated.
    // 2. Must be large enough to hold a tensor describale with spec_
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_;
    TensorSpec spec_;
    // TODO(river): What is the invariant of topology?
    TensorTopology topology_;
};

}  // namespace tt::tt_metal
