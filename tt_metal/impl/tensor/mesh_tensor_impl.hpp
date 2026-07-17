// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <utility>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/assert.hpp>

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
        TT_FATAL(
            mesh_buffer_->size() >= spec_.compute_packed_buffer_size_bytes(),
            "MeshBuffer must be large enough to hold the tensor.");
    }

    const distributed::MeshBuffer& mesh_buffer() const { return *raw_mesh_buffer(); }
    const std::shared_ptr<distributed::MeshBuffer>& raw_mesh_buffer() const { return mesh_buffer_; }
    const TensorSpec& spec() const { return spec_; }
    const TensorTopology& topology() const { return topology_; }
    void update_topology(TensorTopology topology) { topology_ = std::move(topology); }
    void update_spec(TensorSpec spec) {
        TT_FATAL(
            mesh_buffer_->size() >= spec.compute_packed_buffer_size_bytes(),
            "MeshBuffer must be large enough to hold the tensor.");
        spec_ = std::move(spec);
    }

private:
    // Invariant:
    // 1. Cannot be nullptr and must be allocated.
    // 2. Must be large enough to hold a tensor describale with spec_
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_;
    TensorSpec spec_;
    // TODO(river): What is the invariant of topology?
    TensorTopology topology_;
};

}  // namespace tt::tt_metal
