// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

/**
 * Internal implementation header for HostTensor.
 *
 * This header exposes HostTensorImpl so that translation units within tt_metal (e.g. tensor_apis.cpp)
 * can operate directly on the implementation without going through the public HostTensor accessors.
 * It is private to tt_metal and is not part of the installed public API.
 */

namespace tt::tt_metal {

class HostTensorImpl {
public:
    HostTensorImpl(DistributedHostBuffer buffer, TensorSpec spec, TensorTopology topology) :
        buffer_(std::move(buffer)), spec_(std::move(spec)), topology_(std::move(topology)) {}

    HostTensorImpl(const HostTensorImpl& other) = default;
    HostTensorImpl(HostTensorImpl&& other) noexcept = default;
    HostTensorImpl& operator=(const HostTensorImpl& other) = default;
    HostTensorImpl& operator=(HostTensorImpl&& other) noexcept = default;
    ~HostTensorImpl() = default;

    const DistributedHostBuffer& buffer() const& { return buffer_; }
    DistributedHostBuffer& buffer() & { return buffer_; }
    DistributedHostBuffer buffer() const&& { return buffer_; }
    const TensorSpec& spec() const { return spec_; }
    const TensorTopology& topology() const { return topology_; }
    void update_topology(TensorTopology topology) { topology_ = std::move(topology); }

private:
    DistributedHostBuffer buffer_;
    TensorSpec spec_;
    TensorTopology topology_;
};

}  // namespace tt::tt_metal
