// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/details/storage.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

namespace tt::tt_metal {

template <typename Storage>
class TensorAttributes {
public:
    TensorAttributes(Storage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
        storage_(std::move(storage)),
        tensor_spec_(std::move(tensor_spec)),
        tensor_topology_(std::move(tensor_topology)) {}

    TensorAttributes(const TensorAttributes&) = default;
    TensorAttributes(TensorAttributes&&) = default;
    TensorAttributes& operator=(const TensorAttributes&) = default;
    TensorAttributes& operator=(TensorAttributes&&) = default;

    // Getters and setters.
    const Storage& get_storage() const { return storage_; }
    Storage& get_storage() { return storage_; }
    const TensorSpec& get_tensor_spec() const { return tensor_spec_; }
    const TensorTopology& get_tensor_topology() const { return tensor_topology_; }

private:
    Storage storage_;
    TensorSpec tensor_spec_;
    TensorTopology tensor_topology_;
};

template class TensorAttributes<HostStorage>;
template class TensorAttributes<DeviceStorage>;

}  // namespace tt::tt_metal
