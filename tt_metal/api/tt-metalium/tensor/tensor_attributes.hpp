// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt-metalium/tensor/storage.hpp>
#include <tt-metalium/tensor/tensor_spec.hpp>
#include <tt-metalium/distributed/tensor_topology.hpp>

namespace tt::tt_metal {

class TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
public:
    TensorAttributes(Storage storage, TensorSpec tensor_spec, TensorTopology tensor_topology);
    TensorAttributes(const TensorAttributes&) = default;
    TensorAttributes(TensorAttributes&&) = default;
    TensorAttributes& operator=(const TensorAttributes&) = default;
    TensorAttributes& operator=(TensorAttributes&&) = default;

    // Getters and setters.
    const Storage& get_storage() const;
    Storage& get_storage();
    const TensorSpec& get_tensor_spec() const;
    const TensorTopology& get_tensor_topology() const;

    TensorAttributes with_tensor_topology(TensorTopology tensor_topology) const;

private:
    Storage storage_;
    TensorSpec tensor_spec_;
    TensorTopology tensor_topology_;
};

}  // namespace tt::tt_metal
