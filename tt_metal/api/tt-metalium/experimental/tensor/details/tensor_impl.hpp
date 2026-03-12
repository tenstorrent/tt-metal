// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include <tt-metalium/experimental/tensor/details/storage.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

namespace tt::tt_metal {

// These will be brought in at #37692
class HostStorage;
class DeviceStorage;

template <typename Storage>
struct TensorImpl {
    Storage storage_;
    TensorSpec tensor_spec_;
    TensorTopology tensor_topology_;
};

template struct TensorImpl<HostStorage>;
template struct TensorImpl<DeviceStorage>;

}  // namespace tt::tt_metal
