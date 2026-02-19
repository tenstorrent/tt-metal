// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/distributed_tensor_configs.hpp>

namespace tt::tt_metal {

class DistributedTensorSpec final {
public:
    DistributedTensorSpec(TensorSpec tensor_spec, distributed::MeshMapperConfig mapper_config) :
        tensor_spec_(std::move(tensor_spec)), mapper_config_(std::move(mapper_config)) {}

    DistributedTensorSpec(const DistributedTensorSpec&) = default;
    DistributedTensorSpec& operator=(const DistributedTensorSpec&) = default;
    DistributedTensorSpec(DistributedTensorSpec&&) noexcept = default;
    DistributedTensorSpec& operator=(DistributedTensorSpec&&) noexcept = default;

    const TensorSpec& tensor_spec() const { return tensor_spec_; }

    const distributed::MeshMapperConfig& mapper_config() const { return mapper_config_; }

private:
    TensorSpec tensor_spec_;
    distributed::MeshMapperConfig mapper_config_;
};

}  // namespace tt::tt_metal
