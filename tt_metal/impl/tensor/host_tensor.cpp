// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/host_tensor.hpp>

namespace tt::tt_metal {

// TODO: Implement once HostStorage is migrated (#37692)
// HostTensor::HostTensor(HostStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology)

// TODO: Implement once HostStorage is migrated (#37692)
// HostTensor::HostTensor(HostBuffer buffer, TensorSpec spec, TensorTopology topology)

// TODO: Implement once HostStorage is migrated (#37692)
// HostTensor::HostTensor(const HostTensor& other)

// TODO: Implement once HostStorage is migrated (#37692)
// HostTensor& HostTensor::operator=(const HostTensor& other)

// TODO: Implement once HostStorage is migrated (#37692)
// HostTensor::HostTensor(HostTensor&& other) noexcept

// TODO: Implement once HostStorage is migrated (#37692)
// HostTensor& HostTensor::operator=(HostTensor&& other) noexcept

// TODO: Implement once HostStorage is migrated (#37692)
// const TensorSpec& HostTensor::tensor_spec() const

// TODO: Implement once HostStorage is migrated (#37692)
// const TensorTopology& HostTensor::tensor_topology() const

// TODO: Implement once HostStorage is migrated (#37692)
// const HostStorage& HostTensor::get_legacy_host_storage() const

// TODO: Implement once HostStorage is migrated (#37692)
// const DistributedHostBuffer& HostTensor::get_distributed_host_buffer() const

// TODO: Implement once HostStorage is migrated (#37692)
// void HostTensor::update_tensor_topology(TensorTopology tensor_topology)

DataType HostTensor::dtype() const { return tensor_spec().tensor_layout().get_data_type(); }

Layout HostTensor::layout() const { return tensor_spec().tensor_layout().get_layout(); }

const Shape& HostTensor::logical_shape() const { return tensor_spec().logical_shape(); }

const Shape& HostTensor::padded_shape() const { return tensor_spec().padded_shape(); }

HostTensor::volume_type HostTensor::logical_volume() const { return logical_shape().volume(); }

HostTensor::volume_type HostTensor::physical_volume() const { return padded_shape().volume(); }

const MemoryConfig& HostTensor::memory_config() const { return tensor_spec().memory_config(); }

bool HostTensor::is_sharded() const { return tensor_spec().memory_config().is_sharded(); }

const std::optional<ShardSpec>& HostTensor::legacy_shard_spec() const { return memory_config().shard_spec(); }

const std::optional<NdShardSpec>& HostTensor::nd_shard_spec() const { return memory_config().nd_shard_spec(); }

std::size_t HostTensor::element_size() const {
    switch (dtype()) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return sizeof(std::byte);
        default: TT_THROW("Unsupported data type");
    }
}

Strides HostTensor::strides() const { return tensor_spec().tensor_layout().compute_strides(logical_shape()); }

}  // namespace tt::tt_metal
