// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace tt::tt_metal {

/*

// TODO: Implement once DeviceStorage is migrated (#37692)
// MeshTensor::MeshTensor(DeviceStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology)

// TODO: Implement once DeviceStorage is migrated (#37692)
// bool MeshTensor::is_allocated() const

// TODO: Implement once DeviceStorage is migrated (#37692)
// std::shared_ptr<distributed::MeshBuffer> MeshTensor::mesh_buffer_invariant_breaking() const

// TODO: Implement once DeviceStorage is migrated (#37692)
// const TensorSpec& MeshTensor::tensor_spec() const

// TODO: Implement once DeviceStorage is migrated (#37692)
// const TensorTopology& MeshTensor::tensor_topology() const

// TODO: Implement once DeviceStorage is migrated (#37692)
// const DeviceStorage& MeshTensor::get_legacy_device_storage() const

// TODO: Implement once DeviceStorage is migrated (#37692)
// void MeshTensor::update_tensor_topology(TensorTopology tensor_topology)

distributed::MeshBuffer& MeshTensor::mesh_buffer() const { return *mesh_buffer_invariant_breaking(); }

distributed::MeshDevice& MeshTensor::device() const { return *mesh_buffer().device(); }

DeviceAddr MeshTensor::address() const { return mesh_buffer().address(); }

DataType MeshTensor::dtype() const { return tensor_spec().data_type(); }

Layout MeshTensor::layout() const { return tensor_spec().layout(); }

const Shape& MeshTensor::logical_shape() const { return tensor_spec().logical_shape(); }

const Shape& MeshTensor::padded_shape() const { return tensor_spec().padded_shape(); }

MeshTensor::volume_type MeshTensor::logical_volume() const { return logical_shape().volume(); }

MeshTensor::volume_type MeshTensor::physical_volume() const { return padded_shape().volume(); }

const MemoryConfig& MeshTensor::memory_config() const { return tensor_spec().memory_config(); }

bool MeshTensor::is_sharded() const { return memory_config().is_sharded(); }

const std::optional<ShardSpec>& MeshTensor::legacy_shard_spec() const { return memory_config().shard_spec(); }

const std::optional<NdShardSpec>& MeshTensor::nd_shard_spec() const { return memory_config().nd_shard_spec(); }

std::size_t MeshTensor::element_size() const {
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

Strides MeshTensor::strides() const { return tensor_spec().tensor_layout().compute_strides(logical_shape()); }

*/

}  // namespace tt::tt_metal
