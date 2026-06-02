// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "mesh_tensor_impl.hpp"

namespace tt::tt_metal {

MeshTensor::MeshTensor(MeshTensor&& other) noexcept = default;

MeshTensor& MeshTensor::operator=(MeshTensor&& other) noexcept = default;

MeshTensor::MeshTensor(std::shared_ptr<distributed::MeshBuffer> mesh_buffer, TensorSpec spec, TensorTopology topology) :
    impl_(std::make_unique<MeshTensorImpl>(std::move(mesh_buffer), std::move(spec), std::move(topology))) {}

MeshTensor::~MeshTensor() = default;

MeshTensorImpl& MeshTensor::impl() {
    TT_FATAL(impl_ != nullptr, "MeshTensor is in a moved-from state.");
    return *impl_;
}

const MeshTensorImpl& MeshTensor::impl() const {
    TT_FATAL(impl_ != nullptr, "MeshTensor is in a moved-from state.");
    return *impl_;
}

const distributed::MeshBuffer& MeshTensor::mesh_buffer() const { return impl().mesh_buffer(); }

std::shared_ptr<distributed::MeshBuffer> MeshTensor::mesh_buffer_invariant_breaking() const {
    return impl().raw_mesh_buffer();
}

const distributed::MeshDevice& MeshTensor::device() const { return mutable_device(); }

distributed::MeshDevice& MeshTensor::mutable_device() const { return *mesh_buffer().device(); }

const TensorSpec& MeshTensor::tensor_spec() const { return impl().spec(); }

const TensorTopology& MeshTensor::tensor_topology() const { return impl().topology(); }

bool MeshTensor::is_valueless_after_move() const { return impl_ == nullptr; }

DeviceAddr MeshTensor::address() const { return mesh_buffer().address(); }

DataType MeshTensor::dtype() const { return tensor_spec().data_type(); }

Layout MeshTensor::layout() const { return tensor_spec().layout(); }

const Shape& MeshTensor::logical_shape() const { return tensor_spec().logical_shape(); }

const Shape& MeshTensor::padded_shape() const { return tensor_spec().padded_shape(); }

MeshTensor::volume_type MeshTensor::logical_volume() const { return logical_shape().volume(); }

MeshTensor::volume_type MeshTensor::physical_volume() const { return padded_shape().volume(); }

const MemoryConfig& MeshTensor::memory_config() const { return tensor_spec().memory_config(); }

bool MeshTensor::is_sharded() const { return memory_config().is_sharded(); }

const std::optional<ShardSpec>& MeshTensor::shard_spec() const { return memory_config().shard_spec(); }

const std::optional<NdShardSpec>& MeshTensor::nd_shard_spec() const { return memory_config().nd_shard_spec(); }

std::size_t MeshTensor::element_size() const {
    switch (dtype()) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::FP8_E4M3: return sizeof(float8_e4m3);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return sizeof(std::byte);
        default: TT_THROW("Unsupported data type");
    }
}

Strides MeshTensor::strides() const { return tensor_spec().tensor_layout().compute_strides(logical_shape()); }

void MeshTensor::update_tensor_topology(TensorTopology tensor_topology) {
    impl().update_topology(std::move(tensor_topology));
}

MeshTensor MeshTensor::allocate_on_device(
    distributed::MeshDevice& mesh_device, const TensorSpec& spec, const TensorTopology& topology) {
    // Catch-all guard: FP8_E4M3 is only supported on Blackhole. Op-level validators may also
    // check this, but we enforce it here at the device-binding boundary so any path that
    // produces an FP8 tensor on unsupported hardware fails loudly rather than silently
    // generating programs that misbehave later.
    if (spec.data_type() == DataType::FP8_E4M3) {
        TT_FATAL(
            mesh_device.arch() == tt::ARCH::BLACKHOLE,
            "FP8_E4M3 is only supported on Blackhole hardware (got arch {})",
            mesh_device.arch());
    }
    auto mesh_buffer = tensor_impl::allocate_device_buffer(&mesh_device, spec);
    return MeshTensor(std::move(mesh_buffer), spec, topology);
}

}  // namespace tt::tt_metal
