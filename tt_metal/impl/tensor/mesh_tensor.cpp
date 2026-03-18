// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace tt::tt_metal {

class MeshTensorImpl {
public:
    MeshTensorImpl(std::shared_ptr<distributed::MeshBuffer> mesh_buffer, TensorSpec spec, TensorTopology topology) :
        MeshTensorImpl(std::move(mesh_buffer), std::move(spec), std::move(topology), nullptr) {}

    MeshTensorImpl(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer,
        TensorSpec spec,
        TensorTopology topology,
        std::shared_ptr<distributed::MeshBuffer> root_mesh_buffer) :
        mesh_buffer_(std::move(mesh_buffer)),
        spec_(std::move(spec)),
        topology_(std::move(topology)),
        root_mesh_buffer_(std::move(root_mesh_buffer)) {
        TT_FATAL(mesh_buffer_ != nullptr, "MeshBuffer cannot be nullptr.");
        TT_FATAL(mesh_buffer_->is_allocated(), "MeshBuffer must be allocated.");
        TT_FATAL(
            mesh_buffer_->size() >= spec.compute_packed_buffer_size_bytes(),
            "MeshBuffer must be large enough to hold the tensor.");
    }

    // Two step construction for MeshTensor,
    // for transiet purpose.
    MeshTensorImpl(MeshTensorImpl&& other, TensorSpec spec, TensorTopology topology) :
        mesh_buffer_(std::move(other.mesh_buffer_)),
        spec_(std::move(spec)),
        topology_(std::move(topology)),
        root_mesh_buffer_(std::move(other.root_mesh_buffer_)) {}

    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer() const { return mesh_buffer_; }
    const TensorSpec& spec() const { return spec_; }
    const TensorTopology& topology() const { return topology_; }
    void update_topology(TensorTopology topology) { topology_ = std::move(topology); }

private:
    // Invariant:
    // 1. Cannot be nullptr and must be allocated.
    // 2. Must be large enough to hold a tensor describale with spec_
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_;
    TensorSpec spec_;
    // TODO(river): What is the invariant of topology?
    TensorTopology topology_;

    // Experimental feature: Accomodates #38101
    // This keeps mesh_buffer_ alive in limited cases.
    std::shared_ptr<distributed::MeshBuffer> root_mesh_buffer_;
};

MeshTensor::MeshTensor() = default;

MeshTensor::MeshTensor(MeshTensor&& other) noexcept = default;

MeshTensor& MeshTensor::operator=(MeshTensor&& other) noexcept = default;

MeshTensor::MeshTensor(std::shared_ptr<distributed::MeshBuffer> mesh_buffer, TensorSpec spec, TensorTopology topology) :
    impl(std::make_unique<MeshTensorImpl>(std::move(mesh_buffer), std::move(spec), std::move(topology))) {}

MeshTensor::MeshTensor(MeshTensor&& other, TensorSpec spec, TensorTopology topology) :
    impl(std::make_unique<MeshTensorImpl>(std::move(*other.impl), std::move(spec), std::move(topology))) {}

MeshTensor::~MeshTensor() = default;

distributed::MeshBuffer& MeshTensor::mesh_buffer() const { return *mesh_buffer_invariant_breaking(); }

std::shared_ptr<distributed::MeshBuffer> MeshTensor::mesh_buffer_invariant_breaking() const {
    TT_ASSERT(impl != nullptr, "MeshTensor is in default constructed state.");
    return impl->mesh_buffer();
}

distributed::MeshDevice& MeshTensor::device() const { return *mesh_buffer().device(); }

const TensorSpec& MeshTensor::tensor_spec() const {
    TT_ASSERT(impl != nullptr, "MeshTensor is in default constructed state.");
    return impl->spec();
}

const TensorTopology& MeshTensor::tensor_topology() const {
    TT_ASSERT(impl != nullptr, "MeshTensor is in default constructed state.");
    return impl->topology();
}

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

void MeshTensor::update_tensor_topology(TensorTopology tensor_topology) {
    TT_ASSERT(impl != nullptr, "MeshTensor is in default constructed state.");
    impl->update_topology(std::move(tensor_topology));
}

}  // namespace tt::tt_metal
