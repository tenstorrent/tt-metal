// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/details/storage.hpp>

namespace tt::tt_metal {

class HostTensorImpl {
public:
    HostTensorImpl(HostStorage storage, TensorSpec spec, TensorTopology topology) :
        storage_(std::move(storage)), tensor_spec_(std::move(spec)), tensor_topology_(std::move(topology)) {}

    HostStorage storage_;
    TensorSpec tensor_spec_;
    TensorTopology tensor_topology_;
};

HostTensor::HostTensor(HostStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
    impl(std::make_unique<HostTensorImpl>(std::move(storage), std::move(tensor_spec), std::move(tensor_topology))) {}

HostTensor::HostTensor(HostBuffer buffer, TensorSpec spec, TensorTopology topology) :
    impl(std::make_unique<HostTensorImpl>(HostStorage{std::move(buffer)}, std::move(spec), std::move(topology))) {}

HostTensor::HostTensor(const HostTensor& other) :
    impl(other.impl ? std::make_unique<HostTensorImpl>(*other.impl) : nullptr) {}

HostTensor& HostTensor::operator=(const HostTensor& other) {
    if (this == &other) {
        return *this;
    }
    impl = other.impl ? std::make_unique<HostTensorImpl>(*other.impl) : nullptr;
    return *this;
}

HostTensor::HostTensor(HostTensor&& other) noexcept : impl(std::move(other.impl)) {}

HostTensor& HostTensor::operator=(HostTensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    impl = std::move(other.impl);
    return *this;
}

const TensorSpec& HostTensor::tensor_spec() const {
    TT_ASSERT(impl != nullptr, "HostTensor is in default constructed state");
    return impl->tensor_spec_;
}

const TensorTopology& HostTensor::tensor_topology() const {
    TT_ASSERT(impl != nullptr, "HostTensor is in default constructed state");
    return impl->tensor_topology_;
}

const HostStorage& HostTensor::get_legacy_host_storage() const {
    TT_ASSERT(impl != nullptr, "HostTensor is in default constructed state");
    return impl->storage_;
}

const DistributedHostBuffer& HostTensor::get_distributed_host_buffer() const {
    TT_ASSERT(impl != nullptr, "HostTensor is in default constructed state");
    return impl->storage_.buffer();
}

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

void HostTensor::update_tensor_topology(TensorTopology tensor_topology) {
    TT_ASSERT(impl != nullptr, "HostTensor is in default constructed state");
    impl->tensor_topology_ = std::move(tensor_topology);
}

}  // namespace tt::tt_metal
