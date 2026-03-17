// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/host_tensor.hpp>

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
    DistributedHostBuffer buffer() const&& { return buffer_; }
    const TensorSpec& spec() const { return spec_; }
    const TensorTopology& topology() const { return topology_; }

private:
    DistributedHostBuffer buffer_;
    TensorSpec spec_;
    TensorTopology topology_;
};

HostTensor::HostTensor(DistributedHostBuffer buffer, TensorSpec spec, TensorTopology topology) :
    impl(std::make_unique<HostTensorImpl>(std::move(buffer), std::move(spec), std::move(topology))) {}

HostTensor::HostTensor(HostTensor&& other, TensorSpec spec, TensorTopology topology) :
    impl(std::make_unique<HostTensorImpl>(other.impl->buffer(), std::move(spec), std::move(topology))) {}

HostTensor::HostTensor(const HostTensor& other) : impl(std::make_unique<HostTensorImpl>(*other.impl)) {}

HostTensor& HostTensor::operator=(const HostTensor& other) {
    impl = std::make_unique<HostTensorImpl>(*other.impl);
    return *this;
}

HostTensor::HostTensor(HostTensor&& other) noexcept : impl(std::move(other.impl)) {}

HostTensor& HostTensor::operator=(HostTensor&& other) noexcept {
    impl = std::move(other.impl);
    return *this;
}

HostTensor::~HostTensor() = default;

const TensorSpec& HostTensor::tensor_spec() const { return impl->spec(); }

const TensorTopology& HostTensor::tensor_topology() const { return impl->topology(); }

const DistributedHostBuffer& HostTensor::buffer() const { return impl->buffer(); }

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

HostTensor HostTensor::transform(const std::function<HostBuffer(const HostBuffer&)>& callable) const {
    auto transformed_buffer =
        buffer().transform(callable, DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);
    return HostTensor(std::move(transformed_buffer), tensor_spec(), tensor_topology());
}

}  // namespace tt::tt_metal
