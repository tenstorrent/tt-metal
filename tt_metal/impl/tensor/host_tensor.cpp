// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/host_tensor.hpp>

#include <cstdint>
#include <memory>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>

#include "tensor/details/tensor_attributes.hpp"
#include "tensor/details/storage.hpp"
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

namespace tt::tt_metal {

// Special Member Functions

HostTensor::HostTensor() = default;
HostTensor::~HostTensor() = default;

// Deep copy since we use unique_ptr (unlike Tensor which shallow copies shared_ptr)
HostTensor::HostTensor(const HostTensor& other) :
    impl(other.impl ? std::make_unique<TensorAttributes>(*other.impl) : nullptr) {}

HostTensor& HostTensor::operator=(const HostTensor& other) {
    if (this != &other) {
        impl = other.impl ? std::make_unique<TensorAttributes>(*other.impl) : nullptr;
    }
    return *this;
}

// Move - default works for unique_ptr
HostTensor::HostTensor(HostTensor&&) noexcept = default;
HostTensor& HostTensor::operator=(HostTensor&&) noexcept = default;

// Constructor (pattern from tensor.cpp:84-90)
HostTensor::HostTensor(const HostBuffer& buffer, TensorSpec spec, TensorTopology topology) :
    impl(std::make_unique<TensorAttributes>(Storage(HostStorage(buffer)), std::move(spec), std::move(topology))) {}

// Getter Implementations (following tensor.cpp:607-650)

const Shape& HostTensor::logical_shape() const { return impl->get_tensor_spec().logical_shape(); }

const Shape& HostTensor::padded_shape() const { return impl->get_tensor_spec().padded_shape(); }

DataType HostTensor::dtype() const { return impl->get_tensor_spec().tensor_layout().get_data_type(); }

Layout HostTensor::layout() const { return impl->get_tensor_spec().tensor_layout().get_layout(); }

const TensorSpec& HostTensor::tensor_spec() const { return impl->get_tensor_spec(); }

const MemoryConfig& HostTensor::memory_config() const { return tensor_spec().tensor_layout().get_memory_config(); }

const std::optional<ShardSpec>& HostTensor::shard_spec() const { return memory_config().shard_spec(); }

const std::optional<NdShardSpec>& HostTensor::nd_shard_spec() const { return memory_config().nd_shard_spec(); }

const TensorTopology& HostTensor::tensor_topology() const { return impl->get_tensor_topology(); }

Shape HostTensor::strides() const {
    auto s = compute_strides(padded_shape());
    return Shape(ttsl::SmallVector<uint32_t>(s.begin(), s.end()));
}

// Computed Getters (following tensor.cpp:411-461)

HostTensor::volumn_type HostTensor::logical_volume() const { return logical_shape().volume(); }

HostTensor::volumn_type HostTensor::physical_volume() const { return padded_shape().volume(); }

// Host tensors are never sharded
bool HostTensor::is_sharded() const { return false; }

bool HostTensor::is_scalar() const {
    const Shape& shape = logical_shape();
    return shape.rank() == 0 || shape.volume() == 1;
}

// Following tensor_impl.cpp:54-66
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

// Storage Access (following tensor.cpp:629-632 pattern)
HostBuffer HostTensor::get_host_buffer() const {
    const auto& storage = std::get<HostStorage>(impl->get_storage());
    // Get shard at (0,0) for single-device tensor - always exists for HostTensor
    auto buffer = storage.buffer().get_shard(distributed::MeshCoordinate(0, 0));
    TT_ASSERT(buffer.has_value(), "HostTensor must have a buffer at coordinate (0, 0)");
    return *buffer;
}

// String Conversion (tensor.cpp:391 + tensor_impl.cpp:527-529)
std::string HostTensor::write_to_string() const {
    // Simple implementation for now - can expand later
    std::ostringstream os;
    os << "HostTensor(shape=" << logical_shape() << ", dtype=" << dtype() << ", layout=";
    switch (layout()) {
        case Layout::ROW_MAJOR: os << "ROW_MAJOR"; break;
        case Layout::TILE: os << "TILE"; break;
        default: os << "INVALID"; break;
    }
    os << ")";
    return os.str();
}

// Mutation (stub per header TODO)
void HostTensor::reshape(/* */) {
    // TODO: Implement reshape - see tensor.cpp:417-422
}

}  // namespace tt::tt_metal
