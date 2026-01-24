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
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {

// Special Member Functions

// Main Constructor
HostTensor::HostTensor(const HostBuffer& buffer, TensorSpec spec, TensorTopology topology) :
    impl(std::make_unique<TensorAttributes>(Storage(HostStorage(buffer)), std::move(spec), std::move(topology))) {}

// The default tensor spec for an empty Tensor
// TODO: PageConfig(Layout::INVALID) is not valid, need to fix.
TensorSpec DEFAULT_TENSOR_SPEC(Shape(), TensorLayout(DataType::INVALID, PageConfig(Layout::INVALID), MemoryConfig()));

// TODO: when and if we need to check whether a host tensor is empty
HostTensor::HostTensor() : HostTensor(HostBuffer(), DEFAULT_TENSOR_SPEC, TensorTopology{}) {}
HostTensor::~HostTensor() = default;

// Deep copy (of the config and topology?)
HostTensor::HostTensor(const HostTensor& other) :
    impl(other.impl ? std::make_unique<TensorAttributes>(*other.impl) : nullptr) {}

HostTensor& HostTensor::operator=(const HostTensor& other) {
    if (this != &other) {
        impl = other.impl ? std::make_unique<TensorAttributes>(*other.impl) : nullptr;
    }
    return *this;
}

// Move - default works for unique_ptr, this means operating anything on a move-from HostTensor is undefined?
HostTensor::HostTensor(HostTensor&&) noexcept = default;
HostTensor& HostTensor::operator=(HostTensor&&) noexcept = default;

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

// Should just change the return type to some flavor of vector
Shape HostTensor::strides() const {
    // This is a copy just to convert from size_t to uint32_t...
    // Stinky!
    auto s = compute_strides(padded_shape());
    return Shape(ttsl::SmallVector<uint32_t>(s.begin(), s.end()));
}

// Computed Getters (following tensor.cpp:411-461)

HostTensor::volumn_type HostTensor::logical_volume() const { return logical_shape().volume(); }

HostTensor::volumn_type HostTensor::padded_volume() const { return padded_shape().volume(); }

// Host tensors are never sharded
bool HostTensor::is_sharded() const { return false; }

std::size_t HostTensor::element_size() const { return data_type_size(dtype()); }

// Storage Access (following tensor.cpp:629-632 pattern)
// TODO: figure out if we're doing DistributedHostBuffer, hardcoding (0,0) is horrifying.
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
