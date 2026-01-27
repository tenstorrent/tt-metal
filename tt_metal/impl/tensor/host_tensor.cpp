// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/host_tensor.hpp>

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

// Default TensorAttributes for empty tensors
// TODO: lower this as a default constructor for TensorAttributes?
const TensorAttributes DEFAULT_TENSOR_ATTRIBUTES(
    HostStorage(HostBuffer()),
    TensorSpec(Shape(), TensorLayout(DataType::INVALID, PageConfig(Layout::INVALID), MemoryConfig())),
    TensorTopology{});

// Special Member Functions

// Main Constructor
HostTensor::HostTensor(const HostBuffer& buffer, TensorSpec spec, TensorTopology topology) :
    impl(std::make_unique<TensorAttributes>(Storage(HostStorage(buffer)), std::move(spec), std::move(topology))) {}

/*
 * Implementation note:
 * This might be better implmeneted as an empty std::unique_ptr?
 * expectation is this will always be assigned over.
 * it's just we will have to check for nullptr everywhere.
 * Right now we pay by assigning DEFAULT_TENSOR_ATTRIBUTES to the moved-from HostTensor and always allocate.
 */
HostTensor::HostTensor() : impl(std::make_unique<TensorAttributes>(DEFAULT_TENSOR_ATTRIBUTES)) {}
HostTensor::~HostTensor() = default;

HostTensor::HostTensor(const HostTensor& other) : impl(std::make_unique<TensorAttributes>(*other.impl)) {}
HostTensor& HostTensor::operator=(const HostTensor& other) {
    if (this != &other) {
        *impl = *other.impl;
    }
    return *this;
}

HostTensor::HostTensor(HostTensor&& other) noexcept : HostTensor() {
    std::swap(impl, other.impl);
    *other.impl = DEFAULT_TENSOR_ATTRIBUTES;
}
HostTensor& HostTensor::operator=(HostTensor&& other) noexcept {
    if (this != &other) {
        std::swap(impl, other.impl);
        *other.impl = DEFAULT_TENSOR_ATTRIBUTES;
    }
    return *this;
}

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

Strides HostTensor::strides() const { return impl->get_tensor_spec().compute_strides(); }

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
    std::string layout_str;
    switch (layout()) {
        case Layout::ROW_MAJOR: layout_str = "ROW_MAJOR"; break;
        case Layout::TILE: layout_str = "TILE"; break;
        default: layout_str = "INVALID"; break;
    }
    return fmt::format("HostTensor(shape={}, dtype={}, layout={})", logical_shape(), dtype(), layout_str);
}

// Mutation (stub per header TODO)
void HostTensor::reshape(/* */) {
    // TODO: Implement reshape - see tensor.cpp:417-422
}

}  // namespace tt::tt_metal
