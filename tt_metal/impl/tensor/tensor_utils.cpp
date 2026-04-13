// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>

namespace tt::tt_metal {

bool logical_matches_physical(const TensorSpec& tensor_spec) {
    return tensor_spec.layout() == Layout::ROW_MAJOR && tensor_spec.logical_2d_shape() == tensor_spec.physical_shape();
}

namespace host_buffer {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <typename T>
void validate_datatype(DataType dtype) {
    using BaseType = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<BaseType, uint32_t>) {
        TT_FATAL(
            dtype == DataType::UINT32 or dtype == DataType::BFLOAT8_B or dtype == DataType::BFLOAT4_B,
            "Incorrect data type {}",
            dtype);
    } else if constexpr (std::is_same_v<BaseType, int32_t>) {
        TT_FATAL(dtype == DataType::INT32, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, float>) {
        TT_FATAL(dtype == DataType::FLOAT32, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, bfloat16>) {
        TT_FATAL(dtype == DataType::BFLOAT16, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, uint16_t>) {
        TT_FATAL(dtype == DataType::UINT16, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, uint8_t>) {
        TT_FATAL(dtype == DataType::UINT8, "Incorrect data type {}", dtype);
    } else {
        static_assert(sizeof(BaseType) == 0, "Unsupported DataType");
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostBuffer get_host_buffer(const HostTensor& tensor) {
    std::vector<HostBuffer> buffers;
    tensor.buffer().apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
    TT_FATAL(
        buffers.size() == 1,
        "Can't get a single buffer from host storage distributed over mesh shape {}",
        tensor.buffer().shape());
    return buffers.front();
}

template <typename T>
ttsl::Span<const T> get_as(const HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
ttsl::Span<T> get_as(HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
ttsl::Span<const T> get_as(const HostTensor& tensor) {
    CMAKE_UNIQUE_NAMESPACE::validate_datatype<T>(tensor.dtype());
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

template <typename T>
ttsl::Span<T> get_as(HostTensor& tensor) {
    CMAKE_UNIQUE_NAMESPACE::validate_datatype<T>(tensor.dtype());
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

// Explicit template instantiations
#define INSTANTIATE_HOST_BUFFER_FUNCTIONS(T)                         \
    template ttsl::Span<const T> get_as<T>(const HostBuffer&);       \
    template ttsl::Span<const T> get_as<const T>(const HostBuffer&); \
    template ttsl::Span<T> get_as<T>(HostBuffer&);                   \
    template ttsl::Span<const T> get_as<const T>(HostBuffer&);       \
    template ttsl::Span<const T> get_as<T>(const HostTensor&);       \
    template ttsl::Span<const T> get_as<const T>(const HostTensor&); \
    template ttsl::Span<T> get_as<T>(HostTensor&);                   \
    template ttsl::Span<const T> get_as<const T>(HostTensor&);

INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(int32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(float)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(bfloat16)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint16_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint8_t)

#undef INSTANTIATE_HOST_BUFFER_FUNCTIONS

}  // namespace host_buffer

}  // namespace tt::tt_metal
