// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>
#include <type_traits>

#include <internal/tensor/host_to_tensor_spec_apis.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/host_buffer.hpp>

#include <tt_stl/span.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                                  .to_tensor_spec()
// ======================================================================================

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

bool exact_spec_match(const TensorSpec& a, const TensorSpec& b) {
    return a == b && experimental::per_core_allocation::is_per_core_allocation(a.memory_config()) ==
                         experimental::per_core_allocation::is_per_core_allocation(b.memory_config());
}

bool is_bfp_dtype(DataType dtype) { return dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B; }

bool is_integral_dtype(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::INT32;
}

template <typename DestInt, typename PadT>
void validate_float_pad_against_integral(PadT pad_value) {
    const double pad_as_double = static_cast<double>(static_cast<float>(pad_value));
    TT_FATAL(
        std::isfinite(pad_as_double),
        "to_tensor_spec: floating pad_value must be finite when destination dtype is integral (got {})",
        pad_as_double);
    const double lo = static_cast<double>(std::numeric_limits<DestInt>::lowest());
    const double hi = static_cast<double>(std::numeric_limits<DestInt>::max());
    TT_FATAL(
        pad_as_double >= lo && pad_as_double <= hi,
        "to_tensor_spec: floating pad_value {} is out of range for integral destination [{}, {}]",
        pad_as_double,
        lo,
        hi);
}

template <typename PadT>
void validate_pad_for_integral_dest(PadT pad_value, DataType dest_dtype) {
    if constexpr (!(std::is_same_v<PadT, float> || std::is_same_v<PadT, bfloat16>)) {
        return;
    }
    switch (dest_dtype) {
        case DataType::UINT8: validate_float_pad_against_integral<uint8_t>(pad_value); break;
        case DataType::UINT16: validate_float_pad_against_integral<uint16_t>(pad_value); break;
        case DataType::UINT32: validate_float_pad_against_integral<uint32_t>(pad_value); break;
        case DataType::INT32: validate_float_pad_against_integral<int32_t>(pad_value); break;
        default: break;
    }
}

void assert_packed_shard_sizes(const DistributedHostBuffer& buffer, const TensorSpec& spec) {
    const size_t expected_shard_size = spec.compute_packed_buffer_size_bytes();
    for (const auto& coord : buffer.shard_coords()) {
        auto shard = buffer.get_shard(coord);
        if (shard) {
            TT_FATAL(
                shard->view_bytes().size() == expected_shard_size,
                "to_tensor_spec shard size mismatch: actual {} != expected {}",
                shard->view_bytes().size(),
                expected_shard_size);
        }
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

template <typename T>
HostTensor to_tensor_spec(const HostTensor& tensor, const TensorSpec& dest_spec, T pad_value) {
    TT_FATAL(
        tensor.logical_shape().rank() > 0,
        "to_tensor_spec: rank-0 tensors are unsupported (got rank {})",
        tensor.logical_shape().rank());
    TT_FATAL(
        tensor.logical_shape() == dest_spec.logical_shape(),
        "to_tensor_spec: logical shapes must match (src {}, dest {})",
        tensor.logical_shape(),
        dest_spec.logical_shape());
    TT_FATAL(
        tensor.dtype() != DataType::FP8_E4M3 && dest_spec.data_type() != DataType::FP8_E4M3,
        "to_tensor_spec: FP8_E4M3 source/destination is unsupported in v1");

    const bool bfp_src = CMAKE_UNIQUE_NAMESPACE::is_bfp_dtype(tensor.dtype());
    const bool bfp_dst = CMAKE_UNIQUE_NAMESPACE::is_bfp_dtype(dest_spec.data_type());
    const DataType working_encode_dtype = (bfp_src || bfp_dst) ? DataType::FLOAT32 : tensor.dtype();
    const auto pad_dtype = convert_to_data_type<T>();
    TT_FATAL(
        pad_dtype == working_encode_dtype,
        "to_tensor_spec: pad/encode type {} must match working encode dtype {} "
        "(BFP paths require float; otherwise T must match the post-decode source dtype)",
        pad_dtype,
        working_encode_dtype);

    if (CMAKE_UNIQUE_NAMESPACE::exact_spec_match(tensor.tensor_spec(), dest_spec)) {
        return tensor;
    }

    HostTensor source_for_decode = tensor;
    if (working_encode_dtype == DataType::FLOAT32 && tensor.dtype() != DataType::FLOAT32) {
        // Mandatory BFP staging (and BFP-dest float pivot): convert to FLOAT32 before decode.
        source_for_decode = to_dtype(tensor, DataType::FLOAT32);
    }

    const TensorSpec& decode_spec = source_for_decode.tensor_spec();
    const TensorSpec working_spec(
        dest_spec.logical_shape(),
        TensorLayout(
            working_encode_dtype,
            dest_spec.page_config(),
            dest_spec.memory_config(),
            dest_spec.tensor_layout().get_alignment()));

    // If floating pad will land in padded regions that a later integral to_dtype casts, reject bad pads first.
    if (CMAKE_UNIQUE_NAMESPACE::is_integral_dtype(dest_spec.data_type()) &&
        working_spec.logical_2d_shape() != working_spec.physical_shape()) {
        CMAKE_UNIQUE_NAMESPACE::validate_pad_for_integral_dest(pad_value, dest_spec.data_type());
    }

    auto transformed_buffer = source_for_decode.buffer().transform(
        [&](const HostBuffer& buffer) {
            auto physical = buffer.view_as<const T>();
            auto logical = tensor_impl::decode_tensor_data(physical, decode_spec);
            auto encoded = tensor_impl::encode_tensor_data(ttsl::make_const_span(logical), working_spec, pad_value);
            return HostBuffer(std::move(encoded));
        },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    CMAKE_UNIQUE_NAMESPACE::assert_packed_shard_sizes(transformed_buffer, working_spec);

    HostTensor result =
        host_tensor_from_buffer_with_topology(std::move(transformed_buffer), working_spec, get_tensor_topology(tensor));

    if (result.dtype() != dest_spec.data_type()) {
        result = to_dtype(result, dest_spec.data_type());
    }

    TT_FATAL(
        CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec),
        "to_tensor_spec: result does not satisfy exact-spec predicate against dest_spec");
    CMAKE_UNIQUE_NAMESPACE::assert_packed_shard_sizes(result.buffer(), dest_spec);
    TT_FATAL(get_tensor_topology(result) == get_tensor_topology(tensor), "to_tensor_spec: topology must be preserved");

    return result;
}

template HostTensor to_tensor_spec<float>(const HostTensor&, const TensorSpec&, float);
template HostTensor to_tensor_spec<bfloat16>(const HostTensor&, const TensorSpec&, bfloat16);
template HostTensor to_tensor_spec<int32_t>(const HostTensor&, const TensorSpec&, int32_t);
template HostTensor to_tensor_spec<uint32_t>(const HostTensor&, const TensorSpec&, uint32_t);
template HostTensor to_tensor_spec<uint16_t>(const HostTensor&, const TensorSpec&, uint16_t);
template HostTensor to_tensor_spec<uint8_t>(const HostTensor&, const TensorSpec&, uint8_t);

}  // namespace tt::tt_metal
