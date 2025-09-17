// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_util.hpp"

#include "autograd/auto_context.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"

nb::ndarray<nb::numpy> make_numpy_tensor(const tt::tt_metal::Tensor& tensor) {
    const auto cpu_tensor = tensor.cpu(/*blocking=*/true, ttnn::DefaultQueueId);
    const auto tensor_spec = cpu_tensor.tensor_spec();
    const auto impl = [&tensor_spec]<typename T>(const tt::tt_metal::Tensor& t) {
        const tt::tt_metal::Shape& tensor_shape = tensor_spec.logical_shape();

        const auto tensor_shape_rank = tensor_shape.rank();
        std::vector<size_t> numpy_shape(tensor_shape_rank);
        std::copy(tensor_shape.cbegin(), tensor_shape.cend(), numpy_shape.begin());

        const auto tensor_strides = t.strides();
        std::vector<int64_t> numpy_strides(tensor_strides.rank());
        std::copy(tensor_strides.cbegin(), tensor_strides.cend(), numpy_strides.begin());

        const auto tensor_data = tt::tt_metal::host_buffer::get_as<const T>(t);

        const auto return_data_copy_with_capsule = [&](const auto& src) {
            T* numpy_data = new T[src.size()];
            std::copy(src.begin(), src.end(), numpy_data);

            const nb::capsule owner(numpy_data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
            return nb::ndarray<nb::numpy>(
                numpy_data, tensor_shape_rank, numpy_shape.data(), owner, numpy_strides.data(), nb::dtype<T>());
        };

        return return_data_copy_with_capsule(t.to_vector<T>());
        // if (tt::tt_metal::tensor_impl::logical_matches_physical(t.tensor_spec())) {
        //     return return_data_copy_with_capsule(tensor_data);
        // }
        // const auto decoded_data = tt::tt_metal::tensor_impl::decode_tensor_data(tensor_data, t.tensor_spec());
        // return return_data_copy_with_capsule(decoded_data);
    };

    const auto ensure_row_major = [&impl]<typename T>(const tt::tt_metal::Tensor& t) {
        if (t.layout() != tt::tt_metal::Layout::ROW_MAJOR) {
            return impl.template operator()<T>(
                ttnn::untilize_with_unpadding(ttnn::DefaultQueueId, t, t.logical_shape(), std::nullopt));
        }
        return impl.template operator()<T>(t);
    };

    switch (tensor_spec.data_type()) {
        case tt::tt_metal::DataType::INT32: return ensure_row_major.template operator()<int32_t>(tensor);
        case tt::tt_metal::DataType::UINT32: return ensure_row_major.template operator()<uint32_t>(tensor);
        case tt::tt_metal::DataType::FLOAT32: return ensure_row_major.template operator()<float>(tensor);
        case tt::tt_metal::DataType::BFLOAT16: return ensure_row_major.template operator()<bfloat16>(tensor);
        case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
        case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
        case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
        case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
        case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
    }

    TT_THROW("Unsupported type: unknown");
}

tt::tt_metal::Tensor make_metal_tensor(nb::ndarray<> data) {
    const auto data_type = data.dtype();
    TT_FATAL(!(data_type.bits % 8), "Unsupported precision: {} bits", data_type.bits);

    const auto rank = data.ndim();
    tt::tt_metal::ShapeBase::Container shape_container(rank);
    for (size_t dimension = 0; dimension < rank; ++dimension) {
        const auto dimension_size = data.shape(dimension);
        TT_FATAL(
            dimension_size >= std::numeric_limits<uint32_t>::min(),
            "Invalid shape parameter for dimension {}: {} is too small",
            dimension,
            dimension_size);
        TT_FATAL(
            dimension_size <= std::numeric_limits<uint32_t>::max(),
            "Invalid shape parameter for dimension {}: {} is too large",
            dimension,
            dimension_size);
        shape_container[dimension] = dimension_size;
    }
    const tt::tt_metal::Shape tensor_shape(shape_container);
    const tt::tt_metal::MemoryConfig tensor_memory_config{};
    const tt::tt_metal::PageConfig tensor_page_config(tt::tt_metal::Layout::ROW_MAJOR);

    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    const auto impl = [&]<typename T>(tt::tt_metal::DataType tensor_data_type) {
        TT_FATAL(
            data_type.bits == (sizeof(T) * 8),
            "Unsupported precision: expected {} bits, got {} bits",
            sizeof(T) * 8,
            data_type.bits);

        tt::tt_metal::TensorLayout tensor_layout(tensor_data_type, tensor_page_config, tensor_memory_config);
        tt::tt_metal::TensorSpec tensor_spec(tensor_shape, tensor_layout);
        return ttnn::tilize_with_zero_padding(
                   ttnn::DefaultQueueId,
                   tt::tt_metal::Tensor::from_span(
                       tt::stl::Span<const T>(static_cast<const T*>(data.data()), data.size()), tensor_spec, device))
            .to_device(device, tensor_memory_config);
    };

    switch (static_cast<nb::dlpack::dtype_code>(data_type.code)) {
        case nb::dlpack::dtype_code::Int:
            return impl.template operator()<int32_t>(tt::tt_metal::DataType::INT32);
            break;
        case nb::dlpack::dtype_code::UInt:
            return impl.template operator()<uint32_t>(tt::tt_metal::DataType::UINT32);
            break;
        case nb::dlpack::dtype_code::Float:
            return impl.template operator()<float>(tt::tt_metal::DataType::FLOAT32);
            break;
        case nb::dlpack::dtype_code::Bfloat:
            return impl.template operator()<bfloat16>(tt::tt_metal::DataType::BFLOAT16);
            break;
        case nb::dlpack::dtype_code::Complex: TT_THROW("Unsupported type: Complex"); break;
        case nb::dlpack::dtype_code::Bool: TT_THROW("Unsupported type: Bool"); break;
    }

    TT_THROW("Unsupported type: unknown");
}
