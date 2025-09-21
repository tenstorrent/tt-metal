// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_util.hpp"

#include "autograd/auto_context.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"

nb::ndarray<nb::numpy> make_numpy_tensor(
    const tt::tt_metal::Tensor& t, std::optional<tt::tt_metal::DataType> new_type) {
    auto const numpy_tensor_from_data = []<typename U>(
                                            const auto& tensor_data,
                                            const auto& tensor_spec,
                                            const auto& tensor_strides) {
        const tt::tt_metal::Shape& tensor_shape = tensor_spec.logical_shape();

        const auto tensor_shape_rank = tensor_shape.rank();
        std::vector<size_t> numpy_shape(tensor_shape_rank);
        std::copy(tensor_shape.cbegin(), tensor_shape.cend(), numpy_shape.begin());

        // std::vector<int64_t> numpy_strides(tensor_strides.rank());
        // std::copy(tensor_strides.cbegin(), tensor_strides.cend(), numpy_strides.begin());

        U* numpy_data = new U[tensor_data.size()];
        std::copy(tensor_data.begin(), tensor_data.end(), numpy_data);

        const nb::capsule owner(numpy_data, [](void* p) noexcept { delete[] static_cast<U*>(p); });
        return nb::ndarray<nb::numpy>(
            numpy_data, tensor_shape_rank, numpy_shape.data(), owner, nullptr /*numpy_strides.data()*/, nb::dtype<U>());
    };

    auto const convert_to_row_major = [](const tt::tt_metal::Tensor& tensor) {
        tt::tt_metal::Shape output_tensor_end(ttsl::SmallVector<uint32_t>(tensor.logical_shape().rank(), 0));
        int logical_rank = tensor.logical_shape().rank();
        for (int index = -1; index >= -logical_rank; --index) {
            output_tensor_end[index] = tensor.logical_shape()[index] - 1;
        }

        return ttnn::untilize_with_unpadding(tensor, output_tensor_end, std::nullopt);
    };

    auto const impl = [&numpy_tensor_from_data,
                       &convert_to_row_major]<typename T, typename U>(const tt::tt_metal::Tensor& tensor) {
        if (tensor.storage_type() == ttnn::types::StorageType::HOST) {
            if (tensor.layout() != tt::tt_metal::Layout::ROW_MAJOR) {
                auto const row_major_tensor = convert_to_row_major(tensor);
                const auto row_major_tensor_data = tt::tt_metal::host_buffer::get_as<T const>(row_major_tensor);
                return numpy_tensor_from_data.template operator()<U>(
                    row_major_tensor_data, row_major_tensor.tensor_spec(), row_major_tensor.strides());
            }
            const auto tensor_data = tt::tt_metal::host_buffer::get_as<const T>(tensor);
            return numpy_tensor_from_data.template operator()<U>(tensor_data, tensor.tensor_spec(), tensor.strides());
        }
        const auto cpu_tensor = tensor.cpu(/*blocking=*/true);
        const auto cpu_tensor_data = tt::tt_metal::host_buffer::get_as<const T>(cpu_tensor);
        const auto cpu_tensor_spec = tensor.tensor_spec();
        const auto cpu_tensor_strides = tensor.strides();

        if (tt::tt_metal::tensor_impl::logical_matches_physical(cpu_tensor_spec)) {
            return numpy_tensor_from_data.template operator()<U>(cpu_tensor_data, cpu_tensor_spec, cpu_tensor_strides);
        }

        const auto decoded_data = tt::tt_metal::tensor_impl::decode_tensor_data(cpu_tensor_data, cpu_tensor_spec);
        return numpy_tensor_from_data.template operator()<U>(decoded_data, cpu_tensor_spec, cpu_tensor_strides);
    };

    const auto& tensor_spec = t.tensor_spec();
    const auto tensor_type = tensor_spec.data_type();
    if (!new_type.has_value()) {
        switch (tensor_type) {
            case tt::tt_metal::DataType::INT32: return impl.template operator()<int32_t, int32_t>(t);
            case tt::tt_metal::DataType::UINT32: return impl.template operator()<uint32_t, uint32_t>(t);
            case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<float, float>(t);
            case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<bfloat16, bfloat16>(t);
            case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
            case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
            case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
            case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
            case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
        }

        TT_THROW("Unsupported type: unknown");
    }
    switch (tensor_type) {
        case tt::tt_metal::DataType::INT32:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<int32_t, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<int32_t, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<int32_t, float>(t);
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<int32_t, bfloat16>(t);
                case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
                case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
                case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
                case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
                case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
            }
        case tt::tt_metal::DataType::UINT32:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<uint32_t, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<uint32_t, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<uint32_t, float>(t);
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<uint32_t, bfloat16>(t);
                case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
                case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
                case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
                case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
                case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
            }
        case tt::tt_metal::DataType::FLOAT32:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<float, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<float, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<float, float>(t);
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<float, bfloat16>(t);
                case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
                case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
                case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
                case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
                case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
            }
        case tt::tt_metal::DataType::BFLOAT16:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<bfloat16, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<bfloat16, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<bfloat16, float>(t);
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<bfloat16, bfloat16>(t);
                case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
                case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
                case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
                case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
                case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
            }
        case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
        case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
        case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
        case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
        case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
    }

    TT_THROW("Unsupported type: unknown");
}

tt::tt_metal::Tensor make_metal_tensor(
    nb::ndarray<> data, tt::tt_metal::Layout layout, std::optional<tt::tt_metal::DataType> new_type) {
    const auto data_type = data.dtype();
    TT_FATAL(!(data_type.bits % 8), "Unsupported precision: {} bits", data_type.bits);

    const auto rank = data.ndim();

    const auto impl = [&data_type, rank, &data, layout]<typename T>(tt::tt_metal::DataType tensor_data_type) {
        using U = std::remove_cvref_t<T>;
        const auto types_match = [](tt::tt_metal::DataType dt) {
            switch (dt) {
                case tt::tt_metal::DataType::INT32: return std::is_same_v<U, int32_t>;
                case tt::tt_metal::DataType::UINT32: return std::is_same_v<U, uint32_t>;
                case tt::tt_metal::DataType::FLOAT32: return std::is_same_v<U, float>;
                case tt::tt_metal::DataType::BFLOAT16: return std::is_same_v<U, bfloat16>;
                case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
                case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
                case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
                case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
                case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
            }

            TT_THROW("Unsupported type: unknown");
            return false;
        };

        TT_FATAL(
            data_type.bits == (sizeof(T) * 8),
            "Unsupported precision: expected {} bits, got {} bits",
            sizeof(T) * 8,
            data_type.bits);

        auto* device = &ttml::autograd::ctx().get_device();
        // device->enable_program_cache();

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
        tt::tt_metal::TensorLayout tensor_layout(tensor_data_type, tensor_page_config, tensor_memory_config);
        tt::tt_metal::TensorSpec tensor_spec(tensor_shape, tensor_layout);
        if (types_match(tensor_data_type)) {
            auto tensor = tt::tt_metal::Tensor::from_span(
                ttsl::Span<const T>(static_cast<const T*>(data.data()), data.size()), tensor_spec, device);
            if (layout == tt::tt_metal::Layout::ROW_MAJOR) {
                return tensor.to_device(device, tensor_memory_config);
            }

            auto device_tensor = tensor.to_device(device, tensor_memory_config);
            return ttnn::tilize_with_zero_padding(device_tensor);
        }
        const auto convert_to_type = [&]<typename Type>() {
            std::span<U const> data_span(static_cast<const U*>(data.data()), data.size());
            std::vector<Type> new_data;
            new_data.assign(data_span.begin(), data_span.end());
            auto tensor = tt::tt_metal::Tensor::from_vector(new_data, tensor_spec, device);
            if (layout == tt::tt_metal::Layout::ROW_MAJOR) {
                return tensor.to_device(device, tensor_memory_config);
            }
            auto device_tensor = tensor.to_device(device, tensor_memory_config);
            return ttnn::tilize_with_zero_padding(device_tensor);
        };
        switch (tensor_data_type) {
            case tt::tt_metal::DataType::INT32: return convert_to_type.template operator()<int32_t>();
            case tt::tt_metal::DataType::UINT32: return convert_to_type.template operator()<uint32_t>();
            case tt::tt_metal::DataType::FLOAT32: return convert_to_type.template operator()<float>();
            case tt::tt_metal::DataType::BFLOAT16: return convert_to_type.template operator()<bfloat16>();
            case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
            case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
            case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
            case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
            case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
        }
    };

    switch (static_cast<nb::dlpack::dtype_code>(data_type.code)) {
        case nb::dlpack::dtype_code::Int:
            return impl.template operator()<int32_t>(new_type.value_or(tt::tt_metal::DataType::INT32));
        case nb::dlpack::dtype_code::UInt:
            return impl.template operator()<uint32_t>(new_type.value_or(tt::tt_metal::DataType::UINT32));
        case nb::dlpack::dtype_code::Float:
            return impl.template operator()<float>(new_type.value_or(tt::tt_metal::DataType::FLOAT32));
        case nb::dlpack::dtype_code::Bfloat:
            return impl.template operator()<bfloat16>(new_type.value_or(tt::tt_metal::DataType::BFLOAT16));
        case nb::dlpack::dtype_code::Complex: TT_THROW("Unsupported type: Complex"); break;
        case nb::dlpack::dtype_code::Bool: TT_THROW("Unsupported type: Bool"); break;
    }

    TT_THROW("Unsupported type: unknown");
}
