// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_util.hpp"

#include "autograd/auto_context.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"

namespace UnsupportedMessages {
constexpr auto BFLOAT8_B = "Unsupported type: BFLOAT8_B";
constexpr auto BFLOAT4_B = "Unsupported type: BFLOAT4_B";
constexpr auto UINT8 = "Unsupported type: UINT8";
constexpr auto UINT16 = "Unsupported type: UINT16";
constexpr auto INVALID = "Unsupported type: INVALID";
constexpr auto UNKNOWN = "Unsupported type: unknown";
constexpr auto COMPLEX = "Unsupported type: Complex";
constexpr auto BOOL = "Unsupported type: Bool";
constexpr auto BFLOAT = "Unsupported type: Bfloat";
}  // namespace UnsupportedMessages

nb::ndarray<nb::numpy> make_numpy_tensor(
    const tt::tt_metal::Tensor& t, std::optional<tt::tt_metal::DataType> new_type) {
    auto const make_numpy_tensor_from_data =
        []<typename NumpyType>(
            const auto& tensor_data, const auto& tensor_spec, [[maybe_unused]] const auto& tensor_strides) {
            const tt::tt_metal::Shape& tensor_shape = tensor_spec.logical_shape();

            const auto tensor_shape_rank = tensor_shape.rank();
            std::vector<size_t> numpy_shape(tensor_shape_rank);
            std::copy(tensor_shape.cbegin(), tensor_shape.cend(), numpy_shape.begin());

            // std::vector<int64_t> numpy_strides;
            // numpy_strides.assign(tensor_strides.cbegin(), tensor_strides.cend());

            auto* numpy_data = new NumpyType[tensor_data.size()];
            std::copy(tensor_data.begin(), tensor_data.end(), numpy_data);

            const nb::capsule owner(numpy_data, [](void* p) noexcept { delete[] static_cast<NumpyType*>(p); });
            return nb::ndarray<nb::numpy>(
                numpy_data,
                tensor_shape_rank,
                numpy_shape.data(),
                owner,
                nullptr /*numpy_strides.data()*/,
                nb::dtype<NumpyType>());
        };

    auto const convert_to_row_major = [](const tt::tt_metal::Tensor& tensor) {
        tt::tt_metal::Shape output_tensor_end(ttsl::SmallVector<uint32_t>(tensor.logical_shape().rank(), 0));
        int logical_rank = tensor.logical_shape().rank();
        for (int index = -1; index >= -logical_rank; --index) {
            output_tensor_end[index] = tensor.logical_shape()[index] - 1;
        }

        return ttnn::untilize_with_unpadding(tensor, output_tensor_end, std::nullopt);
    };

    auto const impl = [&make_numpy_tensor_from_data, &convert_to_row_major]<typename MetalType, typename NumpyType>(
                          const tt::tt_metal::Tensor& tensor) {
        static_assert(!std::is_same_v<bfloat16, NumpyType>, "Numpy does not support bfloat16, use float");
        if (tensor.storage_type() == ttnn::types::StorageType::HOST) {
            if (tensor.layout() != tt::tt_metal::Layout::ROW_MAJOR) {
                auto const row_major_tensor = convert_to_row_major(tensor);
                const auto row_major_tensor_data = tt::tt_metal::host_buffer::get_as<MetalType const>(row_major_tensor);
                return make_numpy_tensor_from_data.template operator()<NumpyType>(
                    row_major_tensor_data, row_major_tensor.tensor_spec(), row_major_tensor.strides());
            }
            const auto tensor_data = tt::tt_metal::host_buffer::get_as<const MetalType>(tensor);
            return make_numpy_tensor_from_data.template operator()<NumpyType>(
                tensor_data, tensor.tensor_spec(), tensor.strides());
        }
        const auto cpu_tensor = tensor.cpu(/*blocking=*/true);
        const auto cpu_tensor_data = tt::tt_metal::host_buffer::get_as<const MetalType>(cpu_tensor);
        const auto cpu_tensor_spec = cpu_tensor.tensor_spec();
        const auto cpu_tensor_strides = cpu_tensor.strides();

        if (tt::tt_metal::tensor_impl::logical_matches_physical(cpu_tensor_spec)) {
            return make_numpy_tensor_from_data.template operator()<NumpyType>(
                cpu_tensor_data, cpu_tensor_spec, cpu_tensor_strides);
        }

        const auto decoded_data = tt::tt_metal::tensor_impl::decode_tensor_data(cpu_tensor_data, cpu_tensor_spec);
        return make_numpy_tensor_from_data.template operator()<NumpyType>(
            decoded_data, cpu_tensor_spec, cpu_tensor_strides);
    };

    const auto& tensor_spec = t.tensor_spec();
    const auto tensor_type = tensor_spec.data_type();
    if (!new_type.has_value()) {
        switch (tensor_type) {
            case tt::tt_metal::DataType::INT32: return impl.template operator()<int32_t, int32_t>(t);
            case tt::tt_metal::DataType::UINT32: return impl.template operator()<uint32_t, uint32_t>(t);
            case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<float, float>(t);
            // impl<bfloat16, float>(t) is intentional below
            case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<bfloat16, float>(t);
            case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW(UnsupportedMessages::BFLOAT8_B); break;
            case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW(UnsupportedMessages::BFLOAT4_B); break;
            case tt::tt_metal::DataType::UINT8: TT_THROW(UnsupportedMessages::UINT8); break;
            case tt::tt_metal::DataType::UINT16: TT_THROW(UnsupportedMessages::UINT16); break;
            case tt::tt_metal::DataType::INVALID: TT_THROW(UnsupportedMessages::INVALID); break;
        }

        TT_THROW(UnsupportedMessages::UNKNOWN);
    }
    switch (tensor_type) {
        case tt::tt_metal::DataType::INT32:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<int32_t, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<int32_t, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<int32_t, float>(t);
                // impl<int32_t, float>(t) is intentional below
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<int32_t, float>(t);
                case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW(UnsupportedMessages::BFLOAT8_B); break;
                case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW(UnsupportedMessages::BFLOAT4_B); break;
                case tt::tt_metal::DataType::UINT8: TT_THROW(UnsupportedMessages::UINT8); break;
                case tt::tt_metal::DataType::UINT16: TT_THROW(UnsupportedMessages::UINT16); break;
                case tt::tt_metal::DataType::INVALID: TT_THROW(UnsupportedMessages::INVALID); break;
            }
        case tt::tt_metal::DataType::UINT32:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<uint32_t, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<uint32_t, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<uint32_t, float>(t);
                // impl<uint32_t, float>(t) is intentional below
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<uint32_t, float>(t);
                case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW(UnsupportedMessages::BFLOAT8_B); break;
                case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW(UnsupportedMessages::BFLOAT4_B); break;
                case tt::tt_metal::DataType::UINT8: TT_THROW(UnsupportedMessages::UINT8); break;
                case tt::tt_metal::DataType::UINT16: TT_THROW(UnsupportedMessages::UINT16); break;
                case tt::tt_metal::DataType::INVALID: TT_THROW(UnsupportedMessages::INVALID); break;
            }
        case tt::tt_metal::DataType::FLOAT32:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<float, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<float, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<float, float>(t);
                // impl<float, float>(t) is intentional below
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<float, float>(t);
                case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW(UnsupportedMessages::BFLOAT8_B); break;
                case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW(UnsupportedMessages::BFLOAT4_B); break;
                case tt::tt_metal::DataType::UINT8: TT_THROW(UnsupportedMessages::UINT8); break;
                case tt::tt_metal::DataType::UINT16: TT_THROW(UnsupportedMessages::UINT16); break;
                case tt::tt_metal::DataType::INVALID: TT_THROW(UnsupportedMessages::INVALID); break;
            }
        case tt::tt_metal::DataType::BFLOAT16:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<bfloat16, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<bfloat16, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<bfloat16, float>(t);
                // impl<bfloat16, float>(t) is intentional below
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<bfloat16, float>(t);
                case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW(UnsupportedMessages::BFLOAT8_B); break;
                case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW(UnsupportedMessages::BFLOAT4_B); break;
                case tt::tt_metal::DataType::UINT8: TT_THROW(UnsupportedMessages::UINT8); break;
                case tt::tt_metal::DataType::UINT16: TT_THROW(UnsupportedMessages::UINT16); break;
                case tt::tt_metal::DataType::INVALID: TT_THROW(UnsupportedMessages::INVALID); break;
            }
        case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW(UnsupportedMessages::BFLOAT8_B); break;
        case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW(UnsupportedMessages::BFLOAT4_B); break;
        case tt::tt_metal::DataType::UINT8: TT_THROW(UnsupportedMessages::UINT8); break;
        case tt::tt_metal::DataType::UINT16: TT_THROW(UnsupportedMessages::UINT16); break;
        case tt::tt_metal::DataType::INVALID: TT_THROW(UnsupportedMessages::INVALID); break;
    }

    TT_THROW(UnsupportedMessages::UNKNOWN);
}

tt::tt_metal::Tensor make_metal_tensor(
    nb::ndarray<> data, tt::tt_metal::Layout target_layout, std::optional<tt::tt_metal::DataType> new_type) {
    const auto data_type = data.dtype();
    TT_FATAL(!(data_type.bits % 8), "Unsupported precision: {} bits", data_type.bits);

    const auto rank = data.ndim();

    const auto impl = [&data_type, rank, &data, target_layout]<typename NumpyType>(
                          tt::tt_metal::DataType tensor_data_type) {
        static_assert(std::is_same_v<NumpyType, std::remove_cvref_t<NumpyType>>);
        TT_FATAL(
            data_type.bits == (sizeof(NumpyType) * 8),
            "Unsupported precision: expected {} bits, got {} bits",
            sizeof(NumpyType) * 8,
            data_type.bits);

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
        // Our tensor will initially be created from row-major numpy data, regardless of target layout
        const tt::tt_metal::PageConfig tensor_page_config(tt::tt_metal::Layout::ROW_MAJOR);
        tt::tt_metal::TensorLayout tensor_layout(tensor_data_type, tensor_page_config, tensor_memory_config);
        tt::tt_metal::TensorSpec tensor_spec(tensor_shape, tensor_layout);

        const auto make_device_tensor = [&]<typename MetalType>() -> tt::tt_metal::Tensor {
            auto* device = &ttml::autograd::ctx().get_device();
            // device->enable_program_cache();

            std::span<const NumpyType> data_span(static_cast<const NumpyType*>(data.data()), data.size());
            if constexpr (sizeof(MetalType) != sizeof(NumpyType)) {
                std::vector<MetalType> converted_data;
                converted_data.assign(data_span.begin(), data_span.end());

                auto row_major_tensor = tt::tt_metal::Tensor::from_vector(converted_data, tensor_spec, device);

                if (target_layout == tt::tt_metal::Layout::ROW_MAJOR) {
                    return row_major_tensor;
                }
                return ttnn::tilize_with_zero_padding(row_major_tensor);
            } else {
                auto row_major_tensor = tt::tt_metal::Tensor::from_span(data_span, tensor_spec, device);

                if (target_layout == tt::tt_metal::Layout::ROW_MAJOR) {
                    return row_major_tensor;
                }
                return ttnn::tilize_with_zero_padding(row_major_tensor);
            }
        };

        switch (tensor_data_type) {
            case tt::tt_metal::DataType::INT32: return make_device_tensor.template operator()<int32_t>();
            case tt::tt_metal::DataType::UINT32: return make_device_tensor.template operator()<uint32_t>();
            case tt::tt_metal::DataType::FLOAT32: return make_device_tensor.template operator()<float>();
            case tt::tt_metal::DataType::BFLOAT16: return make_device_tensor.template operator()<bfloat16>();
            case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW(UnsupportedMessages::BFLOAT8_B); break;
            case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW(UnsupportedMessages::BFLOAT4_B); break;
            case tt::tt_metal::DataType::UINT8: TT_THROW(UnsupportedMessages::UINT8); break;
            case tt::tt_metal::DataType::UINT16: TT_THROW(UnsupportedMessages::UINT16); break;
            case tt::tt_metal::DataType::INVALID: TT_THROW(UnsupportedMessages::INVALID); break;
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
            //  seems like bfloat is not bfloat16
            // return impl.template operator()<bfloat16>(new_type.value_or(tt::tt_metal::DataType::BFLOAT16));
            TT_THROW(UnsupportedMessages::BFLOAT);
            break;
        case nb::dlpack::dtype_code::Complex: TT_THROW(UnsupportedMessages::COMPLEX); break;
        case nb::dlpack::dtype_code::Bool: TT_THROW(UnsupportedMessages::BOOL); break;
    }

    TT_THROW(UnsupportedMessages::UNKNOWN);
}
