// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_util.hpp"

#include <nanobind/nanobind.h>

#include "autograd/auto_context.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttml::nanobind::util {

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

[[noreturn]] void throw_exception(
    std::source_location source_location,
    nb::exception_type exception_type,
    auto exception_name,
    auto condition_str,
    auto msg) {
    throw nb::builtin_exception(
        exception_type,
        fmt::format(
            "[{}:{}] {}: ({}){}",
            source_location.file_name(),
            source_location.line(),
            exception_name,
            condition_str,
            msg)
            .c_str());
}

nb::ndarray<nb::numpy> make_numpy_tensor(
    const tt::tt_metal::Tensor& t,
    std::optional<tt::tt_metal::DataType> new_type,
    const ttnn::distributed::MeshToTensor* composer) {
    const auto make_numpy_tensor_from_data =
        []<typename NumpyType>(const auto& tensor_data, const auto& tensor_spec /*, const auto& tensor_strides*/) {
            const tt::tt_metal::Shape& tensor_shape = tensor_spec.logical_shape();

            const auto tensor_shape_rank = tensor_shape.rank();
            std::vector<size_t> numpy_shape(tensor_shape_rank);
            std::copy(tensor_shape.cbegin(), tensor_shape.cend(), numpy_shape.begin());

            // Copying strides does not work in all cases, fortunately ndarray will compute them at construction
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

    const auto convert_to_row_major = [](const tt::tt_metal::Tensor& tensor) {
        tt::tt_metal::Shape output_tensor_end(ttsl::SmallVector<uint32_t>(tensor.logical_shape().rank(), 0));
        int logical_rank = tensor.logical_shape().rank();
        for (int index = -1; index >= -logical_rank; --index) {
            output_tensor_end[index] = tensor.logical_shape()[index] - 1;
        }

        return ttnn::untilize_with_unpadding(tensor, output_tensor_end, std::nullopt);
    };

    const auto impl = [&make_numpy_tensor_from_data,
                       &convert_to_row_major,
                       composer]<typename MetalType, typename NumpyType>(const tt::tt_metal::Tensor& tensor) {
        if (tensor.storage_type() == ttnn::types::StorageType::HOST) {
            if (tensor.layout() != tt::tt_metal::Layout::ROW_MAJOR) {
                const auto row_major_tensor = convert_to_row_major(tensor);
                const auto row_major_tensor_data = tt::tt_metal::host_buffer::get_as<MetalType const>(row_major_tensor);
                return make_numpy_tensor_from_data.template operator()<NumpyType>(
                    row_major_tensor_data, row_major_tensor.tensor_spec());
            }
            const auto tensor_data = tt::tt_metal::host_buffer::get_as<const MetalType>(tensor);
            return make_numpy_tensor_from_data.template operator()<NumpyType>(tensor_data, tensor.tensor_spec());
        }

        // Move to CPU and convert to row major
        auto cpu_tensor = tensor.cpu(/*blocking=*/true);
        cpu_tensor = cpu_tensor.to_layout(tt::tt_metal::Layout::ROW_MAJOR);

        // If composer is provided, use it to compose distributed tensor shards
        if (composer != nullptr) {
            auto composed_result = composer->compose<MetalType>(cpu_tensor);
            auto& vec = std::get<0>(composed_result);
            auto& shape = std::get<1>(composed_result);

            // Create a temporary tensor spec with the composed shape
            auto composed_spec = tt::tt_metal::TensorSpec(shape, cpu_tensor.tensor_spec().tensor_layout());
            return make_numpy_tensor_from_data.template operator()<NumpyType>(vec, composed_spec);
        }

        const auto cpu_tensor_data = tt::tt_metal::host_buffer::get_as<const MetalType>(cpu_tensor);
        const auto cpu_tensor_spec = cpu_tensor.tensor_spec();
        const auto cpu_tensor_strides = cpu_tensor.strides();

        if (tt::tt_metal::tensor_impl::logical_matches_physical(cpu_tensor_spec)) {
            return make_numpy_tensor_from_data.template operator()<NumpyType>(cpu_tensor_data, cpu_tensor_spec);
        }

        const auto decoded_data = tt::tt_metal::tensor_impl::decode_tensor_data(cpu_tensor_data, cpu_tensor_spec);
        return make_numpy_tensor_from_data.template operator()<NumpyType>(decoded_data, cpu_tensor_spec);
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
            case tt::tt_metal::DataType::BFLOAT8_B:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT8_B);
                break;
            case tt::tt_metal::DataType::BFLOAT4_B:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT4_B);
                break;
            case tt::tt_metal::DataType::UINT8:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT8);
                break;
            case tt::tt_metal::DataType::UINT16:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT16);
                break;
            case tt::tt_metal::DataType::INVALID:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::INVALID);
                break;
        }

        NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UNKNOWN);
    }
    switch (tensor_type) {
        case tt::tt_metal::DataType::INT32:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<int32_t, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<int32_t, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<int32_t, float>(t);
                // impl<int32_t, float>(t) is intentional below
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<int32_t, float>(t);
                case tt::tt_metal::DataType::BFLOAT8_B:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT8_B);
                    break;
                case tt::tt_metal::DataType::BFLOAT4_B:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT4_B);
                    break;
                case tt::tt_metal::DataType::UINT8:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT8);
                    break;
                case tt::tt_metal::DataType::UINT16:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT16);
                    break;
                case tt::tt_metal::DataType::INVALID:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::INVALID);
                    break;
            }
        case tt::tt_metal::DataType::UINT32:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<uint32_t, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<uint32_t, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<uint32_t, float>(t);
                // impl<uint32_t, float>(t) is intentional below
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<uint32_t, float>(t);
                case tt::tt_metal::DataType::BFLOAT8_B:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT8_B);
                    break;
                case tt::tt_metal::DataType::BFLOAT4_B:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT4_B);
                    break;
                case tt::tt_metal::DataType::UINT8:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT8);
                    break;
                case tt::tt_metal::DataType::UINT16:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT16);
                    break;
                case tt::tt_metal::DataType::INVALID:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::INVALID);
                    break;
            }
        case tt::tt_metal::DataType::FLOAT32:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<float, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<float, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<float, float>(t);
                // impl<float, float>(t) is intentional below
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<float, float>(t);
                case tt::tt_metal::DataType::BFLOAT8_B:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT8_B);
                    break;
                case tt::tt_metal::DataType::BFLOAT4_B:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT4_B);
                    break;
                case tt::tt_metal::DataType::UINT8:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT8);
                    break;
                case tt::tt_metal::DataType::UINT16:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT16);
                    break;
                case tt::tt_metal::DataType::INVALID:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::INVALID);
                    break;
            }
        case tt::tt_metal::DataType::BFLOAT16:
            switch (new_type.value()) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<bfloat16, int32_t>(t);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<bfloat16, uint32_t>(t);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<bfloat16, float>(t);
                // impl<bfloat16, float>(t) is intentional below
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<bfloat16, float>(t);
                case tt::tt_metal::DataType::BFLOAT8_B:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT8_B);
                    break;
                case tt::tt_metal::DataType::BFLOAT4_B:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT4_B);
                    break;
                case tt::tt_metal::DataType::UINT8:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT8);
                    break;
                case tt::tt_metal::DataType::UINT16:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT16);
                    break;
                case tt::tt_metal::DataType::INVALID:
                    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::INVALID);
                    break;
            }
        case tt::tt_metal::DataType::BFLOAT8_B:
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT8_B);
            break;
        case tt::tt_metal::DataType::BFLOAT4_B:
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT4_B);
            break;
        case tt::tt_metal::DataType::UINT8: NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT8); break;
        case tt::tt_metal::DataType::UINT16:
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT16);
            break;
        case tt::tt_metal::DataType::INVALID:
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::INVALID);
            break;
    }

    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UNKNOWN);
}

tt::tt_metal::Tensor make_metal_tensor(
    nb::ndarray<> numpy_data,
    tt::tt_metal::Layout target_layout,
    std::optional<tt::tt_metal::DataType> new_type,
    const ttnn::distributed::TensorToMesh* mapper) {
    const auto numpy_data_type = numpy_data.dtype();
    NB_COND_THROW(
        !(numpy_data_type.bits % 8),
        nb::exception_type::type_error,
        "Unsupported precision: {} bits",
        numpy_data_type.bits);

    const auto rank = numpy_data.ndim();

    const auto impl = [&numpy_data_type, rank, &numpy_data, target_layout, mapper]<typename NumpyType>(
                          tt::tt_metal::DataType tensor_data_type) {
        static_assert(std::is_same_v<NumpyType, std::remove_cvref_t<NumpyType>>);
        NB_COND_THROW(
            (numpy_data_type.bits == (sizeof(NumpyType) * 8)),
            nb::exception_type::type_error,
            "Unsupported precision: expected {} bits, got {} bits",
            sizeof(NumpyType) * 8,
            numpy_data_type.bits);

        tt::tt_metal::ShapeBase::Container shape_container(rank);
        for (size_t dimension = 0; dimension < rank; ++dimension) {
            const auto dimension_size = numpy_data.shape(dimension);
            NB_COND_THROW(
                (dimension_size >= std::numeric_limits<uint32_t>::min()),
                nb::exception_type::type_error,
                "Invalid shape parameter for dimension {}: {} is too small",
                dimension,
                dimension_size);
            NB_COND_THROW(
                (dimension_size <= std::numeric_limits<uint32_t>::max()),
                nb::exception_type::type_error,
                "Invalid shape parameter for dimension {}: {} is too large",
                dimension,
                dimension_size);
            shape_container[dimension] = dimension_size;
        }

        const auto make_device_tensor = [target_layout,
                                         tensor_data_type,
                                         &numpy_data,
                                         &shape_container,
                                         mapper]<typename MetalType>() -> tt::tt_metal::Tensor {
            const tt::tt_metal::Shape tensor_shape(shape_container);
            const tt::tt_metal::MemoryConfig tensor_memory_config{};
            // Our tensor will initially be created from row-major numpy data, regardless of target layout
            const tt::tt_metal::PageConfig tensor_page_config(tt::tt_metal::Layout::ROW_MAJOR);
            tt::tt_metal::TensorLayout tensor_layout(tensor_data_type, tensor_page_config, tensor_memory_config);
            tt::tt_metal::TensorSpec tensor_spec(tensor_shape, tensor_layout);
            auto* device = &ttml::autograd::ctx().get_device();
            // device->enable_program_cache();

            std::span<const NumpyType> numpy_data_span(
                static_cast<const NumpyType*>(numpy_data.data()), numpy_data.size());

            if constexpr (!std::is_same_v<MetalType, NumpyType>) {
                std::vector<MetalType> converted_data;
                converted_data.assign(numpy_data_span.begin(), numpy_data_span.end());

                auto row_major_tensor =
                    (mapper != nullptr)
                        ? ttnn::distributed::create_distributed_tensor(
                              ttsl::make_const_span(converted_data), tensor_shape, tensor_layout, *mapper)
                        : tt::tt_metal::Tensor::from_vector(converted_data, tensor_spec, device);

                // Move distributed tensor to device
                if (mapper != nullptr) {
                    row_major_tensor = ttnn::to_device(row_major_tensor, device, tt::tt_metal::MemoryConfig{});
                }

                if (target_layout == tt::tt_metal::Layout::ROW_MAJOR) {
                    return row_major_tensor;
                }
                return ttnn::tilize_with_zero_padding(row_major_tensor);
            } else {
                auto row_major_tensor =
                    (mapper != nullptr)
                        ? ttnn::distributed::create_distributed_tensor(
                              ttsl::make_const_span(numpy_data_span), tensor_shape, tensor_layout, *mapper)
                        : tt::tt_metal::Tensor::from_span(numpy_data_span, tensor_spec, device);

                // Move distributed tensor to device
                if (mapper != nullptr) {
                    row_major_tensor = ttnn::to_device(row_major_tensor, device, tt::tt_metal::MemoryConfig{});
                }

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
            case tt::tt_metal::DataType::BFLOAT8_B:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT8_B);
                break;
            case tt::tt_metal::DataType::BFLOAT4_B:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT4_B);
                break;
            case tt::tt_metal::DataType::UINT8:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT8);
                break;
            case tt::tt_metal::DataType::UINT16:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT16);
                break;
            case tt::tt_metal::DataType::INVALID:
                NB_THROW(nb::exception_type::type_error, UnsupportedMessages::INVALID);
                break;
        }

        NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UNKNOWN);
    };

    switch (static_cast<nb::dlpack::dtype_code>(numpy_data_type.code)) {
        case nb::dlpack::dtype_code::Int:
            return impl.template operator()<int32_t>(new_type.value_or(tt::tt_metal::DataType::INT32));
        case nb::dlpack::dtype_code::UInt:
            return impl.template operator()<uint32_t>(new_type.value_or(tt::tt_metal::DataType::UINT32));
        case nb::dlpack::dtype_code::Float:
            return impl.template operator()<float>(new_type.value_or(tt::tt_metal::DataType::FLOAT32));
        case nb::dlpack::dtype_code::Bfloat:
            //  seems like bfloat is not bfloat16
            // return impl.template operator()<bfloat16>(new_type.value_or(tt::tt_metal::DataType::BFLOAT16));
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT);
            break;
        case nb::dlpack::dtype_code::Complex:
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::COMPLEX);
            break;
        case nb::dlpack::dtype_code::Bool: NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BOOL); break;
    }

    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UNKNOWN);
}

}  // namespace ttml::nanobind::util
