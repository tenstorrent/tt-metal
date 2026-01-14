// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_util.hpp"

#include <nanobind/nanobind.h>

#include <climits>
#include <cstring>
#include <numeric>

#include "autograd/auto_context.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttml::nanobind::util {

// NumPy dtype.kind single-character code
// See: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
namespace NumpyDtypeKind {
constexpr auto VOID = "V";  // Void/structured (custom dtypes like ml_dtypes.bfloat16)
}  // namespace NumpyDtypeKind

namespace UnsupportedMessages {

constexpr auto BFLOAT8_B = "Unsupported type: BFLOAT8_B";
constexpr auto BFLOAT4_B = "Unsupported type: BFLOAT4_B";
constexpr auto UINT8 = "Unsupported type: UINT8";
constexpr auto UINT16 = "Unsupported type: UINT16";
constexpr auto INVALID = "Unsupported type: INVALID";
constexpr auto UNKNOWN = "Unsupported type: unknown";
constexpr auto COMPLEX = "Unsupported type: Complex";
constexpr auto BOOL = "Unsupported type: Bool";

}  // namespace UnsupportedMessages

// Helper: Check if a DataType is supported for conversion
constexpr bool is_supported_datatype(tt::tt_metal::DataType dt) {
    return dt == tt::tt_metal::DataType::INT32 || dt == tt::tt_metal::DataType::UINT32 ||
           dt == tt::tt_metal::DataType::FLOAT32 || dt == tt::tt_metal::DataType::BFLOAT16;
}

// Helper: Throw appropriate error for unsupported types
[[noreturn]] void throw_unsupported_datatype(tt::tt_metal::DataType dt) {
    switch (dt) {
        case tt::tt_metal::DataType::BFLOAT8_B:
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT8_B);
        case tt::tt_metal::DataType::BFLOAT4_B:
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BFLOAT4_B);
        case tt::tt_metal::DataType::UINT8: NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT8);
        case tt::tt_metal::DataType::UINT16: NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UINT16);
        case tt::tt_metal::DataType::INVALID: NB_THROW(nb::exception_type::type_error, UnsupportedMessages::INVALID);
        default: NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UNKNOWN);
    }
}

// Helper: Dispatch single type (for device tensor creation)
template <typename Impl>
auto dispatch_single_type(tt::tt_metal::DataType data_type, const Impl& impl) {
    if (!is_supported_datatype(data_type)) {
        throw_unsupported_datatype(data_type);
    }

    switch (data_type) {
        case tt::tt_metal::DataType::INT32: return impl.template operator()<int32_t>();
        case tt::tt_metal::DataType::UINT32: return impl.template operator()<uint32_t>();
        case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<float>();
        case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<bfloat16>();
        default: throw_unsupported_datatype(data_type);
    }
}

// Helper: Dispatch type conversion using template magic
template <typename Impl>
auto dispatch_conversion(
    tt::tt_metal::DataType from_type, tt::tt_metal::DataType to_type, const Impl& impl, const auto& tensor) {
    if (!is_supported_datatype(to_type)) {
        throw_unsupported_datatype(to_type);
    }

    // Dispatch on from_type, then to_type
    switch (from_type) {
        case tt::tt_metal::DataType::INT32:
            switch (to_type) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<int32_t, int32_t>(tensor);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<int32_t, uint32_t>(tensor);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<int32_t, float>(tensor);
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<int32_t, bfloat16>(tensor);
                default: throw_unsupported_datatype(to_type);
            }
        case tt::tt_metal::DataType::UINT32:
            switch (to_type) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<uint32_t, int32_t>(tensor);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<uint32_t, uint32_t>(tensor);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<uint32_t, float>(tensor);
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<uint32_t, bfloat16>(tensor);
                default: throw_unsupported_datatype(to_type);
            }
        case tt::tt_metal::DataType::FLOAT32:
            switch (to_type) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<float, int32_t>(tensor);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<float, uint32_t>(tensor);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<float, float>(tensor);
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<float, bfloat16>(tensor);
                default: throw_unsupported_datatype(to_type);
            }
        case tt::tt_metal::DataType::BFLOAT16:
            switch (to_type) {
                case tt::tt_metal::DataType::INT32: return impl.template operator()<bfloat16, int32_t>(tensor);
                case tt::tt_metal::DataType::UINT32: return impl.template operator()<bfloat16, uint32_t>(tensor);
                case tt::tt_metal::DataType::FLOAT32: return impl.template operator()<bfloat16, float>(tensor);
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<bfloat16, bfloat16>(tensor);
                default: throw_unsupported_datatype(to_type);
            }
        default: throw_unsupported_datatype(from_type);
    }
}

// Helper: Create tensor from numpy data span (common logic for both overloads)
template <typename NumpyType, typename MetalType>
tt::tt_metal::Tensor create_tensor_from_span(
    std::span<const NumpyType> numpy_data_span,
    const tt::tt_metal::ShapeBase::Container& shape_container,
    tt::tt_metal::DataType tensor_data_type,
    tt::tt_metal::Layout target_layout,
    const ttnn::distributed::TensorToMesh* mapper) {
    const tt::tt_metal::Shape tensor_shape(shape_container);
    static const tt::tt_metal::MemoryConfig tensor_memory_config{};
    const tt::tt_metal::PageConfig tensor_page_config(tt::tt_metal::Layout::ROW_MAJOR);
    tt::tt_metal::TensorLayout tensor_layout(tensor_data_type, tensor_page_config, tensor_memory_config);
    tt::tt_metal::TensorSpec tensor_spec(tensor_shape, tensor_layout);
    auto* device = &ttml::autograd::ctx().get_device();

    if constexpr (!std::is_same_v<MetalType, NumpyType>) {
        std::vector<MetalType> converted_data;
        converted_data.reserve(numpy_data_span.size());
        converted_data.assign(numpy_data_span.begin(), numpy_data_span.end());

        auto row_major_tensor = (mapper != nullptr)
                                    ? ttnn::distributed::create_distributed_tensor(
                                          ttsl::make_const_span(converted_data), tensor_shape, tensor_layout, *mapper)
                                    : tt::tt_metal::Tensor::from_vector(converted_data, tensor_spec, device);

        if (mapper != nullptr) {
            row_major_tensor = ttnn::to_device(row_major_tensor, device, tt::tt_metal::MemoryConfig{});
        }

        if (target_layout == tt::tt_metal::Layout::ROW_MAJOR) {
            return row_major_tensor;
        }
        return ttnn::tilize_with_zero_padding(row_major_tensor);
    } else {
        auto row_major_tensor = (mapper != nullptr)
                                    ? ttnn::distributed::create_distributed_tensor(
                                          ttsl::make_const_span(numpy_data_span), tensor_shape, tensor_layout, *mapper)
                                    : tt::tt_metal::Tensor::from_span(numpy_data_span, tensor_spec, device);

        if (mapper != nullptr) {
            row_major_tensor = ttnn::to_device(row_major_tensor, device, tt::tt_metal::MemoryConfig{});
        }

        if (target_layout == tt::tt_metal::Layout::ROW_MAJOR) {
            return row_major_tensor;
        }
        return ttnn::tilize_with_zero_padding(row_major_tensor);
    }
}

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

nb::object make_numpy_tensor(
    const tt::tt_metal::Tensor& t,
    std::optional<tt::tt_metal::DataType> new_type,
    const ttnn::distributed::MeshToTensor* composer) {
    // Get ml_dtypes.bfloat16 dtype for bfloat16 arrays
    static nb::object ml_dtypes_bfloat16_dtype = []() -> nb::object {
        try {
            nb::object ml_dtypes = nb::module_::import_("ml_dtypes");
            nb::object bfloat16_dtype = ml_dtypes.attr("bfloat16");
            return bfloat16_dtype;
        } catch (...) {
            // ml_dtypes not available, return None
            return nb::none();
        }
    }();

    const auto make_numpy_tensor_from_data =
        []<typename NumpyType>(
            const auto& tensor_data, const auto& tensor_spec /*, const auto& tensor_strides*/) -> nb::object {
        const tt::tt_metal::Shape& tensor_shape = tensor_spec.logical_shape();

        const auto tensor_shape_rank = tensor_shape.rank();
        std::vector<size_t> numpy_shape(tensor_shape_rank);
        std::copy(tensor_shape.cbegin(), tensor_shape.cend(), numpy_shape.begin());

        // Copying strides does not work in all cases, fortunately ndarray will compute them at construction
        // std::vector<int64_t> numpy_strides;
        // numpy_strides.assign(tensor_strides.cbegin(), tensor_strides.cend());

        // For bfloat16, use ml_dtypes.bfloat16 dtype if available
        if constexpr (std::is_same_v<NumpyType, bfloat16>) {
            if (!ml_dtypes_bfloat16_dtype.is_none()) {
                const size_t num_elements = tensor_data.size();

                // Allocate buffer directly (single allocation, no temporary)
                auto* numpy_data = new bfloat16[num_elements];

                // Create capsule to manage memory
                const nb::capsule owner(numpy_data, [](void* p) noexcept { delete[] static_cast<bfloat16*>(p); });

                // Check if we can use fast memcpy (when types match) or need type conversion
                using TensorDataType = typename std::decay_t<decltype(tensor_data)>::value_type;
                if constexpr (std::is_same_v<TensorDataType, bfloat16>) {
                    // Fast path: direct memcpy when no conversion needed
                    std::memcpy(numpy_data, tensor_data.data(), num_elements * sizeof(bfloat16));
                } else {
                    // Slow path: type conversion needed (e.g., float -> bfloat16)
                    std::copy(tensor_data.begin(), tensor_data.end(), numpy_data);
                }

                // Create ndarray from buffer with uint16 dtype (bfloat16 is 16-bit)
                // This is zero-copy from C++ buffer to NumPy
                nb::object array = nb::cast(nb::ndarray<nb::numpy, uint16_t>(
                    reinterpret_cast<uint16_t*>(numpy_data), tensor_shape_rank, numpy_shape.data(), owner));

                // View as bfloat16 - just changes dtype metadata, no copy
                return array.attr("view")(ml_dtypes_bfloat16_dtype);
            } else {
                NB_THROW(
                    nb::exception_type::type_error,
                    "ml_dtypes package required for bfloat16 NumPy arrays (try pip install ml_dtypes)");
            }
        }

        const size_t num_elements = tensor_data.size();
        auto* numpy_data = new NumpyType[num_elements];
        const nb::capsule owner(numpy_data, [](void* p) noexcept { delete[] static_cast<NumpyType*>(p); });

        // Check if we can use fast memcpy (when types match) or need type conversion
        using TensorDataType = typename std::decay_t<decltype(tensor_data)>::value_type;
        if constexpr (std::is_same_v<TensorDataType, NumpyType>) {
            // Fast path: direct memcpy when no conversion needed (types match exactly)
            std::memcpy(numpy_data, tensor_data.data(), num_elements * sizeof(NumpyType));
        } else {
            // Slow path: type conversion needed (e.g., float -> int32_t, bfloat16 -> float)
            std::copy(tensor_data.begin(), tensor_data.end(), numpy_data);
        }

        // Cast to nb::object for uniform return type
        return nb::cast(nb::ndarray<nb::numpy>(
            numpy_data,
            tensor_shape_rank,
            numpy_shape.data(),
            owner,
            nullptr /*numpy_strides.data*/,
            nb::dtype<NumpyType>()));
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

        if (tt::tt_metal::logical_matches_physical(cpu_tensor_spec)) {
            return make_numpy_tensor_from_data.template operator()<NumpyType>(cpu_tensor_data, cpu_tensor_spec);
        }

        const auto decoded_data = tt::tt_metal::tensor_impl::decode_tensor_data(cpu_tensor_data, cpu_tensor_spec);
        return make_numpy_tensor_from_data.template operator()<NumpyType>(decoded_data, cpu_tensor_spec);
    };

    const auto& tensor_spec = t.tensor_spec();
    const auto tensor_type = tensor_spec.data_type();
    const auto target_type = new_type.value_or(tensor_type);

    // Use helper function to dispatch - eliminates 100+ lines of duplication
    return dispatch_conversion(tensor_type, target_type, impl, t);
}

// Fast path: standard NumPy dtypes validated by nanobind
tt::tt_metal::Tensor make_metal_tensor(
    nb::ndarray<nb::numpy> numpy_data,
    tt::tt_metal::Layout target_layout,
    std::optional<tt::tt_metal::DataType> new_type,
    const ttnn::distributed::TensorToMesh* mapper) {
    // Fast path: use ndarray methods directly (no Python attribute lookups)
    auto numpy_data_type = numpy_data.dtype();
    const size_t rank = numpy_data.ndim();

    NB_COND_THROW(
        !(numpy_data_type.bits % 8),
        nb::exception_type::type_error,
        "Unsupported precision: {} bits",
        numpy_data_type.bits);

    const auto impl = [&numpy_data_type, rank, &numpy_data, target_layout, mapper]<typename NumpyType>(
                          tt::tt_metal::DataType tensor_data_type) {
        static_assert(std::is_same_v<NumpyType, std::remove_cvref_t<NumpyType>>);

        // Precision check for standard dtypes
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
                (dimension_size > std::numeric_limits<uint32_t>::min()),
                nb::exception_type::type_error,
                "Invalid shape parameter for dimension {}: {} is too small",
                dimension,
                dimension_size);
            NB_COND_THROW(
                (dimension_size <= std::numeric_limits<uint32_t>::max()),
                nb::exception_type::type_error,
                "Invalid shape parameter for dimension {}: {} exceeds uint32_t maximum",
                dimension,
                dimension_size);
            shape_container[dimension] = dimension_size;
        }

        // Compute data size and get pointer
        size_t data_size =
            std::accumulate(shape_container.begin(), shape_container.end(), size_t(1), std::multiplies<size_t>());

        // Fast path: get data pointer directly from ndarray (no Python attribute lookup)
        const void* data_ptr = numpy_data.data();
        std::span<const NumpyType> numpy_data_span(static_cast<const NumpyType*>(data_ptr), data_size);

        // Dispatch based on target data type
        switch (tensor_data_type) {
            case tt::tt_metal::DataType::INT32:
                return create_tensor_from_span<NumpyType, int32_t>(
                    numpy_data_span, shape_container, tensor_data_type, target_layout, mapper);
            case tt::tt_metal::DataType::UINT32:
                return create_tensor_from_span<NumpyType, uint32_t>(
                    numpy_data_span, shape_container, tensor_data_type, target_layout, mapper);
            case tt::tt_metal::DataType::FLOAT32:
                return create_tensor_from_span<NumpyType, float>(
                    numpy_data_span, shape_container, tensor_data_type, target_layout, mapper);
            case tt::tt_metal::DataType::BFLOAT16:
                return create_tensor_from_span<NumpyType, bfloat16>(
                    numpy_data_span, shape_container, tensor_data_type, target_layout, mapper);
            default: throw_unsupported_datatype(tensor_data_type);
        }
    };

    // Map dtype codes to appropriate handlers
    switch (numpy_data_type.code) {
        case static_cast<uint8_t>(nb::dlpack::dtype_code::Int):
            return impl.operator()<int32_t>(new_type.value_or(tt::tt_metal::DataType::INT32));
        case static_cast<uint8_t>(nb::dlpack::dtype_code::UInt):
            return impl.operator()<uint32_t>(new_type.value_or(tt::tt_metal::DataType::UINT32));
        case static_cast<uint8_t>(nb::dlpack::dtype_code::Float):
            return impl.operator()<float>(new_type.value_or(tt::tt_metal::DataType::FLOAT32));
        case static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat):
            return impl.operator()<bfloat16>(new_type.value_or(tt::tt_metal::DataType::BFLOAT16));
        case static_cast<uint8_t>(nb::dlpack::dtype_code::Complex):
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::COMPLEX);
        case static_cast<uint8_t>(nb::dlpack::dtype_code::Bool):
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BOOL);
        default:
            NB_THROW(
                nb::exception_type::type_error,
                "Unsupported dtype code: {}. For custom dtypes like ml_dtypes.bfloat16, use the nb::object overload.",
                static_cast<int>(numpy_data_type.code));
    }
}

// Custom dtype handler: only accepts custom dtypes (like ml_dtypes.bfloat16)
// Standard dtypes should use the fast path nb::ndarray<nb::numpy> overload
tt::tt_metal::Tensor make_metal_tensor(
    nb::object numpy_data_obj,
    tt::tt_metal::Layout target_layout,
    std::optional<tt::tt_metal::DataType> new_type,
    const ttnn::distributed::TensorToMesh* mapper) {
    // Validate this is a custom dtype (like ml_dtypes.bfloat16)
    nb::object dtype_obj = numpy_data_obj.attr("dtype");
    nb::object itemsize_obj = dtype_obj.attr("itemsize");
    int itemsize_bytes = nb::cast<int>(itemsize_obj);

    // Check dtype kind - custom dtypes have kind='V' (void/structured)
    nb::object kind_obj = dtype_obj.attr("kind");
    const auto dtype_kind = nb::cast<std::string>(kind_obj);
    const bool is_custom_dtype = (dtype_kind == NumpyDtypeKind::VOID);

    // Reject standard dtypes - they should use the fast path
    if (!is_custom_dtype) {
        NB_THROW(
            nb::exception_type::type_error,
            "Standard dtypes should use the nb::ndarray<nb::numpy> overload, not the nb::object overload. "
            "This function is only for custom dtypes like ml_dtypes.bfloat16.");
    }

    // Validate this is bfloat16 (the only supported custom dtype)
    if (itemsize_bytes != sizeof(bfloat16)) {
        nb::object name_obj = dtype_obj.attr("name");
        std::string dtype_name = nb::cast<std::string>(name_obj);
        NB_THROW(
            nb::exception_type::type_error,
            "Unsupported custom dtype '{}' with size {} bytes. Only ml_dtypes.bfloat16 (2 bytes) is supported.",
            dtype_name,
            itemsize_bytes);
    }

    // Check dtype name
    nb::object name_obj = dtype_obj.attr("name");
    std::string dtype_name = nb::cast<std::string>(name_obj);

    if (dtype_name != "bfloat16") {
        NB_THROW(
            nb::exception_type::type_error,
            "Unsupported custom dtype '{}'. Only ml_dtypes.bfloat16 is supported.",
            dtype_name);
    }

    // Force bfloat16 type if not already specified
    if (!new_type.has_value()) {
        new_type = tt::tt_metal::DataType::BFLOAT16;
    }

    // Get rank and shape from Python directly
    nb::object ndim_obj = numpy_data_obj.attr("ndim");
    size_t rank = nb::cast<size_t>(ndim_obj);

    nb::object shape_obj = numpy_data_obj.attr("shape");
    nb::tuple shape_tuple = nb::cast<nb::tuple>(shape_obj);
    std::vector<size_t> shape_from_python;
    for (size_t i = 0; i < rank; ++i) {
        shape_from_python.push_back(nb::cast<size_t>(shape_tuple[i]));
    }

    const auto impl = [rank, &numpy_data_obj, target_layout, mapper, &shape_from_python]<typename NumpyType>(
                          tt::tt_metal::DataType tensor_data_type) {
        static_assert(std::is_same_v<NumpyType, std::remove_cvref_t<NumpyType>>);
        // Note: precision already validated above (bfloat16 is the only supported custom dtype)

        tt::tt_metal::ShapeBase::Container shape_container(rank);
        for (size_t dimension = 0; dimension < rank; ++dimension) {
            // Always use shape from Python (nanobind doesn't populate shape reliably)
            const auto dimension_size = shape_from_python[dimension];
            NB_COND_THROW(
                (dimension_size <= std::numeric_limits<uint32_t>::max()),
                nb::exception_type::type_error,
                "Invalid shape parameter for dimension {}: {} exceeds uint32_t maximum",
                dimension,
                dimension_size);
            shape_container[dimension] = dimension_size;
        }

        // Compute size from shape
        size_t data_size =
            std::accumulate(shape_container.begin(), shape_container.end(), size_t(1), std::multiplies<size_t>());

        // Get data pointer from Python's __array_interface__
        nb::object array_interface = numpy_data_obj.attr("__array_interface__");
        nb::dict ai_dict = nb::cast<nb::dict>(array_interface);
        nb::tuple data_tuple = nb::cast<nb::tuple>(ai_dict["data"]);
        const void* data_ptr = reinterpret_cast<const void*>(nb::cast<uintptr_t>(data_tuple[0]));

        std::span<const NumpyType> numpy_data_span(static_cast<const NumpyType*>(data_ptr), data_size);

        // Dispatch based on target data type
        switch (tensor_data_type) {
            case tt::tt_metal::DataType::INT32:
                return create_tensor_from_span<NumpyType, int32_t>(
                    numpy_data_span, shape_container, tensor_data_type, target_layout, mapper);
            case tt::tt_metal::DataType::UINT32:
                return create_tensor_from_span<NumpyType, uint32_t>(
                    numpy_data_span, shape_container, tensor_data_type, target_layout, mapper);
            case tt::tt_metal::DataType::FLOAT32:
                return create_tensor_from_span<NumpyType, float>(
                    numpy_data_span, shape_container, tensor_data_type, target_layout, mapper);
            case tt::tt_metal::DataType::BFLOAT16:
                return create_tensor_from_span<NumpyType, bfloat16>(
                    numpy_data_span, shape_container, tensor_data_type, target_layout, mapper);
            default: throw_unsupported_datatype(tensor_data_type);
        }
    };

    // Handle custom dtype (ml_dtypes.bfloat16)
    // Already validated at the beginning of this function
    return impl.template operator()<bfloat16>(new_type.value_or(tt::tt_metal::DataType::BFLOAT16));
}

}  // namespace ttml::nanobind::util
