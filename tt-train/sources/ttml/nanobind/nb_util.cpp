// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_util.hpp"

#include <nanobind/nanobind.h>

#include <climits>
#include <numeric>

#include "autograd/auto_context.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttml::nanobind::util {

// Custom dtype code for NumPy structured/void types (kind='V') like ml_dtypes.bfloat16
// These types don't support DLPack and nanobind can't determine their dtype automatically.
// Value 200 is chosen to be well above standard DLPack codes (Int=0, UInt=1, Float=2, Bfloat=4, Complex=5, Bool=6)
// to minimize collision risk with future DLPack additions.
constexpr uint8_t DLPACK_DTYPE_CODE_CUSTOM = 200;

// NumPy dtype.kind single-character codes
// See: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
namespace NumpyDtypeKind {
constexpr char BOOLEAN = 'b';                          // Boolean
constexpr char SIGNED_INT = 'i';                       // Signed integer
constexpr char UNSIGNED_INT = 'u';                     // Unsigned integer
constexpr char FLOAT = 'f';                            // Floating-point
constexpr char COMPLEX = 'c';                          // Complex floating-point
[[maybe_unused]] constexpr char TIMEDELTA = 'm';       // Timedelta
[[maybe_unused]] constexpr char DATETIME = 'M';        // Datetime
[[maybe_unused]] constexpr char OBJECT = 'O';          // Python object
[[maybe_unused]] constexpr char BYTE_STRING = 'S';     // Byte string
[[maybe_unused]] constexpr char UNICODE_STRING = 'U';  // Unicode string
constexpr char VOID = 'V';                             // Void/structured (custom dtypes like ml_dtypes.bfloat16)
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
// constexpr auto BFLOAT = "Unsupported type: Bfloat";

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
                // Copy data to temporary buffer
                std::vector<bfloat16> temp_data(tensor_data.begin(), tensor_data.end());

                // Create array using NumPy with ml_dtypes.bfloat16 dtype
                nb::object np = nb::module_::import_("numpy");

                // Create buffer - treat bfloat16 data as uint16 view
                nb::object bytes_obj =
                    nb::bytes(reinterpret_cast<const char*>(temp_data.data()), temp_data.size() * sizeof(bfloat16));

                nb::object np_frombuffer = np.attr("frombuffer");
                nb::object uint16_array = np_frombuffer(bytes_obj, nb::arg("dtype") = np.attr("uint16"));

                // Convert shape to tuple for reshape
                nb::list shape_list;
                for (size_t i = 0; i < tensor_shape_rank; ++i) {
                    shape_list.append(numpy_shape[i]);
                }

                // Reshape and view as bfloat16 - return as object
                nb::object reshaped = uint16_array.attr("reshape")(nb::tuple(shape_list));
                return reshaped.attr("view")(ml_dtypes_bfloat16_dtype);
            } else {
                NB_THROW(nb::exception_type::type_error, "ml_dtypes package required for bfloat16 NumPy arrays");
            }
        }

        auto* numpy_data = new NumpyType[tensor_data.size()];
        std::copy(tensor_data.begin(), tensor_data.end(), numpy_data);

        const nb::capsule owner(numpy_data, [](void* p) noexcept { delete[] static_cast<NumpyType*>(p); });

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
            case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<bfloat16, bfloat16>(t);
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
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<int32_t, bfloat16>(t);
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
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<uint32_t, bfloat16>(t);
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
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<float, bfloat16>(t);
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
                case tt::tt_metal::DataType::BFLOAT16: return impl.template operator()<bfloat16, bfloat16>(t);
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
            const tt::tt_metal::PageConfig tensor_page_config(tt::tt_metal::Layout::ROW_MAJOR);
            tt::tt_metal::TensorLayout tensor_layout(tensor_data_type, tensor_page_config, tensor_memory_config);
            tt::tt_metal::TensorSpec tensor_spec(tensor_shape, tensor_layout);
            auto* device = &ttml::autograd::ctx().get_device();

            size_t data_size =
                std::accumulate(shape_container.begin(), shape_container.end(), size_t(1), std::multiplies<size_t>());

            // Fast path: get data pointer directly from ndarray (no Python attribute lookup)
            const void* data_ptr = numpy_data.data();
            std::span<const NumpyType> numpy_data_span(static_cast<const NumpyType*>(data_ptr), data_size);

            if constexpr (!std::is_same_v<MetalType, NumpyType>) {
                std::vector<MetalType> converted_data;
                converted_data.assign(numpy_data_span.begin(), numpy_data_span.end());

                auto row_major_tensor =
                    (mapper != nullptr)
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
                auto row_major_tensor =
                    (mapper != nullptr)
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
        default:
            NB_THROW(
                nb::exception_type::type_error,
                "Unsupported dtype code: {}. Falling back to custom dtype handler.",
                static_cast<int>(numpy_data_type.code));
    }
}

// Fallback: custom dtypes (like ml_dtypes.bfloat16)
tt::tt_metal::Tensor make_metal_tensor(
    nb::object numpy_data_obj,
    tt::tt_metal::Layout target_layout,
    std::optional<tt::tt_metal::DataType> new_type,
    const ttnn::distributed::TensorToMesh* mapper) {
    // Check if this is a custom dtype (like ml_dtypes.bfloat16)
    nb::object dtype_obj = numpy_data_obj.attr("dtype");
    nb::object itemsize_obj = dtype_obj.attr("itemsize");
    int itemsize_bytes = nb::cast<int>(itemsize_obj);

    // Check dtype kind - custom dtypes have kind='V' (void/structured)
    nb::object kind_obj = dtype_obj.attr("kind");
    std::string dtype_kind = nb::cast<std::string>(kind_obj);
    bool is_custom_dtype = (dtype_kind.size() == 1 && dtype_kind[0] == NumpyDtypeKind::VOID);

    // Special handling for 2-byte custom types (bfloat16)
    if (is_custom_dtype && itemsize_bytes == sizeof(bfloat16)) {
        // Check dtype name
        nb::object name_obj = dtype_obj.attr("name");
        std::string dtype_name = nb::cast<std::string>(name_obj);

        if (dtype_name == "bfloat16") {
            // Force bfloat16 type if not already specified
            if (!new_type.has_value()) {
                new_type = tt::tt_metal::DataType::BFLOAT16;
            }
        }
    }

    // Convert Python object to ndarray
    nb::ndarray<> numpy_data(numpy_data_obj.ptr());

    auto numpy_data_type = numpy_data.dtype();

    // Get rank and shape from Python directly (nanobind doesn't populate these reliably)
    nb::object ndim_obj = numpy_data_obj.attr("ndim");
    size_t rank = nb::cast<size_t>(ndim_obj);

    nb::object shape_obj = numpy_data_obj.attr("shape");
    nb::tuple shape_tuple = nb::cast<nb::tuple>(shape_obj);
    std::vector<size_t> shape_from_python;
    for (size_t i = 0; i < rank; ++i) {
        shape_from_python.push_back(nb::cast<size_t>(shape_tuple[i]));
    }

    // Nanobind doesn't reliably populate dtype fields, so we synthesize them from Python
    if (numpy_data_type.bits == 0 && itemsize_bytes > 0) {
        numpy_data_type.bits = static_cast<uint8_t>(itemsize_bytes * 8);
    }

    // Determine dtype code from Python dtype kind
    if (is_custom_dtype) {
        numpy_data_type.code = DLPACK_DTYPE_CODE_CUSTOM;
    } else if (dtype_kind.size() == 1) {
        // Map NumPy dtype kind to DLPack code for standard dtypes
        switch (dtype_kind[0]) {
            case NumpyDtypeKind::FLOAT:
                numpy_data_type.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Float);
                break;
            case NumpyDtypeKind::SIGNED_INT:
                numpy_data_type.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
                break;
            case NumpyDtypeKind::UNSIGNED_INT:
                numpy_data_type.code = static_cast<uint8_t>(nb::dlpack::dtype_code::UInt);
                break;
            case NumpyDtypeKind::BOOLEAN:
                numpy_data_type.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Bool);
                break;
            case NumpyDtypeKind::COMPLEX:
                numpy_data_type.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Complex);
                break;
            default:
                // Keep whatever nanobind gave us for unrecognized kinds
                break;
        }
    }

    NB_COND_THROW(
        !(numpy_data_type.bits % 8),
        nb::exception_type::type_error,
        "Unsupported precision: {} bits",
        numpy_data_type.bits);

    const auto impl = [&numpy_data_type,
                       rank,
                       &numpy_data_obj,
                       target_layout,
                       mapper,
                       is_custom_dtype,
                       &shape_from_python]<typename NumpyType>(tt::tt_metal::DataType tensor_data_type) {
        static_assert(std::is_same_v<NumpyType, std::remove_cvref_t<NumpyType>>);
        // Skip precision check for custom dtypes since we manually set bits
        if (!is_custom_dtype) {
            NB_COND_THROW(
                (numpy_data_type.bits == (sizeof(NumpyType) * 8)),
                nb::exception_type::type_error,
                "Unsupported precision: expected {} bits, got {} bits",
                sizeof(NumpyType) * 8,
                numpy_data_type.bits);
        }

        tt::tt_metal::ShapeBase::Container shape_container(rank);
        for (size_t dimension = 0; dimension < rank; ++dimension) {
            // Always use shape from Python (nanobind doesn't populate shape reliably)
            const auto dimension_size = shape_from_python[dimension];
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
                                         &numpy_data_obj,
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

            // Compute size from shape (nanobind doesn't populate size reliably)
            size_t data_size =
                std::accumulate(shape_container.begin(), shape_container.end(), size_t(1), std::multiplies<size_t>());

            // Get data pointer from Python's __array_interface__ (nanobind doesn't provide reliable data pointer)
            nb::object array_interface = numpy_data_obj.attr("__array_interface__");
            nb::dict ai_dict = nb::cast<nb::dict>(array_interface);
            nb::tuple data_tuple = nb::cast<nb::tuple>(ai_dict["data"]);
            const void* data_ptr = reinterpret_cast<const void*>(nb::cast<uintptr_t>(data_tuple[0]));

            std::span<const NumpyType> numpy_data_span(static_cast<const NumpyType*>(data_ptr), data_size);

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
            return impl.template operator()<bfloat16>(new_type.value_or(tt::tt_metal::DataType::BFLOAT16));
        case nb::dlpack::dtype_code::Complex:
            NB_THROW(nb::exception_type::type_error, UnsupportedMessages::COMPLEX);
            break;
        case nb::dlpack::dtype_code::Bool: NB_THROW(nb::exception_type::type_error, UnsupportedMessages::BOOL); break;
    }

    // Handle custom dtype code (e.g., ml_dtypes.bfloat16 with code=200)
    if (numpy_data_type.code == DLPACK_DTYPE_CODE_CUSTOM) {
        // Currently only bfloat16 is supported as a custom dtype
        return impl.template operator()<bfloat16>(new_type.value_or(tt::tt_metal::DataType::BFLOAT16));
    }

    NB_THROW(nb::exception_type::type_error, UnsupportedMessages::UNKNOWN);
}

}  // namespace ttml::nanobind::util
