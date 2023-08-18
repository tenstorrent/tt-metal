#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/conv/conv_op.hpp"
#include "tt_dnn/op_library/fill_rm/fill_rm_op.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/concat/concat_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_dnn/op_library/groupnorm/groupnorm_op.hpp"
#include "tt_dnn/op_library/pool/average_pool.hpp"
#include "tt_dnn/op_library/pool/max_pool.hpp"
#include "tt_dnn/op_library/fully_connected/fully_connected_op.hpp"
#include "tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.hpp"
#include "tt_dnn/op_library/move/move_op.hpp"
#include "tt_dnn/op_library/rotate_half/rotate_half_op.hpp"
#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_op.hpp"
#include "tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "tt_dnn/op_library/update_cache/update_cache_op.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/borrowed_buffer.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/serialization.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_lib_bindings.hpp"
#include "operations/module.hpp"
#include "type_caster.hpp"

namespace py = pybind11;

namespace tt {

namespace tt_metal {

struct PythonFallbackOperation {
    std::string function_name_;
    tt::stl::reflection::Attributes attributes_;

    std::string get_type_name() const {
        return fmt::format("{} (fallback operation)", this->function_name_);
    }

    tt::stl::reflection::Attributes attributes() const {
        return this->attributes_;
    }
};


namespace detail {


OwnedBuffer create_owned_buffer_from_vector_of_floats(std::vector<float>&& data, DataType data_type) {
    switch (data_type) {
        case DataType::BFLOAT8_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return owned_buffer::create<uint32_t>(std::move(uint32_vector));
        }
        case DataType::FLOAT32: {
            return owned_buffer::create<float>(std::move(data));
        }
        case DataType::BFLOAT16: {
            std::vector<bfloat16> bfloat16_data(data.size());
            std::transform(
                std::begin(data), std::end(data),
                std::begin(bfloat16_data),
                [](float value) { return bfloat16(value); }
            );
            return owned_buffer::create<bfloat16>(std::move(bfloat16_data));
        }
        default: {
            TT_THROW("Cannot create a host buffer!");
        }
    }
}

template<class T>
struct DataTypeToFormatType {
    using type = T;
};

template<>
struct DataTypeToFormatType<bfloat16> {
    using type = uint16_t;
};

template<class CppType, class DataType, class PyType>
void implement_buffer_protocol(PyType& py_buffer_t) {
    py_buffer_t
        .def(
            "__getitem__",
            [](const CppType& self, std::size_t index) {
                return self[index];
            }
        )
        .def(
            "__len__",
            [](const CppType& self) {
                return self.size();
            }
        )
        .def(
            "__iter__",
            [](const CppType& self) {
                return py::make_iterator(self.begin(), self.end());
            },
            py::keep_alive<0, 1>()
        )
        .def_buffer(
            [](CppType& self) -> py::buffer_info {
                using FormatType = typename DataTypeToFormatType<DataType>::type;
                return py::buffer_info(
                    self.begin(),                                /* Pointer to buffer */
                    sizeof(DataType),                            /* Size of one scalar */
                    py::format_descriptor<FormatType>::format(), /* Python struct-style format descriptor */
                    1,                                           /* Number of dimensions */
                    { self.size() },                             /* Buffer dimensions */
                    { sizeof(DataType) }                         /* Strides (in bytes) for each index */
                );
            }
        );
};

Tensor convert_torch_tensor_to_tt_tensor(const py::handle& torch_tensor, std::optional<DataType> optional_data_type = std::nullopt) {
    py::object torch = py::module_::import("torch");
    if (not py::isinstance(torch_tensor, torch.attr("Tensor"))) {
        TT_THROW("The argument must be of type torch.Tensor!");
    }

    auto torch_dtype = torch_tensor.attr("dtype");
    auto shape = py::cast<std::vector<uint32_t>>(torch_tensor.attr("shape"));

    auto contiguous_torch_tensor = torch_tensor.attr("contiguous")();

    // Override the data type if there is an user-provided one
    // Otherwise, figure it out from torch dtype
    DataType data_type;
    if (optional_data_type.has_value()) {
        data_type = optional_data_type.value();
    }
    else if (torch_dtype.equal(torch.attr("float32"))) {
        data_type = DataType::FLOAT32;
    }
    else if (torch_dtype.equal(torch.attr("float16"))) {
        // TODO(arakhmati): add DataType::FLOAT16?
        data_type = DataType::BFLOAT16;
    }
    else if (torch_dtype.equal(torch.attr("bfloat16"))) {
        data_type = DataType::BFLOAT16;
    }
    else if (torch_dtype.equal(torch.attr("int64"))) {
        contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("int32"));
        // TODO(arakhmati): add DataType::INT64?
        data_type = DataType::UINT32;
    }
    else if (torch_dtype.equal(torch.attr("int32"))) {
        contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("int32"));
        // TODO(arakhmati): add DataType::INT32?
        data_type = DataType::UINT32;
    } else {
        TT_THROW(fmt::format("Unsupported DataType: {}", py::repr(torch_dtype)));
    }

    switch (data_type) {
        case DataType::UINT32: {
            contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("int32"));
            break;
        }
        case DataType::FLOAT32: {
            contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("float32"));
            break;
        }
        case DataType::BFLOAT16: {
            contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("bfloat16"));
            break;
        }
        case DataType::BFLOAT8_B: {
            contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("float32"));
            break;
        }
        default: {
            TT_THROW(fmt::format("Unsupported DataType: {}", data_type));
            break;
        }
    }

    auto on_creation_callback = [tensor = contiguous_torch_tensor] { tensor.inc_ref(); };
    auto on_destruction_callback = [tensor = contiguous_torch_tensor] { tensor.dec_ref(); };

    switch (data_type) {
        case DataType::UINT32: {
            auto data_ptr = reinterpret_cast<uint32_t*>(py::cast<std::size_t>(contiguous_torch_tensor.attr("data_ptr")()));
            auto num_elements = py::cast<std::size_t>(contiguous_torch_tensor.attr("numel")());
            auto storage = BorrowedStorage(
                borrowed_buffer::Buffer(data_ptr, num_elements),
                on_creation_callback,
                on_destruction_callback
            );
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR);
        }
        case DataType::FLOAT32: {
            auto data_ptr = reinterpret_cast<float*>(py::cast<std::size_t>(contiguous_torch_tensor.attr("data_ptr")()));
            auto num_elements = py::cast<std::size_t>(contiguous_torch_tensor.attr("numel")());
            auto storage = BorrowedStorage(
                borrowed_buffer::Buffer(data_ptr, num_elements),
                on_creation_callback,
                on_destruction_callback
            );
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR);
        }
        case DataType::BFLOAT16: {
            auto data_ptr = reinterpret_cast<bfloat16*>(py::cast<std::size_t>(contiguous_torch_tensor.attr("data_ptr")()));
            auto num_elements = py::cast<std::size_t>(contiguous_torch_tensor.attr("numel")());
            auto storage = BorrowedStorage(
                borrowed_buffer::Buffer(data_ptr, num_elements),
                on_creation_callback,
                on_destruction_callback
            );
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR);
        }
        case DataType::BFLOAT8_B: {
            auto data_ptr = reinterpret_cast<float*>(py::cast<std::size_t>(contiguous_torch_tensor.attr("data_ptr")()));
            auto num_elements = py::cast<std::size_t>(contiguous_torch_tensor.attr("numel")());
            auto data = std::vector<float>(data_ptr, data_ptr + num_elements);
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto buffer = owned_buffer::create<uint32_t>(std::move(uint32_vector));
            auto storage = OwnedStorage{std::move(buffer)};
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR);
        }
        default: {
            TT_THROW(fmt::format("Unsupported DataType: {}", data_type));
            break;
        }
    }
}


py::object convert_tt_tensor_to_torch_tensor(const Tensor& tt_tensor) {
    TT_ASSERT(tt_tensor.storage_type() == StorageType::OWNED or tt_tensor.storage_type() == StorageType::BORROWED);

    using namespace pybind11::literals;
    py::object torch = py::module_::import("torch");
    auto frombuffer = torch.attr("frombuffer");

    auto buffer = std::visit(
        [] (auto&& storage) -> std::variant<OwnedBuffer, BorrowedBuffer> {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return storage.buffer;
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                TT_THROW("Device tensor cannot be converted to torch");
            }
            else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return storage.buffer;
            }
            else {
                raise_unsupported_storage<T>();
            }
        },
        tt_tensor.storage()
    );

    auto tt_dtype = tt_tensor.dtype();
    if (tt_dtype == DataType::BFLOAT8_B) {
        auto uint32_data = std::get<owned_buffer::Buffer<std::uint32_t>>(std::get<OwnedBuffer>(buffer)).get();
        auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
        buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
        tt_dtype = DataType::FLOAT32;
    }

    const auto tt_dtype_to_torch_dtype = std::map<DataType, py::object> {
        {DataType::UINT32, torch.attr("int32")}, // TODO(arakhmati): add DataType::INT32
        {DataType::FLOAT32, torch.attr("float32")},
        {DataType::BFLOAT16, torch.attr("bfloat16")},
    };
    auto torch_dtype = tt_dtype_to_torch_dtype.at(tt_dtype);

    auto shape = tt_tensor.shape();
    auto torch_shape = std::vector<std::uint32_t>(std::begin(shape), std::end(shape));
    auto tensor = frombuffer(buffer, "dtype"_a=torch_dtype);
    tensor = tensor.attr("reshape")(torch_shape);
    return tensor;
}

template <bool mem_config_arg = true, typename Func, typename... Extra>
void bind_op_with_mem_config(py::module_ &module, std::string op_name, Func &&f, std::string docstring, Extra&&... extra) {
    if constexpr (mem_config_arg) {
        const std::string mem_config_name = "output_mem_config";
        docstring += fmt::format(R"doc(
            "{0}", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is {1} in {2}", "No")doc",
            mem_config_name, operation::DEFAULT_OUTPUT_MEMORY_CONFIG.interleaved ? "interleaved" : "non-interleaved", magic_enum::enum_name(operation::DEFAULT_OUTPUT_MEMORY_CONFIG.buffer_type)
        );
        module.def(op_name.c_str(), f,
            std::forward<Extra>(extra)..., py::arg(mem_config_name.c_str()).noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, docstring.c_str()
        );
    } else {
        module.def(op_name.c_str(), f,
            std::forward<Extra>(extra)..., docstring.c_str()
        );
    }
}

template <bool fused_activations = true, bool mem_config_arg = true, typename Func>
void bind_binary_op(py::module_ &module, std::string op_name, Func &&f, std::string op_desc) {
    std::vector<std::string> arg_name = {"input", "other"};
    op_desc = fmt::format(op_desc, arg_name[0], arg_name[1]);

    std::string docstring = fmt::format(R"doc(
        {0}

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{2}", "First tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "{3}", "Second tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        op_desc, op_name, arg_name[0], arg_name[1]
    );
    if constexpr (fused_activations) {
        const std::string fused_activations_name = "fused_activations";
        const std::optional<std::vector<UnaryWithParam>> default_fused_activations = std::nullopt;
        docstring += fmt::format(R"doc(
            "{0}", "Fused activations after binary computation", "List of FusibleActivation with optional param", "Default is None", "No")doc",
            fused_activations_name
        );
        bind_op_with_mem_config<mem_config_arg>(module, op_name, f, docstring,
            py::arg(arg_name[0].c_str()).noconvert(),
            py::arg(arg_name[1].c_str()).noconvert(),
            py::arg(fused_activations_name.c_str()) = default_fused_activations
        );

    } else {
        bind_op_with_mem_config<mem_config_arg>(module, op_name, f, docstring,
            py::arg(arg_name[0].c_str()).noconvert(),
            py::arg(arg_name[1].c_str()).noconvert()
        );
    }
}

//TODO @tt-aho: Update to handle variable number of params
template <bool mem_config_arg = true, typename Func>
void bind_unary_op(py::module_ &module, std::string op_name, Func &&f, std::string op_desc) {
    const std::string tensor_name = "input";
    op_desc = fmt::format(op_desc, tensor_name);
    std::string docstring = fmt::format(R"doc(
        {0}

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{1}", "Tensor {2} is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        op_desc, tensor_name, op_name
    );

    bind_op_with_mem_config<mem_config_arg>(module, op_name, f, docstring, py::arg(tensor_name.c_str()).noconvert());
}

template <bool mem_config_arg = true, typename Func, typename PyArg, typename std::enable_if<std::is_base_of<py::arg, PyArg>::value, int>::type = 0>
void bind_unary_op_with_param(py::module_ &module, std::string op_name, Func &&f, PyArg param, std::string op_desc, std::string param_desc) {
    const std::string tensor_name = "input";
    std::string param_name = std::string(param.name);
    op_desc = fmt::format(op_desc, tensor_name, param_name);
    const std::string required_param = std::is_same_v<py::arg_v, PyArg> ? "No" : "Yes";
    std::string docstring = fmt::format(R"doc(
        {0}

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{1}", "Tensor {2} is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        op_desc, tensor_name, op_name, param_desc
    );

    docstring += fmt::format(R"doc(
            "{0}", {1}, "{2}")doc",
        param_name, param_desc, required_param
    );
    bind_op_with_mem_config<mem_config_arg>(module, op_name, f, docstring, py::arg(tensor_name.c_str()).noconvert(), param);
}

template <typename E, typename... Extra>
py::enum_<E> export_enum(const py::handle &scope, std::string name = "", Extra&&... extra) {
    py::enum_<E> enum_type(scope, name.empty() ? magic_enum::enum_type_name<E>().data() : name.c_str(), std::forward<Extra>(extra)...);
    for (const auto &[value, name] : magic_enum::enum_entries<E>()) {
        enum_type.value(name.data(), value);
    }
    return enum_type;
}
}

void TensorModule(py::module &m_tensor) {
    // ENUM SECTION

    // bcast enums
    detail::export_enum<BcastOpMath>(m_tensor);
    /** TODO: add these to bcast ops - good to have not required
        .value("GT", BcastOpMath::GT)
        .value("LT", BcastOpMath::LT)
        .value("GE", BcastOpMath::GE)
        .value("LE", BcastOpMath::LE)
        .value("EQ", BcastOpMath::EQ)
        .value("NEQ", BcastOpMath::NE);
    */

    detail::export_enum<BcastOpDim>(m_tensor);

    // reduce enums
    detail::export_enum<ReduceOpMath>(m_tensor);

    detail::export_enum<ReduceOpDim>(m_tensor);

    // layout enums
    detail::export_enum<Layout>(m_tensor);

    detail::export_enum<DataType>(m_tensor);

    detail::export_enum<StorageType>(m_tensor);

    py::enum_<BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1);

    // Fusible Activations
    detail::export_enum<UnaryOpType>(m_tensor, "FusibleActivation");
    py::class_<UnaryWithParam>(m_tensor, "FusibleActivationWithParam")
        .def(py::init<UnaryOpType>())
        .def(py::init<UnaryOpType, float>())
        .def(py::init<>(
            [](std::pair<UnaryOpType, float> arg) {
                return UnaryWithParam{.op_type=arg.first, .param=arg.second};
            }
        ));
    // Allow implicit construction of UnaryWithParam object without user explicitly creating it
    // Can take in just the op type, or sequence container of op type and param value
    py::implicitly_convertible<UnaryOpType, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, float>, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, int>, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, bool>, UnaryWithParam>();

    auto py_core_coord = py::class_<CoreCoord>(m_tensor, "CoreCoord", R"doc(
        Class defining core coordinate
    )doc");

    py_core_coord
        .def(py::init<std::size_t, std::size_t>())
        .def("__repr__", [](const CoreCoord& self) -> std::string {
            return self.str();
        }
        );

    auto pyMemoryConfig = py::class_<MemoryConfig>(m_tensor, "MemoryConfig", R"doc(
        Class defining memory configuration for storing tensor data on TT Accelerator device.
        There are eight DRAM memory banks on TT Accelerator device, indexed as 0, 1, 2, ..., 7.
    )doc");

    pyMemoryConfig
        .def(
            py::init<>(
                [](bool interleaved, BufferType buffer_type) {
                    return MemoryConfig{.interleaved=interleaved, .buffer_type=buffer_type};
                }
            ),
            py::arg("interleaved") = true,
            py::arg("buffer_type") = BufferType::DRAM, R"doc(
                Create MemoryConfig class.
                If interleaved is set to True, tensor data will be interleaved across multiple DRAM banks on TT Accelerator device.
                Otherwise, tensor data will be stored in a DRAM bank selected by dram_channel (valid values are 0, 1, ..., 7).

                Example of creating MemoryConfig specifying that tensor data should be stored in DRAM bank 3.

                .. code-block:: python

                    mem_config = tt_lib.tensor.MemoryConfig(False)
            )doc"
        )
        .def("__repr__", [](const MemoryConfig &memory_config) -> std::string {
            return fmt::format("{}", memory_config);
        }
        )
        .def_readonly("interleaved", &MemoryConfig::interleaved, "Whether tensor data is interleaved across mulitple DRAM channels")
        .def_readonly("buffer_type", &MemoryConfig::buffer_type, "Buffer type to store tensor data. Can be DRAM or L1");

    auto py_owned_buffer_for_uint32_t = py::class_<owned_buffer::Buffer<uint32_t>>(m_tensor, "owned_buffer_for_uint32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<uint32_t>, uint32_t>(py_owned_buffer_for_uint32_t);

    auto py_owned_buffer_for_float32_t = py::class_<owned_buffer::Buffer<float>>(m_tensor, "owned_buffer_for_float32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<float>, float>(py_owned_buffer_for_float32_t);

    auto py_owned_buffer_for_bfloat16_t = py::class_<owned_buffer::Buffer<bfloat16>>(m_tensor, "owned_buffer_for_bfloat16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<bfloat16>, bfloat16>(py_owned_buffer_for_bfloat16_t);

    auto py_borrowed_buffer_for_uint32_t = py::class_<borrowed_buffer::Buffer<std::uint32_t>>(m_tensor, "borrowed_buffer_for_uint32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint32_t>, std::uint32_t>(py_borrowed_buffer_for_uint32_t);

    auto py_borrowed_buffer_for_float32_t = py::class_<borrowed_buffer::Buffer<float>>(m_tensor, "borrowed_buffer_for_float32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<float>, float>(py_borrowed_buffer_for_float32_t);

    auto py_borrowed_buffer_for_bfloat16_t = py::class_<borrowed_buffer::Buffer<bfloat16>>(m_tensor, "borrowed_buffer_for_bfloat16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<bfloat16>, bfloat16>(py_borrowed_buffer_for_bfloat16_t);

    // Tensor constructors that accept device and .to(device) function use keep alive call policy to communicate that Device needs to outlive Tensor.
    // This is because when tensors on device are destroyed they need to deallocate their buffers via device.
    // keep_alive increases the ref count of the Device object being passed into the constructor and .to() function.
    // For additional info see: https://pybind11.readthedocs.io/en/stable/advanced/functions.html#keep-alive
    auto pyTensor = py::class_<Tensor>(m_tensor, "Tensor", R"doc(


        Class constructor supports tensors of rank 4.
        The constructor takes following arguments:

        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        |  Argument  |                 Description                            |       Data type           |           Valid range              | Required |
        +============+========================================================+===========================+====================================+==========+
        | data       | Data to store in TT tensor                             | List[float/int]           |                                    | Yes      |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | shape      | Shape of TT tensor                                     | List[int[4]]              |                                    | Yes      |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | data_type  | Data type of numbers in TT tensor                      | tt_lib.tensor.DataType    | tt_lib.tensor.DataType.BFLOAT16    | Yes      |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | tt_lib.tensor.DataType.FLOAT32     |          |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | tt_lib.tensor.DataType.UINT32      |          |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | tt_lib.tensor.DataType.BFLOAT8_B   |          |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | layout     | Layout of tensor data in memory                        | tt_lib.tensor.Layout      | tt_lib.tensor.Layout.ROW_MAJOR     | Yes      |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | tt_lib.tensor.Layout.TILE          |          |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | device     | Device on which tensor will be created                 | tt_lib.device.Device      | Host or TT accelerator device      | No       |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator device memory banks | tt_lib.tensor.MemoryConfig|                                    | No       |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+

    )doc");

    pyTensor
        .def(
            py::init<>(
                [](std::vector<float>&& data, const std::array<uint32_t, 4>& shape, DataType data_type, Layout layout) {
                    auto owned_buffer = detail::create_owned_buffer_from_vector_of_floats(std::move(data), data_type);
                    return Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
                }
            ),
            py::return_value_policy::move,
            R"doc(
                +---------------+---------------+
                | Argument      | Name          |
                +===============+===============+
                | arg0          | data          |
                +---------------+---------------+
                | arg1          | shape         |
                +---------------+---------------+
                | arg2          | data_type     |
                +---------------+---------------+
                | arg3          | layout        |
                +---------------+---------------+

                Example of creating a TT Tensor on host:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    tt_lib.tensor.Tensor(
                        py_tensor.reshape(-1).tolist(),
                        py_tensor.size(),
                        tt_lib.tensor.DataType.BFLOAT16,
                        tt_lib.tensor.Layout.ROW_MAJOR,
                    )
            )doc"
        )
        .def(
            py::init<>(
                [](std::vector<float>&& data, const std::array<uint32_t, 4>& shape, DataType data_type, Layout layout, Device *device) {
                    auto owned_buffer = detail::create_owned_buffer_from_vector_of_floats(std::move(data), data_type);
                    auto tensor = Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
                    return tensor.to(device, MemoryConfig{});
                }
            ),
            py::keep_alive<1, 6>(),
            py::return_value_policy::move,
            R"doc(
                +---------------+---------------+
                | Argument      | Name          |
                +===============+===============+
                | arg0          | data          |
                +---------------+---------------+
                | arg1          | shape         |
                +---------------+---------------+
                | arg2          | data_type     |
                +---------------+---------------+
                | arg3          | layout        |
                +---------------+---------------+
                | arg3          | device        |
                +---------------+---------------+

                Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B (in TILE layout) are supported on device.

                Note that TT Tensor in ROW_MAJOR layout on TT Accelerator device must have size of last dimension divisble by 2.

                Example of creating a TT Tensor on TT accelerator device:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    tt_device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
                    // ...
                    tt_lib.tensor.Tensor(
                        py_tensor.reshape(-1).tolist(),
                        py_tensor.size(),
                        tt_lib.tensor.DataType.BFLOAT16,
                        tt_lib.tensor.Layout.ROW_MAJOR,
                        tt_device
                    )
            )doc"
        )
        .def(
            py::init<>(
                [](std::vector<float>&& data, const std::array<uint32_t, 4>& shape, DataType data_type, Layout layout, Device *device, const MemoryConfig& memory_config) {
                    auto owned_buffer = detail::create_owned_buffer_from_vector_of_floats(std::move(data), data_type);
                    auto tensor = Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
                    return tensor.to(device, memory_config);
                }
            ),
            py::keep_alive<1, 6>(),
            py::return_value_policy::move,
            R"doc(
                +---------------+---------------+
                | Argument      | Name          |
                +===============+===============+
                | arg0          | data          |
                +---------------+---------------+
                | arg1          | shape         |
                +---------------+---------------+
                | arg2          | data_type     |
                +---------------+---------------+
                | arg3          | layout        |
                +---------------+---------------+
                | arg3          | device        |
                +---------------+---------------+
                | arg3          | mem_config    |
                +---------------+---------------+

                Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B (in TILE layout) are supported on device.

                Note that TT Tensor in ROW_MAJOR layout on TT Accelerator device must have size of last dimension divisble by 2.

                Example of creating a TT Tensor on TT accelerator device with specified mem_config:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    tt_device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
                    mem_config = tt_lib.tensor.MemoryConfig(False)
                    // ...
                    tt_lib.tensor.Tensor(
                        py_tensor.reshape(-1).tolist(),
                        py_tensor.size(),
                        tt_lib.tensor.DataType.BFLOAT16,
                        tt_lib.tensor.Layout.ROW_MAJOR,
                        tt_device,
                        mem_config
                    )
            )doc"
        )
        .def(
            py::init<>(
                [](const py::object& torch_tensor, DataType data_type) {
                    return detail::convert_torch_tensor_to_tt_tensor(torch_tensor, data_type);
                }
            ),
            py::return_value_policy::move,
            R"doc(
                +---------------+---------------+
                | Argument      | Name          |
                +===============+===============+
                | arg0          | torch_tensor  |
                +---------------+---------------+

                Example of creating a TT Tensor that uses torch.Tensor's storage as its own storage:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    tt_lib.tensor.Tensor(py_tensor)
            )doc"
        )
        .def("deallocate", [](Tensor &self) {
            return self.deallocate();
        }, R"doc(
            Dellocates all data of a tensor. This either deletes all host data or deallocates tensor data from device memory.
        )doc"
        )
        .def("to", [](const Tensor &self, Device *device, const MemoryConfig &mem_config) {
            return self.to(device, mem_config);
        }, py::arg().noconvert(), py::arg("mem_config").noconvert() = MemoryConfig{.interleaved = true}, py::keep_alive<0, 2>(), R"doc(
            Move TT Tensor from host device to TT accelerator device.

            Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B (in TILE layout) are supported on device.

            If ``arg1`` is not supplied, default ``MemoryConfig`` with ``interleaved`` set to ``True``.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | Device to which tensor will be moved            | tt_lib.device.Device       | TT accelerator device | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | arg1      | MemoryConfig of tensor of TT accelerator device | tt_lib.tensor.MemoryConfig |                       | No       |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def("cpu", &Tensor::cpu, R"doc(
            Move TT Tensor from TT accelerator device to host device.

            .. code-block:: python

                tt_tensor = tt_tensor.cpu()
        )doc")
        .def("to", py::overload_cast<Layout>(&Tensor::to, py::const_), R"doc(
            Convert TT Tensor to provided memory layout. Available layouts conversions are:

            * ROW_MAJOR to TILE
            * TILE to ROW_MAJOR

            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range                    | Required |
            +===========+=================================================+============================+================================+==========+
            | arg0      | Target memory layout                            | tt_lib.tensor.Layout       | ROW_MAJOR, TILE                | Yes      |
            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_lib.tensor.Layout.TILE)
        )doc")
        .def("pad",
            [] (const Tensor &self, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {
                return self.pad(output_tensor_shape, input_tensor_start, pad_value);
            },
            R"doc(
            Pad TT Tensor with given pad value ``arg2``.

            The input tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor that contains the input tensor at the given input tensor start indices ``arg1`` and the padded value everywhere else.

            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                          | Data type    | Valid range                                         | Required |
            +=====================+======================================================+==============+=====================================================+==========+
            | arg0                | Shape of output tensor                               | List[int[4]] |                                                     | Yes      |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg1                | Start indices to place input tensor in output tensor | List[int[4]] | Values along each dim must be                       | Yes      |
            |                     |                                                      |              |                                                     |          |
            |                     |                                                      |              | <= (output_tensor_shape[i] - input_tensor_shape[i]) |          |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg2                | Value to pad input tensor                            | float        |                                                     | Yes      |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+

            .. code-block:: python

                input_tensor_shape = [1, 1, 3, 3]
                output_tensor_shape = [1, 2, 5, 5]
                input_tensor_start = [0, 1, 1, 1]
                pad_value = 0

                inp = torch.Tensor(
                    [ 1, 2, 3,
                      4, 5, 6,
                      7, 8, 9 ]
                )
                tt_tensor = ttl.tensor.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttl.tensor.DataType.BFLOAT16,
                    ttl.tensor.Layout.ROW_MAJOR,
                )
                tt_tensor_padded = tt_tensor.pad(output_tensor_shape, input_tensor_start, pad_value)

                print("Input tensor:")
                tt_tensor.pretty_print()
                print("\nPadded tensor:")
                tt_tensor_padded.pretty_print()

            Example output:

            .. code-block::

                Input tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]

                Padded tensor:
                [ [[[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                    [[0, 0, 0, 0, 0],
                     [0, 1, 2, 3, 0],
                     [0, 4, 5, 6, 0],
                     [0, 7, 8, 9, 0],
                     [0, 0, 0, 0, 0]]] dtype=bfloat16 ]
        )doc")
        .def("unpad", [](const Tensor &self, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {
            return self.unpad(output_tensor_start, output_tensor_end);
        }, R"doc(
            Unpad this TT Tensor.

            This tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor from output tensor start indices ``arg0`` to output tensor end indices ``arg1`` (inclusive) of the input tensor.

            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                  | Data type    | Valid range                                         | Required |
            +=====================+==============================================+==============+=====================================================+==========+
            | arg0                | Start indices of input tensor                | List[int[4]] | Values along each dim must be                       | Yes      |
            |                     |                                              |              |                                                     |          |
            |                     |                                              |              | < input_tensor_shape[i] and <= output_tensor_end[i] |          |
            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg1                | End indices of input tensor in output tensor | List[int[4]] | Values along each dim must be                       | Yes      |
            |                     |                                              |              |                                                     |          |
            |                     |                                              |              | < input_tensor_shape[i]                             |          |
            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+

            .. code-block:: python

                input_tensor_shape = [1, 1, 5, 5]
                output_tensor_start = [0, 0, 1, 1]
                output_tensor_end = [0, 0, 3, 3]

                inp = torch.Tensor(
                    [ 0, 0, 0, 0, 0,
                      0, 1, 2, 3, 0,
                      0, 4, 5, 6, 0,
                      0, 7, 8, 9, 0,
                      0, 0, 0, 0, 0 ]
                )
                tt_tensor = ttl.tensor.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttl.tensor.DataType.BFLOAT16,
                    ttl.tensor.Layout.ROW_MAJOR,
                )
                tt_tensor_unpadded = tt_tensor.unpad(output_tensor_start, output_tensor_end)

                print("Input tensor:")
                tt_tensor.pretty_print()
                print("\nUnpadded tensor:")
                tt_tensor_unpadded.pretty_print()

            Example output:

            .. code-block::

                Input tensor:
                [ [[[0, 0, 0, 0, 0],
                    [0, 1, 2, 3, 0],
                    [0, 4, 5, 6, 0],
                    [0, 7, 8, 9, 0],
                    [0, 0, 0, 0, 0]]] dtype=bfloat16 ]

                Unpadded tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]
        )doc")
        .def("pad_to_tile", [](const Tensor &self, float pad_value) {
            return self.pad_to_tile(pad_value);
        }, R"doc(
            Pads TT Tensor with given pad value ``arg0``.

            The input tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor that contains the input tensor padded with the padded value in the last two dims to multiples of 32.

            Padding will be added to the right and bottom of the tensor.

            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                          | Data type    | Valid range                                         | Required |
            +=====================+======================================================+==============+=====================================================+==========+
            | arg0                | Value to pad input tensor                            | float        |                                                     | Yes      |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+

            .. code-block:: python

                input_tensor_shape = [1, 1, 3, 3]
                pad_value = 0

                inp = torch.Tensor(
                    [ 1, 2, 3,
                      4, 5, 6,
                      7, 8, 9 ]
                )
                tt_tensor = ttl.tensor.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttl.tensor.DataType.BFLOAT16,
                    ttl.tensor.Layout.ROW_MAJOR,
                )
                tt_tensor_padded = tt_tensor.pad_to_tile(pad_value)

                print("Input tensor:")
                tt_tensor.pretty_print()
                print("\nPadded tensor:")
                tt_tensor_padded.pretty_print()

            Example output:

            .. code-block::

                Input tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]

                Padded tensor:
                [ [[[1, 2, 3, 0, ..., 0],
                    [4, 5, 6, 0, ..., 0],
                    [7, 8, 9, 0, ..., 0],
                    [0, 0, 0, 0, ..., 0],
                    ...,
                    [0, 0, 0, 0, ..., 0]]] dtype=bfloat16 ]
        )doc")
        .def("unpad_from_tile", [](const Tensor &self, const std::array<uint32_t, 4> &output_tensor_shape) {
            return self.unpad_from_tile(output_tensor_shape);
        }, R"doc(
            Unpads TT Tensor from given input tensor ``arg0``.

            The input tensor must be on host and in ROW_MAJOR layout.

            This function expects the real data to aligned on the top left of the tensor.

            Returns an output tensor with padding removed from the right and bottom of the input tensor.

            +---------------------+----------------------------------------------+--------------+------------------------------------------------------------------------------+----------+
            | Argument            | Description                                  | Data type    | Valid range                                                                  | Required |
            +=====================+==============================================+==============+==============================================================================+==========+
            | arg0                | Shape of output tensor                       | List[int[4]] | All dims must match the input tensor dims apart from the last two dims.      | Yes      |
            |                     |                                              |              |                                                                              |          |
            |                     |                                              |              | Last two dims have the following restrictions:                               |          |
            |                     |                                              |              |                                                                              |          |
            |                     |                                              |              | input_tensor_shape[i] must be a multiple of 32                               |          |
            |                     |                                              |              |                                                                              |          |
            |                     |                                              |              | input_tensor_shape[i] - 32 < output_tensor_shape[i] <= input_tensor_shape[i] |          |
            +---------------------+----------------------------------------------+--------------+------------------------------------------------------------------------------+----------+


            .. code-block:: python

                input_tensor_shape = [1, 1, 32, 32]
                output_tensor_shape = [1, 1, 3, 3]

                inp = torch.arange(start=1.0, end=10.0).reshape(1, 1, 3, 3)
                inp = torch.nn.functional.pad(inp, [0, input_tensor_shape[3] - inp.shape[3], 0, input_tensor_shape[2] - inp.shape[2]]).reshape(-1)
                tt_tensor = ttl.tensor.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttl.tensor.DataType.BFLOAT16,
                    ttl.tensor.Layout.ROW_MAJOR,
                )
                tt_tensor_unpadded = tt_tensor.unpad_from_tile(output_tensor_shape)

                print("Input tensor:")
                tt_tensor.pretty_print()
                print("\nUnpadded tensor:")
                tt_tensor_unpadded.pretty_print()

            Example output:

            .. code-block::

                Input tensor:
                [ [[[1, 2, 3, 0, ..., 0],
                    [4, 5, 6, 0, ..., 0],
                    [7, 8, 9, 0, ..., 0],
                    [0, 0, 0, 0, ..., 0],
                    ...,
                    [0, 0, 0, 0, ..., 0]]] dtype=bfloat16 ]

                Unpadded tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]
        )doc")
        .def("print", [](const Tensor &self, Layout print_layout) {
            return self.print(print_layout);
        }, py::arg("print_layout") = Layout::ROW_MAJOR, R"doc(
            Prints the tensor as a flat list of numbers. By default, the tensor will be printed in row major order.

            .. code-block:: python

                tt_tensor.print()

            Example output:

            .. code-block::

                [ 0.722656, 0.0332031, 0.109375, ..., 0.333984, 0.396484, 0.851562 dtype=bfloat16 ]
        )doc")
        .def("pretty_print", [](const Tensor &self) {
            return self.pretty_print();
        }, R"doc(
            Prints the tensor as list of nested lists. Number of levels of nesting is equal to tensor rank.

            .. code-block:: python

                tt_tensor.pretty_print()

            Example output for a rank 4 TT Tensor with shape (1, 1, 32, 32):

            .. code-block::

                [ [[[0.220703, 0.839844, 0.960938, ..., 0.378906, 0.507812],
                [0.03125, 0.511719, 0.0407715, ..., 0.945312, 0.671875],
                ...
                [0.433594, 0.165039, 0.980469, ..., , 0.349609]]] dtype=bfloat16 ]

        )doc")
        .def("shape", [](const Tensor &self) {
            const auto& shape = self.shape();
            return std::vector<std::uint32_t>(std::begin(shape), std::end(shape));
        }, R"doc(
            Get the shape of the tensor as list of integers.

            .. code-block:: python

                shape = tt_tensor.shape()

        )doc")
        .def("storage_type", [](const Tensor &self) {
            return self.storage_type();
        }, R"doc(
            Check if the tensor is on host

            .. code-block:: python

                storage_type = tt_tensor.storage_type()

        )doc")
        .def("device", [](const Tensor &self) {
            return self.device();
        }, R"doc(
            Get the device of the tensor.

            .. code-block:: python

                device = tt_tensor.device()

        )doc")
        .def("to_torch", [](const Tensor& self) -> py::object {
            return detail::convert_tt_tensor_to_torch_tensor(self);
        }, R"doc(
            Convert tensor to torch tensor.

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().to_torch() # move TT Tensor to host and convert it to torch tensor

        )doc")
        .def("buffer", [](const Tensor &self) -> std::variant<OwnedBuffer, BorrowedBuffer> {
            return std::visit(
                [] (auto&& storage) -> std::variant<OwnedBuffer, BorrowedBuffer> {
                    using T = std::decay_t<decltype(storage)>;
                    if constexpr (std::is_same_v<T, OwnedStorage>) {
                        return storage.buffer;
                    }
                    else if constexpr (std::is_same_v<T, DeviceStorage>) {
                        TT_THROW("Device storage doesn't support buffer method");
                    }
                    else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                        return storage.buffer;
                    }
                    else {
                        raise_unsupported_storage<T>();
                    }
                },
                self.storage()
            );
        }, R"doc(
            Get the underlying buffer.

            The tensor must be on the cpu when calling this function.

            .. code-block:: python

                buffer = tt_tensor.cpu().buffer() # move TT Tensor to host and get the buffer

        )doc")
        .def("layout", [](const Tensor &self) {
            return self.layout();
        }, R"doc(
            Get memory layout of TT Tensor.

            .. code-block:: python

                layout = tt_tensor.layout()

        )doc")
        .def("memory_config", [](const Tensor &self) {
            return self.memory_config();
        }, R"doc(
            Get buffer type of TT Tensor.

            .. code-block:: python

                memory_config = tt_tensor.memory_config()

        )doc")
        .def("dtype", [](const Tensor &self) {
            return self.dtype();
        }, R"doc(
            Get dtype of TT Tensor.

            .. code-block:: python

                dtype = tt_tensor.dtype()
        )doc")
        .def("shape_without_padding", [](const Tensor &self) {
            Shape shape_without_padding = self.shape().without_padding();
            std::array<uint32_t, 4> unpadded_shape;
            std::copy(std::begin(shape_without_padding), std::end(shape_without_padding), std::begin(unpadded_shape));
            return unpadded_shape;
        }, R"doc(
            Get shape without padding of TT Tensor.

            .. code-block:: python

                dtype = tt_tensor.shape_without_padding()
        )doc")
        .def("reshape", [](Tensor &self, int N, int C, int H, int W) {
            return self.reshape(N, C, H, W);
        }, R"doc(
            Reshapes TT tensor

            .. code-block:: python

                reshaped_tensor = tt_tensor.reshape(N, C, H, W)
        )doc");

    m_tensor.def("where", &where,
        py::arg("predicate"), py::arg("true_value"), py::arg("false_value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform an ternary where operation on two tensors based on third @predicate.

        where(predicate, true_value, false_value) implements (predicate) ? true_value : false_value.

        All three input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "predicate", "Predicate Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "true_value", "True Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "false_value", "False Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // *** eltwise binary ***

    detail::bind_binary_op(m_tensor, "add", add, R"doc(Perform an eltwise-binary add (``{0} + {1}``) on two tensors.)doc");
    detail::bind_binary_op(m_tensor, "sub", sub, R"doc(Perform an eltwise-binary sub (``{0} - {1}``) on two tensors.)doc");
    detail::bind_binary_op(m_tensor, "mul", mul, R"doc(Perform an eltwise-binary mul (``{0} * {1}``) on two tensors.)doc");
    detail::bind_binary_op(m_tensor, "squared_difference", squared_difference, R"doc(Perform an eltwise-binary squared_difference (``{0} - {1}``)^2 on two tensors.)doc");
    detail::bind_binary_op(m_tensor, "gt", gt, R"doc(Perform an eltwise-binary greater-than (``{0} > {1}``) on two tensors.)doc");
    detail::bind_binary_op(m_tensor, "lt", lt, R"doc(Perform an eltwise-binary less-than (``{0} < {1}``) on two tensors.)doc");
    detail::bind_binary_op(m_tensor, "lte", lte, R"doc(Perform an eltwise-binary less-than-or-equal (``{0} <= {1}``) on two tensors.)doc");
    detail::bind_binary_op(m_tensor, "gte", gte, R"doc(Perform an eltwise-binary greater-than-or-equal (``{0} >= {1}``) on two tensors.)doc");
    detail::bind_binary_op(m_tensor, "eq", eq, R"doc(Perform an eltwise-binary equal (``{0} == {1}``) on two tensors.)doc");
    detail::bind_binary_op(m_tensor, "ne", ne, R"doc(Perform an eltwise-binary not-equal (``{0} != {1}``) on two tensors.)doc");
    detail::bind_binary_op<false, true>(m_tensor, "max", &tt::tt_metal::max, R"doc(Perform an eltwise-binary max on two tensors.)doc");
    detail::bind_binary_op<false, true>(m_tensor, "min", &tt::tt_metal::min, R"doc(Perform an eltwise-binary min on two tensors.)doc");
    detail::bind_binary_op<false, true>(m_tensor, "hypot", &hypot, R"doc(Returns tensor with the hypot activation on elements of the input tensors ``{0}`` and ``{1}``.)doc");
    detail::bind_binary_op<false, true>(m_tensor, "xlogy", &xlogy, R"doc(Performs eltwise-binary xlogy (``{0} * log( {1} )``) on two tensors.)doc");
    detail::bind_binary_op<false, true>(m_tensor, "atan2", &atan2, R"doc(Returns tensor with the atan2 activation on elements of the input tensors ``{0}`` and ``{1}``.)doc");
    detail::bind_binary_op<false, true>(m_tensor, "ldexp", &ldexp, R"doc(Performs eltwise-binary ldexp (``{0} * 2**{1}``) on two tensors.)doc");
    detail::bind_binary_op<false, true>(m_tensor, "logaddexp", &logaddexp, R"doc(Perform an eltwise-binary logaddexp (``log(exp({0}) + exp({1}))``) on two tensors.)doc");
    detail::bind_binary_op<false, true>(m_tensor, "logaddexp2", logaddexp2, R"doc(Perform an eltwise-binary logaddexp2 (``log2(2^({0}) + 2^({1}))``) on two tensors.)doc");

    detail::bind_binary_op(m_tensor, "add_without_autoformat", add_without_autoformat,
        R"doc(Perform an eltwise-binary add (``{0} + {1}``) on two tensors.

        Auto formatting is disabled. Both input tensors must have TILE layout. Output tensor will have TILE layout.)doc"
    );

    detail::bind_unary_op(m_tensor, "mean_hw", tt::tt_metal::mean_hw, R"doc(  Returns a new tensor with the variance of the input tensor ``{0}`` on H,W axes.)doc");
    detail::bind_unary_op(m_tensor, "var_hw", tt::tt_metal::var_hw, R"doc(  Returns a new tensor with the variance of the input tensor ``{0}`` on H,W axes.)doc");
    detail::bind_unary_op(m_tensor, "std_hw", tt::tt_metal::std_hw, R"doc(Returns a new tensor with the standard deviation of the input tensor ``{0}`` on H,W axes.)doc");
    detail::bind_unary_op(m_tensor, "normalize_hw", tt::tt_metal::normalize_hw, R"doc(Returns a new tensor with the Gaussian normalize of the elements of the input tensor ``{0}`` on H,W axes.)doc");

    // *** eltwise unary ***
    m_tensor.def("move", &move,
        py::arg().noconvert(), py::arg("output_mem_config").noconvert() = std::nullopt, R"doc(
        Moves the elements of the input tensor ``arg0`` to a location in memory with specified memory layout.

        If no memory layout is specified, output memory will be the same as the input tensor memory config.

        +----------+----------------------------+----------------------------+---------------------------------+----------+
        | Argument | Description                | Data type                  | Valid range                     | Required |
        +==========+============================+============================+=================================+==========+
        | arg0     | Tensor to move             | Tensor                     | Tensor of shape [W, Z, Y, X]    | Yes      |
        +----------+----------------------------+----------------------------+---------------------------------+----------+
        | arg1     | MemoryConfig of tensor of  | tt_lib.tensor.MemoryConfig | Default is same as input tensor | No       |
        |          | TT accelerator device      |                            |                                 |          |
        +----------+----------------------------+----------------------------+---------------------------------+----------+
    )doc");

    detail::bind_unary_op(m_tensor, "exp", exp, R"doc(Returns a new tensor with the exponential of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "recip", recip, R"doc(Returns a new tensor with the reciprocal of the elements of the input tensor ``recip``.)doc");
    detail::bind_unary_op(m_tensor, "relu", relu, R"doc(Applies the rectified linear unit (ReLU) function to the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "relu6", relu6, R"doc(Returns tensor with the relu6 activation on elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "sqrt", sqrt, R"doc(Returns tensor with the square-root of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "sigmoid", sigmoid, R"doc(Applies the sigmoid function to the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "log", log, R"doc(Returns tensor with the natural logarithm of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "tanh", tanh, R"doc(Returns tensor with the hyperbolic tangent of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "log2", log2, R"doc(Returns tensor with the base 2 logarithm of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "log10", log10, R"doc(Returns tensor with the base 10 logarithm of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "sin", tt::tt_metal::sin, R"doc(Returns tensor with the sine of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "cos", tt::tt_metal::cos, R"doc(Returns tensor with the cosine of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "abs", abs, R"doc(Returns tensor with elementwise absolute value of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "sign", sign, R"doc(Returns tensor with the elementwise signum of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "square", square, R"doc(Returns tensor with the square of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "eqz", eqz, R"doc(Returns tensor with the result of equal to zero of all of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "nez", nez, R"doc(Returns tensor with the not equal zero of all of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "gtz", gtz, R"doc(Returns tensor with the greater than zero of all of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "ltz", ltz, R"doc(Returns tensor with the less than zero of all of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "gez", gez, R"doc(Returns tensor with the greater than equal zero of all of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "lez", lez, R"doc(Returns tensor with the less than equal zero of all of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "exp2", exp2, R"doc(Returns a new tensor with the exp2 (2 power) of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "expm1", expm1,
        R"doc(Returns a new tensor with the expm1 of the elements of the input tensor ``{0}``.
        expm1 = exp(x) - 1)doc"
    );
    detail::bind_unary_op(m_tensor, "signbit", signbit, R"doc(Applies the signbit function to the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "atan", atan, R"doc(Returns a new tensor with the arctan of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "atanh", atanh, R"doc(Returns a new tensor with the inverse hyperbolic tangent of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "asin", asin, R"doc(Returns a new tensor with the arcsine of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "acos", acos, R"doc(Returns a new tensor with the arccosine of the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "log_sigmoid", &log_sigmoid, R"doc(Applies the logsigmoid function to the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op_with_param(
        m_tensor, "gelu", &gelu,
        py::arg("fast_and_approx") = true,
        R"doc(Applies the Gaussian Error Linear Units (GELU) function to the elements of the input tensor ``{0}``.)doc",
        R"doc("Indicate true for approx and fast mode; false for accurate and slow mode", "bool", "default of true")doc"
    );
     detail::bind_unary_op_with_param(
        m_tensor, "erf", &erf,
        py::arg("fast_and_approx") = true,
        R"doc(Computes error function for all elements of the input tensor ``{0}``.)doc",
        R"doc("Indicate true for approx and fast mode; false for accurate and slow mode", "bool", "default of true")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "erfc", &erfc,
        py::arg("fast_and_approx") = true,
        R"doc(Computes complementary error function for all elements of the input tensor ``{0}``.)doc",
        R"doc("Indicate true for approx and fast mode; false for accurate and slow mode", "bool", "default of true")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "rsqrt", &rsqrt,
        py::arg("fast_and_approx") = true,
        R"doc(Returns a new tensor with the reciprocal of the square-root of each of the elements of the input tensor ``{0}``.)doc",
        R"doc("Indicate true for approx and fast mode; false for accurate and slow mode", "bool", "default of true")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "relu_max", relu_max,
        py::arg("upper_limit"),
        R"doc(Returns tensor with the relu max of all of elements of the input tensor ``{0}``. This is equivalent
        to relu_max[x] = relu(min(x, ``{1}``)). It caps off the input to a max value and a min value of 0.)doc",
        R"doc("max value", "float", "")doc"

    );
    detail::bind_unary_op_with_param(
        m_tensor, "relu_min", relu_min,
        py::arg("lower_limit"),
        R"doc(Returns tensor with the relu min of all of elements of the input tensor ``{0}``. This is equivalent
        to relu_min[x] = max(x, ``{1}``). It moves relu function down to carry out operation at minvalue
        instead of the standard 0.)doc",
        R"doc("min value", "float", "")doc"

    );
    detail::bind_unary_op_with_param(
        m_tensor, "elu", elu,
        py::arg("alpha"),
        R"doc(Returns tensor with the elu activation of all of elements of the input tensor ``{0}`` and scale
        factor alpha as ``{1}``. ELU(x) = alpha*(exp(x) - 1) if x < 0 else x.)doc",
        R"doc("alpha value", "float", "")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "heaviside", heaviside,
        py::arg("value"),
        R"doc(Returns tensor with the Heaviside step function of all of elements of the input tensor ``{0}`` and value factor as ``{1}``.

        HEAVISIDE(x) = 0 if x < 0 , 1 if x > 0 , else value.)doc",
        R"doc("value", "float", "")doc"

    );
    detail::bind_unary_op_with_param(
        m_tensor, "power", power,
        py::arg("exponent"),
        R"doc(Returns tensor with the all of elements of the input tensor ``{0}`` raised to ``{1}``.)doc",
        R"doc("exponent value", "int", ">=0")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "leaky_relu", leaky_relu,
        py::arg("slope"),
        R"doc(Returns tensor with the leaky relu of all of elements of the input tensor ``{0}`` with negative slope as ``{1}``.)doc",
        R"doc("slope value", "float", "")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "unary_chain", &unary_chain,
        py::arg("unary_chain"),
        R"doc(Returns tensor with the unary op chain applied to all of elements of the input tensor ``{0}``.)doc",
        R"doc("Unary op chain", "Vector<FusibleActivation>", "At least 1 activation")doc"
    );

    detail::bind_unary_op(m_tensor, "relu_without_autoformat", &relu_without_autoformat,
        R"doc(Applies the rectified linear unit (ReLU) function to the elements of the input tensor ``{0}``.

        Auto formatting is disabled. Input tensor must have TILE layout. Output tensor will have TILE layout.)doc"
    );

    // *** composite unary ops ***
    detail::bind_unary_op(m_tensor, "sinh", &tt::tt_metal::sinh, R"doc(Returns tensor with the hyperbolic sine of elements of the input tensor ``{0}`` in range [-9,9] with high accuracy.)doc");
    detail::bind_unary_op(m_tensor, "cosh", &tt::tt_metal::cosh, R"doc(Returns tensor with the hyperbolic cosine of elements of the input tensor ``{0}`` in range [-9,9] with high accuracy.)doc");
    detail::bind_unary_op(m_tensor, "softsign", &softsign, R"doc(Applies the softsign function to the elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "softplus", &softplus, R"doc(Returns tensor with the softplus activation of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "log1p", &log1p, R"doc(Returns tensor with the natural log of 1 added to all of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "silu", &silu, R"doc(Returns tensor with the silu all of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "swish", swish, R"doc(Returns tensor with the swish all of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "mish", &mish, R"doc(Returns tensor with the mish activation of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "cbrt", &cbrt, R"doc(Returns tensor with the cbrt activation of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "asinh", &asinh, R"doc(Returns tensor with the inverse hyperbolic sine of elements of the input tensor ``{0}`` in range [-1e-6, 1e6].
        for +input , output = asinh(input)
        for -input , output = -asinh(input))doc"
    );
    detail::bind_unary_op(m_tensor, "acosh", &acosh, R"doc(Returns tensor with the inverse hyperbolic cosine of elements of the input tensor ``{0}`` in range [-1e-6, 1e6].
        for  input > 1, output = acosh(input)
        for  input ==1, ouptut = 0
        for  input < 1, output =  nan)doc"
    );
    detail::bind_unary_op(m_tensor, "tanhshrink", &tanhshrink,
        R"doc(Applies tanh on the input tensor ``{0}`` and subtracted from the input tensor.

        ``tanhshrink(x) = x - tanh(x)``)doc"
    );

    detail::bind_unary_op_with_param(
        m_tensor, "softshrink", &softshrink,
        py::arg("lambda"),
        R"doc(Applies the softshrink function to the elements of the input tensor ``{0}`` between limits ``-{1}`` low and
        the ``+{1}`` high limits.)doc",
        R"doc("value limits (-lambda to +lambda)", "float", ">= 0")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "hardshrink", &hardshrink,
        py::arg("lambda"),
        R"doc(Applies the hardshrink function to the elements of the input tensor ``{0}`` between limits ``-{1}`` low and
        the ``+{1}`` high limits.)doc",
        R"doc("value limits (-lambda to +lambda)", "float", ">= 0")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "bias_gelu", &bias_gelu,
        py::arg("bias"),
        R"doc(Applies the Gelu activation function to the elements of the biased ``{1}`` input tensor ``{0}``.)doc",
        R"doc("value limits (-bias to +bias)", "float", ">= 0")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "polyval", &polyval,
        py::arg("coeffs"),
        R"doc(Returns tensor with the polyval of all of elements of the input tensor ``{0}`` with coefficients ``{1}``.)doc",
        R"doc("coefficients value with highest degree first", "List of float", "List size > 0")doc"
    );


    // *** bcast binary tied to unary ***
    detail::bind_unary_op(m_tensor, "add1", &add1, R"doc(Returns tensor with the addition of one with input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "neg", &neg, R"doc(Returns tensor with the negate all of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "deg2rad", &deg2rad, R"doc(Returns tensor with the deg2rad conversion of elements of the input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "rad2deg", &rad2deg, R"doc(Returns tensor with the rad2deg conversion of elements of the input tensor ``{0}``.)doc");

    detail::bind_unary_op_with_param(
        m_tensor, "add_unary", py::overload_cast<const Tensor&, float, const MemoryConfig&>(&add_unary),
        py::arg("scalar"),
        R"doc(Perform an eltwise-binary add on one tensor ``{0}`` and one scalar ``{1}``.)doc",
        R"doc("Scalar", "float", "")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "sub_unary", py::overload_cast<const Tensor&, float, const MemoryConfig&>(&sub_unary),
        py::arg("scalar"),
        R"doc(Perform an eltwise-binary sub on one tensor ``{0}`` and one scalar ``{1}``.)doc",
        R"doc("Scalar", "float", "")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "mul_unary", py::overload_cast<const Tensor&, float, const MemoryConfig&>(&mul_unary),
        py::arg("scalar"),
        R"doc(Perform an eltwise-binary mul on one tensor ``{0}`` and one scalar ``{1}``.)doc",
        R"doc("Scalar", "float", "")doc"
    );
    detail::bind_unary_op_with_param(
        m_tensor, "div_unary", py::overload_cast<const Tensor&, float, const MemoryConfig&>(&div_unary),
        py::arg("scalar"),
        R"doc(Perform an eltwise-binary div on one tensor ``{0}`` and one scalar ``{1}``.)doc",
        R"doc("Scalar", "float", "")doc"
    );

    m_tensor.def("add_unary", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&add_unary),
        py::arg("scalar"), py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform an eltwise-binary add on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "scalar", "Scalar", "float", "", "Yes"
            "input", "Tensor to add", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("sub_unary", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&sub_unary),
        py::arg("scalar"), py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform an eltwise-binary sub on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "scalar", "Scalar", "float", "", "Yes"
            "input", "Tensor to sub", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

    )doc");

    m_tensor.def("mul_unary", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&mul_unary),
        py::arg("scalar"), py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform an eltwise-binary mul on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "scalar", "Scalar", "float", "", "Yes"
            "input", "Tensor to mul", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

    )doc");

    m_tensor.def("div_unary", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&div_unary),
        py::arg("scalar"), py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform an eltwise-binary div on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "scalar", "Scalar", "float", "", "Yes"
            "input", "Tensor to div", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

    )doc");

    m_tensor.def("hardtanh", &hardtanh,
        py::arg("input").noconvert(), py::arg("low") = -1.0f, py::arg("high") = +1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Applies the hard tanh function to the elements of the input tensor ``input``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor hardtanh is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "low", "Low value (PyTorch default)", "float", "default to -1.0f", "No"
            "high", "High value (PyTorch default)", "float", "default to +1.0f", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("clip", &clip,
        py::arg("input").noconvert(), py::arg("low"), py::arg("high"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Applies the clip function to the elements of the input tensor ``input`` between limits ``low`` low and
        the ``high`` high limits.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor hardtanh is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "low", "Low value)", "float", "", "Yes"
            "high", "High value", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");


    m_tensor.def("hardsigmoid", &hardsigmoid,
        py::arg("input").noconvert(), py::arg("scale") = 1.0f/6.0f, py::arg("shift") = 0.5f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Applies the hardsigmoid function to the elements of the input tensor ``input``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor hardsigmoid is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "scale", "Scale value (PyTorch default)", "float", "default to 1.0/6.0f", "No"
            "shift", "Shift value (PyTorch default)", "float", "default to 0.5f", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("lerp", py::overload_cast<const Tensor&, const Tensor&, float, const MemoryConfig&>(&lerp),
        py::arg("input").noconvert(), py::arg("end").noconvert(), py::arg("weight"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
        Applies the linear interpolation of two tensors ``arg0`` (given by input) and ``arg1`` based on a
        scalar ``arg2`` and returns the resulting out tensor.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor lerp is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "end", "End value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "weight", "Weight value", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("lerp", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&lerp),
        py::arg("input").noconvert(), py::arg("end").noconvert(), py::arg("weight").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Applies the linear interpolation of two tensors ``arg0`` (given by input) and ``arg1`` based on a
        tensor ``arg2`` and returns the resulting out tensor.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor lerp is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "end", "End value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "weight", "Weight value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("hardswish", &hardswish,
        py::arg("input").noconvert(), py::arg("scale") = 1.0f/6.0f, py::arg("shift") = 0.5f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Applies the hard swish function to the elements of the input tensor ``input``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor hardswish is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "scale", "Scale value (PyTorch default)", "float", "default to 1.0/6.0f", "No"
            "shift", "Shift value (PyTorch default)", "float", "default to 0.5f", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("subalpha", &subalpha,
        py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Subtracts ``other``, scaled by ``alpha``, from ``input``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor subalpha is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "alpha", "Alpha value", "float", "default to 1.0f", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("full_like", &full_like,
        py::arg("input").noconvert(), py::arg("fill_value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a new tensor filled with the scalar value shaped like reference tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Reference Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "fill_value", "Fill value", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("zeros_like", &zeros_like,
        py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a new tensor filled with zeros shaped like reference tensor ``input``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Reference Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");


    m_tensor.def("ones_like", &ones_like,
        py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a new tensor filled with ones shaped like reference tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Reference Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("zeros",
        [] (const std::array<uint32_t, 4> shape, Layout layout, Device * device, const MemoryConfig& output_mem_config) {
            return zeros(shape, layout, device, output_mem_config);
        },
        py::arg("shape"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a new tensor filled with zeros in shape specified by input ``shape``.

        Input shape is specified as a list of 4 integer elements

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
            "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
            "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("ones",
        [] (const std::array<uint32_t, 4> shape, Layout layout, Device * device, const MemoryConfig& output_mem_config) {
            return ones(shape, layout, device, output_mem_config);
        },
        py::arg("shape"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a new tensor filled with ones in shape specified by input ``shape``.

        Input shape is specified as a list of 4 integer elements

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
            "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
            "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("full",
        [] (const std::array<uint32_t, 4> shape, float value, Layout layout, Device * device, const MemoryConfig& output_mem_config) {
            return full(shape, value, layout, device, output_mem_config);
        },
        py::arg("shape"), py::arg("fill_value"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a new tensor filled with the scalar value in shape specified by input ``shape``.

        Input shape is specified as a list of 4 integer elements

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
            "fill_value", "Fill value ", "float", "", "Yes"
            "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
            "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("arange", &arange,
        py::arg("start"), py::arg("end"), py::arg("step"), py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a new 1D tensor with the incremented values in size specified by inputs ``start``, ``end`` and ``step``.

        Inpute scalars are integers specifying start, end, and step sizes.
        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "start", "Start", "int", "", "Yes"
            "end", "End", "int", "> start", "Yes"
            "step", "Step", "int", "> 0", "Yes"
            "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

#if 0
    m_tensor.def("bitwise_complement", &bitwise_complement, R"doc(
        Returns tensor with the bitwise complement of elements of the input tensor ``arg0``.

        Input tensor must have UINT32 data type.

        Output tensor will have UINT32 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor bitwise complement |           |                              |          |
        |          | '~' is applied to         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");


    m_tensor.def("logical_not", &logical_not, R"doc(
        Returns tensor with the logical notof elements of the input tensor ``arg0``.

        Input tensor must have UINT32 data type.

        Output tensor will have UINT32 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor logical not        |           |                              |          |
        |          | '!' is applied to         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");
#endif


#if 0
    m_tensor.def("mean", &mean, R"doc(
        Returns tensor with the mean of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor mean is computed   | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("std", &tt::tt_metal::std, R"doc(
        Returns tensor with the std of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor std is computed on | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("normalize", &normalize, R"doc(
        Returns tensor with the normalization of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor std normalized     | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");
#endif

     m_tensor.def("addcmul", &addcmul,
        py::arg("input").noconvert(), py::arg("tensor1").noconvert(), py::arg("tensor2").noconvert(), py::arg("value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Performs the element-wise multiplication of tensor1 ``tensor1`` by tensor2 ``tensor2``, multiplies the result
        by the scalar value ``value`` and adds it to input ``input``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor addcmul is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "tensor1", "First Tensor to multiply", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "tensor2", "Second tensor to multiply", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "value", "Value to be multiplied", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("addcdiv", &addcdiv,
        py::arg("input").noconvert(), py::arg("tensor1").noconvert(), py::arg("tensor2").noconvert(), py::arg("value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Performs the element-wise division of tensor1 ``tensor1`` by tensor2 ``tensor2``, multiplies the result
        by the scalar value ``value`` and adds it to input ``input``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor addcdiv is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "tensor1", "Numerator Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "tensor2", "Denominator Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "value", "Value to be multiplied", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");



    m_tensor.def("mac", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&mac),
        py::arg("input").noconvert(), py::arg("tensor1").noconvert(), py::arg("tensor2").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns tensor with the multiply and accumulation of all of elements of the input tensors ``input, tensor1, tensor2``.
        Output is ``input x tensor1 + tensor2`` elementwise operator.
        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor mac is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "tensor1", "Tensor to be multiplied", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "tensor2", "Tensor to be added", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("mac", py::overload_cast<const Tensor&, float, float, const MemoryConfig&>(&mac),
        py::arg("input").noconvert(), py::arg("float1"), py::arg("float2"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns tensor with the multiply and accumulation of all of elements of the input tensor ``input11 with``float1, float2``.
        Output is ``tensor1 x float1 + float2`` elementwise operator.
        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor mac is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "float1", "Value to be multiplied", "float", "", "Yes"
            "float2", "Value to be added", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

     m_tensor.def("threshold", &threshold,
        py::arg("input").noconvert(), py::arg("threshold"), py::arg("value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns tensor with the threshold activation on elements of the input tensors ``arg0`` at threshold ``threshold``,
        and value ``value``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor threshold is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "threshold", "Value to threshold at", "float", "", "Yes"
            "value", "Value to replace with", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // *** matrix multiplication ***
    m_tensor.def("matmul", &matmul,
        py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform a non-batched matrix multiplication ``arg0 x arg1`` with two tensors.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "First tensor to multiply", "Tensor", "Tensor of shape [1, 1, Y, S]", "Yes"
            "other", "Second tensor to multiply", "Tensor", "Tensor of shape [1, 1, S, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("outer", &outer,
        py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform a non-batched outer product multiplication ``arg0 x arg1`` with two tensors.

        Both input tensors must have BFLOAT16 data type but shape [1,1,N,1] and [1,1,1,M] respectively
        or reshapeable with only one major dimension while other 3 being squeezable dimensions.

        Output tensor will have BFLOAT16 data type but of shape [1,1,N,M].

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "First tensor to multiply", "Tensor", "Tensor of shape [1, 1, N, 1]", "Yes"
            "other", "Second tensor to multiply", "Tensor", "Tensor of shape [1, 1, 1, M]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("bmm", &bmm,
        py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform a batched matmul ``arg0 x arg1`` with two tensors, where batch dims match.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "First tensor to multiply", "Tensor", "Tensor of shape [W, Z, Y, S]", "Yes"
            "other", "Second tensor to multiply", "Tensor", "Tensor of shape [W, Z, S, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // *** tensor manipulation ***
    m_tensor.def("concat", py::overload_cast<Tensor&, Tensor&, uint32_t, const MemoryConfig&>(&concat),
        py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("dim"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Concatenates shape of tensors ``arg0`` and ``arg1`` to new shape ``[W, Z, Y, X]`` along the specified dimension ``arg2``.

        Input tensors must be on device, in ROW MAJOR or TILE layout, and have BFLOAT16 data type.

        Output tensor will be on device, in same layout, and have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X], where concat dim %32=0", "Yes"
            "other", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X], where concat dim %32=0", "Yes"
            "dim", "dimension of concat", "int", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("concat", py::overload_cast<std::vector<Tensor>&, uint32_t, const MemoryConfig&>(&concat),
        py::arg("input_tensors").noconvert(), py::arg("dim"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Concatenates shape of tensors ``arg0`` and ``arg1`` to new shape ``[W, Z, Y, X]`` along the specified dimension ``arg1``.

        Input tensors must be on device, in ROW MAJOR or TILE layout, and have BFLOAT16 data type.

        Output tensor will be on device, in same layout, and have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input_tensors", "Input tensors to concat", "List of Tensors", "Tensors of shape [W, Z, Y, X], where concat dim %32=0", "Yes"
            "dim", "dimension of concat", "int", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("reshape", &reshape,
        py::arg("input").noconvert(), py::arg("W"), py::arg("Z"), py::arg("Y"), py::arg("X"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a tensor with the new shape of ``[W, Z, Y, X]``. The X dimension of input and output tensor must have same size.

        Input tensor must be on host device, in TILE layout, and have BFLOAT16 data type.

        Output tensor will be on host device, in TILE layout, and have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "W", "W dim of output tensor", "int", "", "Yes"
            "Z", "Z dim of output tensor", "int", "", "Yes"
            "Y", "Y dim of output tensor", "int", "", "Yes"
            "X", "X dim of output tensor", "int", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    detail::bind_unary_op_with_param(
        m_tensor, "sum", &sum,
        py::arg("dim"),
        R"doc(Returns a tensor that is a sum  of input tensor with shape ``[W, Z, Y, X]`` along dimensions ``{1}``.)doc",
        R"doc("dimension to sum along", "int", "0, 1, 2, or 3")doc"
    );

    m_tensor.def("transpose", py::overload_cast<const Tensor&, uint, uint, const MemoryConfig&>(&transpose),
        py::arg("input").noconvert(), py::arg("dim0"), py::arg("dim1"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``arg1`` and ``arg2`` are swapped.

        Input tensor must have BFLOAT16 data type. Second and third input specify the dimensions of tensor to be transposed.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "dim0", "dimension to transpose", "uint", "0, 1, 2, or 3", "Yes"
            "dim1", "dimension to transpose", "uint", "0, 1, 2, or 3", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    //transpose = transpose_wh
    detail::bind_unary_op(m_tensor, "transpose", py::overload_cast<const Tensor&, const MemoryConfig&>(&transpose), R"doc(Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``X`` and ``Y`` are swapped.)doc");
    detail::bind_unary_op(m_tensor, "transpose_hc", &transpose_hc, R"doc(Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``Y`` and ``Z`` are swapped.)doc");
    detail::bind_unary_op(m_tensor, "transpose_cn", &transpose_cn, R"doc(Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``Z`` and ``W`` are swapped.)doc");
    detail::bind_unary_op(m_tensor, "transpose_nh", &transpose_nh, R"doc(Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``W`` and ``Y`` are swapped.)doc");
    detail::bind_unary_op(m_tensor, "transpose_cw", &transpose_cw, R"doc(Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``Z`` and ``X`` are swapped.)doc");
    detail::bind_unary_op(m_tensor, "transpose_nw", &transpose_nw, R"doc(Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``W`` and ``X`` are swapped.)doc");

    m_tensor.def("permute", &permute,
        py::arg("input").noconvert(), py::arg("W"), py::arg("Z"), py::arg("Y"), py::arg("X"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a tensor that is input tensor ``arg0`` with its dimensions permuted to new order ``[arg1, arg2, arg3, arg4]``.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "W", "Dim to become W", "int", "Unique value between [0, num dims)", "Yes"
            "Z", "Dim to become Z", "int", "Unique value between [0, num dims)", "Yes"
            "Y", "Dim to become Y", "int", "Unique value between [0, num dims)", "Yes"
            "X", "Dim to become X", "int", "Unique value between [0, num dims)", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("tilize", &tilize,
        py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Changes data layout of input tensor to TILE.

        Input tensor must be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X] where Y%32=0 and X%32=0", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("tilize_with_zero_padding", &tilize_with_zero_padding,
        py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Tilizes a given tensor across memory on device. Pads zeroes height-wise and width-wise if required.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("tilize_with_val_padding",
        [] (const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value, const MemoryConfig& output_mem_config) {
            return tilize_with_val_padding(tensor, output_tensor_shape, input_tensor_start, pad_value, output_mem_config);
        },
        py::arg("input").noconvert(), py::arg("output_tensor_shape").noconvert(), py::arg("input_tensor_start"), py::arg("pad_value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Tilizes a given tensor across memory on device. Pads to specified shape before tilizing.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "", "Yes"
            "output_tensor_shape", "Shape of output tensor", "List[int[4]]", "Shape [W, Z, Y, X] where Y%32=0 and X%32=0", "Yes"
            "input_tensor_start", "Start indices to place input tensor in output tensor", "List[int[4]]", "Must be all 0s", "Yes"
            "pad_value", "Value to pad input tensor", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("untilize", &untilize,
        py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Changes data layout of input tensor to ROW_MAJOR.

        Input tensor must be on TT accelerator device, in TILE, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X] where Y%32=0 and X%32=0", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("untilize_with_unpadding",
        [] (const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, const MemoryConfig& output_mem_config) {
            return untilize_with_unpadding(tensor, output_tensor_shape, input_tensor_start, output_mem_config);
        },
        py::arg("input").noconvert(), py::arg("output_tensor_start").noconvert(), py::arg("output_tensor_end"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Changes data layout of input tensor to ROW_MAJOR and unpads/removes elements from the tensor.

        Input tensor must be on TT accelerator device, in TILE, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X] where Y%32=0 and X%32=0", "Yes"
            "output_tensor_start", "Start indices of input tensor", "List[int[4]]", "Must be all 0s", "Yes"
            "output_tensor_end", "End indices of input tensor in output tensor", "List[int[4]]", "Values along each dim must be < input_tensor_shape[i]", "Yes"
            "pad_value", "Value to pad input tensor", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("pad",
        [] (const Tensor &input_tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value, const MemoryConfig& output_mem_config) {
            return pad(input_tensor, output_tensor_shape, input_tensor_start, pad_value, output_mem_config);
        },
        py::arg("input").noconvert(), py::arg("output_tensor_shape").noconvert(), py::arg("input_tensor_start"), py::arg("pad_value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Pad TT Tensor with given pad value ``arg2``.

        The input tensor must be in ROW_MAJOR or TILE layout.

        Returns an output tensor that contains the input tensor at the given input tensor start indices ``arg3`` and the padded value everywhere else.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "", "Yes"
            "output_tensor_shape", "Shape of output tensor", "List[int[4]]", "", "Yes"
            "input_tensor_start", "Start indices to place input tensor in output tensor", "List[int[4]]", "Must be all 0s", "Yes"
            "pad_value", "Value to pad input tensor", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("unpad",
        [] (const Tensor &input_tensor, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end, const MemoryConfig& output_mem_config) {
            return unpad(input_tensor, output_tensor_start, output_tensor_end, output_mem_config);
        },
        py::arg("input").noconvert(), py::arg("output_tensor_start").noconvert(), py::arg("output_tensor_end"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Unpad TT Tensor.

        Returns an output tensor from output tensor start indices ``arg1`` to output tensor end indices ``arg2`` (inclusive) of the input tensor.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "", "Yes"
            "output_tensor_start", "Start indices of input tensor", "List[int[4]]", "Must be all 0s", "Yes"
            "output_tensor_end", "End indices of input tensor in output tensor", "List[int[4]]", "Values along each dim must be < input_tensor_shape[i]", "Yes"
            "pad_value", "Value to pad input tensor", "float", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // *** broadcast and reduce ***
    m_tensor.def("bcast", &bcast,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("math_op"), py::arg("dim"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform a binary elementwise operation ``math_op`` between tensors ``input`` and ``other``, where values from tensor ``other`` are broadcast.

        Let tensor ``input`` have shape ``[W0, Z0, Y0, X0]`` and tensor ``other`` shape ``[W1, Z1, Y1, X1]``. ``arg3`` determines the type of broadcast performed.

        For ``dim=BcastOpDim::W`` broadcast is performed on dimension ``X``. ``Y0`` and ``Y1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``dim=BcastOpDim::H`` broadcast is performed on dimension  ``Y``. ``X0`` and ``X1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``dim=BcastOpDim::HW`` broadcast is performed on dimensions ``X`` and ``Y``. Either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1) must hold for input shapes.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0]", "Yes"
            "other", "Input tensor to broadcast", "Tensor", "Tensor of shape [W1, Z1, Y1, X1]", "Yes"
            "math_op", "Aggregating math operation", " BcastOpMath", "ADD, SUB, MUL", "Yes"
            "dim", "Dimension on which to broadcast", "BcastOpDim", "W, H, HW", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("bcast_without_autoformat", &bcast_without_autoformat,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("math_op"), py::arg("dim"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform a binary elementwise operation ``arg2`` between tensors ``arg0`` and ``arg1``, where values from tensor ``arg1`` are broadcast.

        Let tensor ``arg0`` have shape ``[W0, Z0, Y0, X0]`` and tensor ``arg1`` shape ``[W1, Z1, Y1, X1]``. ``arg3`` determines the type of broadcast performed.

        For ``arg3=BcastOpDim::W`` broadcast is performed on dimension ``X``. ``Y0`` and ``Y1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``arg3=BcastOpDim::H`` broadcast is performed on dimension  ``Y``. ``X0`` and ``X1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``arg3=BcastOpDim::HW`` broadcast is performed on dimensions ``X`` and ``Y``. Either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1) must hold for input shapes.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        Auto formatting is disabled. Input tensors must have TILE layout. Output tensors will have TILE layout.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0], where Y0%32=0 and X0%32=0", "Yes"
            "other", "Input tensor to broadcast", "Tensor", "Tensor of shape [W1, Z1, Y1, X1], where Y1%32=0 and X1%32=0", "Yes"
            "math_op", "Aggregating math operation", " BcastOpMath", "ADD, SUB, MUL", "Yes"
            "dim", "Dimension on which to broadcast", "BcastOpDim", "W, H, HW", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("reduce", &reduce,
        py::arg("input").noconvert(), py::arg("math_op"), py::arg("dim"), py::arg("scaler"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Perform a reduction of input tensor ``arg0`` using mathematical operation ``arg1`` on dimension ``arg2``.

        For ``arg2=ReduceOpDim::W`` reduce is done on dimension X.

        For ``arg2=ReduceOpDim::H`` reduce is done on dimension Y.

        For ``arg2=ReduceOpDim::HW`` reduce is done on dimensions X and Y.

        Input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "math_op", "Aggregating math operation", " ReduceOpMath", "SUM, MAX", "Yes"
            "dim", "Dimension on which reduction is performed", "ReduceOpDim", "W, H, HW", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // *** experimental operations ***
    m_tensor.def("fill_rm", &fill_rm,
        py::arg("N"), py::arg("C"), py::arg("H"), py::arg("W"), py::arg("hOnes"), py::arg("wOnes"), py::arg("any").noconvert(), py::arg("val_hi"), py::arg("val_lo"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Generates an NCHW row-major tensor and fill it with high values up to
        hOnes, wOnes in each HW tile with the rest padded with high values. So
        for H=2, W=3, hFill=1, wFill=2 the following tensor will be generated:

        .. code-block::

            +------------> W
            | hi hi lo
            | lo lo lo
            |
            v H

        H, W are expected to be multiples of 32.

        The 'any' Tensor arg is only used to pass the device and resulting
        tensor dtype.

        val_hi/lo are expected to be floats.

        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | Argument | Description                                                           | Data type             | Valid range            | Required |
        +==========+=======================================================================+=======================+========================+==========+
        | N        | Batch count of output tensor                                          | int                   | N > 0                  | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | C        | Channel count of output tensor                                        | int                   | C > 0                  | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | H        | Height count of output tensor                                         | int                   | H > 0                  | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | W        | Width count of output tensor                                          | int                   | W > 0                  | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | hOnes    | Height of high values region                                          | int                   | hOnes <= H             | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | wOnes    | Width of high values region                                           | int                   | wOnes <= W             | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | any      | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | val_hi   | High value to use                                                     | float                 |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | val_lo   | Low value to use                                                      | float                 |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
    )doc");
    m_tensor.def("fill_ones_rm", &fill_ones_rm,
        py::arg("N"), py::arg("C"), py::arg("H"), py::arg("W"), py::arg("hOnes"), py::arg("wOnes"), py::arg("any").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Same as ``fill_rm``, but ``val_hi`` is set to ``1`` and ``val_lo`` is
        ``0``.

        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | Argument | Description                                                           | Data type             | Valid range            | Required |
        +==========+=======================================================================+=======================+========================+==========+
        | N        | Batch count of output tensor                                          | int                   | N > 0                  | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | C        | Channel count of output tensor                                        | int                   | C > 0                  | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | H        | Height count of output tensor                                         | int                   | H > 0                  | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | W        | Width count of output tensor                                          | int                   | W > 0                  | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | hOnes    | Height of high values region                                          | int                   | hOnes <= H             | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | wOnes    | Width of high values region                                           | int                   | wOnes <= W             | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | any      | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
    )doc");

    // matrix multiplication
    m_tensor.def("bmm_tilize_untilize", &bmm_tilize_untilize, R"doc(
        Perform a batched matmul ``A x B`` with two tensors, where batch and channel dims match.
        This op also supports tiling tensor A and untiling the output.

        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------|
        | Argument                      | Description                                           | Data type | Valid range | Required |
        +===============================+=======================================================+===========+=============+==========+
        | a                             | LHS matmul operand                                    | Tensor    |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
        | b                             | RHS matmul operand                                    | Tensor    |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
        | a_height_nblocks              | Number of blocks along A's height                     | uint32_t  |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
        | a_width_nblocks               | Number of blocks along A's width (= along B's height) | uint32_t  |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
        | b_width_nblocks               | Number of blocks along B's width                      | uint32_t  |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
        | a_block_height_ntiles         | Number of tiles along height of an A block            | uint32_t  |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
        | a_block_width_ntiles          | Number of tiles along width of an A block             | uint32_t  |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
        | b_block_width_ntiles          | Number of tiles along width of a B block              | uint32_t  |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
        | out_subblock_height_ntiles    | Height of subblocks on height for output              | uint32_t  |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
        | out_subblock_width_ntiles     | Number of subblocks on width for output               | uint32_t  |             | Yes      |
        +-------------------------------+-------------------------------------------------------+-----------+-------------+----------+
    )doc");
    m_tensor.def("conv", &conv, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def("conv_with_fast_reader", &conv_with_fast_reader, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def("conv_with_address_map", &conv_with_address_map, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output
        Reader kernel uses an address map which pre-computed on the host to read activations and weights

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    // Custom BERT matmuls/bmms
    m_tensor.def("bert_large_fused_qkv_matmul", &bert_large_fused_qkv_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_fused_qkv non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_ff1_matmul", &bert_large_ff1_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("fuse_gelu_activation") = false, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_ff1 non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_ff2_matmul", &bert_large_ff2_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_ff2 non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_selfout_matmul", &bert_large_selfout_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_selfout non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_pre_softmax_bmm", &bert_large_pre_softmax_bmm,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_pre_softmax_bmm batched matmul ``[9, 16, 384, 64] x [9, 16, 64, 384]`` with two tensors and returns a reshaped output of [9, 1, 6144, 384].
    )doc");
    m_tensor.def("bert_large_post_softmax_bmm", &bert_large_post_softmax_bmm,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_post_softmax_bmm batched matmul by reshaping tensor A to [9, 16, 384, 384] first, then returning ``[9, 16, 384, 384] x [9, 16, 384, 64]``.
    )doc");

    // Custom Falcon matmuls/bmms
    m_tensor.def("falcon_fused_qkv_matmul", &falcon_fused_qkv_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a falcon_fused_qkv non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("falcon_selfout_matmul", &falcon_selfout_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a falcon_selfout non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("falcon_dense_4h_to_h_matmul", &falcon_dense_4h_to_h_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a falcon_dense_4h_to_h non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("falcon_dense_h_to_4h_matmul", &falcon_dense_h_to_4h_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("fuse_gelu_activation") = false, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a falcon_dense_h_to_4h non-batched matmul ``A x B`` with two tensors. This invokes the MULTI_CORE matmul parallelization. This parallelization does not support bias option yet.
    )doc");
    m_tensor.def("falcon_lm_head_matmul", &falcon_lm_head_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Perform a falcon_lm_head non-batched matmul ``A x B`` with two tensors. This invokes the MULTI_CORE matmul parallelization. This parallelization does not support bias option yet.
    )doc");

    // Custom Generic NLP TMs
    // TODO: Uplift nlp_create_qkv_heads to support generic qkv num_heads and head_dim
    // This op should support arbitrary B and S divisible by 32 on DRAM; on L1, might error out due to space
    m_tensor.def("nlp_create_qkv_heads", &nlp_create_qkv_heads,
        py::arg().noconvert(), py::arg("output_mem_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Shuffles [B, 1, S, 4672] fused qkv matrix into 3 heads with shapes [B, 71, S, 64], [B, 1, S, 64], and [B, 1, S, 64].
    )doc");
    // TODO: Uplift nlp_concat_heads to support generic num_heads and head_dim
    // This op should support arbitrary B and S divisible by 32 on DRAM; on L1, might error out due to space
    m_tensor.def("nlp_concat_heads", &nlp_concat_heads,
        py::arg().noconvert(), py::arg("output_mem_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Shuffles [B, 71, S, 64] tensor into tensor with shape [B, 1, S, 4544].
    )doc");

    // groupnorm
    m_tensor.def("groupnorm", &groupnorm,
        py::arg("input").noconvert(), py::arg("group_size").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a groupnorm operation on the channel dimension grouped per group_size, with optional fused with post-multiplication and addition via W-bcast.
    )doc");

    // layernorm
    m_tensor.def("layernorm", &layernorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a layernorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
    )doc");
    m_tensor.def("add_layernorm", &add_layernorm,
        py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a layernorm(a+b)*gamma + beta operation."
    )doc");
    m_tensor.def("rmsnorm", &rmsnorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a rmsnorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
    )doc");
    m_tensor.def("rotate_half", &rotate_half,
        py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a rotate half operation used by RotaryEmbedding.
    )doc");
    m_tensor.def("rotary_embedding", &rotary_embedding,
        py::arg("input").noconvert(), py::arg("cos").noconvert(), py::arg("sin").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs rotary embedding with a given input, cos, and sin tensors. Sequence length is inferred as the second last dim of the input tensor.
    )doc");
    m_tensor.def("update_cache", &update_cache,
         py::arg("cache").noconvert(), py::arg("input").noconvert(), py::arg("batch_idx"), py::arg("update_idx"), R"doc(
        "Fills the cache tensor in place with the values from input at the specified batch_idx, update_idx.
    )doc");


    // input embeddings
    m_tensor.def("embeddings", &embeddings,
        py::arg("input").noconvert(), py::arg("weights").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns specific indices of the embedding table specified by the input tensor
        +----------------+-------------------------------------+------------+-------------------------------+----------+
        | Argument       | Description                         | Data type  | Valid range                   | Required |
        +================+=====================================+============+===============================+==========+
        | num_embeddings | Number of rows in embedding table   | uint32     |                               | Yes      |
        | input          | Tensor containing rows we want      | Tensor     |                               | Yes      |
        | weights        | Entire Embedding Table              | Tensor     |                               | Yes      |
        +----------------+-------------------------------------+------------+-------------------------------+----------+
    )doc");


    // FC
    m_tensor.def("fully_connected", &fully_connected,
        py::arg("act").noconvert(), py::arg("weights").noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Fully connected layer (linear.)

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "act", "Input activations tensor", "Tensor", "", "Yes"
            "weights", "Input weights tensor", "Tensor", "", "Yes"
            "bias", "Input bias tensor", "Tensor", "", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // Pools
    m_tensor.def("average_pool_2d", &average_pool_2d, R"doc(
        Average Pool 2D
        It operates on tensors whose that have channels as the last dimension

        +----------+----------------------------+------------+-------------------------------+----------+
        | Argument | Description                | Data type  | Valid range                   | Required |
        +==========+============================+============+===============================+==========+
        | act      | Input activations tensor   | Tensor     |                               | Yes      |
        +----------+----------------------------+------------+-------------------------------+----------+
    )doc");
    m_tensor.def("max_pool2d", &max_pool2d,
        py::arg("input").noconvert(),
        py::arg("in_h").noconvert(), py::arg("in_w").noconvert(),
        py::arg("kernel_h").noconvert(), py::arg("kernel_w").noconvert(),
        py::arg("stride_h") = 1, py::arg("stride_w") = 1,
        py::arg("pad_h") = 0, py::arg("pad_w") = 0,
        py::arg("dilation_h") = 1, py::arg("dilation_w") = 1,
        py::arg("output_mem_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("nblocks") = 1, R"doc(
        Max Pool 2D
        +-------------------+-------------------------------+---------------+-------------+----------+
        | Argument          | Description                   | Data type     | Valid range | Required |
        +===================+===============================+===============+=============+==========+
        | input             | Input activations tensor      | Tensor        |             | Yes      |
        | in_h              | Input height                  | Tensor        |             | Yes      |
        | in_w              | Input width                   | Tensor        |             | Yes      |
        | kernel_h          | kernel window height          | uint32_t      |             | Yes      |
        | kernel_w          | kernel window width           | uint32_t      |             | Yes      |
        | stride_h          | stride in height dim          | uint32_t      |             | No       |
        | stride_w          | stride in width dim           | uint32_t      |             | No       |
        | pad_h             | padding in height dim         | uint32_t      |             | No       |
        | pad_w             | padding in width dim          | uint32_t      |             | No       |
        | dilation_h        | kernel dilation in height dim | uint32_t      |             | No       |
        | dilation_w        | kernel dilation in width dim  | uint32_t      |             | No       |
        | output_mem_config | output tensor memory config   | MemoryConfig  |             | No       |
        +-------------------+-------------------------------+---------------+-------------+----------+
    )doc");

    // TMs
    m_tensor.def("split_last_dim_two_chunks_tiled", &split_last_dim_two_chunks_tiled, py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Splits a tensor's last dimension in two equal sized chunks. This assumes the last dim is tile sized.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

    )doc");
    m_tensor.def("convert_conv_weight_tensor_to_tiled_layout", &convert_conv_weight_tensor_to_tiled_layout, R"doc(
       Converts convolution weights to 2d matrix tiled layout on host
       Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def("convert_conv_weight_tensor_to_special_padding_tiled_layout", &convert_conv_weight_tensor_to_special_padding_tiled_layout, R"doc(
       Converts convolution weights to 2d matrix tiled layout on host with special block height padding
       Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def(
        "log_fallback_operation",
        [] (const py::function& fallback_operation, const py::args& args, const py::kwargs& kwargs) -> void {

            auto function_name = py::cast<std::string>(fallback_operation.attr("__qualname__"));

            std::vector<Tensor> input_tensors;
            tt::stl::reflection::Attributes attributes;

            auto process_name_and_value = [&function_name, &input_tensors, &attributes] (const auto& name, const auto& value) {
                py::object torch = py::module_::import("torch");
                if (py::isinstance<Tensor>(value)) {
                    auto tensor = py::cast<Tensor>(value);
                    input_tensors.push_back(tensor);
                }
                else if (py::isinstance(value, torch.attr("nn").attr("Module"))) {
                    // do nothing
                }
                else if (py::isinstance(value, torch.attr("Tensor"))) {
                    auto tensor = detail::convert_torch_tensor_to_tt_tensor(value);
                    input_tensors.push_back(tensor);
                }
                else {
                    attributes.push_back({fmt::format("{}", name), fmt::format("{}", value)});
                }
            };

            auto arg_index = 0;
            for (const auto& value : args) {
                auto name = fmt::format("arg_{}", arg_index++);
                process_name_and_value(name, value);
            }

            for (const auto& [name, value] : kwargs) {
                process_name_and_value(name, value);
            }

            auto operation = PythonFallbackOperation{function_name, attributes};
            operation::log_operation(operation, input_tensors);
        }, R"doc(
        Log fallback operation using operation infrastructure.

            +----------+----------------------+-----------+-------------+----------+
            | Argument | Description          | Data type | Valid range | Required |
            +==========+======================+===========+=============+==========+
            | function | Fallback Function    | Function  |             | Yes      |
            +----------+----------------------+-----------+-------------+----------+
            | args     | Packed args          | tuple     |             | No       |
            +----------+----------------------+-----------+-------------+----------+
            | kwargs   | Packed kwargs        | dict      |             | No       |
            +----------+----------------------+-----------+-------------+----------+
        )doc"
    );
    m_tensor.def(
        "format_input_tensor",
        [] (const Tensor &input, Device * device, const std::array<uint32_t, 4>& padded_shape, float pad_value, Layout target_layout, std::optional<MemoryConfig> target_mem_config = std::nullopt) {
            return AutoFormat::format_input_tensor(input, device, padded_shape, pad_value, target_layout, target_mem_config);
        },
        py::arg("input").noconvert(), py::arg("device").noconvert(), py::arg("padded_shape"), py::arg("pad_value"), py::arg("target_layout").noconvert(), py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and pads to padded shape
        )doc"
    );
    m_tensor.def(
        "format_output_tensor",
        [] (const Tensor &output, const std::array<uint32_t, 4>& shape, Device* device, Layout target_layout, std::optional<MemoryConfig> target_mem_config = std::nullopt) {
            return AutoFormat::format_output_tensor(output, shape, device, target_layout, target_mem_config);
        },
        py::arg("output").noconvert(), py::arg("shape"), py::arg("device").noconvert(), py::arg("target_layout").noconvert(), py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and unpads to shape
        )doc"
    );
    m_tensor.def(
        "pad_to_tile_shape",
        [] (const std::array<uint32_t, 4>& unpadded_shape, bool pad_c=false, bool pad_n=false, bool pad_h=true, bool pad_w=true) {
            Shape padded_shape_object = AutoFormat::pad_to_tile_shape(unpadded_shape, pad_c, pad_n, pad_h, pad_w);
            std::array<uint32_t, 4> padded_shape;
            std::copy(std::begin(padded_shape_object), std::end(padded_shape_object), std::begin(padded_shape));
            return padded_shape;
        }, R"doc(
            Returns shape padded to tile shape
        )doc"
    );

    m_tensor.def(
        "dump_tensor",
        &dump_tensor,
        R"doc(
            Dump tensor to file
        )doc"
    );

    m_tensor.def(
        "load_tensor",
        &load_tensor,
        R"doc(
            Load tensor to file
        )doc"
    );
}

void DeviceModule(py::module &m_device) {
    py::enum_<tt::ARCH>(m_device, "Arch", "Enum of types of Tenstorrent accelerator devices.")
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL)
        .value("WORMHOLE_B0", tt::ARCH::WORMHOLE_B0);

    auto pyDevice = py::class_<Device>(m_device, "Device", "Class describing a Tenstorrent accelerator device.");
    pyDevice
        .def(
            py::init<>(
                [](tt::ARCH arch, int pcie_slot) {
                    return Device(arch, pcie_slot);
                }
            ), "Create device."
        );

    m_device.def("CreateDevice", &CreateDevice, R"doc(
        Creates an instance of TT device.

        +------------------+------------------------+---------------------+------------------------------+----------+
        | Argument         | Description            | Data type           | Valid range                  | Required |
        +==================+========================+=====================+==============================+==========+
        | arch             | Type of TT Device      | tt_lib.device.Arch  | tt_lib.device.Arch.GRAYSKULL | Yes      |
        +------------------+------------------------+---------------------+------------------------------+----------+
        | pci_express_slot | PCI Express slot index | int                 |                              | Yes      |
        +------------------+------------------------+---------------------+------------------------------+----------+
    )doc");
    m_device.def("InitializeDevice", &InitializeDevice, py::arg().noconvert(), R"doc(
        Initialize instance of TT accelerator device.

        +-------------------+--------------------------------------------------------+----------------------------------+-------------------------------------------+----------+
        |  Argument         |                 Description                            |       Data type                  |           Valid range                     | Required |
        +===================+========================================================+==================================+============================================+=========+
        | device            | Device to initialize                                   | tt_lib.device.Device             |                                           | Yes      |
        +-------------------+--------------------------------------------------------+----------------------------------+-------------------------------------------+----------+
    )doc");
    m_device.def("CloseDevice", &CloseDevice, R"doc(
        Reset an instance of TT accelerator device to default state and relinquish connection to device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to close     | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("SetDefaultDevice", &AutoFormat::SetDefaultDevice, R"doc(
        Sets the default device to use for ops when inputs aren't on device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to use       | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("GetDefaultDevice", &AutoFormat::GetDefaultDevice, R"doc(
        Gets the default device to use for ops when inputs aren't on device.
    )doc");

    m_device.def("StartDebugPrintServer", &StartDebugPrintServer);

    m_device.def("EnablePersistentKernelCache", &detail::EnablePersistentKernelCache, R"doc(
        Enable kernel compilation cache to be persistent across runs. When this is called, kernels will not be compiled if the output binary path exists.
    )doc");
    m_device.def("DisablePersistentKernelCache", &detail::DisablePersistentKernelCache, R"doc(
        Disables kernel compilation cache from being persistent across runs
    )doc");
    m_device.def("EnableCompilationReports", &detail::EnableCompilationReports, R"doc(
        Enables tt-metal to generate reports of compilation statistics
    )doc");
    m_device.def("DisableCompilationReports", &detail::DisableCompilationReports, R"doc(
        Disables generation of compilation statistics reports in tt-metal
    )doc");

    m_device.def("EnableMemoryReports", &detail::EnableMemoryReports, R"doc(
        Enables tt-metal to generate reports of memory allocation statistics
    )doc");
    m_device.def("DisableMemoryReports", &detail::DisableMemoryReports, R"doc(
        Disables generation of memory allocation statistics reports in tt-metal
    )doc");

    m_device.def("DumpDeviceMemoryState", &detail::DumpDeviceMemoryState, R"doc(
        Generates reports to dump device memory state. Three reports are generated:
        - `l1_usage_summary.csv` has a table with an entry for each program indicating the minimum largest free L1 block and size of largest L1 buffer that can be interleaved across available free L1 blocks
        - `memory_usage_summary.csv` for each program there is an entry indicating total allocatable, allocated, free, and largest free block sizes for each DRAM and L1 bank
        - `detailed_memory_usage.csv` expands on the memory usage summary report by including each memory block address, size, and allocation status

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to dump memory state for  | tt_lib.device.Device  |             | Yes      |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("Synchronize", &detail::Synchronize, R"doc(
        Wait for all kernels on TT device to complete.
    )doc");
}

void ProfilerModule(py::module &m_profiler) {
    py::enum_<op_profiler::OpType>(m_profiler, "OpType")
        .value("python_fallback", op_profiler::OpType::python_fallback)
        .value("custom_zone", op_profiler::OpType::custom_zone);

    m_profiler.def("get_profiler_flag", &op_profiler::get_profiler_flag, R"doc(
        Gets the profiling flag.
    )doc");

    m_profiler.def("set_profiler_location", &op_profiler::set_profiler_location, R"doc(
        Sets the profiling root folder.

        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | profilerLocation | Profiling out folder                           | string                | Valid dir   | Yes      |
        |                  | Default : tt_metal/tools/profiler/logs/ops/    |                       |             |          |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def("append_meta_data", &op_profiler::append_meta_data, R"doc(
        Append extra information regardig the op.

        +------------------+------------------------+-----------------------+------------------+----------+
        | Argument         | Description            | Data type             | Valid range      | Required |
        +==================+========================+=======================+==================+==========+
        | metaData         | Meta Data              | string                | Non-empty string | Yes      |
        +------------------+------------------------+-----------------------+------------------+----------+
    )doc");

    m_profiler.def("append_input_data", &op_profiler::append_input_data, R"doc(
        Append op input information .

        +------------------+------------------------+-----------------------+------------------+----------+
        | Argument         | Description            | Data type             | Valid range      | Required |
        +==================+========================+=======================+==================+==========+
        | input            | Input tensor           | Tensor                | Valid Tensor     | Yes      |
        +------------------+------------------------+-----------------------+------------------+----------+
    )doc");

    m_profiler.def("append_output_data", &op_profiler::append_output_data, R"doc(
        Append op output information .

        +------------------+------------------------+-----------------------+------------------+----------+
        | Argument         | Description            | Data type             | Valid range      | Required |
        +==================+========================+=======================+==================+==========+
        | output           | output tensor          | Tensor                | Valid Tensor     | Yes      |
        +------------------+------------------------+-----------------------+------------------+----------+
    )doc");

    m_profiler.def("set_preferred_name", &op_profiler::set_preferred_name<string>, R"doc(
        Set a name to be appended to the name that profiler started with.

        +------------------+------------------------+-----------------------+------------------+----------+
        | Argument         | Description            | Data type             | Valid range      | Required |
        +==================+========================+=======================+==================+==========+
        | name             | Preferred Name         | String                | Valid String     | Yes      |
        +------------------+------------------------+-----------------------+------------------+----------+
    )doc");

    m_profiler.def("start_profiling",
		  &op_profiler::start_profiling,py::arg("opName"), py::arg("opType") = op_profiler::OpType::custom_zone, R"doc(
        Start profiling op.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | Name             | Name of the op or zone to be profiled          | string                |             | Yes      |
        | Type             | Fallback op or custom zone                     | string                |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def("stop_profiling", &op_profiler::stop_profiling, R"doc(
        Stop profiling op.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | Name             | Name of the op or zone to stop profiling       | string                |             | Yes      |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

}

void DTXModule(py::module &m_dtx) {
    auto pyDataTransformations = py::class_<DataTransformations>(m_dtx, "DataTransformations", "Class describing the data transformations.");
    m_dtx.def("evaluate", [](vector<float> data, vector<uint32_t> address_map, vector<vector<int>> output_shape){
        return evaluate(data, address_map, output_shape);
    }, R"doc(
        Evaluates data transformation on host cpu.
        +------------------+----------------------------+-----------------------+-------------+----------+
        | Argument         | Description                 | Data type            | Valid range | Required |
        +==================+=============================+======================+=============+==========+
        | data             | Input data to transform     | vector of floats     |             | Yes      |
        | address_map      | address mapping from src to dst  |  vector of uint32_t |      | Yes      |
        | output shape     | shape of the dst tensor |  vector of int |      | Yes      |
        +------------------+-----------------------------+----------------------+-------------+----------+
    )doc");
    m_dtx.def("conv_transform", [](vector<int> activation_shape,
                                        vector<int> weight_shape,
                                        vector<int> conv_params,
                                        uint32_t in0_block_h,
                                        uint32_t in0_block_w,
                                        uint32_t in1_block_w,
                                        uint32_t num_blocks_in0_h,
                                        uint32_t num_blocks_in1_w,
                                        uint32_t num_bytes_of_df,
                                        bool skip_activation_transform){
        return conv_transform(activation_shape, weight_shape, conv_params, in0_block_h, in0_block_w, in1_block_w, num_blocks_in0_h, num_blocks_in1_w, num_bytes_of_df, skip_activation_transform);
    });
}



void ProgramCacheModule(py::module &m_program_cache) {
   m_program_cache.def("enable", &tt::tt_metal::program_cache::enable);
   m_program_cache.def("disable_and_clear", &tt::tt_metal::program_cache::disable_and_clear);
   m_program_cache.def("num_entries", &tt::tt_metal::program_cache::num_entries);
}

} // end namespace tt_metal

} // end namespace tt


PYBIND11_MODULE(_C, m) {

    m.attr("__name__") = "tt_lib";
    m.doc() = "Python bindings for TT-Metal";

    py::module_ m_device = m.def_submodule("device", "Submodule defining a host or device");
    tt::tt_metal::DeviceModule(m_device);

    py::module_ m_profiler = m.def_submodule("profiler", "Submodule defining the profiler");
    tt::tt_metal::ProfilerModule(m_profiler);

    py::module_ m_tensor = m.def_submodule("tensor", "Submodule defining an tt_metal tensor");
    tt::tt_metal::TensorModule(m_tensor);

    py::module_ m_dtx = m.def_submodule("dtx", "Submodule defining data transformation engine");
    tt::tt_metal::DTXModule(m_dtx);

    py::module_ m_program_cache = m.def_submodule("program_cache", "Submodule for caching operations");
    tt::tt_metal::ProgramCacheModule(m_program_cache);

    py::module_ m_operations = m.def_submodule("operations", "Submodule for operations");
    tt::operations::py_module(m_operations);
}
