#include "tt_lib_bindings_tensor.hpp"
#include "tensor/tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/conv/conv_op.hpp"
#include "tt_dnn/op_library/conv/optimized_conv_op.hpp"
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
#include "tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.hpp"
#include "tt_dnn/op_library/clone/clone_op.hpp"
#include "tt_dnn/op_library/rotate_half/rotate_half_op.hpp"
#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_op.hpp"
#include "tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "tt_dnn/op_library/update_cache/update_cache_op.hpp"
#include "tt_dnn/op_library/move/move_op.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/borrowed_buffer.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/serialization.hpp"

namespace tt::tt_metal{


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

namespace detail{

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

    detail::export_enum<MathFidelity>(m_tensor);

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
                    tt_device = tt_lib.device.CreateDevice(0)
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
                    tt_device = tt_lib.device.CreateDevice(0)
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
    m_tensor.def("concat", &concat,
        py::arg("input_tensors").noconvert(), py::arg("dim") = 0, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Concatenates shape of tensors ``arg0`` and ``arg1`` to new shape ``[W, Z, Y, X]`` along the specified dimension ``arg1``.

        Input tensors must be on device, in ROW MAJOR or TILE layout, and have matching data type.

        Output tensor will be on device, in same layout, and have same data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input_tensors", "Input tensors to concat", "List of Tensors", "Tensors of shape [W, Z, Y, X], where Y or X must be a multiple of 32 if they are the concat dim", "Yes"
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

    detail::bind_unary_op_with_param(
        m_tensor, "sum", &sum,
        py::arg("dim"),
        R"doc(Returns a tensor that is a sum  of input tensor with shape ``[W, Z, Y, X]`` along dimensions ``{1}``.)doc",
        R"doc("dimension to sum along", "int", "0, 1, 2, or 3")doc"
    );
    detail::bind_unary_op(m_tensor, "clone", &clone, R"doc(  Returns a new tensor which is a new copy of input tensor ``{0}``.)doc");
    detail::bind_unary_op(m_tensor, "mean_hw", tt::tt_metal::mean_hw, R"doc(  Returns a new tensor with the variance of the input tensor ``{0}`` on H,W axes.)doc");

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

    py::class_<OptimizedConvParallelizationConfig>(m_tensor, "OptimizedConvParallelizationConfig")
        .def(
            py::init<>(
                [] (
                    std::tuple<std::size_t, std::size_t> grid_size,
                    uint32_t per_core_act_matrix_height_ntiles
                ) {
                    return OptimizedConvParallelizationConfig{
                        .grid_size={std::get<0>(grid_size), std::get<1>(grid_size)},
                        .per_core_act_matrix_height_ntiles=per_core_act_matrix_height_ntiles
                    };

                }
            ),
            py::kw_only(),
            py::arg("grid_size").noconvert(),
            py::arg("per_core_act_matrix_height_ntiles").noconvert()
        );

    m_tensor.def("optimized_conv", &optimized_conv,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert() = 0, R"doc(
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

    m_tensor.def("conv_with_fast_reader", &conv_with_fast_reader,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4, R"doc(
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
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("fused_activation") = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
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
        py::arg("input").noconvert(), py::arg("cos").noconvert(), py::arg("sin").noconvert(), py::arg("token_idx") = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs rotary embedding with a given input, cos, and sin tensors. Sequence length is inferred as the second last dim of the input tensor.
        If token_idx is passed, this assumes input is transposed to [seq_len, 1, B, head_dim], and seq_len is 1.
    )doc");
    m_tensor.def("fill_cache", &fill_cache,
         py::arg("cache").noconvert(), py::arg("input").noconvert(), py::arg("batch_idx"), R"doc(
        "Fills the cache tensor in place with the values from input at the specified batch_idx.
    )doc");
    m_tensor.def("update_cache", &update_cache,
         py::arg("cache").noconvert(), py::arg("input").noconvert(), py::arg("update_idx"), R"doc(
        "Updates the cache tensor in place with the values from input at the specified update_idx.
    )doc");


    // input embeddings
    m_tensor.def("embeddings", &embeddings,
        py::arg("input").noconvert(), py::arg("weights").noconvert(),
        py::arg("split_weights").noconvert() = false,
        py::arg("tilized").noconvert() = false,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns specific indices of the embedding table specified by the input tensor
        +----------------+----------------------------------------------+------------+-------------------------------+----------+
        | Argument       | Description                                  | Data type  | Valid range                   | Required |
        +================+==============================================+============+===============================+==========+
        | num_embeddings | Number of rows in embedding table            | uint32     |                               | Yes      |
        | input          | Tensor containing rows we want               | Tensor     |                               | Yes      |
        | weights        | Entire Embedding Table                       | Tensor     |                               | Yes      |
        | split_weights  | Parallelizing over weights (instead of input)| Bool       |                               | No       |
        | tilized        | Enable fused tilize on output                | Bool       |                               | No       |
        +----------------+----------------------------------------------+------------+-------------------------------+----------+
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
    m_tensor.def("average_pool_2d", &average_pool_2d,
        py::arg().noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,  R"doc(
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
        py::arg("nblocks") = 1,
        py::arg("use_multicore") = true, R"doc(
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

    detail::TensorModuleCompositeOPs( m_tensor);
}

}
