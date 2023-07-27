#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/conv/conv_op.hpp"
#include "tt_dnn/op_library/fill_rm/fill_rm_op.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_dnn/op_library/groupnorm/groupnorm_op.hpp"
#include "tt_dnn/op_library/pool/average_pool.hpp"
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
#include "tt_dnn/op_library/bert_large_tms/bert_large_tms.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.hpp"
#include "tt_dnn/op_library/move/move_op.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/borrowed_buffer.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_lib_bindings.hpp"
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


OwnedBuffer create_owned_buffer_from_list_of_floats(std::vector<float>&& data, DataType data_type) {
    switch (data_type) {
        case DataType::BFLOAT8_B:
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

Tensor convert_torch_tensor_to_tt_tensor(const py::handle& torch_tensor) {
    py::object torch = py::module_::import("torch");
    if (not py::isinstance(torch_tensor, torch.attr("Tensor"))) {
        TT_THROW("The argument must be of type torch.Tensor!");
    }

    auto torch_dtype = torch_tensor.attr("dtype");
    auto shape = py::cast<std::vector<uint32_t>>(torch_tensor.attr("shape"));

    auto on_creation_callback = [torch_tensor] {
        torch_tensor.inc_ref();
    };

    auto on_destruction_callback = [torch_tensor] {
        torch_tensor.dec_ref();
    };

    if (torch_dtype.equal(torch.attr("float32"))) {
        auto data_ptr = reinterpret_cast<float*>(py::cast<std::size_t>(torch_tensor.attr("data_ptr")()));
        auto storage = BorrowedStorage(
            borrowed_buffer::Buffer(data_ptr, volume(shape)),
            on_creation_callback,
            on_destruction_callback
        );
        return Tensor(std::move(storage), shape, DataType::FLOAT32, Layout::ROW_MAJOR);
    }
    else if (torch_dtype.equal(torch.attr("bfloat16"))) {
        auto data_ptr = reinterpret_cast<bfloat16*>(py::cast<std::size_t>(torch_tensor.attr("data_ptr")()));
        auto storage = BorrowedStorage(
            borrowed_buffer::Buffer(data_ptr, volume(shape)),
            on_creation_callback,
            on_destruction_callback
        );
        return Tensor(std::move(storage), shape, DataType::BFLOAT16, Layout::ROW_MAJOR);
    }
    else if (torch_dtype.equal(torch.attr("int64"))) {
         // TODO(arakhmati): add DataType::INT64
        auto data_ptr = reinterpret_cast<uint32_t*>(py::cast<std::size_t>(torch_tensor.attr("data_ptr")()));
        auto storage = BorrowedStorage(
            borrowed_buffer::Buffer(data_ptr, volume(shape) * 2), // times 2 because INT64 is converted to UINT32
            on_creation_callback,
            on_destruction_callback
        );
        return Tensor(std::move(storage), shape, DataType::UINT32, Layout::ROW_MAJOR);
    }
    else {
        TT_THROW(fmt::format("Unsupported DataType: {}", py::repr(torch_dtype)));
    }
}

}

void TensorModule(py::module &m_tensor) {
    // ENUM SECTION

    // bast enums
    py::enum_<BcastOpMath::Enum>(m_tensor, "BcastOpMath")
        .value("ADD", BcastOpMath::Enum::ADD)
        .value("SUB", BcastOpMath::Enum::SUB)
        .value("MUL", BcastOpMath::Enum::MUL);
    /** TODO: add these to bcast ops - good to have not required
        .value("GT", BcastOpMath::Enum::GT)
        .value("LT", BcastOpMath::Enum::LT)
        .value("GE", BcastOpMath::Enum::GE)
        .value("LE", BcastOpMath::Enum::LE)
        .value("EQ", BcastOpMath::Enum::EQ)
        .value("NEQ", BcastOpMath::Enum::NE);
    */

    py::enum_<BcastOpDim::Enum>(m_tensor, "BcastOpDim")
        .value("H", BcastOpDim::Enum::H)
        .value("W", BcastOpDim::Enum::W)
        .value("HW", BcastOpDim::Enum::HW);

    // reduce enums
    py::enum_<ReduceOpMath::Enum>(m_tensor, "ReduceOpMath")
        .value("SUM", ReduceOpMath::Enum::SUM)
        .value("MAX", ReduceOpMath::Enum::MAX);

    py::enum_<ReduceOpDim::Enum>(m_tensor, "ReduceOpDim")
        .value("H", ReduceOpDim::Enum::H)
        .value("W", ReduceOpDim::Enum::W)
        .value("HW", ReduceOpDim::Enum::HW);

    // layout enums
    py::enum_<Layout>(m_tensor, "Layout")
        .value("ROW_MAJOR", Layout::ROW_MAJOR)
        .value("TILE", Layout::TILE);

    py::enum_<DataType>(m_tensor, "DataType")
        .value("FLOAT32", DataType::FLOAT32)
        .value("BFLOAT16", DataType::BFLOAT16)
        .value("UINT32", DataType::UINT32)
        .value("BFLOAT8_B", DataType::BFLOAT8_B);

    py::enum_<BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1);

    py::enum_<StorageType>(m_tensor, "StorageType")
        .value("OWNED", StorageType::OWNED)
        .value("DEVICE", StorageType::DEVICE)
        .value("BORROWED", StorageType::BORROWED);


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
                    auto owned_buffer = detail::create_owned_buffer_from_list_of_floats(std::move(data), data_type);
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
                    auto owned_buffer = detail::create_owned_buffer_from_list_of_floats(std::move(data), data_type);
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
                    auto owned_buffer = detail::create_owned_buffer_from_list_of_floats(std::move(data), data_type);
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
                [](const py::object& torch_tensor) {
                    return detail::convert_torch_tensor_to_tt_tensor(torch_tensor);
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
        }, py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, py::keep_alive<0, 2>(), R"doc(
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
        .def("data", [](const Tensor &self) -> std::variant<OwnedBuffer, BorrowedBuffer> {
            return std::visit(
                [] (auto&& storage) -> std::variant<OwnedBuffer, BorrowedBuffer> {
                    using T = std::decay_t<decltype(storage)>;
                    if constexpr (std::is_same_v<T, OwnedStorage>) {
                        return storage.buffer;
                    }
                    else if constexpr (std::is_same_v<T, DeviceStorage>) {
                        TT_THROW("Device storage doesn't support data method");
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
            Get data in the tensor as a list of numbers.

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().data() # move TT Tensor to host and get values stored in it

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
        )doc");

    m_tensor.def("where", &where, R"doc(
        Perform an ternary where operation on two tensors based on third @predicate.

        where(predicate, true_value, false_value) implements (predicate) ? true_value : false_value.

        All three input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | Predicate tensor     | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | True tensor          | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg2     | False tensor         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    // *** eltwise binary tied to unary ***
    m_tensor.def("add_unary", py::overload_cast<const Tensor&,float>(&add_unary), R"doc(
        Perform an eltwise-binary add on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | Tensor to add        | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Scalar               | float     |                              | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");
    m_tensor.def("add_unary", py::overload_cast<float,const Tensor&>(&add_unary), R"doc(
        Perform an eltwise-binary add on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | Scalar               | float     |                              | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Tensor to add        | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+

    )doc");

    m_tensor.def("sub_unary", py::overload_cast<const Tensor&,float>(&sub_unary), R"doc(
        Perform an eltwise-binary sub on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | tensor to sub        | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | scalar               | float     |                              | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");
    m_tensor.def("sub_unary", py::overload_cast<float,const Tensor&>(&sub_unary), R"doc(
        Perform an eltwise-binary sub on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | Scalar               | float     |                              | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Tensor to sub        | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+

    )doc");

    m_tensor.def("mul_unary", py::overload_cast<const Tensor&,float>(&mul_unary), R"doc(
        Perform an eltwise-binary mul on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | Tensor to mul        | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Scalar               | float     |                              | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");
    m_tensor.def("mul_unary", py::overload_cast<float,const Tensor&>(&mul_unary), R"doc(
        Perform an eltwise-binary mul on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | Scalar               | float     |                              | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Tensor to mul        | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+

    )doc");

    m_tensor.def("div_unary", py::overload_cast<const Tensor&,float>(&div_unary), R"doc(
        Perform an eltwise-binary div on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | Tensor to div        | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Scalar               | float     |                              | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");
    m_tensor.def("div_unary", py::overload_cast<float,const Tensor&>(&div_unary), R"doc(
        Perform an eltwise-binary div on one tensor and one scalar.

        Both inputs, the tensor and scalar, must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | Scalar               | float     |                              | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Tensor to div        | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+

    )doc");

    // *** eltwise binary ***
    m_tensor.def("add", add, R"doc(
        Perform an eltwise-binary add on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to add  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to add | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("add_without_autoformat", add_without_autoformat, R"doc(
        Perform an eltwise-binary add on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        Auto formatting is disabled. Both input tensors must have TILE layout. Output tensor will have TILE layout.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to add  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to add | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("sub", sub, R"doc(
        Perform an eltwise-binary sub (``arg0 - arg1``) on two tensors.\

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to sub  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to sub | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("mul", mul, R"doc(
        Perform an eltwise-binary mul on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to mul  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to mul | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");


    m_tensor.def("gt", gt, R"doc(
        Perform an eltwise-binary greater-than, >, on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to gt   | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to gt  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("lt", lt, R"doc(
        Perform an eltwise-binary less-than, <, on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to lt   | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to lt  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("lte", lte, R"doc(
        Perform an eltwise-binary less-than, <=, on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to lte  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to lte | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("gte", gte, R"doc(
        Perform an eltwise-binary greater-than, >=, on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to gte  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to gte | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("eq", eq, R"doc(
        Perform an eltwise-binary equal, ==, on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to eq   | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to eq  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("ne", ne, R"doc(
        Perform an eltwise-binary not equal, !=, on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to ne   | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to ne  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");


    m_tensor.def("max", &tt::tt_metal::max, R"doc(
        Perform an eltwise-binary max on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to max  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to max | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("min", &tt::tt_metal::min, R"doc(
        Perform an eltwise-binary min on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------+-----------+------------------------------+----------+
        | Argument | Description          | Data type | Valid range                  | Required |
        +==========+======================+===========+==============================+==========+
        | arg0     | First tensor to min  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to min | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("squared_difference", &squared_difference, R"doc(
        Perform an eltwise-binary squared_difference on two tensors.

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        +----------+-------------------------------------+-----------+------------------------------+----------+
        | Argument | Description                         | Data type | Valid range                  | Required |
        +==========+=====================================+===========+==============================+==========+
        | arg0     | First tensor to squared_difference  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to squared_difference | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------------+-----------+------------------------------+----------+
    )doc");

    // *** eltwise unary ***
    m_tensor.def("move", &move,
        py::arg().noconvert(), py::arg("mem_config").noconvert() = std::nullopt, R"doc(
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
    m_tensor.def("gelu", &gelu,
		 py::arg().noconvert(), py::arg("fast_and_appx") = true, R"doc(
        Applies the Gaussian Error Linear Units (GELU) function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor GELU is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
        | arg1     | Indicate true for approx   | bool      | default of true              | No       |
        |          | and fast mode; false for   |           |                              |          |
        |          | accurate and slow mode     |           |                              |          |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("rsqrt", &rsqrt, R"doc(
        Returns a new tensor with the reciprocal of the square-root of each of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor rsqrt is aplied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("relu", &relu, R"doc(
        Applies the rectified linear unit (ReLU) function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor ReLU is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("relu_without_autoformat", &relu_without_autoformat, R"doc(
        Applies the rectified linear unit (ReLU) function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        Auto formatting is disabled. Input tensor must have TILE layout. Output tensor will have TILE layout.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor ReLU is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("sigmoid", &sigmoid, R"doc(
        Applies the sigmoid function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-------------------------------+-----------+------------------------------+----------+
        | Argument | Description                   | Data type | Valid range                  | Required |
        +==========+===============================+===========+==============================+==========+
        | arg0     | Tensor sigmoid is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("hardtanh", &hardtanh,
		 py::arg().noconvert(), py::arg("low") = -1.0f, py::arg("high") = +1.0f, R"doc(
        Applies the hard tanh function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-------------------------------+-----------+------------------------------+----------+
        | Argument | Description                   | Data type | Valid range                  | Required |
        +==========+===============================+===========+==============================+==========+
        | arg0     | Tensor hardtanh is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
        | low      | Low value (PyTorch default)   | float     | default to -1.0f             | No       |
        +----------+-------------------------------+-----------+------------------------------+----------+
        | high     | High value (PyTorch default)  | float     | default to +1.0f             | No       |
        +----------+-------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("clip", &clip, R"doc(
        Applies the clip function to the elements of the input tensor ``arg0`` between limits ``arg1`` low and
        the ``arg2`` high limits.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-------------------------------+-----------+------------------------------+----------+
        | Argument | Description                   | Data type | Valid range                  | Required |
        +==========+===============================+===========+==============================+==========+
        | arg0     | Tensor clip is applied to     | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
        | low      | Low value (PyTorch default)   | float     |                              | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
        | high     | High value (PyTorch default)  | float     |                              | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("softshrink", &softshrink, R"doc(
        Applies the softshrink function to the elements of the input tensor ``arg0`` between limits ``-arg1`` low and
        the ``+arg1`` high limits.

        Input tensor must have BFLOAT16 data type. Input arg1 is parameter BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------------+-----------+------------------------------+----------+
        | Argument | Description                     | Data type | Valid range                  | Required |
        +==========+=================================+===========+==============================+==========+
        | arg0     | Tensor softshrink is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------------+-----------+------------------------------+----------+
        | arg1     | value limits (-arg1 to +arg1)   | float     | >= 0                         | Yes      |
        +----------+---------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("hardshrink", &hardshrink, R"doc(
        Applies the hardshrink function to the elements of the input tensor ``arg0`` between limits ``-arg1`` low and
        the ``+arg1`` high limits.

        Input tensor must have BFLOAT16 data type. Input arg1 is parameter BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------------+-----------+------------------------------+----------+
        | Argument | Description                     | Data type | Valid range                  | Required |
        +==========+=================================+===========+==============================+==========+
        | arg0     | Tensor hardshrink is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------------+-----------+------------------------------+----------+
        | arg1     | value limits (-arg1 to +arg1)   | float     | >= 0                         | Yes      |
        +----------+---------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("softsign", &softsign, R"doc(
        Applies the softsign function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type. Input arg1 is parameter BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-------------------------------+-----------+------------------------------+----------+
        | Argument | Description                   | Data type | Valid range                  | Required |
        +==========+===============================+===========+==============================+==========+
        | arg0     | Tensor softsign is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("hardsigmoid", &hardsigmoid,
		 py::arg().noconvert(), py::arg("scale") = 1.0f/6.0f, py::arg("shift") = 0.5f, R"doc(
        Applies the hardsigmoid function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-----------------------------------+-----------+------------------------------+----------+
        | Argument | Description                       | Data type | Valid range                  | Required |
        +==========+===================================+===========+==============================+==========+
        | arg0     | Tensor hardsigmoid is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-----------------------------------+-----------+------------------------------+----------+
        | scale    | Scale value (PyTorch default)     | float     | default to 1.0/6.0f          | No       |
        +----------+-----------------------------------+-----------+------------------------------+----------+
        | shift    | Shift value (PyTorch default)     | float     | default to 0.5f              | No       |
        +----------+-----------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("hardswish", &hardswish,
		 py::arg().noconvert(), py::arg("scale") = 1.0f/6.0f, py::arg("shift") = 0.5f, R"doc(
        Applies the hard swish function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------------+-----------+------------------------------+----------+
        | Argument | Description                    | Data type | Valid range                  | Required |
        +==========+================================+===========+==============================+==========+
        | arg0     | Tensor hardswish is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+--------------------------------+-----------+------------------------------+----------+
        | scale    | Scale value (PyTorch default)  | float     | default to 1.0/6.0f          | No       |
        +----------+--------------------------------+-----------+------------------------------+----------+
        | shift    | Shift value (PyTorch default)  | float     | default to 0.5f              | No       |
        +----------+--------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("exp", &exp, R"doc(
        Returns a new tensor with the exponential of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------+-----------+------------------------------+----------+
        | Argument | Description              | Data type | Valid range                  | Required |
        +==========+==========================+===========+==============================+==========+
        | arg0     | Tensor exp is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+--------------------------+-----------+------------------------------+----------+
    )doc");


    m_tensor.def("full_like", &full_like, R"doc(
        Returns a new tensor filled with the scalar value shaped like reference tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------+-----------+------------------------------+----------+
        | Argument | Description              | Data type | Valid range                  | Required |
        +==========+==========================+===========+==============================+==========+
        | arg0     | Reference Tensor         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+--------------------------+-----------+------------------------------+----------+
        | arg1     | Fill value               | float     |                              | Yes      |
        +----------+--------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("zeros_like", &zeros_like, R"doc(
        Returns a new tensor filled with zeros shaped like reference tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------+-----------+------------------------------+----------+
        | Argument | Description              | Data type | Valid range                  | Required |
        +==========+==========================+===========+==============================+==========+
        | arg0     | Reference Tensor         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+--------------------------+-----------+------------------------------+----------+
    )doc");


    m_tensor.def("ones_like", &ones_like, R"doc(
        Returns a new tensor filled with ones shaped like reference tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------+-----------+------------------------------+----------+
        | Argument | Description              | Data type | Valid range                  | Required |
        +==========+==========================+===========+==============================+==========+
        | arg0     | Reference Tensor         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+--------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("zeros",
        [] (const std::array<uint32_t, 4> shape, Layout layout, Device * device) {
            return zeros(shape, layout, device);
        },
        py::arg("shape"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, R"doc(
        Returns a new tensor filled with zeros in shape specified by input ``shape``.

        Input shape is specified as a list of 4 integer elements

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | shape    | Shape vector               | Vector    | [W, Z, Y, X]                 | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
        | layout   | Tensor layout              | Layout    | default is ROW_MAJOR         | No       |
        +----------+----------------------------+-----------+------------------------------+----------+
        | device   | Device tensor is placed on | Device    | default is None (on host)    | No       |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("ones",
        [] (const std::array<uint32_t, 4> shape, Layout layout, Device * device) {
            return ones(shape, layout, device);
        },
        py::arg("shape"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, R"doc(
        Returns a new tensor filled with ones in shape specified by input ``shape``.

        Input shape is specified as a list of 4 integer elements

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | shape    | Shape vector               | Vector    | [W, Z, Y, X]                 | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
        | layout   | Tensor layout              | Layout    | default is ROW_MAJOR         | No       |
        +----------+----------------------------+-----------+------------------------------+----------+
        | device   | Device tensor is placed on | Device    | default is None (on host)    | No       |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("full",
        [] (const std::array<uint32_t, 4> shape, float value, Layout layout, Device * device) {
            return full(shape, value, layout, device);
        },
        py::arg("shape"), py::arg("fill_value"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, R"doc(
        Returns a new tensor filled with the scalar value in shape specified by input ``shape``.

        Input shape is specified as a list of 4 integer elements

        Output tensor will have BFLOAT16 data type.

        +------------+----------------------------+-----------+------------------------------+----------+
        | Argument   | Description                | Data type | Valid range                  | Required |
        +============+============================+===========+==============================+==========+
        | shape      | Shape vector               | Vector    | [W, Z, Y, X]                 | Yes      |
        +------------+----------------------------+-----------+------------------------------+----------+
        | fill_value | Fill value                 | float     |                              | Yes      |
        +------------+----------------------------+-----------+------------------------------+----------+
        | layout     | Tensor layout              | Layout    | default is ROW_MAJOR         | No       |
        +------------+----------------------------+-----------+------------------------------+----------+
        | device     | Device tensor is placed on | Device    | default is None (on host)    | No       |
        +------------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("arange", &arange,
        py::arg("start"), py::arg("end"), py::arg("step"), py::arg("device") = nullptr, R"doc(
        Returns a new 1D tensor with the incremented values in size specified by inputs ``start``, ``end`` and ``step``.

        Inpute scalars are integers specifying start, end, and step sizes.
        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | start    | Start                      | int       |                              | yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
        | end      | End                        | int       | > Start                      | yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
        | step     | Step                       | int       | > 0                          | yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
        | device   | Device tensor is placed on | Device    | default is None (on host)    | No       |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");


    m_tensor.def("recip", &recip, R"doc(
        Returns a new tensor with the reciprocal of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor recip is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("sqrt", &sqrt, R"doc(
        Returns tensor with the square-root of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor sqrt is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("log", &log, R"doc(
        Returns tensor with the natural logarithm of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor log is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");
    m_tensor.def("log2", &log2, R"doc(
        Returns tensor with the base 2 logarithm of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor log2 is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");
    m_tensor.def("log10", &log10, R"doc(
        Returns tensor with the base 10 logarithm of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor log10 is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");
    m_tensor.def("tanhshrink", &tanhshrink, R"doc(
        Applies tanh on the input tensor "arg0" and subtracted from the input tensor.
            tanhshrink(x) = x - tanh(x)

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------------+-----------+------------------------------+----------+
        | Argument | Description                      | Data type | Valid range                  | Required |
        +==========+==================================+===========+==============================+==========+
        | arg0     | Tensor tanhshrink is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("tanh", &tanh, R"doc(
        Returns tensor with the hyperbolic tangent of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor tanh is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
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

    m_tensor.def("sin", &tt::tt_metal::sin, R"doc(
        Returns tensor with the sine of elements of the input tensor ``arg0`` in range [0,2$\pi$].

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor sine is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("cos", &tt::tt_metal::cos, R"doc(
        Returns tensor with the cosine of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type in range [0,2$\pi$].

        Output tensor will have BFLOAT16 data type.

        +----------+-----------------------------+-----------+------------------------------+----------+
        | Argument | Description                 | Data type | Valid range                  | Required |
        +==========+=============================+===========+==============================+==========+
        | arg0     | Tensor cosine is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("sinh", &tt::tt_metal::sinh, R"doc(
        Returns tensor with the hyperbolic sine of elements of the input tensor ``arg0`` in range [-9,9] with high accuracy.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor sinh is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("cosh", &tt::tt_metal::cosh, R"doc(
        Returns tensor with the hyperbolic cosine of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type in range [-9,9] for high accuracy.

        Output tensor will have BFLOAT16 data type.

        +----------+-----------------------------+-----------+------------------------------+----------+
        | Argument | Description                 | Data type | Valid range                  | Required |
        +==========+=============================+===========+==============================+==========+
        | arg0     | Tensor cosh is applied to   | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("square", &square, R"doc(
        Returns tensor with the square of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-----------------------------+-----------+------------------------------+----------+
        | Argument | Description                 | Data type | Valid range                  | Required |
        +==========+=============================+===========+==============================+==========+
        | arg0     | Tensor square is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("mish", &mish, R"doc(
        Returns tensor with the mish activation of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor mish is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("softplus", &softplus, R"doc(
        Returns tensor with the softplus activation of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-------------------------------+-----------+------------------------------+----------+
        | Argument | Description                   | Data type | Valid range                  | Required |
        +==========+===============================+===========+==============================+==========+
        | arg0     | Tensor softplus is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("sign", &sign, R"doc(
        Returns tensor with the elementwise signum of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor sign is aplied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("abs", &abs, R"doc(
        Returns tensor with elementwise absolute value of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor abs is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("neg", &neg, R"doc(
        Returns tensor with the negate all of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor neg is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("log1p", &log1p, R"doc(
        Returns tensor with the natural log of 1 added to all of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor log1p is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("exp2", &exp2, R"doc(
        Returns a new tensor with the exp2 (2 power) of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor exp2 is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("add1", &add1, R"doc(
        Returns tensor with the addition of one with input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor add1 is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("expm1", &expm1, R"doc(
        Returns a new tensor with the expm1 of the elements of the input tensor ``arg0``.
        expm1 = exp(x) - 1

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor expm1 is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("asin", &asin, R"doc(
        Returns a new tensor with the arcsine of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor asin is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("acos", &acos, R"doc(
        Returns a new tensor with the arccosine of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor acos is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("swish", swish, R"doc(
        Returns tensor with the swish all of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor swish is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("silu", &silu, R"doc(
        Returns tensor with the silu all of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor silu is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("eqz", &eqz, R"doc(
        Returns tensor with the result of equal to zero of all of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor eqz is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("nez", &nez, R"doc(
        Returns tensor with the not equal zero of all of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor neq is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("gtz", &gtz, R"doc(
        Returns tensor with the greater than zero of all of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor gtz is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("ltz", &ltz, R"doc(
        Returns tensor with the less than zero of all of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor ltz is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("lez", &lez, R"doc(
        Returns tensor with the less than equal zero of all of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor lez is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("gez", &gez, R"doc(
        Returns tensor with the greater than equal zero of all of the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | Tensor gez is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("leaky_relu", &leaky_relu, R"doc(
        Returns tensor with the leaky relu of all of elements of the input tensor ``arg0`` with negative slope as ``arg1``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | leaky relu operation input| Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
        | arg1     |  slope value              | float     | scalar                       | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("relu_max", &relu_max, R"doc(
        Returns tensor with the relu max of all of elements of the input tensor ``arg0``. This is equivalent
        to relu_max[x] = relu(min(x,max value)). It caps off the input to a max value and a min value of 0.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-------------------------------+-----------+------------------------------+----------+
        | Argument | Description                   | Data type | Valid range                  | Required |
        +==========+===============================+===========+==============================+==========+
        | arg0     | Tensor relu max is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
        | arg1     | max value                     | float     |                              | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("elu", &elu, R"doc(
        Returns tensor with the elu activation of all of elements of the input tensor ``arg0`` and scale
        factor alpha as ``arg1``. ELU(x) = alpha*(exp(x) - 1) if x < 0 else x.

        Input tensor and scale factor alpha must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     |  elu computed for tensor  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
        | arg1     |  alpha value              | float     | scalar                       | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("heaviside", &heaviside, R"doc(
        Returns tensor with the Heaviside step function of all of elements of the input tensor ``arg0`` and value
        factor as ``arg1``. HEAVISIDE(x) = 0 if x < 0 , 1 if x > 0 , else value.

        Input tensor and value (used when input x is zero) must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------------+-----------+------------------------------+----------+
        | Argument | Description                    | Data type | Valid range                  | Required |
        +==========+================================+===========+==============================+==========+
        | arg0     | heaviside computed for tensor  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+--------------------------------+-----------+------------------------------+----------+
        | arg1     | value                          | float     | scalar                       | Yes      |
        +----------+--------------------------------+-----------+------------------------------+----------+
        )doc");

    m_tensor.def("log_sigmoid", &log_sigmoid, R"doc(
        Applies the logsigmoid function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------------+-----------+------------------------------+--------------+
        | Argument | Description                      | Data type | Valid range                  | Required     |
        +==========+==================================+===========+==============================+==============+
        | arg0     | Tensor logsigmoid is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes          |
        +----------+----------------------------------+-----------+------------------------------+--------------+
    )doc");

    m_tensor.def("signbit", &signbit, R"doc(
        Applies the signbit function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------------+-----------+------------------------------+--------------+
        | Argument | Description                      | Data type | Valid range                  | Required     |
        +==========+==================================+===========+==============================+==============+
        | arg0     | Tensor signbit is applied to     | Tensor    | Tensor of shape [W, Z, Y, X] | Yes          |
        +----------+----------------------------------+-----------+------------------------------+--------------+
    )doc");

    m_tensor.def("relu_min", &relu_min, R"doc(
        Returns tensor with the relu min of all of elements of the input tensor ``arg0``. This is equivalent
        to relu_min[x] = max(x,minvalue). It moves relu function down to carry out operation at minvalue
        instead of the standard 0.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-------------------------------+-----------+------------------------------+----------+
        | Argument | Description                   | Data type | Valid range                  | Required |
        +==========+===============================+===========+==============================+==========+
        | arg0     | Tensor relu min is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
        | arg1     | min value                     | float     |                              | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("power", &power, R"doc(
        Returns tensor with the all of elements of the input tensor ``arg0`` raised to ``arg1``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+--------------+------------------------------+----------+
        | Argument | Description                | Data type    | Valid range                  | Required |
        +==========+============================+==============+==============================+==========+
        | arg0     | Tensor power is applied to | Tensor       | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+--------------+------------------------------+----------+
        | arg1     | exponent value             | int          | >=0                          | Yes      |
        +----------+----------------------------+--------------+------------------------------+----------+
    )doc");

    m_tensor.def("mac", &mac, R"doc(
        Returns tensor with the multiply and accumulation of all of elements of the input tensors ``arg0, arg1, arg2``.
        Output is ```arg0 x arg1 + arg2``` elementwise operator.
        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | tensor 1                  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
        | arg2     | tensor 2                  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
        | arg3     | tensor 3                  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("polyval", &polyval, R"doc(
        Returns tensor with the polyval of all of elements of the input tensor ``arg0`` with coefficients ``arg1``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+------------------------------+-----------+------------------------------+----------+
        | Argument | Description                  | Data type | Valid range                  | Required |
        +==========+==============================+===========+==============================+==========+
        | arg0     | Tensor polyval is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+------------------------------+-----------+------------------------------+----------+
        | arg1     | coefficients value           | List of   | List size > 0                | Yes      |
        |          | with highest degree first    | float     |                              |          |
        +----------+------------------------------+-----------+------------------------------+----------+
    )doc");

   m_tensor.def("deg2rad", &deg2rad, R"doc(
        Returns tensor with the deg2rad conversion of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+------------------------------+-----------+------------------------------+----------+
        | Argument | Description                  | Data type | Valid range                  | Required |
        +==========+==============================+===========+==============================+==========+
        | arg0     | Tensor deg2rad is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("rad2deg", &rad2deg, R"doc(
        Returns tensor with the rad2deg conversion of elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+------------------------------+-----------+------------------------------+----------+
        | Argument | Description                  | Data type | Valid range                  | Required |
        +==========+==============================+===========+==============================+==========+
        | arg0     | Tensor rad2deg is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("relu6", &relu6, R"doc(
        Returns tensor with the relu6 activation on elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor relu6 is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+----------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("hypot", &hypot, R"doc(
        Returns tensor with the hypot activation on elements of the input tensors ``arg0`` and ``arg1``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+-------------------------------+-----------+------------------------------+----------+
        | Argument | Description                   | Data type | Valid range                  | Required |
        +==========+===============================+===========+==============================+==========+
        | arg0     | first tensor for hypotenuse   |           |                              |          |
        |          | operation                     | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
        | arg1     | second tensor for hypotenuse  |           |                              |          |
        |          | operation                     | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+-------------------------------+-----------+------------------------------+----------+
    )doc");

     m_tensor.def("threshold", &threshold, R"doc(
        Returns tensor with the threshold activation on elements of the input tensors ``arg0`` at threshold `t`,
        and value 'v'.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +-----------+--------------------------------+-----------+------------------------------+----------+
        | Argument  | Description                    | Data type | Valid range                  | Required |
        +===========+================================+===========+==============================+==========+
        | arg0      | Tensor threshold is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +-----------+--------------------------------+-----------+------------------------------+----------+
        | arg1      | Value to theshold at           | float     |                              | Yes      |
        +-----------+--------------------------------+-----------+------------------------------+----------+
        | arg2      | Value to replace with          | float     |                              | Yes      |
        +-----------+--------------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("cbrt", &cbrt, R"doc(
        Returns tensor with the cbrt activation on elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------------+-----------+------------------------------+----------+
        | Argument | Description                    | Data type | Valid range                  | Required |
        +==========+================================+===========+==============================+==========+
        | arg0     | Tensor cube-root is applied to | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
        +----------+--------------------------------+-----------+------------------------------+----------+
    )doc");

    // *** matrix multiplication ***
    m_tensor.def("matmul", &matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a non-batched matrix multiplication ``arg0 x arg1`` with two tensors.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +------------+------------------------------------+--------------+--------------------------------+----------+
        | Argument   | Description                        | Data type    | Valid range                    | Required |
        +============+====================================+==============+================================+==========+
        | arg0       | First tensor to multiply           | Tensor       | Tensor of shape [1, 1, Y, S]   | Yes      |
        +------------+------------------------------------+--------------+--------------------------------+----------+
        | arg1       | Second tensor to multiply          | Tensor       | Tensor of shape [1, 1, S, X]   | Yes      |
        +------------+------------------------------------+--------------+--------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator | MemoryConfig | Default is interleaved in DRAM | No       |
        |            | device memory banks                |              |                                |          |
        +------------+------------------------------------+--------------+--------------------------------+----------+
    )doc");

    m_tensor.def("outer", &outer, R"doc(
        Perform a non-batched outer product multiplication ``arg0 x arg1`` with two tensors.

        Both input tensors must have BFLOAT16 data type but shape [1,1,N,1] and [1,1,1,M] respectively
        or reshapeable with only one major dimension while other 3 being squeezable dimensions.

        Output tensor will have BFLOAT16 data type but of shape [1,1,N,M].

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | First tensor to multiply  | Tensor    | Tensor of shape [1, 1, N, 1] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to multiply | Tensor    | Tensor of shape [1, 1, 1, M] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+

    )doc");

    m_tensor.def("bmm", &bmm,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a batched matmul ``arg0 x arg1`` with two tensors, where batch dims match.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | First tensor to multiply  | Tensor    | Tensor of shape [1, Z, Y, S] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to multiply | Tensor    | Tensor of shape [1, Z, S, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    // *** tensor manipulation ***
    m_tensor.def("reshape", &reshape, R"doc(
        Returns a tensor with the new shape of ``[W, Z, Y, X]``. The X dimension of input and output tensor must have same size.

        Input tensor must be on host device, in TILE layout, and have BFLOAT16 data type.

        Output tensor will be on host device, in TILE layout, and have BFLOAT16 data type.

        +----------+--------------------------------+------------+--------------------------------------------------------+----------+
        | Argument | Description                    | Data type  | Valid range                                            | Required |
        +==========+================================+============+========================================================+==========+
        | arg0     | Input tensor                   | Tensor     | Tensor of shape [W, Z, Y, X], where Y%32=0 and X%32=0  | Yes      |
        +----------+--------------------------------+------------+--------------------------------------------------------+----------+
        | arg1     | W dim of output tensor         | int        |                                                        | Yes      |
        +----------+--------------------------------+------------+--------------------------------------------------------+----------+
        | arg2     | Z dim of output tensor         | int        |                                                        | Yes      |
        +----------+--------------------------------+------------+--------------------------------------------------------+----------+
        | arg3     | Y dim of output tensor         | int        | Y%32=0                                                 | Yes      |
        +----------+--------------------------------+------------+--------------------------------------------------------+----------+
        | arg4     | X dim of output tensor         | int        | X%32=0                                                 | Yes      |
        +----------+--------------------------------+------------+--------------------------------------------------------+----------+
    )doc");

    m_tensor.def("sum", py::overload_cast<const Tensor&,uint>(&sum), R"doc(
        Returns a tensor that is a sum  of input tensor with shape ``[W, Z, Y, X]`` along dimensions ``arg1``.

        Input tensor must have BFLOAT16 data type. Second and third input specify the dimensions of tensor to be transposed.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------------+------------+-------------------------------+----------+
        | Argument | Description                    | Data type  | Valid range                   | Required |
        +==========+================================+============+===============================+==========+
        | arg0     | Input tensor                   | Tensor     | Tensor of shape [W, Z, Y, X]  | Yes      |
        +----------+--------------------------------+------------+-------------------------------+----------+
        | arg1     | dimension to sum along         | uint       | 0, 1, 2, or 3                 | Yes      |
        +----------+--------------------------------+------------+-------------------------------+----------+
    )doc");

    m_tensor.def("transpose", py::overload_cast<const Tensor&,uint,uint>(&transpose), R"doc(
        Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``arg1`` and ``arg2`` are swapped.

        Input tensor must have BFLOAT16 data type. Second and third input specify the dimensions of tensor to be transposed.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------------+------------+-------------------------------+----------+
        | Argument | Description                    | Data type  | Valid range                   | Required |
        +==========+================================+============+===============================+==========+
        | arg0     | Input tensor                   | Tensor     | Tensor of shape [W, Z, Y, X]  | Yes      |
        +----------+--------------------------------+------------+-------------------------------+----------+
        | arg1     | dimension to transpose         | uint       | 0, 1, 2, or 3                 | Yes      |
        +----------+--------------------------------+------------+-------------------------------+----------+
        | arg2     | dimension to transpose         | uint       | 0, 1, 2, or 3                 | Yes      |
        +----------+--------------------------------+------------+-------------------------------+----------+
    )doc");

    m_tensor.def("transpose", py::overload_cast<const Tensor&>(&transpose), R"doc(
        Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``X`` and ``Y`` are swapped.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------------+------------+-------------------------------+----------+
        | Argument | Description                    | Data type  | Valid range                   | Required |
        +==========+================================+============+===============================+==========+
        | arg0     | Input tensor                   | Tensor     | Tensor of shape [W, Z, Y, X]  | Yes      |
        +----------+--------------------------------+------------+-------------------------------+----------+
    )doc");

    m_tensor.def("transpose_hc", &transpose_hc, R"doc(
        Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``Y`` and ``Z`` are swapped.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------------+------------+-------------------------------+----------+
        | Argument | Description                    | Data type  | Valid range                   | Required |
        +==========+================================+============+===============================+==========+
        | arg0     | Input tensor                   | Tensor     | Tensor of shape [W, Z, Y, X]  | Yes      |
        +----------+--------------------------------+------------+-------------------------------+----------+
    )doc");

    m_tensor.def("transpose_cn", &transpose_cn, R"doc(
        Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``Z`` and ``W`` are swapped.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+--------------------------------+------------+-------------------------------+----------+
        | Argument | Description                    | Data type  | Valid range                   | Required |
        +==========+================================+============+===============================+==========+
        | arg0     | Input tensor                   | Tensor     | Tensor of shape [W, Z, Y, X]  | Yes      |
        +----------+--------------------------------+------------+-------------------------------+----------+
    )doc");

    m_tensor.def("permute", &permute, R"doc(
        Returns a tensor that is input tensor ``arg0`` with its dimensions permuted to new order ``[arg1, arg2, arg3, arg4]``.

        +----------+----------------------+-----------+------------------------------------+----------+
        | Argument | Description          | Data type | Valid range                        | Required |
        +==========+======================+===========+====================================+==========+
        | arg0     | Input tensor         | Tensor    |                                    | Yes      |
        +----------+----------------------+-----------+------------------------------------+----------+
        | arg1     | Dim to become W      | int       | Unique value between [0, num dims) | Yes      |
        +----------+----------------------+-----------+------------------------------------+----------+
        | arg2     | Dim to become Z      | int       | Unique value between [0, num dims) | Yes      |
        +----------+----------------------+-----------+------------------------------------+----------+
        | arg3     | Dim to become Y      | int       | Unique value between [0, num dims) | Yes      |
        +----------+----------------------+-----------+------------------------------------+----------+
        | arg4     | Dim to become X      | int       | Unique value between [0, num dims) | Yes      |
        +----------+----------------------+-----------+------------------------------------+----------+
    )doc");

    m_tensor.def("tilize", &tilize,
        py::arg("input").noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Changes data layout of input tensor to TILE.

        Input tensor must be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

        +------------+------------------------------------+--------------+-----------------------------------------------------------------+----------+
        | Argument   | Description                        | Data type    | Valid range                                                     | Required |
        +============+====================================+==============+=================================================================+==========+
        | input      | Input tensor                       | Tensor       | Tensor of shape [W, Z, Y, X], where Y%32=0 and X%32=0           | Yes      |
        +------------+------------------------------------+--------------+-----------------------------------------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator | MemoryConfig | Default is interleaved in DRAM                                  | No       |
        |            | device memory banks                |              |                                                                 |          |
        +------------+------------------------------------+--------------+-----------------------------------------------------------------+----------+
    )doc");

    m_tensor.def("tilize_with_zero_padding", &tilize_with_zero_padding,
        py::arg("input").noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Tilizes a given tensor across memory on device. Pads zeroes height-wise and width-wise if required.

        +------------+------------------------------------+--------------+--------------------------------+----------+
        | Argument   | Description                        | Data type    | Valid range                    | Required |
        +============+====================================+==============+================================+==========+
        | input      | Input tensor                       | Tensor       |                                | Yes      |
        +------------+------------------------------------+--------------+--------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator | MemoryConfig | Default is interleaved in DRAM | No       |
        |            | device memory banks                |              |                                |          |
        +------------+------------------------------------+--------------+--------------------------------+----------+
    )doc");

    m_tensor.def("tilize_with_val_padding",
        [] (const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value, const MemoryConfig& mem_config) {
            return tilize_with_val_padding(tensor, output_tensor_shape, input_tensor_start, pad_value, mem_config);
        },
        py::arg("input").noconvert(), py::arg("output_tensor_shape").noconvert(), py::arg("input_tensor_start"), py::arg("pad_value"), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Tilizes a given tensor across memory on device. Pads to specified shape before tilizing.

        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | Argument            | Description                                          | Data type    | Valid range                    | Required |
        +=====================+======================================================+==============+================================+==========+
        | input               | Input tensor                                         | Tensor       |                                | Yes      |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | output_tensor_shape | Shape of output tensor                               | List[int[4]] |                                | Yes      |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | input_tensor_start  | Start indices to place input tensor in output tensor | List[int[4]] | Must be all 0s                 | Yes      |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | pad_value           | Value to pad input tensor                            | float        |                                | Yes      |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | mem_config          | Layout of tensor in TT Accelerator                   | MemoryConfig | Default is interleaved in DRAM | No       |
        |                     | device memory banks                                  |              |                                |          |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
    )doc");

    m_tensor.def("untilize", &untilize,
        py::arg("input").noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Changes data layout of input tensor to ROW_MAJOR.

        Input tensor must be on TT accelerator device, in TILE, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        +------------+------------------------------------+--------------+-----------------------------------------------------------------+----------+
        | Argument   | Description                        | Data type    | Valid range                                                     | Required |
        +============+====================================+==============+=================================================================+==========+
        | input      | Input tensor                       | Tensor       | Tensor of shape [W, Z, Y, X], where Y%32=0 and X%32=0           | Yes      |
        +------------+------------------------------------+--------------+-----------------------------------------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator | MemoryConfig | Default is interleaved in DRAM                                  | No       |
        |            | device memory banks                |              |                                                                 |          |
        +------------+------------------------------------+--------------+-----------------------------------------------------------------+----------+
    )doc");

    m_tensor.def("untilize_with_unpadding",
        [] (const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, const MemoryConfig& mem_config) {
            return untilize_with_unpadding(tensor, output_tensor_shape, input_tensor_start, mem_config);
        },
        py::arg("input").noconvert(), py::arg("output_tensor_start").noconvert(), py::arg("output_tensor_end"), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Changes data layout of input tensor to ROW_MAJOR and unpads/removes elements from the tensor.

        Input tensor must be on TT accelerator device, in TILE, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
        | Argument            | Description                                  | Data type    | Valid range                    | Required |
        +=====================+==============================================+==============+================================+==========+
        | input               | Input tensor                                 | Tensor       |                                | Yes      |
        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
        | output_tensor_start | Start indices of input tensor                | List[int[4]] | Must be all 0s                 | Yes      |
        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
        | output_tensor_end   | End indices of input tensor in output tensor | List[int[4]] | Values along each dim must be  | Yes      |
        |                     |                                              |              |                                |          |
        |                     |                                              |              | < input_tensor_shape[i]        |          |
        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
        | mem_config          | Layout of tensor in TT Accelerator           | MemoryConfig | Default is interleaved in DRAM | No       |
        |                     | device memory banks                          |              |                                |          |
        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
    )doc");

    m_tensor.def("pad",
        [] (const Tensor &input_tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value, const MemoryConfig& mem_config) {
            return pad(input_tensor, output_tensor_shape, input_tensor_start, pad_value, mem_config);
        },
        py::arg("input").noconvert(), py::arg("output_tensor_shape").noconvert(), py::arg("input_tensor_start"), py::arg("pad_value"), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Pad TT Tensor with given pad value ``arg2``.

        The input tensor must be in ROW_MAJOR or TILE layout.

        Returns an output tensor that contains the input tensor at the given input tensor start indices ``arg3`` and the padded value everywhere else.

        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | Argument            | Description                                          | Data type    | Valid range                    | Required |
        +=====================+======================================================+==============+================================+==========+
        | input               | Input tensor                                         | Tensor       |                                | Yes      |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | output_tensor_shape | Shape of output tensor                               | List[int[4]] |                                | Yes      |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | input_tensor_start  | Start indices to place input tensor in output tensor | List[int[4]] | Must be all 0s                 | Yes      |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | pad_value           | Value to pad input tensor                            | float        |                                | Yes      |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
        | mem_config          | Layout of tensor in TT Accelerator                   | MemoryConfig | Default is interleaved in DRAM | No       |
        |                     | device memory banks                                  |              |                                |          |
        +---------------------+------------------------------------------------------+--------------+--------------------------------+----------+
    )doc");

    m_tensor.def("unpad",
        [] (const Tensor &input_tensor, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end, const MemoryConfig& mem_config) {
            return unpad(input_tensor, output_tensor_start, output_tensor_end, mem_config);
        },
        py::arg("input").noconvert(), py::arg("output_tensor_start").noconvert(), py::arg("output_tensor_end"), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Unpad TT Tensor.

        Returns an output tensor from output tensor start indices ``arg1`` to output tensor end indices ``arg2`` (inclusive) of the input tensor.

        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
        | Argument            | Description                                  | Data type    | Valid range                    | Required |
        +=====================+==============================================+==============+================================+==========+
        | input               | Input tensor                                 | Tensor       |                                | Yes      |
        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
        | output_tensor_start | Start indices of input tensor                | List[int[4]] | Must be all 0s                 | Yes      |
        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
        | output_tensor_end   | End indices of input tensor in output tensor | List[int[4]] | Values along each dim must be  | Yes      |
        |                     |                                              |              |                                |          |
        |                     |                                              |              | < input_tensor_shape[i]        |          |
        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
        | mem_config          | Layout of tensor in TT Accelerator           | MemoryConfig | Default is interleaved in DRAM | No       |
        |                     | device memory banks                          |              |                                |          |
        +---------------------+----------------------------------------------+--------------+--------------------------------+----------+
    )doc");

    // *** broadcast and reduce ***
    m_tensor.def("bcast", &bcast,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("math_op"), py::arg("dim"), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a binary elementwise operation ``arg2`` between tensors ``arg0`` and ``arg1``, where values from tensor ``arg1`` are broadcast.

        Let tensor ``arg0`` have shape ``[W0, Z0, Y0, X0]`` and tensor ``arg1`` shape ``[W1, Z1, Y1, X1]``. ``arg3`` determines the type of broadcast performed.

        For ``arg3=BcastOpDim::W`` broadcast is performed on dimension ``X``. ``Y0`` and ``Y1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``arg3=BcastOpDim::H`` broadcast is performed on dimension  ``Y``. ``X0`` and ``X1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``arg3=BcastOpDim::HW`` broadcast is performed on dimensions ``X`` and ``Y``. Either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1) must hold for input shapes.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | Argument   | Description                        | Data type    | Valid range                                                 | Required |
        +============+====================================+==============+=============================================================+==========+
        | arg0       | Input tensor                       | Tensor       | Tensor of shape [W0, Z0, Y0, X0], where Y0%32=0 and X0%32=0 | Yes      |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | arg1       | Input tensor                       | Tensor       | Tensor of shape [W1, Z1, Y1, X1], where Y1%32=0 and X1%32=0 | Yes      |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | arg2       | Math operation to perform          | BcastOpMath  | ADD, SUB, MUL                                               | Yes      |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | arg3       | Dimension on which to broadcast    | BcastOpDim   | W, H, HW                                                    | Yes      |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator | MemoryConfig | Default is interleaved in DRAM                              | No       |
        |            | device memory banks                |              |                                                             |          |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
    )doc");

    m_tensor.def("bcast_without_autoformat", &bcast_without_autoformat,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("math_op"), py::arg("dim"), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a binary elementwise operation ``arg2`` between tensors ``arg0`` and ``arg1``, where values from tensor ``arg1`` are broadcast.

        Let tensor ``arg0`` have shape ``[W0, Z0, Y0, X0]`` and tensor ``arg1`` shape ``[W1, Z1, Y1, X1]``. ``arg3`` determines the type of broadcast performed.

        For ``arg3=BcastOpDim::W`` broadcast is performed on dimension ``X``. ``Y0`` and ``Y1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``arg3=BcastOpDim::H`` broadcast is performed on dimension  ``Y``. ``X0`` and ``X1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``arg3=BcastOpDim::HW`` broadcast is performed on dimensions ``X`` and ``Y``. Either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1) must hold for input shapes.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        Auto formatting is disabled. Input tensors must have TILE layout. Output tensors will have TILE layout.

        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | Argument   | Description                        | Data type    | Valid range                                                 | Required |
        +============+====================================+==============+=============================================================+==========+
        | arg0       | Input tensor                       | Tensor       | Tensor of shape [W0, Z0, Y0, X0], where Y0%32=0 and X0%32=0 | Yes      |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | arg1       | Input tensor                       | Tensor       | Tensor of shape [W1, Z1, Y1, X1], where Y1%32=0 and X1%32=0 | Yes      |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | arg2       | Math operation to perform          | BcastOpMath  | ADD, SUB, MUL                                               | Yes      |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | arg3       | Dimension on which to broadcast    | BcastOpDim   | W, H, HW                                                    | Yes      |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator | MemoryConfig | Default is interleaved in DRAM                              | No       |
        |            | device memory banks                |              |                                                             |          |
        +------------+------------------------------------+--------------+-------------------------------------------------------------+----------+
    )doc");

    m_tensor.def("reduce", &reduce, R"doc(
        Perform a reduction of input tensor ``arg0`` using mathematical operation ``arg1`` on dimension ``arg2``.

        For ``arg2=ReduceOpDim::W`` reduce is done on dimension X.

        For ``arg2=ReduceOpDim::H`` reduce is done on dimension Y.

        For ``arg2=ReduceOpDim::HW`` reduce is done on dimensions X and Y.

        Input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------------------------------------+---------------+-------------------------------------------------------+----------+
        | Argument | Description                                             | Data type     | Valid range                                           | Required |
        +==========+=========================================================+===============+=======================================================+==========+
        | arg0     | Input tensor                                            | Tensor        | Tensor of shape [W, Z, Y, X], where Y%32=0 and X%32=0 | Yes      |
        +----------+---------------------------------------------------------+---------------+-------------------------------------------------------+----------+
        | arg1     | Aggregating math operation                              | ReduceOpMath  | SUM, MAX                                              | Yes      |
        +----------+---------------------------------------------------------+---------------+-------------------------------------------------------+----------+
        | arg2     | Dimension on which reduction is performed               | ReduceOpDim   | W, H, HW                                              | Yes      |
        +----------+---------------------------------------------------------+---------------+-------------------------------------------------------+----------+
        | arg3     | Scaling factor applied to each element of output tensor | float         | For HW reduction only value 1.0f is supported         | Yes      |
        +----------+---------------------------------------------------------+---------------+-------------------------------------------------------+----------+
    )doc");

    // *** experimental operations ***
    m_tensor.def("fill_rm", &fill_rm, R"doc(
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
    m_tensor.def("fill_ones_rm", &fill_ones_rm, R"doc(
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

    // Custom BERT TMs
    m_tensor.def("bert_large_create_qkv_heads", &bert_large_create_qkv_heads,
        py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Splits [9, 1, 384, 3072] fused qkv matrix into 3 heads with shapes [9, 16, 384, 64], [9, 16, 64, 384], and [9, 16, 384, 64].
    )doc");
    m_tensor.def("bert_large_split_fused_qkv", &bert_large_split_fused_qkv,
        py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Splits [9, 1, 384, 3072] fused qkv matrix into 3 heads with shape [9, 1, 384, 1024].
    )doc");
    m_tensor.def("bert_large_create_q_head", &bert_large_create_q_head,
        py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Reshuffles [9, 1, 384, 1024] tensor into tensor with shape [9, 16, 384, 64].
    )doc");
    m_tensor.def("bert_large_create_k_head", &bert_large_create_k_head,
        py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Reshuffles [9, 1, 384, 1024] tensor into tensor with shape [9, 16, 64, 384].
    )doc");
    m_tensor.def("bert_large_create_v_head", &bert_large_create_v_head,
        py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Reshuffles [9, 1, 384, 1024] tensor into tensor with shape [9, 16, 384, 64].
    )doc");
    m_tensor.def("bert_large_concat_heads", &bert_large_concat_heads,
        py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Reshuffles [9, 16, 384, 64] tensor into tensor with shape [9, 1, 384, 1024].
    )doc");

    // Custom BERT matmuls/bmms
    m_tensor.def("bert_large_fused_qkv_matmul", &bert_large_fused_qkv_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("mem_config") = MemoryConfig{.interleaved = true}, py::arg("out_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_fused_qkv non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_ff1_matmul", &bert_large_ff1_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("fuse_gelu_activation") = false, py::arg("mem_config") = MemoryConfig{.interleaved = true}, py::arg("out_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_ff1 non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_ff2_matmul", &bert_large_ff2_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("mem_config") = MemoryConfig{.interleaved = true}, py::arg("out_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_ff2 non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_selfout_matmul", &bert_large_selfout_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("mem_config") = MemoryConfig{.interleaved = true}, py::arg("out_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_selfout non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_pre_softmax_bmm", &bert_large_pre_softmax_bmm,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, py::arg("out_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_pre_softmax_bmm batched matmul ``[9, 16, 384, 64] x [9, 16, 64, 384]`` with two tensors and returns a reshaped output of [9, 1, 6144, 384].
    )doc");
    m_tensor.def("bert_large_post_softmax_bmm", &bert_large_post_softmax_bmm,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, py::arg("out_dtype").noconvert() = std::nullopt, R"doc(
        Perform a bert_large_post_softmax_bmm batched matmul by reshaping tensor A to [9, 16, 384, 384] first, then returning ``[9, 16, 384, 384] x [9, 16, 384, 64]``.
    )doc");

    // Custom BERT Layernorm
    m_tensor.def("bert_large_layernorm", &bert_large_layernorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        "Performs a bert_large_layernorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
    )doc");
    m_tensor.def("bert_large_add_layernorm", &bert_large_add_layernorm,
        py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        "Performs a bert_large_layernorm(a+b)*gamma + beta operation."
    )doc");

    // softmax
    m_tensor.def("softmax_in_place", &softmax_in_place,
        "Performs a softmax operation on the last tensor dimension. Returns a reference to the input tensor modified in place.");
    m_tensor.def("scale_mask_softmax_in_place", &scale_mask_softmax_in_place,
        "Performs a fused scale->attention_mask->softmax operation. Returns a reference to the input tensor modified in place.");

    // groupnorm
    m_tensor.def("groupnorm", &groupnorm,
        py::arg("input").noconvert(), py::arg("group_size").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        "Performs a groupnorm operation on the channel dimension grouped per group_size, with optional fused with post-multiplication and addition via W-bcast.
    )doc");

    // layernorm
    m_tensor.def("layernorm", &layernorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        "Performs a layernorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
    )doc");
    m_tensor.def("add_layernorm", &add_layernorm,
        py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        "Performs a layernorm(a+b)*gamma + beta operation."
    )doc");
    m_tensor.def("rmsnorm", &rmsnorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        "Performs a rmsnorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
    )doc");

    // FC
    m_tensor.def("fully_connected", &fully_connected,
        py::arg("act").noconvert(), py::arg("weights").noconvert(), py::arg("bias").noconvert() = std::nullopt, R"doc(
        Fully connected layer (linear.)
        +----------+----------------------------+------------+-------------------------------+----------+
        | Argument | Description                | Data type  | Valid range                   | Required |
        +==========+============================+============+===============================+==========+
        | act      | Input activations tensor   | Tensor     |                               | Yes      |
        | weights  | Input weights tensor       | Tensor     |                               | Yes      |
        | bias     | Input bias tensor          | Tensor     |                               | No       |
        +----------+----------------------------+------------+-------------------------------+----------+
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

    // TMs
    m_tensor.def("split_last_dim_two_chunks_tiled", &split_last_dim_two_chunks_tiled, py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Splits a tensor's last dimension in two equal sized chunks. This assumes the last dim is tiled.

        +----------+--------------------------------+------------+-------------------------------+----------+
        | Argument | Description                    | Data type  | Valid range                   | Required |
        +==========+================================+============+===============================+==========+
        | arg0     | Input tensor                   | Tensor     | Tensor of shape [W, Z, Y, X]  | Yes      |
        +----------+--------------------------------+------------+-------------------------------+----------+

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
        [] (const Tensor &input, Device * device, const std::array<uint32_t, 4>& padded_shape, float pad_value, Layout target_layout) {
            return AutoFormat::format_input_tensor(input, device, padded_shape, pad_value, target_layout);
        }, R"doc(
            Formats tensor to target layout and pads to padded shape
        )doc"
    );
    m_tensor.def(
        "format_output_tensor",
        [] (const Tensor &output, const std::array<uint32_t, 4>& shape, Device* device, Layout target_layout) {
            return AutoFormat::format_output_tensor(output, shape, device, target_layout);
        }, R"doc(
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
    m_device.def("StartDebugPrintServerOnCores", &StartDebugPrintServerOnCores);

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

    m_device.def("Synchronize", &Synchronize, R"doc(
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
                                        uint32_t in0_block_w,
                                        uint32_t in0_block_h,
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
}
