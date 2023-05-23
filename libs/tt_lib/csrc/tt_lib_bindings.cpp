#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/conv/conv_op.hpp"
#include "tt_dnn/op_library/pad_h_rm/pad_h_rm_op.hpp"
#include "tt_dnn/op_library/fill_rm/fill_rm_op.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_dnn/op_library/transpose_rm/transpose_rm_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"
#include "tensor/tensor_utils.hpp"

#include "tt_lib_bindings.hpp"
#include "type_caster.hpp"

namespace py = pybind11;

namespace tt {

namespace tt_metal {

extern void EnableCompileCache();
extern int  DisableCompileCache();
extern bool GetCompileCacheEnabled();

void TensorModule(py::module &m_tensor) {
    // ENUM SECTION

    // bast enums
    py::enum_<BcastOpMath::Enum>(m_tensor, "BcastOpMath")
        .value("ADD", BcastOpMath::Enum::ADD)
        .value("SUB", BcastOpMath::Enum::SUB)
        .value("MUL", BcastOpMath::Enum::MUL);

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
        .value("TILE", Layout::TILE)
        .value("CHANNELS_LAST", Layout::CHANNELS_LAST);

    // TODO(agrebenisan): This should probably be in its own module, but here for now.
    py::enum_<Initialize>(m_tensor, "Initialize")
        .value("ZEROS", Initialize::ZEROS)
        .value("ONES",Initialize::ONES)
        .value("INCREMENT", Initialize::INCREMENT)
        .value("RANDOM", Initialize::RANDOM);

    py::enum_<DataType>(m_tensor, "DataType")
        .value("FLOAT32", DataType::FLOAT32)
        .value("BFLOAT16", DataType::BFLOAT16)
        .value("UINT32", DataType::UINT32)
        .value("BFLOAT8_B", DataType::BFLOAT8_B);

    py::enum_<BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1);

    auto pyMemoryConfig = py::class_<MemoryConfig>(m_tensor, "MemoryConfig", R"doc(
        Class defining memory configuration for storing tensor data on TT Accelerator device.
        There are eight DRAM memory banks on TT Accelerator device, indexed as 0, 1, 2, ..., 7.
    )doc");

    pyMemoryConfig
        .def(
            py::init<>(
                [](bool interleaved, int bank_id, BufferType buffer_type) {
                    return MemoryConfig{.interleaved=interleaved, bank_id=bank_id, buffer_type=buffer_type};
                }
            ),
            py::arg("interleaved") = true,
            py::arg("bank_id") = -1,
            py::arg("buffer_type") = BufferType::DRAM, R"doc(
                Create MemoryConfig class.
                If interleaved is set to True, tensor data will be interleaved across multiple DRAM banks on TT Accelerator device.
                Otherwise, tensor data will be stored in a DRAM bank selected by dram_channel (valid values are 0, 1, ..., 7).

                Example of creating MemoryConfig specifying that tensor data should be stored in DRAM bank 3.

                .. code-block:: python

                    mem_config = tt_lib.tensor.MemoryConfig(False, 3)
            )doc"
        )
        .def_readonly("interleaved", &MemoryConfig::interleaved, "Whether tensor data is interleaved across mulitple DRAM channels")
        .def_readonly("bank_id", &MemoryConfig::bank_id, "DRAM channel holding tensor data. Only used when tensor is not interleaved")
        .def_readonly("buffer_type", &MemoryConfig::buffer_type, "Buffer type to store tensor data. Can be DRAM or L1");

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
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | tt_lib.tensor.Layout.CHANNELS_LAST |          |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | device     | Device on which tensor will be created                 | tt_lib.device.Device      | Host or TT accelerator device      | No       |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator device memory banks | tt_lib.tensor.MemoryConfig|                                    | No       |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+

    )doc");

    pyTensor
        .def(
            py::init<>(
                [](std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout) {
                    return Tensor(data, shape, data_type, layout);
                }
            ), R"doc(
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
                [](std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device) {
                    return Tensor(data, shape, data_type, layout, device);
                }
            ), py::keep_alive<1, 6>(), R"doc(
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
                [](std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device, const MemoryConfig &mem_config) {
                    return Tensor(data, shape, data_type, layout, device, mem_config);
                }
            ), py::keep_alive<1, 6>(), R"doc(
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
                    mem_config = tt_lib.tensor.MemoryConfig(False, 3)
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
                [](std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout) {
                    return Tensor(data, shape, data_type, layout);
                }
            ), R"doc(
                Not supported.
            )doc"
        )
        .def(
            py::init<>(
                [](std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device) {
                    return Tensor(data, shape, data_type, layout, device);
                }
            ), py::keep_alive<1, 6>(), R"doc(
                Not supported.
            )doc"
        )
        .def(
            py::init<>(
                [](std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device, const MemoryConfig &mem_config) {
                    return Tensor(data, shape, data_type, layout, device, mem_config);
                }
            ), py::keep_alive<1, 6>(), R"doc(
                Not supported.
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
        .def("to", py::overload_cast<Host*>(&Tensor::to, py::const_), R"doc(
            Move TT Tensor from TT accelerator device to host device.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | Device to which tensor will be moved            | tt_lib.device.Host         | Host device           | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(host)
        )doc")
        .def("to", py::overload_cast<Layout>(&Tensor::to, py::const_), R"doc(
            Convert TT Tensor to provided memory layout. Available layouts conversions are:

            * ROW_MAJOR to TILE or CHANNELS_LAST
            * TILE to ROW_MAJOR
            * CHANNELS_LAST to ROW_MAJOR

            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range                    | Required |
            +===========+=================================================+============================+================================+==========+
            | arg0      | Target memory layout                            | tt_lib.tensor.Layout       | ROW_MAJOR, TILE, CHANNELS_LAST | Yes      |
            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_lib.tensor.Layout.TILE)
        )doc")
        .def("pad", [](const Tensor &self, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {
            return self.pad(output_tensor_shape, input_tensor_start, pad_value);
        }, R"doc(
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
            return self.shape();
        }, R"doc(
            Get the shape of the tensor as list of integers.

            .. code-block:: python

                shape = tt_tensor.shape()

        )doc")
        .def("on_host", [](const Tensor &self) {
            return self.on_host();
        }, R"doc(
            Check if the tensor is on host

            .. code-block:: python

                on_host = tt_tensor.on_host()

        )doc")
        .def("device", [](const Tensor &self) {
            return self.device();
        }, R"doc(
            Get the device of the tensor.

            .. code-block:: python

                device = tt_tensor.device()

        )doc")
        .def("data", [](const Tensor &self) {
            std::vector<uint32_t> empty_vec;
            TT_ASSERT(self.data_ptr() != nullptr);
            switch (self.dtype()) {
                case DataType::BFLOAT16: {
                    return py::cast(*reinterpret_cast<std::vector<bfloat16>*>(self.data_ptr()));
                }
                break;
                case DataType::FLOAT32:
                    return py::cast(*reinterpret_cast<std::vector<float>*>(self.data_ptr()));
                break;
                case DataType::UINT32:
                    return py::cast(*reinterpret_cast<std::vector<uint32_t>*>(self.data_ptr()));
                break;
                case DataType::BFLOAT8_B:
                    return py::cast(*reinterpret_cast<std::vector<float>*>(self.data_ptr()));
                break;
                default:
                    TT_ASSERT(false && "Unsupported data type!");
                break;
            }
            return py::cast(empty_vec);
        }, R"doc(
            Get data in the tensor as a list of numbers.

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.to(host).data() # move TT Tensor to host and get values stored in it

        )doc")
        .def("layout", [](const Tensor &self) {
            return self.layout();
        }, R"doc(
            Get memory layout of TT Tensor.

            .. code-block:: python

                layout = tt_tensor.layout()

        )doc")
        .def("buffer_type", [](const Tensor &self) {
            return self.buffer_type();
        }, R"doc(
            Get buffer type of TT Tensor.

            .. code-block:: python

                buffer_type = tt_tensor.buffer_type()

        )doc")
        .def("dtype", [](const Tensor &self) {
            return self.dtype();
        }, R"doc(
            Get dtype of TT Tensor.

            .. code-block:: python

                dtype = tt_tensor.dtype()

        )doc");;

    // *** eltwise binary ***
    m_tensor.def("add", &add, R"doc(
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

    m_tensor.def("sub", &sub, R"doc(
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

    m_tensor.def("mul", &mul, R"doc(
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

    // *** eltwise unary ***
    m_tensor.def("gelu", &gelu, R"doc(
        Applies the Gaussian Error Linear Units (GELU) function to the elements of the input tensor ``arg0``.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+----------------------------+-----------+------------------------------+----------+
        | Argument | Description                | Data type | Valid range                  | Required |
        +==========+============================+===========+==============================+==========+
        | arg0     | Tensor GELU is applied to  | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
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

    // *** matrix multiplication ***
    m_tensor.def("matmul", &matmul, R"doc(
        Perform a non-batched matrix multiplication ``arg0 x arg1`` with two tensors.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +----------+---------------------------+-----------+------------------------------+----------+
        | Argument | Description               | Data type | Valid range                  | Required |
        +==========+===========================+===========+==============================+==========+
        | arg0     | First tensor to multiply  | Tensor    | Tensor of shape [1, 1, Y, S] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
        | arg1     | Second tensor to multiply | Tensor    | Tensor of shape [1, 1, S, X] | Yes      |
        +----------+---------------------------+-----------+------------------------------+----------+
    )doc");

    m_tensor.def("bmm", &bmm, R"doc(
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
        Changes shape of tensor ``arg0`` to new shape ``[W, Z, Y, X]``. The X dimension of input and output tensor must have same size.

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

    m_tensor.def("transpose", &transpose, R"doc(
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

    // *** broadcast and reduce ***
    m_tensor.def("bcast", &bcast, R"doc(
        Perform a binary elementwise operation ``arg2`` between tensors ``arg0`` and ``arg1``, where values from tensor ``arg1`` are broadcast.

        Let tensor ``arg0`` have shape ``[W0, Z0, Y0, X0]`` and tensor ``arg1`` shape ``[W1, Z1, Y1, X1]``. ``arg3`` determines the type of broadcast performed.

        For ``arg3=BcastOpDim::W`` broadcast is performed on dimension ``X``. ``Y0`` and ``Y1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``arg3=BcastOpDim::H`` broadcast is performed on dimension  ``Y``. ``X0`` and ``X1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``arg3=BcastOpDim::HW`` broadcast is performed on dimensions ``X`` and ``Y``. Either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1) must hold for input shapes.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        +-----------+---------------------------------+--------------+-------------------------------------------------------------+----------+
        | Argument  | Description                     | Data type    | Valid range                                                 | Required |
        +===========+=================================+==============+=============================================================+==========+
        | arg0      | Input tensor                    | Tensor       | Tensor of shape [W0, Z0, Y0, X0], where Y0%32=0 and X0%32=0 | Yes      |
        +-----------+---------------------------------+--------------+-------------------------------------------------------------+----------+
        | arg1      | Input tensor                    | Tensor       | Tensor of shape [W1, Z1, Y1, X1], where Y1%32=0 and X1%32=0 | Yes      |
        +-----------+---------------------------------+--------------+-------------------------------------------------------------+----------+
        | arg2      | Math operation to perform       | BcastOpMath  | ADD, SUB, MUL                                               | Yes      |
        +-----------+---------------------------------+--------------+-------------------------------------------------------------+----------+
        | arg3      | Dimension on which to broadcast | BcastOpDim   | W, H, HW                                                    | Yes      |
        +-----------+---------------------------------+--------------+-------------------------------------------------------------+----------+
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
    m_tensor.def("transpose_hc_rm", &transpose_hc_rm, R"doc(
        Transposes a given tensor's H and C dimensions, row-major wise.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");
    m_tensor.def("tilize", &tilize, R"doc(
        Changes data layout of input tensor to TILE.

        Input tensor must be on TT accelerator device, in ROW_MAJOR or CHANNELS_LAST layout, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

        +----------+--------------------------------+------------+---------------------------------------------------------+----------+
        | Argument | Description                    | Data type  | Valid range                                             | Required |
        +==========+================================+============+=========================================================+==========+
        | arg0     | Input tensor                   | Tensor     | Tensor of shape [W, Z, Y, X], where Y%32=0 and X%32=0   | Yes      |
        +----------+--------------------------------+------------+---------------------------------------------------------+----------+
    )doc");
    m_tensor.def("untilize", &untilize, R"doc(
        Changes data layout of input tensor to ROW_MAJOR.

        Input tensor must be on TT accelerator device, in TILE, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        +----------+--------------------------------+------------+-----------------------------------------------------------------+----------+
        | Argument | Description                    | Data type  | Valid range                                                     | Required |
        +==========+================================+============+=================================================================+==========+
        | arg0     | Input tensor                   | Tensor     | Tensor of shape [W, Z, Y, X], where Y%32=0 and X%32=0           | Yes      |
        +----------+--------------------------------+------------+-----------------------------------------------------------------+----------+
    )doc");
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

        val_hi/lo are expected to be uint16 encodings of bfloat16 numbers, so
        0x3f80 for 1.0 etc.

        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | Argument | Description                                                           | Data type             | Valid range            | Required |
        +==========+=======================================================================+=======================+========================+==========+
        | N        | Batch count of output tensor                                          | int                   |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | C        | Channel count of output tensor                                        | int                   |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | H        | Height count of output tensor                                         | int                   |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | W        | Width count of output tensor                                          | int                   |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | hOnes    | Height of high values region                                          | int                   | hOnes <= H             | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | wOnes    | Width of high values region                                           | int                   | wOnes <= W             | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | any      | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | val_hi   | High value to use                                                     | int                   | Valid bfloat16 integer | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | val_lo   | Low value to use                                                      | int                   | Valid bfloat16 integer | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
    )doc");
    m_tensor.def("fill_ones_rm", &fill_ones_rm, R"doc(
        Same as ``fill_rm``, but ``val_hi`` is set to ``1`` and ``val_lo`` is
        ``0``.

        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | Argument | Description                                                           | Data type             | Valid range            | Required |
        +==========+=======================================================================+=======================+========================+==========+
        | N        | Batch count of output tensor                                          | int                   |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | C        | Channel count of output tensor                                        | int                   |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | H        | Height count of output tensor                                         | int                   |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | W        | Width count of output tensor                                          | int                   |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | hOnes    | Height of high values region                                          | int                   | hOnes <= H             | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | wOnes    | Width of high values region                                           | int                   | wOnes <= W             | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        | any      | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
        +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
    )doc");
    m_tensor.def("pad_h_rm", &pad_h_rm, R"doc(
        Pads a given tensor's on the H dimension (2nd dimension from lowest)
        with 0s until the H reaches dimension ``paddedH``.

        +----------+----------------------+-----------+--------------+----------+
        | Argument | Description          | Data type | Valid range  | Required |
        +==========+======================+===========+==============+==========+
        | a        | Tensor to pad        | Tensor    |              | Yes      |
        +----------+----------------------+-----------+--------------+----------+
        | paddedH  | New H dim            | int       | >= current H | Yes      |
        +----------+----------------------+-----------+--------------+----------+
    )doc");

    // matrix multiplication

    m_tensor.def("large_bmm", &large_bmm, R"doc(
        Perform a batched matmul ``A x B`` with two tensors, where batch dims match.
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | LHS matmul operand                                                                         | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | RHS matmul operand                                                                         | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");
    m_tensor.def("large_bmm_single_block", &large_bmm_single_block, R"doc(
        Perform a batched matmul ``A x B`` with two tensors, where batch dims match.
        This op also supports tilizing tensor A and untilizing the output if you so choose.

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | LHS matmul operand                                                                         | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | RHS matmul operand                                                                         | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | tilize_a     | Whether or not to tilize a (useful if a is in row major layout)                            | bool      |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | untilize_out | Whether or not to untilize the output (useful if a consuming op requires row major layout) | bool      |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");
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
        Perform a batched matmul ``A x B`` with two tensors, where batch dims match.
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
    m_tensor.def("bert_large_fused_qkv_matmul", &bert_large_fused_qkv_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a bert_large_fused_qkv non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_ff1_matmul", &bert_large_ff1_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a bert_large_ff1 non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_ff2_matmul", &bert_large_ff2_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a bert_large_ff2 non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_selfout_matmul", &bert_large_selfout_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a bert_large_selfout non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_pre_softmax_bmm", &bert_large_pre_softmax_bmm,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a bert_large_pre_softmax_bmm batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bert_large_post_softmax_bmm", &bert_large_post_softmax_bmm,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
        Perform a bert_large_post_softmax_bmm batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("compute_conv_op_block_info", &compute_conv_op_block_info);

    // softmax
    m_tensor.def("softmax_in_place", &softmax_in_place,
        "Performs a softmax operation on the last tensor dimension. Returns a reference to the input tensor modified in place.");
    m_tensor.def("scale_mask_softmax_in_place", &scale_mask_softmax_in_place,
        "Performs a fused scale->attention_mask->softmax operation. Returns a reference to the input tensor modified in place.");

    // layernorm
    m_tensor.def("layernorm", &layernorm, "Performs a layernorm operation on the last tensor dimension.");
    m_tensor.def("layernorm_gamma", &layernorm_gamma, "Performs a layernorm operation on the last tensor dimension fused with post-multiplication via W-bcast.");
    m_tensor.def("layernorm_gamma_beta", &layernorm_gamma_beta, "Performs a layernorm operation on the last tensor dimension fused with post-multiplication and addition via W-bcast.");
    m_tensor.def("add_layernorm_gamma_beta", &add_layernorm_gamma_beta, "Performs a layernorm(a+b)*gamma + beta operation.");

    // TMs



    m_tensor.def("tilize_with_zero_padding", &tilize_with_zero_padding, R"doc(
        Tilizes a given tensor across memory on device. Pads zeroes height-wise if required.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");
    m_tensor.def("tilize_conv_activation", &tilize_conv_activation, R"doc(
        Converts conv activation to 2d Matrix and tilizes it on device. Pads zeroes height-wise if required.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
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
}

void DeviceModule(py::module &m_device) {
    py::enum_<tt::ARCH>(m_device, "Arch", "Enum of types of Tenstorrent accelerator devices.")
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL);

    py::enum_<tt::tt_metal::MemoryAllocator>(m_device, "MemoryAllocator", "Enum of types of memory allocation schemes.")
        .value("BASIC", tt::tt_metal::MemoryAllocator::BASIC)
        .value("L1_BANKING", tt::tt_metal::MemoryAllocator::L1_BANKING);

    auto pyDevice = py::class_<Device>(m_device, "Device", "Class describing a Tenstorrent accelerator device.");
    pyDevice
        .def(
            py::init<>(
                [](tt::ARCH arch, int pcie_slot) {
                    return Device(arch, pcie_slot);
                }
            ), "Create device."
        );

    auto pyHost = py::class_<Host>(m_device, "Host", "Class describing the host machine.");

    pyHost
        .def(
            py::init<>(
                []() {
                    return Host();
                }
            ), "Create host."
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
    m_device.def("InitializeDevice", &InitializeDevice, py::arg().noconvert(), py::arg("memory_allocator") = tt::tt_metal::MemoryAllocator::BASIC, R"doc(
        Initialize instance of TT accelerator device.

        +-------------------+--------------------------------------------------------+----------------------------------+-------------------------------------------+----------+
        |  Argument         |                 Description                            |       Data type                  |           Valid range                     | Required |
        +===================+========================================================+==================================+============================================+=========+
        | device            | Device to initialize                                   | tt_lib.device.Device             |                                           | Yes      |
        +-------------------+--------------------------------------------------------+----------------------------------+-------------------------------------------+----------+
        | memory_allocator  | Type of memory allocator scheme to use                 | tt_lib.device.MemoryAllocator    | tt_lib.device.MemoryAllocator.BASIC       | No       |
        |                   |                                                        |                                  |                                           |          |
        |                   |                                                        |                                  | tt_lib.device.MemoryAllocator.L1_BANKING  |          |
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

    m_device.def("SetDefaultDevice", &AutoPad::SetDefaultDevice, R"doc(
        Sets the default device to use for ops when inputs aren't on device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to use       | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("GetDefaultDevice", &AutoPad::GetDefaultDevice, R"doc(
        Gets the default device to use for ops when inputs aren't on device.
    )doc");

    m_device.def("StartDebugPrintServer", &StartDebugPrintServer);
    m_device.def("StartDebugPrintServerOnCores", &StartDebugPrintServerOnCores);
    m_device.def("SetProfilerDir", &SetProfilerDir);

    m_device.def("EnableCompileCache", &EnableCompileCache);
    m_device.def("DisableCompileCache", &DisableCompileCache);
    m_device.def("GetCompileCacheEnabled", &GetCompileCacheEnabled);

    m_device.def("GetHost", &GetHost, R"doc(
        Get a reference to host machine of a TT accelerator device, usually a reference to the host
        machine executing Python code.
    )doc");
}

void DTXModule(py::module &m_dtx) {
    auto pyDataTransformations = py::class_<DataTransformations>(m_dtx, "DataTransformations", "Class describing the data transformations.");
    m_dtx.def("evaluate", [](vector<float> data, vector<uint32_t> address_map, vector<int> output_shape){
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
    m_dtx.def("conv_transform", [](vector<int> shape, vector<int> conv_params, std::pair<vector<int>,vector<int>> block_info, uint32_t num_bytes_of_df){
        return conv_transform(shape, conv_params, block_info, num_bytes_of_df);
    });
}

} // end namespace tt_metal

} // end namespace tt


PYBIND11_MODULE(_C, m) {

    m.attr("__name__") = "tt_lib";
    m.doc() = "Python bindings for TT-Metal";

    py::module_ m_device = m.def_submodule("device", "Submodule defining a host or device");
    tt::tt_metal::DeviceModule(m_device);

    py::module_ m_tensor = m.def_submodule("tensor", "Submodule defining an tt_metal tensor");
    tt::tt_metal::TensorModule(m_tensor);

    py::module_ m_dtx = m.def_submodule("dtx", "Submodule defining data transformation engine");
    tt::tt_metal::DTXModule(m_dtx);
}
