#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/conv/conv_op.hpp"
#include "tt_dnn/op_library/pad_h_rm/pad_h_rm_op.hpp"
#include "tt_dnn/op_library/fill_rm/fill_rm_op.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_dnn/op_library/transpose_rm/transpose_rm_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"

#include "tt_lib_bindings.hpp"
#include "type_caster.hpp"

namespace py = pybind11;

namespace tt {

namespace tt_metal {

extern void SetForceRecompiles(int newval);
extern int  GetForceRecompiles();
extern void EnableCompileCache();
extern int  DisableCompileCache();
extern bool GetCompileCacheEnabled();
extern void EnableBinaryCache();
extern int  DisableBinaryCache();
extern bool GetBinaryCacheEnabled();

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
        .value("UINT32", DataType::UINT32);

    auto pyMemoryConfig = py::class_<MemoryConfig>(m_tensor, "MemoryConfig", R"doc(
        Class defining memory configuration for storing tensor data on TT Accelerator device.
        There are eight DRAM memory banks on TT Accelerator device, indexed as 0, 1, 2, ..., 7.
    )doc");

    pyMemoryConfig
        .def(
            py::init<>(
                [](bool interleaved, int dram_channel) {
                    return MemoryConfig{.interleaved=interleaved, dram_channel=dram_channel};
                }
            ),
            py::arg("interleaved") = true,
            py::arg("dram_channel") = -1, R"doc(
                Create MemoryConfig class.
                If interleaved is set to True, tensor data will be interleaved across multiple DRAM banks on TT Accelerator device.
                Otherwise, tensor data will be stored in a DRAM bank selected by dram_channel (valid values are 0, 1, ..., 7).

                Example of creating MemoryConfig specifying that tensor data should be stored in DRAM bank 3.

                .. code-block:: python

                    mem_config = tt_lib.tensor.MemoryConfig(False, 3)
            )doc"
        )
        .def_readonly("interleaved", &MemoryConfig::interleaved, "Whether tensor data is interleaved across mulitple DRAM channels")
        .def_readonly("dram_channel", &MemoryConfig::dram_channel, "DRAM channel holding tensor data. Only used when tensor is not interleaved");

    auto pyTensor = py::class_<Tensor>(m_tensor, "Tensor", R"doc(


        Class constructor supports tensors of rank 4 where the size of both last two dimensions is a multiple of 32.
        The constructor takes following arguments:

        +------------+--------------------------------------------------------+---------------------------+---------------------------------+----------+
        |  Argument  |                 Description                            |       Data type           |           Valid range           | Required |
        +============+========================================================+===========================+=================================+==========+
        | data       | Data to store in TT tensor                             | List[float/int]           |                                 | Yes      |
        +------------+--------------------------------------------------------+---------------------------+---------------------------------+----------+
        | shape      | Shape of TT tensor                                     | List[int[4]]              |                                 | Yes      |
        +------------+--------------------------------------------------------+---------------------------+---------------------------------+----------+
        | data_type  | Data type of numbers in TT tensor                      | tt_lib.tensor.DataType    | tt_lib.tensor.DataType.BFLOAT16 | Yes      |
        +------------+--------------------------------------------------------+---------------------------+---------------------------------+----------+
        | layout     | Layout of tensor data in memory                        | tt_lib.tensor.Layout      | tt_lib.tensor.Layout.ROW_MAJOR  | Yes      |
        +------------+--------------------------------------------------------+---------------------------+---------------------------------+----------+
        | device     | Device on whihc tensor will be created                 | tt_lib.device.Device      | Host or TT accelerator device   | No       |
        +------------+--------------------------------------------------------+---------------------------+---------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator device memory banks | tt_lib.tensor.MemoryConfig|                                 | No       |
        +------------+--------------------------------------------------------+---------------------------+---------------------------------+----------+

    )doc");

    pyTensor
        .def(
            py::init<>(
                [](std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout) {
                    return Tensor(data, shape, data_type, layout);
                }
            ), R"doc(
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
            ), R"doc(
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
            ), R"doc(
                Example of creating a TT Tensor on TT accelerator device with specified mem_config:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    tt_device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
                    mem_config = tt_lib.tensor.MemoryConfig(Fasle, 3)
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
            ), R"doc(
                Not supported.
            )doc"
        )
        .def(
            py::init<>(
                [](std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device, const MemoryConfig &mem_config) {
                    return Tensor(data, shape, data_type, layout, device, mem_config);
                }
            ), R"doc(
                Not supported.
            )doc"
        )
        .def("to", [](const Tensor &self, Device *device, const MemoryConfig &mem_config) {
            return self.to(device, mem_config);
        }, py::arg().noconvert(), py::arg("mem_config") = MemoryConfig{.interleaved = true}, R"doc(
            Moves TT Tensor form host device to TT accelerator device.

            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def("to", py::overload_cast<Host*>(&Tensor::to, py::const_), R"doc(
            Move TT Tensor form TT accelerator device to host device.

            .. code-block:: python

                tt_tensor = tt_tensor.to(host)
        )doc")
        .def("to", py::overload_cast<Layout>(&Tensor::to, py::const_), R"doc(
            Convert TT Tensor to provided memory layout. Available layouts are TILE and ROW_MAJOR.

            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_lib.tensor.Layout.TILE)
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

        )doc");

    // Tensor functions
    // eltwise binary
    const std::string add_doc = R"doc(
        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | First tensor to add  | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
        | b        | Second tensor to add | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc";
    m_tensor.def("add", &add, "Perform an eltwise-binary add on two tensors.");

    const std::string sub_doc = R"doc(
        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | First tensor to sub  | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
        | b        | Second tensor to sub | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc";

    m_tensor.def("sub", &sub, "Perform an eltwise-binary sub on two tensors.");
    m_tensor.def("mul", &mul, "Perform an eltwise-binary mul on two tensors.");

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
    m_tensor.def("matmul", &matmul, R"doc(
        Perform a non-batched matmul ``A x B`` with two tensors.
    )doc");
    m_tensor.def("bmm", &bmm, R"doc(
        Perform a batched matmul ``A x B`` with two tensors, where batch dims match.
    )doc");
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
    m_tensor.def("conv_as_large_bmm_single_core_single_block", &conv_as_large_bmm_single_core_single_block, R"doc(
        Perform a batched matmul ``A x B`` with two tensors, where batch dims match.
        This op also supports tilizing tensor A and untilizing the output if you so choose.

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | LHS matmul operand                                                                         | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | RHS matmul operand                                                                         | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | untilize_out | Whether or not to untilize the output (useful if a consuming op requires row major layout) | bool      |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | use_single_bank_reader | Whether or not to use single bank reader kernel. Useful for debugging.           | bool      |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");
    m_tensor.def("conv_as_large_bmm_single_core", &conv_as_large_bmm_single_core, R"doc(
        Perform a batched matmul ``A x B`` with two tensors, where batch dims match.
        This op also supports tilizing tensor A and untilizing the output if you so choose.

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | LHS matmul operand                                                                         | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | RHS matmul operand                                                                         | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | untilize_out | Whether or not to untilize the output (useful if a consuming op requires row major layout) | bool      |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");
    // broadcast math
    m_tensor.def("bcast", &bcast, R"doc(
        Perform a broadcasted binary math operation between two tensors.

        The first tensor, ``a``, is the one to be broadcast.

        +-----------+-------------------------------+----------------------------+-------------+----------+
        | Argument  | Description                   | Data type                  | Valid range | Required |
        +===========+===============================+============================+=============+==========+
        | a         | Input tensor                  | tt_lib.tensor.Tensor       |             | Yes      |
        +-----------+-------------------------------+----------------------------+-------------+----------+
        | b         | Input tensor                  | tt_lib.tensor.Tensor       |             | Yes      |
        +-----------+-------------------------------+----------------------------+-------------+----------+
        | bcast_op  | Math operation to perform     | tt_lib.tensor.BcastOpMath  |             | Yes      |
        +-----------+-------------------------------+----------------------------+-------------+----------+
        | bcast_dim | Height count of output tensor | tt_lib.tensor.BcastOpDim   |             | Yes      |
        +-----------+-------------------------------+----------------------------+-------------+----------+
    )doc");

    // reduce
    m_tensor.def("reduce", &reduce, R"doc(
        Perform a reduce with a specified aggregation function on a tensor.

        +-------------+---------------------------------------+-----------------------------+-------------+----------+
        | Argument    | Description                           | Data type                   | Valid range | Required |
        +=============+=======================================+=============================+=============+==========+
        | a           | Input tensor                          | tt_lib.tensor.Tensor        |             | Yes      |
        +-------------+---------------------------------------+-----------------------------+-------------+----------+
        | reduce_math | Aggregating math operation            | tt_lib.tensor.ReduceOpMath  |             | Yes      |
        +-------------+---------------------------------------+-----------------------------+-------------+----------+
        | reduce_dim  | Dim to perform aggregation over       | tt_lib.tensor.ReduceOpDim   |             | Yes      |
        +-------------+---------------------------------------+-----------------------------+-------------+----------+
        | scalar      | Scalar to apply during math operation | float                       |             | Yes      |
        +-------------+---------------------------------------+-----------------------------+-------------+----------+
    )doc");

    // eltwise unary SFPU
    m_tensor.def("exp", &exp, "Performs a unary exp operation on a tensor.");
    m_tensor.def("recip", &recip, "Performs a unary recip operation on a tensor.");
    m_tensor.def("gelu", &gelu, "Performs a unary gelu operation on a tensor.");
    m_tensor.def("relu", &relu, "Performs a unary relu operation on a tensor.");
    m_tensor.def("sqrt", &sqrt, "Performs a unary sqrt operation on a tensor.");
    m_tensor.def("sigmoid", &sigmoid, "Performs a unary sigmoid operation on a tensor.");
    m_tensor.def("log", &log, "Performs a unary log operation on a tensor.");
    m_tensor.def("tanh", &tanh, "Performs a unary tanh operation on a tensor.");

    // TMs
    m_tensor.def("reshape", &reshape, R"doc(
        Reshapes a tensor given new N, C, H, and W dimensions and returns
        a tensor (a new view).

        +----------+--------------------------------+-----------------------+-------------+----------+
        | Argument | Description                    | Data type             | Valid range | Required |
        +==========+================================+=======================+=============+==========+
        | a        | Input tensor                   | tt_lib.tensor.Tensor  |             | Yes      |
        +----------+--------------------------------+-----------------------+-------------+----------+
        | N        | Batch count of output tensor   | int                   |             | Yes      |
        +----------+--------------------------------+-----------------------+-------------+----------+
        | C        | Channel count of output tensor | int                   |             | Yes      |
        +----------+--------------------------------+-----------------------+-------------+----------+
        | H        | Height count of output tensor  | int                   |             | Yes      |
        +----------+--------------------------------+-----------------------+-------------+----------+
        | W        | Width count of output tensor   | int                   |             | Yes      |
        +----------+--------------------------------+-----------------------+-------------+----------+
    )doc");

    m_tensor.def("transpose", &transpose, R"doc(
        Transposes a given tensor's H and W dimensions.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");
    m_tensor.def("transpose_hc", &transpose_hc, R"doc(
        Transposes a given tensor's H and C dimensions.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");
    m_tensor.def("transpose_hc_rm", &transpose_hc_rm, R"doc(
        Transposes a given tensor's H and C dimensions, row-major wise.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");
    m_tensor.def("tilize", &tilize, R"doc(
        Tilizes a given tensor across memory on device.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");
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
    m_tensor.def("untilize", &untilize, R"doc(
        Untilizes a given tensor tilized across memory on device.

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
    m_device.def("InitializeDevice", &InitializeDevice, R"doc(
        Initialize instance of TT accelerator device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | Device to initialize   | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");
    m_device.def("CloseDevice", &CloseDevice, R"doc(
        Reset an instance of TT accelerator device to default state and relinquish connection to device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to close     | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("StartDebugPrintServer", &StartDebugPrintServer);
    m_device.def("SetProfilerDir", &SetProfilerDir);

    m_device.def("SetForceRecompiles", &SetForceRecompiles);
    m_device.def("GetForceRecompiles", &GetForceRecompiles);
    m_device.def("EnableCompileCache", &EnableCompileCache);
    m_device.def("DisableCompileCache", &DisableCompileCache);
    m_device.def("GetCompileCacheEnabled", &GetCompileCacheEnabled);
    m_device.def("EnableBinaryCache", &EnableBinaryCache);
    m_device.def("DisableBinaryCache", &DisableBinaryCache);
    m_device.def("GetBinaryCacheEnabled", &GetBinaryCacheEnabled);

    m_device.def("GetHost", &GetHost, R"doc(
        Get a reference to host machine of a TT accelerator device, usually a reference to the host
        machine executing Python code.
    )doc");
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
}
