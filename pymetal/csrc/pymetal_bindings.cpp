#include "tt_metal/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_metal/op_library/bmm/bmm_op.hpp"
#include "tt_metal/op_library/pad_h_rm/pad_h_rm_op.hpp"
#include "tt_metal/op_library/fill_rm/fill_rm_op.hpp"
#include "tt_metal/op_library/bcast/bcast_op.hpp"
#include "tt_metal/op_library/reduce/reduce_op.hpp"
#include "tt_metal/op_library/transpose/transpose_op.hpp"
#include "tt_metal/op_library/transpose_rm/transpose_rm_op.hpp"
#include "tt_metal/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/op_library/tilize/tilize_op.hpp"
#include "tt_metal/op_library/untilize/untilize_op.hpp"
#include "tt_metal/op_library/reshape/reshape_op.hpp"

#include "pymetal_bindings.hpp"
#include "pymetal/csrc/type_caster.hpp"

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

    auto pyTensor = py::class_<Tensor>(m_tensor, "Tensor");

    pyTensor
        .def(
            py::init<>(
                [](std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout) {
                    return Tensor(data, shape, data_type, layout);
                }
            )
        )
        .def(
            py::init<>(
                [](std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device) {
                    return Tensor(data, shape, data_type, layout, device);
                }
            )
        )
        .def(
            py::init<>(
                [](std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout) {
                    return Tensor(data, shape, data_type, layout);
                }
            )
        )
        .def(
            py::init<>(
                [](std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device) {
                    return Tensor(data, shape, data_type, layout, device);
                }
            )
        )
        .def("to", py::overload_cast<Device*>(&Tensor::to, py::const_), "Moves the tensor to device")
        .def("to", py::overload_cast<Host*>(&Tensor::to, py::const_), "Moves the tensor to CPU")
        .def("print", [](const Tensor &self, Layout print_layout = Layout::ROW_MAJOR) {
            return self.print(print_layout);
        }, "Prints the tensor")
        .def("pretty_print", [](const Tensor &self) {
            return self.pretty_print();
        }, "Pretty prints the tensor")
        .def("shape", [](const Tensor &self) {
            return self.shape();
        }, "Returns the shape of the tensor")
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
        }, "Returns the data in the output tensor as a 1D vector")
        .def("layout", [](const Tensor &self) {
            return self.layout();
        }, "Returns the layout of the tensor");

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
        | any      | Any input tensor with desired device and data types for output tensor | ttmetal.tensor.Tensor |                        | Yes      |
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
        | any      | Any input tensor with desired device and data types for output tensor | ttmetal.tensor.Tensor |                        | Yes      |
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

    // broadcast math
    m_tensor.def("bcast", &bcast, R"doc(
        Perform a broadcasted binary math operation between two tensors.

        The first tensor, ``a``, is the one to be broadcast.

        +-----------+-------------------------------+----------------------------+-------------+----------+
        | Argument  | Description                   | Data type                  | Valid range | Required |
        +===========+===============================+============================+=============+==========+
        | a         | Input tensor                  | ttmetal.tensor.Tensor      |             | Yes      |
        +-----------+-------------------------------+----------------------------+-------------+----------+
        | b         | Input tensor                  | ttmetal.tensor.Tensor      |             | Yes      |
        +-----------+-------------------------------+----------------------------+-------------+----------+
        | bcast_op  | Math operation to perform     | ttmetal.tensor.BcastOpMath |             | Yes      |
        +-----------+-------------------------------+----------------------------+-------------+----------+
        | bcast_dim | Height count of output tensor | ttmetal.tensor.BcastOpDim  |             | Yes      |
        +-----------+-------------------------------+----------------------------+-------------+----------+
    )doc");

    // reduce
    m_tensor.def("reduce", &reduce, R"doc(
        Perform a reduce with a specified aggregation function on a tensor.

        +-------------+---------------------------------------+-----------------------------+-------------+----------+
        | Argument    | Description                           | Data type                   | Valid range | Required |
        +=============+=======================================+=============================+=============+==========+
        | a           | Input tensor                          | ttmetal.tensor.Tensor       |             | Yes      |
        +-------------+---------------------------------------+-----------------------------+-------------+----------+
        | reduce_math | Aggregating math operation            | ttmetal.tensor.ReduceOpMath |             | Yes      |
        +-------------+---------------------------------------+-----------------------------+-------------+----------+
        | reduce_dim  | Dim to perform aggregation over       | ttmetal.tensor.ReduceOpDim  |             | Yes      |
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
        | a        | Input tensor                   | ttmetal.tensor.Tensor |             | Yes      |
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
    py::enum_<tt::ARCH>(m_device, "Arch", "Type of Tenstorrent accelerator device")
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL);

    auto pyDevice = py::class_<Device>(m_device, "Device", "A Tenstorrent accelerator device");

    pyDevice
        .def(
            py::init<>(
                [](tt::ARCH arch, int pcie_slot) {
                    return Device(arch, pcie_slot);
                }
            ), "Create device"
        );

    auto pyHost = py::class_<Host>(m_device, "Host", "A host machine");

    pyHost
        .def(
            py::init<>(
                []() {
                    return Host();
                }
            ), "Create host"
        );

    m_device.def("CreateDevice", &CreateDevice, R"doc(
        Creates a device instance.

        +------------------+------------------------+---------------------+-------------+----------+
        | Argument         | Description            | Data type           | Valid range | Required |
        +==================+========================+=====================+=============+==========+
        | arch             | Device type            | ttmetal.device.Arch |             | Yes      |
        +------------------+------------------------+---------------------+-------------+----------+
        | pci_express_slot | PCI Express slot index | int                 |             | Yes      |
        +------------------+------------------------+---------------------+-------------+----------+
    )doc");
    m_device.def("InitializeDevice", &InitializeDevice, R"doc(
        Initialize device instance with default params.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | Device to initialize   | ttmetal.device.Device |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");
    m_device.def("CloseDevice", &CloseDevice, R"doc(
        Close a device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | Device to close        | ttmetal.device.Device |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("StartDebugPrintServer", &StartDebugPrintServer);
    m_device.def("setProfilerDir", &setProfilerDir);

    m_device.def("SetForceRecompiles", &SetForceRecompiles);
    m_device.def("GetForceRecompiles", &GetForceRecompiles);
    m_device.def("EnableCompileCache", &EnableCompileCache);
    m_device.def("DisableCompileCache", &DisableCompileCache);
    m_device.def("GetCompileCacheEnabled", &GetCompileCacheEnabled);
    m_device.def("EnableBinaryCache", &EnableBinaryCache);
    m_device.def("DisableBinaryCache", &DisableBinaryCache);
    m_device.def("GetBinaryCacheEnabled", &GetBinaryCacheEnabled);

    m_device.def("GetHost", &GetHost);
}

} // end namespace tt_metal

} // end namespace tt


PYBIND11_MODULE(_C, m) {

    m.attr("__name__") = "ttmetal";
    m.doc() = "General purpose AI python bindings";

    py::module_ m_device = m.def_submodule("device", "Submodule defining a host or device");
    tt::tt_metal::DeviceModule(m_device);

    py::module_ m_tensor = m.def_submodule("tensor", "Submodule defining an tt_metal tensor");
    tt::tt_metal::TensorModule(m_tensor);
}
