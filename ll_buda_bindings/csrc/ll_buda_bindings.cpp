#include "ll_buda/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "ll_buda/op_library/bmm/bmm_op.hpp"
#include "ll_buda/op_library/bcast/bcast_op.hpp"
#include "ll_buda/op_library/reduce/reduce_op.hpp"
#include "ll_buda/op_library/transpose/transpose_op.hpp"
#include "ll_buda/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "ll_buda_bindings.hpp"

namespace py = pybind11;

namespace tt {

namespace ll_buda {

void reshape(Tensor& a, int N, int C, int H, int W) {
    a.reshape(N, C, H, W);
}

void TensorModule(py::module &m_tensor) {

    py::class_<Tensor>(m_tensor, "Tensor")
        .def(
            py::init<>(
                [](std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout) {
                    return Tensor(data, shape, data_type, layout); 
                }
            )
        )
        .def(
            py::init<>(
                [](std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout, Device *device) {
                    return Tensor(data, shape, data_type, layout, device);
                }
            )
        )
        .def("to", py::overload_cast<Device*>(&Tensor::to, py::const_), "Moves the tensor to device")
        .def("to", py::overload_cast<Host*>(&Tensor::to, py::const_), "Moves the tensor to CPU")
        .def("print", [](const Tensor &self, Layout print_layout = Layout::ROW_MAJOR) {
            return self.print(print_layout);
        }, "Prints the tensor")
        .def("shape", [](const Tensor &self) {
            return self.shape();
        }, "Returns the shape of the tensor")
        .def("data", [](const Tensor &self) {
            return self.data();
        }, "Returns the data in the output tensor as a 1D vector");
    
    // Tensor functions
    // eltwise binary
    m_tensor.def("reshape", &reshape);
        //.def("reshape", [](Tensor &self, int N, int C, int H, int W) {
        //    reshape(self, N, C, H, W);
        //}, "Sets the shape of the tensor")
    m_tensor.def("add", &add);
    m_tensor.def("sub", &sub);
    m_tensor.def("mul", &mul);

    // matrix multiplication
    m_tensor.def("matmul", &matmul);

    // broadcast math
    m_tensor.def("bcast", &bcast);

    // reduce
    m_tensor.def("reduce", &reduce);

    // eltwise unary SFPU
    m_tensor.def("exp", &exp);
    m_tensor.def("recip", &recip);
    m_tensor.def("gelu", &gelu);
    m_tensor.def("relu", &relu);
    m_tensor.def("sqrt", &sqrt);

    // TMs
    // transpose
    m_tensor.def("transpose", &transpose);
    m_tensor.def("transpose_hc", &transpose_hc);

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
    
    py::enum_<DataFormat>(m_tensor, "DataFormat")
        .value("FLOAT32", DataFormat::Float32)
        .value("BFLOAT16", DataFormat::Float16_b);
}

void DeviceModule(py::module &m_device) {
    py::class_<Device>(m_device, "Device")
        .def(
            py::init<>(
                [](tt::ARCH arch, int pcie_slot) {
                    return Device(arch, pcie_slot);
                }
            )
        );
    py::class_<Host>(m_device, "Host")
        .def(
            py::init<>(
                []() {
                    return Host();
                }
            )
        );
    m_device.def("CreateDevice", &CreateDevice);
    m_device.def("InitializeDevice", &InitializeDevice);
    m_device.def("CloseDevice", &CloseDevice);

    m_device.def("GetHost", &GetHost);
    
    py::enum_<tt::ARCH>(m_device, "Arch")
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL);

}

} // end namespace ll_buda
} // end namespace tt


PYBIND11_MODULE(_C, m) {

    m.attr("__name__") = "ll_buda_bindings._C";
    m.doc() = "ll_buda python bindings";

    py::module_ m_tensor = m.def_submodule("tensor", "Submodule defining an ll_buda tensor");
    tt::ll_buda::TensorModule(m_tensor);

    py::module_ m_device = m.def_submodule("device", "Submodule defining a host or device");
    tt::ll_buda::DeviceModule(m_device);
}
