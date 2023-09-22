#pragma once

#include "tt_lib_bindings.hpp"

namespace tt::tt_metal{

namespace detail{
    void TensorModuleCompositeOPs( py::module & m_tensor);
    void TensorModuleDMOPs( py::module & m_tensor);
    void TensorModuleCustomAndBMMOPs( py::module & m_tensor);
    void TensorModulePyTensor( py::module & m_tensor);

}

void TensorModule(py::module &m_tensor);

}
