#pragma once

#include "tt_lib_bindings.hpp"

namespace tt::tt_metal{

namespace detail{
    void TensorModuleCompositeOPs( py::module & m_tensor);
}

void TensorModule(py::module &m_tensor);

}
