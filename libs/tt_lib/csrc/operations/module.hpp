#pragma once

#include "primary/module.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


namespace tt {
namespace operations {

void py_module(py::module& m_operations) {
    auto m_primary = m_operations.def_submodule("primary", "Primary operations");
    primary::py_module(m_primary);
}

}  // namespace operations

}  // namespace tt
