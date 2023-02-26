#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ll_buda/host_api.hpp"
#include "ll_buda/tensor/tensor.hpp"

namespace py = pybind11;

namespace tt {

void PyTensor(py::module &tensor);

}
