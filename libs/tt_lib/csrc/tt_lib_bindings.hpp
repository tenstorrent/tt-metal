#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"

namespace py = pybind11;

namespace tt {

void PyTensor(py::module &tensor);

}
