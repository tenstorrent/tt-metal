/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tt_numpy/functions.hpp"

#include "primary/module.hpp"

namespace py = pybind11;


namespace tt {
namespace operations {

void py_module(py::module& m_operations) {
    auto m_primary = m_operations.def_submodule("primary", "Primary operations");
    primary::py_module(m_primary);

    m_operations.def("zeros",
        [] (const std::vector<uint32_t>& shape, DataType dtype, Layout layout) {
            return tt::numpy::zeros(shape, dtype, layout);
        },
        py::arg("shape"),
        py::arg("dtype").noconvert() = DataType::BFLOAT16,
        py::arg("layout").noconvert() = Layout::ROW_MAJOR,
        R"doc(
            Returns a new tensor filled with zeros of shape specified by argument ``shape``.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "shape", "Shape vector", "vector<int>", "", "Yes"
                "dtype", "Data type", "DataType", "", "Yes"
                "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
        )doc"
    );


    m_operations.def("random",
        [] (const std::vector<uint32_t>& shape, DataType dtype, Layout layout) {
            return tt::numpy::random::random(shape, dtype, layout);
        },
        py::arg("shape"),
        py::arg("dtype").noconvert() = DataType::BFLOAT16,
        py::arg("layout").noconvert() = Layout::ROW_MAJOR,
        R"doc(
            Returns a new tensor filled with random values of shape specified by argument ``shape``.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "shape", "Shape vector", "vector<int>", "", "Yes"
                "dtype", "Data type", "DataType", "", "Yes"
                "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
        )doc"
    );
}

}  // namespace operations

}  // namespace tt
