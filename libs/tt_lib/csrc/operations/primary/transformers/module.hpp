#pragma once

#include "tt_dnn/op_library/transformer_tms/transformer_tms.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace tt {
namespace operations {
namespace primary {
namespace transformers {


void py_module(py::module& m_transformers) {
    m_transformers.def("split_fused_qkv_and_split_heads", &split_fused_qkv_and_split_heads,
        py::arg().noconvert(), py::arg("compute_with_storage_grid_size").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Splits [9, 1, 384, 3072] fused qkv matrix into 3 heads with shapes [9, 16, 384, 64], [9, 16, 64, 384], and [9, 16, 384, 64].
    )doc");

    m_transformers.def("concatenate_heads", &concatenate_heads,
        py::arg().noconvert(), py::arg("compute_with_storage_grid_size").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Reshuffles [9, 16, 384, 64] tensor into tensor with shape [9, 1, 384, 1024].
    )doc");

    m_transformers.def("scale_mask_softmax_in_place", &scale_mask_softmax_in_place,
        "Performs a fused scale->attention_mask->softmax operation. Returns a reference to the input tensor modified in place.");
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
