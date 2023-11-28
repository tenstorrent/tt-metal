// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "tt_dnn/op_library/backward/backward_ops.hpp"

namespace tt::tt_metal::detail{
    void TensorModuleBackwardOPs( py::module & m_tensor){

    m_tensor.def("addalpha_bw", &tt::tt_metal::addalpha_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for multiplication of ``other`` and ``alpha`` tensors with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor addalpha is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("add_bw", &tt::tt_metal::add_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for addition of ``other`` tensors with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor add is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("unary_mul_bw", &tt::tt_metal::unary_mul_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("scalar") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for multiplication with given ``grad`` and ``scalar``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "scalar", "Scalar value", "float", "default to 1.0f", "No"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("mul_bw", &tt::tt_metal::mul_bw,
            py::arg("grad").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for multiplication of two input tensors with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensors will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input_a", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input_b", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("exp_bw", &tt::tt_metal::exp_bw,
            py::arg("grad").noconvert(), py::arg("exp_result").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for exponential with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "exp_result", "Result tensor of an exponentinal forward operation", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("addcmul_bw", &tt::tt_metal::addcmul_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("tensor1").noconvert(), py::arg("tensor2").noconvert(), py::arg("value") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for multiplication of ``tensor1``, ``tensor2`` and ``value`` tensors with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor addcmul_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor1", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor2", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "value", "Value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("unary_assign_bw", &tt::tt_metal::unary_assign_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for assign with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("unary_div_bw", &tt::tt_metal::unary_div_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("scalar") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for division with given ``grad`` and ``scalar``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "scalar", "Scalar value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("div_bw", &tt::tt_metal::div_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for division of ``other`` with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor addalpha is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");
    }

}
