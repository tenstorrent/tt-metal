// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "tt_dnn/op_library/backward/backward_ops.hpp"

namespace tt::tt_metal::detail{
    void TensorModuleBackwardOPs( py::module & m_tensor){

    m_tensor.def("addalpha_bw", &tt::tt_metal::addalpha_bw,
            py::arg("grad").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for multiplication of ``input_b`` and ``alpha`` tensors with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_a", "Tensor addalpha is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("conj_bw", &tt::tt_metal::conj_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for conjugate for complex tensor ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
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
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for exponential with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"


                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");


    m_tensor.def("exp2_bw", &tt::tt_metal::exp2_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for exp2 with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"


                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("expm1_bw", &tt::tt_metal::expm1_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for expm1 with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"


                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
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
                "input", "Tensor addcmul is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
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

    m_tensor.def("binary_assign_bw", &tt::tt_metal::binary_assign_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for binary assign on ``other`` with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("tan_bw", &tt::tt_metal::tan_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for tangent with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("tanh_bw", &tt::tt_metal::tanh_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for tanh with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("sigmoid_bw", &tt::tt_metal::tanh_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for sigmoid with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("addcdiv_bw", &tt::tt_metal::addcdiv_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("tensor1").noconvert(), py::arg("tensor2").noconvert(), py::arg("value") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for multiplication and division of ``tensor1``, ``tensor2`` and ``value`` tensors with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor addcdiv is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor1", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor2", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "value", "Value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("add_bw", &tt::tt_metal::add_bw,
            py::arg("grad").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for addition of ``input_b`` tensors with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_a", "Tensor add is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("relu_bw", &tt::tt_metal::relu_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for relu of ``input`` tensors with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor relu is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("unary_add_bw", &tt::tt_metal::unary_add_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for addition with given ``grad`` and ``alpha``.

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "alpha", "Alpha value", "float", "default to 1.0f", "No"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("unary_pow_bw", &tt::tt_metal::unary_pow_bw,
        py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("exponent") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for power with given ``grad`` and ``exponent`` where exponent value is greater than 1.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "exponent", "Exponent value", "integer", "default to 1", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("sqrt_bw", &tt::tt_metal::sqrt_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for sqrt with given ``grad``.

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("unary_div_bw", &tt::tt_metal::unary_div_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("scalar") = 1.0f, py::arg("round_mode") = "None", py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
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
            py::arg("grad").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("round_mode") = "None", py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for division of ``input_b`` with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_a", "Tensor div is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("rdiv_bw", &tt::tt_metal::rdiv_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("scalar") = 1.0f, py::arg("round_mode") = "None", py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for division for ``input`` tensor and ``scalar`` with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "scalar", "Scalar value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("min_bw", &tt::tt_metal::min_bw,
            py::arg("grad").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for minimum of ``input_b`` with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_a", "Tensor min is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("max_bw", &tt::tt_metal::max_bw,
            py::arg("grad").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for maximum of ``input_b`` with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_a", "Tensor max is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


    m_tensor.def("where_bw", &tt::tt_metal::where_bw,
            py::arg("grad").noconvert(), py::arg("condition").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for where selected from either ``input_a`` or ``input_b``, depending on ``condition`` with given ``grad``.
            When condition True (nonzero), yield grad, otherwise yield zero's.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "condition", "Tensor", "Bool", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_a", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


    m_tensor.def("fill_zero_bw", &tt::tt_metal::fill_zero_bw,
            py::arg("grad").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor


            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("fill_bw", &tt::tt_metal::fill_bw,
            py::arg("grad").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor like ``grad`` tensor with sum of tensor values

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("embedding_bw", &tt::tt_metal::embedding_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("weight").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for embedding_bw function and it returns specific indices of the embedding table specified by the grad tensor.
            The input tensor should be unique.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor containing rows we want", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "weight", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("sub_bw", &tt::tt_metal::sub_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for subraction of ``other`` and ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("unary_sub_bw", &tt::tt_metal::unary_sub_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for subraction of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor add is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("rsub_bw", &tt::tt_metal::rsub_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for subraction of ``input`` from ``other`` tensors with given ``grad`` (reversed order of subtraction operator).

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor add is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("neg_bw", &tt::tt_metal::neg_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for negative with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("lt_bw", &tt::tt_metal::lt_bw,
            py::arg("grad").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor


            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("gt_bw", &tt::tt_metal::gt_bw,
            py::arg("grad").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor


            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("ne_bw", &tt::tt_metal::ne_bw,
            py::arg("grad").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor


            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("log_bw", &tt::tt_metal::log_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for logarithm of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor add is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("abs_bw", &tt::tt_metal::abs_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for abs of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor add is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("rsqrt_bw", &tt::tt_metal::rsqrt_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for rsqrt of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor add is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("clamp_bw", &tt::tt_metal::clamp_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("min").noconvert(), py::arg("max").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for clamp of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "min", "Minimum Value", "float", , "Yes"
                "max", "Maximum Value", "float", , "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("clamp_min_bw", &tt::tt_metal::clamp_min_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("min").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for clamp min of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "min", "Minimum Value", "float", , "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("clamp_max_bw", &tt::tt_metal::clamp_max_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("max").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for clamp max of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "max", "Maximum Value", "float", , "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("binary_le_bw", &tt::tt_metal::binary_le_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor and ``input`` tensor.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor LE is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("lerp_bw", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&>(&lerp_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("end").noconvert(), py::arg("weight"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
            Applies the linear interpolation of two tensors ``arg0`` (given by input) and ``arg1`` based on a
            scalar ``arg2``  with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor lerp is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "end", "End value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "weight", "Weight value", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("lerp_bw", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&lerp_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("end").noconvert(), py::arg("weight").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Applies the linear interpolation of two tensors ``arg0`` (given by input) and ``arg1`` based on a
            tensor ``arg2`` with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor lerp is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "end", "End value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "weight", "Weight value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("atan2_bw", &tt::tt_metal::atan2_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for atan2 of ``input`` and ``other`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor atan2 is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor atan2 is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("hardshrink_bw", &tt::tt_metal::hardshrink_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("lambda") = 0.5f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for hardshrink to the elements of the input tensor ``{0}`` between limits ``-{1}`` low and the ``+{1}`` high limits with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor hardshrink_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "lambda", "lambda value", "float", "float", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("softshrink_bw", &tt::tt_metal::softshrink_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("lambd") = 0.5f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for softshrink to the elements of the input tensor ``{input}`` between limits ``-{lambd}`` low and the ``+{lambd}`` high limits with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor softshrink_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "lambd", "lambda value", "float", "float value >= 0", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("hypot_bw", &tt::tt_metal::hypot_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for hypotenuse of ``input`` and ``other`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "First input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Second input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("gelu_bw", &tt::tt_metal::gelu_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("approximate").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for gelu of ``input`` tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor gelu is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "approximate", "Approximation type", "String", "None, tanh", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("bias_gelu_bw", &tt::tt_metal::bias_gelu_bw,
            py::arg("grad").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("approximate").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for binary gelu of ``input_a`` and ``input_b`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_a", "Tensor gelu is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor gelu is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "approximate", "Approximation type", "String", "None, tanh", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("bias_gelu_unary_bw", &tt::tt_metal::bias_gelu_unary_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("bias").noconvert(), py::arg("approximate").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for unary gelu of ``input`` tensor with given ``grad`` at given ``bias``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor gelu is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "bias", "Bias value", "float", "float", "Yes"
                "approximate", "Approximation type", "String", "None, tanh", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("squared_difference_bw", &tt::tt_metal::squared_difference_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for squared difference of ``input`` and ``other`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor squared difference is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor squared difference is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("ldexp_bw", &tt::tt_metal::ldexp_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for ldexp of ``input`` and ``other`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("hardswish_bw", &tt::tt_metal::hardswish_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for hardswish_bw of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("softplus_bw", &tt::tt_metal::softplus_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("beta")=1.0f, py::arg("threshold") = 20.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for softplus_bw of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "beta", "Beta value", "float", "default to 1.0f", "No"
                "threshold", "Threshold value", "float", "default to 20.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

   m_tensor.def("xlogy_bw", &tt::tt_metal::xlogy_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for xlogy  for ``input`` and  ``other`` with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

   m_tensor.def("logaddexp_bw", &tt::tt_metal::logaddexp_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for logaddexp (``input``, ``other``)  with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("logaddexp2_bw", &tt::tt_metal::logaddexp2_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for logaddexp2 (``input``, ``other``)  with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("concat_bw", &tt::tt_metal::concat_bw,
            py::arg("grad").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(),  py::arg("dim") = 0, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for concat of ``input_a`` and ``input_b`` tensors with given ``grad``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_a", "Tensor concat is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor concat is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "Dim", "Dim value", "int", "default to 0", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("hardtanh_bw", &tt::tt_metal::hardtanh_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("min")= -1.0f, py::arg("max") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for hardtanh activation function on ``input`` tensor with given ``grad``, ``min`` and ``max``.

            Input tensor must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor hardtanh is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "Min", "Min value", "Float", "default to -1.0f", "Yes"
                "Max", "Max value", "Float", "default to 1.0f", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("hardsigmoid_bw", &tt::tt_metal::hardsigmoid_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for hardsigmoid of ``input`` tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor hardsigmoid is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("asin_bw", &tt::tt_metal::asin_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for asin_bw of ``input`` tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor asin_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("asinh_bw", &tt::tt_metal::asinh_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for asinh_bw of ``input`` tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor asinh_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("leaky_relu_bw", &tt::tt_metal::leaky_relu_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("negative_slope") = 0.01f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for leaky_relu_bw of ``input`` tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor leaky_relu_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("i0_bw", &tt::tt_metal::i0_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for i0 of ``input`` and tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor i0 is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("polygamma_bw", &tt::tt_metal::polygamma_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("n").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for polygamma of ``input`` tensor for order ``n`` with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor polygamma is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "n", "order of the polygamma", "int", "1 to 10", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

     m_tensor.def("atan_bw", &tt::tt_metal::atan_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for atan of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("atanh_bw", &tt::tt_metal::atanh_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for atanh of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("cos_bw", &tt::tt_metal::cos_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for cos of the ``input`` with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("cosh_bw", &tt::tt_metal::cosh_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for hyperbolic cos of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


    m_tensor.def("acosh_bw", &tt::tt_metal::acosh_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for acosh of ``input`` and tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("angle_bw", &tt::tt_metal::angle_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for angle for the ``input`` with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("acos_bw", &tt::tt_metal::acos_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for acos for the ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("sin_bw", &tt::tt_metal::sin_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for sin of the ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("sinh_bw", &tt::tt_metal::sinh_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for hyperbolic sin of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("celu_bw", &tt::tt_metal::celu_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(),  py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for Celu of ``input`` tensor  with given ``grad`` and ``alpha``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor Celu is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

     m_tensor.def("binary_lt_bw", &tt::tt_metal::binary_lt_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor and ``input`` tensor.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor LT is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("elu_bw", &tt::tt_metal::elu_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("alpha").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for elu for the ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "alpha", "alpha value", "float", " ", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

     m_tensor.def("erfinv_bw", &tt::tt_metal::erfinv_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for erfinv of ``input`` tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


    m_tensor.def("subalpha_bw", &tt::tt_metal::subalpha_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(

            Performs backward operations for subraction of ``other`` and ``input`` tensors with alpha for the given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


    m_tensor.def("log10_bw", &tt::tt_metal::log10_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for log10 of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


    m_tensor.def("log1p_bw", &tt::tt_metal::log1p_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for log1p of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("binary_ne_bw", &tt::tt_metal::binary_ne_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor and ``input`` tensor.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

     m_tensor.def("erf_bw", &tt::tt_metal::erf_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for erf of ``input`` tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

     m_tensor.def("erfc_bw", &tt::tt_metal::erfc_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for erfc of ``input`` tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("digamma_bw", &tt::tt_metal::digamma_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for digamma for the ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("deg2rad_bw", &tt::tt_metal::deg2rad_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for deg2rad for the ``input`` with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("rad2deg_bw", &tt::tt_metal::rad2deg_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for rad2deg for the ``input`` with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("reciprocal_bw", &tt::tt_metal::reciprocal_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for reciprocal for the ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

     m_tensor.def("relu6_bw", &tt::tt_metal::relu6_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of backward operation of relu6 for ``input`` tensor and ``grad`` tensor.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor relu6 is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("rpow_bw", &tt::tt_metal::rpow_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("exponent").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for rpow for the ``input`` and ``exponent`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "exponent", "exponent", "float", ">0.0", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("silu_bw", &tt::tt_metal::silu_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for silu sin of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor silu_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


    m_tensor.def("selu_bw", &tt::tt_metal::selu_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for selu sin of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor selu_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("binary_ge_bw", &tt::tt_metal::binary_ge_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor and ``input`` tensor.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("binary_eq_bw", &tt::tt_metal::binary_eq_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor and ``input`` tensor.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("binary_gt_bw", &tt::tt_metal::binary_gt_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor and ``input`` tensor.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("square_bw", &tt::tt_metal::square_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward square operations on ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor square_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("lgamma_bw", &tt::tt_metal::lgamma_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for lgamma of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor lgamma is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("frac_bw", &tt::tt_metal::frac_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for frac of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor frac is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("trunc_bw", &tt::tt_metal::trunc_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for trunc of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor trunc is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("log_sigmoid_bw", &tt::tt_metal::log_sigmoid_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for log_sigmoid ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor log_sigmoid is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("tanhshrink_bw", &tt::tt_metal::tanhshrink_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward tanhshrink operations on ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor tanhshrink_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("threshold_bw", &tt::tt_metal::threshold_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("threshold"), py::arg("value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward threshold operation on ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor threshold_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "threshold", "Value to threshold at", "float", "", "Yes"
                "value", "Value to replace with", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("unary_eq_bw", &tt::tt_metal::unary_eq_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Value to compare", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("logit_bw", &tt::tt_metal::logit_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for logit of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


   m_tensor.def("logiteps_bw", &tt::tt_metal::logiteps_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("eps") = 0.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for logit of ``input`` tensors with `eps` for the given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "eps", "eps value", "float", "default to 0.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("softsign_bw", &tt::tt_metal::softsign_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward softsign operations on ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor softsign_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("sign_bw", &tt::tt_metal::sign_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward sign operations on ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor sign_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("ceil_bw", &tt::tt_metal::ceil_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward ceil operations on ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor ceil_bw is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("log2_bw", &tt::tt_metal::log2_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for log2 of ``input`` tensors with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("ge_bw", &tt::tt_metal::ge_bw,
            py::arg("grad").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("le_bw", &tt::tt_metal::le_bw,
            py::arg("grad").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns an tensor of zeros like ``grad`` tensor

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

   m_tensor.def("unary_fmod_bw", &tt::tt_metal::unary_fmod_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("scalar"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for fmod of ``input`` tensors with `scalar` for the given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "scalar", "scalar value", "float", "float", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

   m_tensor.def("unary_remainder_bw", &tt::tt_metal::unary_remainder_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("scalar"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for ramainder of ``input`` tensors with `scalar` for the given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "scalar", "scalar value", "float", "float", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("imag_bw", &tt::tt_metal::imag_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for imaginary part of complex tensor ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("real_bw", &tt::tt_metal::real_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for real part of complex tensor ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("multigammaln_bw", &tt::tt_metal::multigammaln_bw,
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for multigammaln of ``input`` tensors with given ``grad`` and value of P is taken as 4.

            mvlgamma is refered as multigammaln.

            Input value must be greater than 2.5f

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor mvlgamma is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");
    }
}
