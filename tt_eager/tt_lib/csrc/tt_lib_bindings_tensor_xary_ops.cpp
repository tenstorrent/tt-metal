// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"

namespace tt::tt_metal::detail {
    void TensorModuleXaryOPs( py::module & m_tensor){
        // *** eltwise unary ***
        detail::bind_unary_op(m_tensor, "identity", identity, R"doc(Returns a copy of same tensor ``input``; useful for profiling the SFPU.
        this shouldn't normally be used; users should normally use clone operation instead for same functionality as this would be lower performance.
        )doc");
    detail::bind_unary_op(
        m_tensor,
        "identity_uint32",
        identity_uint32,
        R"doc(Returns a copy of same tensor ``input``; useful for profiling the SFPU.
        this shouldn't normally be used; users should normally use clone operation instead for same functionality as this would be lower performance.
        Use this version of identity only if input is in uint32 format
        )doc");
        detail::bind_unary_op(m_tensor, "sigmoid_accurate", sigmoid_accurate, R"doc(Applies the sigmoid_accurate function to the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "tiled_prod", tiled_prod, R"doc(Performs tile-wise multiplication on input tensor ``{0}`` and store the result in the last tile of the input tensor.)doc");
        detail::bind_unary_op(m_tensor, "floor", floor, R"doc(Applies floor to the elements of the input tensor ``{0}``. Support provided only for Wormhole_B0.)doc");
        detail::bind_unary_op(m_tensor, "log_sigmoid", &log_sigmoid, R"doc(Applies the logsigmoid function to the elements of the input tensor ``{0}`` for input range [-4,10].)doc");

        m_tensor.def("eltwise_typecast", &eltwise_typecast,
            py::arg("input").noconvert(), py::arg("tt_input_dtype"), py::arg("tt_output_dtype"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns tensor with all elements of the input tensor ``{0}`` typecasted.
            Supported typecasts:
                BFLOAT16 <-> FLOAT32
                BFLOAT16 <-> INT32
                BFLOAT16 <-> UINT16
                BFLOAT16 <-> UINT32
                FLOAT32 <-> INT32
                FLOAT32 <-> UINT16
                FLOAT32 <-> UINT32
                BFLOAT8_B <-> INT32
                BFLOAT8_B <-> UINT16
                BFLOAT8_B <-> UINT32

            Input tensor must have tt_input_dtype data type.

            Output tensor will have tt_output_dtype data type.

            Note: This operation is not supported on Grayskull.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor softplus is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tt_input_dtype", "Input tensor DataType", "DataType", "One of supported input DataTypes", "Yes"
                "tt_output_dtype", "Desired output tensor DataType", "DataType", "One of supported output DataTypes", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


        detail::bind_unary_op_with_param(
            m_tensor, "rsqrt", &rsqrt,
            py::arg("fast_and_approx") = true,
            R"doc(Returns a new tensor with the reciprocal of the square-root of each of the elements of the input tensor ``{0}`` for the input range -10 to 10.)doc",
            R"doc("Indicate true for approx and fast mode; false for accurate and slow mode", "bool", "default of true")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "relu_max", relu_max,
            py::arg("upper_limit"),
            R"doc(Returns tensor with the relu max of all of elements of the input tensor ``{0}``. This is equivalent
            to relu_max[x] = relu(min(x, ``{1}``)). It caps off the input to a max value and a min value of 0.)doc",
        R"doc("max value", "float", "")doc"

    );
    detail::bind_unary_op_with_param(
        m_tensor,
        "relu_min",
        relu_min,
        py::arg("lower_limit"),
        R"doc(Returns tensor with the relu min of all of elements of the input tensor ``{0}``. This is equivalent
            to relu_min[x] = max(x, ``{1}``). It moves relu function down to carry out operation at minvalue
            instead of the standard 0.)doc",
        R"doc("min value", "float", "")doc"

    );

        m_tensor.def("bitwise_xor",bitwise_xor,
            py::arg("input").noconvert(),py::arg("value"),py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
            Computes bitwise_xor of input tensor ``input`` by a scalar ``value``. Input tensor needs to be positive. Support provided only for Wormhole_B0.

            Input tensor must have INT32 data type.

            Output tensor will have INT32 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "value", "scalar value", "int", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");

        m_tensor.def("bitwise_not",bitwise_not,
            py::arg("input").noconvert(),py::arg("value"),py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
            Computes bitwise_not of input tensor ``input``. Input tensor needs to be in the range [-2147483647, 2147483647]. Support provided only for Wormhole_B0.

            Input tensor must have INT32 data type.

            Output tensor will have INT32 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");

        m_tensor.def("rsub",
        [](const Tensor& input,
            float value,
            const MemoryConfig& output_mem_config,
            std::optional<Tensor> output_tensor,
            uint8_t queue_id){
                return rsub(queue_id, input, value, output_mem_config, output_tensor);
            },
            py::arg("input"),
            py::arg("value"),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Returns tensor  with respective elements of the input tensor ``input`` subtracted from the ``value``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor square operation is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional Output Tensor", "Tensor", "Default value is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

        )doc");


        m_tensor.def("rsub",
        [](const Tensor& input,
            float value,
            const MemoryConfig& output_mem_config,
            std::optional<Tensor> output_tensor,
            uint8_t queue_id){
                return rsub(queue_id, input, value, output_mem_config, output_tensor);
            },
            py::arg("input"),
            py::arg("value"),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Returns tensor  with respective elements of the input tensor ``input`` subtracted from the ``value``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor square operation is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional Output Tensor", "Tensor", "Default value is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

        )doc");

        m_tensor.def("bitwise_and",bitwise_and,
            py::arg("input").noconvert(),py::arg("value"),py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
            Computes bitwise_and of input tensor ``input`` by  a scalar ``value``. Input tensor needs to be positive. Support provided only for Wormhole_B0.

            Input tensor must have INT32 data type.

            Output tensor will have INT32 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "value", "Scalar value", "int", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");

        m_tensor.def("bitwise_or",bitwise_or,
            py::arg("input").noconvert(),py::arg("value"),py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
            Computes bitwise_or of input tensor ``input`` by  a scalar ``value``. Input tensor needs to be positive. Support provided only for Wormhole_B0.

            Input tensor must have INT32 data type.

            Output tensor will have INT32 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "value", "scalar value", "int", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");

        m_tensor.def("right_shift",right_shift,
            py::arg("input").noconvert(),py::arg("shift_amt"),py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
            Computes right shift of input tensor ``input`` by ``shift_amt`` bits. ``shift_amt`` range must be [0, 31]. Support provided only for Wormhole_B0.

            Input tensor must have INT32 data type.

            Output tensor will have INT32 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "shift_amt", "Number of shift bits", "int", "[0, 31]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");
        m_tensor.def("left_shift",left_shift,
            py::arg("input").noconvert(),py::arg("shift_amt"),py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
            Computes left shift of input tensor ``input`` by ``shift_amt`` bits. ``shift_amt`` range must be [0, 31]. Support provided only for Wormhole_B0.

            Input tensor must have INT32 data type.

            Output tensor will have INT32 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "shift_amt", "Number of shift bits", "int", "[0, 31]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");

        detail::bind_unary_op_with_param(
            m_tensor, "unary_remainder", unary_remainder,
            py::arg("value"),
            R"doc(Perform an eltwise-modulus operation on ``{0}`` and ``{1}``. Formula : ``a - a.div(b, rounding_mode="floor") * b`` . Support provided only for WH_B0.)doc",
            R"doc("value", "float", "")doc"

        );
        detail::bind_unary_op_with_param(
            m_tensor, "unary_fmod", unary_fmod,
            py::arg("value"),
            R"doc(Perform an eltwise-fmod operation on ``{0}`` and ``{1}``. Formula : ``a - a.div(b, rounding_mode="trunc") * b`` . Support provided only for WH_B0.)doc",
            R"doc("value", "float", "")doc"

        );
        detail::bind_unary_op_with_param(
            m_tensor, "unary_ne", unary_ne,
            py::arg("value"),
            R"doc(Perform an eltwise-unary not-equal (``{0} != {1}``) on input tensor.)doc",
            R"doc("value", "float", "")doc"

        );
        detail::bind_unary_op_with_param(
            m_tensor, "rdiv", rdiv,
            py::arg("denominator"),
            R"doc(Returns tensor  with value ``{1}`` divided by each of respective elements of the input tensor ``{0}``.)doc",
            R"doc("denominator value which is actually calculated as numerator", "float", ">=0.0")doc"
        );

        detail::bind_unary_op_with_param(
            m_tensor, "prelu", prelu,
            py::arg("weight"),
            R"doc(Returns tensor with the prelu of all of elements of the input tensor ``{0}`` with negative slope as ``{1}``.)doc",
            R"doc("weight value", "float", "")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "unary_chain", &unary_chain,
            py::arg("unary_chain"),
            R"doc(Returns tensor with the unary op chain applied to all of elements of the input tensor ``{0}``.)doc",
            R"doc("Unary op chain", "Vector<FusibleActivation>", "At least 1 activation")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "unary_gt", unary_gt,
            py::arg("value"),
            R"doc(Perform an eltwise-unary greater-than (``{0} > {1}``) on input tensor.)doc",
            R"doc("value", "float", "")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "unary_lt", unary_lt,
            py::arg("value"),
            R"doc(Perform an eltwise-unary less-than (``{0} < {1}``) on input tensor.)doc",
            R"doc("value", "float", "")doc"
        );

        // *** bcast binary tied to unary ***
        detail::bind_unary_op(m_tensor, "add1", &add1, R"doc(Returns tensor with the addition of one with input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "deg2rad", &deg2rad, R"doc(Returns tensor with the deg2rad conversion of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "rad2deg", &rad2deg, R"doc(Returns tensor with the rad2deg conversion of elements of the input tensor ``{0}``.)doc");


        m_tensor.def("mul_unary",
        [](float value,
            const Tensor& input_tensor,
            const MemoryConfig& output_mem_config,
            std::optional<Tensor> output_tensor,
            uint8_t queue_id){
                return mul_unary(queue_id, value, input_tensor, output_mem_config, output_tensor);
            },
            py::arg("scalar"),
            py::arg("input"),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Perform an eltwise-binary mul on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "scalar", "Scalar", "float", "", "Yes"
                "input", "Tensor to mul", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional Output Tensor", "Tensor", "Default value is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

        )doc");

        m_tensor.def("mul_unary",
        [](const Tensor& input_tensor,
            float value,
            const MemoryConfig& output_mem_config,
            std::optional<Tensor> output_tensor,
            uint8_t queue_id){
                return mul_unary(queue_id, input_tensor, value, output_mem_config, output_tensor);
            },
            py::arg("input"),
            py::arg("scalar"),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Perform an eltwise-binary mul on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor to mul", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "scalar", "Scalar", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional Output Tensor", "Tensor", "Default value is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

        )doc");

        m_tensor.def("div_unary",
        [](float value,
            const Tensor& input_tensor,
            const MemoryConfig& output_mem_config,
            std::optional<Tensor> output_tensor,
            uint8_t queue_id){
                return div_unary(queue_id, value, input_tensor, output_mem_config, output_tensor);
            },
            py::arg("scalar"),
            py::arg("input"),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Perform an eltwise-binary div on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "scalar", "Scalar", "float", "", "Yes"
                "input", "Tensor to div", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional Output Tensor", "Tensor", "Default value is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

        )doc");

        m_tensor.def("div_unary",
        [](const Tensor& input_tensor,
            float value,
            const MemoryConfig& output_mem_config,
            std::optional<Tensor> output_tensor,
            uint8_t queue_id){
                return div_unary(queue_id, input_tensor, value, output_mem_config, output_tensor);
            },
            py::arg("input"),
            py::arg("scalar"),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Perform an eltwise-binary div on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor to div", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "scalar", "Scalar", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional Output Tensor", "Tensor", "Default value is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

        )doc");

    m_tensor.def(
        "sub_unary",
        py::overload_cast<float, const Tensor &, const MemoryConfig &>(&sub_unary),
        py::arg("scalar"),
        py::arg("input"),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
            Perform an eltwise-binary sub on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "scalar", "Scalar", "float", "", "Yes"
                "input", "Tensor to sub", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");

        m_tensor.def("add_unary",
        [](float value,
            const Tensor& input_tensor,
            const MemoryConfig& output_mem_config,
            std::optional<Tensor> output_tensor,
            uint8_t queue_id){
                return add_unary(queue_id, value, input_tensor, output_mem_config, output_tensor);
            },
            py::arg("scalar"),
            py::arg("input"),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Perform an eltwise-binary add on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "scalar", "Scalar", "float", "", "Yes"
                "input", "Tensor to add", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional Output Tensor", "Tensor", "Default value is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

        )doc");

        m_tensor.def("add_unary",
        [](const Tensor& input_tensor,
            float value,
            const MemoryConfig& output_mem_config,
            std::optional<Tensor> output_tensor,
            uint8_t queue_id){
                return add_unary(queue_id, input_tensor, value, output_mem_config, output_tensor);
            },
            py::arg("input"),
            py::arg("scalar"),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Perform an eltwise-binary add on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor to add", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "scalar", "Scalar", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional Output Tensor", "Tensor", "Default value is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

        )doc");

    detail::bind_unary_op_with_param(
        m_tensor,
        "sub_unary",
        py::overload_cast<const Tensor &, float, const MemoryConfig &>(&sub_unary),
        py::arg("scalar"),
        R"doc(Perform an eltwise-binary sub on one tensor ``{0}`` and one scalar ``{1}``.)doc",
        R"doc("Scalar", "float", "")doc");

    // softmax
    m_tensor.def(
        "softmax",
        &softmax,
        py::arg("input").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a softmax operation on the last tensor dimension.");

    // softmax with scale and mask, regular mask has a dim of (batch, 1, 1, seq_len), causal mask has a dim of (batch,
    // 1, seq_len, seq_len)
    m_tensor.def(
        "scale_mask_softmax",
        &transformers::scale_mask_softmax,
        py::arg("input").noconvert(),
        py::arg("scale"),
        py::arg("mask").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("is_causal_mask").noconvert() = false,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a fused scale->attention_mask->softmax operation.");
}
}  // namespace tt::tt_metal::detail
