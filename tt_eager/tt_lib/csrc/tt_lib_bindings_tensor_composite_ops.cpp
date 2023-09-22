// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"

namespace tt::tt_metal::detail{
    void TensorModuleCompositeOPs( py::module & m_tensor){

        m_tensor.def("outer", &outer,
            py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform a non-batched outer product multiplication ``arg0 x arg1`` with two tensors.

            Both input tensors must have BFLOAT16 data type but shape [1,1,N,1] and [1,1,1,M] respectively
            or reshapeable with only one major dimension while other 3 being squeezable dimensions.

            Output tensor will have BFLOAT16 data type but of shape [1,1,N,M].

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "First tensor to multiply", "Tensor", "Tensor of shape [1, 1, N, 1]", "Yes"
                "other", "Second tensor to multiply", "Tensor", "Tensor of shape [1, 1, 1, M]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("where", &where,
            py::arg("predicate"), py::arg("true_value"), py::arg("false_value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform an ternary where operation on two tensors based on third @predicate.

            where(predicate, true_value, false_value) implements (predicate) ? true_value : false_value.

            All three input tensors must have BFLOAT16 data type, and be of equal shape.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "predicate", "Predicate Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "true_value", "True Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "false_value", "False Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        // *** eltwise binary ***

        detail::bind_binary_op(m_tensor, "add", add, R"doc(Perform an eltwise-binary add (``{0} + {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "sub", sub, R"doc(Perform an eltwise-binary sub (``{0} - {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "mul", mul, R"doc(Perform an eltwise-binary mul (``{0} * {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "squared_difference", squared_difference, R"doc(Perform an eltwise-binary squared_difference (``{0} - {1}``)^2 on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "logical_and", logical_and, R"doc(Performs the element-wise logical AND of the given input tensors ``{0}`` && ``{1}``, Zeros are treated as False and nonzeros are treated as True.)doc");
        detail::bind_binary_op(m_tensor, "bias_gelu", bias_gelu, R"doc(Perform an eltwise-binary bias_gelu (``{0} + {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "gt", gt, R"doc(Perform an eltwise-binary greater-than (``{0} > {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "lt", lt, R"doc(Perform an eltwise-binary less-than (``{0} < {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "lte", lte, R"doc(Perform an eltwise-binary less-than-or-equal (``{0} <= {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "gte", gte, R"doc(Perform an eltwise-binary greater-than-or-equal (``{0} >= {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "eq", eq, R"doc(Perform an eltwise-binary equal (``{0} == {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "ne", ne, R"doc(Perform an eltwise-binary not-equal (``{0} != {1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "ldexp", ldexp, R"doc(Performs eltwise-binary ldexp (``{0} * 2**{1}``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "logaddexp", logaddexp, R"doc(Perform an eltwise-binary logaddexp (``log(exp({0}) + exp({1}))``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "logaddexp2", logaddexp2, R"doc(Perform an eltwise-binary logaddexp2 (``log2(2^({0}) + 2^({1}))``) on two tensors.)doc");
        detail::bind_binary_op(m_tensor, "logical_or", logical_or, R"doc(Perform an eltwise-binary logical OR (``{0} || {1}``) on two tensors.)doc");
        detail::bind_binary_op<false, true>(m_tensor, "max", &tt::tt_metal::max, R"doc(Perform an eltwise-binary max on two tensors.)doc");
        detail::bind_binary_op<false, true>(m_tensor, "min", &tt::tt_metal::min, R"doc(Perform an eltwise-binary min on two tensors.)doc");
        detail::bind_binary_op<false, true>(m_tensor, "hypot", &hypot, R"doc(Returns tensor with the hypot activation on elements of the input tensors ``{0}`` and ``{1}``.)doc");
        detail::bind_binary_op<false, true>(m_tensor, "xlogy", &xlogy, R"doc(Performs eltwise-binary xlogy (``{0} * log( {1} )``) on two tensors.)doc");
        detail::bind_binary_op<false, true>(m_tensor, "atan2", &atan2, R"doc(Returns tensor with the atan2 activation on elements of the input tensors ``{0}`` and ``{1}``.)doc");
        detail::bind_binary_op<false, true>(m_tensor, "logical_xor", &logical_xor, R"doc(Performs eltwise-binary logical_xor (``{0} ^ {1}``) on two tensors.)doc");

        detail::bind_binary_op(m_tensor, "add_without_autoformat", add_without_autoformat,
            R"doc(Perform an eltwise-binary add (``{0} + {1}``) on two tensors.

            Auto formatting is disabled. Both input tensors must have TILE layout. Output tensor will have TILE layout.)doc"
        );

        detail::bind_unary_op(m_tensor, "var_hw", tt::tt_metal::var_hw, R"doc(  Returns a new tensor with the variance of the input tensor ``{0}`` on H,W axes.)doc");
        detail::bind_unary_op(m_tensor, "std_hw", tt::tt_metal::std_hw, R"doc(Returns a new tensor with the standard deviation of the input tensor ``{0}`` on H,W axes.)doc");
        detail::bind_unary_op(m_tensor, "normalize_hw", tt::tt_metal::normalize_hw, R"doc(Returns a new tensor with the Gaussian normalize of the elements of the input tensor ``{0}`` on H,W axes.)doc");

        // *** eltwise unary ***

        detail::bind_unary_op(m_tensor, "exp", exp, R"doc(Returns a new tensor with the exponential of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "recip", recip, R"doc(Returns a new tensor with the reciprocal of the elements of the input tensor ``recip``.)doc");
        detail::bind_unary_op(m_tensor, "relu", relu, R"doc(Applies the rectified linear unit (ReLU) function to the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "relu6", relu6, R"doc(Returns tensor with the relu6 activation on elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "sqrt", sqrt, R"doc(Returns tensor with the square-root of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "sigmoid", sigmoid, R"doc(Applies the sigmoid function to the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "log", log, R"doc(Returns tensor with the natural logarithm of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "tanh", tanh, R"doc(Returns tensor with the hyperbolic tangent of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "log2", log2, R"doc(Returns tensor with the base 2 logarithm of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "log10", log10, R"doc(Returns tensor with the base 10 logarithm of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "sin", tt::tt_metal::sin, R"doc(Returns tensor with the sine of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "cos", tt::tt_metal::cos, R"doc(Returns tensor with the cosine of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "abs", abs, R"doc(Returns tensor with elementwise absolute value of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "isfinite", isfinite, R"doc(Returns boolean tensor that is True where input tensor ``{0}``, is finite and False elsewhere.)doc");
        detail::bind_unary_op(m_tensor, "isinf", isinf, R"doc(Returns boolean tensor that is True where input tensor ``{0}``, is infinite and False elsewhere.)doc");
        detail::bind_unary_op(m_tensor, "isposinf", isposinf, R"doc(Returns each element of input tensor ``{0}``, is positive infinity or not.)doc");
        detail::bind_unary_op(m_tensor, "isneginf", isneginf, R"doc(Returns each element of input tensor ``{0}``, is negative infinity or not.)doc");
        detail::bind_unary_op(m_tensor, "isnan", isnan, R"doc(Returns boolean tensor that is True where tensor ``{0}``, is NaN and False elsewhere.)doc");
        detail::bind_unary_op(m_tensor, "sign", sign, R"doc(Returns tensor with the elementwise signum of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "square", square, R"doc(Returns tensor with the square of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "eqz", eqz, R"doc(Returns tensor with the result of equal to zero of all of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "nez", nez, R"doc(Returns tensor with the not equal zero of all of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "gtz", gtz, R"doc(Returns tensor with the greater than zero of all of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "ltz", ltz, R"doc(Returns tensor with the less than zero of all of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "gez", gez, R"doc(Returns tensor with the greater than equal zero of all of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "lez", lez, R"doc(Returns tensor with the less than equal zero of all of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "exp2", exp2, R"doc(Returns a new tensor with the exp2 (2 power) of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "expm1", expm1,
            R"doc(Returns a new tensor with the expm1 of the elements of the input tensor ``{0}``.
            expm1 = exp(x) - 1)doc"
        );
        detail::bind_unary_op(m_tensor, "signbit", signbit, R"doc(Applies the signbit function to the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "atan", atan, R"doc(Returns a new tensor with the arctan of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "atanh", atanh, R"doc(Returns a new tensor with the inverse hyperbolic tangent of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "asin", asin, R"doc(Returns a new tensor with the arcsine of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "acos", acos, R"doc(Returns a new tensor with the arccosine of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "logical_not_unary", logical_not_unary, R"doc(Returns a new tensor with the logical not of the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "log_sigmoid", &log_sigmoid, R"doc(Applies the logsigmoid function to the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op_with_param(
            m_tensor, "glu", &glu,
        py::arg("dim") = -1,
            R"doc(Applies the Gated Linear Units (GLU) function to the elements of the input tensor ``{0}`` split along dim ``{1}``.)doc",
        R"doc(dimension to split)doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "geglu", &geglu,
        py::arg("dim") = -1,
            R"doc(Applies the Gaussian Error Gated Linear Units function to the elements of the input tensor ``{0}`` split along dim ``{1}``.)doc",
        R"doc(dimension to split)doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "reglu", &reglu,
            py::arg("dim") = -1,
            R"doc(Applies the Rectified Linear Gated Linear Units (ReGLU) function to the elements of the input tensor ``{0}`` split along dim ``{1}``.)doc",
        R"doc(dimension to split)doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "swiglu", &swiglu,
            py::arg("dim") = -1,
            R"doc(Applies the Swish Gated Linear Units (SwiGLU) function to the elements of the input tensor ``{0}`` split along dim ``{1}``.)doc",
        R"doc(dimension to split)doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "gelu", &gelu,
            py::arg("fast_and_approx") = true,
            R"doc(Applies the Gaussian Error Linear Units (GELU) function to the elements of the input tensor ``{0}``.)doc",
            R"doc("Indicate true for approx and fast mode; false for accurate and slow mode", "bool", "default of true")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "logical_andi", &logical_andi,
            py::arg("immediate"),
            R"doc(Perform an eltwise logical AND (``{0} && {1}``) on input tensor and immediate value.)doc",
            R"doc("Scalar", "float", "")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "erf", &erf,
            py::arg("fast_and_approx") = true,
            R"doc(Computes error function for all elements of the input tensor ``{0}``.)doc",
            R"doc("Indicate true for approx and fast mode; false for accurate and slow mode", "bool", "default of true")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "erfc", &erfc,
            py::arg("fast_and_approx") = true,
            R"doc(Computes complementary error function for all elements of the input tensor ``{0}``.)doc",
            R"doc("Indicate true for approx and fast mode; false for accurate and slow mode", "bool", "default of true")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "rsqrt", &rsqrt,
            py::arg("fast_and_approx") = true,
            R"doc(Returns a new tensor with the reciprocal of the square-root of each of the elements of the input tensor ``{0}``.)doc",
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
            m_tensor, "relu_min", relu_min,
            py::arg("lower_limit"),
            R"doc(Returns tensor with the relu min of all of elements of the input tensor ``{0}``. This is equivalent
            to relu_min[x] = max(x, ``{1}``). It moves relu function down to carry out operation at minvalue
            instead of the standard 0.)doc",
            R"doc("min value", "float", "")doc"

        );
        detail::bind_unary_op_with_param(
            m_tensor, "elu", elu,
            py::arg("alpha"),
            R"doc(Returns tensor with the elu activation of all of elements of the input tensor ``{0}`` and scale
            factor alpha as ``{1}``. ELU(x) = alpha*(exp(x) - 1) if x < 0 else x.)doc",
            R"doc("alpha value", "float", "")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "heaviside", heaviside,
            py::arg("value"),
            R"doc(Returns tensor with the Heaviside step function of all of elements of the input tensor ``{0}`` and value factor as ``{1}``.

            HEAVISIDE(x) = 0 if x < 0 , 1 if x > 0 , else value.)doc",
            R"doc("value", "float", "")doc"

        );
        detail::bind_unary_op_with_param(
            m_tensor, "power", power,
            py::arg("exponent"),
            R"doc(Returns tensor with the all of elements of the input tensor ``{0}`` raised to ``{1}``.)doc",
            R"doc("exponent value", "int", ">=0")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "leaky_relu", leaky_relu,
            py::arg("slope"),
            R"doc(Returns tensor with the leaky relu of all of elements of the input tensor ``{0}`` with negative slope as ``{1}``.)doc",
            R"doc("slope value", "float", "")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "unary_chain", &unary_chain,
            py::arg("unary_chain"),
            R"doc(Returns tensor with the unary op chain applied to all of elements of the input tensor ``{0}``.)doc",
            R"doc("Unary op chain", "Vector<FusibleActivation>", "At least 1 activation")doc"
        );

        detail::bind_unary_op(m_tensor, "relu_without_autoformat", &relu_without_autoformat,
            R"doc(Applies the rectified linear unit (ReLU) function to the elements of the input tensor ``{0}``.

            Auto formatting is disabled. Input tensor must have TILE layout. Output tensor will have TILE layout.)doc"
        );

        // *** composite unary ops ***
        detail::bind_unary_op(m_tensor, "sinh", &tt::tt_metal::sinh, R"doc(Returns tensor with the hyperbolic sine of elements of the input tensor ``{0}`` in range [-9,9] with high accuracy.)doc");
        detail::bind_unary_op(m_tensor, "cosh", &tt::tt_metal::cosh, R"doc(Returns tensor with the hyperbolic cosine of elements of the input tensor ``{0}`` in range [-9,9] with high accuracy.)doc");
        detail::bind_unary_op(m_tensor, "softsign", &softsign, R"doc(Applies the softsign function to the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "softplus", &softplus, R"doc(Returns tensor with the softplus activation of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "log1p", &log1p, R"doc(Returns tensor with the natural log of 1 added to all of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "silu", &silu, R"doc(Returns tensor with the silu all of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "swish", swish, R"doc(Returns tensor with the swish all of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "mish", &mish, R"doc(Returns tensor with the mish activation of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "cbrt", &cbrt, R"doc(Returns tensor with the cbrt activation of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "asinh", &asinh, R"doc(Returns tensor with the inverse hyperbolic sine of elements of the input tensor ``{0}`` in range [-1e-6, 1e6].
            for +input , output = asinh(input)
            for -input , output = -asinh(input))doc"
        );
        detail::bind_unary_op(m_tensor, "acosh", &acosh, R"doc(Returns tensor with the inverse hyperbolic cosine of elements of the input tensor ``{0}`` in range [-1e-6, 1e6].
            for  input > 1, output = acosh(input)
            for  input ==1, ouptut = 0
            for  input < 1, output =  nan)doc"
        );
        detail::bind_unary_op(m_tensor, "tanhshrink", &tanhshrink,
            R"doc(Applies tanh on the input tensor ``{0}`` and subtracted from the input tensor.

            ``tanhshrink(x) = x - tanh(x)``)doc"
        );

        detail::bind_unary_op_with_param(
            m_tensor, "softshrink", &softshrink,
            py::arg("lambda"),
            R"doc(Applies the softshrink function to the elements of the input tensor ``{0}`` between limits ``-{1}`` low and
            the ``+{1}`` high limits.)doc",
            R"doc("value limits (-lambda to +lambda)", "float", ">= 0")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "hardshrink", &hardshrink,
            py::arg("lambda"),
            R"doc(Applies the hardshrink function to the elements of the input tensor ``{0}`` between limits ``-{1}`` low and
            the ``+{1}`` high limits.)doc",
            R"doc("value limits (-lambda to +lambda)", "float", ">= 0")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "bias_gelu_unary", &bias_gelu_unary,
            py::arg("bias"),
            R"doc(Applies the Gelu activation function to the elements of the biased ``{1}`` input tensor ``{0}``.)doc",
            R"doc("value limits (-bias to +bias)", "float", ">= 0")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "polyval", &polyval,
            py::arg("coeffs"),
            R"doc(Returns tensor with the polyval of all of elements of the input tensor ``{0}`` with coefficients ``{1}``.)doc",
            R"doc("coefficients value with highest degree first", "List of float", "List size > 0")doc"
        );

        // *** bcast binary tied to unary ***
        detail::bind_unary_op(m_tensor, "add1", &add1, R"doc(Returns tensor with the addition of one with input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "neg", &neg, R"doc(Returns tensor with the negate all of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "deg2rad", &deg2rad, R"doc(Returns tensor with the deg2rad conversion of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "rad2deg", &rad2deg, R"doc(Returns tensor with the rad2deg conversion of elements of the input tensor ``{0}``.)doc");

        detail::bind_unary_op_with_param(
            m_tensor, "add_unary", py::overload_cast<const Tensor&, float, const MemoryConfig&>(&add_unary),
            py::arg("scalar"),
            R"doc(Perform an eltwise-binary add on one tensor ``{0}`` and one scalar ``{1}``.)doc",
            R"doc("Scalar", "float", "")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "logical_noti", &logical_noti,
            py::arg("immediate"),
            R"doc(Perform an eltwise logical NOT (``!{1}``) on immediate value.)doc",
            R"doc("immediate", "float", "")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "sub_unary", py::overload_cast<const Tensor&, float, const MemoryConfig&>(&sub_unary),
            py::arg("scalar"),
            R"doc(Perform an eltwise-binary sub on one tensor ``{0}`` and one scalar ``{1}``.)doc",
            R"doc("Scalar", "float", "")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "mul_unary", py::overload_cast<const Tensor&, float, const MemoryConfig&>(&mul_unary),
            py::arg("scalar"),
            R"doc(Perform an eltwise-binary mul on one tensor ``{0}`` and one scalar ``{1}``.)doc",
            R"doc("Scalar", "float", "")doc"
        );
        detail::bind_unary_op_with_param(
            m_tensor, "div_unary", py::overload_cast<const Tensor&, float, const MemoryConfig&>(&div_unary),
            py::arg("scalar"),
            R"doc(Perform an eltwise-binary div on one tensor ``{0}`` and one scalar ``{1}``.)doc",
            R"doc("Scalar", "float", "")doc"
        );

        detail::bind_unary_op_with_param(
            m_tensor, "logical_ori", &logical_ori,
            py::arg("immediate"),
            R"doc(Perform an eltwise logical OR (``{0} || {1}``) on input tensor and immediate value.)doc",
            R"doc("Scalar", "float", "")doc"
        );

        m_tensor.def("add_unary", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&add_unary),
            py::arg("scalar"), py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform an eltwise-binary add on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "scalar", "Scalar", "float", "", "Yes"
                "input", "Tensor to add", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("sub_unary", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&sub_unary),
            py::arg("scalar"), py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform an eltwise-binary sub on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "scalar", "Scalar", "float", "", "Yes"
                "input", "Tensor to sub", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");

        m_tensor.def("mul_unary", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&mul_unary),
            py::arg("scalar"), py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform an eltwise-binary mul on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "scalar", "Scalar", "float", "", "Yes"
                "input", "Tensor to mul", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");

        m_tensor.def("div_unary", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&div_unary),
            py::arg("scalar"), py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform an eltwise-binary div on one tensor and one scalar.

            Both inputs, the tensor and scalar, must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "scalar", "Scalar", "float", "", "Yes"
                "input", "Tensor to div", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

        )doc");

        m_tensor.def("hardtanh", &hardtanh,
            py::arg("input").noconvert(), py::arg("low") = -1.0f, py::arg("high") = +1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Applies the hard tanh function to the elements of the input tensor ``input``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor hardtanh is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "low", "Low value (PyTorch default)", "float", "default to -1.0f", "No"
                "high", "High value (PyTorch default)", "float", "default to +1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("clip", &clip,
            py::arg("input").noconvert(), py::arg("low"), py::arg("high"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Applies the clip function to the elements of the input tensor ``input`` between limits ``low`` low and
            the ``high`` high limits.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor hardtanh is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "low", "Low value)", "float", "", "Yes"
                "high", "High value", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


        m_tensor.def("hardsigmoid", &hardsigmoid,
            py::arg("input").noconvert(), py::arg("scale") = 1.0f/6.0f, py::arg("shift") = 0.5f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Applies the hardsigmoid function to the elements of the input tensor ``input``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor hardsigmoid is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "scale", "Scale value (PyTorch default)", "float", "default to 1.0/6.0f", "No"
                "shift", "Shift value (PyTorch default)", "float", "default to 0.5f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("lerp", py::overload_cast<const Tensor&, const Tensor&, float, const MemoryConfig&>(&lerp),
            py::arg("input").noconvert(), py::arg("end").noconvert(), py::arg("weight"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
            Applies the linear interpolation of two tensors ``arg0`` (given by input) and ``arg1`` based on a
            scalar ``arg2`` and returns the resulting out tensor.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor lerp is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "end", "End value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "weight", "Weight value", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("lerp", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&lerp),
            py::arg("input").noconvert(), py::arg("end").noconvert(), py::arg("weight").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Applies the linear interpolation of two tensors ``arg0`` (given by input) and ``arg1`` based on a
            tensor ``arg2`` and returns the resulting out tensor.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor lerp is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "end", "End value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "weight", "Weight value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("hardswish", &hardswish,
            py::arg("input").noconvert(), py::arg("scale") = 1.0f/6.0f, py::arg("shift") = 0.5f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Applies the hard swish function to the elements of the input tensor ``input``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor hardswish is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "scale", "Scale value (PyTorch default)", "float", "default to 1.0/6.0f", "No"
                "shift", "Shift value (PyTorch default)", "float", "default to 0.5f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("subalpha", &subalpha,
            py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Subtracts ``other``, scaled by ``alpha``, from ``input``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor subalpha is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("addalpha", &addalpha,
            py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Add ``other``, scaled by ``alpha``, from ``input``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor addalpha is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "other", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("full_like", &full_like,
            py::arg("input").noconvert(), py::arg("fill_value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor filled with the scalar value shaped like reference tensor ``arg0``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Reference Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "fill_value", "Fill value", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("zeros_like", &zeros_like,
            py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor filled with zeros shaped like reference tensor ``input``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Reference Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


        m_tensor.def("ones_like", &ones_like,
            py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor filled with ones shaped like reference tensor ``arg0``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Reference Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("zeros",
            [] (const std::array<uint32_t, 4> shape, Layout layout, Device * device, const MemoryConfig& output_mem_config) {
                return zeros(shape, layout, device, output_mem_config);
            },
            py::arg("shape"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor filled with zeros in shape specified by input ``shape``.

            Input shape is specified as a list of 4 integer elements

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
                "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
                "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("ones",
            [] (const std::array<uint32_t, 4> shape, Layout layout, Device * device, const MemoryConfig& output_mem_config) {
                return ones(shape, layout, device, output_mem_config);
            },
            py::arg("shape"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor filled with ones in shape specified by input ``shape``.

            Input shape is specified as a list of 4 integer elements

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
                "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
                "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("full",
            [] (const std::array<uint32_t, 4> shape, float value, Layout layout, Device * device, const MemoryConfig& output_mem_config) {
                return full(shape, value, layout, device, output_mem_config);
            },
            py::arg("shape"), py::arg("fill_value"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor filled with the scalar value in shape specified by input ``shape``.

            Input shape is specified as a list of 4 integer elements

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
                "fill_value", "Fill value ", "float", "", "Yes"
                "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
                "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("arange", &arange,
            py::arg("start"), py::arg("end"), py::arg("step"), py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new 1D tensor with the incremented values in size specified by inputs ``start``, ``end`` and ``step``.

            Inpute scalars are integers specifying start, end, and step sizes.
            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "start", "Start", "int", "", "Yes"
                "end", "End", "int", "> start", "Yes"
                "step", "Step", "int", "> 0", "Yes"
                "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    #if 0
        m_tensor.def("bitwise_complement", &bitwise_complement, R"doc(
            Returns tensor with the bitwise complement of elements of the input tensor ``arg0``.

            Input tensor must have UINT32 data type.

            Output tensor will have UINT32 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor bitwise complement |           |                              |          |
            |          | '~' is applied to         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");


        m_tensor.def("logical_not", &logical_not, R"doc(
            Returns tensor with the logical notof elements of the input tensor ``arg0``.

            Input tensor must have UINT32 data type.

            Output tensor will have UINT32 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor logical not        |           |                              |          |
            |          | '!' is applied to         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");
    #endif


    #if 0
        m_tensor.def("mean", &mean, R"doc(
            Returns tensor with the mean of elements of the input tensor ``arg0``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor mean is computed   | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");

        m_tensor.def("std", &tt::tt_metal::std, R"doc(
            Returns tensor with the std of elements of the input tensor ``arg0``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor std is computed on | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");

        m_tensor.def("normalize", &normalize, R"doc(
            Returns tensor with the normalization of elements of the input tensor ``arg0``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor std normalized     | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");
    #endif

        m_tensor.def("addcmul", &addcmul,
            py::arg("input").noconvert(), py::arg("tensor1").noconvert(), py::arg("tensor2").noconvert(), py::arg("value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs the element-wise multiplication of tensor1 ``tensor1`` by tensor2 ``tensor2``, multiplies the result
            by the scalar value ``value`` and adds it to input ``input``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor addcmul is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor1", "First Tensor to multiply", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor2", "Second tensor to multiply", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "value", "Value to be multiplied", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("addcdiv", &addcdiv,
            py::arg("input").noconvert(), py::arg("tensor1").noconvert(), py::arg("tensor2").noconvert(), py::arg("value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs the element-wise division of tensor1 ``tensor1`` by tensor2 ``tensor2``, multiplies the result
            by the scalar value ``value`` and adds it to input ``input``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor addcdiv is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor1", "Numerator Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor2", "Denominator Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "value", "Value to be multiplied", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");



        m_tensor.def("mac", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&mac),
            py::arg("input").noconvert(), py::arg("tensor1").noconvert(), py::arg("tensor2").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns tensor with the multiply and accumulation of all of elements of the input tensors ``input, tensor1, tensor2``.
            Output is ``input x tensor1 + tensor2`` elementwise operator.
            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor mac is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor1", "Tensor to be multiplied", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor2", "Tensor to be added", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("mac", py::overload_cast<const Tensor&, float, float, const MemoryConfig&>(&mac),
            py::arg("input").noconvert(), py::arg("float1"), py::arg("float2"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns tensor with the multiply and accumulation of all of elements of the input tensor ``input11 with``float1, float2``.
            Output is ``tensor1 x float1 + float2`` elementwise operator.
            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor mac is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "float1", "Value to be multiplied", "float", "", "Yes"
                "float2", "Value to be added", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("threshold", &threshold,
            py::arg("input").noconvert(), py::arg("threshold"), py::arg("value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns tensor with the threshold activation on elements of the input tensors ``arg0`` at threshold ``threshold``,
            and value ``value``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor threshold is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "threshold", "Value to threshold at", "float", "", "Yes"
                "value", "Value to replace with", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        detail::bind_unary_op_with_param(
            m_tensor, "logit", &logit,
            py::arg("eps"),
            R"doc(Returns a tensor that is a logit  of input tensor with shape ``[W, Z, Y, X]`` along clamp ``{1}``.)doc",
            R"doc("dimension to logit along", "int", "0, 1, 2, or 3")doc"
        );

        detail::bind_unary_op_with_param(
            m_tensor, "logical_xori", &logical_xori,
            py::arg("immediate"),
            R"doc(Perform an eltwise logical XOR (``{0} ^ {1}``) on input tensor and immediate value.)doc",
            R"doc("Scalar", "float", "")doc"
        );


    }





}
