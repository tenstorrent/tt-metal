// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/complex/complex_ops.hpp"
#include "tt_eager/tt_dnn/op_library/loss/loss_op.hpp"
#include "tt_eager/tt_dnn/op_library/optimizer/optimizer_ops.hpp"

namespace tt::tt_metal::detail{
    void TensorModuleCompositeOPs( py::module & m_tensor){

	m_tensor.def("repeat", &tt::tt_metal::repeat,
            py::arg("input"), py::arg("shape"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                    Returns a new tensor filled with repetition of input ``input`` tensor according to number of times specified in ``shape``. The rank of ``shape`` should be same as rank of tensor ``input_a`` and contain positive integer entries - a limitation in our implementation.

                    Output tensor will have BFLOAT16 data type.

                    .. csv-table::
                        :header: "Argument", "Description", "Data type", "Valid range", "Required"

                        "input", "Input tensor for which repetition is computed", "Tensor", "Tensor of any shape", "Yes"
                        "shape", "Shape value", "Shape", "The number of times to repeat this tensor along each dimension", "Yes"
                        "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                )doc");
	m_tensor.def("pow", py::overload_cast<const Tensor&, float, const MemoryConfig&>(&tt::tt_metal::pow),
		     py::arg("input"), py::arg("exponent"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                    Returns a new tensor filled with power of input ``input`` raised to value of ``exponent``.

                    Output tensor will have BFLOAT16 data type.

                    .. csv-table::
                        :header: "Argument", "Description", "Data type", "Valid range", "Required"

                        "input", "Input tensor for which power is computed", "Tensor", "Tensor of any shape", "Yes"
                        "exponent", "exponent value", "float", "positive floating point value", "Yes"
                        "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                )doc");
    m_tensor.def("pow", py::overload_cast<const Tensor&, int, const MemoryConfig&>(&tt::tt_metal::pow),
		     py::arg("input"), py::arg("exponent"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                    Returns a new tensor filled with power of input ``input`` raised to value of ``exponent``.

                    Output tensor will have BFLOAT16 data type.

                    .. csv-table::
                        :header: "Argument", "Description", "Data type", "Valid range", "Required"

                        "input", "Input tensor for which power is computed", "Tensor", "Tensor of any shape", "Yes"
                        "exponent", "exponent value", "integer", "positive integer value", "Yes"
                        "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                )doc");
        m_tensor.def("sfpu_eps", &tt::tt_metal::sfpu_eps,
                py::arg("shape"), py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                    Returns a new tensor filled with the machine epsilon value in shape specified by input ``shape``.

                    Input shape is specified as a list of 4 integer elements

                    Output tensor will have BFLOAT16 data type.

                    .. csv-table::
                        :header: "Argument", "Description", "Data type", "Valid range", "Required"
                        "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
                        "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
                        "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
                        "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                )doc");


        m_tensor.def("outer", &outer,
            py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform a non-batched outer product multiplication ``arg0 x arg1`` with two tensors.

            Both input tensors must have BFLOAT16 data type but shape [1,1,N,1] and [1,1,1,M] respectively
            or reshapeable with only one major dimension while other 3 being squeezable dimensions.

            Output tensor will have BFLOAT16 data type but of shape [1,1,N,M].

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_a", "First tensor to multiply", "Tensor", "Tensor of shape [1, 1, N, 1]", "Yes"
                "input_b", "Second tensor to multiply", "Tensor", "Tensor of shape [1, 1, 1, M]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("where", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&where),
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
        m_tensor.def("where", py::overload_cast<const Tensor&, float, const Tensor&, const MemoryConfig&>(&where),
            py::arg("predicate"), py::arg("true_value"), py::arg("false_value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform an ternary where operation on two tensors based on third @predicate.

            where(predicate, true_value, false_value) implements (predicate) ? true_value : false_value.

            All three input tensors must have BFLOAT16 data type, and be of equal shape.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "predicate", "Predicate Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "true_value", "float", "float", "float scalar", "Yes"
                "false_value", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");
        m_tensor.def("where", py::overload_cast<const Tensor&, const Tensor&, const float, const MemoryConfig&>(&where),
            py::arg("predicate"), py::arg("true_value"), py::arg("false_value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform an ternary where operation on two tensors based on third @predicate.

            where(predicate, true_value, false_value) implements (predicate) ? true_value : false_value.

            All three input tensors must have BFLOAT16 data type, and be of equal shape.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "predicate", "Predicate Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "true_value", "True Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "false_value", "float", "float", "float scalar", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");
        m_tensor.def("where", py::overload_cast<const Tensor&, const float, const float, const MemoryConfig&>(&where),
            py::arg("predicate"), py::arg("true_value"), py::arg("false_value"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Perform an ternary where operation on two tensors based on third @predicate.

            where(predicate, true_value, false_value) implements (predicate) ? true_value : false_value.

            All three input tensors must have BFLOAT16 data type, and be of equal shape.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "predicate", "Predicate Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "true_value", "float", "float", "Tensor of shape [W, Z, Y, X]", "Yes"
                "false_value", "float", "float", "float scalar", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");
        // *** composite unary ops ***
        detail::bind_unary_op(m_tensor, "normalize_hw", tt::tt_metal::normalize_hw, R"doc(Returns a new tensor with the Gaussian normalize of the elements of the input tensor ``{0}`` on H,W axes.)doc");
        detail::bind_unary_op(m_tensor, "normalize_global", tt::tt_metal::normalize_global, R"doc(Returns a new tensor with the Gaussian normalize of the elements of the input tensor ``{0}`` on N,C,H,W axes.)doc");
        detail::bind_unary_op(m_tensor, "var_hw", tt::tt_metal::var_hw, R"doc(  Returns a new tensor with the variance of the input tensor ``{0}`` on H,W axes.)doc");
        detail::bind_unary_op(m_tensor, "std_hw", tt::tt_metal::std_hw, R"doc(Returns a new tensor with the standard deviation of the input tensor ``{0}`` on H,W axes.)doc");
        detail::bind_unary_op(m_tensor, "sinh", &tt::tt_metal::sinh, R"doc(Returns tensor with the hyperbolic sine of elements of the input tensor ``{0}`` in range [-9,9] with high accuracy.)doc");
        detail::bind_unary_op(m_tensor, "cosh", &tt::tt_metal::cosh, R"doc(Returns tensor with the hyperbolic cosine of elements of the input tensor ``{0}`` in range [-9,9] with high accuracy.)doc");
        detail::bind_unary_op(m_tensor, "softsign", &softsign, R"doc(Applies the softsign function to the elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "softplus", &softplus, R"doc(Returns tensor with the softplus activation of elements of the input tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "log1p", &log1p, R"doc(Returns tensor with the natural log of 1 added to all of elements of the input tensor ``{0}``.)doc");
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
        detail::bind_unary_op(m_tensor, "digamma", &digamma, R"doc(Computes the logarithmic derivative of the gamma function on input tensor ``{0}`` for the input range 1 to inf.)doc");
        detail::bind_unary_op(m_tensor, "lgamma", &lgamma, R"doc(Computes the natural logarithm of the absolute value of the gamma function on the  ``{0}`` tensor for inputs greater than 0.)doc");
        detail::bind_unary_op(m_tensor, "multigammaln", &multigammaln, R"doc(Computes the multivariate log-gamma function with dimension 4 element-wise on the input tensor ``{0}`` for inputs greater than 1.5f.)doc");

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
            m_tensor, "logical_andi", &logical_andi,
            py::arg("immediate"),
            R"doc(Perform an eltwise logical AND (``{0} && {1}``) on input tensor and immediate value.)doc",
            R"doc("Scalar", "float", "")doc"
        );


        detail::bind_unary_op_with_param(
            m_tensor, "logical_noti", &logical_noti,
            py::arg("immediate"),
            R"doc(Perform an eltwise logical NOT (``!{1}``) on immediate value.)doc",
            R"doc("immediate", "float", "")doc"
        );

        detail::bind_unary_op_with_param(
            m_tensor, "rpow", rpow,
            py::arg("base"),
            R"doc(Returns tensor  raising ``{1}`` value to power of respective elements of the input exponent tensor ``{0}``.)doc",
            R"doc("base value", "float", ">0.0")doc"
        );

        detail::bind_unary_op_with_param(
            m_tensor, "logical_ori", &logical_ori,
            py::arg("immediate"),
            R"doc(Perform an eltwise logical OR (``{0} || {1}``) on input tensor and immediate value.)doc",
            R"doc("Scalar", "float", "")doc"
        );

        m_tensor.def("argmax", &argmax,
            py::arg("input").noconvert(), py::arg("dim"), py::arg("all") = false, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns the indices of the maximum value of elements in the ``input`` tensor
            If ``all`` is set to ``true`` irrespective of given dimension it will return the indices of maximum value of all elements in given ``input``

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor argmax is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "dim", "Dimension to perform argmax", "int", "", "Yes"
                "all", "Consider all dimension (ignores ``dim`` param)", "bool", "default to false", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("argmin", &argmin,
            py::arg("input").noconvert(), py::arg("dim"), py::arg("all") = false, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns the indices of the minimum value of elements in the ``input`` tensor
            If ``all`` is set to ``true`` irrespective of given dimension it will return the indices of minimum value of all elements in given ``input``

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor argmin is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "dim", "Dimension to perform argmin", "int", "", "Yes"
                "all", "Consider all dimension (ignores ``dim`` param)", "bool", "default to false", "No"
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

        m_tensor.def("isclose", &isclose,
            py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("rtol") = 1e-05f, py::arg("atol") = 1e-08f, py::arg("equal_nan") = false, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Applies the isclose function to the elements of the input tensor ``input_a`` and ``input_b``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            if equal_nan True, then two NaN s will be considered equal, else not equal.

            isclose(input_a, input_b, rtol, atol) = ∣input_a−input_B∣ ≤ atol+rtol×∣input_b∣.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_a", "Tensor isclose is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor isclose is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "rtol", "rtol value", "float", "default to 1e-05f", "No"
                "atol", "atol value", "float", "default to 1e-08f", "No"
                "equal_nan", "equal_nan value", "bool", "default to false", "No"
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
            py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Subtracts ``input_b``, scaled by ``alpha``, from ``input_a``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_a", "Tensor subalpha is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("addalpha", &addalpha,
            py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Add ``input_b``, scaled by ``alpha``, from ``input_a``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_a", "Tensor addalpha is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("repeat_interleave", &repeat_interleave,
            py::arg("input").noconvert(), py::arg("repeat"), py::arg("dim"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Repeated tensor which has the same shape as ``input``, except along the given axis.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor input is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "repeat", "Repeat value", "int", "1 to inf", "Yes"
                "dim", "dim value", "int", "0 to 2", "Yes"
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

        m_tensor.def("triu",
	     &triu, py::arg("input"), py::arg("diag") = 0
            , py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor with upper triangular elements of input with rest being zero.

            Input tensor will have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "tensor input to be upper triangular processed", "Tensor", "", "Yes"
                "diag", "diagonal to be chosen (default to 0)", "int32_t", "-dim to +dim (default to 0)", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("tril",
	    &tril, py::arg("input"), py::arg("diag") = 0
            , py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor with lower triangular elements of input with rest being zero.

            Input tensor will have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "tensor input to be lower triangular processed", "Tensor", "", "Yes"
                "diag", "diagonal to be chosen", "int32_t", "-dim to +dim (default to 0)", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("zeros", &zeros,
            py::arg("shape"), py::arg("data_type").noconvert() = DataType::BFLOAT16, py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor filled with zeros in shape specified by input ``shape``.

            Input shape is specified as a list of 4 integer elements

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
                "data_type", "Tensor data type", "DataType", "default is BFLOAT16", "No"
                "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
                "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("empty", &empty,
            py::arg("shape"), py::arg("data_type").noconvert() = DataType::BFLOAT16, py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new empty tensor (on device) in shape specified by input ``shape``.

            Input shape is specified as a list of 4 integer elements

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
                "data_type", "Tensor data type", "DataType", "default is BFLOAT16", "No"
                "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
                "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("ones", &ones,
            py::arg("shape"), py::arg("data_type").noconvert() = DataType::BFLOAT16, py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor filled with ones in shape specified by input ``shape``.

            Input shape is specified as a list of 4 integer elements

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
                "data_type", "Tensor data type", "DataType", "default is BFLOAT16", "No"
                "layout", "Tensor layout", "Layout", "default is ROW_MAJOR", "No"
                "device", "Device tensor is placed on", "Device", "default is None (on host)", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("full", &full,
            py::arg("shape"), py::arg("fill_value"), py::arg("data_type").noconvert() = DataType::BFLOAT16, py::arg("layout").noconvert() = Layout::ROW_MAJOR, py::arg("device") = nullptr, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a new tensor filled with the scalar value in shape specified by input ``shape``.

            Input shape is specified as a list of 4 integer elements

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "shape", "Shape vector", "Vector<int>", "[W, Z, Y, X]", "Yes"
                "fill_value", "Fill value ", "float", "", "Yes"
                "data_type", "Tensor data type", "DataType", "default is BFLOAT16", "No"
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

        m_tensor.def("lamb_optimizer", &lamb_optimizer,
            py::arg("data").noconvert(), py::arg("grad").noconvert(), py::arg("exp_avg").noconvert(), py::arg("exp_avg_sq").noconvert(), py::arg("beta1"), py::arg("beta2"), py::arg("step_size"), py::arg("eps"), py::arg("weight_decay"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns tensor with the threshold activation on elements of the input tensors ``arg0`` at threshold ``threshold``,
            and value ``value``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "data", "Tensor data is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "grad", "Tensor grad is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "exp_avg", "Tensor exp_avg is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "exp_avg_sq", "exp_avg_sq threshold is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "beta1", "Value to beta1 at", "float", "", "Yes"
                "beta2", "Value to beta2 with", "float", "", "Yes"
                "step_size", "Value to beta1 at", "float", "", "Yes"
                "eps", "Value to beta2 with", "float", "", "Yes"
                "weight_decay", "Value to beta1 at", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        detail::bind_unary_op_with_param(
            m_tensor, "logit", &logit,
            py::arg("eps"),
            R"doc(Returns a tensor that is a logit  of input tensor with shape ``[W, Z, Y, X]`` along clamp ``{1}``.)doc",
            R"doc("dimension to logit along", "int", "0, 1, 2, or 3")doc"
        );

        detail::bind_unary_op_with_param(
            m_tensor, "polygamma", &polygamma,
            py::arg("n"),
            R"doc(Returns a tensor that is a polygamma of input tensor where the range supports from 1 to 10 with shape ``[W, Z, Y, X]`` along n ``{1}``.)doc",
            R"doc("the order of the polygamma along", "int", "1 to 10")doc"
        );

        detail::bind_unary_op_with_param(
            m_tensor, "logical_xori", &logical_xori,
            py::arg("immediate"),
            R"doc(Perform an eltwise logical XOR (``{0} ^ {1}``) on input tensor and immediate value.)doc",
            R"doc("Scalar", "float", "")doc"
        );

        detail::bind_unary_op(m_tensor, "atanh", atanh, R"doc(Returns a new tensor with the inverse hyperbolic tangent of the elements of the input tensor ``{0}``.)doc");

        // *** complex operations ***
        detail::bind_unary_op(m_tensor, "angle", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::angle), R"doc(Returns elementwise angle of complex tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "real", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::real), R"doc(Returns real portion of complex tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "imag", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::imag), R"doc(Returns imag portion of complex tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "is_real", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::is_real), R"doc(Returns true if complex tensor ``{0}``  is real.)doc");
        detail::bind_unary_op(m_tensor, "is_imag", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::is_imag), R"doc(Returns true if complex tensor ``{0}``  is imaginary.)doc");
        detail::bind_unary_op(m_tensor, "complex_abs", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_abs), R"doc(Returns elementwise abs value of complex tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "conj", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::conj), R"doc(Returns elementwise complex conjugate of tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "complex_recip", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_recip), R"doc(Returns elementwise reciprocal of complex tensor ``{0}``.)doc");

        m_tensor.def("complex_mul", py::overload_cast<const Tensor&,const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_mul),
            py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an eltwise-binary multiplication ``input_a * input_b`` on two complex tensors.)doc");

        m_tensor.def("complex_div", py::overload_cast<const Tensor&,const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_div),
            py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an eltwise-binary divide ``input_a / input_b`` on two complex tensors.)doc");

        m_tensor.def("complex_add", py::overload_cast<const Tensor&,const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_add),
            py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an eltwise-binary addition ``input_a + input_b`` on two complex tensors.)doc");

        m_tensor.def("complex_sub", py::overload_cast<const Tensor&,const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_sub),
            py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an eltwise-binary subtraction ``input_a - input_b`` on two complex tensors.)doc");

        m_tensor.def("polar", py::overload_cast<const ComplexTensor&, const MemoryConfig&>(&tt::tt_metal::polar),
	    py::arg("input_a"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an polar to Cartesian transformation of the input.real(r), input.imag(theta) into x + i*y generating a type-2 complex tensor.)doc");

        m_tensor.def("polar", py::overload_cast<const Tensor&,const Tensor&, const MemoryConfig&>(&tt::tt_metal::polar),
	    py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an polar to Cartesian transformation of the input_a (r), input_b(theta) into x + i*y generating a type-2 complex tensor.)doc");

        detail::bind_binary_op<false, true, false>(m_tensor, "logical_xor", &logical_xor, R"doc(Performs eltwise-binary logical_xor (``{0} ^ {1}``) on two tensors.)doc");
        detail::bind_binary_op<false, true, false>(m_tensor, "max", &tt::tt_metal::max, R"doc(Perform an eltwise-binary max on two tensors.)doc");
        detail::bind_binary_op<false, true, false>(m_tensor, "min", &tt::tt_metal::min, R"doc(Perform an eltwise-binary min on two tensors.)doc");
        detail::bind_binary_op<false, true, false>(m_tensor, "hypot", &hypot, R"doc(Returns tensor with the hypot activation on elements of the input tensors ``{0}`` and ``{1}``.)doc");
        detail::bind_binary_op<false, true, false>(m_tensor, "scatter", &tt::tt_metal::scatter, R"doc(Performs scatter operation on elements of the input tensors ``{0}`` and ``{1}``,specifically to copy channel data.)doc");
        detail::bind_binary_op<false, true, false>(m_tensor, "xlogy", &xlogy, R"doc(Performs eltwise-binary xlogy (``{0} * log( {1} )``) on two tensors.)doc");
        detail::bind_binary_op<false, true, false>(m_tensor, "atan2", &atan2, R"doc(Returns tensor with the atan2 activation on elements of the input tensors ``{0}`` and ``{1}``.)doc");
        detail::bind_binary_op<false, true, false>(m_tensor, "nextafter", &nextafter, R"doc(Returns the next floating-point value after input_a towards input_b of the input tensors ``{0}`` and ``{1}``.)doc");

	    // *** type-2 complex operations in new submodule 'type2_complex' ***
        auto m_type2_cplx = m_tensor.def_submodule("complex", "Complex type2");
        py::class_<tt::tt_metal::ComplexTensor> pycplx_cls(m_type2_cplx, "ComplexTensor");

        pycplx_cls.def_property_readonly("real",&tt::tt_metal::ComplexTensor::real);
        pycplx_cls.def_property_readonly("imag",&tt::tt_metal::ComplexTensor::imag);
        pycplx_cls.def("deallocate",&tt::tt_metal::ComplexTensor::deallocate);

        m_tensor.def("complex_tensor",
		     [](Tensor& r, Tensor& i) -> tt::tt_metal::ComplexTensor {
		       return tt::tt_metal::ComplexTensor({r,i});
		     },
            py::arg("real"),
            py::arg("imag"),
	        R"doc(Create a complex tensor object from real and imag parts ``{0}`` and ``{1}``.)doc"
        );

        m_tensor.def("is_real",
		     py::overload_cast<const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::is_real),
            py::arg("input"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns boolean tensor if value of complex tensor ``{0}`` is real.)doc"
        );

        m_tensor.def("is_imag",
		    py::overload_cast<const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::is_imag),
            py::arg("input"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns boolean tensor if value of complex tensor ``{0}`` is imaginary.)doc"
        );

        m_tensor.def("complex_abs",
		    py::overload_cast<const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::complex_abs),
            py::arg("input"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns absolute value of complex tensor ``{0}``.)doc"
        );

        m_tensor.def("real",
		    py::overload_cast<const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::real),
            py::arg("input"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns real value of complex tensor ``{0}``.)doc"
        );

        m_tensor.def("imag",
		    py::overload_cast<const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::imag),
            py::arg("input"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns imaginary value of complex tensor ``{0}``.)doc"
        );

        m_tensor.def("angle",
		    py::overload_cast<const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::angle),
            py::arg("input"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns angle of a complex tensor ``{0}``.)doc"
        );

        m_tensor.def("conj",
		    py::overload_cast<const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::conj),
            py::arg("input"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns complex conjugate value of complex tensor ``{0}``.)doc"
        );

        m_tensor.def("complex_recip",
		    py::overload_cast<const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::complex_recip),
            py::arg("input"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns complex reciprocal value of complex tensor ``{0}``.)doc"
        );

        m_tensor.def("complex_add",
		    py::overload_cast<const ComplexTensor&,const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::complex_add),
            py::arg("input_a"),
            py::arg("input_b"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns addition of a complex tensor ``{0}`` with ``{1}``.)doc"
        );

        m_tensor.def("complex_sub",
            py::overload_cast<const ComplexTensor&,const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::complex_sub),
            py::arg("input_a"),
            py::arg("input_b"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns subtraction of a complex tensor ``{1}`` from ``{0}``.)doc"
        );

        m_tensor.def("complex_mul",
		    py::overload_cast<const ComplexTensor&,const ComplexTensor&,const MemoryConfig&>(tt::tt_metal::complex_mul),
            py::arg("input_a"),
            py::arg("input_b"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns addition of a complex multiplication of ``{0}`` and ``{1}``.)doc"
        );

        m_tensor.def("complex_div",
		    py::overload_cast<const ComplexTensor&,const ComplexTensor&, const MemoryConfig&>(tt::tt_metal::complex_div),
            py::arg("input_a"),
            py::arg("input_b"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns addition of a complex division of ``{0}`` by ``{1}``.)doc"
        );

        //loss functions
        m_tensor.def("mseloss",
		    py::overload_cast<const Tensor&,const Tensor&,const LossReductionMode,const MemoryConfig&>(tt::tt_metal::mseloss),
            py::arg("input_reference"),
            py::arg("input_prediction"),
            py::arg("reduce_mode"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns mean squared error loss function for ``{0}`` and ``{1}``.)doc"
        );

        m_tensor.def("maeloss",
		    py::overload_cast<const Tensor&,const Tensor&,const LossReductionMode,const MemoryConfig&>(tt::tt_metal::maeloss),
            py::arg("input_reference"),
            py::arg("input_prediction"),
            py::arg("reduce_mode"),
	        py::arg("output_mem_config").noconvert() = std::nullopt,
	        R"doc(Returns mean absolute error loss function for ``{0}`` and ``{1}``.)doc"
        );
    }
}
