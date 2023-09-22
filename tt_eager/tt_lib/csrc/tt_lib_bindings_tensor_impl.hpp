#pragma once

#include "tt_lib_bindings.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

namespace tt::tt_metal{

namespace detail {

// template<class T>
// struct DataTypeToFormatType {
//     using type = T;
// };

// template<>
// struct DataTypeToFormatType<bfloat16> {
//     using type = uint16_t;
// };

// template<class CppType, class DataType, class PyType>
// void implement_buffer_protocol(PyType& py_buffer_t) {
//     py_buffer_t
//         .def(
//             "__getitem__",
//             [](const CppType& self, std::size_t index) {
//                 return self[index];
//             }
//         )
//         .def(
//             "__len__",
//             [](const CppType& self) {
//                 return self.size();
//             }
//         )
//         .def(
//             "__iter__",
//             [](const CppType& self) {
//                 return py::make_iterator(self.begin(), self.end());
//             },
//             py::keep_alive<0, 1>()
//         )
//         .def_buffer(
//             [](CppType& self) -> py::buffer_info {
//                 using FormatType = typename DataTypeToFormatType<DataType>::type;
//                 return py::buffer_info(
//                     self.begin(),                                /* Pointer to buffer */
//                     sizeof(DataType),                            /* Size of one scalar */
//                     py::format_descriptor<FormatType>::format(), /* Python struct-style format descriptor */
//                     1,                                           /* Number of dimensions */
//                     { self.size() },                             /* Buffer dimensions */
//                     { sizeof(DataType) }                         /* Strides (in bytes) for each index */
//                 );
//             }
//         );
// };


template <bool mem_config_arg = true, typename Func, typename... Extra>
void bind_op_with_mem_config(py::module_ &module, std::string op_name, Func &&f, std::string docstring, Extra&&... extra) {
    if constexpr (mem_config_arg) {
        const std::string mem_config_name = "output_mem_config";
        docstring += fmt::format(R"doc(
            "{0}", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is {1} in {2}", "No")doc",
            mem_config_name, operation::DEFAULT_OUTPUT_MEMORY_CONFIG.interleaved ? "interleaved" : "non-interleaved", magic_enum::enum_name(operation::DEFAULT_OUTPUT_MEMORY_CONFIG.buffer_type)
        );
        module.def(op_name.c_str(), f,
            std::forward<Extra>(extra)..., py::arg(mem_config_name.c_str()).noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, docstring.c_str()
        );
    } else {
        module.def(op_name.c_str(), f,
            std::forward<Extra>(extra)..., docstring.c_str()
        );
    }
}

template <bool fused_activations = true, bool mem_config_arg = true, typename Func>
void bind_binary_op(py::module_ &module, std::string op_name, Func &&f, std::string op_desc) {
    std::vector<std::string> arg_name = {"input", "other"};
    op_desc = fmt::format(op_desc, arg_name[0], arg_name[1]);

    std::string docstring = fmt::format(R"doc(
        {0}

        Both input tensors must have BFLOAT16 data type, and be of equal shape.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{2}", "First tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "{3}", "Second tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        op_desc, op_name, arg_name[0], arg_name[1]
    );
    if constexpr (fused_activations) {
        const std::string fused_activations_name = "fused_activations";
        const std::optional<std::vector<UnaryWithParam>> default_fused_activations = std::nullopt;
        docstring += fmt::format(R"doc(
            "{0}", "Fused activations after binary computation", "List of FusibleActivation with optional param", "Default is None", "No")doc",
            fused_activations_name
        );
        bind_op_with_mem_config<mem_config_arg>(module, op_name, f, docstring,
            py::arg(arg_name[0].c_str()).noconvert(),
            py::arg(arg_name[1].c_str()).noconvert(),
            py::arg(fused_activations_name.c_str()) = default_fused_activations
        );

    } else {
        bind_op_with_mem_config<mem_config_arg>(module, op_name, f, docstring,
            py::arg(arg_name[0].c_str()).noconvert(),
            py::arg(arg_name[1].c_str()).noconvert()
        );
    }
}

//TODO @tt-aho: Update to handle variable number of params
template <bool mem_config_arg = true, typename Func>
void bind_unary_op(py::module_ &module, std::string op_name, Func &&f, std::string op_desc) {
    const std::string tensor_name = "input";
    op_desc = fmt::format(op_desc, tensor_name);
    std::string docstring = fmt::format(R"doc(
        {0}

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{1}", "Tensor {2} is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        op_desc, tensor_name, op_name
    );

    bind_op_with_mem_config<mem_config_arg>(module, op_name, f, docstring, py::arg(tensor_name.c_str()).noconvert());
}

template <bool mem_config_arg = true, typename Func, typename PyArg, typename std::enable_if<std::is_base_of<py::arg, PyArg>::value, int>::type = 0>
void bind_unary_op_with_param(py::module_ &module, std::string op_name, Func &&f, PyArg param, std::string op_desc, std::string param_desc) {
    const std::string tensor_name = "input";
    std::string param_name = std::string(param.name);
    op_desc = fmt::format(op_desc, tensor_name, param_name);
    const std::string required_param = std::is_same_v<py::arg_v, PyArg> ? "No" : "Yes";
    std::string docstring = fmt::format(R"doc(
        {0}

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{1}", "Tensor {2} is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        op_desc, tensor_name, op_name, param_desc
    );

    docstring += fmt::format(R"doc(
            "{0}", {1}, "{2}")doc",
        param_name, param_desc, required_param
    );
    bind_op_with_mem_config<mem_config_arg>(module, op_name, f, docstring, py::arg(tensor_name.c_str()).noconvert(), param);
}

template <typename E, typename... Extra>
py::enum_<E> export_enum(const py::handle &scope, std::string name = "", Extra&&... extra) {
    py::enum_<E> enum_type(scope, name.empty() ? magic_enum::enum_type_name<E>().data() : name.c_str(), std::forward<Extra>(extra)...);
    for (const auto &[value, name] : magic_enum::enum_entries<E>()) {
        enum_type.value(name.data(), value);
    }
    return enum_type;
}
}

}
