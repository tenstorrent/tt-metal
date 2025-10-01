// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nanobind/nb_autograd.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <core/ttnn_all_includes.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/graph.hpp"
#include "autograd/tensor.hpp"
#include "nanobind/nb_export_enum.hpp"
#include "nanobind/nb_util.hpp"
#include "ops/binary_ops.hpp"

namespace ttml::nanobind::autograd {
using namespace ttml::autograd;

void py_module_types(nb::module_& m) {
    ttml::nanobind::util::export_enum<ttnn::DataType>(m);
    ttml::nanobind::util::export_enum<GradMode>(m);
    ttml::nanobind::util::export_enum<PreferredPrecision>(m);

    nb::class_<AutoContext>(m, "AutoContext");
    nb::class_<AutocastTensor>(m, "AutocastTensor");
    nb::class_<Graph>(m, "Graph");
    nb::class_<GraphNode>(m, "GraphNode");
    nb::class_<Tensor>(m, "Tensor");
}

void py_module(nb::module_& m) {
    {
        auto py_graph_node = static_cast<nb::class_<GraphNode>>(m.attr("GraphNode"));
        py_graph_node.def(nb::init<>());
        py_graph_node.def_rw("grad_function", &GraphNode::grad_function);
    }

    {
        auto py_graph = static_cast<nb::class_<Graph>>(m.attr("Graph"));
        py_graph.def(nb::init<>());
        py_graph.def("get_edges", &Graph::get_edges);
        py_graph.def("get_graph_nodes", &Graph::get_graph_nodes);
        py_graph.def("add_node", &Graph::add_node);
    }

    {
        auto py_tensor = static_cast<nb::class_<Tensor>>(m.attr("Tensor"));
        py_tensor.def(nb::init<>());
        py_tensor.def(nb::init<const Tensor&>());
        py_tensor.def(nb::init<Tensor&&>());
        py_tensor.def(nb::init<const tt::tt_metal::Tensor&, bool>());
        py_tensor.def("set_value", &Tensor::set_value, nb::arg("value"));
        py_tensor.def("set_grad", &Tensor::set_grad, nb::arg("grad"));
        py_tensor.def("set_node", &Tensor::set_node, nb::arg("node"));
        py_tensor.def("clean_node", &Tensor::clean_node);
        py_tensor.def("add_grad", &Tensor::add_grad, nb::arg("grad"));
        py_tensor.def("set_requires_grad", &Tensor::set_requires_grad, nb::arg("requires_grad"));
        py_tensor.def("get_value", &Tensor::get_value, nb::arg("precision") = PreferredPrecision::HALF);
        py_tensor.def("get_grad", nb::overload_cast<>(&Tensor::get_grad, nb::const_));
        py_tensor.def("get_grad_rw", nb::overload_cast<>(&Tensor::get_grad));
        py_tensor.def("get_requires_grad", &Tensor::get_requires_grad);
        py_tensor.def("get_node", &Tensor::get_node);
        py_tensor.def("get_shape", &Tensor::get_shape);
        py_tensor.def("get_rank", &Tensor::get_rank);
        py_tensor.def("backward", &Tensor::backward, nb::arg("retain_graph"));
        py_tensor.def("is_grad_initialized", &Tensor::is_grad_initialized);
        py_tensor.def_static(
            "from_numpy",
            [](nb::ndarray<> numpy_tensor,
               tt::tt_metal::Layout layout,
               std::optional<tt::tt_metal::DataType> new_type) {
                return create_tensor(ttml::nanobind::util::make_metal_tensor(numpy_tensor, layout, new_type));
            },
            nb::arg("numpy_tensor"),
            nb::arg("layout") = tt::tt_metal::Layout::TILE,
            nb::arg("new_type") = std::nullopt);
        py_tensor.def(
            "to_numpy",
            [](const Tensor& tensor, std::optional<tt::tt_metal::DataType> new_type) {
                return ttml::nanobind::util::make_numpy_tensor(tensor.get_value(PreferredPrecision::FULL), new_type);
            },
            nb::arg("new_type") = std::nullopt);
        py_tensor.def("to_string", [](const Tensor& tensor) {
            return tensor.get_value(PreferredPrecision::FULL).write_to_string();
        });
        py_tensor.def("shape", [](const Tensor& tensor) {
            const tt::tt_metal::Shape& shape = tensor.get_shape();
            nb::list ret;
            for (auto it = shape.cbegin(); it != shape.cend(); ++it) {
                ret.append(*it);
            }
            return ret;
        });
        py_tensor.def("dtype", [](const Tensor& tensor) { return tensor.get_value(PreferredPrecision::FULL).dtype(); });
        py_tensor.def(
            "__add__",
            [](const TensorPtr& self, const AutocastTensor& other) { return ttml::ops::operator+(self, other); },
            nb::arg("other"),
            nb::is_operator());
        py_tensor.def(
            "__add__",
            [](const TensorPtr& self, const TensorPtr& other) { return ttml::ops::operator+(self, other); },
            nb::arg("other"),
            nb::is_operator());
        py_tensor.def(
            "__mul__",
            [](const TensorPtr& self, const TensorPtr& other) { return ttml::ops::operator*(self, other); },
            nb::arg("other"),
            nb::is_operator());
        py_tensor.def(
            "__mul__",
            [](const TensorPtr& self, float other) { return ttml::ops::operator*(self, other); },
            nb::arg("other"),
            nb::is_operator());
        py_tensor.def(
            "__sub__",
            [](const TensorPtr& self, const TensorPtr& other) { return ttml::ops::operator-(self, other); },
            nb::arg("other"),
            nb::is_operator());
        py_tensor.def(
            "__div__",
            [](const TensorPtr& self, const TensorPtr& other) { return ttml::ops::operator/(self, other); },
            nb::arg("other"),
            nb::is_operator());
    }

    {
        auto py_autocast_tensor = static_cast<nb::class_<AutocastTensor>>(m.attr("AutocastTensor"));
        py_autocast_tensor.def(nb::init<>());
        // py_autocast_tensor.def(nb::init<tt::tt_metal::Tensor&>());
        py_autocast_tensor.def(nb::init<const AutocastTensor&>());
        py_autocast_tensor.def(nb::init<AutocastTensor&&>());
        py_autocast_tensor.def("set_tensor", &AutocastTensor::set_tensor, nb::arg("tensor"));
        py_autocast_tensor.def("get_tensor", &AutocastTensor::get_tensor);
    }

    {
        auto py_auto_context = static_cast<nb::class_<AutoContext>>(m.attr("AutoContext"));
        py_auto_context.def_static("get_instance", &AutoContext::get_instance, nb::rv_policy::reference);
        py_auto_context.def("set_seed", &AutoContext::set_seed, nb::arg("seed"));
        py_auto_context.def("get_seed", &AutoContext::get_seed);
        py_auto_context.def(
            "add_backward_node", &AutoContext::add_backward_node, nb::arg("grad_function"), nb::arg("links"));
        py_auto_context.def("reset_graph", &AutoContext::reset_graph);
        py_auto_context.def("set_gradient_mode", &AutoContext::set_gradient_mode, nb::arg("mode"));
        py_auto_context.def("open_device", &AutoContext::open_device, nb::arg("mesh_shape"), nb::arg("device_ids"));
        py_auto_context.def("close_device", &AutoContext::close_device);
        py_auto_context.def("get_device", &AutoContext::get_device);
        // TODO: argv's char** not supported
        // py_auto_context.def("initialize_distributed_context", &AutoContext::initialize_distributed_context);
        py_auto_context.def(
            "initialize_distributed_context",
            [](AutoContext& auto_context, nb::args args) {
                const auto argc = args.size();
                std::vector<const char*> argv(argc);

                for (const auto& arg : args) {
                    argv.push_back(nb::str(arg).c_str());
                }
                argv.push_back(nullptr);

                auto_context.initialize_distributed_context(argc, const_cast<char**>(argv.data()));
            },
            nb::arg("args"));
        py_auto_context.def("get_distributed_context", &AutoContext::get_distributed_context);
        py_auto_context.def("get_profiler", &AutoContext::get_profiler);
        py_auto_context.def("close_profiler", &AutoContext::close_profiler);
        py_auto_context.def("get_ccl_resources", &AutoContext::get_ccl_resources);
    }
}

}  // namespace ttml::nanobind::autograd
