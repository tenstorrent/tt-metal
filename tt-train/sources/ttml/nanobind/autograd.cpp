// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/graph.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"

namespace ttml::autograd {

void py_module_types(nb::module_& m) {
    nb::enum_<GradMode>(m, "GradMode")
        .value("ENABLED", GradMode::ENABLED)
        .value("DISABLED", GradMode::DISABLED)
        .export_values();
    nb::enum_<PreferredPrecision>(m, "PreferredPrecision")
        .value("HALF", PreferredPrecision::HALF)
        .value("FULL", PreferredPrecision::FULL)
        .export_values();
    nb::enum_<RunMode>(m, "RunMode").value("TRAIN", RunMode::TRAIN).value("EVAL", RunMode::EVAL).export_values();

    nb::class_<GraphNode>(m, "GraphNode");
    nb::class_<Graph>(m, "Graph");
    nb::class_<ModuleBase>(m, "ModuleBase");
    nb::class_<Tensor>(m, "Tensor");
    nb::class_<AutocastTensor>(m, "AutocastTensor");
    nb::class_<AutoContext>(m, "AutoContext");
}

void py_module(nb::module_& m) {
    auto py_graph_node = static_cast<nb::class_<GraphNode>>(m.attr("GraphNode"));
    py_graph_node.def(nb::init<>());
    py_graph_node.def_rw("grad_function", &GraphNode::grad_function);

    auto py_graph = static_cast<nb::class_<Graph>>(m.attr("Graph"));
    py_graph.def(nb::init<>());
    py_graph.def("get_edges", &Graph::get_edges);
    py_graph.def("get_graph_nodes", &Graph::get_graph_nodes);
    py_graph.def("add_node", &Graph::add_node);

    auto py_module_base = static_cast<nb::class_<ModuleBase>>(m.attr("ModuleBase"));
    py_module_base.def(nb::init<>());
    py_module_base.def(nb::init<const ModuleBase&>());
    py_module_base.def(nb::init<ModuleBase&&>());
    py_module_base.def("get_name", &ModuleBase::get_name);
    py_module_base.def("parameters", &ModuleBase::parameters);
    py_module_base.def("train", &ModuleBase::train);
    py_module_base.def("eval", &ModuleBase::eval);
    py_module_base.def("set_run_mode", &ModuleBase::set_run_mode);

    auto py_tensor = static_cast<nb::class_<Tensor>>(m.attr("Tensor"));
    py_tensor.def(nb::init<const Tensor&>());
    py_tensor.def(nb::init<Tensor&&>());
    // py_tensor.def(nb::init<const tt::tt_metal::Tensor&, bool>());
    py_tensor.def("set_value", &Tensor::set_value);
    py_tensor.def("set_grad", &Tensor::set_grad);
    py_tensor.def("set_node", &Tensor::set_node);
    py_tensor.def("clean_node", &Tensor::clean_node);
    py_tensor.def("add_grad", &Tensor::add_grad);
    py_tensor.def("set_requires_grad", &Tensor::set_requires_grad);
    py_tensor.def("get_value", &Tensor::get_value);
    py_tensor.def("get_grad", nb::overload_cast<>(&Tensor::get_grad, nb::const_));
    py_tensor.def("get_grad_rw", nb::overload_cast<>(&Tensor::get_grad));
    py_tensor.def("get_requires_grad", &Tensor::get_requires_grad);
    py_tensor.def("get_node", &Tensor::get_node);
    py_tensor.def("get_shape", &Tensor::get_shape);
    py_tensor.def("get_rank", &Tensor::get_rank);
    py_tensor.def("backward", &Tensor::backward);
    py_tensor.def("is_grad_initialized", &Tensor::is_grad_initialized);

    auto py_autocast_tensor = static_cast<nb::class_<AutocastTensor>>(m.attr("AutocastTensor"));
    py_autocast_tensor.def(nb::init<>());
    // py_autocast_tensor.def(nb::init<const tt::tt_metal::Tensor&>());
    py_autocast_tensor.def(nb::init<const AutocastTensor&>());
    py_autocast_tensor.def(nb::init<AutocastTensor&&>());
    // py_autocast_tensor.def("set_tensor", &AutocastTensor::set_tensor);
    // py_autocast_tensor.def("get_tensor", &AutocastTensor::get_tensor);
    py_autocast_tensor.def("from_numpy", [](AutocastTensor& autocast_tensor, const nb::ndarray<>& data) {
        // TODO
        throw std::runtime_error("no impl");
    });
    py_autocast_tensor.def("to_numpy", [](const AutocastTensor& autocast_tensor) {
        // TODO
        throw std::runtime_error("no impl");
        // auto const & tensor = autocast_tensor.get_tensor(PreferredPrecision::FULL);

        // return nb::ndarray(
        //     tensor.buffer(),
    });

    auto py_auto_context = static_cast<nb::class_<AutoContext>>(m.attr("AutoContext"));
    py_auto_context.def_static("get_instance", &AutoContext::get_instance);
    py_auto_context.def("get_generator", &AutoContext::get_generator);
    py_auto_context.def("set_generator", &AutoContext::set_generator);
    py_auto_context.def("set_seed", &AutoContext::set_seed);
    py_auto_context.def("get_seed", &AutoContext::get_seed);
    py_auto_context.def("add_backward_node", &AutoContext::add_backward_node);
    py_auto_context.def("reset_graph", &AutoContext::reset_graph);
    py_auto_context.def("set_gradient_mode", &AutoContext::set_gradient_mode);
    py_auto_context.def("open_device", &AutoContext::open_device);
    py_auto_context.def("close_device", &AutoContext::close_device);
    py_auto_context.def(
        "initialize_distributed_context", [](AutoContext& auto_context, const std::vector<std::string>& args) {
            auto const argc = args.size();
            std::vector<char const*> argv(argc);

            for (auto const& arg : args) {
                argv.push_back(arg.c_str());
            }
            argv.push_back(nullptr);

            auto_context.initialize_distributed_context(argc, const_cast<std::remove_const_t<char**>>(argv.data()));
        });
    py_auto_context.def("get_distributed_context", &AutoContext::get_distributed_context);
    py_auto_context.def("get_profiler", &AutoContext::get_profiler);
    py_auto_context.def("close_profiler", &AutoContext::close_profiler);
    py_auto_context.def("get_ccl_resources", &AutoContext::get_ccl_resources);
}

}  // namespace ttml::autograd
