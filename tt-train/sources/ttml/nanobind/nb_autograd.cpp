// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nanobind/nb_autograd.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

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
    ttml::nanobind::util::export_enum<GradMode>(m);
    ttml::nanobind::util::export_enum<PreferredPrecision>(m);

    nb::class_<AutoContext>(m, "AutoContext");
    nb::class_<AutocastTensor>(m, "AutocastTensor");
    nb::class_<Graph>(m, "Graph");
    nb::class_<GraphNode>(m, "GraphNode");
    nb::class_<NodeId>(m, "NodeId");
    nb::class_<Tensor>(m, "Tensor");
    nb::class_<ParallelismContext>(m, "ParallelismContext");
    nb::class_<DistributedConfig>(m, "DistributedConfig");
}

void py_module(nb::module_& m) {
    {
        auto py_graph_node = static_cast<nb::class_<GraphNode>>(m.attr("GraphNode"));
        py_graph_node.def(nb::init<>());
        py_graph_node.def_rw("grad_function", &GraphNode::grad_function, "Get/set gradient function");
    }

    {
        auto py_graph = static_cast<nb::class_<Graph>>(m.attr("Graph"));
        py_graph.def(nb::init<>());
        py_graph.def("get_edges", &Graph::get_edges, "Get graph edges");
        py_graph.def("get_graph_nodes", &Graph::get_graph_nodes, "Get graph nodes");
        py_graph.def("add_node", &Graph::add_node, "Add graph node");
    }

    {
        auto py_tensor = static_cast<nb::class_<Tensor>>(m.attr("Tensor"));
        py_tensor.def(nb::init<>());
        py_tensor.def(nb::init<const Tensor&>());
        py_tensor.def(nb::init<Tensor&&>());
        py_tensor.def(nb::init<const tt::tt_metal::Tensor&, bool>());
        py_tensor.def("set_value", &Tensor::set_value, nb::arg("value"), "Set underlying tensor");
        py_tensor.def("set_grad", &Tensor::set_grad, nb::arg("grad"), "Set gradient");
        py_tensor.def(
            "set_grad_from_tensor",
            [](Tensor& self, const TensorPtr& grad_tensor) { self.set_grad(grad_tensor->get_value()); },
            nb::arg("grad_tensor"),
            "Set gradient from tensor");
        py_tensor.def("set_node", &Tensor::set_node, nb::arg("node"), "Set node");
        py_tensor.def("clean_node", &Tensor::clean_node, "Clean(unset) node");
        py_tensor.def("add_grad", &Tensor::add_grad, nb::arg("grad"), "Add to gradient");
        py_tensor.def(
            "set_requires_grad", &Tensor::set_requires_grad, nb::arg("requires_grad"), "Set gradient requirement flag");
        py_tensor.def(
            "get_value",
            &Tensor::get_value,
            nb::arg("precision") = PreferredPrecision::HALF,
            "Get underlying tensor value");
        py_tensor.def("get_grad", nb::overload_cast<>(&Tensor::get_grad, nb::const_), "Get gradient");
        py_tensor.def("get_grad_rw", nb::overload_cast<>(&Tensor::get_grad), "Get/set gradient");
        // Return gradient wrapped as TensorPtr for Python use (e.g., gradient clipping)
        py_tensor.def(
            "get_grad_tensor",
            [](const Tensor& self) -> TensorPtr {
                if (!self.is_grad_initialized()) {
                    return nullptr;
                }
                return create_tensor(self.get_grad());
            },
            "Get gradient as a Tensor object (for Python operations like gradient clipping)");
        py_tensor.def("get_requires_grad", &Tensor::get_requires_grad, "Get gradient requirement flag");
        py_tensor.def("get_node", &Tensor::get_node, "Get node");
        py_tensor.def("get_shape", &Tensor::get_shape, "Get shape");
        py_tensor.def("get_rank", &Tensor::get_rank, "Get rank");
        py_tensor.def(
            "backward", &Tensor::backward, nb::arg("retain_graph"), "Call gradient function on graph nodes in reverse");

        py_tensor.def("is_grad_initialized", &Tensor::is_grad_initialized, "Check if gradient is initialized");
        py_tensor.def(
            "assign",
            [](const TensorPtr& self, const TensorPtr& other) { self->set_value(other->get_value()); },
            nb::arg("other"));
        py_tensor.def_static(
            "from_numpy",
            [](nb::ndarray<nb::numpy> numpy_tensor,
               tt::tt_metal::Layout layout,
               std::optional<tt::tt_metal::DataType> new_type,
               ttnn::distributed::TensorToMesh* mapper) {
                return create_tensor(ttml::nanobind::util::make_metal_tensor(numpy_tensor, layout, new_type, mapper));
            },
            nb::arg("numpy_tensor"),
            nb::arg("layout") = tt::tt_metal::Layout::TILE,
            nb::arg("new_type") = std::nullopt,
            nb::arg("mapper") = nullptr,
            "Construct a Tensor from a numpy tensor");

        // Fallback: custom dtypes (like ml_dtypes.bfloat16)
        py_tensor.def_static(
            "from_numpy",
            [](nb::object numpy_tensor_obj,
               tt::tt_metal::Layout layout,
               std::optional<tt::tt_metal::DataType> new_type,
               ttnn::distributed::TensorToMesh* mapper) {
                return create_tensor(
                    ttml::nanobind::util::make_metal_tensor(numpy_tensor_obj, layout, new_type, mapper));
            },
            nb::arg("numpy_tensor"),
            nb::arg("layout") = tt::tt_metal::Layout::TILE,
            nb::arg("new_type") = std::nullopt,
            nb::arg("mapper") = nullptr,
            "Construct a Tensor from a numpy tensor with custom dtype");
        py_tensor.def(
            "to_numpy",
            [](const Tensor& tensor,
               std::optional<tt::tt_metal::DataType> new_type,
               ttnn::distributed::MeshToTensor* composer) {
                return ttml::nanobind::util::make_numpy_tensor(
                    tensor.get_value(PreferredPrecision::FULL), new_type, composer);
            },
            nb::arg("new_type") = std::nullopt,
            nb::arg("composer") = nullptr,
            "Construct a numpy tensor from a Tensor");
        py_tensor.def(
            "to_string",
            [](const Tensor& tensor) { return tensor.get_value(PreferredPrecision::FULL).write_to_string(); },
            "Return string representation of the Tensor");
        py_tensor.def(
            "shape",
            [](const Tensor& tensor) {
                const tt::tt_metal::Shape& shape = tensor.get_shape();
                nb::list ret;
                for (auto it = shape.cbegin(); it != shape.cend(); ++it) {
                    ret.append(*it);
                }
                return ret;
            },
            "Get Tensor shape as list");
        py_tensor.def(
            "dtype",
            [](const Tensor& tensor) { return tensor.get_value(PreferredPrecision::FULL).dtype(); },
            "Get Tensor data type");
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
        py_autocast_tensor.def(nb::init<const AutocastTensor&>());
        py_autocast_tensor.def(nb::init<AutocastTensor&&>());
        py_autocast_tensor.def("set_tensor", &AutocastTensor::set_tensor, nb::arg("tensor"), "Set underlying Tensor");
        py_autocast_tensor.def("get_tensor", &AutocastTensor::get_tensor, "Get underlying Tensor");
    }

    {
        auto py_auto_context = static_cast<nb::class_<AutoContext>>(m.attr("AutoContext"));
        py_auto_context.def_static(
            "get_instance", &AutoContext::get_instance, nb::rv_policy::reference, "Get singleton AutoContext instance");
        py_auto_context.def("set_seed", &AutoContext::set_seed, nb::arg("seed"), "Set seed");
        py_auto_context.def("get_seed", &AutoContext::get_seed, "Get seed");
        py_auto_context.def(
            "add_backward_node",
            [](AutoContext& self, GradFunction grad_function, std::optional<nb::list> links_obj) {
                // Handle empty list case where nanobind can't infer element type
                std::vector<NodeId> links;
                if (links_obj.has_value() && nb::len(*links_obj) > 0) {
                    links = nb::cast<std::vector<NodeId>>(*links_obj);
                }
                return self.add_backward_node(std::move(grad_function), links);
            },
            nb::arg("grad_function"),
            nb::arg("links"),
            "Add backward graph node");
        py_auto_context.def("reset_graph", &AutoContext::reset_graph, "Reset graph");
        py_auto_context.def("set_gradient_mode", &AutoContext::set_gradient_mode, nb::arg("mode"), "Set gradient mode");
        py_auto_context.def("get_gradient_mode", &AutoContext::get_gradient_mode, "Get gradient mode");
        py_auto_context.def(
            "open_device",
            [](AutoContext& self, nb::object mesh_shape_obj, nb::object device_ids_obj) {
                tt::tt_metal::distributed::MeshShape mesh_shape(1, 1);

                if (!mesh_shape_obj.is_none()) {
                    if (nb::isinstance<nb::list>(mesh_shape_obj) || nb::isinstance<nb::tuple>(mesh_shape_obj)) {
                        const auto dims = nb::cast<std::vector<int>>(mesh_shape_obj);
                        if (dims.size() != 2) {
                            throw std::runtime_error("mesh_shape must be a list/tuple of 2 integers: [rows, cols]");
                        }
                        mesh_shape = tt::tt_metal::distributed::MeshShape(dims[0], dims[1]);
                    } else {
                        mesh_shape = nb::cast<tt::tt_metal::distributed::MeshShape>(mesh_shape_obj);
                    }
                }

                std::vector<int> device_ids;
                if (!device_ids_obj.is_none()) {
                    device_ids = nb::cast<std::vector<int>>(device_ids_obj);
                }

                self.open_device(mesh_shape, device_ids);
            },
            nb::arg("mesh_shape") = nb::none(),
            nb::arg("device_ids") = nb::none(),
            "Open a mesh device");
        py_auto_context.def("close_device", &AutoContext::close_device, "Close mesh device");
        py_auto_context.def("get_device", &AutoContext::get_device, nb::rv_policy::reference, "Get mesh device");
        // TODO: argv's char** not supported
        py_auto_context.def(
            "initialize_distributed_context",
            [](AutoContext& auto_context, nb::args args) {
                const auto argc = args.size();

                std::vector<std::string> storage;
                storage.reserve(argc);
                for (const auto& arg : args) {
                    storage.emplace_back(nb::str(arg).c_str());
                }

                std::vector<char*> argv;
                argv.reserve(argc);
                for (auto& s : storage) {
                    argv.push_back(s.data());
                }
                argv.push_back(nullptr);

                auto_context.initialize_distributed_context(static_cast<int>(argc), argv.data());
            },
            nb::arg("args"),
            "Initialize distributed context");
        py_auto_context.def(
            "get_distributed_context",
            [](AutoContext& self) -> tt::tt_metal::distributed::multihost::DistributedContext* {
                return self.get_distributed_context().get();
            },
            nb::rv_policy::reference,
            "Get distributed context");
        py_auto_context.def("get_profiler", &AutoContext::get_profiler, "Get profiler");
        py_auto_context.def("close_profiler", &AutoContext::close_profiler, "Close profiler");
        py_auto_context.def("get_ccl_resources", &AutoContext::get_ccl_resources, "Get CCL resources");

        // Socket manager controls
        py_auto_context.def(
            "initialize_socket_manager",
            &AutoContext::initialize_socket_manager,
            nb::arg("socket_type"),
            "Initialize socket manager");
        py_auto_context.def(
            "get_socket_manager", &AutoContext::get_socket_manager, nb::rv_policy::reference, "Get socket manager");

        // Parallelism context controls
        py_auto_context.def(
            "initialize_parallelism_context",
            &AutoContext::initialize_parallelism_context,
            nb::arg("config"),
            "Initialize parallelism context with DistributedConfig");
        py_auto_context.def(
            "get_parallelism_context",
            [](AutoContext& self) -> const ParallelismContext& { return self.get_parallelism_context(); },
            nb::rv_policy::reference,
            "Get parallelism context");
        py_auto_context.def(
            "is_parallelism_context_initialized",
            &AutoContext::is_parallelism_context_initialized,
            "Check if parallelism context has been initialized");
    }

    {
        auto py_parallelism_context = static_cast<nb::class_<ParallelismContext>>(m.attr("ParallelismContext"));
        py_parallelism_context.def(
            "get_ddp_axis",
            [](const ParallelismContext& self) -> std::optional<uint32_t> { return self.get_ddp_axis(); },
            "Get DDP axis (mesh dimension for data parallelism)");
        py_parallelism_context.def(
            "get_tp_axis",
            [](const ParallelismContext& self) -> std::optional<uint32_t> { return self.get_tp_axis(); },
            "Get TP axis (mesh dimension for tensor parallelism)");
        py_parallelism_context.def("get_ddp_size", &ParallelismContext::get_ddp_size, "Get number of DDP devices");
        py_parallelism_context.def("get_tp_size", &ParallelismContext::get_tp_size, "Get number of TP devices");
        py_parallelism_context.def("is_ddp_enabled", &ParallelismContext::is_ddp_enabled, "Check if DDP is enabled");
        py_parallelism_context.def("is_tp_enabled", &ParallelismContext::is_tp_enabled, "Check if TP is enabled");
    }

    {
        auto py_distributed_config = static_cast<nb::class_<DistributedConfig>>(m.attr("DistributedConfig"));
        py_distributed_config.def(nb::init<>());
        py_distributed_config.def(nb::init<bool, bool>(), nb::arg("enable_ddp") = false, nb::arg("enable_tp") = false);
        py_distributed_config.def_rw("enable_ddp", &DistributedConfig::enable_ddp, "Enable data parallelism");
        py_distributed_config.def_rw("enable_tp", &DistributedConfig::enable_tp, "Enable tensor parallelism");
    }

    // Module-level create_tensor functions for creating autograd tensors
    m.def(
        "create_tensor",
        [](const tt::tt_metal::Tensor& value, bool requires_grad) -> TensorPtr {
            return create_tensor(value, requires_grad);
        },
        nb::arg("value"),
        nb::arg("requires_grad") = true,
        "Create an autograd Tensor from a tt::tt_metal::Tensor");

    m.def("create_tensor", []() -> TensorPtr { return create_tensor(); }, "Create an empty autograd Tensor");
}

}  // namespace ttml::nanobind::autograd
