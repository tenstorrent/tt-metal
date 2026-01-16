// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_core.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <ttnn/distributed/distributed_tensor.hpp>
#include "nanobind/nb_export_enum.hpp"
#include "serialization/serializable.hpp"

// Make NamedParameters opaque - must be before unordered_map include
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/unordered_map.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/tensor.hpp"
#include "core/distributed/distributed.hpp"
#include "core/distributed/socket_manager.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace ttml::nanobind::core {

void py_module_types(nb::module_& m) {
    m.def_submodule("distributed");
    auto py_distributed = static_cast<nb::module_>(m.attr("distributed"));
    // Note: TensorToMesh, MeshToTensor, MeshComposerConfig are already registered by ttnn
    // They are imported and re-exported in the Python __init__.py
    // Expose SocketManager (ttml-specific type)
    nb::class_<ttml::core::distributed::SocketManager>(py_distributed, "SocketManager");
    // Expose SocketType enum (not exposed by ttnn)
    ttml::nanobind::util::export_enum<ttnn::distributed::SocketType>(py_distributed);
    // Expose multihost DistributedContext under core.distributed as a non-owning type (not exposed by ttnn)
    nb::class_<tt::tt_metal::distributed::multihost::DistributedContext>(py_distributed, "DistributedContext");
}

void py_module(nb::module_& m) {
    // Core utility functions
    m.def(
        "empty_like",
        [](const ttml::autograd::TensorPtr& tensor) -> ttml::autograd::TensorPtr {
            auto empty = ttnn::empty_like(tensor->get_value());
            return ttml::autograd::create_tensor(empty);
        },
        nb::arg("tensor"),
        "Create an empty tensor with the same shape and properties as the input tensor");

    {
        auto py_distributed = static_cast<nb::module_>(m.attr("distributed"));
        py_distributed.def("enable_fabric", &ttnn_fixed::distributed::enable_fabric);

        // Returns std::unique_ptr<TensorToMesh>
        py_distributed.def(
            "shard_tensor_to_mesh_mapper",
            &ttnn::distributed::shard_tensor_to_mesh_mapper,
            nb::arg("device"),
            nb::arg("rank"));

        // Returns std::unique_ptr<MeshToTensor> - composer for combining distributed tensors
        py_distributed.def(
            "concat_mesh_to_tensor_composer",
            &ttnn::distributed::concat_mesh_to_tensor_composer,
            nb::arg("mesh_device"),
            nb::arg("dim"));

        py_distributed.def(
            "create_mesh_composer",
            &ttnn::distributed::create_mesh_composer,
            nb::arg("mesh_device"),
            nb::arg("config"));
        py_distributed.def(
            "create_mesh_composer_config",
            [](nb::list dims, nb::list override) -> ttnn::distributed::MeshComposerConfig {
                ttsl::SmallVector<int> sdims;
                ttsl::SmallVector<uint32_t> soverride;
                for (nb::handle h : dims) sdims.push_back(nb::cast<int>(h));
                for (nb::handle h : override) soverride.push_back(nb::cast<int>(h));
                return ttnn::distributed::MeshComposerConfig(sdims, tt::tt_metal::distributed::MeshShape{soverride});
            },
            nb::arg("dims"),
            nb::arg("mesh_shape_override"));
        // Synchronize gradients across devices for DDP
        py_distributed.def(
            "synchronize_gradients", &ttml::core::distributed::synchronize_gradients, nb::arg("parameters"));

        // Bind DistributedContext methods
        using DistributedContext = tt::tt_metal::distributed::multihost::DistributedContext;
        auto py_dist_ctx = static_cast<nb::class_<DistributedContext>>(py_distributed.attr("DistributedContext"));
        py_dist_ctx.def("size", [](DistributedContext& self) { return *self.size(); });
        py_dist_ctx.def("rank", [](DistributedContext& self) { return *self.rank(); });
        py_dist_ctx.def("barrier", [](DistributedContext& self) { self.barrier(); });
        py_dist_ctx.def(
            "create_sub_context",
            [](DistributedContext& self, const std::vector<int>& ranks) {
                return self.create_sub_context(ttsl::Span<int>(const_cast<int*>(ranks.data()), ranks.size()));
            },
            nb::arg("ranks"));

        // Bind SocketManager methods
        auto py_socket_manager =
            static_cast<nb::class_<ttml::core::distributed::SocketManager>>(py_distributed.attr("SocketManager"));
        using SocketManager = ttml::core::distributed::SocketManager;
        using Rank = ttml::core::distributed::Rank;
        using SocketType = ttnn::distributed::SocketType;
        py_socket_manager.def(nb::init<SocketType>());
        py_socket_manager.def(
            "send",
            [](SocketManager& self,
               const ttml::autograd::Tensor& tensor,
               DistributedContext* distributed_ctx,
               int rank,
               bool use_grad) {
                // TODO: Refactor binding of DistributedContext so we don't need this hack
                std::shared_ptr<DistributedContext> ctx(distributed_ctx, [](DistributedContext*) {});
                if (use_grad) {
                    self.send(tensor.get_grad(), ctx, Rank{rank});
                } else {
                    self.send(tensor.get_value(), ctx, Rank{rank});
                }
            },
            nb::arg("tensor"),
            nb::arg("distributed_ctx"),
            nb::arg("rank"),
            nb::arg("use_grad") = false);
        py_socket_manager.def(
            "recv",
            [](SocketManager& self,
               ttml::autograd::Tensor& tensor,
               DistributedContext* distributed_ctx,
               int rank,
               bool use_grad) -> ttml::autograd::Tensor& {
                // TODO: Refactor binding of DistributedContext so we don't need this hack
                std::shared_ptr<DistributedContext> ctx(distributed_ctx, [](DistributedContext*) {});
                if (use_grad) {
                    if (!tensor.is_grad_initialized()) {
                        tensor.set_grad(ttnn::empty_like(tensor.get_value()));
                    }
                    auto filled = self.recv(tensor.get_grad(), ctx, Rank{rank});
                    tensor.set_grad(filled);
                } else {
                    auto filled = self.recv(tensor.get_value(), ctx, Rank{rank});
                    tensor.set_value(filled);
                }
                return tensor;
            },
            nb::arg("tensor"),
            nb::arg("distributed_ctx"),
            nb::arg("rank"),
            nb::arg("use_grad") = false,
            nb::rv_policy::reference);
    }
}

}  // namespace ttml::nanobind::core
