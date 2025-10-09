// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_core.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>

#include <ttnn/distributed/distributed_tensor.hpp>

#include "serialization/serializable.hpp"

// Make NamedParameters opaque - must be before unordered_map include
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/unordered_map.h>

#include <core/ttnn_all_includes.hpp>

#include "core/distributed/distributed.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace ttml::nanobind::core {

void py_module_types(nb::module_& m) {
    m.def_submodule("distributed");
    auto py_distributed = static_cast<nb::module_>(m.attr("distributed"));
    // Expose TensorToMesh so functions returning unique_ptr<TensorToMesh> can be converted
    nb::class_<ttnn::distributed::TensorToMesh>(py_distributed, "TensorToMesh");
    // Expose MeshToTensor composer for composing distributed tensors back to single tensor
    nb::class_<ttnn::distributed::MeshToTensor>(py_distributed, "MeshToTensor");
    nb::class_<ttnn::distributed::MeshComposerConfig>(py_distributed, "MeshComposerConfig");
}

void py_module(nb::module_& m) {
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
            "synchronize_parameters", &ttml::core::distributed::synchronize_parameters, nb::arg("parameters"));
    }
}

}  // namespace ttml::nanobind::core
