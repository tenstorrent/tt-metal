// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "core/ttnn_all_includes.hpp"
#include "serialization/serializable.hpp"

// Make NamedParameters opaque so we can bind it explicitly
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/unordered_map.h>

#include "nb_autograd.hpp"
#include "nb_core.hpp"
#include "nb_export_enum.hpp"
#include "nb_models.hpp"
#include "nb_ops.hpp"
#include "nb_optimizers.hpp"

namespace ttml::nanobind {
using namespace ::nanobind;

NB_MODULE(_ttml, m) {
    // Bind NamedParameters as a proper map type at the top level
    nb::bind_map<ttml::serialization::NamedParameters>(m, "NamedParameters");

    ttml::nanobind::util::export_enum<tt::tt_metal::Layout>(m);
    auto m_autograd = m.def_submodule("autograd", "autograd");
    auto m_models = m.def_submodule("models", "models");
    auto m_modules = m.def_submodule("modules", "modules");
    auto m_ops = m.def_submodule("ops", "ops");
    auto m_core = m.def_submodule("core", "core");
    auto m_optimizers = m.def_submodule("optimizers", "optimizers");

    // TYPES
    ttml::nanobind::autograd::py_module_types(m_autograd);
    ttml::nanobind::models::py_module_types(m_models, m_modules);
    ttml::nanobind::ops::py_module_types(m_ops);
    ttml::nanobind::core::py_module_types(m_core);
    ttml::nanobind::optimizers::py_module_types(m_optimizers);

    // FUNCTIONS / OPERATIONS
    ttml::nanobind::autograd::py_module(m_autograd);
    ttml::nanobind::models::py_module(m_models, m_modules);
    ttml::nanobind::ops::py_module(m_ops);
    ttml::nanobind::core::py_module(m_core);
    ttml::nanobind::optimizers::py_module(m_optimizers);

    // Add MeshDevice as non-owning
    nb::class_<tt::tt_metal::distributed::MeshDevice>(m, "MeshDevice");
}

}  // namespace ttml::nanobind
