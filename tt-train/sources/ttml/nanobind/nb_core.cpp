// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_core.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <iomanip>
#include <sstream>
#include <ttnn/distributed/distributed_tensor.hpp>

#include "nanobind/nb_export_enum.hpp"
#include "serialization/serializable.hpp"

// Make NamedParameters opaque - must be before unordered_map include
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/unordered_map.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/tensor.hpp"
#include "core/clip_grad_norm.hpp"
#include "core/distributed/distributed.hpp"
#include "core/distributed/socket_manager.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "utils/memory_utils.hpp"

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

    // Utils submodule for memory tracking
    m.def_submodule("utils");
    auto py_utils = static_cast<nb::module_>(m.attr("utils"));

    // Bind DRAMUsage struct
    nb::class_<ttml::utils::DRAMUsage>(py_utils, "DRAMUsage")
        .def_ro("peak", &ttml::utils::DRAMUsage::peak, "Peak memory usage in bytes")
        .def_ro("total_allocations", &ttml::utils::DRAMUsage::total_allocations, "Total memory allocated in bytes")
        .def_ro(
            "total_deallocations", &ttml::utils::DRAMUsage::total_deallocations, "Total memory deallocated in bytes")
        .def(
            "__repr__",
            [](const ttml::utils::DRAMUsage& usage) {
                std::stringstream ss;
                ss << "DRAMUsage(peak=" << usage.peak << ", total_allocations=" << usage.total_allocations
                   << ", total_deallocations=" << usage.total_deallocations << ")";
                return ss.str();
            })
        .def("__str__", [](const ttml::utils::DRAMUsage& usage) {
            constexpr double MB = 1024.0 * 1024.0;
            std::stringstream ss;
            ss << "DRAM Usage:\n"
               << "  Peak:          " << std::fixed << std::setprecision(2) << usage.peak / MB << " MB\n"
               << "  Allocations:   " << usage.total_allocations / MB << " MB\n"
               << "  Deallocations: " << usage.total_deallocations / MB << " MB";
            return ss.str();
        });

    // Note: L1UsagePerCore (alias for ttnn::graph::PeakMemoryUsagePerCore) is already
    // registered by ttnn. Use ttnn.graph.PeakMemoryUsagePerCore in Python.
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

    // Gradient clipping
    m.def(
        "clip_grad_norm",
        [](const ttml::serialization::NamedParameters& parameters,
           float max_norm,
           float p_norm_type,
           bool error_if_nonfinite) -> ttml::autograd::TensorPtr {
            return ttml::core::clip_grad_norm(parameters, max_norm, p_norm_type, error_if_nonfinite);
        },
        nb::arg("parameters"),
        nb::arg("max_norm"),
        nb::arg("p_norm_type") = 2.0f,
        nb::arg("error_if_nonfinite") = false,
        "Clip gradients of parameters to a maximum norm. Returns the total norm after clipping.");

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

    // MemoryUsageTracker bindings under core.utils
    {
        auto py_utils = static_cast<nb::module_>(m.attr("utils"));

        // Create a MemoryUsageTracker submodule
        py_utils.def_submodule("MemoryUsageTracker", "Memory usage tracking utilities");
        auto py_tracker = static_cast<nb::module_>(py_utils.attr("MemoryUsageTracker"));

        // Note: RunMode enum is already registered by ttnn as ttnn.graph.RunMode
        // Users should use ttnn.graph.RunMode.NORMAL or ttnn.graph.RunMode.NO_DISPATCH

        // Internal holder for the non-movable ScopeGuard - allocated on heap
        struct ScopeGuardHolder {
            ttnn::ScopeGuard guard;

            explicit ScopeGuardHolder(tt::tt_metal::IGraphProcessor::RunMode mode) :
                guard(ttml::utils::MemoryUsageTracker::begin_capture(mode)) {
            }

            void release() {
                guard.release();
            }
        };

        // MemoryUsageGuard - Python-friendly wrapper using shared_ptr to heap-allocated holder
        struct MemoryUsageGuard {
            std::shared_ptr<ScopeGuardHolder> holder;

            explicit MemoryUsageGuard(tt::tt_metal::IGraphProcessor::RunMode mode) :
                holder(std::make_shared<ScopeGuardHolder>(mode)) {
            }

            MemoryUsageGuard() = default;

            void release() {
                if (holder) {
                    holder->release();
                }
            }
        };

        nb::class_<MemoryUsageGuard>(py_tracker, "MemoryUsageGuard")
            .def(
                nb::init<tt::tt_metal::IGraphProcessor::RunMode>(),
                nb::arg("mode") = tt::tt_metal::IGraphProcessor::RunMode::NORMAL)
            .def("release", &MemoryUsageGuard::release, "Release the guard without calling cleanup (end_capture/clear)")
            .def("__enter__", [](MemoryUsageGuard& self) -> MemoryUsageGuard& { return self; })
            .def("__exit__", [](MemoryUsageGuard& self, nb::object, nb::object, nb::object) {
                // On context exit, release the guard to prevent automatic cleanup
                // User is expected to call end_capture/print_memory_usage/clear manually
                self.release();
                return false;
            });

        py_tracker.def(
            "begin_capture",
            [](tt::tt_metal::IGraphProcessor::RunMode mode) { return MemoryUsageGuard(mode); },
            nb::arg("mode") = tt::tt_metal::IGraphProcessor::RunMode::NORMAL,
            R"doc(
            Begin capturing memory usage.

            Args:
                mode: Run mode for the graph processor (NORMAL or NO_DISPATCH).
                      Use NO_DISPATCH to measure memory usage of models that don't fit in device memory.

            Returns:
                A MemoryUsageGuard object. The guard will automatically call end_capture()
                and clear() when destroyed, unless release() is called first.

            Example:
                # Manual control (recommended for matching C++ behavior)
                guard = MemoryUsageTracker.begin_capture()
                # ... operations ...
                MemoryUsageTracker.snapshot("CHECKPOINT")
                # ... more operations ...
                MemoryUsageTracker.end_capture("FINAL")
                MemoryUsageTracker.print_memory_usage()
                MemoryUsageTracker.clear()
                guard.release()  # Prevent double cleanup

            Warning:
                Not thread safe.
            )doc");

        py_tracker.def(
            "end_capture",
            &ttml::utils::MemoryUsageTracker::end_capture,
            nb::arg("name") = ttml::utils::MemoryUsageTracker::kDefaultTraceName,
            R"doc(
            End capturing memory usage and store the trace with the given name.

            Args:
                name: The name to store the trace under (default: "END_TRACE")

            Note:
                If capture is not active, this function prints a warning.
                Not thread safe.
            )doc");

        py_tracker.def(
            "snapshot",
            &ttml::utils::MemoryUsageTracker::snapshot,
            nb::arg("name"),
            R"doc(
            Create a checkpoint: save current trace with given name and start a new capture.

            This function:
            1. Ends the current capture and saves the trace with the given name
            2. Starts a new capture session

            Args:
                name: The name for this checkpoint

            Raises:
                RuntimeError: If capture is not active

            Note:
                Not thread safe.
            )doc");

        py_tracker.def(
            "get_dram_usage",
            &ttml::utils::MemoryUsageTracker::get_dram_usage,
            nb::arg("name") = ttml::utils::MemoryUsageTracker::kDefaultTraceName,
            R"doc(
            Get DRAM usage of captured trace by name.

            Args:
                name: The name of the trace (default: "END_TRACE")

            Returns:
                DRAMUsage object with peak, total_allocations, and total_deallocations fields

            Raises:
                RuntimeError: If the named trace doesn't exist

            Note:
                Not thread safe.
            )doc");

        py_tracker.def(
            "get_dram_usage_all",
            &ttml::utils::MemoryUsageTracker::get_dram_usage_all,
            R"doc(
            Get DRAM usage of all captured traces.

            Returns:
                List of tuples (trace_name, DRAMUsage) in capture order

            Note:
                Not thread safe.
            )doc");

        py_tracker.def(
            "get_l1_usage",
            &ttml::utils::MemoryUsageTracker::get_l1_usage,
            nb::arg("name") = ttml::utils::MemoryUsageTracker::kDefaultTraceName,
            R"doc(
            Get L1 usage of captured trace by name.

            Args:
                name: The name of the trace (default: "END_TRACE")

            Returns:
                L1UsagePerCore object with peak_cb, peak_l1, and peak_total fields

            Raises:
                RuntimeError: If the named trace doesn't exist

            Note:
                Not thread safe.
            )doc");

        py_tracker.def(
            "get_l1_usage_all",
            &ttml::utils::MemoryUsageTracker::get_l1_usage_all,
            R"doc(
            Get L1 usage of all captured traces.

            Returns:
                List of tuples (trace_name, L1UsagePerCore) in capture order

            Note:
                Not thread safe.
            )doc");

        py_tracker.def(
            "get_trace_names",
            &ttml::utils::MemoryUsageTracker::get_trace_names,
            R"doc(
            Get all trace names in order they were captured.

            Returns:
                List of trace names

            Note:
                Not thread safe.
            )doc");

        py_tracker.def(
            "print_memory_usage",
            &ttml::utils::MemoryUsageTracker::print_memory_usage,
            R"doc(
            Print memory usage summary for all captured traces.

            Prints detailed information including:
            - Per-segment DRAM usage (peak, allocations, deallocations)
            - Cumulative DRAM usage (peak, current)
            - L1 usage per core (CB, buffer, total)

            Note:
                Not thread safe.
            )doc");

        py_tracker.def(
            "clear",
            &ttml::utils::MemoryUsageTracker::clear,
            R"doc(
            Clear all stored traces.

            Note:
                Not thread safe.
            )doc");
    }
}

}  // namespace ttml::nanobind::core
