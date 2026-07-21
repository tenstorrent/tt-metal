// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_core.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
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
#include "tt-metalium/distributed_context.hpp"
#include "ttnn/distributed/create_socket.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/full_like/full_like.hpp"
#include "ttnn/tensor/tensor.hpp"

// Make NamedParameters opaque - must be before unordered_map include
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/unordered_map.h>

#include "autograd/tensor.hpp"
#include "core/clip_grad_norm.hpp"
#include "core/distributed/distributed.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/tt_profiler.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"
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

    // TTProfiler binding
    nb::class_<ttml::core::TTProfiler>(m, "TTProfiler");

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

    m.def(
        "zeros_like",
        [](const tt::tt_metal::Tensor& tensor) -> tt::tt_metal::Tensor {
            return ttnn::moreh_full_like(tensor, 0.F, tensor.dtype(), tensor.layout(), tensor.memory_config());
        },
        nb::arg("tensor"),
        "Create a zero tensor with the same shape and properties as the input tensor");

    m.def(
        "ones_like",
        [](const tt::tt_metal::Tensor& tensor) -> tt::tt_metal::Tensor {
            return ttnn::moreh_full_like(tensor, 1.F, tensor.dtype(), tensor.layout(), tensor.memory_config());
        },
        nb::arg("tensor"),
        "Create a ones tensor with the same shape and properties as the input tensor");

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
        py_distributed.def(
            "disable_fabric",
            &ttnn_fixed::distributed::disable_fabric,
            "Tear down the process-global fabric config (SetFabricConfig(DISABLED)). "
            "Safe to call only when no devices are open; close the mesh device first.");

        // Returns std::unique_ptr<TensorToMesh>
        py_distributed.def(
            "shard_tensor_to_mesh_mapper",
            static_cast<std::unique_ptr<ttnn::distributed::TensorToMesh> (*)(
                ttnn::distributed::MeshDevice&, int, std::optional<int>)>(
                &ttnn::distributed::shard_tensor_to_mesh_mapper),
            nb::arg("device"),
            nb::arg("dim"),
            nb::arg("cluster_axis") = nb::none());

        py_distributed.def(
            "replicate_tensor_to_mesh_mapper",
            static_cast<std::unique_ptr<ttnn::distributed::TensorToMesh> (*)(ttnn::distributed::MeshDevice&)>(
                &ttnn::distributed::replicate_tensor_to_mesh_mapper),
            nb::arg("device"));

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
            "synchronize_gradients",
            static_cast<void (*)(const ttml::serialization::NamedParameters&)>(
                &ttml::core::distributed::synchronize_gradients),
            nb::arg("parameters"));
        py_distributed.def(
            "synchronize_gradients",
            static_cast<void (*)(const ttml::serialization::NamedParameters&, const std::vector<uint32_t>&)>(
                &ttml::core::distributed::synchronize_gradients),
            nb::arg("parameters"),
            nb::arg("cluster_axes"));

        // Raw (non-autograd) CCL collectives on ttnn tensors.
        // Unlike ttml.ops.distributed.* these do NOT register graph nodes, so they can be
        // freely invoked from FSDP pre/post hooks without polluting the autograd graph.
        py_distributed.def(
            "all_gather",
            &ttml::ttnn_fixed::distributed::all_gather,
            nb::arg("tensor"),
            nb::arg("dim"),
            nb::arg("cluster_axis") = nb::none(),
            "Raw all_gather without autograd tracking. Returns a new tt::tt_metal::Tensor.");
        py_distributed.def(
            "reduce_scatter",
            &ttml::ttnn_fixed::distributed::reduce_scatter,
            nb::arg("tensor"),
            nb::arg("dim"),
            nb::arg("cluster_axis") = nb::none(),
            "Raw reduce_scatter without autograd tracking. Returns a new tt::tt_metal::Tensor.");
        py_distributed.def(
            "all_reduce",
            &ttml::ttnn_fixed::distributed::all_reduce,
            nb::arg("tensor"),
            nb::arg("cluster_axis") = nb::none(),
            "Raw all_reduce without autograd tracking. Returns a new tt::tt_metal::Tensor.");

        // Bind DistributedContext methods
        using DistributedContext = tt::tt_metal::distributed::multihost::DistributedContext;
        using DistRank = tt::tt_metal::distributed::multihost::Rank;
        using DistTag = tt::tt_metal::distributed::multihost::Tag;
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
        // Byte-level point-to-point primitives — pure host-MPI, no device or
        // mesh-socket dependency. These are the operations the C++ training
        // stack uses underneath SocketManager (e.g. for the multi-host socket
        // descriptor handshake in mesh_socket_utils.cpp). Exposed here so
        // tests can validate them directly without going through MeshSocket,
        // which requires sender_mesh_id != receiver_mesh_id (not satisfied
        // on Galaxy single-mesh layouts).
        py_dist_ctx.def(
            "send",
            [](DistributedContext& self, nb::bytes data, int dest, int tag) {
                // DistributedContext::send takes Span<std::byte> (non-const),
                // but underlying MPI_Send treats it as readonly.
                auto* ptr = reinterpret_cast<std::byte*>(const_cast<char*>(data.c_str()));
                self.send(ttsl::Span<std::byte>(ptr, data.size()), DistRank{dest}, DistTag{tag});
            },
            nb::arg("data"),
            nb::arg("dest"),
            nb::arg("tag") = 0);
        py_dist_ctx.def(
            "recv",
            [](DistributedContext& self, std::size_t nbytes, int source, int tag) -> nb::bytes {
                std::vector<std::byte> buffer(nbytes);
                self.recv(ttsl::Span<std::byte>(buffer.data(), buffer.size()), DistRank{source}, DistTag{tag});
                return nb::bytes(reinterpret_cast<const char*>(buffer.data()), buffer.size());
            },
            nb::arg("nbytes"),
            nb::arg("source"),
            nb::arg("tag") = 0);

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

    // TTProfiler method bindings
    {
        auto py_profiler = static_cast<nb::class_<ttml::core::TTProfiler>>(m.attr("TTProfiler"));
        py_profiler.def("is_enabled", &ttml::core::TTProfiler::is_enabled, "Check if profiler is enabled");
        py_profiler.def("enable", &ttml::core::TTProfiler::enable, "Enable profiler");
        py_profiler.def("disable", &ttml::core::TTProfiler::disable, "Disable profiler");
        py_profiler.def(
            "get_naive_profiling",
            &ttml::core::TTProfiler::get_naive_profiling,
            "Check if TTML_NAIVE_PROFILER env var is set");
        py_profiler.def(
            "read_results",
            [](const ttml::core::TTProfiler& self,
               ttnn::distributed::MeshDevice& device,
               const std::string& noop_identifier,
               bool dump_results) {
                self.read_results(&device, noop_identifier, dump_results, 5U, tt::tt_metal::ProfilerReadState::NORMAL);
            },
            nb::arg("device"),
            nb::arg("noop_identifier") = "noop_identifier",
            nb::arg("dump_results") = false,
            "Insert a profiler marker. Synchronizes device and emits a timestamp (naive mode) "
            "or inserts noop markers for Tracy profiling. "
            "When dump_results is True, also flushes device profiling data to disk.");
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

    // DramFootprintTracker bindings under core.utils
    {
        auto py_utils = static_cast<nb::module_>(m.attr("utils"));

        nb::class_<ttml::utils::DramFootprint>(
            py_utils, "DramFootprint", "Peak DRAM footprint over a tracking window, bytes per device.")
            .def_ro("peak_allocated_bytes", &ttml::utils::DramFootprint::peak_allocated_bytes, "Highest usage reached.")
            .def_ro(
                "min_largest_free_bytes",
                &ttml::utils::DramFootprint::min_largest_free_bytes,
                "Largest single buffer still allocatable at the tightest point (the OOM-limiting contiguity).");

        py_utils.def(
            "dram_reserved_bytes",
            &ttml::utils::dram_reserved_bytes,
            R"doc(
            DRAM reserved outside the allocator arena, in bytes per device (physical - arena).
            A static device property (firmware base + any trace region), independent of footprint tracking.
            )doc");

        py_utils.def(
            "dram_arena_bytes",
            &ttml::utils::dram_arena_bytes,
            R"doc(
            DRAM arena (allocatable) size, in bytes per device -- the budget peak usage competes for.
            OOM is gated by this, not physical DRAM. physical = arena + reserved.
            )doc");

        py_utils.def_submodule(
            "DramFootprintTracker", "Zero-overhead peak DRAM footprint tracking via the real device allocator.");
        auto py_tracker = static_cast<nb::module_>(py_utils.attr("DramFootprintTracker"));

        py_tracker.def(
            "begin",
            &ttml::utils::DramFootprintTracker::begin,
            R"doc(
            Begin zero-overhead peak DRAM footprint tracking on the active device; resets the footprint.

            While active, every DRAM allocation samples the allocator on the allocation path itself
            (O(size classes) -- no op hooks, no capture buffers), so it measures the true peak DRAM
            footprint of the enclosed region at effectively zero cost, without perturbing op timings.
            Unlike MemoryUsageTracker this reads the real device allocator rather than an op-graph estimate.

            Raises:
                RuntimeError: if tracking is already active (not nestable).

            Note:
                Not thread safe. Prefer the ttml.track_dram_footprint() context manager.
            )doc");

        py_tracker.def(
            "footprint",
            &ttml::utils::DramFootprintTracker::footprint,
            R"doc(
            The DramFootprint since begin() (peak_allocated_bytes, min_largest_free_bytes), or zeros if not active.

            Note:
                Not thread safe.
            )doc");

        py_tracker.def(
            "end",
            &ttml::utils::DramFootprintTracker::end,
            R"doc(
            Stop tracking and return the final DramFootprint (zeros, with a warning, if not active).

            Note:
                Not thread safe.
            )doc");
    }
}

}  // namespace ttml::nanobind::core
