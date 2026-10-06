// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec_validation/validate_spec.hpp"
#include "core_descriptor.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/metal2_host_api/helpers.hpp"
#include "hostdev/remote_dfb_config_layout.h"  // PREFETCHER_PIPE_MAX_CREDIT_LANES
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"

namespace tt::tt_metal::experimental {

//////////////////////////////////////////////////
// Validate PrefetcherPipeParameters and relay DFBs
//////////////////////////////////////////////////
//
// Everything here is decidable from the spec alone (geometry and kernel placement); the pipe
// object arrives later via ProgramRunArgs and is reconciled against this geometry then.
//
// The spec never names a pipe's sender: that is the pipe object's (a consumer Program need not
// know it, and a DRAM-resident sender has no worker node to name). A Program that runs the sender
// kernel places it through a WorkUnitSpec; the supplied pipe's sender must be one of those nodes,
// which is checked when the pipe is bound.
//
// Rules per parameter:
//  1. Geometry: non-empty receivers, ring_size > 0, entry_size > 0, L1-aligned and <= ring_size.
// Rules per accessor group (one KernelAdvancedOptions::PrefetcherPipeBinding; its pipes share one device
// slot on every node the kernel runs on, so one binary serves them all):
//  2. The binding kernel is a data-movement kernel (compute reaches the ring via a relay DFB).
//  3. Tiling: the group's pipes agree on ring_size / entry_size and their receiver sets are
//     pairwise disjoint. The kernel's nodes equal EITHER the union of the receiver sets
//     (receiver role) OR avoid every receiver and number one per pipe (sender role); mixed or
//     partial coverage is rejected. Per pipe, at most one kernel plays sender and at most one
//     plays receiver, so exactly one kernel instance owns the credit counters on each node. Roles
//     may be split across Programs (sender op vs consumer op).
//  4. Receiver-side credit lanes P: the receiver kernel's num_threads (and, with a relay, the
//     relay's PRODUCER kernels' num_threads) must agree, fit the architecture's lane capacity,
//     and, when P > 1, divide the ring's entry count.
// Rules per relay DFB:
//  5. Not also borrowed_from. Every relayed pipe shares ring_size / entry_size; the DFB's
//     entry_size divides that entry_size (the relay may page one pipe entry as several pages, e.g.
//     a K-block as tiles, or one entry per consumer; only with a single-threaded producer) and
//     entry_size * num_entries is the pipe's
//     whole entries: ring_size rounded down to a multiple of the pipe's entry_size (the DFB is
//     exactly the ring the pipe uses; the pipe skips any trailing gap at the wrap).
//  6. The relayed pipes' receiver sets are pairwise disjoint and their union equals the DFB's
//     node set; every PRODUCER kernel binds exactly the relayed pipe set under one accessor (so
//     it is those pipes' receiver kernel and can drive the protocol the relay depends on).
void ValidatePrefetcherPipeSpec(const ProgramSpec& spec, const CollectedSpecData& collected, const Hal& hal) {
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const uint32_t lane_capacity = is_gen2_arch(hal) ? PREFETCHER_PIPE_MAX_CREDIT_LANES : 1u;

    std::unordered_map<PrefetcherPipeParamName, NodeRangeSet> pipe_receiver_set;
    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        // Rule 1: geometry.
        const NodeRangeSet receivers = to_node_range_set(pipe.receivers);
        TT_FATAL(receivers.num_cores() > 0, "PrefetcherPipeParameter '{}' has no receiver nodes", pipe.unique_id);
        TT_FATAL(pipe.ring_size > 0, "PrefetcherPipeParameter '{}' has ring_size = 0", pipe.unique_id);
        TT_FATAL(pipe.entry_size > 0, "PrefetcherPipeParameter '{}' has entry_size = 0", pipe.unique_id);
        TT_FATAL(
            pipe.entry_size % l1_alignment == 0,
            "PrefetcherPipeParameter '{}' entry_size {} must be a multiple of the L1 alignment ({})",
            pipe.unique_id,
            pipe.entry_size,
            l1_alignment);
        TT_FATAL(
            pipe.entry_size <= pipe.ring_size,
            "PrefetcherPipeParameter '{}' entry_size {} exceeds ring_size {}",
            pipe.unique_id,
            pipe.entry_size,
            pipe.ring_size);
        pipe_receiver_set.emplace(pipe.unique_id, receivers);
    }

    // Rules 2 and 3: per accessor group. Derive each group's role once, then record the
    // sender / receiver kernel of every pipe in it.
    std::unordered_map<PrefetcherPipeParamName, const KernelSpec*> sender_kernel_of;
    std::unordered_map<PrefetcherPipeParamName, const KernelSpec*> receiver_kernel_of;
    for (const auto& kernel : spec.kernels) {
        if (kernel.advanced_options.prefetcher_pipe_bindings.empty()) {
            continue;
        }
        TT_FATAL(
            kernel.is_data_movement_kernel(),
            "Kernel '{}' binds PrefetcherPipeParameter(s) (accessor '{}') but is a compute kernel. Only "
            "data-movement kernels bind a pipe; compute consumes through a relay DFB "
            "(DFBAdvancedOptions::prefetcher_pipe_relays).",
            kernel.unique_id,
            kernel.advanced_options.prefetcher_pipe_bindings[0].accessor_name);
        const NodeRangeSet& nodes = collected.kernel_node_set.at(kernel.unique_id);
        TT_FATAL(
            nodes.num_cores() > 0,
            "Kernel '{}' binds PrefetcherPipeParameter(s) but its WorkUnitSpecs place it on no nodes",
            kernel.unique_id);

        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            const PrefetcherPipeParameter* first =
                collected.prefetcher_pipe_by_name.at(binding.pipe_parameter_names[0]);
            NodeRangeSet group_receivers;
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                const PrefetcherPipeParameter* pipe = collected.prefetcher_pipe_by_name.at(pipe_name);
                TT_FATAL(
                    pipe->ring_size == first->ring_size && pipe->entry_size == first->entry_size,
                    "Kernel '{}' accessor '{}' names PrefetcherPipeParameters '{}' (ring_size {}, entry_size {}) "
                    "and '{}' (ring_size {}, entry_size {}); pipes sharing an accessor must share ring_size and "
                    "entry_size (one compiled kernel, one geometry)",
                    kernel.unique_id,
                    binding.accessor_name,
                    first->unique_id,
                    first->ring_size,
                    first->entry_size,
                    pipe->unique_id,
                    pipe->ring_size,
                    pipe->entry_size);
                const NodeRangeSet& receivers = pipe_receiver_set.at(pipe_name);
                TT_FATAL(
                    !receivers.intersects(group_receivers),
                    "Kernel '{}' accessor '{}' names PrefetcherPipeParameter '{}' whose receiver nodes overlap "
                    "another pipe's in the same accessor; pipes sharing an accessor must occupy disjoint nodes (one "
                    "pipe per node, so the accessor resolves to exactly one pipe on every node)",
                    kernel.unique_id,
                    binding.accessor_name,
                    pipe_name);
                group_receivers = group_receivers.merge(receivers);
            }

            // Role: the kernel's nodes are exactly the group's receivers, or one sender node per
            // pipe, none of them a receiver.
            const size_t num_pipes = binding.pipe_parameter_names.size();
            const bool is_receiver_role = is_prefetcher_pipe_receiver_role(nodes, group_receivers);
            const bool is_sender_role =
                !is_receiver_role && is_prefetcher_pipe_sender_role(nodes, group_receivers, num_pipes);
            if (!is_sender_role && !is_receiver_role) {
                const uint32_t on_receivers = nodes.intersection(group_receivers).num_cores();
                TT_THROW(
                    "Kernel '{}' accessor '{}' ({} pipe(s)): the kernel's WorkUnitSpec nodes must equal either the "
                    "union of the group's receiver nodes (receiver role) or be {} node(s) outside them, one per pipe "
                    "(sender role). Kernel covers {} node(s): {} of the {} receiver node(s), {} outside the "
                    "receivers. A role cannot be partial, mixed with the other role, or spill onto extra nodes.",
                    kernel.unique_id,
                    binding.accessor_name,
                    num_pipes,
                    num_pipes,
                    nodes.num_cores(),
                    on_receivers,
                    group_receivers.num_cores(),
                    nodes.num_cores() - on_receivers);
            }

            auto& role_map = is_sender_role ? sender_kernel_of : receiver_kernel_of;
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                auto [it, inserted] = role_map.try_emplace(pipe_name, &kernel);
                if (!inserted) {
                    TT_THROW(
                        "Kernels '{}' and '{}' both bind PrefetcherPipeParameter '{}' as its {}. Only one "
                        "data-movement kernel may own a pipe's {} credits.",
                        it->second->unique_id,
                        kernel.unique_id,
                        pipe_name,
                        is_sender_role ? "sender" : "receiver",
                        is_sender_role ? "sender" : "receiver");
                }
            }
        }
    }

    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        auto receiver_it = receiver_kernel_of.find(pipe.unique_id);
        const KernelSpec* receiver_kernel = receiver_it == receiver_kernel_of.end() ? nullptr : receiver_it->second;

        // Rule 4: receiver-side credit lanes. Sources: the receiver binding kernel and every
        // relay's PRODUCER kernels (uniform per role by the DFB checks above).
        std::optional<uint32_t> lanes;
        const KernelSpec* lanes_source = nullptr;
        auto take_lanes = [&](const KernelSpec* kernel) {
            if (!lanes.has_value()) {
                lanes = kernel->num_threads;
                lanes_source = kernel;
                return;
            }
            TT_FATAL(
                *lanes == kernel->num_threads,
                "PrefetcherPipeParameter '{}' receiver-side kernels disagree on thread count: '{}' has {} "
                "threads, '{}' has {}. The receiver kernel and every relay DFB producer must use the same "
                "num_threads (this is the pipe's credit lane count).",
                pipe.unique_id,
                lanes_source->unique_id,
                *lanes,
                kernel->unique_id,
                kernel->num_threads);
        };
        if (receiver_kernel != nullptr) {
            take_lanes(receiver_kernel);
        }
        // A pipe bound only by kernels (no relay DFB) has no entry.
        if (const auto relays_it = collected.prefetcher_pipe_relays.find(pipe.unique_id);
            relays_it != collected.prefetcher_pipe_relays.end()) {
            for (const DataflowBufferSpec* relay : relays_it->second) {
                for (const auto& rec : collected.dfb_endpoints.at(relay->unique_id).producers) {
                    take_lanes(rec.kernel);
                }
            }
        }
        if (lanes.has_value() && *lanes > 1) {
            TT_FATAL(
                *lanes <= lane_capacity,
                "PrefetcherPipeParameter '{}' receiver kernel '{}' has {} threads, but a pipe supports at most {} "
                "credit lanes on this architecture",
                pipe.unique_id,
                lanes_source->unique_id,
                *lanes,
                lane_capacity);
            TT_FATAL(
                pipe.ring_size % pipe.entry_size == 0,
                "PrefetcherPipeParameter '{}' with {} credit lanes requires entry_size {} to divide ring_size {}",
                pipe.unique_id,
                *lanes,
                pipe.entry_size,
                pipe.ring_size);
            TT_FATAL(
                (pipe.ring_size / pipe.entry_size) % *lanes == 0,
                "PrefetcherPipeParameter '{}' ring holds {} entries of {} bytes, which is not a multiple of {} "
                "credit lanes (receiver kernel '{}' num_threads)",
                pipe.unique_id,
                pipe.ring_size / pipe.entry_size,
                pipe.entry_size,
                *lanes,
                lanes_source->unique_id);
        }
    }

    // Rules 5 and 6: relay DFBs.
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.advanced_options.prefetcher_pipe_relays.empty()) {
            continue;
        }
        TT_FATAL(
            !dfb.borrowed_from.has_value(),
            "DFB '{}' sets both prefetcher_pipe_relays and borrowed_from; a relay DFB's backing memory is the "
            "pipe ring",
            dfb.unique_id);

        const PrefetcherPipeParameter* first =
            collected.prefetcher_pipe_by_name.at(dfb.advanced_options.prefetcher_pipe_relays[0]);
        NodeRangeSet relayed_receivers;
        for (const auto& pipe_name : dfb.advanced_options.prefetcher_pipe_relays) {
            const PrefetcherPipeParameter* pipe = collected.prefetcher_pipe_by_name.at(pipe_name);
            TT_FATAL(
                pipe->ring_size == first->ring_size && pipe->entry_size == first->entry_size,
                "DFB '{}' relays PrefetcherPipeParameters '{}' (ring_size {}, entry_size {}) and '{}' (ring_size "
                "{}, entry_size {}); every pipe relayed by one DFB must share ring_size and entry_size",
                dfb.unique_id,
                first->unique_id,
                first->ring_size,
                first->entry_size,
                pipe->unique_id,
                pipe->ring_size,
                pipe->entry_size);
            const NodeRangeSet& receivers = pipe_receiver_set.at(pipe_name);
            TT_FATAL(
                !relayed_receivers.intersects(receivers),
                "DFB '{}' relays PrefetcherPipeParameter '{}' whose receiver nodes overlap another relayed pipe's; "
                "relayed pipes must have disjoint receivers",
                dfb.unique_id,
                pipe_name);
            relayed_receivers = relayed_receivers.merge(receivers);
        }

        TT_FATAL(
            dfb.entry_size != 0 && first->entry_size % dfb.entry_size == 0,
            "DFB '{}' entry_size {} must divide relayed PrefetcherPipeParameter '{}' entry_size {}: a relay DFB "
            "pages each pipe entry as a whole number of its own entries",
            dfb.unique_id,
            dfb.entry_size,
            first->unique_id,
            first->entry_size);
        if (dfb.entry_size != first->entry_size) {
            // Credit lanes stripe whole pipe entries over the relay's producer threads; a relay paged
            // finer than the pipe is only implemented for one producer thread.
            for (const auto& rec : collected.dfb_endpoints.at(dfb.unique_id).producers) {
                TT_FATAL(
                    rec.kernel->num_threads == 1,
                    "DFB '{}' pages relayed PrefetcherPipeParameter '{}' entry_size {} as entries of {} bytes, which "
                    "needs a single-threaded relay producer, but kernel '{}' has {} threads",
                    dfb.unique_id,
                    first->unique_id,
                    first->entry_size,
                    dfb.entry_size,
                    rec.kernel->unique_id,
                    rec.kernel->num_threads);
            }
        }
        const uint32_t usable_ring_size = first->ring_size - first->ring_size % first->entry_size;
        TT_FATAL(
            static_cast<uint64_t>(dfb.entry_size) * dfb.num_entries == usable_ring_size,
            "DFB '{}' (entry_size {} * num_entries {} = {} bytes) must exactly cover the {} bytes of whole entries in "
            "relayed PrefetcherPipeParameter '{}' (ring_size {}, entry_size {})",
            dfb.unique_id,
            dfb.entry_size,
            dfb.num_entries,
            static_cast<uint64_t>(dfb.entry_size) * dfb.num_entries,
            usable_ring_size,
            first->unique_id,
            first->ring_size,
            first->entry_size);

        const NodeRangeSet& dfb_nodes = collected.dfb_node_set.at(dfb.unique_id);
        TT_FATAL(
            same_node_set(dfb_nodes, relayed_receivers),
            "DFB '{}' relays PrefetcherPipe(s) whose receiver nodes do not match the DFB's node set (union of its "
            "bound kernels' WorkUnitSpec nodes). The relay must live on exactly the receiver nodes.",
            dfb.unique_id);

        // Every PRODUCER must be the relayed pipes' receiver kernel: it binds exactly this pipe
        // set under one accessor. (Binding implies data-movement by rule 2; the tiling rule then
        // makes its nodes the receiver union, i.e. the DFB's nodes.) Without the binding the
        // producer could not drive the pipe protocol the relay depends on.
        const std::unordered_set<PrefetcherPipeParamName> relayed_set(
            dfb.advanced_options.prefetcher_pipe_relays.begin(), dfb.advanced_options.prefetcher_pipe_relays.end());
        for (const auto& rec : collected.dfb_endpoints.at(dfb.unique_id).producers) {
            const bool binds_relayed_set = std::any_of(
                rec.kernel->advanced_options.prefetcher_pipe_bindings.begin(),
                rec.kernel->advanced_options.prefetcher_pipe_bindings.end(),
                [&](const KernelAdvancedOptions::PrefetcherPipeBinding& binding) {
                    return binding.pipe_parameter_names.size() == relayed_set.size() &&
                           std::all_of(
                               binding.pipe_parameter_names.begin(),
                               binding.pipe_parameter_names.end(),
                               [&](const PrefetcherPipeParamName& n) { return relayed_set.contains(n); });
                });
            TT_FATAL(
                binds_relayed_set,
                "Kernel '{}' is a PRODUCER of relay DFB '{}' but has no PrefetcherPipe accessor naming exactly the "
                "relayed pipe set ({} pipe(s), first '{}'). A relay's producer is the relayed pipes' receiver "
                "data-movement kernel; it must bind them (KernelAdvancedOptions::prefetcher_pipe_bindings) under one "
                "accessor.",
                rec.kernel->unique_id,
                dfb.unique_id,
                relayed_set.size(),
                first->unique_id);
        }
    }
}

// ----------------------------------------------------------------------------
// ValidateNodeBounds: Node coordinate bounds checking
// ----------------------------------------------------------------------------
//
// Validates that every NodeCoord referenced by a WorkUnitSpec or SemaphoreSpec
// is within the compute worker grid on this device.
// (Kernel and DFB placement is derived from WorkUnitSpec membership, so
// bounds-checking the WorkUnitSpecs covers them too.)
//
// NOTE: We're dealing in logical coordinates. (Harvesting is handled by UMD.)
//
// ASSUMPTION: All chips in a MeshDevice are identical, so chip 0 is
// representative of every device in the mesh.

void ValidateNodeBounds(const ProgramSpec& spec, MetalContext& metal_ctx) {
    MetalEnvImpl& env_impl = MetalEnvAccessor(metal_ctx.get_env()).impl();

    // Handle the mock device case (for cheap unit testing)
    const bool is_mock = metal_ctx.get_cluster().get_target_device_type() == tt::TargetDevice::Mock;

    // A default DispatchCoreConfig and 1 CQ is sufficient to look up the compute grid size
    // from the YAML descriptor, and both are available in mock mode.
    DispatchCoreConfig dispatch_core_config{};
    uint8_t num_hw_cqs = 1;
    constexpr ChipId chip_id = 0;

    // But, best get the real dispatch_core_config and num_hw_cqs
    // (Makes no difference now, but hardbaking that assumption could be brittle)
    if (!is_mock) {
        auto& dispatch_mgr = metal_ctx.get_dispatch_core_manager();
        dispatch_core_config = dispatch_mgr.get_dispatch_core_config();
        num_hw_cqs = dispatch_mgr.get_num_hw_cqs();
    }

    // The compute_grid already accounts for the dispatch row/col
    // No need for dispatch-specific checks (and dispatch-specific error messages confuse users)
    const CoreCoord compute_grid = tt::get_compute_grid_size(env_impl, chip_id, num_hw_cqs, dispatch_core_config);

    auto check_target_nodes =
        [&](const Nodes& target_nodes, std::string_view entity_type, std::string_view entity_name) {
            const NodeRangeSet range_set = to_node_range_set(target_nodes);
            for (const NodeRange& range : range_set.ranges()) {
                for (const NodeCoord& node : range) {
                    TT_FATAL(
                        node.x < compute_grid.x && node.y < compute_grid.y,
                        "{} '{}' targets node ({},{}), which is out of bounds. "
                        "The compute worker grid on this device is {}x{}.",
                        entity_type,
                        entity_name,
                        node.x,
                        node.y,
                        compute_grid.x,
                        compute_grid.y);
                }
            }
        };

    for (const auto& work_unit : spec.work_units) {
        check_target_nodes(work_unit.target_nodes, "WorkUnitSpec", work_unit.name);
    }
    for (const auto& sem : spec.semaphores) {
        check_target_nodes(sem.target_nodes, "SemaphoreSpec", sem.unique_id.get());
    }
    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        check_target_nodes(pipe.receivers, "PrefetcherPipeParameter", pipe.unique_id.get());
    }
}

// ValidateProgramSpec: Semantic validation
// ----------------------------------------------------------------------------
//
// This function checks SEMANTIC rules (that don't affect the CollectedSpecData structure):
//   - Architecture requirements
//   - Resource limits
//   - Feature support
//   - Target node constraints (work_unit overlap, node coverage, node validity)
//
// Assumes CollectedSpecData is already built.
void ValidateProgramSpec(
    const ProgramSpec& spec, const CollectedSpecData& collected, MetalContext& metal_ctx, const Allocator& allocator) {
    const Hal& hal = metal_ctx.hal();
    // Sanity check for supported architecture.
    TT_FATAL(is_gen1_arch(hal) || is_gen2_arch(hal), "Unsupported architecture.");

    //////////////////////////////
    // Node bounds checks
    //////////////////////////////

    ValidateNodeBounds(spec, metal_ctx);

    //////////////////////////////
    // Validate KernelSpecs
    //////////////////////////////

    // A Program needs at least one kernel
    TT_FATAL(!spec.kernels.empty(), "A ProgramSpec must have at least one KernelSpec");

    // Validate named RTA/CRTA schema and named CTAs
    for (const auto& kernel : spec.kernels) {
        // All three kinds share the args:: namespace — their names must be mutually unique.
        std::unordered_map<std::string, const char*> seen;  // name -> kind
        auto check_name = [&](const std::string& name, const char* kind) {
            TT_FATAL(
                IsValidCppIdentifier(name),
                "KernelSpec '{}' {} name '{}' is not a valid C++ identifier.",
                kernel.unique_id,
                kind,
                name);
            auto [it, inserted] = seen.try_emplace(name, kind);
            TT_FATAL(
                inserted,
                "KernelSpec '{}' has a naming collision: '{}' is declared as both a {} and a {}.",
                kernel.unique_id,
                name,
                it->second,
                kind);
        };
        for (const auto& name : kernel.runtime_arg_schema.runtime_arg_names) {
            check_name(name, "named RTA");
        }
        for (const auto& name : kernel.runtime_arg_schema.common_runtime_arg_names) {
            check_name(name, "named CRTA");
        }
        for (const auto& [name, value] : kernel.compile_time_args) {
            (void)value;
            check_name(name, "named CTA");
        }
    }

    // Validate kernel thread counts
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(kernel.num_threads > 0, "KernelSpec '{}' has no threads!", kernel.unique_id);
        if (kernel.is_compute_kernel()) {
            if (is_gen2_arch(hal)) {
                TT_FATAL(
                    kernel.num_threads <= QUASAR_TENSIX_ENGINES_PER_NODE,
                    "KernelSpec '{}' has too many threads. The architecture supports up to {} for compute kernels.",
                    kernel.unique_id,
                    QUASAR_TENSIX_ENGINES_PER_NODE);
                // On Quasar, we're not allowing 3-thread compute kernels.
                TT_FATAL(
                    kernel.num_threads != 3,
                    "KernelSpec '{}' has 3 threads, which is not supported for compute kernels. Legal values are 1, 2, "
                    "and 4.",
                    kernel.unique_id);
            } else {
                TT_FATAL(
                    kernel.num_threads == 1,
                    "KernelSpec '{}' specifies {} compute threads, but the target architecture does not support "
                    "multi-threaded kernels.",
                    kernel.unique_id,
                    kernel.num_threads);
            }
        }
        if (kernel.is_data_movement_kernel()) {
            if (is_gen2_arch(hal)) {
                TT_FATAL(
                    kernel.num_threads <= QUASAR_USER_DM_CORES_PER_NODE,
                    "KernelSpec '{}' has too many data movement threads. The maximum is {}.",
                    kernel.unique_id,
                    QUASAR_USER_DM_CORES_PER_NODE);
            } else {
                TT_FATAL(
                    kernel.num_threads == 1,
                    "KernelSpec '{}' specifies {} DM threads, but the target architecture does not support "
                    "multi-threaded kernels. "
                    "num_threads must be 1.",
                    kernel.unique_id,
                    kernel.num_threads);
            }
        }
    }

    // On Gen1, a DM kernel must supply config_1xx: processor and NOC have no
    // default. Architecture still comes from the device, not from which optional is set.
    // Gen1 has exactly two DM processors: RISCV_0 (BRISC) and RISCV_1 (NCRISC).
    // RISCV_2..RISCV_7 exist only on Gen2/Quasar. Reject them here, mirroring the legacy
    // CreateDataMovementKernel "DM0 or DM1 only" guard.
    if (is_gen1_arch(hal)) {
        for (const auto& kernel : spec.kernels) {
            if (!kernel.is_data_movement_kernel()) {
                continue;
            }
            const auto& data_movement_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);
            TT_FATAL(
                data_movement_config.config_1xx.has_value(),
                "KernelSpec '{}' is a data-movement kernel on Gen1 but has no config_1xx processor/NOC. "
                "Those settings are required to build a Gen1 data-movement kernel. Supply a DataMovement1XXConfig "
                "(e.g. CreateReaderDataMovementConfig()/CreateWriterDataMovementConfig()).",
                kernel.unique_id);
            const DataMovementProcessor processor = data_movement_config.config_1xx->processor;
            TT_FATAL(
                processor == DataMovementProcessor::RISCV_0 || processor == DataMovementProcessor::RISCV_1,
                "KernelSpec '{}' targets Gen1 (WH/BH) but requests DM processor RISCV_{}. Gen1 has only "
                "RISCV_0 and RISCV_1; RISCV_2..RISCV_7 exist only on Gen2/Quasar.",
                kernel.unique_id,
                static_cast<int>(processor));
        }
    }

    // On Gen1 (WH/BH), the DM kernels sharing a node must be mutually coherent:
    //   1. Distinct DM processors (RISCV_0 vs RISCV_1) — no two kernels may pin the same RISC.
    //   2. Agreeing NOC mode. noc_mode configures shared per-core NOC hardware (command-buffer
    //      partitioning + completion-counter location) and is compiled into each kernel binary as
    //      the NOC_MODE define (see kernel.cpp), so two kernels on a node with different modes are
    //      incoherent. Mirrors the KernelGroup-construction guard ("KernelGroup must have the same
    //      noc mode for all kernels"), surfaced here at spec-validation time with a clearer message.
    //   3. In DM_DEDICATED_NOC mode, distinct NOCs. Each DM kernel's NoC traffic is statically
    //      compiled to NOC_INDEX == config.noc (see kernel.cpp), so two dedicated-NOC kernels
    //      pinned to the same NOC deadlock the device. This enforces the NOC-distinctness invariant
    //      that KernelGroup finalize silently relies on (it writes brisc_noc_id = arg.noc for
    //      RISCV_0 vs 1 - arg.noc for RISCV_1, which agree only when the two NOCs differ -- "safe
    //      due to prior correctness validation"). The legacy CheckDataMovementConfig intended this
    //      check but did not reliably enforce it for the common reader+writer pair (it runs before
    //      the second DM kernel is registered). DM_DYNAMIC_NOC kernels are exempt: they may
    //      intentionally share a NOC, freeing the other NOC for fabric.
    // (Each kernel's effective node set is derived from WorkUnitSpec membership.)
    if (is_gen1_arch(hal)) {
        // (node, processor) -> the kernel that already claimed it.
        std::map<std::pair<NodeCoord, DataMovementProcessor>, KernelSpecName> claimed_processor;
        // node -> (noc mode, the kernel that first set it) — all DM kernels on a node must agree.
        std::map<NodeCoord, std::pair<NOC_MODE, KernelSpecName>> node_noc_mode;
        // (node, noc) -> the kernel that already claimed it (dedicated-NOC kernels only).
        std::map<std::pair<NodeCoord, NOC>, KernelSpecName> claimed_noc;
        for (const auto& kernel : spec.kernels) {
            if (!kernel.is_data_movement_kernel()) {
                continue;
            }
            const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);
            if (!dm_config.config_1xx.has_value()) {
                continue;
            }
            const auto& gen1 = *dm_config.config_1xx;
            const NodeRangeSet& nodes = collected.kernel_node_set.at(kernel.unique_id);
            for (const auto& range : nodes.ranges()) {
                for (const auto& node : range) {
                    auto [proc_it, proc_inserted] =
                        claimed_processor.try_emplace(std::make_pair(node, gen1.processor), kernel.unique_id);
                    TT_FATAL(
                        proc_inserted,
                        "KernelSpec '{}' conflicts with '{}' on node ({}, {}): both claim the same DM processor. ",
                        kernel.unique_id,
                        proc_it->second,
                        node.x,
                        node.y);

                    // All DM kernels on a node must agree on NOC mode -- it configures shared per-core NOC
                    // hardware. Independent of the NOC-distinctness check below (which is gated per-kernel on
                    // DM_DEDICATED_NOC); their source order does not affect behavior.
                    auto [mode_it, mode_inserted] =
                        node_noc_mode.try_emplace(node, std::make_pair(gen1.noc_mode, kernel.unique_id));
                    TT_FATAL(
                        mode_inserted || mode_it->second.first == gen1.noc_mode,
                        "KernelSpec '{}' conflicts with '{}' on node ({}, {}): they set different NOC modes (one "
                        "DM_DEDICATED_NOC, the other DM_DYNAMIC_NOC). All data movement kernels on a node must use "
                        "the same NOC mode.",
                        kernel.unique_id,
                        mode_it->second.second,
                        node.x,
                        node.y);

                    // NOC-distinctness applies only to statically-pinned (dedicated-NOC) kernels.
                    if (gen1.noc_mode == NOC_MODE::DM_DEDICATED_NOC) {
                        auto [noc_it, noc_inserted] =
                            claimed_noc.try_emplace(std::make_pair(node, gen1.noc), kernel.unique_id);
                        TT_FATAL(
                            noc_inserted,
                            "KernelSpec '{}' conflicts with '{}' on node ({}, {}): both are dedicated-NOC data "
                            "movement kernels pinned to NOC_{}, which hangs the device. Give them distinct NOCs, or "
                            "use DM_DYNAMIC_NOC mode to intentionally share a NOC.",
                            kernel.unique_id,
                            noc_it->second,
                            node.x,
                            node.y,
                            static_cast<int>(gen1.noc));
                    }
                }
            }
        }
    }

    // Validate compute kernel unpack_modes entries against the per-DFB unpack legality table.
    //
    // "Unpack to Dest" means the unpacker writes a consumed DFB straight into the Dest register,
    // bypassing SrcA/B. Its legality depends on the Dest width (enable_32_bit_dest), the DFB's
    // element width, the binding role, and the generation:
    //
    //   UnpackToSrc                          → always accepted (the default path).
    //   UnpackToDest, producer-only binding  → inert (the DFB is never unpacked): tolerated.
    //   UnpackToDest, consumer, enable=true  → accepted (Dest is 32-bit; the choice is coherent).
    //   UnpackToDest, consumer, enable=false, 32-bit format (Float32/Int32/UInt32/RawUInt32)
    //                                        → REJECTED on every generation: a 32-bit datum cannot
    //                                          be unpacked into a 16-bit Dest register.
    //   UnpackToDest, consumer, enable=false, <=16-bit format, Gen1
    //                                        → REJECTED: bad for perf (bypasses SrcA/B for no gain).
    //   UnpackToDest, consumer, enable=false, <=16-bit format, Gen2
    //                                        → accepted: Gen2 has no unpack-to-Dest penalty.
    //   (A compute self-loop DFB binds both roles; the consumer rules govern it.)
    //
    // Separately, where the Src-vs-Dest choice is REAL an explicit entry is REQUIRED rather than
    // silently defaulting to UnpackToSrc: a consumed Float32 DFB with enable_32_bit_dest=true.
    //
    // INTENTIONAL INTERMEDIATE GAP — do not "fix" without the follow-up. The require-an-explicit-
    // entry rule is Float32-only. The choice is just as real for a consumed Int32/UInt32 DFB with
    // enable_32_bit_dest=true, and the end goal is to require an entry there too — but that is a
    // legality tightening that would reject roughly a dozen already-ported ops, so it is deferred to
    // a follow-up PR (see issue #49936). Until then, an unspecified int32/uint32 consumer silently
    // defaults to UnpackToSrc (its 32-bit value truncated to ~19 bits): wrong, but it preserves
    // existing behavior. (Some accepted UnpackToDest cases are also silently mishandled by the LLK
    // today — a codegen gap being fixed LLK-side, not a host-validation concern.)

    // A DataFormat whose elements are 32 bits wide, and so cannot be held by a 16-bit Dest register.
    // (Note: datum_size() throws on the block/MX formats.)
    auto is_32bit_element_format = [](tt::DataFormat fmt) {
        switch (fmt) {
            case tt::DataFormat::Float32:
            case tt::DataFormat::Int32:
            case tt::DataFormat::UInt32:
            case tt::DataFormat::RawUInt32: return true;
            default: return false;
        }
    };

    for (const auto& kernel : spec.kernels) {
        if (!kernel.is_compute_kernel()) {
            continue;
        }
        const auto& compute_config = std::get<ComputeHardwareConfig>(kernel.hw_config);
        const auto& unpack_modes = compute_config.unpack_modes;
        const bool enable_32_bit_dest = compute_config.enable_32_bit_dest;
        const bool is_gen2 = is_gen2_arch(hal);

        // Index the kernel's DFB bindings: which it binds at all, and which it CONSUMES. A self-loop
        // DFB appears as two separate bindings (one PRODUCER, one CONSUMER — there is no BOTH endpoint
        // type); indexing by name into a set dedups them, and membership in consumed_dfbs makes the
        // consumer rules govern it.
        std::unordered_set<DFBSpecName> bound_dfbs;
        std::unordered_set<DFBSpecName> consumed_dfbs;
        for (const auto& binding : kernel.dfb_bindings) {
            bound_dfbs.insert(binding.dfb_spec_name);
            if (binding.endpoint_type == DFBEndpointType::CONSUMER) {
                consumed_dfbs.insert(binding.dfb_spec_name);
            }
        }

        // Validate each explicit entry, tracking which DFBs got one (to require one below where the
        // choice is real). Duplicate DFB entries are impossible: unpack_modes is a Table with unique
        // keys, so a repeated DFB overwrites the prior value.
        std::unordered_set<DFBSpecName> dfbs_with_entry;
        for (const auto& [dfb_name, mode] : unpack_modes) {
            dfbs_with_entry.insert(dfb_name);
            TT_FATAL(
                bound_dfbs.contains(dfb_name),
                "Kernel '{}' unpack_modes entry references DFB '{}', which the kernel does not bind",
                kernel.unique_id,
                dfb_name);

            if (mode == UnpackMode::UnpackToSrc) {
                continue;  // Always allowed.
            }
            //////////////////////////
            // mode == UnpackToDest
            //////////////////////////
            if (!consumed_dfbs.contains(dfb_name)) {
                continue;  // Compute kernel is bound as the DFB Producer: inert, tolerated.
            }

            // Compute kernel is the DFB's consumer.

            if (enable_32_bit_dest) {
                continue;  // UnpackTo Dest, with 32-bit Dest: always permitted
            }

            // UnpackToDest into a 16-bit Dest:
            // Legality checks are gen-specific, and depends on the element width.

            const DataflowBufferSpec* dfb_spec = collected.dfb_by_name.at(dfb_name);
            if (!dfb_spec->data_format_metadata.has_value()) {
                continue;  // Format unknown (deferred to the data_format-required check).
            }

            const tt::DataFormat fmt = dfb_spec->data_format_metadata.value();
            TT_FATAL(
                !is_32bit_element_format(fmt),
                "Compute kernel '{}' unpack_modes entry for DFB '{}' specifies UnpackToDest, but the DFB entries use a "
                "32-bit format ({}) and enable_32_bit_dest is false. A 32-bit datum cannot be unpacked into "
                "a 16-bit Dest register. Set enable_32_bit_dest=true, or use UnpackToSrc.",
                kernel.unique_id,
                dfb_name,
                fmt);
            TT_FATAL(
                is_gen2,
                "Compute kernel '{}' unpack_modes entry for DFB '{}' specifies UnpackToDest, but "
                "enable_32_bit_dest=false "
                "and the data type is not a 32-bit type. On Gen1 architectures, bypassing the SrcA/B path (with no "
                "precision benefit) is not permitted because it leads to worse performance. Use UnpackToSrc instead.",
                kernel.unique_id,
                dfb_name);
            // On Gen2, <=16-bit format + UnpackToDest + enable_32_bit_dest=false
            // is permitted. Unpacking to dest on Gen2 does not carry the performance penalty it does on Gen1.
        }

        // Require an explicit entry (i.e. don't assume a default) if the following conditions are all true:
        //  - the compute kernel is the DFB consumer
        //  - the data format is FP32
        //  - enable_32_bit_dest=true
        // NOTE: Int32/UInt32 are also 32-bit formats, but they are deliberately NOT required here yet.
        //       See the INTENTIONAL INTERMEDIATE GAP note above.
        //       This check should be extended to int32/uint32. (TODO: Issue #49936)
        if (enable_32_bit_dest) {
            for (const auto& binding : kernel.dfb_bindings) {
                if (binding.endpoint_type != DFBEndpointType::CONSUMER) {
                    continue;
                }
                const DataflowBufferSpec* dfb_spec = collected.dfb_by_name.at(binding.dfb_spec_name);
                if (!dfb_spec->data_format_metadata.has_value()) {
                    continue;  // Format unknown (deferred to the data_format-required check).
                }

                // FP32 only for now
                if (dfb_spec->data_format_metadata.value() != tt::DataFormat::Float32) {
                    continue;
                }
                TT_FATAL(
                    dfbs_with_entry.contains(binding.dfb_spec_name),
                    "Compute kernel '{}' consumes FP32 DFB '{}' with enable_32_bit_dest=true, but provides no "
                    "unpack_modes entry for this DFB. This configuration requires an explicit choice "
                    "between UnpackMode::UnpackToSrc and UnpackMode::UnpackToDest.",
                    kernel.unique_id,
                    binding.dfb_spec_name);
            }
        }
    }

    // Blackhole supports local semaphore bindings on UNPACK and PACK (SemScope::COMPUTE_ATOMIC).
    // Wormhole has no compute implementation and Quasar compute remains out of scope.
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            !kernel.is_compute_kernel() || kernel.semaphore_bindings.empty() || hal.get_arch() == tt::ARCH::BLACKHOLE,
            "KernelSpec '{}' has semaphore bindings. "
            "Semaphore bindings on compute kernels are supported only on Blackhole.",
            kernel.unique_id);
    }

    // A compute semaphore is an UNPACK <-> PACK mechanism (the Tensix hardware semaphore, driven by
    // Tensix instructions a DM core cannot issue) and may not be shared with a DM kernel. Reject it
    // here rather than resolve a scope that cannot serve both.
    {
        std::unordered_set<std::string_view> sem_has_compute;
        std::unordered_set<std::string_view> sem_has_dm;
        for (const auto& kernel : spec.kernels) {
            for (const auto& binding : kernel.semaphore_bindings) {
                (kernel.is_compute_kernel() ? sem_has_compute : sem_has_dm).insert(*binding.semaphore_spec_name);
            }
        }
        for (const auto& name : sem_has_compute) {
            TT_FATAL(
                !sem_has_dm.contains(name),
                "SemaphoreSpec '{}' is bound by both a compute kernel and a data-movement kernel. "
                "Compute semaphores synchronize UNPACK and PACK with each other and cannot be shared "
                "with a DM kernel; use separate semaphores for the compute and data-movement handoffs.",
                name);
        }
        // Every compute semaphore maps onto the single free Tensix hardware semaphore (index 3), so two in
        // one program would alias the same hardware state.
        TT_FATAL(
            sem_has_compute.size() <= 1,
            "{} semaphores are bound by compute kernels; a program may bind at most one compute semaphore "
            "(Blackhole has a single free Tensix hardware semaphore).",
            sem_has_compute.size());
        // The compute semaphore lives in the Tensix Sync Unit, which the host cannot write; it is seeded
        // to 0 by compute_kernel_hw_startup() on the device, so no other initial value can be honored.
        // Its capacity (max_value) is a 4-bit hardware field and has no meaning for a DM semaphore.
        for (const auto& sem : spec.semaphores) {
            const bool compute_bound = sem_has_compute.contains(std::string_view{*sem.unique_id});
            const uint32_t init_value = sem.advanced_options.initial_value;
            const uint32_t max_value = sem.advanced_options.max_value;
            TT_FATAL(
                !compute_bound || init_value == 0,
                "SemaphoreSpec '{}' is bound by a compute kernel but has initial_value={}. Compute "
                "semaphores always start at 0 (seeded by compute_kernel_hw_startup on the device).",
                sem.unique_id,
                init_value);
            TT_FATAL(
                !compute_bound || max_value <= 15,
                "SemaphoreSpec '{}' has max_value={}; a compute semaphore's capacity is at most 15 (4-bit "
                "Tensix hardware semaphore).",
                sem.unique_id,
                max_value);
            TT_FATAL(
                compute_bound || max_value == 0,
                "SemaphoreSpec '{}' has max_value={} but is not bound by a compute kernel; max_value is the "
                "capacity of a compute semaphore and has no effect on a data-movement semaphore.",
                sem.unique_id,
                max_value);
        }
    }

    // Validate DM kernel disable_dfb_implicit_sync_for entries.
    //
    // Implicit sync is a Gen2-only, DM-only mechanism (ISR-based credit posting from NoC
    // transaction completion). A DM kernel can opt out per-DFB by listing the DFB's name in
    // config_2xx->disable_dfb_implicit_sync_for, or opt out of all the DFBs it binds at
    // once via config_2xx->disable_dfb_implicit_sync_for_all. If config_2xx is not
    // engaged, implicit sync stays at its default (on for every bound DFB). Either way the
    // opt-out applies to the side(s) of the DFB this kernel binds (producer, consumer, or
    // both for a self-loop).
    //
    // Per-kernel rule: every listed name references a DFB the kernel binds (typo guard).
    //
    // Cross-kernel rule (per DFB): on each side independently, all DM kernels must agree on the
    // opt-out — either all disable it (by list or by _all), or none do. (Producer-side and
    // consumer-side are checked separately; the underlying hardware mechanism is per-side, with
    // one mask per side.)
    {
        // Per-kernel pass: typo guard.
        for (const auto& kernel : spec.kernels) {
            if (!kernel.is_data_movement_kernel()) {
                continue;
            }
            const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);
            if (!dm_config.config_2xx.has_value()) {
                continue;
            }
            std::unordered_set<DFBSpecName> bound_dfbs;
            for (const auto& binding : kernel.dfb_bindings) {
                bound_dfbs.insert(binding.dfb_spec_name);
            }
            for (const auto& dfb_name : dm_config.config_2xx->disable_dfb_implicit_sync_for) {
                TT_FATAL(
                    bound_dfbs.contains(dfb_name),
                    "Kernel '{}' disable_dfb_implicit_sync_for entry references DFB '{}', which the kernel does not "
                    "bind",
                    kernel.unique_id,
                    dfb_name);
            }
        }

        // Cross-kernel pass: per-DFB producer-side and consumer-side agreement.
        // Note: a single DFB can be bound by multiple producer KernelSpecs and multiple
        // consumer KernelSpecs — ops sometimes specialize the same kernel source by CTAs,
        // producing several KernelSpecs that share a DFB.
        auto check_side_agreement =
            [&](const std::vector<CollectedSpecData::DFBEndpointInfo::EndpointRecord>& endpoints,
                const DFBSpecName& dfb_name,
                std::string_view side_label) {
                const KernelSpec* canonical = nullptr;
                bool canonical_disables = false;
                for (const auto& ep : endpoints) {
                    if (!ep.kernel->is_data_movement_kernel()) {
                        continue;
                    }
                    const auto& dm_config = std::get<DataMovementHardwareConfig>(ep.kernel->hw_config);
                    if (!is_gen2_arch(hal)) {
                        // Gen1 device — can't physically participate in Gen2 implicit sync; abstains.
                        continue;
                    }
                    const bool disables = DmKernelDisablesImplicitSync(dm_config, dfb_name);
                    if (canonical == nullptr) {
                        canonical = ep.kernel;
                        canonical_disables = disables;
                        continue;
                    }
                    TT_FATAL(
                        disables == canonical_disables,
                        "DFB '{}' has disagreeing implicit-sync opt-out state on the {} side",
                        dfb_name,
                        side_label);
                }
            };
        for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
            check_side_agreement(endpoint_info.producers, dfb_name, "producer");
            check_side_agreement(endpoint_info.consumers, dfb_name, "consumer");
        }
    }

    //////////////////////////////////
    // Validate DataflowBufferSpecs
    //////////////////////////////////

    // Device slots are allocated per core (two DFBs may share a slot iff their node sets are
    // disjoint), so the arch limit is on how many DFBs land on any single node — not on
    // ProgramSpec::dataflow_buffers.size(). Gen1 lowers each slot to a circular buffer; Gen2
    // indexes the packed config by device slot up to dfb::NUM_DFBS. Tile-counter exhaustion on
    // Gen2 is still checked later at enqueue.
    {
        const uint32_t max_slots_per_core =
            hal.has_tile_counter_registers() ? static_cast<uint32_t>(::dfb::NUM_DFBS) : hal.get_num_dataflow_buffers();

        std::unordered_map<NodeCoord, uint32_t> dfbs_per_node;
        for (const auto& dfb : spec.dataflow_buffers) {
            for (const NodeCoord& node : corerange_to_cores(collected.dfb_node_set.at(dfb.unique_id))) {
                dfbs_per_node[node]++;
            }
        }

        for (const auto& [node, count] : dfbs_per_node) {
            if (count <= max_slots_per_core) {
                continue;
            }
            if (is_gen1_arch(hal)) {
                TT_THROW(
                    "ProgramSpec '{}' places {} DataflowBufferSpecs on node ({}, {}), but Gen1 "
                    "supports at most {} device slots per core (disjoint cores may reuse slots).",
                    spec.name,
                    count,
                    node.x,
                    node.y,
                    max_slots_per_core);
            } else if (is_gen2_arch(hal)) {
                TT_THROW(
                    "ProgramSpec '{}' places {} DataflowBufferSpecs on node ({}, {}), but the "
                    "target architecture supports at most {} device slots per core. The true "
                    "limit is also configuration-dependent (tile counters) and is checked at "
                    "enqueue.",
                    spec.name,
                    count,
                    node.x,
                    node.y,
                    max_slots_per_core);
            } else {
                TT_FATAL(false, "Unknown architecture");
            }
        }
    }

    // Validate per-DFB sizing: entry_size and num_entries must be set to non-zero values.
    // (Sizes may still be overridden at runtime via ProgramRunArgs, but a ProgramSpec value is required.)
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            dfb.entry_size > 0,
            "DataflowBufferSpec '{}' has entry_size = 0. entry_size must be set to a non-zero value.",
            dfb.unique_id);
        TT_FATAL(
            dfb.num_entries > 0,
            "DataflowBufferSpec '{}' has num_entries = 0. num_entries must be set to a non-zero value.",
            dfb.unique_id);
    }

    // The allow_instance_multi_binding escape hatch is Gen1-only. On Gen2 the DFB's per-RISC
    // tile-counter / remapper machinery is driven by the producer/consumer masks, so a multi-bound
    // instance cannot be lowered. Reject the flag itself on Gen2, independent of whether any instance
    // is actually multi-bound — a Gen2 spec carrying it is never valid.
    if (is_gen2_arch(hal)) {
        for (const auto& dfb : spec.dataflow_buffers) {
            TT_FATAL(
                !dfb.advanced_options.allow_instance_multi_binding,
                "DFB '{}' sets allow_instance_multi_binding, which is only supported on Gen1 (WH/BH) "
                "architectures. On Gen2 a DFB instance must have exactly one producer and one consumer.",
                dfb.unique_id);
        }
    }

    // Validate local DFB endpoint placement and multi-binding consistency.
    //
    // The hardware invariant is local: a local DFB lives in shared SRAM on each node, so at every
    // node where the DFB is instantiated, exactly one producer kernel instance and exactly one
    // consumer kernel instance must run on that node. Metal 2.0 permits multiple PRODUCER
    // KernelSpecs (and multiple CONSUMER KernelSpecs) per DFB, so we enforce that invariant directly
    // as a per-node census, plus per-role uniformity of the binding-site config:
    //   1./2. Placement: every node hosting the DFB runs exactly one producer instance and exactly
    //         one consumer instance (the per-node census below). This subsumes both "no node has two
    //         same-role instances" and "producer and consumer node coverage coincide".
    //   3. All bindings on the same role have matching `access_pattern` (the DFB scheduler
    //      config is shared per role).
    //   4. All KernelSpecs on the same role have matching `num_threads` (the per-side
    //      credit-tracking config is shared per role).
    // Self-loop (a kernel that appears in both producers and consumers of a DFB) is currently
    // restricted to the simple single-producer-single-consumer case.
    for (const auto& dfb : spec.dataflow_buffers) {
        const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);

        // allow_instance_multi_binding (Gen1-only; rejected on Gen2 earlier in this function) turns
        // the per-node DFB into a plain shared circular buffer, which has no per-role hardware config
        // to share — no processor mask, DFB scheduler, or credit machinery. Every per-role uniformity
        // requirement below exists solely to guarantee such a shared config, so none apply under the
        // flag: the role-uniformity checks are skipped and the per-node census relaxes its "exactly
        // one" counts to "at least one".
        const bool allow_multi = dfb.advanced_options.allow_instance_multi_binding;

        // (3) and (4): per-role uniformity of binding-site parameters, plus kernel kind.
        // Kind (compute vs DM) must agree because the DFB's hardware config carries a single
        // processor mask per role, and compute / DM masks live in disjoint bit ranges (bits
        // 0-7 vs 8-15 on Gen2; orthogonal RISC encodings on Gen1) — mismatched kinds cannot
        // share a mask. (All three are skipped under allow_multi — see above.)
        auto check_role_uniformity = [&](const auto& records, std::string_view role) {
            if (records.size() < 2) {
                return;
            }
            const auto first_pattern = records[0].binding->access_pattern;
            const auto first_threads = records[0].kernel->num_threads;
            const bool first_is_compute = records[0].kernel->is_compute_kernel();
            const auto& first_kernel = records[0].kernel->unique_id;
            for (size_t i = 1; i < records.size(); ++i) {
                TT_FATAL(
                    records[i].binding->access_pattern == first_pattern,
                    "DFB '{}' has multiple {} bindings with mismatched access_pattern (kernel '{}' vs kernel '{}')",
                    dfb.unique_id,
                    role,
                    first_kernel,
                    records[i].kernel->unique_id);
                TT_FATAL(
                    records[i].kernel->num_threads == first_threads,
                    "DFB '{}' has multiple {} KernelSpecs with mismatched num_threads (kernel '{}' = {} vs kernel '{}' "
                    "= {})",
                    dfb.unique_id,
                    role,
                    first_kernel,
                    first_threads,
                    records[i].kernel->unique_id,
                    records[i].kernel->num_threads);
                TT_FATAL(
                    records[i].kernel->is_compute_kernel() == first_is_compute,
                    "DFB '{}' has multiple {} KernelSpecs mixing compute and data-movement kinds "
                    "('{}' is a {} kernel; '{}' is a {} kernel). All KernelSpecs bound to the same "
                    "DFB role must be of the same kind — the DFB's hardware config carries a single "
                    "processor mask per role.",
                    dfb.unique_id,
                    role,
                    first_kernel,
                    first_is_compute ? "compute" : "data-movement",
                    records[i].kernel->unique_id,
                    first_is_compute ? "data-movement" : "compute");
            }
        };
        if (!allow_multi) {
            check_role_uniformity(endpoints.producers, "PRODUCER");
            check_role_uniformity(endpoints.consumers, "CONSUMER");
        }

        // (1)/(2) Placement — per-node census. A local DFB lives in shared SRAM on each node, so
        // every node it is instantiated on must run exactly one producer instance and exactly one
        // consumer instance. Tally instances per node directly from the bindings: this subsumes the
        // old within-role disjointness check (a node with >1 same-role instance) and the cross-role
        // coverage check (a node with one role but not the other), and reports the offending node in
        // node terms rather than WorkUnitSpec terms. It counts actual node occupancy, so overlapping
        // same-role placements are caught regardless of how the WorkUnitSpec bookkeeping produced
        // them. (Self-loops fall out naturally: a self-looping kernel tallies as one producer AND
        // one consumer on each of its nodes.)
        std::unordered_map<NodeCoord, std::vector<const KernelSpec*>> producers_on_node;
        std::unordered_map<NodeCoord, std::vector<const KernelSpec*>> consumers_on_node;
        auto tally_role = [&](const auto& records, auto& on_node) {
            for (const auto& rec : records) {
                for (const NodeCoord& node : corerange_to_cores(collected.kernel_node_set.at(rec.kernel->unique_id))) {
                    on_node[node].push_back(rec.kernel);
                }
            }
        };
        tally_role(endpoints.producers, producers_on_node);
        tally_role(endpoints.consumers, consumers_on_node);

        // Footprint = every node hosting any instance of either role. A std::set gives deterministic
        // iteration order, hence deterministic error messages.
        std::set<NodeCoord> footprint;
        for (const auto& [node, kernels] : producers_on_node) {
            footprint.insert(node);
        }
        for (const auto& [node, kernels] : consumers_on_node) {
            footprint.insert(node);
        }

        auto names_at = [](const std::unordered_map<NodeCoord, std::vector<const KernelSpec*>>& on_node,
                           const NodeCoord& node) -> std::string {
            auto it = on_node.find(node);
            if (it == on_node.end() || it->second.empty()) {
                return "none";
            }
            std::string names;
            for (const KernelSpec* k : it->second) {
                names += (names.empty() ? "'" : ", '") + k->unique_id.get() + "'";
            }
            return names;
        };

        // Per-node census. Under allow_multi (see top of loop) the "exactly one" upper bound relaxes
        // to "at least one": a node may host multiple instances of a role, but must still host at
        // least one producer AND one consumer, or the FIFO is half-wired.
        for (const NodeCoord& node : footprint) {
            auto p_it = producers_on_node.find(node);
            auto c_it = consumers_on_node.find(node);
            const size_t num_producers = p_it == producers_on_node.end() ? 0 : p_it->second.size();
            const size_t num_consumers = c_it == consumers_on_node.end() ? 0 : c_it->second.size();
            const bool node_ok =
                allow_multi ? (num_producers >= 1 && num_consumers >= 1) : (num_producers == 1 && num_consumers == 1);
            if (node_ok) {
                continue;
            }
            std::string_view guidance;
            if (num_producers == 0) {
                guidance =
                    "This node has a consumer but no producer — ensure a producer kernel covers it "
                    "(via its WorkUnitSpec membership).";
            } else if (num_consumers == 0) {
                guidance =
                    "This node has a producer but no consumer — ensure a consumer kernel covers it "
                    "(via its WorkUnitSpec membership).";
            } else {
                guidance =
                    "Multiple same-role kernel instances land on this node — their placements overlap; "
                    "give each disjoint nodes.";
            }
            TT_FATAL(
                false,
                "Local DFB '{}' is malformed at node {}: {} producer instance(s) ({}) and {} consumer "
                "instance(s) ({}). A local DFB lives in shared SRAM on each node, so every node it is "
                "instantiated on must run exactly one producer and one consumer kernel instance. {}",
                dfb.unique_id,
                node.str(),
                num_producers,
                names_at(producers_on_node, node),
                num_consumers,
                names_at(consumers_on_node, node),
                guidance);
        }

        // Find a self-loop participant: a kernel bound to this DFB as both producer and consumer.
        // Stays nullptr if the DFB is not self-looped. Iterating producers in vector order keeps the
        // pick — and any resulting error message — deterministic across runs.
        const KernelSpec* self_loop_kernel = nullptr;
        for (const auto& p : endpoints.producers) {
            for (const auto& c : endpoints.consumers) {
                if (p.kernel == c.kernel) {
                    self_loop_kernel = p.kernel;
                    break;
                }
            }
            if (self_loop_kernel != nullptr) {
                break;
            }
        }

        if (self_loop_kernel != nullptr) {
            // A data-movement kernel may self-loop a DFB (bind it as both PRODUCER and CONSUMER) only
            // on Gen1 (WH/BH), where a DFB lowers to a plain circular buffer that a single DM RISC can
            // both fill and drain. On Gen2 the DFB's tile-counter credit machinery requires disjoint
            // producer/consumer RISCs, so a DM self-loop cannot be lowered. Catch it here (with a clear
            // message) rather than let it fall through to a confusing "producer_risc_mask and
            // consumer_risc_mask must not overlap" error in the DFB backend. (Compute self-loops are
            // always legal: they lower to the intra-Tensix packer->unpacker flow.)
            TT_FATAL(
                !(is_gen2_arch(hal) && self_loop_kernel->is_data_movement_kernel()),
                "DataflowBuffer '{}' is self-looped by data-movement kernel '{}' (bound as both PRODUCER "
                "and CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2 "
                "architectures. Consider using a scratchpad or LocalTensorAccessor instead.",
                dfb.unique_id,
                self_loop_kernel->unique_id);

            // Self-loop interplay with multi-binding: the producer set must equal the consumer set
            // as sets of KernelSpec*. This permits the natural pattern of multiple same-source
            // KernelSpecs each self-looping the DFB on their disjoint node ranges, while rejecting
            // the case where a self-looping kernel shares the DFB with an unrelated kernel (which
            // would make the producer/consumer mask and lowering semantics ambiguous).
            std::unordered_set<const KernelSpec*> producer_kernels;
            std::unordered_set<const KernelSpec*> consumer_kernels;
            for (const auto& p : endpoints.producers) {
                producer_kernels.insert(p.kernel);
            }
            for (const auto& c : endpoints.consumers) {
                consumer_kernels.insert(c.kernel);
            }
            TT_FATAL(
                producer_kernels == consumer_kernels,
                "DFB '{}' is self-looped (some kernel appears as both producer and consumer), but "
                "the set of producer KernelSpecs differs from the set of consumer KernelSpecs. "
                "When a DFB is self-looped, every same-side binding must come from a self-loop "
                "participant (i.e. a kernel that appears on both sides).",
                dfb.unique_id);
        }
    }

    // Cross-node DFBs are not yet supported.
    //
    // TODO: When cross-node DFB is supported, add a validation checks. Enforce that
    //       each (producer_node, consumer_node) entry in producer_consumer_map has
    //       p_node != c_node.

    TT_FATAL(
        spec.cross_node_dataflow_buffers.empty(),
        "CrossNodeDataflowBufferSpec is part of the Metal 2.0 API surface but is not yet supported "
        "by the runtime. (ProgramSpec '{}' has {} cross-node DFB(s).)",
        spec.name,
        spec.cross_node_dataflow_buffers.size());

    // Scratchpad placement census (multi-binding rule).
    //
    // A scratchpad is private, node-local L1. More than one KernelSpec may bind the same
    // ScratchpadSpec, but only on disjoint nodes: each node hosting the scratchpad must run exactly
    // one binding kernel instance, so the per-node region stays private to that one kernel.
    // (Allocation and CRTA delivery are per-binding-kernel — allocate_scratchpads stacks each
    // kernel's scratchpad onto its own cores' allocators — so disjoint bindings never interact.) Two
    // binding kernels on the same node would be true sharing, which is deferred behind a future
    // AdvancedOption and rejected here. Mirrors the local-DFB per-node census above. (A kernel
    // binding the same scratchpad twice is already rejected during collection, so every binder here
    // is a distinct kernel.)
    for (const auto& scratchpad : spec.scratchpads) {
        auto binders_it = collected.scratchpad_binders.find(scratchpad.unique_id);
        if (binders_it == collected.scratchpad_binders.end() || binders_it->second.size() < 2) {
            continue;  // unbound (caught earlier) or single binder — always legal.
        }
        // Tally binding-kernel instances per node. A std::map keeps iteration (and error messages)
        // deterministic.
        std::map<NodeCoord, std::vector<const KernelSpec*>> binders_on_node;
        for (const KernelSpec* kernel : binders_it->second) {
            for (const NodeCoord& node : corerange_to_cores(collected.kernel_node_set.at(kernel->unique_id))) {
                binders_on_node[node].push_back(kernel);
            }
        }
        for (const auto& [node, kernels] : binders_on_node) {
            if (kernels.size() <= 1) {
                continue;
            }
            std::string names;
            for (const KernelSpec* kernel : kernels) {
                names += (names.empty() ? "'" : ", '") + kernel->unique_id.get() + "'";
            }
            TT_FATAL(
                false,
                "ScratchpadSpec '{}' is bound by {} kernel instances on node {} ({}). A scratchpad is "
                "private node-local L1; multiple kernels may bind the same scratchpad only on disjoint "
                "nodes, so each node's instance stays private to one kernel. Sharing one node's "
                "scratchpad across kernels is not yet supported — give each binding kernel disjoint nodes.",
                scratchpad.unique_id,
                kernels.size(),
                node.str(),
                names);
        }
    }

    //////////////////////////////////////////////////
    // Validate PrefetcherPipeParameters and relay DFBs
    //////////////////////////////////////////////////

    ValidatePrefetcherPipeSpec(spec, collected, hal);

    // Validate borrowed-memory DFBs.
    //
    // A borrowed-memory DFB names a TensorParameter via DataflowBufferSpec::borrowed_from. The
    // backing MeshTensor flows through ProgramRunArgs::tensor_args at execution time.
    // We enforce only the safety-relevant checks:
    //  - the named parameter exists
    //  - the TensorSpec places storage in L1,
    //  - the spec is large enough
    // We don't validate any layout considerations (interleaved vs sharded, page / tile sizes, etc.)
    // That's on the user; this is an advanced feature.
    for (const auto& dfb : spec.dataflow_buffers) {
        if (!dfb.borrowed_from.has_value()) {
            continue;
        }
        const TensorParamName& tp_name = *dfb.borrowed_from;
        auto it = collected.tensor_parameter_by_name.find(tp_name);
        TT_FATAL(
            it != collected.tensor_parameter_by_name.end(),
            "DFB '{}' borrows memory from TensorParameter '{}', but no such TensorParameter is declared in the "
            "ProgramSpec.",
            dfb.unique_id,
            tp_name);
        const TensorSpec& tensor_spec = it->second->spec;
        TT_FATAL(
            tensor_spec.memory_config().is_l1(),
            "DFB '{}' borrows memory from TensorParameter '{}', but its TensorSpec is not L1-resident (L1 is "
            "required). Both L1 and L1_SMALL are accepted.",
            dfb.unique_id,
            tp_name);
        // Spec-time sizing check. A borrowed DFB lives in ONE core's slice of the backing buffer,
        // so the bound is that buffer's per-bank allocation.
        //
        // TensorSpec yields the per-bank figure without a Buffer: sharded specs take pages-per-bank
        // from the shard spec or the distribution spec, interleaved specs divide their page count by
        // num_banks.
        // The attach-time check in AttachBorrowedDFBBuffers (program_run_args.cpp) stays
        // authoritative; this one just stops deferring a rejection it can already make.
        //
        // Caveat: this is the default allocator, which has no sub-device context. A sub-device
        // allocator owns fewer banks, so an interleaved tensor allocated there has a LARGER per-bank
        // slice than what we compute, and a DFB sized to it would be rejected here even though
        // attach time would take it. Sharded specs ignore num_banks and so are unaffected. Thread a
        // SubDeviceId in here if that combination ever shows up.
        const uint32_t num_banks = allocator.get_num_banks(tensor_spec.memory_config().buffer_type());
        const size_t dfb_bytes = static_cast<size_t>(dfb.entry_size) * static_cast<size_t>(dfb.num_entries);
        const size_t tensor_bytes =
            tensor_spec.compute_consumed_memory_bytes_per_bank(hal.get_alignment(HalMemType::L1), num_banks);
        TT_FATAL(
            dfb_bytes <= tensor_bytes,
            "DFB '{}' (entry_size {} * num_entries {} = {} bytes) is larger than the per-bank allocation of its "
            "borrowed TensorParameter '{}' ({} bytes).",
            dfb.unique_id,
            dfb.entry_size,
            dfb.num_entries,
            dfb_bytes,
            tp_name,
            tensor_bytes);
    }

    // Validate DFB alias groups.
    // Rules:
    //  1. Transitivity: every DFB in an alias group must list every other member in its
    //     alias_with field. This strict requirement is redundant by design. This is a
    //     "dangerous" feature that a kernel author should use deliberately.
    //  2. Same total size: entry_size * num_entries must match within a group.
    //  3. Same node coverage: each DFB in the group must cover the same set of nodes
    //  4. Consistent borrowed_from: either no member borrows, or all members borrow from
    //     the same TensorParameter. (Aliased borrows from the same memory object is a
    //     weird-but-valid scenario.)
    {
        // The "extended group" of a DFB is its alias_with plus the DFB itself. Two DFBs
        // are in the same alias group iff their extended groups are equal.
        auto extended_group = [](const DataflowBufferSpec& d) {
            std::set<DFBSpecName> s(dfb_alias_with(d).begin(), dfb_alias_with(d).end());
            s.insert(d.unique_id);
            return s;
        };

        // Pre-pass: every name in every alias_with must refer to a real DFB and must not
        // be self-referential.
        for (const auto& dfb : spec.dataflow_buffers) {
            for (const auto& alias_name : dfb_alias_with(dfb)) {
                TT_FATAL(
                    collected.dfb_by_name.contains(alias_name),
                    "DFB '{}' lists unknown alias '{}' in alias_with",
                    dfb.unique_id,
                    alias_name);
                TT_FATAL(alias_name != dfb.unique_id, "DFB '{}' lists itself in alias_with", dfb.unique_id);
            }
        }

        for (const auto& dfb : spec.dataflow_buffers) {
            if (dfb_alias_with(dfb).empty()) {
                continue;
            }
            const size_t total_size_a = static_cast<size_t>(dfb.entry_size) * static_cast<size_t>(dfb.num_entries);
            const auto group_a = extended_group(dfb);
            const auto& nodes_a = collected.dfb_node_set.at(dfb.unique_id);

            for (const auto& alias_name : dfb_alias_with(dfb)) {
                const DataflowBufferSpec* alias_spec = collected.dfb_by_name.at(alias_name);

                // Rule 1: full clique declaration.
                const auto group_b = extended_group(*alias_spec);
                if (group_a != group_b) {
                    TT_THROW(
                        "DFBs '{}' and '{}' do not declare the same alias group. Every DFB in an "
                        "alias group must list every other member in its alias_with field.",
                        dfb.unique_id,
                        alias_name);
                }

                // Rule 2: same total size.
                const size_t total_size_b =
                    static_cast<size_t>(alias_spec->entry_size) * static_cast<size_t>(alias_spec->num_entries);
                TT_FATAL(
                    total_size_a == total_size_b,
                    "Aliased DFBs '{}' and '{}' have different total sizes ({} vs {} bytes). "
                    "Aliased DFBs must have the same total size (entry_size * num_entries).",
                    dfb.unique_id,
                    alias_name,
                    total_size_a,
                    total_size_b);

                // Rule 3: same node coverage.
                const auto& nodes_b = collected.dfb_node_set.at(alias_name);
                TT_FATAL(
                    nodes_a == nodes_b,
                    "Aliased DFBs '{}' and '{}' cover different sets of nodes. Aliased DFBs must "
                    "cover the same node coverage (their bound kernels' WorkUnitSpec membership "
                    "must yield identical target_nodes unions) — the shared L1 region must be "
                    "reserved at the same cores for all members.",
                    dfb.unique_id,
                    alias_name);

                // Rule 4: consistent borrowed_from.
                TT_FATAL(
                    dfb.borrowed_from == alias_spec->borrowed_from,
                    "Aliased DFBs '{}' and '{}' have inconsistent borrowed_from. Either no member "
                    "of an alias group borrows, or all members borrow from the same TensorParameter.",
                    dfb.unique_id,
                    alias_name);
            }
        }
    }

    // Data format metadata (optional param) MUST be specified for a DFB with a compute endpoint
    auto any_compute_endpoint = [](const auto& records) {
        for (const auto& rec : records) {
            if (rec.kernel->is_compute_kernel()) {
                return true;
            }
        }
        return false;
    };
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        if (any_compute_endpoint(endpoint_info.producers) || any_compute_endpoint(endpoint_info.consumers)) {
            const DataflowBufferSpec* dfb_spec = collected.dfb_by_name.at(dfb_name);
            TT_FATAL(
                dfb_spec->data_format_metadata.has_value(),
                "DFB '{}' is used by a compute kernel, but no data_format_metadata is specified",
                dfb_name);
        }
    }

    // Data format must be valid for the architecture
    const tt::ARCH arch = hal.get_arch();
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.data_format_metadata.has_value()) {
            TT_FATAL(
                tt::is_data_format_supported(dfb.data_format_metadata.value(), arch),
                "DFB '{}' has data format '{}' which is not supported on architecture {}",
                dfb.unique_id,
                dfb.data_format_metadata.value(),
                arch);
        }
    }

    for (const auto& scratchpad : spec.scratchpads) {
        const bool has_format = scratchpad.data_format_metadata.has_value();
        const bool has_tile = scratchpad.tile_format_metadata.has_value();
        TT_FATAL(
            has_format || !has_tile,
            "ScratchpadSpec '{}' has tile_format_metadata but no data_format_metadata",
            scratchpad.unique_id);
        if (has_format) {
            TT_FATAL(
                tt::is_data_format_supported(scratchpad.data_format_metadata.value(), arch),
                "ScratchpadSpec '{}' has data format '{}' which is not supported on architecture {}",
                scratchpad.unique_id,
                scratchpad.data_format_metadata.value(),
                arch);
        }
    }

    //////////////////////////////////
    // Validate SemaphoreSpecs
    //////////////////////////////////

    for (const auto& sem : spec.semaphores) {
        const uint32_t init_value = sem.advanced_options.initial_value;
        if (is_gen2_arch(hal)) {
            TT_FATAL(
                init_value == 0,
                "SemaphoreSpec '{}' has initial_value={} but only zero is supported on Quasar",
                sem.unique_id,
                init_value);
        }
    }

    //////////////////////////////
    // Validate WorkUnitSpecs
    //////////////////////////////

    // WorkUnitSpec is required: a valid ProgramSpec has at least one WorkUnitSpec.
    const auto& work_units = spec.work_units;
    TT_FATAL(!work_units.empty(), "At least one WorkUnitSpec is required");

    // WorkUnitSpecs may not overlap in their target nodes
    for (const auto& work_unit : work_units) {
        for (const auto& other_work_unit : work_units) {
            if (work_unit.name == other_work_unit.name) {
                continue;
            }
            if (nodes_intersect(work_unit.target_nodes, other_work_unit.target_nodes)) {
                TT_FATAL(
                    false, "WorkUnitSpecs '{}' and '{}' overlap in target nodes", work_unit.name, other_work_unit.name);
            }
        }
    }

    // A WorkUnitSpec must have at least one kernel
    for (const auto& work_unit : work_units) {
        TT_FATAL(!work_unit.kernels.empty(), "WorkUnitSpec '{}' has no kernels", work_unit.name);
    }

    // Does the WorkUnit have enough cores to run all of its kernels?
    for (const auto& work_unit : work_units) {
        uint32_t dm_cores_needed = 0;
        uint32_t compute_engines_needed = 0;
        for (const auto& kernel_name : work_unit.kernels) {
            const auto& kernel_spec = collected.kernel_by_name.at(kernel_name);
            if (kernel_spec->is_compute_kernel()) {
                compute_engines_needed += kernel_spec->num_threads;
            }
            if (kernel_spec->is_data_movement_kernel()) {
                dm_cores_needed += kernel_spec->num_threads;
            }
        }
        if (is_gen2_arch(hal)) {
            TT_FATAL(
                compute_engines_needed <= QUASAR_TENSIX_ENGINES_PER_NODE,
                "WorkUnitSpec '{}' needs {} Tensix engines, but only {} are available",
                work_unit.name,
                compute_engines_needed,
                QUASAR_TENSIX_ENGINES_PER_NODE);
            TT_FATAL(
                dm_cores_needed <= QUASAR_USER_DM_CORES_PER_NODE,
                "WorkUnitSpec '{}' requests {} data movement cores. This exceeds the permitted maximum of {}.",
                work_unit.name,
                dm_cores_needed,
                QUASAR_USER_DM_CORES_PER_NODE);
        }
        if (is_gen1_arch(hal)) {
            TT_FATAL(
                compute_engines_needed <= 1,
                "WorkUnitSpec '{}' has {} compute kernels. The target architecture supports at most one.",
                work_unit.name,
                compute_engines_needed);
            TT_FATAL(
                dm_cores_needed <= 2,
                "WorkUnitSpec '{}' has {} data movement kernels. The target architecture supports at most two.",
                work_unit.name,
                dm_cores_needed);
        }
    }

    // A work_unit can have at most one compute kernel
    for (const auto& work_unit : work_units) {
        uint32_t num_compute_kernels = 0;
        for (const auto& kernel_name : work_unit.kernels) {
            const auto& kernel_spec = collected.kernel_by_name.at(kernel_name);
            if (kernel_spec->is_compute_kernel()) {
                num_compute_kernels++;
            }
        }
        TT_FATAL(num_compute_kernels <= 1, "WorkUnitSpec '{}' has more than one compute kernel", work_unit.name);
    }

    // NOTE:
    // Placement consistency between kernels, DFBs, and WorkUnitSpecs is now structural,
    // not validated:
    //  - Kernels' effective node sets ARE the union of their containing WorkUnitSpecs' target_nodes
    //  - DFBs' allocation node sets are the union of their binding kernels' node sets
}

//////////////////////////////////////////////////
// PostCollectionValidate
//////////////////////////////////////////////////

namespace {

template <typename KernelId>
void ValidateAccessorNameLength(const KernelId& kernel_id, std::string_view kind, std::string_view name) {
    TT_FATAL(
        name.size() <= MAX_ACCESSOR_NAME_LENGTH,
        "Kernel '{}' {} accessor_name '{}' is {} characters; an accessor_name must be at most {} characters",
        kernel_id,
        kind,
        name,
        name.size(),
        MAX_ACCESSOR_NAME_LENGTH);
}

}  // namespace

void PostCollectionValidate(const ProgramSpec& spec, const CollectedSpecData& collected) {
    // ------------------------------------------------------------------------
    // DFB bindings: accessor names, self-loop pairs, role aliasing
    // ------------------------------------------------------------------------
    for (const auto& kernel : spec.kernels) {
        // Track per-accessor-name signatures within this kernel. Reusing a single
        // accessor_name across two DFBBindings is permitted as a "self-loop pair":
        // both bindings target the same DFB with opposite endpoint types (one PRODUCER,
        // one CONSUMER). This lets a kernel that both produces and consumes the same DFB
        // use a single device-side accessor name instead of two aliasing wrappers.
        struct AccessorBindingInfo {
            DFBSpecName dfb_spec_name;
            bool has_producer = false;
            bool has_consumer = false;
        };
        std::unordered_map<std::string, AccessorBindingInfo> accessor_bindings;
        // Track, per DFB, which endpoint roles this kernel has already bound. Within a kernel a DFB
        // may be bound at most once per role; the only multi-binding form is the self-loop pair (one
        // PRODUCER + one CONSUMER, whose accessor names may differ). A second binding of the same
        // role under a different accessor name is the forbidden "one buffer, two names" aliasing
        // (see the check below). Scoped to the kernel so it resets per iteration — a DFB legitimately
        // carries different accessor names on different kernels (producer 'out', consumer 'in'), so
        // this must not be global.
        struct DFBBoundRoles {
            bool has_producer = false;
            bool has_consumer = false;
        };
        std::unordered_map<DFBSpecName, DFBBoundRoles> dfb_bound_roles;
        for (const auto& dfb_binding : kernel.dfb_bindings) {
            auto [it, inserted] = accessor_bindings.try_emplace(
                dfb_binding.accessor_name, AccessorBindingInfo{dfb_binding.dfb_spec_name});
            AccessorBindingInfo& info = it->second;
            if (inserted) {
                TT_FATAL(
                    IsValidCppIdentifier(dfb_binding.accessor_name),
                    "Kernel '{}' DFB accessor_name '{}' must be a valid C++ identifier",
                    kernel.unique_id,
                    dfb_binding.accessor_name);
                ValidateAccessorNameLength(kernel.unique_id, "DFB", dfb_binding.accessor_name);
            } else {
                TT_FATAL(
                    info.dfb_spec_name == dfb_binding.dfb_spec_name,
                    "Kernel '{}' uses accessor_name '{}' for two different DFBs ('{}' and '{}'). "
                    "Reusing a name is only permitted when both bindings target the same DFB (self-loop pair).",
                    kernel.unique_id,
                    dfb_binding.accessor_name,
                    info.dfb_spec_name,
                    dfb_binding.dfb_spec_name);
            }
            const bool is_producer = (dfb_binding.endpoint_type == DFBEndpointType::PRODUCER);
            bool& seen_this_type = is_producer ? info.has_producer : info.has_consumer;
            TT_FATAL(
                !seen_this_type,
                "Kernel '{}' has duplicate {} binding for accessor_name '{}'",
                kernel.unique_id,
                is_producer ? "PRODUCER" : "CONSUMER",
                dfb_binding.accessor_name);
            seen_this_type = true;

            // Forbid binding the same DFB twice in the same role within this kernel (e.g. two CONSUMER
            // bindings under different accessor names). The legitimate multi-binding form is the
            // self-loop pair — one PRODUCER + one CONSUMER — which this allows regardless of whether
            // the two bindings share an accessor name. The same-role same-name case is already caught
            // above (duplicate {PRODUCER,CONSUMER} binding for accessor_name); this closes the
            // different-name gap. "One buffer, two names" in kernel code must be a handle alias
            // (constexpr auto x = dfb::y) over a single binding, not a second binding — two accessors /
            // DataflowBuffer objects for one FIFO break the object<->DFB identity that device-side
            // debug tooling relies on.
            DFBBoundRoles& bound_roles = dfb_bound_roles[dfb_binding.dfb_spec_name];
            bool& role_already_bound = is_producer ? bound_roles.has_producer : bound_roles.has_consumer;
            TT_FATAL(
                !role_already_bound,
                "Kernel '{}' has two {} bindings to DFB '{}' under different accessor names. Within a "
                "kernel a DFB may be bound at most once per role (the only multi-binding form is the "
                "self-loop pair: one PRODUCER + one CONSUMER). To refer to one buffer by multiple names "
                "in kernel code, alias the handle (constexpr auto x = dfb::y) instead of adding a second binding.",
                kernel.unique_id,
                is_producer ? "PRODUCER" : "CONSUMER",
                dfb_binding.dfb_spec_name);
            role_already_bound = true;
        }
    }

    // ------------------------------------------------------------------------
    // Cross-node DFBs: unbound
    // ------------------------------------------------------------------------
    // Every declared cross-node DFB must be bound by some kernel (local DFBs are checked in collection).
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        const DFBSpecName& name = cross_node_dfb.dfb_spec.unique_id;
        TT_FATAL(
            collected.dfb_endpoints.contains(name),
            "CrossNodeDataflowBufferSpec '{}' is defined but not bound by any kernel",
            name);
    }

    // ------------------------------------------------------------------------
    // Semaphore bindings: accessor names
    // ------------------------------------------------------------------------
    for (const auto& kernel : spec.kernels) {
        std::unordered_set<std::string> accessor_names;
        for (const auto& binding : kernel.semaphore_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate semaphore accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' semaphore accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            ValidateAccessorNameLength(kernel.unique_id, "semaphore", binding.accessor_name);
        }
    }

    // ------------------------------------------------------------------------
    // Scratchpads: sizes, accessor names, unbound
    // ------------------------------------------------------------------------
    for (const auto& scratchpad : spec.scratchpads) {
        TT_FATAL(
            scratchpad.size_per_node != 0,
            "ScratchpadSpec '{}' has size_per_node == 0; a scratchpad must reserve a non-zero number of bytes "
            "(did you forget to set size_per_node?).",
            scratchpad.unique_id);
    }
    for (const auto& kernel : spec.kernels) {
        std::unordered_set<std::string> accessor_names;
        for (const auto& binding : kernel.scratchpad_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate scratchpad accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' scratchpad accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            ValidateAccessorNameLength(kernel.unique_id, "scratchpad", binding.accessor_name);
        }
    }
    // Every declared scratchpad must be bound by some kernel: an unbound scratchpad would reserve L1
    // that no kernel can reach.
    for (const auto& scratchpad : spec.scratchpads) {
        TT_FATAL(
            collected.scratchpad_binders.contains(scratchpad.unique_id),
            "ScratchpadSpec '{}' is declared but not bound by any kernel.",
            scratchpad.unique_id);
    }

    // ------------------------------------------------------------------------
    // Tensor bindings: accessor names, binding sequences, unused parameters
    // ------------------------------------------------------------------------
    for (const auto& kernel : spec.kernels) {
        // A tensor binding is legal on both DM and compute kernels:
        //   - a DM kernel can use the binding token to construct a TensorAccessor or LocalTensorAccessor
        //   - a compute kernel can only use LocalTensorAccessor (NOC-free, local-L1 only)

        std::unordered_set<std::string> accessor_names;
        for (const auto& binding : kernel.tensor_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate tensor accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' tensor accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            ValidateAccessorNameLength(kernel.unique_id, "tensor", binding.accessor_name);
        }

        std::unordered_set<std::string> reserved_type_aliases;
        reserved_type_aliases.reserve(accessor_names.size());
        for (const auto& binding_name : accessor_names) {
            reserved_type_aliases.insert(binding_name + "_t");
        }

        std::unordered_set<std::string> sequence_names;
        for (const auto& sequence : kernel.advanced_options.tensor_binding_sequences) {
            TT_FATAL(
                IsValidCppIdentifier(sequence.sequence_name),
                "Kernel '{}' tensor binding sequence_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                sequence.sequence_name);
            TT_FATAL(
                !accessor_names.contains(sequence.sequence_name),
                "Kernel '{}' tensor binding sequence_name '{}' collides with a TensorBinding accessor_name",
                kernel.unique_id,
                sequence.sequence_name);
            TT_FATAL(
                !reserved_type_aliases.contains(sequence.sequence_name),
                "Kernel '{}' tensor binding sequence_name '{}' collides with generated type alias '{}'",
                kernel.unique_id,
                sequence.sequence_name,
                sequence.sequence_name);
            auto [sit, sinserted] = sequence_names.insert(sequence.sequence_name);
            TT_FATAL(
                sinserted,
                "Kernel '{}' has duplicate tensor binding sequence_name '{}'",
                kernel.unique_id,
                sequence.sequence_name);

            std::unordered_set<std::string> member_names;
            for (const auto& member : sequence.members) {
                TT_FATAL(
                    accessor_names.contains(member),
                    "Kernel '{}' tensor binding sequence '{}' references unknown tensor accessor_name '{}'",
                    kernel.unique_id,
                    sequence.sequence_name,
                    member);
                auto [mit, minserted] = member_names.insert(member);
                TT_FATAL(
                    minserted,
                    "Kernel '{}' tensor binding sequence '{}' has duplicate member '{}'",
                    kernel.unique_id,
                    sequence.sequence_name,
                    member);
            }
        }
    }

    // Every declared TensorParameter must be referenced by some kernel binding or a DFB
    // borrowed_from. (Same usage requirement as DFBs; an unused tensor parameter is a user error.)
    // A borrowed-memory DFB uses its backing TensorParameter via DataflowBufferSpec::borrowed_from
    // (resolved by name at runtime) even when no kernel binds it, so that counts as a use. Only local
    // DFBs are walked: borrowed memory is a local-L1 feature (cross-node DFBs are runtime-unsupported).
    // Existence of the borrowed_from referent is validated in ValidateProgramSpec's borrowed-DFB checks.
    std::unordered_set<TensorParamName> used_tensor_parameters;
    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.tensor_bindings) {
            used_tensor_parameters.insert(binding.tensor_parameter_name);
        }
    }
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.borrowed_from.has_value()) {
            used_tensor_parameters.insert(*dfb.borrowed_from);
        }
    }
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        TT_FATAL(
            used_tensor_parameters.contains(tensor_parameter.unique_id),
            "TensorParameter '{}' is defined but not bound by any kernel",
            tensor_parameter.unique_id);
    }

    // ------------------------------------------------------------------------
    // PrefetcherPipes: accessor names, relays, unused parameters
    // ------------------------------------------------------------------------
    for (const auto& kernel : spec.kernels) {
        std::unordered_set<std::string> accessor_names;
        // A kernel binds a given pipe at most once, within and across accessors: a second binding
        // would be a second device object over the same credit counters (two names for one pipe is a
        // handle alias, not a binding).
        std::unordered_set<PrefetcherPipeParamName> bound_pipes;
        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate PrefetcherPipe accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' PrefetcherPipe accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            ValidateAccessorNameLength(kernel.unique_id, "PrefetcherPipe", binding.accessor_name);
            TT_FATAL(
                !binding.pipe_parameter_names.empty(),
                "Kernel '{}' PrefetcherPipe accessor '{}' names no PrefetcherPipeParameter",
                kernel.unique_id,
                binding.accessor_name);
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                auto [pit, pinserted] = bound_pipes.insert(pipe_name);
                TT_FATAL(
                    pinserted,
                    "Kernel '{}' binds PrefetcherPipeParameter '{}' more than once (latest under accessor_name '{}'). "
                    "A kernel may bind a given pipe at most once.",
                    kernel.unique_id,
                    pipe_name,
                    binding.accessor_name);
            }
        }
    }
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        TT_FATAL(
            cross_node_dfb.dfb_spec.advanced_options.prefetcher_pipe_relays.empty(),
            "CrossNodeDataflowBufferSpec '{}' sets prefetcher_pipe_relays; only a local DFB can relay a "
            "PrefetcherPipe",
            cross_node_dfb.dfb_spec.unique_id);
    }
    // Every declared PrefetcherPipeParameter must be used by a kernel binding or a relay DFB.
    // (An unused pipe parameter would demand a run arg nothing reads.)
    std::unordered_set<PrefetcherPipeParamName> used_pipes;
    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            used_pipes.insert(binding.pipe_parameter_names.begin(), binding.pipe_parameter_names.end());
        }
    }
    for (const auto& [pipe_name, relays] : collected.prefetcher_pipe_relays) {
        used_pipes.insert(pipe_name);
    }
    for (const auto& pipe_parameter : spec.advanced_options.prefetcher_pipe_parameters) {
        TT_FATAL(
            used_pipes.contains(pipe_parameter.unique_id),
            "PrefetcherPipeParameter '{}' is defined but not bound by any kernel or relay DFB",
            pipe_parameter.unique_id);
    }
}

}  // namespace tt::tt_metal::experimental
