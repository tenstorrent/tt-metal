// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/kernel_lowering.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"

namespace tt::tt_metal::experimental {

namespace {

// ----------------------------------------------------------------------------
// MakeKernelSource: Create a KernelSource from a KernelSpec
// ----------------------------------------------------------------------------

KernelSource MakeKernelSource(const KernelSpec& kernel_spec, ContextId context_id) {
    return std::visit(
        [&](const auto& src) -> KernelSource {
            using T = std::decay_t<decltype(src)>;
            if constexpr (std::is_same_v<T, std::filesystem::path>) {
                TT_FATAL(!src.empty(), "KernelSpec '{}' has empty source file path", kernel_spec.unique_id);
                return KernelSource::from_path(context_id, src);
            } else if constexpr (std::is_same_v<T, KernelSpec::SourceCode>) {
                TT_FATAL(!src.code.empty(), "KernelSpec '{}' has empty inline source code", kernel_spec.unique_id);
                return KernelSource::from_source(src.code);
            } else {
                static_assert(!sizeof(T*), "Unhandled KernelSpec::source alternative");
            }
        },
        kernel_spec.source);
}

// ----------------------------------------------------------------------------
// MakeGen1DataMovementConfig: Create a DataMovementConfig (WH/BH) from a KernelSpec
// ----------------------------------------------------------------------------

// (Temporary) Shims
// ProgramSpec APIs use vector<pair> for conceptually map-like data structures.
// This is deliberate, done so ProgramSpec stays hashable for TTNN's program caching.
// For now, just convert to the map types that the core runtime expects.
// TODO: Fix this inefficiency eventually.
std::unordered_map<std::string, uint32_t> to_named_compile_args_map(const KernelSpec::CompileTimeArgs& bindings) {
    return std::unordered_map<std::string, uint32_t>(bindings.begin(), bindings.end());
}
std::map<std::string, std::string> to_defines_map(const KernelSpec::CompilerOptions::Defines& defines) {
    return std::map<std::string, std::string>(defines.begin(), defines.end());
}

DataMovementConfig MakeGen1DataMovementConfig(const KernelSpec& kernel_spec) {
    TT_FATAL(kernel_spec.is_data_movement_kernel(), "Expected a DM kernel");
    const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel_spec.hw_config);
    TT_FATAL(
        dm_config.config_1xx.has_value(),
        "KernelSpec '{}' is a data-movement kernel on Gen1 but has no config_1xx processor/NOC. "
        "Those settings are required to build a Gen1 data-movement kernel. Supply a DataMovement1XXConfig "
        "(e.g. CreateReaderDataMovementConfig()/CreateWriterDataMovementConfig()).",
        kernel_spec.unique_id);
    const auto& gen1 = *dm_config.config_1xx;

    return DataMovementConfig{
        .processor = gen1.processor,
        .noc = gen1.noc,
        .noc_mode = gen1.noc_mode,
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// ----------------------------------------------------------------------------
// BuildUnpackToDestModeVector:
// Translate the Metal 2.0 user-facing DFB-name->mode map into the (gross)
// CB-indexed vector that the JIT data-format machinery expects.
//
// This DFB/CB translation layer is confusing. The gory details:
//   - The JIT consumer (get_unpack_dst_formats) will read unpack_to_dest_mode at
//     index cb_id, where cb_id is the slot used by set_dfb_data_fmt_and_tile
//     in buf_dataformat_arr (aka, dfb->id).
//   - The unpack_mode for a DFB "d" needs to be at unpack_modes[d->id]
//   - The vector must be at least max_dfbs long, or the consumer gets angry
//     (it iterates buf_formats up to max_dfbs).
//   - This is true on WH, BH, and Quasar. (Yes, Quasar too.)
//
// What is the max DFB slot count?
//   - WH/BH: Hardcoded as max_dfbs. Different number on WH vs. BH.
//   - Quasar has a variable cap, based on tile-counter registers.
//     In actual practice, we'll run out LONG before we get the HAL-reported
//     limit of 64.
// ----------------------------------------------------------------------------

std::vector<UnpackToDestMode> BuildUnpackToDestModeVector(
    const ComputeHardwareConfig::ComputeUnpackModes& user_modes,
    const DFBNameToSlotMap& dfb_name_to_slot,
    const Hal& hal) {
    const uint32_t max_dfbs = hal.get_num_dataflow_buffers();
    std::vector<UnpackToDestMode> unpack_modes(max_dfbs, UnpackToDestMode::Default);
    for (const auto& [dfb_name, mode] : user_modes) {
        // Indexed by device slot: this vector is consumed by the HLK alongside the CB-indexed data
        // formats, which set_dfb_data_fmt_and_tile also keys by slot.
        uint32_t dfb_slot = dfb_name_to_slot.at(dfb_name);
        // This TT_FATAL is unreachable, provided that validation wasn't skipped.
        TT_FATAL(
            dfb_slot < max_dfbs,
            "Internal Error: DFB '{}' has device slot {} which exceeds the JIT data-format "
            "slot count ({}); compute kernels cannot reference DFBs past this limit",
            dfb_name,
            dfb_slot,
            max_dfbs);
        // Public UnpackMode -> internal UnpackToDestMode. UnpackToDest keeps full FP32 by
        // unpacking straight to Dest; UnpackToSrc is the SrcA/B path (the internal "Default").
        unpack_modes[dfb_slot] =
            (mode == UnpackMode::UnpackToDest) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
    }
    return unpack_modes;
}

// ----------------------------------------------------------------------------
// MakeGen1ComputeConfig: Create a ComputeConfig (WH/BH) from a KernelSpec
// ----------------------------------------------------------------------------

ComputeConfig MakeGen1ComputeConfig(
    const KernelSpec& kernel_spec, const DFBNameToSlotMap& dfb_name_to_slot, const Hal& hal) {
    TT_FATAL(kernel_spec.is_compute_kernel(), "Expected a compute kernel");
    const auto& compute_config = std::get<ComputeHardwareConfig>(kernel_spec.hw_config);

    std::vector<UnpackToDestMode> unpack_dst_modes =
        BuildUnpackToDestModeVector(compute_config.unpack_modes, dfb_name_to_slot, hal);

    // bfp_pack_precision_mode is TT-1.x.x-only. If config_1xx is not engaged, use the
    // historical default (Approximate).
    const Precision bfp_pack_precision_mode = compute_config.config_1xx.has_value()
                                                  ? compute_config.config_1xx->bfp_pack_precision_mode
                                                  : Precision::Approximate;

    return ComputeConfig{
        .math_fidelity = compute_config.fpu_math_fidelity,
        .fp32_dest_acc_en = compute_config.enable_32_bit_dest,
        .dst_full_sync_en = !compute_config.double_buffer_dest,
        .unpack_to_dest_mode = unpack_dst_modes,
        .bfp8_pack_precise = (bfp_pack_precision_mode == Precision::Precise),
        .math_approx_mode = (compute_config.sfpu_precision_mode == Precision::Approximate),
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// ----------------------------------------------------------------------------
// MakeQuasarDataMovementConfig: Create a QuasarDataMovementConfig from a KernelSpec
// ----------------------------------------------------------------------------

experimental::quasar::QuasarDataMovementConfig MakeQuasarDataMovementConfig(const KernelSpec& kernel_spec) {
    TT_FATAL(kernel_spec.is_data_movement_kernel(), "Expected a DM kernel");

    return experimental::quasar::QuasarDataMovementConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .is_legacy_kernel = false,
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// ----------------------------------------------------------------------------
// MakeGen2ComputeConfig: Create a QuasarComputeConfig from a KernelSpec
// ----------------------------------------------------------------------------

experimental::quasar::QuasarComputeConfig MakeGen2ComputeConfig(
    const KernelSpec& kernel_spec, const DFBNameToSlotMap& dfb_name_to_slot, const Hal& hal) {
    TT_FATAL(kernel_spec.is_compute_kernel(), "Expected a compute kernel");
    const auto& compute_config = std::get<ComputeHardwareConfig>(kernel_spec.hw_config);

    std::vector<UnpackToDestMode> unpack_dst_modes =
        BuildUnpackToDestModeVector(compute_config.unpack_modes, dfb_name_to_slot, hal);

    return experimental::quasar::QuasarComputeConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .math_fidelity = compute_config.fpu_math_fidelity,
        .fp32_dest_acc_en = compute_config.enable_32_bit_dest,
        .dst_full_sync_en = !compute_config.double_buffer_dest,
        .unpack_to_dest_mode = unpack_dst_modes,
        .math_approx_mode = (compute_config.sfpu_precision_mode == Precision::Approximate),
        .compile_args = {},  // Compile args are passed via named_compile_args
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// --------------------------------------------------------------------------------
// GetDMProcessorSet: Convert a DMProcessorMask to a set of DataMovementProcessor
// --------------------------------------------------------------------------------

std::set<DataMovementProcessor> GetDMProcessorSet(DMProcessorMask mask) {
    std::set<DataMovementProcessor> processors;
    for (uint8_t i = 0; i < QUASAR_DM_CORES_PER_NODE; ++i) {
        if (mask.test(i)) {
            processors.insert(static_cast<DataMovementProcessor>(i));
        }
    }
    return processors;
}

// ------------------------------------------------------------------------------------------
// GetComputeProcessorSet: Convert a ComputeEngineMask to a set of QuasarComputeProcessor
// ------------------------------------------------------------------------------------------
//
// The ComputeEngineMask represents the active Tensix engines on a node.
// (Based on the number of compute kernel threads running on that node.)
// Each Tensix engine has 4 compute processors.
// So if bit i is set in the mask, we include all 4 processors for that engine:
//   Engine 0 -> NEO_0_COMPUTE_{0,1,2,3}
//   Engine 1 -> NEO_1_COMPUTE_{0,1,2,3}
//   etc.

std::set<experimental::quasar::QuasarComputeProcessor> GetComputeProcessorSet(ComputeEngineMask mask) {
    using QuasarComputeProcessor = experimental::quasar::QuasarComputeProcessor;
    constexpr uint8_t PROCESSORS_PER_ENGINE = experimental::quasar::QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE;

    std::set<QuasarComputeProcessor> processors;
    for (uint8_t engine = 0; engine < QUASAR_TENSIX_ENGINES_PER_NODE; ++engine) {
        if (mask.test(engine)) {
            // Add all 4 compute processors for this engine
            for (uint8_t proc = 0; proc < PROCESSORS_PER_ENGINE; ++proc) {
                uint8_t processor_id = (engine * PROCESSORS_PER_ENGINE) + proc;
                processors.insert(static_cast<QuasarComputeProcessor>(processor_id));
            }
        }
    }
    return processors;
}

// A KernelSpec lowered into a core-runtime Kernel, plus the runtime-arg schema the ProgramImpl
// registers for it (consumed by ValidateProgramRunArgs / SetProgramRunArgs).
struct LoweredKernel {
    std::shared_ptr<Kernel> kernel;
    detail::ProgramImpl::KernelRTASchema rta_schema;
};

LoweredKernel LowerKernel(
    const KernelSpec& kernel_spec,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    const KernelRiscMaskMap& risc_masks,
    const ProgramResources& resources,
    const Hal& hal,
    ContextId context_id) {
    KernelSource kernel_src = MakeKernelSource(kernel_spec, context_id);
    const NodeRangeSet& node_ranges = collected.kernel_node_set.at(kernel_spec.unique_id);

    // Make the local accessor name -> DFB device slot map for this kernel
    const KernelResourceBindings resource_bindings = ResolveKernelResourceBindings(kernel_spec, collected, resources);
    const tt::tt_metal::DataflowBufferBindingHandleMap& dfb_handles = resource_bindings.dfbs;
    const tt::tt_metal::SemaphoreBindingHandleMap& semaphore_handles = resource_bindings.semaphores;

    // Resolve TensorBindings for this kernel:
    //  - pack each binding's pre-resolved CTA payload into the kernel's positional CTA buffer
    //    (after the user CTA-vararg prefix)
    //  - assign each binding a slot in the kernel's CRTA buffer (TensorBinding address section)
    const auto& user_named_crtas = kernel_spec.runtime_arg_schema.common_runtime_arg_names;
    const auto& cta_varargs = kernel_spec.advanced_options.compile_time_varargs;
    const uint32_t vararg_cta_count = static_cast<uint32_t>(cta_varargs.size());
    TensorBindingsForKernel ta_bindings = ResolveTensorBindingsForKernel(
        kernel_spec,
        resources.tensor_parameters,
        /*base_named_crta_count=*/user_named_crtas.size(),
        /*base_cta_offset=*/vararg_cta_count);

    // Create TensorBindingHandles for this kernel
    const std::vector<TensorBindingHandle>& tensor_binding_handles = ta_bindings.handles;

    // Resolve scratchpad bindings for this kernel. The scratchpad section follows the TensorBinding
    // section and precedes varargs, so it begins at the tensor-binding resolution's (pre-scratchpad)
    // vararg offset; we then push the vararg section out by the scratchpad section's width so the
    // crta_layout that flows into the kernel ctor reflects all four sections.
    ScratchpadBindingsForKernel sp_bindings = ResolveScratchpadBindingsForKernel(
        kernel_spec,
        collected.scratchpad_by_name,
        /*scratchpad_base_crta_word=*/ta_bindings.crta_layout.vararg_section_offset);
    ta_bindings.crta_layout.scratchpad_section_words = sp_bindings.section_words;
    ta_bindings.crta_layout.vararg_section_offset += sp_bindings.section_words;

    // Named-args schema fields passed to the Kernel ctor. The names are used at JIT time to
    // emit kernel_args_generated.h and factor into the kernel cache key. The TensorBinding
    // address section is tracked separately (via tensor_binding_handles), so we pass the user
    // CRTA list through unchanged.
    const auto& named_rtas = kernel_spec.runtime_arg_schema.runtime_arg_names;

    // Positional CTAs: [ user CTA varargs | TensorBinding CTA payloads ]
    std::vector<uint32_t> compile_args = cta_varargs;
    compile_args.insert(compile_args.end(), ta_bindings.cta_words.begin(), ta_bindings.cta_words.end());

    // Create the kernel object
    std::shared_ptr<Kernel> kernel;

    // Kernel creation APIs accept a "is_metal2_kernel" bool, which fences Metal 2.0 JIT machinery
    constexpr bool is_metal2_kernel = true;

    if (is_gen2_arch(hal)) {
        uint16_t risc_mask = risc_masks.at(&kernel_spec);
        if (kernel_spec.is_data_movement_kernel()) {
            auto config = MakeQuasarDataMovementConfig(kernel_spec);
            config.compile_args = std::move(compile_args);
            auto processors = GetDMProcessorSet(DMProcessorMask{static_cast<uint8_t>(risc_mask & 0xFF)});
            kernel = std::make_shared<experimental::quasar::QuasarDataMovementKernel>(
                context_id,
                kernel_src,
                node_ranges,
                config,
                processors,
                is_metal2_kernel,
                dfb_handles,
                semaphore_handles,
                named_rtas,
                user_named_crtas,
                tensor_binding_handles,
                ta_bindings.crta_layout);
        } else {
            auto config = MakeGen2ComputeConfig(kernel_spec, resources.dfbs.slot, hal);
            config.compile_args = std::move(compile_args);
            auto processors = GetComputeProcessorSet(ComputeEngineMask{static_cast<uint8_t>(risc_mask >> 8)});
            kernel = std::make_shared<experimental::quasar::QuasarComputeKernel>(
                context_id,
                kernel_src,
                node_ranges,
                config,
                processors,
                is_metal2_kernel,
                dfb_handles,
                semaphore_handles,
                named_rtas,
                user_named_crtas,
                tensor_binding_handles,
                ta_bindings.crta_layout);
        }
    } else {  // gen1
        if (kernel_spec.is_data_movement_kernel()) {
            auto config = MakeGen1DataMovementConfig(kernel_spec);
            config.compile_args = std::move(compile_args);
            kernel = std::make_shared<DataMovementKernel>(
                context_id,
                kernel_src,
                node_ranges,
                config,
                is_metal2_kernel,
                dfb_handles,
                semaphore_handles,
                named_rtas,
                user_named_crtas,
                tensor_binding_handles,
                ta_bindings.crta_layout);
        } else {
            auto config = MakeGen1ComputeConfig(kernel_spec, resources.dfbs.slot, hal);
            config.compile_args = std::move(compile_args);
            // Bake the compute semaphore's capacity into the kernel (Semaphore::wait_not_full and the
            // SEMINITs read COMPUTE_SEMAPHORE_MAX). At most one compute semaphore per program
            // (ValidateProgramSpec), so at most one define.
            for (const auto& binding : kernel_spec.semaphore_bindings) {
                if (resources.semaphores.scope.at(binding.semaphore_spec_name) != SemScope::COMPUTE_ATOMIC) {
                    continue;
                }
                const auto sem =
                    std::find_if(spec.semaphores.begin(), spec.semaphores.end(), [&](const SemaphoreSpec& s) {
                        return s.unique_id == binding.semaphore_spec_name;
                    });
                // The host option is the source of truth: always bake the resolved capacity, overriding
                // any user-supplied COMPUTE_SEMAPHORE_MAX define. max_value 0 means the default (the 4-bit
                // hardware ceiling, 15), which is what the kernel assumes when the define is absent.
                const uint32_t capacity = (sem != spec.semaphores.end() && sem->advanced_options.max_value != 0)
                                              ? sem->advanced_options.max_value
                                              : 15u;
                config.defines["COMPUTE_SEMAPHORE_MAX"] = std::to_string(capacity);
            }
            kernel = std::make_shared<ComputeKernel>(
                context_id,
                kernel_src,
                node_ranges,
                config,
                is_metal2_kernel,
                dfb_handles,
                semaphore_handles,
                named_rtas,
                user_named_crtas,
                tensor_binding_handles,
                ta_bindings.crta_layout);
        }
    }

    // Attach the resolved scratchpad bindings to the kernel (set post-construction: their sizes are
    // part of the kernel cache key, so this must run before the kernel is compiled). allocate_scratchpads
    // will later fill each handle's allocated_address.
    kernel->set_scratchpad_binding_handles(std::move(sp_bindings.handles));

    // PrefetcherPipe accessors -> program slot ids (also part of the kernel cache key).
    if (!resource_bindings.prefetcher_pipes.empty()) {
        kernel->set_prefetcher_pipe_binding_handles(resource_bindings.prefetcher_pipes);
    }

    std::vector<TensorBindingSequenceHandle> tensor_binding_sequences;
    tensor_binding_sequences.reserve(kernel_spec.advanced_options.tensor_binding_sequences.size());
    for (const auto& sequence : kernel_spec.advanced_options.tensor_binding_sequences) {
        tensor_binding_sequences.push_back(
            TensorBindingSequenceHandle{.sequence_name = sequence.sequence_name, .members = sequence.members});
    }
    kernel->set_tensor_binding_sequences(std::move(tensor_binding_sequences));

    // Prefix length for device get_compile_time_vararg* bounds (values are in compile_time_args_).
    kernel->set_compile_time_vararg_count(vararg_cta_count);

    // Build the RTA+CRTA schema (named lists + vararg counts) for the ProgramImpl.
    // Used by ValidateProgramRunArgs and SetProgramRunArgs to validate and serialize
    // the user-provided values at dispatch time.
    //
    // User-facing vararg RTA specification (see kernel_spec.hpp):
    //   - num_runtime_varargs (scalar): default count applied to every node the kernel
    //     runs on.
    //   - num_runtime_varargs_per_node (optional): sparse per-node overrides on top of
    //     the scalar default. Unlisted nodes fall back to the scalar.
    // We apply the scalar first across target_nodes, then overlay each override entry.
    // An explicit override of 0 erases the scalar-default entry so run-params treats
    // that node as having no varargs (rather than requiring an "empty" value list).
    // Overlapping override entries (two entries covering the same node) are an error.
    const auto& user_schema = kernel_spec.runtime_arg_schema;
    detail::ProgramImpl::KernelRTASchema runtime_schema;
    runtime_schema.runtime_arg_names = user_schema.runtime_arg_names;

    // Pass the user CRTA list through.
    // NOTE: The TensorBinding address section is tracked separately on the Kernel
    // (via tensor_binding_handles) and its slot offsets are baked into each binding handle's
    // addr_crta_offset; SetProgramRunArgs uses BOTH to assemble the per-enqueue CRTA buffer.
    runtime_schema.common_runtime_arg_names = user_named_crtas;

    // Precompute the name -> slot-index maps so the hot UpdateProgramRunArgs path does O(1)
    // lookups rather than rebuilding a map per call.
    for (size_t i = 0; i < runtime_schema.runtime_arg_names.size(); ++i) {
        runtime_schema.runtime_arg_name_to_slot.emplace(runtime_schema.runtime_arg_names[i], i);
    }
    for (size_t i = 0; i < runtime_schema.common_runtime_arg_names.size(); ++i) {
        runtime_schema.common_runtime_arg_name_to_slot.emplace(runtime_schema.common_runtime_arg_names[i], i);
    }

    // Varargs schema now lives on KernelAdvancedOptions.
    const uint32_t num_runtime_varargs = kernel_spec.advanced_options.num_runtime_varargs;
    const uint32_t num_common_runtime_varargs = kernel_spec.advanced_options.num_common_runtime_varargs;
    const bool has_per_node_override = !kernel_spec.advanced_options.num_runtime_varargs_per_node.empty();

    if (num_runtime_varargs > 0) {
        for (const NodeRange& range : node_ranges.ranges()) {
            for (const NodeCoord& node : range) {
                runtime_schema.num_runtime_varargs_per_node[node] = num_runtime_varargs;
            }
        }
    }
    if (has_per_node_override) {
        std::unordered_set<NodeCoord> seen_overrides;
        for (const auto& [nodes_spec, num_varargs] : kernel_spec.advanced_options.num_runtime_varargs_per_node) {
            const NodeRangeSet expanded = to_node_range_set(nodes_spec);
            for (const NodeRange& range : expanded.ranges()) {
                for (const NodeCoord& node : range) {
                    const bool inserted = seen_overrides.insert(node).second;
                    TT_FATAL(
                        inserted,
                        "KernelSpec '{}' num_runtime_varargs_per_node has overlapping entries "
                        "for node {}",
                        kernel_spec.unique_id,
                        node.str());
                    if (num_varargs > 0) {
                        runtime_schema.num_runtime_varargs_per_node[node] = num_varargs;
                    } else {
                        // Explicit zero override: drop any scalar-default entry so
                        // run-params treats this node as missing (→ 0 expected).
                        runtime_schema.num_runtime_varargs_per_node.erase(node);
                    }
                }
            }
        }
    }
    runtime_schema.num_common_runtime_varargs = num_common_runtime_varargs;
    return LoweredKernel{.kernel = std::move(kernel), .rta_schema = std::move(runtime_schema)};
}

}  // namespace

void AddKernels(
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    const KernelRiscMaskMap& risc_masks,
    const ProgramResources& resources,
    const Hal& hal,
    detail::ProgramImpl& program_impl) {
    for (const KernelSpec& kernel_spec : spec.kernels) {
        LoweredKernel lowered =
            LowerKernel(kernel_spec, spec, collected, risc_masks, resources, hal, program_impl.get_context_id());
        KernelHandle handle = program_impl.add_kernel(lowered.kernel, HalProgrammableCoreType::TENSIX);
        program_impl.register_kernel_spec_name(kernel_spec.unique_id.get(), handle);
        program_impl.register_kernel_rta_schema(kernel_spec.unique_id.get(), lowered.rta_schema);
    }
}

}  // namespace tt::tt_metal::experimental
