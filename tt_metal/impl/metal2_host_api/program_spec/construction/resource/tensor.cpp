// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // fmt::formatter<tt::DataFormat> for TT_FATAL messages
#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental {

namespace {

LLKMetadata LLKMetadataFromTensorSpec(const TensorSpec& spec) {
    return LLKMetadata{.format = datatype_to_dataformat_converter(spec.data_type()), .tile = spec.tile()};
}

}  // namespace

namespace {

// Resolve a TensorParameter's static layout into a CTA payload + an extra CRTA word
// count for any runtime-resolved fields.
//
// CTA layout produced:
//  - word 0 is the args_config raw byte
//  - word 1 is aligned_page_size
// For sharded tensors only:
//  - word 2 is rank
//  - word 3 is num_banks
//  - remaining words: per-dim tensor_shape_in_pages (omitted if dynamic_tensor_shape),
//    per-dim shard_shape_in_pages, and packed bank coordinates (two per uint32)
//
// The tensor base address always lives in CRTAs (filled in per-enqueue from the
// corresponding TensorArgument). When dynamic_tensor_shape is set on a sharded tensor,
// the runtime tensor's shape is also written into CRTAs at enqueue time, in
// `extra_crta_words` slots immediately after the address slot.
// (See also ResolveTensorBindingsForKernel below.)
ResolvedTensorParameter ResolveTensorParameterStaticCTAs(
    const TensorParameter& tensor_parameter, const distributed::MeshDevice& mesh_device) {
    const TensorSpec& spec = tensor_parameter.spec;
    const MemoryConfig& memory_config = spec.memory_config();
    const BufferType buffer_type = memory_config.buffer_type();
    const bool is_dram = (buffer_type == BufferType::DRAM);
    const bool is_sharded = memory_config.is_sharded();
    // The tensor SHAPE only rides the CTA/CRTA payload for a sharded tensor: an interleaved payload
    // never carried it in the first place, and the device-side accessor doesn't read it. So this
    // particular induction is sharded-only.
    //
    // That is a statement about the shape words, NOT about the flag. dynamic_tensor_shape is a
    // dynamic relaxation on every layout -- see dyn_page immediately below, which moves an
    // interleaved ROW-MAJOR page size out of the CTAs. (match_padded_shape_only is the flag that is
    // purely a host-side validation loosening with no CTA/CRTA effect; do not transplant its
    // description onto this one.)
    const bool dyn_shape = tensor_parameter.relaxations.dynamic_tensor_shape && is_sharded;
    // dynamic_tensor_shape lets the bound tensor's logical shape vary. For an interleaved ROW-MAJOR
    // tensor the page size (= last_dim_width * elem_size) is part of that varying shape, so it must
    // ride a runtime CRTA word too -- otherwise it goes stale on a program-cache hit and the
    // accessor strides by the wrong number of bytes. We induce that here rather than expose a flag
    // for it: a useful page-size change is ALWAYS a shape change on row-major (you can't vary the
    // width without varying the logical shape), so there is no "page size varies but shape doesn't"
    // case to give a flag to. Tiled page size is dtype-fixed and sharded page size is spec-fixed, so
    // neither triggers this; sharded dynamic_tensor_shape carries shape-in-pages words instead
    // (dyn_shape above). dyn_shape and dyn_page are mutually exclusive by layout.
    //
    // match_page_size opts out of the induction, for the CONVERSE case: shape varies, width does
    // not. That implication runs only one way, so the flag does not reopen the reasoning above --
    // it declares a narrower equivalence class in which the page size is pinned, and the match
    // enforces it (tensor_spec_relaxations.cpp), so the CTA below cannot go stale.
    const bool dyn_page = tensor_parameter.relaxations.dynamic_tensor_shape &&
                          !tensor_parameter.relaxations.match_page_size && !is_sharded &&
                          spec.layout() == Layout::ROW_MAJOR;

    tensor_accessor::ArgsConfig args_config;
    if (is_sharded) {
        args_config.set(tensor_accessor::ArgConfig::Sharded);
    }
    if (is_dram) {
        args_config.set(tensor_accessor::ArgConfig::IsDram);
    }
    if (dyn_shape) {
        args_config.set(tensor_accessor::ArgConfig::RuntimeTensorShape);
    }
    if (dyn_page) {
        args_config.set(tensor_accessor::ArgConfig::RuntimePageSize);
    }

    // aligned_page_size: align the unaligned page size up to the buffer-type alignment.
    const size_t unaligned_page_size = spec.compute_page_size_bytes();
    const uint32_t alignment = mesh_device.allocator()->get_alignment(buffer_type);
    const size_t aligned_page_size = align(unaligned_page_size, static_cast<size_t>(alignment));
    TT_FATAL(
        aligned_page_size <= std::numeric_limits<uint32_t>::max(),
        "TensorParameter '{}' aligned page size {} exceeds uint32_t max {}",
        tensor_parameter.unique_id,
        aligned_page_size,
        std::numeric_limits<uint32_t>::max());

    ResolvedTensorParameter result{.llk_metadata = LLKMetadataFromTensorSpec(spec)};
    std::vector<uint32_t>& cta_payload = result.cta_payload;

    // Common header (always emitted, sharded or not):
    cta_payload.push_back(args_config.raw());
    // If the page size is static, it rides as a CTA.
    // (If it's dynamic, it will live in a CRTA word instead.)
    if (!dyn_page) {
        cta_payload.push_back(static_cast<uint32_t>(aligned_page_size));
    } else {
        TT_FATAL(!is_sharded, "Internal error: dynamic page size should not occur on a sharded tensor parameter");

        // One runtime field: the page size, re-derived from the bound buffer each dispatch
        // and emitted immediately after the base-address word (see EmitBindingCrtaValues).
        result.extra_crta_words = 1;
        result.runtime_field_is_page_size = true;
    }

    // The rest of the logic in this function pertains to sharded tensors only.
    // Early return for a non-sharded tensor.
    if (!is_sharded) {
        return result;
    }

    //////////////////////////////////
    // Sharded tensor handling
    //////////////////////////////////

    // Sharded: emit rank, num_banks, tensor_shape_in_pages (CTA only when static),
    // shard_shape_in_pages, bank_coords.
    const BufferShardingArgs sharding_args = spec.compute_buffer_sharding_args();
    const std::optional<BufferDistributionSpec>& bds_opt = sharding_args.buffer_distribution_spec();
    TT_FATAL(
        bds_opt.has_value(),
        "TensorParameter '{}' is sharded but TensorSpec produced no BufferDistributionSpec",
        tensor_parameter.unique_id);
    const BufferDistributionSpec& bds = *bds_opt;

    const Shape& tensor_shape = bds.tensor_shape_in_pages();
    const Shape& shard_shape = bds.shard_shape_in_pages();
    const std::vector<CoreCoord>& bank_coords = bds.cores();
    const size_t rank = tensor_shape.rank();
    const size_t n_banks = bank_coords.size();

    cta_payload.push_back(static_cast<uint32_t>(rank));
    TT_FATAL(
        n_banks < tensor_accessor::ShardContiguousBit,
        "TensorParameter '{}' has too many banks ({}) to pack the shard-contiguous flag",
        tensor_parameter.unique_id,
        n_banks);
    cta_payload.push_back(tensor_accessor::pack_num_banks(
        static_cast<uint32_t>(n_banks), bds.shard_distribution_strategy() == ShardDistributionStrategy::CONTIGUOUS_1D));

    if (!dyn_shape) {
        for (size_t i = 0; i < rank; ++i) {
            cta_payload.push_back(static_cast<uint32_t>(tensor_shape[i]));
        }
    } else {
        // Shape lives in CRTAs (one word per dim, written at enqueue time from the bound MeshTensor).
        result.extra_crta_words = static_cast<uint32_t>(rank);
    }
    for (size_t i = 0; i < rank; ++i) {
        cta_payload.push_back(static_cast<uint32_t>(shard_shape[i]));
    }

    // Bank coords packed two-per-uint32.
    // Non-DRAM coords are virtualized; DRAM coords are kept logical (DRAM bank id == logical x).
    const CoreType core_type = is_dram ? CoreType::DRAM : CoreType::WORKER;
    auto resolve_coord = [&](const CoreCoord& logical) -> CoreCoord {
        if (is_dram) {
            return logical;
        }
        return mesh_device.virtual_core_from_logical_core(logical, core_type);
    };
    for (size_t i = 0; i < n_banks; i += 2) {
        const CoreCoord c1 = resolve_coord(bank_coords[i]);
        if (i + 1 < n_banks) {
            const CoreCoord c2 = resolve_coord(bank_coords[i + 1]);
            cta_payload.push_back(((c2.x & 0xFF) << 24) | ((c2.y & 0xFF) << 16) | ((c1.x & 0xFF) << 8) | (c1.y & 0xFF));
        } else {
            cta_payload.push_back(((c1.x & 0xFF) << 8) | (c1.y & 0xFF));
        }
    }

    return result;
}

}  // namespace

// Resolve the tensor bindings for a single kernel:
//  1. Walk the kernel's tensor_bindings in declaration order
//  2. Pack each binding's CTA payload into a contiguous positional buffer
//  3. Assign each binding a slot in the kernel's TensorBinding section of the CRTA buffer.
//     The section is structurally separate, immediately following the user-named CRTAs.
//     Each binding occupies (1 + extra_crta_words) words: the always-present base address
//     word, plus any runtime accessor field words (e.g. shape, when dynamic_tensor_shape
//     is set on a sharded TensorParameter).
//  4. Record the resulting CRTA buffer layout (the three section sizes) on the output, so
//     the headergen and runtime can consult it directly instead of re-summing the bindings.
//
// (SetProgramRunArgs will fill the address slots and runtime field slots at enqueue
// time, extracting info from the TensorArgs.)
TensorBindingsForKernel ResolveTensorBindingsForKernel(
    const KernelSpec& kernel,
    const std::unordered_map<TensorParamName, ResolvedTensorParameter>& resolved_tensor_parameters,
    size_t base_named_crta_count,
    uint32_t base_cta_offset) {
    TensorBindingsForKernel out;
    out.handles.reserve(kernel.tensor_bindings.size());

    // Absolute word index into the unified positional CTA buffer:
    //   [ CTA varargs (base_cta_offset words) | TensorBinding payloads ... ]
    uint32_t cta_word_offset = base_cta_offset;
    size_t crta_word_index = base_named_crta_count;
    uint32_t binding_section_words = 0;
    for (const auto& binding : kernel.tensor_bindings) {
        const ResolvedTensorParameter& resolved = resolved_tensor_parameters.at(binding.tensor_parameter_name);
        const std::vector<uint32_t>& binding_ctas = resolved.cta_payload;

        TensorBindingHandle handle;
        handle.accessor_name = binding.accessor_name;
        handle.tensor_parameter_name = binding.tensor_parameter_name.get();
        handle.cta_offset = cta_word_offset;
        handle.addr_crta_offset = static_cast<uint32_t>(crta_word_index * sizeof(uint32_t));
        handle.num_runtime_field_crta_words = resolved.extra_crta_words;
        handle.runtime_field_is_page_size = resolved.runtime_field_is_page_size;
        handle.llk_metadata = resolved.llk_metadata;

        out.cta_words.insert(out.cta_words.end(), binding_ctas.begin(), binding_ctas.end());
        cta_word_offset += static_cast<uint32_t>(binding_ctas.size());
        const uint32_t binding_words = 1u + resolved.extra_crta_words;
        crta_word_index += binding_words;
        binding_section_words += binding_words;

        out.handles.push_back(std::move(handle));
    }

    out.crta_layout.num_named_words = static_cast<uint32_t>(base_named_crta_count);
    out.crta_layout.binding_section_words = binding_section_words;
    out.crta_layout.vararg_section_offset = static_cast<uint32_t>(base_named_crta_count) + binding_section_words;

    return out;
}

std::unordered_map<TensorParamName, ResolvedTensorParameter> RegisterTensorParameters(
    const ProgramSpec& spec, const distributed::MeshDevice& mesh_device, detail::ProgramImpl& program_impl) {
    // Resolve TensorParameters against the MeshDevice into static CTA payloads.
    //
    // TensorBindings ride two existing kernel-arg channels:
    //   - Static layout (rank, shape, bank coords, ...) flows through the kernel's positional
    //     CTA buffer, after any user CTA-vararg prefix (KernelAdvancedOptions::compile_time_varargs).
    //   - Per-enqueue base address flows through a reserved-prefix named CRTA, appended to
    //     the kernel's user-named CRTAs and filled by SetProgramRunArgs from the
    //     corresponding TensorArgument entry. TensorParameters that opt into a dynamic accessor
    //     field (currently: dynamic_tensor_shape, sharded only) carry additional CRTA words
    //     immediately after the address slot, also filled at enqueue time.
    std::unordered_map<TensorParamName, ResolvedTensorParameter> resolved_tensor_parameters;
    resolved_tensor_parameters.reserve(spec.tensor_parameters.size());
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        resolved_tensor_parameters.emplace(
            tensor_parameter.unique_id, ResolveTensorParameterStaticCTAs(tensor_parameter, mesh_device));
    }

    // Register TensorParameters with the program for ValidateProgramRunArgs to consult at enqueue.
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        program_impl.register_tensor_parameter(
            tensor_parameter.unique_id.get(), tensor_parameter.spec, tensor_parameter.relaxations);
    }
    return resolved_tensor_parameters;
}

}  // namespace tt::tt_metal::experimental
