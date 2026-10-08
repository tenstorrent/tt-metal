// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_set>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec_validation/validate_spec.hpp"

namespace tt::tt_metal::experimental {

namespace {

// A DataFormat whose elements are 32 bits wide, and so cannot be held by a 16-bit Dest register.
// (Note: datum_size() throws on the block/MX formats.)
bool is_32bit_element_format(tt::DataFormat fmt) {
    switch (fmt) {
        case tt::DataFormat::Float32:
        case tt::DataFormat::Int32:
        case tt::DataFormat::UInt32:
        case tt::DataFormat::RawUInt32: return true;
        default: return false;
    }
}

// On Gen1, a DM kernel must supply config_1xx: processor and NOC have no
// default. Architecture still comes from the device, not from which optional is set.
// Gen1 has exactly two DM processors: RISCV_0 (BRISC) and RISCV_1 (NCRISC).
// RISCV_2..RISCV_7 exist only on Gen2/Quasar. Reject them here, mirroring the legacy
// CreateDataMovementKernel "DM0 or DM1 only" guard.
void ValidateGen1DataMovementConfig(const KernelSpec& kernel, tt::ARCH arch) {
    if (!is_gen1_arch(arch) || !kernel.is_data_movement_kernel()) {
        return;
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
void ValidateUnpackModes(const KernelSpec& kernel, const CollectedSpecData& collected, tt::ARCH arch) {
    if (!kernel.is_compute_kernel()) {
        return;
    }
    const auto& compute_config = std::get<ComputeHardwareConfig>(kernel.hw_config);
    const auto& unpack_modes = compute_config.unpack_modes;
    const bool enable_32_bit_dest = compute_config.enable_32_bit_dest;
    const bool is_gen2 = is_gen2_arch(arch);

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

// Validate DM kernel disable_dfb_implicit_sync_for entries.
//
// Implicit sync is a Gen2-only, DM-only mechanism (ISR-based credit posting from NoC
// transaction completion). A DM kernel can opt out per-DFB by listing the DFB's name in
// config_2xx->disable_dfb_implicit_sync_for, or opt out of all the DFBs it binds at
// once via config_2xx->disable_dfb_implicit_sync_for_all.
//
// Per-kernel rule: every listed name references a DFB the kernel binds (typo guard).
// (The cross-kernel agreement rule is in ValidateDFBEndpoints.)
void ValidateImplicitSyncOptOuts(const KernelSpec& kernel) {
    if (!kernel.is_data_movement_kernel()) {
        return;
    }
    const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);
    if (!dm_config.config_2xx.has_value()) {
        return;
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

}  // namespace

void ValidateKernelHardwareConfig(const KernelSpec& kernel, const ValidationContext& ctx, tt::ARCH arch) {
    ValidateGen1DataMovementConfig(kernel, arch);
    ValidateUnpackModes(kernel, ctx.collected, arch);
    ValidateImplicitSyncOptOuts(kernel);
}

}  // namespace tt::tt_metal::experimental
