// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include <tt-metalium/base_types.hpp>  // For MathFidelity, UnpackToDestMode

namespace tt::tt_metal::experimental {

// ============================================================================
//  ComputeHardwareConfig
// ============================================================================
//
// The ComputeHardwareConfig describes the configuration of the Tensix compute
// accelerator hardware resources controlled by a compute kernel.
//
// You must specify a ComputeHardwareConfig for every compute kernel.
//
// The Tensix Engine pipeline consists of Unpack, Math, and Pack stages.
// There are two math engines:
//  - FPU reads operands from the SrcA / SrcB register files (~19-bit),
//    and writes to the Dest register file (16- or 32-bit, configurable).
//  - SFPU runs SIMD transcendentals. It can only access Dest.
//
// The ComputeHardwareConfig configures this pipeline.
//
// Different generations of Tenstorrent accelerators have slightly different
// Tensix compute hardware. The compute configuration is therefore generation-
// specific (though many fields are common). ComputeHardwareConfig is a variant
// object that holds one generation's config; you must specify the correct
// config for the hardware your compute kernel will run on.
//
// NOTE: The Unpack, Math, and Pack stages are hardware pipeline stages internal
//       to a single kernel thread. Not to be confused with KernelSpec::num_threads!
//       In a multi-threaded compute kernel, each thread runs its own independent
//       Unpack/Math/Pack pipeline.
//
// ============================================================================

// Per-DFB unpack-to-dest mode table. See gen-specific configs for details.
using ComputeUnpackToDestModes = Table<DFBSpecName, tt::tt_metal::UnpackToDestMode>;

struct Compute1xxConfig {
    ////////////////////////////////////////////////
    // General accuracy / performance tradeoffs
    ////////////////////////////////////////////////

    // Number of multiply passes the FPU runs.
    // The higher the fidelity, the greater the precision (more mantissa bits are used),
    // but higher fidelity means more multiply passes, slowing the computation.
    MathFidelity fpu_math_fidelity = MathFidelity::HiFi4;

    // Accuracy / performance tradeoff for the SFPU transcendentals.
    // Select either fast-and-approximate mode or slow-and-precise mode.
    bool spfu_approx_mode = false;

    // Pack stage precision tweak, specific to the Bfp8 block-float format.
    // Affects how exponents are reconciled when converting Dest contents to Bfp8 in in
    // the Pack stage. Select either precise (slower) or approximate (faster) mode.
    // NOTE: This setting has no effect on non-Bfp8 formats.
    bool bfp8_pack_precise = false;

    ////////////////////////////////////////////////
    // Dest register configuration (32-bit formats)
    ////////////////////////////////////////////////

    // Configure Dest register to hold 32-bit elements (instead of the default 16-bit).
    // A 32-bit Dest register is required to hold full FP32 precision, but it
    // ... (??? is slower how???)
    bool enable_32_bit_dest = false;

    // Dest register synchronization mode:
    //   false (Half) — Dest is split in half; double-buffered between math and pack pipeline stages
    //   true  (Full) — Dest is one buffer; twice the capacity, no math/pack overlap
    // You may only configure this setting to Full (true) if enable_32_bit_dest is also true.
    // This setting affects performance, but not accuracy.)
    // (TODO: What? This still makes no sense. Why is this separate from enable_32_bit_dest?)
    // (This also feels like it should be an enum -- Half vs Full, not a bool. Thoughts?)

    DestSyncMode dest_sync_mode = DestSyncMode::Half;
    // bool dst_full_sync_en = false;  // <-- can we give this guy a better name, and an enum??

    // Choice to unpack data into the Dest or into the SrcA / SrcB register file.
    // By default, the Unpacker unpacks to SrcA / SrcB. This keeps the FPU available,
    // and it is faster (on Wormhole and Blackhole). However, if:
    //  - the data format is Float32,
    //  - the data will be consumed by the SFPU (not the FPU), and
    //  - you wish to preserve full FP32 precision,
    // you may choose instead to unpack directly into the Dest register file.
    //
    // Configuration mode (this selected per-DFB):
    //   Default          — Unpack via SrcA/B
    //   UnpackToDestFp32 — Unpack to Dest (valid only for FP32)
    //
    // This setting is ONLY relevant if enable_32_bit_dest is true!
    // You MUST suppply an unpack_to_dest_mode entry for any consumed-from DFB
    // with Float32 data format. (Failing to do so triggers a compile-time error.)
    // In all other cases, Default is assumed and the unpack_to_dest_mode setting
    // has no effect.
    // NOTE: Validation will reject incoherent UnpackToDestFp32 settings.
    //
    ComputeUnpackToDestModes unpack_to_dest_mode;
};

struct Compute2xxConfig {
    ////////////////////////////////////////////////
    // General accuracy / performance tradeoffs
    ////////////////////////////////////////////////

    // See the Compute1xxConfig for details on fpu_math_fidelity
    MathFidelity fpu_math_fidelity = MathFidelity::HiFi4;

    // See the Compute1xxConfig for details on spfu_approx_mode
    bool spfu_approx_mode = false;

    // Note: 2xx architectures replace BFP data formats with MXFP formats;
    //       the bfp8_pack_precise setting is not relevant for 2xx.

    ////////////////////////////////////////////////
    // Dest register configuration (32-bit formats)
    ////////////////////////////////////////////////

    // See the Compute1xxConfig for details on enable_32_bit_dest
    bool enable_32_bit_dest = false;

    // See the Compute1xxConfig for details on enable_32_bit_dest
    bool dst_full_sync_en = false;

    // See the Compute1xxConfig for details on unpack_to_dest_mode
    // NOTE: The semantics of this setting may change on Quasar
    //       (Or it might disappear altogether?)
    ComputeUnpackToDestModes unpack_to_dest_mode;

    ////////////////////////////////////////////////
    // Temporary hacks that WILL BE REMOVED
    ////////////////////////////////////////////////

    // When true, the unpacker packs two values into each source-register
    // slot instead of one, so the math engine reads twice as many elements per
    // pass. Only the matmul family of instructions work with this format — matmul (MVMUL/MVMULDI) and the GAPOOL
    // instruction that column reduce ops are built on.
    //
    // So set this true only for kernels whose inputs are consumed solely by a matmul or a column reduce.
    bool enable_2x_src_format = false;
    // This setting is specific to Mxfp4 data format (... its name should have "mxfp4" in it...)
    // This is temporary hack! (?)
    // This this is a kernel-global setting, its value should be automatically inferred at compile time,
    // (in the Metal 2.0 validation code) from the data formats of the kernel's consumed-from DFBs.

    // Explicitly route this kernel's unpacked operands into dest
    // running the unpack→math→pack semaphore handshake, independent of operand data format.
    bool unpack_to_dest_en = false;
    // This is a strictly temporary hack!
    // Its presence alters the semantics of unpack_to_dest_mode.
    // This creates a suprising misalignment between 1xx and 2xx behavior
    // It also creates a reachable, unvalidated misconfiguration if these are set inconsistently!
    // Tracked in issue #49445.

    ///////////////////////////////////////////////////////////////////////////////////////////////
};

// A compute kernel's hardware config holds exactly one generation's config.
using ComputeHardwareConfig = std::variant<ComputeGen1Config, ComputeGen2Config>;

}  // namespace tt::tt_metal::experimental
