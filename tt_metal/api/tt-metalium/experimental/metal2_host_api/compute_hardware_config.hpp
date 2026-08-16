// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
// Different generations of Tenstorrent accelerators have slightly different Tensix compute
// hardware. Most of the configuration describes the pipeline itself and means the same thing
// on every generation; those settings are fields of ComputeHardwareConfig directly. The few
// settings that name hardware only one generation has live in a generation-specific block:
//
//     ComputeHardwareConfig{
//         .fpu_math_fidelity = MathFidelity::HiFi2,   // generation-independent
//         .enable_32_bit_dest = true,                 // generation-independent
//         .gen1 = {.unpack_modes = {{"in0", UnpackMode::UnpackToDest}}},  // Gen1 only
//     }
//
// A config may populate both generation blocks; the block matching the target architecture is
// applied and the other is ignored. Leaving a block default is the common case, and is legal on
// every generation — a config that sets only the generation-independent fields runs anywhere.
//
// NOTE: The Unpack, Math, and Pack stages are hardware pipeline stages internal
//       to a single kernel thread. Not to be confused with KernelSpec::num_threads!
//       In a multi-threaded compute kernel, each thread runs its own independent
//       Unpack/Math/Pack pipeline.
//
// ============================================================================

// Selects, per (consumed-from) DFB, whether the unpacker writes into the SrcA / SrcB register
// files or straight into the Dest register file.
//
//  UnpackToSrc  — Unpack to SrcA/B. Both FPU and SFPU can consume the data (copied to Dest for
//                 the SFPU), but precision is reduced to 19 bits (precision is lost for FP32;
//                 32-bit integers are truncated).
//  UnpackToDest — Unpack to Dest directly, preserving full 32-bit precision. Requires a 32-bit
//                 Dest register (see enable_32_bit_dest), and the data must be consumed by the
//                 SFPU rather than the FPU.
//
// UnpackToSrc is assumed for any DFB with no entry. Which entries you should write, and which
// are required, is generation-specific: see ComputeGen1Config / ComputeGen2Config below.
using ComputeUnpackModes = Table<DFBSpecName, tt::tt_metal::UnpackMode>;

// Compute settings specific to Gen1 architectures:
//  - Wormhole  (TT-1.1.0)
//  - Blackhole (TT-1.2.0)
//
// Ignored when the kernel runs on a Gen2 architecture.
struct ComputeGen1Config {
    // Pack stage precision tweak for block-float formats.
    // Affects how exponents are reconciled when converting Dest contents to BFP in
    // the Pack stage. Select either precise (slower) or approximate (faster).
    // NOTE: This setting has no effect on non-BFP formats.
    Precision bfp_pack_precision_mode = Precision::Approximate;

    // Per-DFB unpack destination; see ComputeUnpackModes above for the mechanism.
    //
    // On Wormhole and Blackhole, UnpackToSrc is the fastest option, so UnpackToDest should be
    // used only if:
    //  - The data format has 32-bit precision, AND enable_32_bit_dest is set to true
    //  - You want to preserve the full precision
    //  - The data will be consumed by the SFPU (not the FPU)
    //
    // If enable_32_bit_dest is true and the DFB carries a 32-bit format, you must EXPLICITLY
    // specify an UnpackMode for that DFB: at that combination the choice is a real precision
    // and throughput tradeoff, so there is no default to fall back on. (Enforced by validation
    // checks.)
    ComputeUnpackModes unpack_modes;
};

// Compute settings specific to Gen2 architectures:
//  - Quasar (TT-2.0.0)
//  - Quasar derivatives (TT-2.0.x)
//
// Ignored when the kernel runs on a Gen1 architecture.
//
// Note: Gen2 architectures replace BFP data formats with MXFP formats; the Gen1
//       bfp_pack_precision_mode setting has no Gen2 counterpart.
struct ComputeGen2Config {
    // Per-DFB unpack destination; see ComputeUnpackModes above for the mechanism.
    //
    // On Gen2 architectures there is NO performance penalty for unpacking directly to Dest, so
    // UnpackToDest is the preferred mode for any SFPU-consumed data — which is why this table is
    // generation-specific rather than shared. A table tuned for Gen1 remains legal here, but
    // leaves precision on the table for SFPU consumers.
    //
    // The explicit-entry requirement is the same as on Gen1: if enable_32_bit_dest is true and
    // the DFB carries a 32-bit format, an entry is required. Porting a Gen1-tuned kernel to
    // Quasar therefore surfaces this table as something to revisit rather than inherit.
    ComputeUnpackModes unpack_modes;

    ///////////////////////////////////////////
    // Temporary configs (these will change!)
    ///////////////////////////////////////////

    // When true, the unpacker packs two values into each source-register slot instead of one.
    // The math engine reads twice as many elements per pass, effectively doubling throughput.
    //
    // This is currently ONLY supported for Mxfp4 data format. The setting is ignored for all
    // other formats.
    //
    // WARNING: Only the matmul family of instructions work with this format:
    //  - matmul (MVMUL/MVMULDI)
    //  - the GAPOOL instruction that column reduce ops are built on
    //
    // Invoking other instructions on Mxfp4 data with the setting enabled will produce garbage
    // math results! Enable this setting ONLY for kernels whose inputs are consumed solely by
    // a matmul or a column reduce.
    //
    // This API is not final and subject to change!
    // It should most likely become a per-DFB setting, similar to unpack_modes.
    bool enable_2x_src_register = false;

    ///////////////////////////////////////////////////////////////////////////////////////////////
};

struct ComputeHardwareConfig {
    ////////////////////////////////////////////////
    // General accuracy / performance tradeoffs
    ////////////////////////////////////////////////

    // Number of multiply passes the FPU runs.
    // The higher the fidelity, the greater the precision (more mantissa bits are used),
    // but higher fidelity means more multiply passes, slowing the computation.
    MathFidelity fpu_math_fidelity = MathFidelity::HiFi4;

    // Accuracy / performance tradeoff for the SFPU transcendentals.
    // Select either fast-and-approximate mode or slow-and-precise mode.
    Precision sfpu_precision_mode = Precision::Precise;

    /////////////////////////////////////
    // Dest register file configuration
    /////////////////////////////////////

    // Configure the Dest register to hold 32-bit elements (instead of the default 16-bit).
    // A 32-bit Dest register is required in order to hold full 32-bit precision formats.
    // (But, this halves the number of tiles that can be stored in the Dest register file.)
    // NOTE: When used for FPU accumulation, pair this with fpu_math_fidelity=HiFi3 or
    //       HiFi4; otherwise the extra precision buys little.
    //       When using the SFPU, pair this with UnpackMode=UnpackToDest to preserve 32-bit
    //       precision input data.
    bool enable_32_bit_dest = false;

    // Dest register double-buffering mode.
    // This setting trades off per-step tile capacity for pipeline throughput.
    // It affects performance and tile budget only (no effect on precision).
    //
    // Configuration options:
    //  true -  Double buffered. The Dest register is split in two. Math and Pack stages run
    //          in parallel, but a single compute step has only half the Dest register capacity.
    //  false - Single buffered. Dest is a single buffer. Math must wait for Pack to drain
    //          before reusing, but the full tile capacity is available for each compute step.
    //
    // Always enable double buffering unless a single compute step requires more capacity than
    // the double-buffered (half-capacity) mode allows.
    // NOTE: The enable_32_bit_dest flag (though orthogonal) also affects the tile capacity, and
    // makes it more likely that single-buffering mode will be necessary.
    bool double_buffer_dest = true;

    ///////////////////////////////////////////
    // Generation-specific settings
    ///////////////////////////////////////////

    // Only the block matching the target architecture is applied; the other is ignored.
    // Both may be left default — a kernel needing no generation-specific settings runs on
    // either generation as-is.
    ComputeGen1Config gen1;
    ComputeGen2Config gen2;
};

}  // namespace tt::tt_metal::experimental
