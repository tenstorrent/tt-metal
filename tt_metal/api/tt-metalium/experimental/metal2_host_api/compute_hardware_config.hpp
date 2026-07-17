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

// Type used for unpack_modes; see configuration structs below for details
using ComputeUnpackModes = Table<DFBSpecName, tt::tt_metal::UnpackMode>;

// Compute configuration for Gen1 architectures:
//  - Wormhole  (TT-1.1.0)
//  - Blackhole (TT-1.2.0)
struct ComputeGen1Config {
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

    // Pack stage precision tweak for block-float formats.
    // Affects how exponents are reconciled when converting Dest contents to BFP in
    // the Pack stage. Select either precise (slower) or approximate (faster).
    // NOTE: This setting has no effect on non-BFP formats.
    Precision bfp_pack_precision_mode = Precision::Approximate;

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

    // Unpack data into the Dest or into the SrcA / SrcB register file.
    // This choice is specified per (consumed-from) DFB, rather than kernel-wide.
    // Configuration options:
    //  UnpackToSrc  — Unpack to SrcA/B
    //  UnpackToDest — Unpack to Dest directly
    //
    // UnpackToSrc is the default.
    //  - Both FPU and SFPU can consume the data (copied to Dest for the SFPU).
    //  - Data precision is reduced to 19 bits.
    //    (Precision is lost for FP32; 32-bit integers are truncated).
    //  - This is the fastest option on Wormhole and Blackhole.
    //
    // UnpackToDest should be used (on Wormhole and Blackhole) only if:
    //  - The data format has 32-bit precision, AND enable_32_bit_dest is set to true
    //  - You want to preserve the full precision
    //  - The data will be consumed by the SFPU (not the FPU)
    //
    // If no mode is specified for a (consumed-from) DFB, UnpackToSrc is assumed.
    // However, if enable_32_bit_dest is true and the DFB carries a 32-bit format, you must
    // EXPLICITLY specify an UnpackMode for that DFB. (Enforced by validation checks.)
    //
    ComputeUnpackModes unpack_modes;
};

// Compute configuration for Gen2 architectures:
//  - Quasar (TT-2.0.0)
//  - Quasar derivatives (TT-2.0.x)
struct ComputeGen2Config {
    ////////////////////////////////////////////////
    // General accuracy / performance tradeoffs
    ////////////////////////////////////////////////

    // See ComputeGen1Config for details on fpu_math_fidelity
    MathFidelity fpu_math_fidelity = MathFidelity::HiFi4;

    // See ComputeGen1Config for details on sfpu_precision_mode
    Precision sfpu_precision_mode = Precision::Precise;

    // Note: Gen2 architectures replace BFP data formats with MXFP formats;
    //       the bfp_pack_precision_mode setting is not relevant for Gen2.

    /////////////////////////////////////
    // Dest register file configuration
    /////////////////////////////////////

    // See ComputeGen1Config for details on enable_32_bit_dest
    bool enable_32_bit_dest = false;

    // See ComputeGen1Config for details on double_buffer_dest
    bool double_buffer_dest = true;

    // See ComputeGen1Config for details on unpack_modes
    //
    // NOTE: On Gen2 architectures, there is NO performance penalty for unpacking directly to
    //       Dest, so UnpackMode=UnpackToDest is the preferred mode for any SFPU-consumed data.
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

    // Explicitly route this kernel's unpacked operands into dest
    // running the unpack→math→pack semaphore handshake, independent of operand data format.
    //
    // NOTE: This is a strictly TEMPORARY HACK!
    //       Removal / fix is tracked in issue #49445.
    // ISSUES: Its presence alters the semantics of unpack_modes.
    //         This creates a surprising misalignment between Gen1 and Gen2 behavior.
    //         It also creates a reachable, unvalidated misconfiguration if unpack_to_dest_en and
    //         unpack_modes are set inconsistently.
    bool unpack_to_dest_en = false;

    ///////////////////////////////////////////////////////////////////////////////////////////////
};

// A compute kernel's hardware config holds exactly one generation's config.
using ComputeHardwareConfig = std::variant<ComputeGen1Config, ComputeGen2Config>;

}  // namespace tt::tt_metal::experimental
