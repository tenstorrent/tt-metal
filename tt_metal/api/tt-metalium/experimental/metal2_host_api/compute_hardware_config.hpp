// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>
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
// NOTE: The Unpack, Math, and Pack stages are hardware pipeline stages internal
//       to a single kernel thread. Not to be confused with KernelSpec::num_threads!
//       In a multi-threaded compute kernel, each thread runs its own independent
//       Unpack/Math/Pack pipeline.
//
// ============================================================================

using ComputeUnpackModes = Table<DFBSpecName, tt::tt_metal::UnpackMode>;

struct ComputeHardwareConfig {
    // ---- Generation-agnostic ("common") fields ----

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
    // On Gen2 architectures, there is NO performance penalty for unpacking directly to
    // Dest, so UnpackMode=UnpackToDest is the preferred mode for any SFPU-consumed data.
    //
    // If no mode is specified for a (consumed-from) DFB, UnpackToSrc is assumed.
    // However, if enable_32_bit_dest is true and the DFB carries a 32-bit format, you must
    // EXPLICITLY specify an UnpackMode for that DFB. (Enforced by validation checks.)
    //
    ComputeUnpackModes unpack_modes;

    // ---- Generation-specific configs ----
    // Optional and unset by default. Each config applies only to the architecture named in
    // its header, and is ignored on any other.
    // NOTE: The target architecture is selected at program construction time.
    //       See MakeProgramFromSpec for more details.

    // ---- TT-1.x.x specific (Wormhole, Blackhole) ----
    struct Compute1XXConfig {
        // Pack-stage precision tweak for block-float formats.
        // Affects how exponents are reconciled when converting Dest contents to BFP in
        // the Pack stage. Select either precise (slower) or approximate (faster).
        // NOTE: This setting has no effect on non-BFP formats.
        Precision bfp_pack_precision_mode = Precision::Approximate;
    };
    std::optional<Compute1XXConfig> gen1_specific = std::nullopt;

    // ---- TT-2.x.x specific (Quasar and derivatives) ----
    // Empty today.
    struct Compute2XXConfig {};
    std::optional<Compute2XXConfig> gen2_specific = std::nullopt;
};

}  // namespace tt::tt_metal::experimental
