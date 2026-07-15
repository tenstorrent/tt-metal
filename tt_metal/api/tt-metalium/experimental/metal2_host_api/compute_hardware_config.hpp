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
//    writes to the Dest register file (16- or 32-bit, configurable).
//  - SFPU runs SIMD transcendentals. It can only access Dest.
//
// The ComputeHardwareConfig configures this pipeline.
//
// Different generations of Tenstorrent accelerators have slightly different
// Tensix compute hardware. The compute configuration is therefore generation-
// specific (though many fields are common).
//
// ComputeHardwareConfig is a variant object that holds one generation's config.
// You must specify the correct generation-specific config for the hardware
// your compute kernel will run on.
//
// NOTE: The Unpack, Math, and Pack stages are hardware pipeline stages internal
//       to a single kernel thread. Not to be confused with KernelSpec::num_threads!
//       In a multi-threaded compute kernel, each thread runs its own independent
//       Unpack/Math/Pack pipeline.
//
// ============================================================================

// Per-DFB unpack-to-dest mode table. See gen-specific configs for details.
using ComputeUnpackToDestModes = Table<DFBSpecName, tt::tt_metal::UnpackToDestMode>;

struct ComputeGen1Config {
    // Number of multiply passes the FPU runs to use more mantissa bits
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    // Configure Dest register to hold 32-bit elements (instead of the default 16-bit)
    bool fp32_dest_acc_en = false;

    // Dest register sync mode:
    //   false (Half) — Dest is split in half; math and pack pipeline (double-buffered)
    //   true  (Full) — Dest is one buffer; twice the capacity, no math/pack overlap
    bool dst_full_sync_en = false;

    // Select fast-and-approximate vs slow-and-precise variants of SFPU transcendentals
    bool math_approx_mode = false;

    // Pack-side precision tweak for the Bfp8 block-float format.
    // (Affects how exponents are reconciled when converting Dest contents to Bfp8)
    bool bfp8_pack_precise = false;

    // Per-DFB choice for unpacking Float32 data into this compute kernel.
    //
    // For Float32 data, the unpacker can deliver the data two ways:
    //   Default          — Reduce to ~19-bit Tf32 and unpack via SrcA/B.
    //                      Keeps the FPU available (matmul / binary eltwise).
    //   UnpackToDestFp32 — Keep full FP32. It only fits in the 32-bit Dest register,
    //                      so it unpacks straight to Dest — and since the FPU reads
    //                      only SrcA/B, this DFB becomes SFPU-only.
    //
    // NOTE: Since SrcA/B are 19-bit and can't hold FP32, keeping full FP32 precision
    //       forces the "Dest" path.
    //
    // A real choice exists (and an entry is REQUIRED) when all of the following hold:
    //   1. The compute kernel is the DFB's consumer endpoint
    //   2. The DFB's data format is Float32, and
    //   3. fp32_dest_acc_en is true (else Dest is 16-bit and full FP32 is impossible).
    //
    // The two options differ in precision and which math engine can use the data, so
    // you MUST provide an unpack_to_dest_mode entry for a DFB if the above conditions
    // hold (no default is assumed). Failing to do so will trigger a compile-time error.
    //
    // If the above conditions do not hold, an unpack_to_dest_mode entry is optional.
    // Default is always accepted; UnpackToDestFp32 is tolerated in "don't care" cases.
    //
    ComputeUnpackToDestModes unpack_to_dest_mode;
};

struct ComputeGen2Config {
    // Number of multiply passes the FPU runs to use more mantissa bits
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    // Configure Dest register to hold 32-bit elements (instead of the default 16-bit)
    bool fp32_dest_acc_en = false;

    // Dest register sync mode:
    //   false (Half) — Dest is split in half; math and pack pipeline (double-buffered)
    //   true  (Full) — Dest is one buffer; twice the capacity, no math/pack overlap
    bool dst_full_sync_en = false;

    // Select fast-and-approximate vs slow-and-precise variants of SFPU transcendentals
    bool math_approx_mode = false;

    // When true, the unpacker packs two values into each source-register
    // slot instead of one, so the math engine reads twice as many elements per
    // pass. Only the matmul family of instructions work with this format — matmul (MVMUL/MVMULDI) and the GAPOOL
    // instruction that column reduce ops are built on.
    //
    // So set this true only for kernels whose inputs are consumed solely by a matmul or a column reduce.
    bool enable_2x_src_format = false;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // TODO: Need to fix the problems that arise with unpack_to_dest_en and unpack_to_dest_mode.
    //   - Confusing naming and misleading comments.
    //   - Surprising misalignment between Gen1 and Gen2 behavior of unpack_to_dest_mode
    //   - Reachable, unvalidated misconfiguration if these are set inconsistently

    // Explicitly route this kernel's unpacked operands into dest
    // running the unpack→math→pack semaphore handshake, independent of operand data format.
    bool unpack_to_dest_en = false;

    // NOTE: On Gen2 this field does NOT carry the semantics documented on ComputeGen1Config,
    // and its final Gen2 behavior is still being settled.
    // Key difference: on Quasar the SrcA/B-vs-Dest routing is driven by unpack_to_dest_en,
    // applied regardless of data format — it is NOT inferred from this per-DFB mode the way
    // it is on Gen1. Do NOT rely on a Gen1 reading of this field here.
    //
    // Tracked in issue #49445.
    ComputeUnpackToDestModes unpack_to_dest_mode;
    ///////////////////////////////////////////////////////////////////////////////////////////////
};

// A compute kernel's hardware config holds exactly one generation's config.
using ComputeHardwareConfig = std::variant<ComputeGen1Config, ComputeGen2Config>;

}  // namespace tt::tt_metal::experimental
