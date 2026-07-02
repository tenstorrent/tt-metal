// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include <tt-metalium/base_types.hpp>  // For MathFidelity, UnpackToDestMode (global scope)

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
// The compute hardware differs between Gen1 architectures (Wormhole, Blackhole)
// and Gen2 architectures (Quasar and derivatives). A compute kernel targets exactly
// one generation, so the config is a variant holding either a ComputeGen1Config or
// a ComputeGen2Config — the runtime uses the one matching the target architecture.
//
// The Tensix Engine pipeline consists of Unpack, Math, and Pack stages.
// There are two math engines:
//  - FPU reads operands from the SrcA / SrcB register files (~19-bit),
//    writes to the Dest register file (16- or 32-bit, configurable).
//  - SFPU runs SIMD transcendentals. It can only access Dest.
//
// The Gen1Config / Gen2Config fields configure this pipeline. Most fields are
// common to both generations; the differences are called out per field.
//
// NOTE: The Unpack, Math, and Pack stages are hardware pipeline stages internal
//       to a single kernel thread, not to be confused with KernelSpec::num_threads!
//       Each thread of a multi-threaded compute kernel runs its own independent
//       Unpack/Math/Pack pipeline.
//
// ============================================================================

// Per-DFB choice of how the unpacker delivers data into the math stage:
//   Default          — unpack via SrcA/B regs (~19-bit elements; full FPU access)
//   UnpackToDestFp32 — unpack via Dest regs with full FP32 precision (SFPU only)
//
// This choice matters only when ALL of the following hold for the DFB binding:
//   1. The kernel is the consumer endpoint (unpacking data into the kernel)
//   2. The DFB's data format is Float32.
//   3. fp32_dest_acc_en is true (Dest must be 32-bit-wide to hold FP32).
//
// You MUST provide an unpack_to_dest_mode entry for the DFB when all three conditions hold;
// failing to do so will trigger an error. Outside the triple an entry is optional: Default is
// always accepted, and UnpackToDestFp32 is tolerated where it is inert (non-consumer or
// non-Float32 — the hardware ignores the mode there, and legacy ops set it unconditionally).
// UnpackToDestFp32 always requires fp32_dest_acc_en=true, even where inert; otherwise it is
// rejected as incoherent (Dest is 16-bit and cannot hold full FP32).
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

    // Per-DFB choice of how the unpacker delivers data into the math stage.
    // (See the ComputeUnpackToDestModes doc comment above.)
    ComputeUnpackToDestModes unpack_to_dest_mode;

    // Pack-side precision tweak for the Bfp8 block-float format.
    // (Affects how exponents are reconciled when converting Dest contents to Bfp8)
    bool bfp8_pack_precise = false;
};

struct ComputeGen2Config {
    // The fields below are identical to ComputeGen1Config — see the ComputeGen1Config comments above.
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    bool math_approx_mode = false;
    ComputeUnpackToDestModes unpack_to_dest_mode;

    // (Quasar only).
    //
    // When true, the unpacker packs two values into each source-register
    // slot instead of one, so the math engine reads twice as many elements per
    // pass. Only the matmul family of instructions work with this format — matmul (MVMUL/MVMULDI) and the GAPOOL
    // instruction that column reduce ops are built on.
    //
    // So set this true only for kernels whose inputs are consumed solely by a matmul or a column reduce.
    bool enable_2x_src_format = false;

    // (Quasar only) Explicitly route this kernel's unpacked operands into dest
    // running the unpack→math→pack semaphore handshake, independent of operand data format.
    // Default false. On WH/BH this is ignored (unpack-to-dest stays inferred from 32-bit format).
    bool unpack_to_dest_en = false;
};

// A compute kernel's hardware config holds exactly one generation's config.
using ComputeHardwareConfig = std::variant<ComputeGen1Config, ComputeGen2Config>;

}  // namespace tt::tt_metal::experimental
