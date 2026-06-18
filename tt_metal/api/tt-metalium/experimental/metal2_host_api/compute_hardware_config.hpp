// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
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
// The configuration is common to all Tenstorrent accelerator families.
//
// The Tensix Engine pipeline consists of Unpack, Math, and Pack stages.
// There are two math engines:
//  - FPU reads operands from the SrcA / SrcB register files (~19-bit),
//    writes to the Dest register file (16- or 32-bit, configurable).
//  - SFPU runs SIMD transcendentals. It can only access Dest.
//
// The ComputeHardwareConfig fields configure this pipeline.
//
// ============================================================================

struct ComputeHardwareConfig {
    // Number of multiply passes the FPU runs to use more mantissa bits
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    // Configure Dest register to hold 32-bit elements (instead of the default 16-bit)
    bool fp32_dest_acc_en = false;

    // Dest register sync mode:
    //   false (Half) — Dest is split in half; math and pack pipeline (double-buffered)
    //   true  (Full) — Dest is one buffer; twice the capacity, no math/pack overlap
    bool dst_full_sync_en = false;

    // Pack-side precision tweak for the Bfp8 block-float format.
    // (Affects how exponents are reconciled when converting Dest contents to Bfp8)
    bool bfp8_pack_precise = false;

    // Select fast-and-approximate vs slow-and-precise variants of SFPU transcendentals
    bool math_approx_mode = false;

    // (Quasar only).
    //
    // When true, the unpacker packs two values into each source-register
    // slot instead of one, so the math engine reads twice as many elements per
    // pass. Only the matmul family of instructions work with this format — matmul (MVMUL/MVMULDI) and the GAPOOL
    // instruction that column reduce ops are built on.
    //
    // So set this true only for kernels whose inputs are consumed solely by a matmul or a column reduce.
    bool enable_2x_src_format = false;

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
    using UnpackToDestModes = Table<DFBSpecName, tt::tt_metal::UnpackToDestMode>;
    UnpackToDestModes unpack_to_dest_mode;
};

}  // namespace tt::tt_metal::experimental
