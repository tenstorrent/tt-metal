// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/base_types.hpp>  // For MathFidelity, UnpackToDestMode (global scope)

namespace tt::tt_metal::experimental::metal2_host_api {

// KernelComputeConfig: Tensix hardware resource configuration for compute kernels.
// (Common to all Tenstorrent accelerators.)
struct KernelComputeConfig {

    // The Tensix Engine pipeline consists of Unpack, Math, and Pack stages.
    // There are two math engines:
    //  - FPU reads operands from the SrcA / SrcB register files (~19-bit),
    //    writes to the Dest register file (16- or 32-bit, configurable).
    //  - SFPU runs SIMD transcendentals. It can only access Dest.
    // The fields below configure this pipeline.

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

    // Per-DFB choice of how the unpacker delivers data into the math stage:
    //   Default          — unpack via SrcA/B regs (~19-bit elements; full FPU access)
    //   UnpackToDestFp32 — unpack via Dest regs with full FP32 precision (SFPU only)
    //
    // This choice matters only when ALL of the following hold for the DFB binding:
    //   1. The kernel is the consumer endpoint (unpacking data into the kernel)
    //   2. The DFB's data format is Float32.
    //   3. fp32_dest_acc_en is true (Dest must be 32-bit-wide to hold FP32).
    //
    // You MUST provide an unpack_to_dest_mode entry for the DFB if these conditions hold;
    // failing to do so will trigger an error. Otherwise, supplying an entry is optional
    // and only Default is accepted.
    struct DFBUnpackToDestMode {
        DFBSpecName dfb_spec_name;
        tt::tt_metal::UnpackToDestMode mode;
    };
    std::vector<DFBUnpackToDestMode> unpack_to_dest_mode;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
