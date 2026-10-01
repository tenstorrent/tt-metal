// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Blackhole compute hardware cleanup.
 *
 * Assumes the compute sentinel is off: cleanup restores hardware state only;
 * the next MicroOp is responsible for its own format / geometry configuration
 * (full hw_configure or explicit reconfig).
 *
 * The underlying LLKs exist only in tt_llk_blackhole, but this header ships in
 * HW_JIT_API_HEADERS for every arch, so both the includes and the API body are
 * guarded on ARCH_BLACKHOLE (matching experimental/hadamard.h and
 * experimental/rope_sfpu.h).
 */

#include "api/compute/common_globals.h"

#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_unpack_hw_cleanup.h"
#endif
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_hw_cleanup.h"
#endif
#if defined(TRISC_PACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_pack_hw_cleanup.h"
#endif

// clang-format off
/**
 * Normalizes a documented subset of mutable compute hardware using
 * fixed, CB-independent Float16_b formats. Tries to match the state
 * established by compute_kernel_hw_startup.
 *
 * May be called repeatedly between completed logical operations provided all
 * three TRISCs reach the call, mailboxes are empty, MATH_PACK and UNPACK_SYNC
 * can drain to zero, and cfg_state_id matches hardware.
 *
 * Rendezvouses T0/T1/T2 through hardware mailboxes, serializes configuration
 * in T0→T1→T2 order, and programs both cfg banks identically to:
 *   - source and pack formats: Float16_b;
 *   - tile geometry: one 32x32 tile;
 *   - faces: four 16x16 faces;
 *   - tile size: 2048 bytes.
 * It leaves cfg bank 0 selected.
 *
 * It also normalizes:
 *   - math/pack Dest semaphore and ping-pong state;
 *   - Blackhole destination read-address remap;
 *   - the address counters reset by pack Dest init;
 *   - Default pack ADDR_MOD.
 *
 * Startup does not establish ambient unpack/math MOP or math ADDR_MOD state;
 * Cleanup uses NOP MOPs for unpack/math and zeros math ADDR_MOD_0..7.
 * Pack MOP, strides, and PAC X are poisoned as well, so a following op must call
 * pack_init or pack_reconfig_data_format<true> before packing.
 *
 * Compile-time DST_ACCUM_MODE and DST_SYNC_MODE are preserved (re-asserted via the
 * hw_configure helpers, not changed to a different mode).
 */
// clang-format on
#if defined(ARCH_BLACKHOLE)
ALWI void compute_kernel_hw_cleanup() {
    UNPACK((_llk_unpack_hw_cleanup_canonical_<DST_ACCUM_MODE>()));
    MATH((_llk_math_hw_cleanup_canonical_<DST_SYNC_MODE, DST_ACCUM_MODE>()));
    PACK((_llk_pack_hw_cleanup_canonical_<DST_SYNC_MODE, DST_ACCUM_MODE>()));
}
#endif
