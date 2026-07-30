// SPDX-License-Identifier: Apache-2.0
//
// Force-included (via -include) when an editor indexes a dojo kernel. It is
// never part of a real build and has no effect on `dojo test` / `dojo bench`.
//
// Kernel .cpp files are not standalone translation units: the JIT build
// #includes them into a firmware .cc, compiled by the RISC-V/SFPI toolchain,
// with per-kernel generated headers and per-core defines. A host clang has none
// of that context, so this file supplies the minimum needed for the includes to
// resolve and the API to become navigable.

#pragma once

#include <stdint.h>

// risc_common.h uses MY_NOC_ENCODING(), but the header that defines it
// (noc_nonblocking_api.h) includes risc_common.h itself before reaching its own
// #define. When the kernel is the translation-unit root that ordering doesn't
// hold, so predefine it; the real definition supersedes this later.
#ifndef MY_NOC_ENCODING
#define MY_NOC_ENCODING(noc_index) 0
#endif

// The RISC-V toolchain understands these Tenstorrent attributes; host clang
// rejects them outright. Neutralise them for indexing only.
#include "internal/risc_attribs.h"
#undef tt_l1_ptr
#undef tt_reg_ptr
#define tt_l1_ptr
#define tt_reg_ptr

// chlkc_list.h (pulled in by the compute API) defines run_kernel(), which calls
// kernel_main() before the kernel body has been seen. In a real build the
// generated wrapper includes the kernel first; here a forward declaration does
// the job.
void kernel_main();

// Compute kernels only. chlkc_list.h includes chlkc_descriptors.h under
// UCK_CHLKC_*, which we deliberately do not define (it would also pull in the
// generated per-kernel wrapper that does not exist outside a real build). So
// pull the descriptors in here, and supply the scalars that the real generated
// header defines only under those same UCK_CHLKC_* guards.
//
// This mirrors tt_metal/jit_build/fake_kernels_target/fake_jit_prelude.h.
#if defined(TRISC_MATH) || defined(TRISC_UNPACK) || defined(TRISC_PACK)
#include "chlkc_descriptors.h"
constexpr bool DST_ACCUM_MODE = false;
#define DST_SYNC_MODE DstSync::SyncHalf
constexpr bool APPROX = true;
constexpr std::int32_t MATH_FIDELITY = 255;
#endif
