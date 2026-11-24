// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar: the 2.0 (id-free) and legacy (DFB-id) compute APIs must never be mixed in one kernel. Both share the legacy
// Dest sync (tile_regs_*), but they program buffer descriptors differently: 2.0 op inits program fixed descriptor ids
// at each operand's base, legacy inits program per-DFB descriptors, so a mix would leave one API's descriptors stale.
//
// Each API's chokepoints claim the kernel with an assembler symbol and stop the build if the other API claimed it
// already, so a kernel that uses both fails to build whichever API comes first. Only functions the kernel actually
// uses emit the claim (unused inline/template code emits nothing), and the claim itself emits no instructions or
// data. With -flto the check runs when the kernel is linked.
//   2.0:    compute_kernel_hw_startup and every op init (op_init).
//   legacy: the buffer-descriptor programming of every unpack / pack init.
// Extended asm with empty operand lists: GCC treats non-empty basic asm as a memory clobber, which would make it reload
// state around every claim.
#define LLK_COMPUTE_API_CLAIM(api, other)                                                 \
    asm volatile(".ifdef __compute_api_" #other                                           \
                 "\n.error \"Quasar: 2.0 and legacy compute APIs mixed in one kernel\"\n" \
                 ".endif\n.set __compute_api_" #api ", 1" ::)
