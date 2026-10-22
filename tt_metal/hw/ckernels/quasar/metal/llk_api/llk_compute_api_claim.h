// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar: the 2.0 (id-free) and legacy (DFB-id) compute APIs must never be mixed in one kernel. 2.0 syncs Dest with
// the dvalid chain, legacy with semaphores, and the two schemes cannot share Dest.
//
// Each API's chokepoints claim the kernel with an assembler symbol and stop the build if the other API claimed it
// already, so a kernel that uses both fails to build whichever API comes first. Only functions the kernel actually
// uses emit the claim (unused inline/template code emits nothing), and the claim itself emits no instructions or
// data. With -flto the check runs when the kernel is linked.
//   2.0:    compute_kernel_hw_startup and every op init (dest_join).
//   legacy: the Dest-sync waits (tile_regs_acquire / tile_regs_wait / acquire_dst) and the buffer-descriptor
//           programming of every unpack / pack init.
#define LLK_COMPUTE_API_CLAIM(api, other)                                         \
    asm volatile(                                                                 \
        ".ifdef __compute_api_" #other                                            \
        "\n.error \"Quasar: 2.0 and legacy compute APIs mixed in one kernel\"\n" \
        ".endif\n.set __compute_api_" #api ", 1")
