// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"
#include "api/compute/common.h"

/**
 * Bug: DEVICE_PRINT inside an `inline` function is silently dropped by the host.
 *
 * Why:
 *   DEVICE_PRINT expands to a lambda containing a
 *     `static __attribute__((section(".device_print_strings_info"))) ... allocated_string_info`
 *   Inside an `inline` function the lambda's static gets emitted into a COMDAT/linkonce (not my
 *   area of expertise, but this is my current understanding) group so it can be deduplicated
 *   across TUs. The compiler might drop the `section` attribute and place the variable in `.data`
 *   instead.
 *
 *   The device-side `begin_message_write` then receives a `.data` address (~0xFFB...)
 *   instead of a `.device_print_strings_info` address (~0x065...). Its
 *     `info_id = ((ptr - strings_info_base) / sizeof(DevicePrintStringInfo)) & 0xFFFF`
 *   computation produces a garbage info_id, the host can't resolve the string, and the
 *   message is silently dropped.
 *
 * Evidence — broken (function declared `inline`):
 *   `allocated_string_info` for the inline function ends up in `.data`:
 *
 * ```
 * ffb00820 l     O .data	00000010 _ZZZ28inline_function_device_printvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info
 * 06500000 l     O .device_print_strings_info	00000010 _ZZZ11kernel_mainvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info
 * ```
 *
 * so the call site passes a `.data` pointer (0xFFB00820) to begin_message_write:
 *
 * ```
 * DEVICE_PRINT("INLINE!!!\n");
 * 6a68:	03018593          	addi	a1,gp,48 # ffb00820 <_ZZZ28inline_function_device_printvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info>
 * 6a6c:	00500513          	li      a0,5
 * 6a70:	188000ef          	jal	    6bf8 <_ZN19device_print_detail19begin_message_writeENS_10structures17DevicePrintHeaderEj.isra.0>
 * ```
 *
 * Evidence — fixed (remove `inline`):
 *   `allocated_string_info` is correctly placed in `.device_print_strings_info`:
 *
 * ```
 * 06500010 l     O .device_print_strings_info	00000010 _ZZZ28inline_function_device_printvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info
 * 06500020 l     O .device_print_strings_info	00000010 _ZZZ11kernel_mainvENKUlDpOT_E2_clIJEEEDaS1_E21allocated_string_info
 * ```
 *
 * and the call site passes the correct address (0x06500010):
 *
 * ```
 * DEVICE_PRINT("INLINE!!!\n");
 * 6a64:	01040593          	addi	a1,s0,16 # 6500010 <_ZZZ28inline_function_device_printvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info>
 * 6a68:	00500513          	li      a0,5
 * 6a6c:	188000ef          	jal     6bf4  <_ZN19device_print_detail19begin_message_writeENS_10structures17DevicePrintHeaderEj.isra.0>
 * ```
 *
 * What I don't understand is why this doesn't happen when I remove the unrelated const definition from
 * the inline function >:( .
 */
inline void inline_function_device_print() {
    const ct_string kernel = CTSTR("THIS IS REQUIRED TO BREAK THE FOLLOWING DEVICE_PRINT");
    DEVICE_PRINT("INLINE!!!\n");
}

void kernel_main() {
    DEVICE_PRINT("BEFORE!!!\n");

    inline_function_device_print();

    DEVICE_PRINT("AFTER!!!\n");
}
