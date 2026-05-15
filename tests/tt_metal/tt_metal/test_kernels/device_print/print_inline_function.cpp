// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"
#include "api/compute/common.h"

/**
 * IIUC what happens is that the DEVICE_PRINT (roughly)expands to a lambda has a
 * []() {
 *     static __attribute__((section("..."))) ...
 * }();
 *
 * When this expansion happens inside of an inline function,
 * because there can be many definition of the inline function across different TUs, there can be many definitions of
 * the static variable. The compiler will put the definitions into a comdata (or linkonce, not certain which) section so
 * that it can later merge the together, so that there is only one object behind the static variable. When this happens,
 * it will in certian casses ignore the __attribute__((section("..."))) and put the variable into .data instead.
 *
 * This will then result in the device implementation calling begin_write_message with a 0xFFB... (where .data is
 * located on device) pointer instead of 0x065.... (where string_info should be located) pointer. begin write message
 * will then do (ptr - 0x065... / sizeof(string_info)) & 0xFFFF resulting in a unexpected info_id being passed to the
 * host. The host will then be unable to interpret the DEVICE_PRINT and siliently drop it.
 *
 * If you compile this code and look at the objdump you will see
 * ```
 * ffb00820 l     O .data	00000010 _ZZZ28inline_function_device_printvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info
 * 06500000 l     O .device_print_strings_info	00000010
 * _ZZZ11kernel_mainvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info
 * ```
 * as you can see the allocated_string_info in the inline function silently moved to .data
 * resulting in the following function call
 *
 * ```
 * DEVICE_PRINT("INLINE!!!\n");
 * 6a68:	03018593          	addi	a1,gp,48 # ffb00820
 * <_ZZZ28inline_function_device_printvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info> 6a6c:	00500513          	li
 * a0,5 6a70:	188000ef          	jal	6bf8
 * <_ZN19device_print_detail19begin_message_writeENS_10structures17DevicePrintHeaderEj.isra.0>
 * ```
 *
 * as you can see the address being passed in is 0xFFB00820 instead of 0x065xxxxx, which causes the host to not
 * recognize the message and drop it.
 *
 * however, if you remove the inline from the inline_function_device_print, you will see the following objdump
 * ```
 * 06500010 l     O .device_print_strings_info	00000010
 * _ZZZ28inline_function_device_printvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info 06500020 l     O
 * .device_print_strings_info	00000010 _ZZZ11kernel_mainvENKUlDpOT_E2_clIJEEEDaS1_E21allocated_string_info
 * ```
 * the allocated_string_info is correctly placed in the .device_print_strings_info section, and the function call will
 * be
 * ```
 * DEVICE_PRINT("INLINE!!!\n");
 * 6a64:	01040593          	addi	a1,s0,16 # 6500010
 * <_ZZZ28inline_function_device_printvENKUlDpOT_E_clIJEEEDaS1_E21allocated_string_info> 6a68:	00500513          	li
 * a0,5 6a6c:	188000ef          	jal	6bf4
 * <_ZN19device_print_detail19begin_message_writeENS_10structures17DevicePrintHeaderEj.isra.0>
 * ```
 * which is correct.
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
