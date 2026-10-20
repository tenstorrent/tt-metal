// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Records where a kernel's stack starts on this ERISC: the stack pointer at kernel entry and the end of the
// kernel's local-data-memory sections (__ldm_bss_end), which is as far down as the stack can grow. Every kernel on a
// given ERISC is entered through the same firmware path, so the router kernel starts from the same stack pointer.

#include <cstdint>

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t sp;
    asm volatile("mv %0, sp" : "=r"(sp));
    extern uint32_t __ldm_bss_end[];
    volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(out_addr);
    out[0] = sp;
    out[1] = reinterpret_cast<uint32_t>(__ldm_bss_end);
    out[2] = 0x5AC0DE;
}
