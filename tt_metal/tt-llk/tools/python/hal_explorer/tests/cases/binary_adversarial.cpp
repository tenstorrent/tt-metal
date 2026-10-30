// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// un-inlined call
__attribute__((noinline)) void helper_issue() { TTI_NOP; TTI_NOP; TTI_NOP; }
void calls_helper() { TTI_NOP; helper_issue(); }
__attribute__((noinline)) void helper_issue_arg(std::uint32_t n) { TT_SETC16(n, 1); }
void calls_helper_arg(std::uint32_t n) { helper_issue_arg(n); helper_issue_arg(n + 1); }
// switch
void sw(const std::uint32_t m) {
    switch (m) { case 0: TTI_NOP; break; case 1: TTI_NOP; TTI_NOP; break; case 2: TTI_NOP; TTI_NOP; TTI_NOP; break; default: break; }
    TTI_SETC16(1, 1);
}
void sw_fall(const std::uint32_t m) {
    switch (m) { case 0: TTI_NOP; [[fallthrough]]; case 1: TTI_NOP; break; case 5: TTI_NOP;TTI_NOP;TTI_NOP;TTI_NOP; break; }
}
// trap path
void trap_path(const bool c) { if (c) { __builtin_trap(); } TTI_NOP; TTI_NOP; }
void unreach_path(const bool c) { if (c) { __builtin_unreachable(); } TTI_NOP; }
void noreturn_then(const bool c) { TTI_NOP; if (c) { __builtin_abort(); } TTI_NOP; TTI_NOP; }
// instrn_buffer direct
void ib_direct_idx(const std::uint32_t w) { ckernel::instrn_buffer[0] = w; ckernel::instrn_buffer[1] = w; }
void ib_raw_sym(const std::uint32_t w) { __instrn_buffer[0] = w; }
void ib_ptr_alias(const std::uint32_t w) { volatile std::uint32_t *p = ckernel::instrn_buffer; p[0] = w; p[0] = w + 1; }
void ib_const_addr(const std::uint32_t w) { *reinterpret_cast<volatile std::uint32_t *>(INSTRN_BUF_BASE) = w; }
void ib_const_addr_imm() { *reinterpret_cast<volatile std::uint32_t *>(INSTRN_BUF_BASE) = TT_OP_NOP; }
void ib_thread_addr(const std::uint32_t w) { *reinterpret_cast<volatile std::uint32_t *>(INSTRN_BUF_BASE + 0x10000) = w; }
void ib_offset4(const std::uint32_t w) { ckernel::instrn_buffer[4] = w; }
// store blocking
void store_blocking_ib(const std::uint32_t w) { store_blocking(&ckernel::instrn_buffer[0], w); }
// pc_buf: not an instruction
void pc_buf() { volatile std::uint32_t *p = reinterpret_cast<volatile std::uint32_t *>(PC_BUF_BASE); p[0] = 1; (void)p[1]; }
// duplicated across paths
void dup_paths(const std::uint32_t a, const std::uint32_t b) { if (a) { TTI_NOP; } else { TTI_SETC16(1, 2); } TT_SETC16(a, b); if (b) { TTI_NOP; } }
// runtime loop with constant body
void rt_loop(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; ++i) { TTI_NOP; TTI_SETC16(1, 2); } }
void const_loop() { for (std::uint32_t i = 0; i < 100; ++i) { TT_SETC16(i, 2); } }
void const_loop_big() { for (std::uint32_t i = 0; i < 5000; ++i) { TT_SETC16(i, 2); } }
void while_loop(const std::uint32_t n) { std::uint32_t i = n; while (i) { TT_SETC16(i, 2); --i; } }
void nested_loops(const std::uint32_t n, const std::uint32_t m) { for (std::uint32_t i = 0; i < n; ++i) { TTI_NOP; for (std::uint32_t j = 0; j < m; ++j) { TTI_SETC16(1, 2); } } }
// do-while with runtime early break
void loop_break(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; ++i) { TT_SETC16(i, 2); if (i == 7) break; } TTI_NOP; }
// inline asm volatile nop (not tensix)
void plain_asm() { asm volatile("nop"); TTI_NOP; }
// multiple instrs same word
void same_twice() { TTI_NOP; TTI_NOP; TTI_NOP; }
