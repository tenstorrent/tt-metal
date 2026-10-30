// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void raw_tti() { TTI_NOP; TTI_SETC16(3, 7); TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK); }
void raw_tt(const std::uint32_t a, const std::uint32_t b) { TT_SETC16(a, b); TT_SETDMAREG(0, a, 0, 3); TTI_NOP; }
void raw_tt_branch(const std::uint32_t a, const bool c) { if (c) { TT_SETC16(a, 3); } TTI_NOP; }
void rmw(const std::uint32_t v) { cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1); cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(v); }
void ck_helpers() { TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);  }
void ck_semaphores() { t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU); t6_semaphore_get<p_stall::MATH>(semaphore::FPU_SFPU); }
void ck_semaphore_wait() { t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::MATH_PACK); }
void ck_mutex() { t6_mutex_acquire(mutex::SFPU); t6_mutex_release(mutex::SFPU); }
void ck_wait_idle() { tensix_sync(); mop_sync(); }
void ck_addr_mod() { addr_mod_t{.srca = {.incr = 1}, .srcb = {.incr = 2}, .dest = {.incr = 3}}.set(ADDR_MOD_0); }
void ck_set_dst_write_addr() { TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0); }
void ck_reg_write() { reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1); }
