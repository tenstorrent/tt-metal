// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// Version: FFN1.3.0

#ifndef __OVERLAY_INTERRUPTS_HPP__
#define __OVERLAY_INTERRUPTS_HPP__

#include "overlay_hwtest.h"
#include "metal/drivers/riscv_cpu.h"

uint64_t __ig_mcause_ovl(void);
void __ig_metal_interrupt_external_enable_ovl(void);
void __ig_metal_interrupt_global_enable_ovl(void);
void register_interrupt(size_t core, uint32_t irq, void (*custom_handler)());
void disable_interrupt(size_t core, uint32_t irq);
void claim_interrupt(uint32_t irq);
void clear_interrupt(uint32_t irq);

#endif
