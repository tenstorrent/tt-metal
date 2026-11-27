// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.

#include "hal/cfg.h"

namespace cfg = hal::cfg;

extern "C" __attribute__((noinline, used)) void add_scratch_thread()
{
    cfg::add_scratch<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::ScratchSec::Thread>();
}

extern "C" __attribute__((noinline, used)) void reference_add_scratch_thread()
{
    TTI_CFGSHIFTMASK(
        1 /*disable_mask_on_old_val*/,
        0b011 /*operation*/,
        32 - 1 /*mask_width*/,
        0 /*right_cshift_amt*/,
        0b11 /*scratch_sel*/,
        THCON_SEC0_REG3_Base_address_ADDR32);
}

extern "C" __attribute__((noinline, used)) void add_scratch_sections()
{
    cfg::add_scratch<cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0, cfg::ScratchSec::S0>();
    cfg::add_scratch<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::ScratchSec::S1>();
    cfg::add_scratch<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::ScratchSec::S2>();
}

extern "C" __attribute__((noinline, used)) void reference_add_scratch_sections()
{
    TTI_CFGSHIFTMASK(
        1 /*disable_mask_on_old_val*/,
        3 /*operation*/,
        32 - 1 /*mask_width*/,
        0 /*right_cshift_amt*/,
        0 /*scratch_sel*/,
        THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
    TTI_CFGSHIFTMASK(
        1 /*disable_mask_on_old_val*/, 3 /*operation*/, 32 - 1 /*mask_width*/, 0 /*right_cshift_amt*/, 1 /*scratch_sel*/, THCON_SEC0_REG3_Base_address_ADDR32);
    TTI_CFGSHIFTMASK(
        1 /*disable_mask_on_old_val*/, 3 /*operation*/, 32 - 1 /*mask_width*/, 0 /*right_cshift_amt*/, 2 /*scratch_sel*/, THCON_SEC0_REG3_Base_address_ADDR32);
}

extern "C" __attribute__((noinline, used)) void add_scratch_wait()
{
    cfg::add_scratch<cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0, cfg::ScratchSec::Thread, cfg::WrcfgCompletion::Wait>();
}

extern "C" __attribute__((noinline, used)) void reference_add_scratch_wait()
{
    TTI_CFGSHIFTMASK(
        1 /*disable_mask_on_old_val*/,
        0b011 /*operation*/,
        32 - 1 /*mask_width*/,
        0 /*right_cshift_amt*/,
        0b11 /*scratch_sel*/,
        THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    TTI_NOP;
}
