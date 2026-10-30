// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"
using cfg::Access;
using cfg::Sec;

void st_alu_srcb(const std::uint32_t v)
{
    cfg::write<Access::MMIO, cfg::AluFormatSpecReg::SrcB_val, Sec::S0>(v);
}
void st_alu_round(const std::uint32_t v)
{
    cfg::write<Access::MMIO, cfg::AluRoundingMode::Fpu_srnd_en, Sec::S0>(v);
}
void st_pack(const std::uint32_t v)
{
    cfg::write<Access::MMIO, cfg::Pck0AddrCtrlXyReg0::Xstride, Sec::S0>(v);
}
void st_thcon(const std::uint32_t v)
{
    cfg::write<Access::MMIO, cfg::Thcon[cfg::Reg3].Base_address, Sec::S0>(v);
}
void st_two_fields(const std::uint32_t v, const std::uint32_t w)
{
    cfg::write<Access::MMIO>(cfg::set<cfg::AluFormatSpecReg::SrcA_val, Sec::S0>(v), cfg::set<cfg::AluFormatSpecReg::SrcB_val, Sec::S0>(w));
}
std::uint32_t th_read()
{
    return cfg::read<Access::MMIO, cfg::CfgStateId::StateID, Sec::S0, cfg::ThreadTarget::T1>();
}
void sem()
{
    sync::semaphore::post<sync::Access::MMIO, sync::Semaphore::S1>();
}
void sem_get(const std::uint32_t i)
{
    sync::semaphore::get<sync::Access::MMIO>(static_cast<sync::Semaphore>(i));
}
void direct_mmio()
{
    ckernel::mailbox_write(1, 5);
    const std::uint32_t r = ckernel::regfile[7];
    ckernel::regfile[8]   = r;
    *reinterpret_cast<volatile std::uint32_t *>(0xFFE90004) = 1;
    *reinterpret_cast<volatile std::uint32_t *>(0x20000)    = 1;
    *reinterpret_cast<volatile std::uint32_t *>(0xFFB13004) = 1;
}
