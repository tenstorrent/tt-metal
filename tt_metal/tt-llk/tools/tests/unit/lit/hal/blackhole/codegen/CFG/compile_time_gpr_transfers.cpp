// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include "hal/cfg.h"

namespace cfg = hal::cfg;

extern "C" __attribute__((noinline, used)) void read_cfg_to_common_gpr()
{
    cfg::read<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(hal::gpr<5>());
}

// CHECK-LABEL: <read_cfg_to_common_gpr>:
// CHECK-NEXT: ttrdcfg 5,186
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void read_cfg_to_policy_gpr()
{
    cfg::read<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S1>(cfg::gpr<6>());
}

// CHECK-LABEL: <read_cfg_to_policy_gpr>:
// CHECK-NEXT: ttrdcfg 6,112
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_common_gpr_wait()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>());
}

// CHECK-LABEL: <write_common_gpr_wait>:
// CHECK-NEXT: ttwrcfg 4,0,76
// CHECK-NEXT: ttnop
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_policy_gpr_deferred()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0>(
        cfg::gpr<5, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>());
}

// CHECK-LABEL: <write_policy_gpr_deferred>:
// CHECK-NEXT: ttwrcfg 5,0,77
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_policy_gpr_128_wait()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S1>(cfg::gpr<16, cfg::GprTransferSize::Bits128>());
}

// CHECK-LABEL: <write_policy_gpr_128_wait>:
// CHECK-NEXT: ttwrcfg 16,1,112
// CHECK-NEXT: ttnop
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_thcon_common_gpr_scalar()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>());
}

// CHECK-LABEL: <write_thcon_common_gpr_scalar>:
// CHECK-NEXT: ttreg2flop 1,0,0,0,12,4
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_thcon_policy_gpr_128_scalar()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S1>(cfg::gpr<16, cfg::GprTransferSize::Bits128>());
}

// CHECK-LABEL: <write_thcon_policy_gpr_128_scalar>:
// CHECK-NEXT: ttreg2flop 0,0,0,0,48,16
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_state_reset_via_gpr()
{
    // A field-granular Tensix write to the state-reset register lowers to
    // RMWCIB, which hardware ignores; a whole-word GPR transfer reaches it.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::StateReset::EN, cfg::Sec::S0>(hal::gpr<4>());
}

// CHECK-LABEL: <write_state_reset_via_gpr>:
// CHECK-NEXT: ttwrcfg 4,0,4
// CHECK-NEXT: ttnop
// CHECK-NEXT: ret
