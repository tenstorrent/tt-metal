// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"

namespace adc      = hal::counters::adc;
namespace cfg      = hal::cfg;
namespace counters = hal::counters;
namespace mop      = hal::mop;
namespace replay   = hal::replay;
namespace scalar   = hal::scalar;
namespace sync     = hal::sync;
using namespace ckernel;

// Original: tt_llk_blackhole/llk_lib/llk_pack_common.h, _llk_pack_reduce_mask_clear_
void reference_pack_reduce_mask_clear()
{
    _llk_pack_reduce_mask_clear_();
}

// HAL rewrite
void pack_reduce_mask_clear()
{
    constexpr auto pass_through_mask = hal::gpr<p_gpr_pack::TMP0>();
    constexpr auto zero              = hal::gpr<p_gpr::ZERO>();

    // Every packer uses row-set mapping 0, whose mask passes all 16 datums.
    scalar::set<0xffffu>(pass_through_mask);

    // Wait for the packer before changing its configuration.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();

    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>());

    cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0>(pass_through_mask);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeOffsetSec1::mask, cfg::Sec::S0>(pass_through_mask);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[0][0], cfg::Sec::S0>(zero);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[1][0], cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(zero);
}
