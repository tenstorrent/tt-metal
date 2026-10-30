// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_rows.h"

namespace adc      = hal::counters::adc;
namespace cfg      = hal::cfg;
namespace counters = hal::counters;
namespace mop      = hal::mop;
namespace replay   = hal::replay;
namespace scalar   = hal::scalar;
namespace sync     = hal::sync;
using namespace ckernel;

// Original: tt_llk_blackhole/llk_lib/llk_pack_rows.h, _llk_pack_rows_init_
void reference_pack_rows_init(const std::uint32_t num_rows)
{
    _llk_pack_rows_init_(num_rows);
}

// HAL rewrite.
// The HAL has no PACR builder yet, so the two MOP ops stay encoded with TT_OP_PACR;
// they are data written into the MOP configuration, not instructions issued here.
constexpr std::uint32_t pack_row(const std::uint32_t addr_mod, const std::uint32_t last)
{
    return TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        addr_mod,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::SINGLE_INTF_ACTIVE,
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        p_pacr::NO_CTXT_CTRL,
        0 /*Flush*/,
        last);
}

void pack_rows_init(const std::uint32_t num_rows)
{
    constexpr std::uint32_t row_datums = FACE_C_DIM;

    // ADDR_MOD_0 advances the source Y counter by one row; ADDR_MOD_1 clears it.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Incr, cfg::Sec::S0, 1>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Clear, cfg::Sec::S1, 1>());

    // One PACR per row; the final row closes the block and resets Y.
    mop::program(mop::loop(1 /*count*/, mop::loop(num_rows, pack_row(ADDR_MOD_0, 0 /*last*/)))
                     .patch(mop::last_outer, pack_row(ADDR_MOD_1, 1 /*last*/)));

    // Y_POS resets after one read per XY plane.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>());

    // Pack row_datums datums per row, then start Z and W from zero.
    constexpr auto x_range = adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = row_datums - 1}}.for_clients<adc::Client::Packers>();
    constexpr auto zw_reset =
        adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Packers>();
    counters::set<x_range>();
    counters::set<zw_reset>();
}
