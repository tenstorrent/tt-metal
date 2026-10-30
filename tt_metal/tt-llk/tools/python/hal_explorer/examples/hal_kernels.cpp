// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

namespace adc      = hal::counters::adc;
namespace cfg      = hal::cfg;
namespace counters = hal::counters;
namespace mop      = hal::mop;
namespace replay   = hal::replay;
namespace scalar   = hal::scalar;
namespace sync     = hal::sync;
using namespace ckernel;

// Pack rows: HAL version of _llk_pack_rows_init_ (llk_pack_rows.h).
// The PACR words are data written into the MOP; the HAL has no PACR builder yet.
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
    constexpr std::uint32_t row_datums = 16;

    // ADDR_MOD_0 advances the source Y counter by one row; ADDR_MOD_1 clears it.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Incr, cfg::Sec::S0, 1>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Clear, cfg::Sec::S1, 1>());

    // One PACR per row; the last row closes the block and resets Y.
    mop::program(mop::loop(1 /*count*/, mop::loop(num_rows, pack_row(ADDR_MOD_0, 0 /*last*/)))
                     .patch(mop::last_outer, pack_row(ADDR_MOD_1, 1 /*last*/)));

    // Y_POS resets after one read per XY plane.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>());

    // Pack row_datums datums per row, then start Z and W from zero.
    constexpr auto x_range  = adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = row_datums - 1}}.for_clients<adc::Client::Packers>();
    constexpr auto zw_reset = adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Packers>();
    counters::set<x_range>();
    counters::set<zw_reset>();
}

// Pack reduce mask clear: HAL version of _llk_pack_reduce_mask_clear_ (llk_pack_common.h).
void pack_reduce_mask_clear()
{
    constexpr auto pass_through_mask = hal::gpr<p_gpr_pack::TMP0>();
    constexpr auto zero              = hal::gpr<p_gpr::ZERO>();

    scalar::set<0xffffu>(pass_through_mask);
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0>(pass_through_mask);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeOffsetSec1::mask, cfg::Sec::S0>(pass_through_mask);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[0][0], cfg::Sec::S0>(zero);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[1][0], cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(zero);
}

// Matmul operand A: advance the L1 base address of both config contexts by one tile,
// recorded into the replay buffer and replayed (llk_unpack_AB_matmul.h:64).
void unpack_matmul_advance_a(const bool partial_face)
{
    constexpr auto reset_z = adc::AddrCounter {.channel0 = {.z = 0}, .channel1 = {.z = 0}}.for_clients<adc::Client::Unpacker0>();
    constexpr auto tmp     = hal::gpr<p_gpr_unpack::TMP0>();

    replay::record(
        replay::BufferRange {0, partial_face ? 12u : 10u},
        [&]
        {
            if (partial_face)
            {
                counters::set<reset_z>();
            }
            cfg::read<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(tmp);
            scalar::add(tmp, hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), tmp);
            sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
            cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(tmp);

            if (partial_face)
            {
                counters::set<reset_z>();
            }
            cfg::read<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0>(tmp);
            scalar::add(tmp, hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), tmp);
            sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
            cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(tmp);
        });
    replay::run(replay::BufferRange {0, partial_face ? 12u : 10u});
}

// Math/pack handshake: math posts a full Dest section, pack waits for it and frees it.
void math_publish_dest()
{
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
    sync::semaphore::post<sync::Semaphore::S1>();
}

void pack_consume_dest()
{
    sync::wait::semaphore<sync::StallTarget::Pack, sync::SemaphoreCondition::WhileZero, sync::Semaphore::S1>();
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::PackerIdle>();
    sync::semaphore::get<sync::Semaphore::S1>();
}
