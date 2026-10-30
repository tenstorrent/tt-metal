// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

using hal::mop::MopTemplate;
using replay::BufferRange;
void br_scalar(const bool c, const std::uint32_t v) {
    if (c) { scalar::set<0x1234>(hal::gpr<5>()); } else { scalar::add(hal::gpr<1>(), hal::gpr<2>(), scalar::immediate(v)); scalar::add(hal::gpr<1>(), hal::gpr<2>(), scalar::immediate<3>()); }
    sync::semaphore::post<sync::Semaphore::S1>();
}
void br_sync(const bool c) {
    if (c) { sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>(); sync::semaphore::post<sync::Semaphore::S1>(); }
    else { sync::wait::semaphore<sync::StallTarget::Pack, sync::SemaphoreCondition::WhileZero, sync::Semaphore::S1>(); }
    sync::semaphore::get<sync::Semaphore::S1>();
}
void br_cfg(const bool c, const std::uint32_t v) {
    if (c) { cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>(); }
    else { cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(v); cfg::write<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0, 0x12345678>(); }
    sync::semaphore::post<sync::Semaphore::S2>();
}
void br_cfg_mmio(const bool c, const std::uint32_t v) {
    if (c) { cfg::write<cfg::Access::MMIO, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(v); } else { cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(v); }
}
void br_counters(const bool c, const std::uint32_t x) {
    constexpr auto reset_z = adc::AddrCounter {.channel0 = {.z = 0}, .channel1 = {.z = 0}}.for_clients<adc::Client::Unpacker0>();
    if (c) { counters::set<reset_z>(); } else { TTI_NOP; TTI_NOP; }
    TT_SETADCXX(p_setadc::UNP_A, x, 0);
}
void br_mop(const bool c, const std::uint32_t n) {
    if (c) { mop::program(mop::loop(1 /*count*/, mop::loop(n, TT_OP_NOP))); } else { mop::program(mop::loop(1 /*count*/, mop::loop(4, TT_OP_NOP))); }
    mop::Runner<MopTemplate::Template1>::run();
}
void br_replay(const bool a, const bool b) {
    replay::record(BufferRange{0, (a ? 2u : 1u) + (b ? 3u : 1u)}, [&] {
        if (a) { TTI_NOP; TTI_NOP; } else { TTI_SETC16(1, 1); }
        if (b) { TTI_NOP; TTI_NOP; TTI_NOP; } else { TTI_SETC16(2, 2); }
    });
    replay::run(BufferRange{0, (a ? 2u : 1u) + (b ? 3u : 1u)});
}
void br_loop_hal(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; ++i) { scalar::add(hal::gpr<1>(), hal::gpr<1>(), scalar::immediate(i)); } }
void br_loop_hal_const() { for (std::uint32_t i = 0; i < 40; ++i) { scalar::add(hal::gpr<1>(), hal::gpr<1>(), scalar::immediate(i)); TTI_NOP; } }
void br_gpr_index_loop(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; ++i) { scalar::add(hal::gpr(i), hal::gpr(i), scalar::immediate<1>()); } }
void br_misc(const bool c, const std::uint32_t v) { if (c) { TT_SETC16(v, 1); } else if (v == 3) { TTI_NOP; } else { TTI_NOP; TTI_NOP; } TTI_NOP; }
void br_chain(const bool a, const bool b, const bool c) { if (a) TTI_NOP; if (b) TTI_SETC16(1,1); if (c) { TTI_NOP; TTI_NOP; } else { TTI_SETC16(2, 2); } }
void br_ternary(const bool a) { TT_SETC16(a ? 5u : 6u, 1); if (a) { TTI_NOP; } else { TTI_NOP; TTI_NOP; } }
