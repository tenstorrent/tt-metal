// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"

void r_const()
{
    replay::record<replay::BufferRange {3, 5}>([&] { TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; });
    replay::run<replay::BufferRange {3, 5}>();
}
void r_wrap()
{
    replay::record<replay::BufferRange {30, 5}>([&] { TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; });
    replay::run<replay::BufferRange {30, 5}>();
}
void r_32()
{
    replay::record<replay::BufferRange {0, 32}, replay::RecordBehavior::RecordAndExecute>([&] { for (int i=0;i<32;i++) TTI_NOP; });
    replay::run<replay::BufferRange {0, 32}>();
}
void r_rt(const std::uint32_t n)
{
    replay::record(replay::BufferRange {4, n}, [&] { TTI_NOP; TTI_NOP; });
    replay::run(replay::BufferRange {4, n});
}
void r_rt_exec(const std::uint32_t n)
{
    replay::record<replay::RecordBehavior::RecordAndExecute>(replay::BufferRange {4, n}, [&] { TTI_NOP; TTI_NOP; });
}
void r_rt_ternary(const bool b)
{
    replay::run(replay::BufferRange {7, b ? 12u : 10u});
}
