// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: blackhole-cfg
// RUN: %clangxx -std=c++17 -Wall -Wextra -Werror -I %{llk_root}/tt_llk_blackhole/llk_lib -I %{llk_root}/common -I %{blackhole_cfg_include} %s -o %t
// RUN: %t

#include <cassert>
#include <cstdint>

#include "hal/cfg/detail/write_plan.h"

using namespace hal::cfg;
using namespace hal::cfg::detail;

inline constexpr Field state_low {RegisterScope::State, 32, 5, 0, 0, 4, 1, 0};
inline constexpr Field state_nibble {RegisterScope::State, 32, 5, 0, 4, 4, 1, 0};
inline constexpr Field state_high {RegisterScope::State, 32, 5, 0, 8, 4, 1, 0};
inline constexpr Field state_tail {RegisterScope::State, 32, 5, 0, 16, 4, 1, 0};
inline constexpr Field thread_low {RegisterScope::Thread, 16, 5, 0, 0, 4, 1, 0};
inline constexpr Field thread_high {RegisterScope::Thread, 16, 5, 0, 8, 4, 1, 0};
inline constexpr Field transfer {RegisterScope::State, 32, 8, 0, 0, 32, 1, 0};
inline constexpr Field transfer_last {RegisterScope::State, 32, 11, 0, 0, 4, 1, 0};
inline constexpr Field transfer_next {RegisterScope::State, 32, 12, 0, 0, 4, 1, 0};
inline constexpr Field sectioned {RegisterScope::State, 32, 64, 0, 0, 4, 2, 1536};

using StateLow   = FieldAssignment<state_low, Sec::S0>;
using StateHigh  = FieldAssignment<state_high, Sec::S0>;
using ThreadLow  = ConstantFieldAssignment<thread_low, Sec::S0, 2>;
using ThreadHigh = ConstantFieldAssignment<thread_high, Sec::S0, 3>;
using Transfer   = GprWrite<transfer, Sec::S0, 4, GprTransferSize::Bits128, WrcfgCompletion::Wait>;

// Numeric addresses collide across scopes; nonadjacent thread fields must still
// share one SETC16 group. Constants are classified per group, not per call.
constexpr auto scopes = write_plan_v<ThreadLow, StateLow, ThreadHigh, StateHigh>;
static_assert(scopes.disjoint && scopes.group_count == 2);
static_assert(scopes.groups[0].scope == RegisterScope::Thread && scopes.groups[1].scope == RegisterScope::State);
static_assert(scopes.groups[0].addr == 5 && scopes.groups[1].addr == 5);
static_assert(scopes.groups[0].mask == 0xf0f && scopes.groups[0].data == 0x302);
static_assert(scopes.groups[0].all_constant && !scopes.groups[1].all_constant);
static_assert(scopes.groups[0].first == 0 && scopes.groups[0].count == 2 && scopes.group_of[0] == 0 && scopes.group_of[2] == 0);
static_assert(scopes.groups[1].first == 1 && scopes.groups[1].count == 2 && scopes.group_of[1] == 1 && scopes.group_of[3] == 1);

// Direct word slots collect masks; the active list keeps first-use order.
// Thread masks become [0, 1, 0, 6, ...], State masks [0, 0, 2, ...].
constexpr auto slots = build_write_plan(std::array<WriteOperandMeta, 4> {{
    {RegisterScope::Thread, 1, 0x01, 1, false, false, 0},
    {RegisterScope::Thread, 3, 0x02, 1, false, false, 0},
    {RegisterScope::State, 2, 0x02, 1, false, false, 0},
    {RegisterScope::Thread, 3, 0x04, 1, false, false, 0},
}});
static_assert(slots.disjoint && slots.group_count == 3);
static_assert(slots.groups[0].addr == 1 && slots.groups[0].mask == 1 && slots.groups[0].first == 0);
static_assert(slots.groups[1].addr == 3 && slots.groups[1].mask == 6 && slots.groups[1].first == 1);
static_assert(slots.groups[2].scope == RegisterScope::State && slots.groups[2].addr == 2 && slots.groups[2].mask == 2 && slots.groups[2].first == 2);
static_assert(slots.group_of[0] == 0 && slots.group_of[1] == 1 && slots.group_of[2] == 2 && slots.group_of[3] == 1);
static_assert(slots.groups[0].count == 1 && slots.groups[1].count == 2 && slots.groups[2].count == 1);

// Resolve sections before grouping. Different sections may name different words.
constexpr auto sections = write_plan_v<FieldAssignment<sectioned, Sec::S1>, FieldAssignment<sectioned, Sec::S0>>;
static_assert(sections.disjoint && sections.group_count == 2);
static_assert(sections.groups[0].addr == 112 && sections.groups[1].addr == 64);

// Fields on both sides of a GPR transfer share one group. Thread fields must be
// composed into one SETC16, so the second field does not clear the first one.
constexpr auto across_gpr = write_plan_v<ThreadLow, Transfer, ThreadHigh>;
static_assert(across_gpr.disjoint && across_gpr.group_count == 2);
static_assert(across_gpr.groups[0].first == 0 && across_gpr.groups[1].first == 1);
static_assert(across_gpr.group_of[0] == 0 && across_gpr.group_of[1] == 1 && across_gpr.group_of[2] == 0);
static_assert(across_gpr.groups[0].count == 2 && across_gpr.groups[1].count == 1);
static_assert(across_gpr.groups[0].mask == 0xf0f && across_gpr.groups[0].data == 0x302 && across_gpr.groups[0].all_constant);
constexpr auto constant_byte = write_plan_v<ConstantFieldAssignment<state_low, Sec::S0, 3>, Transfer, ConstantFieldAssignment<state_nibble, Sec::S0, 5>>;
static_assert(constant_byte.disjoint && constant_byte.group_count == 2);
static_assert(constant_byte.groups[0].mask == 0xff && constant_byte.groups[0].data == 0x53 && constant_byte.groups[0].all_constant);
static_assert(constant_byte.group_of[0] == 0 && constant_byte.group_of[1] == 1 && constant_byte.group_of[2] == 0);
static_assert(constant_byte.groups[0].count == 2 && constant_byte.groups[1].first == 1);
constexpr auto across_gpr_groups = write_plan_v<StateLow, ThreadLow, Transfer, StateHigh, ThreadHigh, FieldAssignment<state_tail, Sec::S0>>;
static_assert(across_gpr_groups.disjoint && across_gpr_groups.group_count == 3);
static_assert(across_gpr_groups.groups[0].first == 0 && across_gpr_groups.groups[1].first == 1 && across_gpr_groups.groups[2].first == 2);
static_assert(across_gpr_groups.group_of[0] == 0 && across_gpr_groups.group_of[3] == 0 && across_gpr_groups.group_of[5] == 0);
static_assert(across_gpr_groups.group_of[1] == 1 && across_gpr_groups.group_of[4] == 1 && across_gpr_groups.group_of[2] == 2);
static_assert(across_gpr_groups.groups[0].count == 3 && across_gpr_groups.groups[1].count == 2 && across_gpr_groups.groups[2].count == 1);
static_assert(across_gpr_groups.groups[0].mask == 0xf0f0f && !across_gpr_groups.groups[0].all_constant);
static_assert(across_gpr_groups.groups[1].data == 0x302 && across_gpr_groups.groups[1].all_constant);

// A mixed group's constants are precombined even when its first operand is
// constant. Runtime data must be combined with those bits during emission.
constexpr auto constant_first = write_plan_v<ConstantFieldAssignment<state_low, Sec::S0, 3>, Transfer, StateHigh>;
constexpr auto runtime_first  = write_plan_v<StateHigh, Transfer, ConstantFieldAssignment<state_low, Sec::S0, 3>>;
static_assert(constant_first.disjoint && runtime_first.disjoint);
static_assert(constant_first.groups[0].count == 2 && !constant_first.groups[0].all_constant && constant_first.groups[0].data == 3);
static_assert(runtime_first.groups[0].count == 2 && !runtime_first.groups[0].all_constant && runtime_first.groups[0].data == 3);
static_assert(constant_first.group_of[0] == 0 && constant_first.group_of[2] == 0 && runtime_first.group_of[0] == 0 && runtime_first.group_of[2] == 0);

// A group can span multiple transfers. Groups and transfers retain the order
// of their first occurrence, even when the call starts with a GPR transfer.
using NextTransfer           = GprWrite<transfer_next, Sec::S0, 8, GprTransferSize::Bits32, WrcfgCompletion::Deferred>;
constexpr auto multiple_gprs = write_plan_v<Transfer, StateLow, NextTransfer, StateHigh>;
static_assert(multiple_gprs.disjoint && multiple_gprs.group_count == 3);
static_assert(multiple_gprs.groups[0].first == 0 && multiple_gprs.groups[1].first == 1 && multiple_gprs.groups[2].first == 2);
static_assert(multiple_gprs.group_of[0] == 0 && multiple_gprs.group_of[1] == 1 && multiple_gprs.group_of[2] == 2 && multiple_gprs.group_of[3] == 1);
static_assert(multiple_gprs.groups[0].count == 1 && multiple_gprs.groups[1].count == 2 && multiple_gprs.groups[2].count == 1);

// Overlap checks still cover the entire call and every GPR destination word.
static_assert(!write_plan_v<StateLow, Transfer, StateLow>.disjoint);
static_assert(!write_plan_v<StateLow, StateLow>.disjoint);
static_assert(!write_plan_v<Transfer, Transfer>.disjoint);
static_assert(!write_plan_v<Transfer, FieldAssignment<transfer_last, Sec::S0>>.disjoint);
static_assert(!write_plan_v<FieldAssignment<transfer_last, Sec::S0>, Transfer>.disjoint);
static_assert(write_plan_v<Transfer, FieldAssignment<transfer_next, Sec::S0>>.disjoint);

// Every word in both hardware spaces is usable as a distinct bucket. Reversing
// addresses checks that emission order is first occurrence, not address order.
constexpr auto all_words = []
{
    std::array<WriteOperandMeta, StateCfgWordCount + ThreadCfgWordCount> operands {};
    std::size_t i = 0;
    for (std::uint32_t addr = StateCfgWordCount; addr != 0; --addr)
    {
        operands[i++] = {RegisterScope::State, addr - 1u, 0xffffffffu, 1u, false, true, 0u};
    }
    for (std::uint32_t addr = ThreadCfgWordCount; addr != 0; --addr)
    {
        operands[i++] = {RegisterScope::Thread, addr - 1u, 0xffffu, 1u, false, true, 0u};
    }
    return build_write_plan(operands);
}();
static_assert(all_words.disjoint && all_words.group_count == StateCfgWordCount + ThreadCfgWordCount);
static_assert(all_words.groups[0].addr == StateCfgWordCount - 1u);
static_assert(all_words.groups[StateCfgWordCount].addr == ThreadCfgWordCount - 1u);
static_assert(all_words.groups.back().addr == 0);
static_assert(write_plan_v<>.disjoint && write_plan_v<>.group_count == 0);

// Compare generated operation lists with a separate quadratic reference model.
// It finds groups by scanning all earlier field operands and checks
// overlaps by pairwise span intersections, without using a word-slot table.
int main()
{
    std::uint32_t seed = 1;
    auto random        = [&]()
    {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 16;
    };
    for (unsigned trial = 0; trial < 256; ++trial)
    {
        constexpr std::size_t count = 32;
        std::array<WriteOperandMeta, count> operands {};
        for (auto& operand : operands)
        {
            operand.is_gpr      = random() % 8 == 0;
            operand.scope       = operand.is_gpr || random() % 2 ? RegisterScope::State : RegisterScope::Thread;
            operand.addr        = random() % 16;
            operand.words       = operand.is_gpr ? (random() % 2 ? 4u : 1u) : 1u;
            operand.mask        = operand.is_gpr ? 0xffffffffu : 1u << (random() % 16);
            operand.is_constant = !operand.is_gpr && random() % 2;
            operand.data        = operand.is_constant && random() % 2 ? operand.mask : 0u;
        }

        const auto plan = build_write_plan(operands);
        std::array<std::size_t, count> expected_group {};
        std::size_t groups = 0;
        bool disjoint      = true;
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto& operand = operands[i];
            std::size_t first   = i;
            if (!operand.is_gpr)
            {
                for (std::size_t j = 0; j < i; ++j)
                {
                    if (!operands[j].is_gpr && operands[j].scope == operand.scope && operands[j].addr == operand.addr)
                    {
                        first = j;
                        break;
                    }
                }
            }
            expected_group[i] = first == i ? groups++ : expected_group[first];
            for (std::size_t j = 0; j < i; ++j)
            {
                const auto& prior = operands[j];
                if (prior.scope == operand.scope && prior.addr < operand.addr + operand.words && operand.addr < prior.addr + prior.words &&
                    (prior.mask & operand.mask) != 0)
                {
                    disjoint = false;
                }
            }
        }
        assert(plan.disjoint == disjoint && plan.group_count == groups);
        std::size_t visited = 0;
        for (std::size_t group = 0; group < groups; ++group)
        {
            std::size_t members = 0;
            std::uint32_t mask  = 0;
            std::uint32_t data  = 0;
            bool all_constant   = true;
            for (std::size_t i = 0; i < count; ++i)
            {
                if (expected_group[i] == group)
                {
                    assert(plan.group_of[i] == group);
                    if (members == 0)
                    {
                        assert(plan.groups[group].first == i);
                    }
                    ++members;
                    ++visited;
                    assert(plan.groups[group].scope == operands[i].scope && plan.groups[group].addr == operands[i].addr);
                    mask |= operands[i].mask;
                    data |= operands[i].data;
                    all_constant &= operands[i].is_constant;
                }
            }
            assert(plan.groups[group].count == members);
            assert(plan.groups[group].mask == mask && plan.groups[group].data == data && plan.groups[group].all_constant == all_constant);
        }
        assert(visited == count);
    }
}
