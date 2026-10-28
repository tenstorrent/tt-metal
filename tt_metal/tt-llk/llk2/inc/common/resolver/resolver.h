// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The Resolver's driver (report 3, section 4), for a chain of one: each Compute API call resolves only itself.
// An op provides three pure functions in an OpApi; the driver runs them in order, at compile time:
//   options(in0, in1, mods, settings) -> OpConfig   negotiation; an op that cannot run them fails the build
//   work(in0, in1, config)            -> Work       loop counts
//   plan(in0, in1, config, work)      -> Plan       what each unit does
// resolved<Api, A, B, Mods> names the result, so it is a compile-time object that can be passed by reference.

#pragma once

#include "kernel_settings.h"
#include "resolver/plan.h"

namespace llk2
{

template <class Mods>
struct OpApi
{
    OpConfig (*options)(const TensorSpec<6>& in0, const TensorSpec<6>& in1, const Mods& mods, const KernelSettings& settings);
    Work (*work)(const TensorSpec<6>& in0, const TensorSpec<6>& in1, const OpConfig& config);
    Plan (*plan)(const TensorSpec<6>& in0, const TensorSpec<6>& in1, const Mods& mods, const OpConfig& config, const Work& work);
};

struct Resolved
{
    OpConfig config;
    Work work;
    Plan plan;
};

template <class Mods>
constexpr Resolved resolve(const OpApi<Mods>& api, const TensorSpec<6>& in0, const TensorSpec<6>& in1, const Mods& mods, const KernelSettings& settings)
{
    const OpConfig config = api.options(in0, in1, mods, settings);
    const Work work       = api.work(in0, in1, config);
    return {config, work, api.plan(in0, in1, mods, config, work)};
}

template <const auto& Api, const TensorSpec<6>& A, const TensorSpec<6>& B, auto Mods>
inline constexpr Resolved resolved = resolve(Api, A, B, Mods, kernel_settings);

// A failed check inside a constexpr evaluation is a compile error at the call that needed it.
constexpr void check(bool ok)
{
    if (!ok)
    {
        __builtin_trap();
    }
}

} // namespace llk2
