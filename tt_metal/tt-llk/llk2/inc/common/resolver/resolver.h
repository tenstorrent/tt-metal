// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The Resolver's driver (tt-llk-lab llk-tensor, report 3, section 4), for a chain of one: each Compute API call
// resolves only itself. An op provides four pure functions in an OpApi:
//   accepts(in0, in1, mods, settings)       can the op run on these tensors at all
//   options(in0, in1, mods, settings)       negotiation: the representation it runs in      -> OpConfig
//   work(in0, in1, config)                  allocation: loop counts                         -> Work
//   plan(in0, in1, mods, config, work)      planning: what each unit does                   -> Plan
// The driver runs them in that order, at compile time. accepted<...> and resolved<...> name the results, so they
// are compile-time objects; the Compute API static_asserts accepted<...> before it reads resolved<...>, so an op
// that cannot run on its tensors fails the build at the call, with a message.
//
// OpApi is a class template over the op's modifier struct, so every op's functions take their own modifier type.
// The functions themselves are plain constexpr functions reached through pointers: they cannot be templates, and do
// not need to be, because everything they read arrives as a constexpr argument.

#pragma once

#include "kernel_settings.h"
#include "resolver/plan.h"

namespace llk2
{

template <class Mods>
struct OpApi
{
    bool (*accepts)(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& mods, const KernelSettings& settings);
    OpConfig (*options)(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& mods, const KernelSettings& settings);
    Work (*work)(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const OpConfig& config);
    Plan (*plan)(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& mods, const OpConfig& config, const Work& work);
};

struct Resolved
{
    OpConfig config;
    Work work;
    Plan plan;
};

template <class Mods>
constexpr Resolved resolve(const OpApi<Mods>& api, const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& mods, const KernelSettings& settings)
{
    const OpConfig config = api.options(in0, in1, mods, settings);
    const Work work       = api.work(in0, in1, config);
    return {config, work, api.plan(in0, in1, mods, config, work)};
}

template <const auto& Api, const TensorSpec<tensor_rank>& A, const TensorSpec<tensor_rank>& B, auto Mods>
inline constexpr bool accepted = Api.accepts(A, B, Mods, kernel_settings);

template <const auto& Api, const TensorSpec<tensor_rank>& A, const TensorSpec<tensor_rank>& B, auto Mods>
inline constexpr Resolved resolved = resolve(Api, A, B, Mods, kernel_settings);

} // namespace llk2
