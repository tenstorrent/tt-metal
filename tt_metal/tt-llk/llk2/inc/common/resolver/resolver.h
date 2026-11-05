// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The Resolver's driver (tt-llk-lab llk-tensor, report 3, section 4:
// https://github.com/tenstorrent/tt-llk-lab/tree/lpremovic/llk-tensor/experiments/llk-tensor), for a chain of one:
// each Compute API call resolves only itself. An op provides four pure functions in an OpApi:
//   accepts(in0, in1, mods, settings)       can the op run on these tensors at all
//   options(in0, in1, mods, settings)       negotiation: the representation it runs in      -> OpConfig
//   work(in0, in1, config)                  allocation: loop counts                         -> Work
//   plan(in0, in1, mods, config, work)      planning: what each unit does                   -> Plan
// The driver runs them in that order, at compile time. accepted<...> and resolved<...> name the results, so they
// are compile-time objects; the Compute API static_asserts accepted<...> before it reads resolved<...>, so an op
// that cannot run on its tensors fails the build at the call, with a message.
//
// OpApi is a class template over what the op's Compute API call hands the Resolver, so every op's functions take
// their own type. For most ops that is the op's modifier struct. The SFPU activations share one OpApi (vec_op) and
// one modifier struct (vec_modifiers), so their calls hand over a vec_call (resolver/vec_function.h): the
// activation plus the modifiers. The functions themselves are plain constexpr functions reached through pointers:
// they cannot be templates, and do not need to be, because everything they read arrives as a constexpr argument.

#pragma once

#include "kernel_settings.h"
#include "resolver/plan.h"

namespace llk2
{

/// An op's four Resolver steps. Mods is what the op's Compute API call hands over: its modifier struct (vec_call
/// for the SFPU activations).
template <class Mods>
struct OpApi
{
    /// Whether the op can run on in0 and in1 with these modifiers and settings.
    bool (*accepts)(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& mods, const KernelSettings& settings);
    /// Negotiation: the unit, the register formats, the fidelity and the result spec.
    OpConfig (*options)(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& mods, const KernelSettings& settings);
    /// Allocation: loop counts and tile geometry.
    Work (*work)(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const OpConfig& config);
    /// Planning: what each unit does.
    Plan (*plan)(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& mods, const OpConfig& config, const Work& work);
};

/// The three results of resolving one call.
struct Resolved
{
    OpConfig config; ///< options' result
    Work work;       ///< work's result
    Plan plan;       ///< plan's result
};

/// Runs options, work and plan of api on the call's tensors, modifiers and settings. Does not check accepts.
template <class Mods>
constexpr Resolved resolve(
    const OpApi<Mods>& api, const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& mods, const KernelSettings& settings)
{
    const OpConfig config = api.options(in0, in1, mods, settings);
    const Work work       = api.work(in0, in1, config);
    return {config, work, api.plan(in0, in1, mods, config, work)};
}

/// Whether Api accepts A and B with Mods under the kernel's settings, as a compile-time object.
template <const auto& Api, const TensorSpec<tensor_rank>& A, const TensorSpec<tensor_rank>& B, auto Mods>
inline constexpr bool accepted = Api.accepts(A, B, Mods, kernel_settings);

/// The resolved call of Api on A and B with Mods under the kernel's settings, as a compile-time object.
template <const auto& Api, const TensorSpec<tensor_rank>& A, const TensorSpec<tensor_rank>& B, auto Mods>
inline constexpr Resolved resolved = resolve(Api, A, B, Mods, kernel_settings);

} // namespace llk2
