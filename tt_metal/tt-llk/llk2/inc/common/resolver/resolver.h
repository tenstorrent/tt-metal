// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The Resolver's driver (tt-llk-lab llk-tensor, report 3, section 4:
// https://github.com/tenstorrent/tt-llk-lab/tree/lpremovic/llk-tensor/experiments/llk-tensor), for a chain of one:
// each Compute API call resolves only itself. An op provides four pure functions in an OpApi:
//   accepts(in0, in1, mods, settings)       can the op run on these tensors at all       -> Accepts
//   options(in0, in1, mods, settings)       negotiation: the representation it runs in      -> OpConfig
//                                           (called options because in a chain of several ops each offers several
//                                           and the driver picks; a chain of one has one, the OpConfig)
//   work(in0, in1, config)                  allocation: loop counts                         -> Work
//   plan(in0, in1, mods, config, work)      planning: what each unit does                   -> Plan
// The driver runs them in that order, at compile time. accepted_parts<...> (and accepted<...>, all of them) and
// resolved<...> name the results, so they are compile-time objects; the Compute API static_asserts each part of
// accepted_parts<...> before it reads resolved<...>, so an op that cannot run on its tensors fails the build at the
// call, with the sentence of the part that refused it.
//
// OpApi is a class template over what the op's Compute API call hands the Resolver, so every op's functions take
// their own type. For most ops that is the op's modifier struct. The SFPU activations share one OpApi (vec_op) and
// one modifier struct (vec_modifiers), so their calls hand over a vec_call (resolver/modifiers.h): the
// activation plus the modifiers. The functions themselves are plain constexpr functions reached through pointers:
// they cannot be templates, and do not need to be, because everything they read arrives as a constexpr argument.
//
// The form of an op's file (resolver/<op>.h, per arch), the same for every op:
//   header comment   "<Op>'s OpApi on <arch>" and where its operands go; the four steps, one line each (accepts /
//                    options / work / plan); the op's own sections (its unpack walk, its formats); a last paragraph
//                    "Not ported: ..."
//   helpers          constexpr functions named <op>_<what>
//   <op>_accepts     made of named parts, those that apply, in this order: storage, tiles, shapes, layout, formats,
//                    modifiers, dest_mode (what the op needs of the kernel's Dest mode), runtime (which sizes may be
//                    run-time counts: resolver/work_counts.h), returned by name in an Accepts. A part that rejects
//                    what the hardware can do says "not ported"
//   <op>_options, <op>_work, <op>_plan
//                    each builds its result (OpConfig, Work, the Plan's parts) field by field, by name
//   <op>_op          inline constexpr OpApi<<op>_modifiers> <op>_op {&<op>_accepts, &<op>_options, &<op>_work, &<op>_plan};
// The tensors are named as the Compute API names the op's operands, the rest as OpApi names them: mods, settings,
// config, work. in0 and in1 are the op's two inputs. An op with one tensor gets it twice and does not read the second
// (copy, the SFPU activations); pack's in1 is the tensor it writes, because that tensor's format is the caller's
// choice and decides the Plan.

#pragma once

#include "kernel_settings.h"
#include "resolver/plan.h"

namespace llk2
{

/// What accepts returns: whether the op can run, as the named parts of its rule (the form of an op's file, above).
/// A part an op does not have stays true. The Compute API asserts each part an op has with that part's sentence, so a
/// refused call fails the build with the one rule it broke.
struct Accepts
{
    bool storage   = true; ///< where the tensors are (L1, Dest)
    bool tiles     = true; ///< the tile size
    bool shapes    = true; ///< the tensors' tile grids, against each other
    bool layout    = true; ///< how the tiles sit in L1 or Dest: dense, a window of a wider tile grid
    bool formats   = true; ///< the data formats
    bool modifiers = true; ///< the modifiers, against the op and the formats
    bool dest_mode = true; ///< what the op needs of the kernel's Dest mode
    bool runtime   = true; ///< which sizes may be run-time counts

    /// Whether the op can run: every part holds.
    constexpr bool all() const
    {
        return storage && tiles && shapes && layout && formats && modifiers && dest_mode && runtime;
    }
};

/// An op's four Resolver steps. Mods is what the op's Compute API call hands over: its modifier struct (vec_call
/// for the SFPU activations).
template <class Mods>
struct OpApi
{
    /// Whether the op can run on in0 and in1 with these modifiers and settings, part by part.
    Accepts (*accepts)(const Spec& in0, const Spec& in1, const Mods& mods, const KernelSettings& settings);
    /// Negotiation: the register formats, the fidelity and the result spec.
    OpConfig (*options)(const Spec& in0, const Spec& in1, const Mods& mods, const KernelSettings& settings);
    /// Allocation: loop counts and tile geometry.
    Work (*work)(const Spec& in0, const Spec& in1, const OpConfig& config);
    /// Planning: what each unit does.
    Plan (*plan)(const Spec& in0, const Spec& in1, const Mods& mods, const OpConfig& config, const Work& work);
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
constexpr Resolved resolve(const OpApi<Mods>& api, const Spec& in0, const Spec& in1, const Mods& mods, const KernelSettings& settings)
{
    const OpConfig config = api.options(in0, in1, mods, settings);
    const Work work       = api.work(in0, in1, config);
    return {config, work, api.plan(in0, in1, mods, config, work)};
}

/// What Api's accepts says of A and B with Mods under the kernel's settings, part by part, as a compile-time object.
template <const auto& Api, const Spec& A, const Spec& B, auto Mods>
inline constexpr Accepts accepted_parts = Api.accepts(A, B, Mods, kernel_settings);

/// Whether Api accepts A and B with Mods under the kernel's settings: every part holds.
template <const auto& Api, const Spec& A, const Spec& B, auto Mods>
inline constexpr bool accepted = accepted_parts<Api, A, B, Mods>.all();

/// The resolved call of Api on A and B with Mods under the kernel's settings, as a compile-time object.
template <const auto& Api, const Spec& A, const Spec& B, auto Mods>
inline constexpr Resolved resolved = resolve(Api, A, B, Mods, kernel_settings);

} // namespace llk2
