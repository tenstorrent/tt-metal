// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>

// Support types for the optional-binding lookups that kernel_bindings_generated.h declares:
//
//   template <::binding::Name NAME> constexpr auto try_get_tensor_binding();
//   template <::binding::Name NAME> constexpr std::optional<DFBBindingToken> try_get_dfb_binding();
//   template <::binding::Name NAME> constexpr std::optional<ScratchpadBindingToken> try_get_scratchpad_binding();
//
// THE PROBLEM THESE SOLVE
//
// A Metal 2.0 kernel names a bound resource through a codegen-emitted namespace-scope token:
// `tensor::in`, `dfb::gamma`, `scratch::tmp`. A token is emitted only if the host actually bound
// that resource to that kernel, so a kernel that merely *mentions* a resource the host did not
// bind fails to compile -- the name does not exist. That makes a genuinely optional parameter
// inexpressible: `if constexpr (have_gamma) { DataflowBuffer g(dfb::gamma); }` does not help,
// because `dfb::gamma` must still resolve for the discarded branch to be parsed. Today's
// workaround is a host-defined `-DFUSE_GAMMA` and an `#ifdef` around the use, so the text is
// removed before the compiler sees it. That spreads a single "is gamma present?" decision across
// the program factory, the define, and the kernel.
//
// HOW THE LOOKUPS SOLVE IT
//
// A lookup takes the resource name as a compile-time template argument rather than as an
// identifier, so the name is a *value* being compared, not a symbol being resolved. Every kernel
// gets all three lookups whether or not it has bindings of that kind, so a name that was not bound
// is not an error -- it simply returns an empty optional. Emptiness is a compile-time constant, so
// the absent branch is provably dead and the compiler deletes it; the generated code is identical
// to the `#ifdef` version.
//
//   constexpr auto gamma = try_get_dfb_binding<"gamma">();
//   if (gamma.has_value()) {
//       DataflowBuffer g(*gamma);
//       ...
//   }
//
// THE TRADE-OFF, STATED PLAINLY
//
// Because an unbound name is legal, a *misspelled* name is legal too: it compiles and reports
// absent forever. A kernel that requires a binding should therefore assert on it, which turns a
// typo back into a compile error:
//
//   static_assert(try_get_tensor_binding<"in">().has_value(), "kernel requires tensor binding 'in'");

namespace binding {

// Maximum binding-name length a lookup can distinguish. The cap exists because `Name` must be a
// structural type with a fixed-size member (see below); 63 is far beyond any plausible accessor
// name.
//
// This limit is ENFORCED ON THE HOST, not here: ProgramSpec validation rejects any tensor, DFB or
// scratchpad accessor_name longer than this (see ValidateBindingNameLength in program_spec.cpp,
// which includes this header so the two cannot drift). Without that check a longer name would be
// silently truncated by the constructor below, and two names agreeing on their first
// MAX_BINDING_NAME_LEN characters would become indistinguishable — a lookup would quietly hand
// back the wrong binding. The host check turns that into a loud error naming the kernel.
inline constexpr size_t MAX_BINDING_NAME_LEN = 63;

// A binding name usable as a non-type template parameter.
//
// This is deliberately a concrete, fixed-capacity type rather than the more idiomatic
// `template <size_t N> struct StringLiteral { char value[N]; }` with a deduction guide: device
// kernels build with `-std=c++17 -ftt-nttp`, and while that TT extension permits class-type NTTPs,
// it does not permit NTTPs of *deduced* class type (a C++20 feature). A single non-template type
// keeps `try_get_tensor_binding<"in">()` spelled the same way while staying inside C++17.
//
// All members are public and it has no user-declared copy/move/destructor, so it is a structural
// type and therefore valid as an NTTP.
struct Name {
    // Always NUL-padded to full capacity: `value` is zero-initialized and the constructor only ever
    // writes up to MAX_BINDING_NAME_LEN characters. That makes the fixed-length comparison in
    // operator== well-defined, and it makes two Names with the same characters compare equal
    // regardless of how they were spelled.
    char value[MAX_BINDING_NAME_LEN + 1] = {};

    // Implicit by design, so a call site can write try_get_dfb_binding<"gamma">() rather than
    // try_get_dfb_binding<::binding::Name("gamma")>().
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    constexpr Name(const char* str) {
        for (size_t i = 0; i < MAX_BINDING_NAME_LEN && str[i] != '\0'; ++i) {
            value[i] = str[i];
        }
    }

    // Compares the whole fixed-capacity buffer rather than stopping at the first NUL. Both operands
    // are NUL-padded (see above), so this is equivalent to a string comparison, and it avoids
    // needing a constexpr strcmp in a header that device kernels include.
    constexpr bool operator==(const Name& other) const {
        for (size_t i = 0; i <= MAX_BINDING_NAME_LEN; ++i) {
            if (value[i] != other.value[i]) {
                return false;
            }
        }
        return true;
    }
    constexpr bool operator!=(const Name& other) const { return !(*this == other); }
};

// The value type of an *absent* try_get_tensor_binding result.
//
// The DFB and scratchpad lookups need no equivalent: their tokens (DFBBindingToken,
// ScratchpadBindingToken) are concrete types, so present and absent can both be
// std::optional<Token>. A tensor binding token is TensorBindingToken<CTA_OFFSET,
// ADDR_CRTA_OFFSET> -- a distinct type per binding -- so there is no such token type to be empty
// of when the binding does not exist. This stand-in supplies one.
//
// LocalTensorAccessor has a constructor taking this type. That is not optional politeness: a
// discarded `if constexpr` branch is only left uninstantiated inside a template, and kernel_main
// is not a template, so the absent branch is still fully type-checked. Were the accessor to reject
// this token, a kernel could not compile the very branch these lookups exist to let it write.
struct NullTensorBindingToken {};

}  // namespace binding
