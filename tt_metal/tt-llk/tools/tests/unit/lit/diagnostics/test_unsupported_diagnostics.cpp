// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time diagnostics tests for llk::san::unsupported(), the marker for an LLK operation the
// sanitizer does not model. The rule under test is when it fires, which is the whole of its design:
//
//   reached in a sanitized build    error, so the kernel cannot silently run unchecked
//   never reached                   silent, so marking one API function does not cost every kernel
//                                   that merely includes the header
//   sanitizer off                   silent, because there is no model to be missing from
//
// "Reached" means the enclosing function is instantiated (template) or called (non-template). Both
// shapes are covered, because both exist in llk_api and a static_assert would only have deferred for
// the first: in a non-template function a static_assert fires when the header is parsed, which is
// every kernel, called or not. That is why the marker is an attribute on a declared-but-never-defined
// function rather than a static_assert, and it is what these cases pin down.
//
// SAN_HOOK expands its argument in both builds, so unlike the other entry points the sanitizer-off
// case is carried entirely by the no-op unsupported() in the disabled namespace, not by the macro.
//
// See test_state_diagnostics.cpp for how -verify, split-file and @*:* work.
//
// Unlike the rest of this suite these cases compile for real, with -c, instead of -fsyntax-only. The
// attribute is a codegen-time diagnostic -- it fires when a call survives to code generation, which
// is exactly what makes it silent for the functions a kernel never reaches -- so -fsyntax-only would
// see nothing and every case would pass while testing nothing. -verify still collects it. The
// expectations name no line because the diagnostic points into api.h, not into the case.
//
// -O2 rather than -O0 so that the diagnostic is reached through the inliner, which is how a kernel
// build sees it. The unreached cases are the ones that would break if that ever stopped being true.

// api.h reaches output.h, so the sanitized host builds need deps/host.h and libfmt. The sanitizer-off
// build takes neither: settings.h rejects LLK_SAN_SETTING_HOST_DEPS without LLK_SAN_ENABLE outright,
// and the disabled branch of api.h never reaches output.h. Hence the split between %{base} and %{on}.
// REQUIRES: fmt
// DEFINE: %{base} = -std=c++17 -O2 -c -o /dev/null -DCOMPILE_FOR_TRISC=0 -DDEBUG_PRINT_ENABLED
// DEFINE: %{on} = %{base} -DLLK_SAN_ENABLE -DLLK_SAN_SETTING_HOST_DEPS=1 %{fmt_flags}
// DEFINE: %{verify} = -Xclang -verify -Xclang -verify-ignore-unexpected=note
// DEFINE: %{check} = %clangxx %{on} %{verify} -I %{sanitizer_include}
// DEFINE: %{check_off} = %clangxx %{base} %{verify} -I %{sanitizer_include}
// RUN: split-file %s %t
//
// RUN: %{check} %t/unreached.cpp
// RUN: %{check} %t/instantiated_template.cpp
// RUN: %{check} %t/called_non_template.cpp
// RUN: %{check_off} %t/sanitizer_off.cpp

//--- unreached.cpp
// The counterweight to the two positive cases: a marked template that is never instantiated and a
// marked non-template that is never called cost nothing. Without this case a marker that fired on
// every include would pass both of them.
#include <cstdint>

#include "sanitizer/api.h"

template <int N>
inline void llk_unmodelled_templated_op()
{
    SAN_HOOK(unsupported());
}

inline void llk_unmodelled_op()
{
    SAN_HOOK(unsupported());
}

// expected-no-diagnostics

int main()
{
    return 0;
}

//--- instantiated_template.cpp
// Reached by instantiating the enclosing template. This is the case a static_assert would also have
// caught.
#include <cstdint>

#include "sanitizer/api.h"

template <int N>
inline void llk_unmodelled_templated_op()
{
    SAN_HOOK(unsupported());
}

// expected-error@*:* {{not modelled by the sanitizer}}

int main()
{
    llk_unmodelled_templated_op<0>();
    return 0;
}

//--- called_non_template.cpp
// Reached by calling a non-template enclosing function. This is the case a static_assert could not
// express: it would have fired on unreached.cpp too.
#include <cstdint>

#include "sanitizer/api.h"

inline void llk_unmodelled_op()
{
    SAN_HOOK(unsupported());
}

// expected-error@*:* {{not modelled by the sanitizer}}

int main()
{
    llk_unmodelled_op();
    return 0;
}

//--- sanitizer_off.cpp
// Sanitizer off. Nothing is being checked, so nothing can be missing from the check, and a kernel
// that uses an unmodelled operation builds as it always did. SAN_HOOK still expands the call here,
// so this is the no-op unsupported() in the disabled namespace doing the work.
#include <cstdint>

#include "sanitizer/api.h"

inline void llk_unmodelled_op()
{
    SAN_HOOK(unsupported());
}

// expected-no-diagnostics

int main()
{
    llk_unmodelled_op();
    return 0;
}
