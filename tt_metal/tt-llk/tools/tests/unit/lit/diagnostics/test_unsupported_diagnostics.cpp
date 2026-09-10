// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: fmt
// DEFINE: %{cflags} = -std=c++17 -O2 -c -o /dev/null -DLLK_SAN_ENABLE -DCOMPILE_FOR_TRISC=0 -DLLK_SAN_SETTING_HOST_DEPS=1 -DDEBUG_PRINT_ENABLED %{fmt_flags}
// DEFINE: %{verify} = -Xclang -verify -Xclang -verify-ignore-unexpected=note
// DEFINE: %{check} = %clangxx %{cflags} %{verify} -I %{sanitizer_include}
// RUN: split-file %s %t
//
// RUN: %{check} %t/unreached.cpp
// RUN: %{check} %t/template_instantiated.cpp
// RUN: %{check} %t/called.cpp

//--- unreached.cpp
#include "sanitizer/api.h"

template <int N>
inline void llk_unmodelled_templated_op()
{
    llk::san::unsupported();
}

inline void llk_unmodelled_op()
{
    llk::san::unsupported();
}

// expected-no-diagnostics

int main()
{
    return 0;
}

//--- template_instantiated.cpp
#include "sanitizer/api.h"

template <int N>
inline void llk_unmodelled_templated_op()
{
    llk::san::unsupported();
}

// expected-error@*:* {{not modelled by the sanitizer}}

int main()
{
    llk_unmodelled_templated_op<0>();
    return 0;
}

//--- called.cpp
#include "sanitizer/api.h"

inline void llk_unmodelled_op()
{
    llk::san::unsupported();
}

// expected-error@*:* {{not modelled by the sanitizer}}

int main()
{
    llk_unmodelled_op();
    return 0;
}
