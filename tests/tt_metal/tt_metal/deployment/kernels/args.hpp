// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _ARGS_H
#define _ARGS_H

#define ARGENUM(t, x) A_##x,
#define ARGDECL(t, x) constexpr t x = get_compile_time_arg_val(A_##x);
#define ARGRUNTIMEDECL(t, x) t x = get_arg_val<t>(A_##x);

#define ARG_INIT(ARGS)      \
    enum { ARGS(ARGENUM) }; \
    ARGS(ARGDECL)           \
    do {                    \
    } while (0)

#define ARG_RUNTIME_INIT(ARGS) \
    enum { ARGS(ARGENUM) };    \
    ARGS(ARGRUNTIMEDECL)       \
    do {                       \
    } while (0)

#endif /* _ARGS_H */
