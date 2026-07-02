// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _ARGS_H
#define _ARGS_H

#define ARGENUM(t, x) A_##x,
#define ARGDECL(t, x) constexpr t x = get_compile_time_arg_val(A_##x);
#define ARGINITPARAM(t, x) p.x = get_arg_val<t>(A_##x);

#define ARG_INIT(ARGS)      \
    enum { ARGS(ARGENUM) }; \
    ARGS(ARGDECL)           \
    do {                    \
    } while (0)

/* NOTE: Assumes a struct called `p` into which to store the params */
#define ARG_INIT_PARAMS(ARGS) \
    enum { ARGS(ARGENUM) };   \
    ARGS(ARGINITPARAM)        \
    do {                      \
    } while (0)

#endif /* _ARGS_H */
