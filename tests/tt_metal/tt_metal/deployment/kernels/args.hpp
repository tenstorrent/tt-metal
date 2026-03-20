#ifndef _ARGS_H
#define _ARGS_H

#define ARGENUM(t, x) A_##x,
#define ARGDECL(t, x) constexpr t x = get_compile_time_arg_val(A_##x);

#define ARG_INIT(ARGS)      \
    enum { ARGS(ARGENUM) }; \
    ARGS(ARGDECL)           \
    do {                    \
    } while (0)

#endif /* _ARGS_H */
