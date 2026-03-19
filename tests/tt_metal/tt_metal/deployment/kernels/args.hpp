#ifndef _ARGS_H
#define _ARGS_H

#define ARGENUM(x) A_##x,
#define ARGDECL(x) constexpr uint32_t x = get_compile_time_arg_val(A_##x);

#define ARG_INIT(ARGS)      \
    enum { ARGS(ARGENUM) }; \
    ARGS(ARGDECL)           \
    do {                    \
    } while (0)

#endif /* _ARGS_H */
