// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#ifdef ARCH_BLACKHOLE

// Compile-time marks a kernel's source sets as it is parsed: completing set<N> makes used(tag<N>) callable, so
// is_used<N>(0) reads true from that point on. tag 0: a plain matmul call runs the ttsync unpack, read by the TRISC
// wrapper after the kernel's include; tag 3: the kernel names a fast tilize init; tag 4: a plain matmul call polls; tag
// 5: a plain call of constant block keys runs the ttsync unpack on its cell in a kernel the class rule keeps polling;
// tags 7 and 8: a plain matmul call whose ct and rt are (8) or are not (7) constant expressions; tag 13: a call or init
// of a constant one-tile block (14 unread); tag 15: a math init or call of constant kt 1 (16 unread); tag 17: an unpack
// call of a constant one-tile block of constant kt 1, 18 any other unpack call; tag 19: a copy_block call.
namespace ckernel::matmul_fe_mark {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
// tag 21: a matmul_block call in a LoFi fp32 half-sync bmm cell that keeps main's section end (22 unread).
// The next free tag is 23.
template <int>
struct tag {
    friend auto used(tag);
};
#pragma GCC diagnostic pop
template <int N>
struct set {
    friend auto used(tag<N>) { return true; }
};
template <int N = 0, typename = decltype(used(tag<N>{}))>
constexpr bool is_used(int) {
    return true;
}
template <int N = 0>
constexpr bool is_used(...) {
    return false;
}
// is_used for a template body: the probe depends on D, so it reads the marks of the whole kernel (at the end of the unit)
template <int N, typename D, typename = decltype(used(tag<N + 0 * sizeof(D)>{}))>
constexpr bool is_used_d(int) {
    return true;
}
template <int N, typename D>
constexpr bool is_used_d(...) {
    return false;
}
constexpr unsigned dim(const unsigned x) {
    return x;
}
// every kernel build checks the pattern on spare tags, so a compiler that stops honouring it fails here, not silently
inline constexpr auto self_test_set = sizeof(set<1>);
static_assert(is_used<1>(0), "matmul_fe_mark: a completed set<N> must make used(tag<N>) callable");
static_assert(!is_used<2>(0), "matmul_fe_mark: an untouched tag<N> must read unused");
// The forms key of the plain calls of run-time ct and rt, one per kernel: tag 11 keyed, 10 not, 12 unread. The reads and the set
// take the call's cb argument type D and its line, so a call in a template whose cb argument is dependent reads at instantiation,
// after the init that a kernel_main names before it.
template <int N, typename D>
struct set_d : set<N> {};
template <int N, typename D, int L, typename = decltype(used(tag<N>{}))>
constexpr bool is_used_dl(int) {
    return true;
}
template <int N, typename D, int L>
constexpr bool is_used_dl(...) {
    return false;
}
}  // namespace ckernel::matmul_fe_mark

// Auto TTSync slows fast tilize: a kernel that names a fast tilize init before its first plain matmul call keeps the
// polling unpack. The first plain call decides and every later one follows it, so the kernel's calls and the TRISC
// wrapper agree whatever the order.
#define MATMUL_FE_TTSYNC_OK()                                                                         \
    (::ckernel::matmul_fe_mark::is_used<0>(0) || ::ckernel::matmul_fe_mark::is_used<5>(0) || \
     !::ckernel::matmul_fe_mark::is_used<3>(0))
// 1: the plain call may run the ttsync unpack; 2: it polls
#define MATMUL_FE_MARK() \
    (sizeof(::ckernel::matmul_fe_mark::set<MATMUL_FE_TTSYNC_OK() ? 0 : 4>) + (MATMUL_FE_TTSYNC_OK() ? 0 : 1))
// 3: a call of constant cbs, ct, rt and kt whose cell keeps the ttsync unpack (matmul_fe_ttsync_cell) where the class
// rule polls; 4: a call of constant cbs and run-time ct or rt that runs it on a one-tile block
// (matmul_fe_rt_ttsync_kt); otherwise MATMUL_FE_MARK(). The keys are direct template arguments of matmul_fe_cell_mark:
// nested in a constant-evaluated call, a dim that folds to one value without being a constant expression (a tail equal
// to the full width) does not compile.
#define MATMUL_FE_CELL_MARK(a, b, ct, rt, kt)   \
    (::ckernel::matmul_fe_cell_mark<            \
             MATMUL_FE_C(a),                    \
             MATMUL_FE_C(b),                    \
             MATMUL_FE_CK(ct),                  \
             MATMUL_FE_CK(rt),                  \
             MATMUL_FE_CK(kt),                  \
             MATMUL_FE_TTSYNC_OK()>::value >= 3 \
         ? ::ckernel::matmul_fe_cell_mark<      \
               MATMUL_FE_C(a),                  \
               MATMUL_FE_C(b),                  \
               MATMUL_FE_CK(ct),                \
               MATMUL_FE_CK(rt),                \
               MATMUL_FE_CK(kt),                \
               MATMUL_FE_TTSYNC_OK()>::value    \
         : MATMUL_FE_MARK())
// a block key: the constant itself, 0xFF where the argument is not a constant expression; taken by value, since a reference to a
// run-time dim moved its register allocation (minimal_matmul)
#define MATMUL_FE_CK(x) (__builtin_constant_p(::ckernel::matmul_fe_cb(::ckernel::matmul_fe_mark::dim(x))) ? ::ckernel::matmul_fe_mark::dim(x) : 0xFFu)
// 0, completing tag 15 for a math init or call whose kt is the constant 1 (16 unread)
#define MATMUL_FE_KT1(kt) (0 * sizeof(::ckernel::matmul_fe_mark::set < (MATMUL_FE_CK(kt) == 1u) ? 15 : 16 >))
// 0, completing tag 17 for an unpack call whose ct, rt and kt are the constant 1, 18 for any other
#if defined(UCK_CHLKC_UNPACK)
#define MATMUL_FE_ONE_TILE_KT1(ct, rt, kt)                                                                             \
    (0 *                                                                                                               \
     sizeof(                                                                                                           \
         ::ckernel::matmul_fe_mark::set < (MATMUL_FE_CK(ct) == 1u && MATMUL_FE_CK(rt) == 1u && MATMUL_FE_CK(kt) == 1u) \
             ? 17                                                                                                      \
             : 18 >))
#else
#define MATMUL_FE_ONE_TILE_KT1(ct, rt, kt) 0
#endif
#define FAST_TILIZE_MARK() sizeof(::ckernel::matmul_fe_mark::set<3>)
// 0, completing tag 8 for a plain call whose ct and rt are constant expressions and tag 7 otherwise; the dims pass by value, so
// the test takes no address of them (a reference moved the register allocation of run-time-dim kernels), and through matmul_fe_cb,
// so a run-time dim the optimizer folds still reads as run time
#define MATMUL_FE_SHAPE(ct, rt)                                                                      \
    (0 * sizeof(                                                                                     \
             ::ckernel::matmul_fe_mark::set <                                                        \
                     (__builtin_constant_p(::ckernel::matmul_fe_cb(::ckernel::matmul_fe_dim(ct))) && \
                      __builtin_constant_p(::ckernel::matmul_fe_cb(::ckernel::matmul_fe_dim(rt))))   \
                 ? 8                                                                                 \
                 : 7 >) +                                                                            \
     0 * sizeof(::ckernel::matmul_fe_mark::set < (MATMUL_FE_CK(ct) == 1 && MATMUL_FE_CK(rt) == 1) ? 13 : 14 >))
// The forms key decision of a plain call (1 keyed, 0 not): the kernel's first call or init decides, keyed only where it is an init
// that can key (want: matmul_fe_keyable), and every later one follows it, so the inits and calls of a kernel run one form whatever
// the order the compiler reads them in. x: the call's in0 cb argument.
#define MATMUL_FE_KUNDECIDED(x)                                                   \
    (!::ckernel::matmul_fe_mark::is_used_dl<10, decltype(x), __LINE__>(0) && \
     !::ckernel::matmul_fe_mark::is_used_dl<11, decltype(x), __LINE__>(0))
#define MATMUL_FE_KDEC(x, want)                                                                                         \
    (sizeof(::ckernel::matmul_fe_mark::set_d<MATMUL_FE_KUNDECIDED(x) ? ((want) ? 11 : 10) : 12, decltype(x)>) * 0 + \
     (MATMUL_FE_KUNDECIDED(x) ? static_cast<bool>(want) : ::ckernel::matmul_fe_mark::is_used_dl<11, decltype(x), __LINE__>(0)))
#endif
