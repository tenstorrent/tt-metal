// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#ifdef ARCH_BLACKHOLE

// Compile-time marks a kernel's source sets as it is parsed: completing set<N> makes used(tag<N>) callable, so
// is_used<N>(0) reads true from that point on. tag 0: a plain matmul call runs the ttsync unpack, read by the TRISC
// wrapper after the kernel's include; tag 3: the kernel names a fast tilize init; tag 4: a plain matmul call polls; tags 7 and 8:
// a plain matmul call whose ct and rt are (8) or are not (7) constant expressions.
namespace ckernel::matmul_fe_mark {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
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
// every kernel build checks the pattern on spare tags, so a compiler that stops honouring it fails here, not silently
inline constexpr auto self_test_set = sizeof(set<1>);
static_assert(is_used<1>(0), "matmul_fe_mark: a completed set<N> must make used(tag<N>) callable");
static_assert(!is_used<2>(0), "matmul_fe_mark: an untouched tag<N> must read unused");
}  // namespace ckernel::matmul_fe_mark

// Auto TTSync slows fast tilize: a kernel that names a fast tilize init before its first plain matmul call keeps the
// polling unpack. The first plain call decides and every later one follows it, so the kernel's calls and the TRISC
// wrapper agree whatever the order.
#define MATMUL_FE_TTSYNC_OK() (::ckernel::matmul_fe_mark::is_used<0>(0) || !::ckernel::matmul_fe_mark::is_used<3>(0))
// 1: the plain call may run the ttsync unpack; 2: it polls
#define MATMUL_FE_MARK() \
    (sizeof(::ckernel::matmul_fe_mark::set<MATMUL_FE_TTSYNC_OK() ? 0 : 4>) + (MATMUL_FE_TTSYNC_OK() ? 0 : 1))
#define FAST_TILIZE_MARK() sizeof(::ckernel::matmul_fe_mark::set<3>)
// 0, completing tag 8 for a plain call whose ct and rt are constant expressions and tag 7 otherwise; the dims pass by value, so
// the test takes no address of them (a reference moved the register allocation of run-time-dim kernels), and through matmul_fe_cb,
// so a run-time dim the optimizer folds still reads as run time
#define MATMUL_FE_SHAPE(ct, rt)                                                                 \
    (0 * sizeof(::ckernel::matmul_fe_mark::set<                                                 \
                (__builtin_constant_p(::ckernel::matmul_fe_cb(::ckernel::matmul_fe_dim(ct))) && \
                 __builtin_constant_p(::ckernel::matmul_fe_cb(::ckernel::matmul_fe_dim(rt))))   \
                    ? 8                                                                         \
                    : 7>))
#endif
