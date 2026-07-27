// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Control-flow combinators.
//
// A plain C++ `for` loop cannot express "configure on iteration 0, skip after"
// with a single compile-time state: the state variable is loop-invariant, so
// every iteration sees the same `Tag` and would either all-configure or
// all-skip. The `loop` combinator solves this by threading two states:
//
//   * iteration 0 runs at the ENTRY state (which does not yet match the op, so
//     the reconfigure IS emitted, exactly once), and
//   * iterations 1..N-1 run at the loop's FIXED POINT (already-configured, so
//     the reconfigure is elided via `if constexpr`).
//
// The fixed point is `body(entry)` — the state after one pass. For this
// prototype's scope (constant tile shapes, the dst_untilize case) the body is
// idempotent: body(fixed_point) == fixed_point, which we assert. A general
// version would iterate the merge to convergence (widening to unknown when the
// body changes state every iteration, which correctly forces a reconfigure per
// iteration — the right answer for a shape-changing loop).
//
// `branch` shows the join side: divergent arms phi-merge, widening any field
// the arms disagree on so the next op reconfigures.

#ifndef SST_CONTROL_H
#define SST_CONTROL_H

#include <cstddef>
#include <type_traits>
#include <utility>

#include "defs.h"
#include "state.h"

namespace sst {

// ---------------------------------------------------------------------------
// loop
// ---------------------------------------------------------------------------

// Fixed point = the state after running the body once from `Entry`.
// `value` is a static constexpr data member (has linkage) so it can anchor a
// `Tag<...>` NTTP.
template <const State& Entry, typename Body>
struct LoopFixedPoint {
    using Result = std::decay_t<decltype(std::declval<Body&>()(Tag<Entry>{}, std::size_t{}))>;
    static constexpr State value = Result::state;
};

template <const State& Entry, typename Body>
ALWI auto loop(Tag<Entry>, std::size_t n, Body&& body) {
    using FP = LoopFixedPoint<Entry, std::decay_t<Body>>;

    // Guard for this prototype's scope: the body must reach its fixed point
    // after a single pass, i.e. running it again at the fixed point does not
    // move the state. This holds for constant-shape loops like dst_untilize.
    using ReRun = std::decay_t<decltype(std::declval<Body&>()(Tag<FP::value>{}, std::size_t{}))>;
    static_assert(
        ReRun::state == FP::value,
        "loop body must be idempotent at its fixed point (constant-shape scope). "
        "A shape-changing loop needs the widening fixed-point variant.");

    if (n == 0) {
        return Tag<FP::value>{};
    }

    // Iteration 0 at ENTRY: this instantiation still contains the reconfigure
    // code, so the (single) configure is emitted here.
    body(Tag<Entry>{}, std::size_t{0});

    // Iterations 1..N-1 at the FIXED POINT: this is a DIFFERENT instantiation
    // of the body in which every reconfigure guard is compile-time false, so
    // no configure code exists in it at all.
    for (std::size_t i = 1; i < n; ++i) {
        body(Tag<FP::value>{}, i);
    }

    return Tag<FP::value>{};
}

// ---------------------------------------------------------------------------
// branch: pick the first arm whose runtime condition is true; phi-merge the
// compile-time state across ALL arms so the result is conservatively correct
// regardless of which arm runs.
// ---------------------------------------------------------------------------

template <typename Cond, typename Fn>
struct When {
    Cond cond;
    Fn fn;
};
template <typename Fn>
struct Otherwise {
    Fn fn;
};

template <typename Cond, typename Fn>
constexpr When<std::decay_t<Cond>, std::decay_t<Fn>> when(Cond&& c, Fn&& f) {
    return {std::forward<Cond>(c), std::forward<Fn>(f)};
}
template <typename Fn>
constexpr Otherwise<std::decay_t<Fn>> otherwise(Fn&& f) {
    return {std::forward<Fn>(f)};
}

namespace detail {

template <bool Cond, typename Cb>
constexpr bool eval(const Cb& cb) {
    if constexpr (std::is_invocable_v<const Cb&>) {
        return cb();
    } else {
        return static_cast<bool>(cb);
    }
}

// Runtime dispatch: run the first arm whose condition holds; the default arm
// always runs if reached.
template <const State& S, typename Fn>
ALWI void run(bool& done, const Otherwise<Fn>& br) {
    if (!done) {
        br.fn(Tag<S>{});
        done = true;
    }
}
template <const State& S, typename Cond, typename Fn>
ALWI void run(bool& done, const When<Cond, Fn>& br) {
    if (!done && detail::eval<false>(br.cond)) {
        br.fn(Tag<S>{});
        done = true;
    }
}

}  // namespace detail

// phi of the output states of a When arm.
template <const State& S, typename Cond, typename Fn>
constexpr auto arm_out(const When<Cond, Fn>&) {
    return std::decay_t<decltype(std::declval<Fn&>()(Tag<S>{}))>{};
}
template <const State& S, typename Fn>
constexpr auto arm_out(const Otherwise<Fn>&) {
    return std::decay_t<decltype(std::declval<Fn&>()(Tag<S>{}))>{};
}

template <const State& S, typename... Arms>
ALWI auto branch(Tag<S>, Arms&&... arms) {
    bool done = false;
    (detail::run<S>(done, arms), ...);
    // Merge every arm's resulting state.
    return phi(arm_out<S>(arms)...);
}

// ---------------------------------------------------------------------------
// match: the value-dispatched sibling of `branch` (a `switch`). Each `when(v,
// fn)` arm runs iff its key `v` equals the dispatch `value`; `otherwise` is the
// default. Same phi-join as `branch`: the result state is the merge of ALL arms
// (widening any field the arms disagree on), so an op after the match is forced
// to reconfigure exactly when the arms could have left the hardware in different
// states — regardless of which arm ran at runtime. `branch(when(x==k, ...))`
// expresses the same thing; `match` is the ergonomic form when dispatching one
// value against several keys.
// ---------------------------------------------------------------------------
namespace detail {

template <const State& S, typename V, typename Fn>
ALWI void run_match(bool& done, const V&, const Otherwise<Fn>& br) {
    if (!done) {
        br.fn(Tag<S>{});
        done = true;
    }
}
template <const State& S, typename V, typename Key, typename Fn>
ALWI void run_match(bool& done, const V& value, const When<Key, Fn>& br) {
    if (!done && (br.cond == value)) {
        br.fn(Tag<S>{});
        done = true;
    }
}

}  // namespace detail

template <const State& S, typename V, typename... Arms>
ALWI auto match(Tag<S>, const V& value, Arms&&... arms) {
    bool done = false;
    (detail::run_match<S>(done, value, arms), ...);
    // Merge every arm's resulting state (same phi-join as branch).
    return phi(arm_out<S>(arms)...);
}

}  // namespace sst

#endif  // SST_CONTROL_H
