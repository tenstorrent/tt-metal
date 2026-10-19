// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Bounded waiting.
//
// Every firmware wait in the component goes through here so that "poll until
// it happens" cannot be written without a deadline, and so that a timeout
// always carries the last observation rather than just the elapsed time.
//
// Templated on the clock and the sleep so tests can drive the timeout logic
// with a fake clock instead of real time.

#pragma once

#include <algorithm>
#include <chrono>
#include <thread>

namespace x280 {

template <typename Clock>
class BasicDeadline {
public:
    using duration = std::chrono::milliseconds;

    explicit BasicDeadline(duration budget) : start_(Clock::now()), budget_(budget) {}

    bool expired() const { return elapsed() >= budget_; }
    duration budget() const { return budget_; }
    duration elapsed() const { return std::chrono::duration_cast<duration>(Clock::now() - start_); }

    duration remaining() const {
        const duration used = elapsed();
        return used >= budget_ ? duration::zero() : budget_ - used;
    }

private:
    typename Clock::time_point start_;
    duration budget_;
};

using Deadline = BasicDeadline<std::chrono::steady_clock>;

struct PollResult {
    bool satisfied = false;
    int polls = 0;
};

// Evaluates pred() at least once, then again every `interval` until it returns
// true or the deadline expires. One final evaluation happens after expiry, so a
// slow last iteration is not reported as a timeout.
//
// The wall-clock bound is the configured budget plus one pred() call. A caller
// whose predicate performs UMD I/O must bound that operation separately.
//
// Each sleep is clamped to what is left of the budget, so an interval coarser
// than the budget costs one extra poll rather than an unbounded overshoot. The
// clamp is only reached before expiry, so the sleep is never zero.
template <typename Clock, typename Pred, typename Sleep>
PollResult poll_until(
    const BasicDeadline<Clock>& deadline, std::chrono::milliseconds interval, Pred pred, Sleep sleep_fn) {
    PollResult result;
    while (true) {
        result.polls++;
        if (pred()) {
            result.satisfied = true;
            return result;
        }
        if (deadline.expired()) {
            return result;
        }
        sleep_fn(std::min(interval, deadline.remaining()));
    }
}

template <typename Pred>
PollResult poll_until(const Deadline& deadline, std::chrono::milliseconds interval, Pred pred) {
    return poll_until(deadline, interval, pred, [](std::chrono::milliseconds d) { std::this_thread::sleep_for(d); });
}

}  // namespace x280
