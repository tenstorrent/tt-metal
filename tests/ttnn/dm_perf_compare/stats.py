# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The comparison rule, as pure functions over per-round timings.

One method for every question: per round, compare two sides' best-of-N times; the call's change is
the median of the per-round changes; it counts as faster/slower only when it clears the noise
measured on the same machine and the rounds agree on the direction.
"""

import math
import statistics

# A noise estimate from a handful of control calls can come out near zero by luck. Launch-bound
# calls (tens of microseconds) moved 2-3% between runs of identical builds, so below this a
# "change" is not worth reporting whatever the control said.
MIN_NOISE_PCT = 3.0
# Above this the control pair itself was disturbed (the host was busy in some rounds), so a "no
# change" or "default is fastest" answer from that run proves nothing.
MAX_NOISE_PCT = 5.0
# Share of rounds that must point the same way as the median for the call to count.
AGREE_FRACTION = 0.8


def change_pct(before, after):
    """Percent change in time: negative is faster, positive is slower."""
    return (after / before - 1.0) * 100.0


def median(values):
    return statistics.median(values)


def per_round_changes(before, after):
    """Paired per-round changes; a round missing on either side is dropped, not zero-filled."""
    return [change_pct(b, a) for b, a in zip(before, after) if b is not None and a is not None and b > 0]


def noise_pct(control_changes):
    """Noise band from A-vs-A pairs: `control_changes` maps a control call to its per-round changes.

    The largest per-call median is used rather than the spread of rounds, because the verdict
    compares a per-call median against it.
    """
    medians = [abs(median(changes)) for changes in control_changes.values() if changes]
    return max([MIN_NOISE_PCT] + medians)


def rounds_agree(changes, direction):
    if not changes:
        return False
    same = sum(1 for c in changes if (c > 0) == (direction > 0) and c != 0)
    return same / len(changes) >= AGREE_FRACTION


def verdict(changes, noise):
    """`faster`, `slower` or `no change` for one call's per-round changes."""
    if not changes:
        return "no data"
    m = median(changes)
    if abs(m) <= noise or not rounds_agree(changes, m):
        return "no change"
    return "slower" if m > 0 else "faster"


def geomean_change_pct(changes):
    """Geometric mean of the per-call changes, as a percent change."""
    if not changes:
        return None
    logs = [math.log1p(c / 100.0) for c in changes]
    return (math.exp(sum(logs) / len(logs)) - 1.0) * 100.0
