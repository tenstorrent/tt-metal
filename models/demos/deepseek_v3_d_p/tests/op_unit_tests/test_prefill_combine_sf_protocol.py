# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only model of the store-and-forward combine protocol.

Scope, stated plainly: this validates the *protocol* -- the level-indexed staging FIFOs, the credit
admission rule, the biased arrival counter and the per-level end-of-stream chain. It does not
validate the kernels; the two are independent transcriptions of the same design, so agreement here is
evidence about the design and nothing more.

It exists because a ring is the case the design is actually built for -- the level index is what makes
the buffer dependency graph acyclic, and on a line that acyclicity comes free from chip position
instead -- and no ring-capable hardware is reachable from the development box. Everything here mirrors
combine_sf.hpp and the scheduler in reader_combine.cpp; if either changes, this must change with it.
"""

import random

import pytest


def max_distance(extent, is_ring):
    return extent // 2 if is_ring else extent - 1


def num_levels(extent, is_ring):
    m = max_distance(extent, is_ring)
    return m - 1 if m >= 2 else 0


def max_distance_in_dir(pos, extent, is_ring, positive):
    if is_ring:
        return extent // 2
    return extent - 1 - pos if positive else pos


def out_live(pos, extent, is_ring, positive, level):
    return level + 1 <= max_distance_in_dir(pos, extent, is_ring, positive)


def upstream_pos(pos, extent, is_ring, positive):
    if is_ring:
        return (pos - 1) % extent if positive else (pos + 1) % extent
    if positive:
        return None if pos == 0 else pos - 1
    return None if pos + 1 >= extent else pos + 1


def in_live(pos, extent, is_ring, positive, level):
    up = upstream_pos(pos, extent, is_ring, positive)
    return up is not None and out_live(up, extent, is_ring, positive, level)


def downstream_pos(pos, extent, is_ring, positive):
    if is_ring:
        return (pos + 1) % extent if positive else (pos - 1) % extent
    if positive:
        return None if pos + 1 >= extent else pos + 1
    return None if pos == 0 else pos - 1


def ring_distance(a, b, extent, is_ring):
    direct = abs(a - b)
    return min(direct, extent - direct) if is_ring else direct


class Chip:
    """One chip's scheduler state. Mirrors the locals in reader_combine.cpp's relay path."""

    def __init__(self, pos, extent, is_ring, levels, slots):
        self.pos = pos
        self.levels = levels
        self.slots = slots
        # Inbound staging FIFOs, indexed [direction][level - 1]. Bounded: a slot is only reusable
        # once the upstream sender has been told it was freed.
        self.fifo = [[[] for _ in range(levels)] for _ in range(2)]
        self.arrived = [[0] * levels for _ in range(2)]
        self.closed = [[False] * levels for _ in range(2)]
        self.pool_rd = [[0] * levels for _ in range(2)]
        self.staged = [[0] * levels for _ in range(2)]
        self.credit = [[0] * levels for _ in range(2)]
        self.eos_out = [[False] * levels for _ in range(2)]
        self.inject = []  # tokens this chip originates
        self.untilizers_done = False
        self.transit_run = 0

    def has_room(self, d, level, head_of_line=False):
        # Level zero is the destination's own output page: preallocated, so it can never be full.
        # This is the base of the deadlock argument.
        if level == 0:
            return True
        if head_of_line:
            # The rejected design: one full level stalls the shared queue, so nothing behind it moves.
            for dd in range(2):
                for rr in range(1, self.levels + 1):
                    if self.staged[dd][rr - 1] - self.credit[dd][rr - 1] >= self.slots:
                        return False
        return self.staged[d][level - 1] - self.credit[d][level - 1] < self.slots

    def drained(self, d, r):
        return self.closed[d][r - 1] and self.pool_rd[d][r - 1] == self.arrived[d][r - 1]


class Fabric:
    """Carries pages and counter increments with a delay, so nothing here can accidentally rely on
    instantaneous delivery."""

    def __init__(self, delay):
        self.delay = delay
        self.pending = []

    def send(self, now, fn):
        self.pending.append((now + self.delay, fn))

    def deliver(self, now):
        ready = [f for (t, f) in self.pending if t <= now]
        self.pending = [(t, f) for (t, f) in self.pending if t > now]
        for f in ready:
            f()

    def idle(self):
        return not self.pending


def run_protocol(
    extent,
    is_ring,
    slots,
    tokens_per_chip,
    seed,
    fabric_delay=2,
    quantum=8,
    max_steps=4_000_000,
    head_of_line=False,
    eos_rule="per_level",
):
    """Drive every chip's scheduler until quiescent.

    `head_of_line` and `eos_rule` exist to be set wrong on purpose. They select the two designs that
    were rejected -- a sender that waits on a credit rather than trying another source, and an
    end-of-stream that waits on the upstream's close at the same level -- so the tests below can show
    this model actually detects them. A model that cannot fail proves nothing about the one that
    passes.
    """
    rng = random.Random(seed)
    levels = num_levels(extent, is_ring)
    assert levels > 0, "nothing to model when no relay level exists"

    chips = [Chip(p, extent, is_ring, levels, slots) for p in range(extent)]
    fabric = Fabric(fabric_delay)
    delivered = []
    tie_toggle = [0]

    def route_dir(src, dst):
        """Shorter way round; ties alternate, mirroring the stateful tie-break in get_route."""
        fwd = (dst - src) % extent
        rev = (src - dst) % extent
        if not is_ring:
            return 0 if dst > src else 1
        if fwd < rev:
            return 0
        if rev < fwd:
            return 1
        tie_toggle[0] ^= 1
        return tie_toggle[0]

    for c in chips:
        for _ in range(tokens_per_chip):
            dst = rng.randrange(extent)
            if dst == c.pos:
                continue  # local tokens never touch the relay path
            c.inject.append(dst)

    expected = sum(len(c.inject) for c in chips)

    def stage_into(sender, d, level, token):
        """One single-hop write into the downstream chip's FIFO for this level."""
        nbr = downstream_pos(sender.pos, extent, is_ring, d == 0)
        assert nbr is not None
        sender.staged[d][level - 1] += 1
        target = chips[nbr]

        def land():
            target.fifo[d][level - 1].append(token)
            target.arrived[d][level - 1] += 1

        fabric.send(step, land)

    def final_write(sender, d, token):
        nbr = downstream_pos(sender.pos, extent, is_ring, d == 0)
        assert nbr is not None and nbr == token, f"final write from {sender.pos} to {nbr}, token wants {token}"
        fabric.send(step, lambda: delivered.append(token))

    def try_transit(c):
        for d in range(2):
            for r in range(1, levels + 1):
                if not in_live(c.pos, extent, is_ring, d == 0, r):
                    continue
                if c.pool_rd[d][r - 1] == c.arrived[d][r - 1]:
                    continue
                if not c.has_room(d, r - 1, head_of_line):
                    continue
                token = c.fifo[d][r - 1].pop(0)
                assert (
                    ring_distance(c.pos, token, extent, is_ring) == r
                ), f"chip {c.pos} level {r} holds a token {ring_distance(c.pos, token, extent, is_ring)} hops away"
                if r == 1:
                    final_write(c, d, token)
                else:
                    stage_into(c, d, r - 1, token)
                c.pool_rd[d][r - 1] += 1
                # A slot is returnable as soon as its page is out of the buffer.
                up = upstream_pos(c.pos, extent, is_ring, d == 0)
                if up is not None:

                    def return_credit(u=up, dd=d, rr=r):
                        chips[u].credit[dd][rr - 1] += 1

                    fabric.send(step, return_credit)
                return True
        return False

    def try_inject(c):
        if not c.inject:
            if not c.untilizers_done:
                c.untilizers_done = True
                return True
            return False
        token = c.inject[0]
        d = route_dir(c.pos, token)
        distance = ring_distance(c.pos, token, extent, is_ring)
        target_level = distance - 1
        if not c.has_room(d, target_level, head_of_line):
            return False  # no commit; the row and its credit stay put
        c.inject.pop(0)
        if target_level == 0:
            final_write(c, d, token)
        else:
            stage_into(c, d, target_level, token)
        return True

    def try_emit_eos(c):
        if not c.untilizers_done:
            return
        for d in range(2):
            for r in range(levels, 0, -1):
                if not out_live(c.pos, extent, is_ring, d == 0, r) or c.eos_out[d][r - 1]:
                    continue
                if eos_rule == "naive":
                    # Waits on the upstream's close at this same level. On a ring every chip is fed,
                    # so no chip can ever start and the chain never begins.
                    if in_live(c.pos, extent, is_ring, d == 0, r) and not c.drained(d, r):
                        continue
                else:
                    fed_from_above = r + 1 <= levels and in_live(c.pos, extent, is_ring, d == 0, r + 1)
                    if fed_from_above and not c.drained(d, r + 1):
                        continue
                nbr = downstream_pos(c.pos, extent, is_ring, d == 0)
                if nbr is not None:

                    def close(n=nbr, dd=d, rr=r):
                        chips[n].closed[dd][rr - 1] = True

                    fabric.send(step, close)
                c.eos_out[d][r - 1] = True

    def all_done(c):
        if not c.untilizers_done or c.inject:
            return False
        for d in range(2):
            for r in range(1, levels + 1):
                if in_live(c.pos, extent, is_ring, d == 0, r) and not c.drained(d, r):
                    return False
                if out_live(c.pos, extent, is_ring, d == 0, r) and not c.eos_out[d][r - 1]:
                    return False
        return True

    step = 0
    while step < max_steps:
        fabric.deliver(step)
        progressed = False
        for c in chips:
            if all_done(c):
                continue
            moved = False
            if c.transit_run < quantum:
                moved = try_transit(c)
                if moved:
                    c.transit_run += 1
            if not moved:
                c.transit_run = 0
                moved = try_inject(c)
            try_emit_eos(c)
            progressed = progressed or moved
        # Outstanding buffer occupancy must never exceed what credits allow.
        for c in chips:
            for d in range(2):
                for r in range(levels):
                    assert c.staged[d][r] - c.credit[d][r] <= c.slots, f"chip {c.pos} overran ring ({d},{r + 1})"
        step += 1
        if all(all_done(c) for c in chips) and fabric.idle():
            break
        if not progressed and fabric.idle():
            pytest.fail(f"stalled at step {step} with work outstanding: deadlock")
    else:
        pytest.fail(f"did not terminate within {max_steps} steps")

    return delivered, expected, step


@pytest.mark.parametrize("slots", [1, 2, 4], ids=lambda s: f"slots{s}")
@pytest.mark.parametrize(
    "extent,is_ring", [(8, True), (4, True), (8, False), (4, False)], ids=["ring8", "ring4", "line8", "line4"]
)
def test_sf_protocol_terminates_and_delivers(extent, is_ring, slots):
    """Ring is the case with a real cycle: every chip both feeds and is fed in both directions, so a
    protocol that waited on an upstream close would never start."""
    for seed in range(6):
        delivered, expected, steps = run_protocol(extent, is_ring, slots, tokens_per_chip=12, seed=seed)
        assert len(delivered) == expected, f"delivered {len(delivered)} of {expected} tokens"


def test_sf_protocol_eos_base_case_is_local():
    """The top level must never be fed, on any chip or direction. That is what lets end-of-stream
    start without waiting on a neighbour -- the property a ring otherwise lacks."""
    for extent, is_ring in [(8, True), (4, True), (8, False), (4, False)]:
        levels = num_levels(extent, is_ring)
        if levels == 0:
            continue
        for pos in range(extent):
            for positive in (True, False):
                assert not in_live(pos, extent, is_ring, positive, levels + 1)


def test_sf_protocol_masks_agree_across_neighbours():
    """A chip's inbound mask must equal its upstream's outbound mask, or the two disagree about
    whether a FIFO is used and the drain/end-of-stream handshake mismatches."""
    for extent, is_ring in [(8, True), (4, True), (8, False), (4, False), (16, True)]:
        levels = num_levels(extent, is_ring)
        for pos in range(extent):
            for positive in (True, False):
                up = upstream_pos(pos, extent, is_ring, positive)
                for r in range(1, levels + 1):
                    mine = in_live(pos, extent, is_ring, positive, r)
                    theirs = up is not None and out_live(up, extent, is_ring, positive, r)
                    assert mine == theirs


def test_model_detects_head_of_line_blocking():
    """The credit wait belongs in the reader, where a full level makes it try another source rather
    than wait. Putting it in the writer instead stalls the shared queue -- and that closes a cycle on
    a line as well as a ring, which is why it is the structural decision the rest depends on."""
    # expect_error is for device faults; it emits CI triage markers, which a deliberate model
    # failure must not do.
    with pytest.raises(BaseException, match="deadlock"):  # allow-pytest.raises: host-only model self-check
        run_protocol(8, True, 1, 12, 0, max_steps=200_000, head_of_line=True)


def test_model_detects_naive_end_of_stream():
    """End-of-stream is emitted per level with a purely local base case at the top. The obvious
    alternative -- wait for the upstream to close the same level -- is circular on a ring."""
    with pytest.raises(BaseException, match="deadlock"):  # allow-pytest.raises: host-only model self-check
        run_protocol(8, True, 4, 12, 0, max_steps=200_000, eos_rule="naive")
