# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A walk split at ANY layer reproduces the undivided walk, on device -- bit for bit.

`rank_site_groups` lets a window start mid-block: the walk seals on global multiples of the
block size and reads the rest of an inherited block against the inherited set. The host tests
(`test_attn_res_pipeline_split.py`) pin the bookkeeping and drive the walk with a stub op; this
one runs it, all 93 layers deep, split at block edges and off them.

A split differs from the undivided walk in exactly one arithmetic step: `handoff` settles the
deferred residual write with a plain `ttnn.add`, where the undivided walk would fold it into its
next read. Everything else is the same computation -- a head group batches different sites, but
every per-site result is independent of the batch -- EXCEPT where a head group is one site wide
at sealed depth >= 4 (an aligned entry at 48, 60, ...), which takes the folded dot reduce. So the
primary oracle is an undivided walk that settles its write at the same boundaries:

* off those fold entries the split must equal it BITWISE -- output, live stream at every block
  end and handoff, and every sealed plane. That is what makes a schedule error visible at all:
  against a merely-close oracle, one swapped query moved the output by about one bf16 ulp
  (relative L2 ~4e-3), the same as the handoff's rounding;
* at a fold entry, PCC >= 0.99999 and relative L2 <= 1e-2 instead;
* against the plain undivided walk, and an fp32 torch walk (`attn_res_stack`), within rounding.

A deliberately wrong walk (one query swapped inside a split rank's head group) must break the
bitwise oracle, which is what proves the oracle can fail.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS, attn_res_stack
from models.demos.deepseek_v3_d_p.tests.attn_res.assertions import assert_accurate
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import (
    HIDDEN_SIZE,
    PER_CHIP_TOKENS,
    blackhole_only,
    compose,
    generator,
    place,
    placements,
    random_hidden,
    random_queries,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import BLOCK_SIZE, TtAttnResWalk, sealed_before

LAYERS = 93
# Single splits at and around block edges, and one three-rank cut whose middle rank both enters
# and leaves mid-block.
SPLITS = ((1,), (5,), (11,), (12,), (13,), (47,), (48,), (49,), (84,), (85,), (92,), (5, 20))
SPLIT_PCC = 0.99999
SPLIT_REL_L2 = 1e-2
REFERENCE_PCC = 0.9999
MODULE_SCALE = 0.05

pytestmark = blackhole_only


def _rel_l2(want, got):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


def _assert_close(want, got, name, pcc=SPLIT_PCC, rel_l2=SPLIT_REL_L2):
    measured = assert_accurate(want, got, name=name, pcc_threshold=pcc)
    error = _rel_l2(want, got)
    assert error <= rel_l2, f"{name}: relative L2 {error:.3e} > {rel_l2:.1e} (PCC {measured:.8f})"
    return measured, error


class _Modules:
    """Per-feature stand-ins: `h * w[layer, site]`, `w` ~ MODULE_SCALE * N(1, 0.5) per feature."""

    def __init__(self, op, rng):
        self.host = {
            (layer, site): MODULE_SCALE * (1.0 + 0.5 * torch.randn(HIDDEN_SIZE, generator=rng))
            for layer in range(LAYERS)
            for site in ("attn", "mlp")
        }
        self.device = {
            key: ttnn.from_torch(
                w.reshape(1, 1, 1, HIDDEN_SIZE),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=op.mesh_device,
                mesh_mapper=op.vector_mapper,
            )
            for key, w in self.host.items()
        }

    def torch_fns(self, site):
        return [lambda h, w=self.host[(layer, site)]: h * w for layer in range(LAYERS)]

    def deallocate(self):
        for tensor in self.device.values():
            ttnn.deallocate(tensor)


def _run_window(op, modules, hidden, queries, first, count, inherited=None, observe_live_at=(), settle_at=()):
    """Global layers `[first, first + count)` through one walk, as a pipeline rank runs them.

    Returns `(out_or_live, sealed, observed)`: `("finish", out, sealed_clone)` on the last window,
    `("handoff", live, sealed)` before it. `observed[L]` is the live stream after global layer
    `L - 1` (running sum plus any deferred write, summed on host so the walk is not disturbed).
    `settle_at` names layers after which the deferred write is settled the way `handoff` does it.
    """
    q_pre, q_post, q_out = queries
    walk = TtAttnResWalk(
        op,
        hidden,
        q_pre[first : first + count],
        q_post[first : first + count],
        q_out,
        count,
        inherited_block_residual=inherited,
        first_layer_idx=first,
    )
    observed = {}
    for local in range(count):
        layer = first + local
        h, borrowed = walk.open_layer(local)
        walk.write(ttnn.multiply(h, modules.device[(layer, "attn")]))
        if not borrowed:
            ttnn.deallocate(h)
        h = walk.read()
        walk.write(ttnn.multiply(h, modules.device[(layer, "mlp")]))
        ttnn.deallocate(h)
        if layer + 1 in settle_at:
            walk.stream.running_sum  # settles the deferred write with ttnn.add, as handoff does
        if layer + 1 in observe_live_at:
            stream = walk.stream
            live = compose(op, stream._running_sum).float()
            if stream._pending is not None:
                live = live + compose(op, stream._pending).float()
            observed[layer + 1] = live
    if first + count < LAYERS:
        live, sealed = walk.handoff()
        return "handoff", live, sealed, observed
    # Reading `block_residual` directly does not flush the deferred write.
    sealed = ttnn.clone(walk.stream.block_residual)
    return "finish", walk.finish(), sealed, observed


# The live stream just before every seal, and after the last layer: where a wrong read in the
# block that just ended has not yet been diluted by the layers after it.
BLOCK_ENDS = frozenset([*range(BLOCK_SIZE, LAYERS, BLOCK_SIZE), LAYERS])


def _run_split(op, modules, embed, queries, cuts):
    """Run the stack cut at `cuts`; returns `(out, final_sealed, handoffs, live_at_block_ends)` with
    each handoff's `(boundary, live, sealed planes)` composed on host."""
    edges = (0, *cuts, LAYERS)
    hidden, inherited, handoffs, observed = embed(), None, [], {}
    for first, end in zip(edges[:-1], edges[1:]):
        kind, a, b, seen = _run_window(
            op, modules, hidden, queries, first, end - first, inherited=inherited, observe_live_at=BLOCK_ENDS
        )
        observed.update(seen)
        if kind == "handoff":
            assert b.shape[1] == sealed_before(
                end
            ), f"split at {end}: {b.shape[1]} planes, expected {sealed_before(end)}"
            handoffs.append((end, compose(op, a).float(), compose(op, b)))
            hidden, inherited = a, b
        else:
            out, sealed = compose(op, a), compose(op, b)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            return out, sealed, handoffs, observed


def _takes_fold(cut):
    """An entry whose head group is the single straddling read at sealed depth >= 4 folds its dots."""
    return cut % BLOCK_SIZE == 0 and sealed_before(cut) >= 4


class _Case:
    """One op (production folding, or not) with its own placed queries and modules."""

    def __init__(self, mesh_device, fold_stats, rng_seed, hidden_states):
        self.op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, fold_stats=fold_stats)
        rng = generator(rng_seed)
        random_hidden(rng, hidden_states.shape[0])  # keep the draw order of the host case
        self.q_pre, self.q_post = random_queries(rng, LAYERS), random_queries(rng, LAYERS)
        self.q_out = random_queries(rng, 1)[0]
        self.modules = _Modules(self.op, rng)
        self.queries = (
            [self.op.to_query(q) for q in self.q_pre],
            [self.op.to_query(q) for q in self.q_post],
            self.op.to_query(self.q_out),
        )
        self.embed = lambda: place(self.op, hidden_states.unsqueeze(0).unsqueeze(0))

    def undivided(self, settle_at=(), queries=None):
        _, out, sealed, live = _run_window(
            self.op,
            self.modules,
            self.embed(),
            queries or self.queries,
            0,
            LAYERS,
            observe_live_at=BLOCK_ENDS | set(settle_at),
            settle_at=settle_at,
        )
        result = compose(self.op, out), compose(self.op, sealed), live
        ttnn.deallocate(out)
        ttnn.deallocate(sealed)
        return result

    def split(self, cuts, queries=None):
        got, sealed, handoffs, block_ends = _run_split(self.op, self.modules, self.embed, queries or self.queries, cuts)
        observed = {**block_ends, **{boundary: live for boundary, live, _ in handoffs}}
        return got, sealed, observed

    def deallocate(self):
        self.modules.deallocate()
        for tensor in (self.queries[2], *self.queries[0], *self.queries[1]):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device, device_params", placements(), indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(3600)
def test_split_anywhere_matches_undivided_walk(mesh_device, device_params):
    rows = PER_CHIP_TOKENS * mesh_device.shape[0]
    hidden_states = random_hidden(generator(1), rows)
    # Bitwise oracle on an op that never folds, so EVERY entry -- aligned ones at sealed depth >= 4
    # included -- is held to exact equality. The production (folding) op is then held to exact
    # equality up to each cut and to closeness after it, where its folded head read may round
    # differently (`_collective` is not bit-neutral in general).
    exact = _Case(mesh_device, False, 1, hidden_states)
    folding = _Case(mesh_device, True, 1, hidden_states)

    want = attn_res_stack(
        hidden_states,
        folding.q_pre,
        folding.q_post,
        folding.q_out,
        folding.modules.torch_fns("attn"),
        folding.modules.torch_fns("mlp"),
        block_size=BLOCK_SIZE,
        eps=EPS,
    )
    plain, _, _ = folding.undivided()
    assert_accurate(want, plain, name="undivided vs torch", pcc_threshold=REFERENCE_PCC)

    worst = (1.0, 0.0)
    for cuts in SPLITS:
        oracle, oracle_sealed, oracle_live = exact.undivided(settle_at=cuts)
        got, sealed, observed = exact.split(cuts)
        for at, live in observed.items():
            assert torch.equal(oracle_live[at].float(), live.float()), f"split {cuts}: live at {at} not bitwise"
        assert torch.equal(oracle_sealed, sealed), f"split {cuts}: sealed set not bitwise"
        assert torch.equal(oracle, got), f"split {cuts}: output not bitwise"

        # The production op: exact before the first cut (rank 0 is the undivided walk) and on the
        # planes it handed over; within rounding after.
        oracle, oracle_sealed, oracle_live = folding.undivided(settle_at=cuts)
        got, sealed, observed = folding.split(cuts)
        handed = sealed_before(cuts[0]) * rows
        assert torch.equal(oracle_sealed[:handed], sealed[:handed]), f"split {cuts}: rank-0 planes differ"
        for at, live in observed.items():
            if at <= cuts[0]:
                assert torch.equal(oracle_live[at].float(), live.float()), f"split {cuts}: live at {at} not bitwise"
            else:
                _assert_close(oracle_live[at], live, f"split {cuts} live at {at}")
        if sealed.shape[0] > handed:
            _assert_close(oracle_sealed[handed:], sealed[handed:], f"split {cuts} resealed")
        pcc, error = _assert_close(oracle, got, f"split {cuts} output")
        _assert_close(plain, got, f"split {cuts} vs plain undivided")
        assert_accurate(want, got, name=f"split {cuts} vs torch", pcc_threshold=REFERENCE_PCC)
        worst = (min(worst[0], pcc), max(worst[1], error))
        exactly = torch.equal(oracle, got)
        logger.info(f"split {cuts}: bitwise (no-fold op); production op {'bitwise' if exactly else f'PCC {pcc:.8f}'}")

    logger.info(f"worst production-op split: PCC {worst[0]:.8f}, relative L2 {worst[1]:.2e}")

    # Negative control: one query swapped inside the split rank's head group (layer 50, split at
    # 49). It must break the bitwise oracle -- and it is smaller than the close oracle's tolerance,
    # which is why a close-but-not-equal oracle could not have caught it.
    oracle, _, _ = exact.undivided(settle_at=(49,))
    swapped = (list(exact.queries[0]), list(exact.queries[1]), exact.queries[2])
    swapped[0][50], swapped[1][50] = exact.queries[1][50], exact.queries[0][50]
    wrong, _, _ = exact.split((49,), queries=swapped)
    error = _rel_l2(oracle, wrong)
    assert not torch.equal(oracle, wrong), "the bitwise oracle cannot see a swapped query"
    assert (
        error <= SPLIT_REL_L2
    ), f"control moved the output {error:.2e}; the case for a bitwise oracle needs < tolerance"
    logger.info(f"negative control (layer 50 queries swapped): relative L2 {error:.2e}, caught only bitwise")

    exact.deallocate()
    folding.deallocate()
