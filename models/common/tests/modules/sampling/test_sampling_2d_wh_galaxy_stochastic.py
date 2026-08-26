# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stochastic-path hardware qualification for Sampling2D on a Wormhole Galaxy.

``test_sampling_2d_wh_galaxy.py`` qualifies the greedy path only: it passes both
``forced_argmax=True`` and ``temperature=0.0``, so ``_update_call_buffers`` collapses
every slot to ``k=1, p=0.0, temp=1.0`` and ``ttnn.sampling`` never takes its stochastic
branch. This file covers what that leaves open - seeded stochastic, unseeded stochastic,
the top-p nucleus, and per-slot heterogeneous request parameters - on the same qualified
``(8, 4)`` geometry.

Device tokens are never compared against ``Sampling2D.sample_host``. The two paths draw
from different generators with different seed widths (``_device_seed`` masks the blake2b
digest to 31 bits, ``_host_seed`` to 63) so token-for-token equality is not a property of
the design. What is assertable is support containment, padded-vocabulary exclusion,
seeded determinism, and unseeded freshness.
"""

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.sampling.sampling_2d import Sampling2D
from models.common.tests.modules._hf_reference import hf_valid_token_set

VOCAB_SIZE = 151936
PADDED_VOCAB_SIZE = 152064
BATCH = 32

# Synthetic logits are built from an explicit per-slot candidate set rather than from
# ``torch.randn``: a bfloat16 draw over 151936 tokens produces many exact ties at the
# top-k threshold, which inflates the HuggingFace reference set (``TopKLogitsWarper``
# keeps every token tied with the k-th value) and weakens containment. A tie-free ladder
# makes the eligible set exact.
_CANDIDATES = 64
_LOGIT_TOP = 8.0
_LOGIT_STEP = 0.25  # exactly representable in bfloat16 for every value on the ladder
_BASELINE = -20.0

_DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
_MESH_8X4 = pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)


def _deallocate(tensor):
    if tensor is not None:
        tensor.deallocate(True)


def _sampler(mesh_device):
    """Build Sampling2D on the geometry qualified by the greedy Milestone A test."""
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    sub_core_grid_topk = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9))])
    return Sampling2D(
        VOCAB_SIZE,
        PADDED_VOCAB_SIZE,
        mesh_device,
        sub_core_grids=sub_core_grids,
        sub_core_grid_topk=sub_core_grid_topk,
        start_core=ttnn.CoreCoord(1, 0),
    )


def _candidate_logits(seed: int, *, tail_value: float = _BASELINE, flat: bool = False, candidates: int = _CANDIDATES):
    """Return ``([1, 1, 32, padded], ids)`` logits with an exact candidate set per slot.

    Every slot gets ``candidates`` distinct token ids drawn uniformly from the valid
    vocabulary. With ``flat=False`` they carry a strictly descending ladder starting at
    ``_LOGIT_TOP``; every other valid column sits at ``_BASELINE``, far below the ladder,
    so the global top-k for any ``k <= candidates`` is exactly the first ``k`` ladder
    entries and the argmax is exactly ``ids[slot, 0]``.

    The ladder is also spread across the 8 vocabulary shards, so no shard ever holds more
    than ``max_top_k=32`` candidates in expectation and the per-shard top-32 gather cannot
    drop an eligible token.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.full((1, 1, BATCH, PADDED_VOCAB_SIZE), _BASELINE, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(VOCAB_SIZE, generator=generator)[:candidates] for _ in range(BATCH)])
    if flat:
        values = torch.zeros(candidates, dtype=torch.bfloat16)
    else:
        values = _LOGIT_TOP - _LOGIT_STEP * torch.arange(candidates, dtype=torch.bfloat16)
    logits[0, 0].scatter_(1, ids, values.expand(BATCH, candidates).contiguous())
    logits[..., VOCAB_SIZE:] = tail_value
    return logits, ids


def _to_device(logits, mesh_device):
    return ttnn.from_torch(
        logits,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=(8, 4)),
    )


def _sampled_tokens(output):
    return to_torch_auto_compose(output).reshape(-1)[:BATCH].to(torch.int64)


def _containment_violations(logits, tokens, *, k, p, temp, slots=None):
    """Return ``(slot, token, valid_set_size)`` for every token outside the HF valid set."""
    rows = logits[0, 0, :, :VOCAB_SIZE]
    violations = []
    for slot in range(BATCH) if slots is None else slots:
        valid = hf_valid_token_set(rows[slot], k=k, p=p, temp=temp)
        token = int(tokens[slot])
        if token not in valid:
            violations.append((slot, token, len(valid)))
    return violations


def _report(label, violations, allowed, total):
    """Emit the observed violation count so passing runs record their own calibration."""
    print(
        f"[sampling2d-stochastic] {label}: violations={len(violations)}/{total} allowed={allowed}"
        + ("" if not violations else " " + str(violations[:5]))
    )


def _assert_within_tolerance(label, violations, allowed, total):
    _report(label, violations, allowed, total)
    assert len(violations) <= allowed, (
        f"{label}: {len(violations)}/{total} tokens outside the HF valid set "
        f"(max allowed={allowed} for the bfloat16 nucleus boundary):\n"
        + "\n".join(f"  slot {slot}: token {token} not in {size}-token valid set" for slot, token, size in violations)
    )


@_DEVICE_PARAMS
@_MESH_8X4
@pytest.mark.parametrize(
    "top_k, top_p, temperature, max_boundary_violations",
    [
        # p == 0.0 or p == 1.0: no nucleus threshold exists, so the eligible set is
        # exactly the top-k and any violation is a real defect. Zero tolerance.
        pytest.param(1, 0.0, 1.0, 0, id="k1-p0-t1"),
        pytest.param(8, 1.0, 1.0, 0, id="k8-p1-t1"),
        pytest.param(32, 1.0, 1.0, 0, id="k32-p1-t1"),
        # p in (0, 1): ttnn.sampling computes its softmax and cumulative sum in bfloat16
        # while the HuggingFace reference uses float32, so a token sitting on the nucleus
        # cutoff can land on either side. The bound of 1 below is headroom for that, not
        # an observed requirement - the observed count was 0 - and it is not inherited
        # from the 1D suite. See the docstring.
        pytest.param(32, 0.9, 1.0, 1, id="k32-p0.9-t1"),
        pytest.param(32, 0.5, 0.8, 1, id="k32-p0.5-t0.8"),
    ],
)
def test_sampling_2d_wh_galaxy_stochastic_token_in_valid_set(
    mesh_device, top_k, top_p, temperature, max_boundary_violations
):
    """Every stochastically sampled token lies in the HuggingFace-derived eligible set.

    For each ``(k, p, temp)`` the pipeline
    ``TemperatureLogitsWarper -> TopKLogitsWarper -> TopPLogitsWarper`` defines which
    tokens are eligible; a correct sampler can only draw from that set. Containment is
    exactly valid here because ``k <= max_top_k = 32``: the true global top-k is always a
    subset of the union of the eight per-shard top-32 sets, so the all-gather cannot drop
    an eligible token before ``ttnn.sampling`` runs.

    Calibrated on this host from runs 03-05 (three fresh processes, recorded under
    ``tttv2_milestone_a_gap1_evidence/``): the observed violation count was 0 in every
    case, in every invocation, including both ``p in (0, 1)`` nucleus cases. The bound of
    1 on those two is one above that observed maximum - deliberate headroom for the
    bfloat16 softmax/cumsum boundary noted in the parametrization, which did not manifest
    on this geometry at all once the reciprocal-temperature defect in
    ``_update_call_buffers`` was fixed. Pre-fix, ``k32-p0.5-t0.8`` showed 4/32
    (``run02_prefix_defect_demo.log``). A run that reports violations here is a
    regression, not noise.
    """
    logits, _ = _candidate_logits(seed=17)
    sampler = _sampler(mesh_device)
    tt_logits = _to_device(logits, mesh_device)
    label = f"k{top_k}-p{top_p}-t{temperature}"

    try:
        for invocation in range(2):
            output = sampler.decode_forward(
                tt_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=None,
                forced_argmax=False,
            )
            try:
                tokens = _sampled_tokens(output)
                assert torch.all(tokens < VOCAB_SIZE), f"{label}: sampled a padded vocabulary id: {tokens.tolist()}"
                violations = _containment_violations(logits, tokens, k=top_k, p=top_p, temp=temperature)
                _assert_within_tolerance(f"{label} invocation {invocation}", violations, max_boundary_violations, BATCH)
            finally:
                _deallocate(output)
    finally:
        _deallocate(tt_logits)
        sampler.release()


@_DEVICE_PARAMS
@_MESH_8X4
def test_sampling_2d_wh_galaxy_stochastic_excludes_padded_vocab(mesh_device):
    """Padded LM-head columns stay out of the candidate set under stochastic sampling.

    The greedy Milestone A test cannot make this assertion: with ``k=1`` a single
    dominant valid logit hides a partially broken mask. Here the padded tail
    ``[151936:152064]`` is raised to ``+1000.0`` - far above every valid logit - and the
    sampler draws stochastically over the full 32-wide support. ``invalid_vocab_mask`` is
    added before ``ttnn.topk``, so a correct implementation never admits a padded column
    to the per-shard top-32, let alone to the draw.

    A failure here is a real defect in the mask or in its placement, never a tolerance
    question: ``p=1.0`` means no nucleus threshold exists and the eligible set is exact.
    """
    logits, _ = _candidate_logits(seed=23, tail_value=1000.0)
    sampler = _sampler(mesh_device)
    tt_logits = _to_device(logits, mesh_device)

    try:
        for invocation in range(8):
            output = sampler.decode_forward(
                tt_logits,
                top_k=32,
                top_p=1.0,
                temperature=1.0,
                seed=None,
                forced_argmax=False,
            )
            try:
                tokens = _sampled_tokens(output)
                assert torch.all(
                    tokens < VOCAB_SIZE
                ), f"invocation {invocation}: sampled a padded vocabulary id: {tokens.tolist()}"
                violations = _containment_violations(logits, tokens, k=32, p=1.0, temp=1.0)
                _assert_within_tolerance(f"padded-tail invocation {invocation}", violations, 0, BATCH)
            finally:
                _deallocate(output)
    finally:
        _deallocate(tt_logits)
        sampler.release()


@_DEVICE_PARAMS
@_MESH_8X4
def test_sampling_2d_wh_galaxy_seeded_sampling_is_repeatable_and_slot_stable(mesh_device):
    """A seeded stochastic call is repeatable, and each slot's draw depends only on its own seed.

    ``_update_call_buffers`` refills all 32 seed values with ``secrets.randbits(31)`` on
    every call and only then overwrites the slots whose seed is not ``None``, so a fully
    seeded call is deterministic. ``_device_seed(seed, slot)`` derives a distinct per-slot
    seed from one integer and ``slot_placement`` maps slot -> ``(mesh column, local
    index)``; slot *i*'s token must therefore depend on ``(seed_i, logits row i)`` and
    nothing else.

    The perturbation call reseeds slots 16-31 only. Slots 0-15 must be byte-identical -
    anything else is cross-slot RNG contamination, a serving-correctness bug. Slots 16-31
    must differ somewhere: with a 32-token support whose probabilities follow the
    ``exp(-0.25)`` ladder, an independent redraw reproduces a given slot with probability
    ``sum(p_i^2) ~ 0.12``, so a correct implementation fails this half of the assertion
    with probability about ``0.12 ** 16 ~ 5e-15``.
    """
    logits, _ = _candidate_logits(seed=31)
    sampler = _sampler(mesh_device)
    tt_logits = _to_device(logits, mesh_device)
    kwargs = dict(top_k=32, top_p=1.0, temperature=1.0, forced_argmax=False)

    def sample(seed):
        output = sampler.decode_forward(tt_logits, seed=seed, **kwargs)
        try:
            return _sampled_tokens(output)
        finally:
            _deallocate(output)

    try:
        first = sample(1234)
        repeated = sample(1234)
        assert torch.equal(first, repeated), (
            "seed=1234 produced different tokens across two calls:\n"
            f"  call 1: {first.tolist()}\n  call 2: {repeated.tolist()}"
        )

        perturbed = sample([1234] * 16 + [9999] * 16)
        assert torch.equal(perturbed[:16], first[:16]), (
            "reseeding slots 16-31 changed slots 0-15, so per-slot RNG state is not isolated:\n"
            f"  baseline:  {first[:16].tolist()}\n  perturbed: {perturbed[:16].tolist()}"
        )
        assert not torch.equal(
            perturbed[16:], first[16:]
        ), f"reseeding slots 16-31 left every token unchanged: {first[16:].tolist()}"
    finally:
        _deallocate(tt_logits)
        sampler.release()


@_DEVICE_PARAMS
@_MESH_8X4
def test_sampling_2d_wh_galaxy_unseeded_sampling_uses_fresh_randomness(mesh_device):
    """An unseeded stochastic call draws fresh randomness on every invocation.

    With ``seed=None`` every slot receives a fresh ``secrets.randbits(31)`` per call. The
    logits here are deliberately flat - 32 candidates at an identical logit over a
    ``-20.0`` baseline - so the support is exactly those 32 tokens and each draw is
    uniform over them.

    The assertion is that the eight token vectors are not all identical. For a correct
    implementation that requires all 32 slots to reproduce their call-0 token in each of
    seven further calls: ``(32 ** -32) ** 7``, about ``1e-337``. Nothing stronger about
    the distribution is asserted here.
    """
    logits, ids = _candidate_logits(seed=37, flat=True, candidates=32)
    sampler = _sampler(mesh_device)
    tt_logits = _to_device(logits, mesh_device)
    candidate_sets = [set(ids[slot].tolist()) for slot in range(BATCH)]

    try:
        draws = []
        for invocation in range(8):
            output = sampler.decode_forward(
                tt_logits,
                top_k=32,
                top_p=1.0,
                temperature=1.0,
                seed=None,
                forced_argmax=False,
            )
            try:
                tokens = _sampled_tokens(output)
            finally:
                _deallocate(output)
            assert torch.all(
                tokens < VOCAB_SIZE
            ), f"invocation {invocation}: sampled a padded vocabulary id: {tokens.tolist()}"
            outside = [
                (slot, int(tokens[slot])) for slot in range(BATCH) if int(tokens[slot]) not in candidate_sets[slot]
            ]
            assert not outside, f"invocation {invocation}: tokens outside the flat 32-token support: {outside}"
            draws.append(tokens)

        distinct = {tuple(tokens.tolist()) for tokens in draws}
        print(f"[sampling2d-stochastic] unseeded: {len(distinct)} distinct token vectors across 8 invocations")
        assert len(distinct) > 1, f"eight unseeded invocations produced identical tokens: {draws[0].tolist()}"
    finally:
        _deallocate(tt_logits)
        sampler.release()


@_DEVICE_PARAMS
@_MESH_8X4
def test_sampling_2d_wh_galaxy_per_slot_heterogeneous_parameters(mesh_device):
    """One call mixes greedy and stochastic slots, each honouring its own parameters.

    Production sends a different ``top_k`` / ``top_p`` / ``temperature`` per request and
    ``_broadcast`` accepts per-slot sequences, but no device test exercises that. The four
    groups here are:

    * slots 0-7 - ``forced_argmax=True``: must equal argmax exactly.
    * slots 8-15 - ``temperature=0.0`` with ``forced_argmax=False``: pins the collapse in
      ``_update_call_buffers``, which forces ``k=1, p=0.0, temp=1.0``; must equal argmax.
    * slots 16-23 - ``k=8, p=1.0``: exact top-8 containment, zero tolerance.
    * slots 24-31 - ``k=32, p=0.9, temp=0.8``: nucleus containment, calibrated tolerance.

    The candidate ladder makes the argmax unique (``ids[slot, 0]`` at ``+8.0``, the next
    candidate 0.25 below), so the greedy groups are compared for exact equality.

    Calibrated on this host from runs 03-05 (three fresh processes, recorded under
    ``tttv2_milestone_a_gap1_evidence/``): the slots 24-31 violation count was 0 in every
    invocation. Its bound of 1 is one above that observed maximum - deliberate headroom
    for the bfloat16 softmax/cumsum nucleus boundary, which did not manifest here once the
    reciprocal-temperature defect in ``_update_call_buffers`` was fixed. Pre-fix, this
    group showed 2/8 (``run02_prefix_defect_demo.log``). A run that reports violations
    here is a regression, not noise. The other three groups admit no violation at all.
    """
    logits, ids = _candidate_logits(seed=41)
    expected_argmax = ids[:, 0].to(torch.int64)
    sampler = _sampler(mesh_device)
    tt_logits = _to_device(logits, mesh_device)

    groups = [
        # (slots, top_k, top_p, temperature, forced_argmax)
        (range(0, 8), 32, 1.0, 1.0, True),
        (range(8, 16), 32, 1.0, 0.0, False),
        (range(16, 24), 8, 1.0, 1.0, False),
        (range(24, 32), 32, 0.9, 0.8, False),
    ]
    top_k = [0] * BATCH
    top_p = [0.0] * BATCH
    temperature = [0.0] * BATCH
    forced_argmax = [False] * BATCH
    for slots, k, p, temp, forced in groups:
        for slot in slots:
            top_k[slot], top_p[slot], temperature[slot], forced_argmax[slot] = k, p, temp, forced

    try:
        for invocation in range(2):
            output = sampler.decode_forward(
                tt_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=None,
                forced_argmax=forced_argmax,
            )
            try:
                tokens = _sampled_tokens(output)
                assert torch.all(tokens < VOCAB_SIZE), f"invocation {invocation}: sampled a padded vocabulary id"

                greedy = list(range(0, 16))
                assert torch.equal(tokens[greedy], expected_argmax[greedy]), (
                    f"invocation {invocation}: greedy slots did not equal argmax:\n"
                    f"  actual:   {tokens[greedy].tolist()}\n  expected: {expected_argmax[greedy].tolist()}"
                )

                _assert_within_tolerance(
                    f"heterogeneous k8-p1-t1 slots 16-23 invocation {invocation}",
                    _containment_violations(logits, tokens, k=8, p=1.0, temp=1.0, slots=range(16, 24)),
                    0,
                    8,
                )
                _assert_within_tolerance(
                    f"heterogeneous k32-p0.9-t0.8 slots 24-31 invocation {invocation}",
                    _containment_violations(logits, tokens, k=32, p=0.9, temp=0.8, slots=range(24, 32)),
                    1,
                    8,
                )
            finally:
                _deallocate(output)
    finally:
        _deallocate(tt_logits)
        sampler.release()
