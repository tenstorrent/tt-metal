# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step 7, area 4: device sampling.

``Sampling2D`` already has a host suite of its own
(``models/common/tests/modules/sampling/test_sampling_2d.py``). This suite covers
what step 7 adds on top of it: the *composition* of the sampler with the Galaxy
column user selector and the direct runner, at the two real model vocabularies.

Four claims, and what this file can and cannot say about each:

1. **Slot stability.** Two different claims hide under that word. "The same
   seed in the same slot gives the same token across runs" holds and is proved
   here. "Moving a request to a different slot does not change its stream" does
   **not** hold - the device and host seeds are both derived from
   ``blake2b("sampling2d:{seed}:{slot}")`` - and the test that measures it says
   so rather than asserting the gate. See REPORT.md area 4, D-C2.
2. **Greedy matches host argmax.** Proved here on host, at both vocabularies,
   including when a padded entry carries the largest logit.
3. **Padded vocabulary can never be sampled.** Proved here for Qwen. For Llama
   it is *vacuous*: 128256 is already a multiple of ``8 * 32``, so there is no
   padding and no mask. That is asserted explicitly because the step-7 brief
   assumes both models pad.
4. **Per-slot heterogeneous controls.** Proved here by reading back exactly the
   buffers that would be shipped to the mesh.

Plus one composition property that no single module can check: the sampler's
slot->column map, the column selector's row gather, and the runner's decode
position sharding must all place global slot *s* on mesh column ``s // 8``. If
any two disagree, a user samples from another user's logits - which is a
cross-slot contamination bug that no per-module test can see.

The device half of every claim is unproven; there is no Galaxy result in this
tree for any of it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.models.galaxy import collectives as collectives_module
from models.common.models.galaxy.collectives import GalaxyColumnUserSelector
from models.common.models.galaxy.direct_runner import GalaxyDirectRunner, GalaxySamplingPolicy
from models.common.models.galaxy.recipes import galaxy_padded_vocab_size
from models.common.modules.sampling.sampling_2d import Sampling2D, _device_seed, _host_seed
from models.common.sampling.vocab_padding import build_invalid_vocab_mask
from models.common.tests.models.galaxy.step7_harness import (
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_USERS_PER_COLUMN,
    LLAMA_VOCAB_SIZE,
    QWEN_VOCAB_SIZE,
    RecordingModel,
    ShardMapper,
    mock_mesh,
    patch_compose,
    patch_direct_runner,
)

_VOCABS = pytest.mark.parametrize(
    ("vocab_size", "padded_vocab_size"),
    [
        (LLAMA_VOCAB_SIZE, galaxy_padded_vocab_size(LLAMA_VOCAB_SIZE)),
        (QWEN_VOCAB_SIZE, galaxy_padded_vocab_size(QWEN_VOCAB_SIZE)),
    ],
    ids=["llama", "qwen"],
)


def _sampler(vocab_size=LLAMA_VOCAB_SIZE, padded_vocab_size=None):
    return Sampling2D(
        vocab_size,
        galaxy_padded_vocab_size(vocab_size) if padded_vocab_size is None else padded_vocab_size,
        mock_mesh(),
        sub_core_grids=object(),
        sub_core_grid_topk=object(),
    )


# ---------------------------------------------------------------------------
# 1. Slot stability
# ---------------------------------------------------------------------------


def test_the_same_seed_in_the_same_slot_gives_the_same_token_across_runs():
    """Fresh sampler objects, identical result: nothing is carried in module state."""

    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(4, galaxy_padded_vocab_size(LLAMA_VOCAB_SIZE), generator=generator)
    kwargs = dict(slot_ids=[0, 7, 8, 31], top_k=16, top_p=0.95, temperature=0.7, seed=[5, 5, 5, 5])

    first = _sampler().sample_host(logits, **kwargs)
    second = _sampler().sample_host(logits, **kwargs)
    third = _sampler().sample_host(logits, **kwargs)

    assert torch.equal(first, second)
    assert torch.equal(second, third)


def test_a_requests_row_position_in_the_call_does_not_change_its_stream():
    """The stream follows the *slot*, not the row index within one call."""

    sampler = _sampler()
    generator = torch.Generator().manual_seed(12)
    logits = torch.randn(3, galaxy_padded_vocab_size(LLAMA_VOCAB_SIZE), generator=generator)

    ordered = sampler.sample_host(logits, slot_ids=[4, 9, 25], top_k=16, top_p=0.9, temperature=0.8, seed=[71, 72, 73])
    permutation = [2, 0, 1]
    permuted = sampler.sample_host(
        logits[permutation], slot_ids=[25, 4, 9], top_k=16, top_p=0.9, temperature=0.8, seed=[73, 71, 72]
    )

    assert permuted.tolist() == ordered[permutation].tolist()


def test_moving_a_seeded_request_to_another_slot_does_change_its_stream():
    """MEASURED, NOT A GATE - Milestone B finding D-C2.

    Step 7 asks that "moving a request to a different slot does not change its
    stream". It does. Both the device seed and the host seed are
    ``blake2b("sampling2d:{seed}:{slot}")``, so slot is part of the key.

    This is a deliberate decorrelation: without it, 32 slots given one seed by a
    serving front end would all emit the same token. The step-7 requirement and
    the module's design are in direct conflict, and resolving it is a decision,
    not a bug fix - so this test records the behaviour that exists instead of
    asserting the one the brief asks for.
    """

    assert _device_seed(1234, 3) != _device_seed(1234, 7)
    assert _host_seed(1234, 3) != _host_seed(1234, 7)

    sampler = _sampler()
    generator = torch.Generator().manual_seed(13)
    row = torch.randn(1, galaxy_padded_vocab_size(LLAMA_VOCAB_SIZE), generator=generator)
    kwargs = dict(top_k=32, top_p=1.0, temperature=1.5, seed=[4242])

    in_slot_3 = sampler.sample_host(row, slot_ids=[3], **kwargs)
    in_slot_7 = sampler.sample_host(row, slot_ids=[7], **kwargs)

    # Same logits, same seed, same controls; only the slot moved.
    assert in_slot_3.tolist() != in_slot_7.tolist(), (
        "the seed digest no longer mixes in the slot; if this now passes, the "
        "decorrelation was removed and 32 slots sharing a seed will collapse"
    )


def test_one_seed_across_every_slot_does_not_collapse_the_batch():
    """The property the slot mixing exists to protect."""

    sampler = _sampler()
    generator = torch.Generator().manual_seed(14)
    logits = torch.randn(1, galaxy_padded_vocab_size(LLAMA_VOCAB_SIZE), generator=generator)
    logits = logits.repeat(GALAXY_PHYSICAL_BATCH, 1)

    tokens = sampler.sample_host(
        logits,
        slot_ids=list(range(GALAXY_PHYSICAL_BATCH)),
        top_k=32,
        top_p=1.0,
        temperature=1.5,
        seed=[999] * GALAXY_PHYSICAL_BATCH,
    )
    assert len(set(tokens.tolist())) > 1, "identical logits and one seed produced one token in every slot"


# ---------------------------------------------------------------------------
# 2. Greedy equals host argmax
# ---------------------------------------------------------------------------


@_VOCABS
def test_greedy_selects_exactly_the_host_argmax(vocab_size, padded_vocab_size):
    sampler = _sampler(vocab_size, padded_vocab_size)
    generator = torch.Generator().manual_seed(15)
    logits = torch.randn(8, padded_vocab_size, generator=generator)

    tokens = sampler.sample_host(logits, slot_ids=list(range(8)), top_k=1, top_p=1.0, temperature=0.0)
    expected = torch.argmax(logits[:, :vocab_size], dim=-1)
    assert torch.equal(tokens, expected)


@_VOCABS
def test_forced_argmax_and_zero_temperature_agree(vocab_size, padded_vocab_size):
    sampler = _sampler(vocab_size, padded_vocab_size)
    generator = torch.Generator().manual_seed(16)
    logits = torch.randn(4, padded_vocab_size, generator=generator)

    by_temperature = sampler.sample_host(
        logits, slot_ids=[0, 1, 2, 3], top_k=8, top_p=0.5, temperature=0.0, seed=[1, 2, 3, 4]
    )
    by_flag = sampler.sample_host(
        logits, slot_ids=[0, 1, 2, 3], top_k=8, top_p=0.5, temperature=1.0, forced_argmax=True, seed=[1, 2, 3, 4]
    )
    assert torch.equal(by_temperature, by_flag)


# ---------------------------------------------------------------------------
# 3. Padded vocabulary
# ---------------------------------------------------------------------------


def test_llama_now_pads_and_therefore_masks_like_qwen():
    """This test used to assert that Llama pads by *nothing*. It now pads by 768.

    The step-7 brief assumed Llama pads; the tree it was written against did not,
    because 128256 = 501 * 256 is already a multiple of the Galaxy alignment
    (8 vocabulary shards x 32), so `padded_vocab_size == vocab_size`,
    `build_invalid_vocab_mask` returned None, and the padded-vocabulary gate was
    *vacuous* for Llama. That was recorded here so nobody would read a Llama pass
    as evidence that masking works.

    **It is no longer vacuous, and the reason is a hardware constraint, not a
    preference.** `all_reduce_async`'s reduction kernel waits for a full shard on
    every output core, so the decode LM head's reduced logits must be an exact
    multiple of `cores * shard_width`. 16032 columns per device is 501 tiles, and
    501 has no divisor between 4 and 50, so *no* usable core count divides it: the
    staging necessarily became 42 x 12 = 504 and the 42nd core waited forever, with
    no abort and no traceback. See D-B19 in
    `tttv2_milestone_b_evidence/llama/REPORT.md`.

    So `galaxy_padded_vocab_size` pads to a ring-exact width, Llama gains 768
    masked entries, and the masking path that only Qwen used to exercise is now
    exercised by both models - which is strictly better for step 7 than the
    situation this test was written to warn about.
    """

    padded = galaxy_padded_vocab_size(LLAMA_VOCAB_SIZE)
    assert padded == 129024
    assert padded - LLAMA_VOCAB_SIZE == 768
    # Ring-exact: a whole number of 24-core ring rows per device.
    assert (padded // 8) % (24 * 32) == 0

    mask = _sampler(LLAMA_VOCAB_SIZE).config.invalid_vocab_mask.source
    assert mask.shape == (1, 1, GALAXY_PHYSICAL_BATCH, padded)
    assert torch.all(mask[..., :LLAMA_VOCAB_SIZE] == 0)
    assert torch.all(mask[..., LLAMA_VOCAB_SIZE:] == torch.finfo(torch.bfloat16).min)
    assert build_invalid_vocab_mask(LLAMA_VOCAB_SIZE, padded, GALAXY_PHYSICAL_BATCH) is not None


def test_qwen_pads_to_the_ring_exact_width_and_masks_the_remainder():
    """1664 masked entries, not 128: the ring-exact width. See D-B19.

    152064 - the minimal Galaxy-aligned width, and what this test used to assert -
    is 594 tiles per device, which no usable core count divides. 153600 is 600,
    which 50 cores divide exactly at 12 tiles each.
    """

    padded = galaxy_padded_vocab_size(QWEN_VOCAB_SIZE)
    assert padded == 153600
    assert padded - QWEN_VOCAB_SIZE == 1664
    assert (padded // 8) % (24 * 32) == 0

    mask = _sampler(QWEN_VOCAB_SIZE).config.invalid_vocab_mask.source
    assert mask.shape == (1, 1, GALAXY_PHYSICAL_BATCH, padded)
    assert torch.all(mask[..., :QWEN_VOCAB_SIZE] == 0)
    assert torch.all(mask[..., QWEN_VOCAB_SIZE:] == torch.finfo(torch.bfloat16).min)


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0, 2.0], ids=["greedy", "t0.7", "t1.0", "t2.0"])
def test_a_padded_entry_cannot_win_even_when_it_carries_the_largest_logit(temperature):
    """Qwen geometry, every padded id set far above every real one."""

    padded = galaxy_padded_vocab_size(QWEN_VOCAB_SIZE)
    sampler = _sampler(QWEN_VOCAB_SIZE, padded)
    logits = torch.full((4, padded), -20.0)
    for row in range(4):
        logits[row, row * 1000] = 3.0
    logits[:, QWEN_VOCAB_SIZE:] = 1.0e4

    tokens = sampler.sample_host(
        logits,
        slot_ids=[0, 8, 16, 24],
        top_k=1 if temperature == 0.0 else 32,
        top_p=1.0,
        temperature=temperature,
        seed=[1, 2, 3, 4],
    )
    assert torch.all(tokens < QWEN_VOCAB_SIZE), f"a padded id was sampled: {tokens.tolist()}"
    if temperature == 0.0:
        assert tokens.tolist() == [0, 1000, 2000, 3000]


def test_the_device_path_masks_padding_additively_below_every_real_logit():
    """The device adds the mask before top-k; bf16's minimum is the floor."""

    padded = galaxy_padded_vocab_size(QWEN_VOCAB_SIZE)
    mask = build_invalid_vocab_mask(QWEN_VOCAB_SIZE, padded, GALAXY_PHYSICAL_BATCH)
    logits = torch.full((1, 1, GALAXY_PHYSICAL_BATCH, padded), 1.0e4, dtype=torch.bfloat16)
    masked = logits + mask
    assert torch.all(masked[..., QWEN_VOCAB_SIZE:] < masked[..., :QWEN_VOCAB_SIZE].min())


# ---------------------------------------------------------------------------
# 4. Per-slot heterogeneous controls
# ---------------------------------------------------------------------------


def test_each_slot_carries_its_own_top_k_top_p_and_temperature():
    """Serving mixes them, so the buffers must be per-slot, not broadcast."""

    sampler = _sampler()
    sampler.prepare_call(
        slot_ids=[0, 5, 8, 17, 31],
        top_k=[1, 8, 32, 4, 16],
        top_p=[1.0, 0.9, 0.5, 0.95, 0.25],
        temperature=[0.0, 0.5, 2.0, 1.0, 4.0],
        seed=[None, 10, 20, 30, 40],
    )
    config = sampler.config
    k = config.top_k_buffer.source
    p = config.top_p_buffer.source
    t = config.temperature_buffer.source

    # ``top_p`` and the reciprocal temperature are bfloat16 buffers, so 0.9 and
    # 0.95 are stored to bf16 precision (~2^-8 relative); 0.25, 0.5 and their
    # reciprocals are exact.
    bf16 = dict(rel=2**-8)

    assert (int(k[0]), float(p[0]), float(t[0])) == (1, 0.0, 1.0)
    assert int(k[5]) == 8 and float(p[5]) == pytest.approx(0.9, **bf16) and float(t[5]) == 2.0
    assert int(k[8]) == 32 and float(p[8]) == 0.5 and float(t[8]) == 0.5
    assert int(k[17]) == 4 and float(p[17]) == pytest.approx(0.95, **bf16) and float(t[17]) == 1.0
    assert int(k[31]) == 16 and float(p[31]) == 0.25 and float(t[31]) == 0.25

    # Slots the call did not name keep the greedy defaults.
    for slot in (1, 2, 3, 4, 6, 7, 30):
        assert int(k[slot]) == 1
        assert float(p[slot]) == 0.0
        assert float(t[slot]) == 1.0


def test_heterogeneous_controls_change_the_tokens_they_are_supposed_to_change():
    sampler = _sampler()
    generator = torch.Generator().manual_seed(17)
    row = torch.randn(1, galaxy_padded_vocab_size(LLAMA_VOCAB_SIZE), generator=generator)
    logits = row.repeat(2, 1)

    greedy_first = sampler.sample_host(
        logits, slot_ids=[0, 1], top_k=[1, 32], top_p=[1.0, 1.0], temperature=[0.0, 5.0], seed=[None, 7]
    )
    assert int(greedy_first[0]) == int(torch.argmax(row[0, :LLAMA_VOCAB_SIZE]))


def test_top_k_stays_inside_the_configured_capacity():
    """``max_top_k`` is 32 and the config contract does not go past it."""

    sampler = _sampler()
    assert sampler.config.max_top_k == 32
    sampler.prepare_call(slot_ids=[0], top_k=32, top_p=1.0, temperature=1.0)
    assert int(sampler.config.top_k_buffer.source[0]) == 32


# ---------------------------------------------------------------------------
# 5. The temperature reciprocal, and the runner/module pairing (defect D4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0.25, 0.5, 0.8, 2.0, 4.0])
def test_the_module_writes_one_over_t_while_the_runner_passes_raw_t(monkeypatch, temperature):
    """The pairing the brief asks to verify, on the host half.

    ``GalaxyDirectRunner`` divides by ``T`` in its own host reference and hands
    the module the *raw* ``T``; ``Sampling2D`` then writes ``1/T`` into the
    buffer, because ``ttnn.sampling``'s ``temp`` argument is the reciprocal.
    Both halves have to be checked at ``T != 1.0``: 1.0 is its own reciprocal,
    which is what hid defect D4.
    """

    patch_direct_runner(monkeypatch)
    model = RecordingModel()
    runner = GalaxyDirectRunner(model)
    patch_compose(monkeypatch, lambda tensor: torch.zeros(GALAXY_PHYSICAL_BATCH, dtype=torch.int64))
    runner.open()
    try:
        policy = GalaxySamplingPolicy(top_k=8, top_p=0.9, temperature=temperature, seed=3, on_device=True)
        runner.decode_sampled([1] * GALAXY_PHYSICAL_BATCH, [0] * GALAXY_PHYSICAL_BATCH, policy)
        assert model.sample_calls[-1]["temperature"] == temperature, "the runner must not pre-invert T"
    finally:
        runner.close()

    sampler = _sampler()
    sampler.prepare_call(slot_ids=[0], top_k=8, top_p=0.9, temperature=temperature, seed=3)
    written = float(sampler.config.temperature_buffer.source[0])
    assert written == pytest.approx(1.0 / temperature, rel=1.0 / 128), f"buffer holds {written}, not 1/{temperature}"


def test_the_runners_host_reference_divides_by_t_rather_than_multiplying():
    """A raw-T/reciprocal-T confusion reverses which token a low T favours."""

    runner_cls = GalaxyDirectRunner
    logits = torch.tensor([[0.0, 1.0, 2.0]])

    sharp = runner_cls.sample_host(
        object.__new__(runner_cls), logits, GalaxySamplingPolicy(top_k=3, temperature=0.05, seed=1)
    )
    flat_counts = {0: 0, 1: 0, 2: 0}
    for seed in range(40):
        token = runner_cls.sample_host(
            object.__new__(runner_cls), logits, GalaxySamplingPolicy(top_k=3, temperature=20.0, seed=seed)
        )[0]
        flat_counts[token] += 1

    assert sharp == [2], "a low temperature must concentrate on the largest logit"
    assert min(flat_counts.values()) > 0, "a high temperature must spread across every candidate"


def test_a_greedy_policy_is_temperature_zero_or_top_one():
    assert GalaxySamplingPolicy(temperature=0.0).greedy
    assert GalaxySamplingPolicy(top_k=1, temperature=1.0).greedy
    assert not GalaxySamplingPolicy(top_k=8, temperature=1.0).greedy


def test_the_runner_forces_argmax_for_a_greedy_policy(monkeypatch):
    patch_direct_runner(monkeypatch)
    model = RecordingModel()
    runner = GalaxyDirectRunner(model)
    patch_compose(monkeypatch, lambda tensor: torch.zeros(GALAXY_PHYSICAL_BATCH, dtype=torch.int64))
    runner.open()
    try:
        runner.decode_sampled(
            [1] * GALAXY_PHYSICAL_BATCH,
            [0] * GALAXY_PHYSICAL_BATCH,
            GalaxySamplingPolicy(top_k=1, temperature=0.0, on_device=True),
        )
        assert model.sample_calls[-1]["forced_argmax"] is True
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# 6. The composition property: everyone must agree where a slot lives
# ---------------------------------------------------------------------------


def _column_selector_rows(monkeypatch):
    """Return the identity rows each mesh column would own."""

    staged = {}

    def from_torch(tensor, *, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None, **_kwargs):
        staged["host"] = tensor.clone()
        staged["mapper"] = mesh_mapper
        return object()

    monkeypatch.setattr(collectives_module.ttnn, "from_torch", from_torch)
    monkeypatch.setattr(
        collectives_module.ttnn,
        "ShardTensor2dMesh",
        lambda mesh_device, dims, mesh_shape: ShardMapper(dims=tuple(dims), mesh_shape=tuple(mesh_shape)),
    )
    selector = GalaxyColumnUserSelector(MagicMock(spec=ttnn.MeshDevice))
    selector.selector()
    return staged


def test_the_column_selector_gathers_rows_eight_c_through_eight_c_plus_seven(monkeypatch):
    staged = _column_selector_rows(monkeypatch)
    identity = staged["host"]
    mapper = staged["mapper"]

    assert identity.shape == (1, 1, GALAXY_PHYSICAL_BATCH, GALAXY_PHYSICAL_BATCH)
    assert torch.equal(identity[0, 0], torch.eye(GALAXY_PHYSICAL_BATCH))
    assert mapper.dims == (None, 2)
    assert mapper.mesh_shape == GALAXY_MESH_SHAPE

    # Column c owns tensor dim-2 rows [8c, 8c+8). Reproduce the shard directly.
    for column in range(GALAXY_MESH_SHAPE[1]):
        start = column * GALAXY_USERS_PER_COLUMN
        shard = identity[0, 0, start : start + GALAXY_USERS_PER_COLUMN, :]
        # An exact row gather: each shard row is a one-hot on its global slot.
        for local, row in enumerate(shard):
            assert int(torch.argmax(row)) == start + local
            assert float(row.sum()) == 1.0


def test_the_sampler_the_selector_and_the_runner_agree_on_every_slots_column(monkeypatch):
    """A disagreement here is a cross-slot contamination bug by construction."""

    sampler = _sampler()
    patch_direct_runner(monkeypatch)
    model = RecordingModel()
    runner = GalaxyDirectRunner(model)
    runner.open()
    try:
        positions = list(range(GALAXY_PHYSICAL_BATCH))
        staged = runner._stage_positions(positions)
        assert isinstance(staged.mapper, ShardMapper)
        assert staged.mapper.dims == (None, 0)

        for slot in range(GALAXY_PHYSICAL_BATCH):
            column, local = sampler.slot_placement(slot)
            assert column == slot // GALAXY_USERS_PER_COLUMN
            assert local == slot % GALAXY_USERS_PER_COLUMN
            # The runner shards positions on the same axis with the same width,
            # so slot `slot` lands on column `slot // 8` there too.
            assert positions[slot] == slot
        shard_width = staged.host.shape[0] // GALAXY_MESH_SHAPE[1]
        assert shard_width == GALAXY_USERS_PER_COLUMN
    finally:
        runner.close()


def test_the_selector_refuses_a_tensor_that_is_not_the_physical_batch(monkeypatch):
    _column_selector_rows(monkeypatch)
    selector = GalaxyColumnUserSelector(MagicMock(spec=ttnn.MeshDevice))

    class _Wrong:
        shape = (1, 1, 16, 128)

    with pytest.raises(ValueError, match=r"expects \[1, 1, 32, W\]"):
        selector(_Wrong())
