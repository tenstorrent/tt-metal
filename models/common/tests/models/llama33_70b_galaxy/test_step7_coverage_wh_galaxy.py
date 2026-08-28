# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step-7 device coverage for Galaxy Llama-3.3-70B: the gaps.

``test_full_model_wh_galaxy.py`` already covers full-model prefill/decode,
teacher-forced accuracy, batch-32 slot isolation, prefix-cached prefill, the
4K/32K/128K smokes and repeated requests. ``demo.py`` covers the batch-1 and
batch-32 demos, concat-32 at 128 and greedy device sampling.

This file covers what neither of them reaches, and what the step-7 brief asks
for by name:

* **paged versus contiguous** - the same prompt through both cache layouts, at
  PCC >= 0.99. Nothing in this tree has ever compared them;
* **late capacity resolution** - a paged geometry installed after the model was
  constructed, then bound and used;
* **concat-32 at active batches 16, 31 and 32**, and at sequence lengths 128
  through 2048 in ascending order, checking that a padded row neither changes
  an active row's KV nor contributes a logit;
* **prefix-cached then normal, and a mix across slots** - the interaction, not
  just the feature;
* **device sampling** - slot stability, greedy against host argmax, and per-slot
  heterogeneous top-k/top-p/temperature.

* **the padded vocabulary can never be sampled.** This was written as "not
  applicable to Llama", on the premise that 128256 is already a multiple of
  ``8 vocab shards * 32``. That premise is **false at this tree**:
  ``galaxy_padded_vocab_size(128256) == 129024``, because the width must also be
  a whole number of 24-core ring rows per device (D-B19), so Llama carries **768**
  invalid ids. The case is therefore live for Llama too, and it is below.

This file was written on 2026-08-27 with eleven of the mesh's 32 boards off the
PCIe bus, so ``ttnn`` could not open a cluster at all, and it was committed
unexecuted. It was **first executed on 2026-08-27/28 by `mb-coverage` attempt
2**; what each case measured is in
``tttv2_milestone_b_evidence/coverage/REPORT.md`` §A2.

Run::

    pytest models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py -v
"""

from __future__ import annotations

import gc
import os
from typing import Any

import pytest
import torch

import ttnn
from models.common.models.galaxy.direct_runner import GalaxyDirectRunner, GalaxySamplingPolicy
from models.common.models.galaxy.kv_contract import GalaxyPagedAttentionConfig
from models.common.models.llama33_70b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, from_pretrained
from models.common.tests.models.galaxy.galaxy_hardware import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    hf_config_or_skip,
    load_reference_tokens,
)

_REFERENCE_NAME = "Llama-3.3-70B-Instruct"
_BLOCK_SIZE = 32
_PAGED_VS_CONTIGUOUS_PCC = 0.99
_GREEDY = GalaxySamplingPolicy(top_k=1, temperature=0.0)

#: Ascending, as the brief requires: qualify 128, then work outwards.
_BATCHED_LENGTHS = (128, 256, 512, 1024, 2048)


def _hf_model() -> str:
    return os.getenv("LLAMA33_70B_HF_MODEL", DEFAULT_HF_MODEL)


def _layers() -> int | None:
    value = os.getenv("LLAMA33_70B_GALAXY_TEST_LAYERS")
    return int(value) if value else None


def _paged_config(*, context: int, active_slots: int) -> GalaxyPagedAttentionConfig:
    blocks_per_user = -(-context // _BLOCK_SIZE)
    sinks = GALAXY_PHYSICAL_BATCH - active_slots
    return GalaxyPagedAttentionConfig(block_size=_BLOCK_SIZE, max_num_blocks=blocks_per_user * active_slots + sinks)


def _load(mesh_device: ttnn.MeshDevice, **overrides: Any):
    hf_model = _hf_model()
    hf_config_or_skip(hf_model)
    kwargs: dict[str, Any] = dict(
        hf_model=hf_model,
        max_seq_len=2048,
        prefill_sequence_lengths=(128,),
        n_layers=_layers(),
    )
    kwargs.update(overrides)
    return from_pretrained(mesh_device, **kwargs)


def _close(handle: Any) -> None:
    try:
        handle.close()
    finally:
        del handle
        gc.collect()


def _pcc(expected: torch.Tensor, actual: torch.Tensor, threshold: float):
    from models.common.utility_functions import comp_pcc

    return comp_pcc(expected.unsqueeze(0).float(), actual.unsqueeze(0).float(), threshold)


def _prompt(length: int) -> list[int]:
    reference_tokens, _ = load_reference_tokens(_REFERENCE_NAME)
    if len(reference_tokens) <= length:
        pytest.skip(f"reference sequence has {len(reference_tokens)} tokens, need more than {length}")
    return [int(value) for value in reference_tokens[:length]]


def _distinct_rows(length: int, count: int) -> list[list[int]]:
    """``count`` distinct prompts of ``length`` tokens, from one reference text.

    The reference file holds 1024 tokens, so the straight window walk below runs
    out at ``length + count > 1024`` - which is every concat-32 length at or
    above 1024, exactly the lengths the step-7 brief asks for last. Rather than
    *skip* those (a skip is not a result), fall back to cyclic windows: rotation
    ``r`` of a 1024-token text is still real text and still distinct from
    rotation ``r'``, and the property under test - that row ``i``'s KV and logits
    do not depend on row ``j`` - does not care whether the rows share tokens.
    `mb-coverage` attempt 2 added the fallback; the exact-window path is
    unchanged, so every result taken before it is comparable.
    """

    reference_tokens, _ = load_reference_tokens(_REFERENCE_NAME)
    source = [int(value) for value in reference_tokens]
    if len(source) < length + count:
        if len(source) < count:
            pytest.skip(f"reference sequence has {len(source)} tokens, need at least {count}")
        return [[source[(offset + index) % len(source)] for index in range(length)] for offset in range(count)]
    return [source[offset : offset + length] for offset in range(count)]


_GALAXY = pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
_PARAMS = pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)


# ---------------------------------------------------------------------------
# Area 1: paged KV
# ---------------------------------------------------------------------------


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable(mesh_device: ttnn.MeshDevice):
    """Two different block allocations must serve the same request identically.

    **This case was written as "paged versus contiguous" and it could not be.**
    `mb-coverage` attempt 2 measured why: ``from_pretrained`` does not accept
    "no paged cache". `hf_adaptor.from_pretrained` does

        paged = paged_attention_config or default_paged_attention_config(params)

    so ``paged_attention_config=None`` installs a *default* pool of
    ``ceil(max_seq_len / 32) * max_batch_size`` blocks - which, at
    ``max_seq_len=2048`` and batch 32, is the identical 2048-block pool the
    "paged" arm asked for. The original test therefore compared a paged cache
    against **the same paged cache** and would have passed while proving nothing.
    That is finding D-C4; the contiguous path exists in `Attention2D` and in the
    KV contract, and is simply not reachable through either model's adaptor.

    What is reachable, and what this measures instead: the same 32 requests
    through **two different pools** - the default 2048-block pool and a
    4096-block one, which gives every slot a different run of block ids - must
    produce the same prefill logits and the same decode logits, at PCC >= 0.99.
    A page table that is read as anything other than "slot u owns this run of
    blocks" moves under that change; correct addressing does not.
    """

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    pools = {
        "default-2048": None,
        "explicit-4096": _paged_config(context=4096, active_slots=GALAXY_PHYSICAL_BATCH),
    }
    outputs = {}
    for name, paged in pools.items():
        handle = _load(mesh_device, max_seq_len=4096 if paged else 2048, paged_attention_config=paged)
        try:
            specs = handle.model.kv_specs
            configs = {spec.paged_attention_config for spec in specs}
            assert len(configs) == 1
            resolved = next(iter(configs))
            assert resolved is not None, (
                "D-C4 has been fixed: from_pretrained now builds a contiguous cache. "
                "Restore the paged-versus-contiguous comparison this case was written for."
            )
            print(f"[pool] {name}: block_size={resolved.block_size} max_num_blocks={resolved.max_num_blocks}")
            with GalaxyDirectRunner(handle.model) as runner:
                prefill = torch.cat([runner.prefill_row(row, slot=slot) for slot, row in enumerate(rows)], dim=0)
                tokens = [1] * GALAXY_PHYSICAL_BATCH
                positions = [128] * GALAXY_PHYSICAL_BATCH
                outputs[name] = (prefill.clone(), runner.decode_logits(tokens, positions).clone())
        finally:
            _close(handle)

    first, second = outputs["default-2048"], outputs["explicit-4096"]
    for index, stage in enumerate(("prefill", "decode")):
        for slot in range(GALAXY_PHYSICAL_BATCH):
            passed, message = _pcc(first[index][slot], second[index][slot], _PAGED_VS_CONTIGUOUS_PCC)
            assert passed, f"{stage} slot {slot}, 2048-block pool vs 4096-block pool: {message}"


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_paged_capacity_resolved_after_construction_serves_a_request(mesh_device: ttnn.MeshDevice, expect_error):
    """Late capacity resolution, end to end.

    The model is built with no paged geometry at all; the block pool is chosen
    afterwards, installed with ``configure_paged_attention``, and only then does
    the runner allocate and bind. Reconfiguring while bound must be refused.
    """

    prompt = _prompt(128)
    handle = _load(mesh_device, paged_attention_config=None)
    try:
        model = handle.model
        # D-C4: `paged_attention_config=None` does not mean "contiguous" - the
        # adaptor substitutes `default_paged_attention_config`. So the starting
        # point of late capacity resolution is a *default* pool, not no pool, and
        # the reachable claim is that the geometry can still be replaced before
        # anything is bound. Attempt 2 corrected this from the original
        # `is None` assertion, which failed for that reason and not because late
        # resolution is broken.
        installed = {spec.paged_attention_config for spec in model.kv_specs}
        assert len(installed) == 1 and next(iter(installed)) is not None
        default_pool = next(iter(installed))
        print(f"[pool] as constructed: {default_pool}")

        pool = _paged_config(context=4096, active_slots=32)
        assert pool != default_pool, "the replacement pool must differ from the default to prove anything"
        model.configure_paged_attention(block_size=pool.block_size, max_num_blocks=pool.max_num_blocks)
        assert all(spec.paged_attention_config == pool for spec in model.kv_specs)

        with GalaxyDirectRunner(model) as runner:
            with expect_error(RuntimeError, "cannot be reconfigured"):
                model.configure_paged_attention(block_size=32, max_num_blocks=pool.max_num_blocks + 32)
            logits = runner.prefill_row(prompt, slot=0)
            assert 0 <= int(torch.argmax(logits[0])) < model.vocab_size

        # Unbound again: the geometry may be replaced.
        model.configure_paged_attention(block_size=32, max_num_blocks=pool.max_num_blocks + 32)
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_a_write_for_one_user_never_appears_in_another_users_blocks(mesh_device: ttnn.MeshDevice):
    """Cross-slot contamination, measured rather than argued.

    Slot 0 is prefilled with prompt A while every other slot holds prompt B.
    The run is repeated with slot 0 holding prompt B as well. If any slot's
    decode logits move between the two runs because of what slot 0 held, some
    slot read slot 0's blocks.
    """

    rows = _distinct_rows(128, 2)
    handle = _load(mesh_device, paged_attention_config=_paged_config(context=2048, active_slots=32))
    try:
        observed = []
        for first_row in (rows[0], rows[1]):
            with GalaxyDirectRunner(handle.model) as runner:
                prompts = [first_row] + [rows[1]] * (GALAXY_PHYSICAL_BATCH - 1)
                for slot, prompt in enumerate(prompts):
                    runner.prefill_row(prompt, slot=slot)
                tokens = [1] * GALAXY_PHYSICAL_BATCH
                positions = [len(rows[1])] * GALAXY_PHYSICAL_BATCH
                observed.append(runner.decode_logits(tokens, positions).clone())

        moved = [
            slot for slot in range(1, GALAXY_PHYSICAL_BATCH) if not torch.equal(observed[0][slot], observed[1][slot])
        ]
        assert not moved, f"slots {moved} changed when only slot 0's prompt changed"
        assert not torch.equal(observed[0][0], observed[1][0]), "slot 0 ignored its own prompt"
    finally:
        _close(handle)


# ---------------------------------------------------------------------------
# Area 2: concat-32 physical prefill
# ---------------------------------------------------------------------------


@_PARAMS
@_GALAXY
@pytest.mark.parametrize("length", _BATCHED_LENGTHS, ids=[f"len{value}" for value in _BATCHED_LENGTHS])
@torch.no_grad()
def test_llama_concat32_matches_sequential_prefill_at_each_length(mesh_device: ttnn.MeshDevice, length: int):
    """Qualify 128 first; only then the longer recipes.

    Each length is a separate resolved recipe and a separate set of collective
    resources, so a pass at 128 says nothing about 2048.
    """

    rows = _distinct_rows(length, GALAXY_PHYSICAL_BATCH)
    handle = _load(
        mesh_device,
        max_seq_len=max(2048, length),
        prefill_sequence_lengths=(length,),
        batched_prefill_sequence_lengths=(length,),
        paged_attention_config=_paged_config(context=max(2048, length), active_slots=32),
    )
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            sequential = torch.cat([runner.prefill_row(row, slot=slot) for slot, row in enumerate(rows)], dim=0)
        with GalaxyDirectRunner(handle.model) as runner:
            batched = runner.prefill_batched(rows)

        mismatched = [
            slot
            for slot in range(GALAXY_PHYSICAL_BATCH)
            if int(torch.argmax(sequential[slot])) != int(torch.argmax(batched[slot]))
        ]
        assert not mismatched, f"concat-32 at length {length} diverged from sequential in slots {mismatched}"
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@pytest.mark.parametrize("active", [16, 31, 32], ids=["active16", "active31", "active32"])
@torch.no_grad()
def test_llama_concat32_padded_rows_change_no_active_rows_logits(mesh_device: ttnn.MeshDevice, active: int):
    """Padding inactive rows must not write KV or return logits for them.

    The concatenated prefill always runs 32 physical rows. Rows ``active..31``
    carry a one-token filler. Changing that filler must not move any of the
    first ``active`` rows' logits.
    """

    length = 128
    rows = _distinct_rows(length, active)
    handle = _load(
        mesh_device,
        batched_prefill_sequence_lengths=(length,),
        paged_attention_config=_paged_config(context=2048, active_slots=32),
    )
    try:
        outputs = []
        for filler in (rows[0][:1], rows[-1][:1]):
            padded = list(rows) + [list(filler) for _ in range(GALAXY_PHYSICAL_BATCH - active)]
            with GalaxyDirectRunner(handle.model) as runner:
                outputs.append(runner.prefill_batched(padded).clone())

        moved = [slot for slot in range(active) if not torch.equal(outputs[0][slot], outputs[1][slot])]
        assert not moved, f"active slots {moved} moved when only the padding rows changed"
    finally:
        _close(handle)


# ---------------------------------------------------------------------------
# Area 3: prefix-cached and chunked prefill, and the interaction
# ---------------------------------------------------------------------------


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_a_prefix_cached_request_then_a_normal_one(mesh_device: ttnn.MeshDevice):
    """The interaction, not just the feature.

    Slot 0 is chunk-prefilled; slot 1 is then prefilled normally and must agree
    with the same request run on a runner that never did a chunked prefill.
    """

    chunked_prompt = _prompt(256)
    plain_prompt = _prompt(128)
    handle = _load(
        mesh_device,
        prefill_sequence_lengths=(128,),
        chunked_prefill_sequence_lengths=(128,),
        paged_attention_config=_paged_config(context=2048, active_slots=32),
    )
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            baseline = runner.prefill_row(plain_prompt, slot=1).clone()
        with GalaxyDirectRunner(handle.model) as runner:
            runner.prefill_chunked(chunked_prompt, slot=0, chunk_length=128)
            after = runner.prefill_row(plain_prompt, slot=1).clone()

        passed, message = _pcc(baseline[0], after[0], _PAGED_VS_CONTIGUOUS_PCC)
        assert passed, f"a preceding prefix-cached request changed the next plain request: {message}"
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_prefix_cached_and_plain_requests_mixed_across_slots(mesh_device: ttnn.MeshDevice):
    """A mix of both in one batch, then one decode over all of them."""

    chunked_prompt = _prompt(256)
    plain_prompt = _prompt(128)
    handle = _load(
        mesh_device,
        prefill_sequence_lengths=(128,),
        chunked_prefill_sequence_lengths=(128,),
        paged_attention_config=_paged_config(context=2048, active_slots=32),
    )
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            for slot in range(GALAXY_PHYSICAL_BATCH):
                if slot % 2:
                    runner.prefill_chunked(chunked_prompt, slot=slot, chunk_length=128)
                else:
                    runner.prefill_row(plain_prompt, slot=slot)

            tokens = [1] * GALAXY_PHYSICAL_BATCH
            positions = [len(plain_prompt) if slot % 2 == 0 else len(chunked_prompt) for slot in range(32)]
            logits = runner.decode_logits(tokens, positions)

        even = [logits[slot] for slot in range(0, GALAXY_PHYSICAL_BATCH, 2)]
        odd = [logits[slot] for slot in range(1, GALAXY_PHYSICAL_BATCH, 2)]
        for index, row in enumerate(even[1:], start=1):
            assert torch.equal(even[0], row), f"plain slot {2 * index} disagreed with plain slot 0"
        for index, row in enumerate(odd[1:], start=1):
            assert torch.equal(odd[0], row), f"cached slot {2 * index + 1} disagreed with cached slot 1"
        assert not torch.equal(even[0], odd[0]), "the two request kinds produced identical logits"
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_chunked_prefill_matches_a_single_uncached_prefill(mesh_device: ttnn.MeshDevice):
    """The gate: prefix-cached output matches uncached execution.

    A 256-token prompt as two 128-token chunks, against the same 256 tokens as
    one prefill. This differs from ``test_full_model_wh_galaxy``'s version in
    that it also decodes afterwards, so the cache the chunks *wrote* is read.
    """

    prompt = _prompt(256)
    handle = _load(
        mesh_device,
        prefill_sequence_lengths=(128, 256),
        chunked_prefill_sequence_lengths=(128,),
        paged_attention_config=_paged_config(context=2048, active_slots=32),
    )
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            uncached = runner.prefill_row(prompt, slot=0, sequence_length=256).clone()
            tokens = [0] * GALAXY_PHYSICAL_BATCH
            positions = [0] * GALAXY_PHYSICAL_BATCH
            tokens[0] = int(torch.argmax(uncached[0]))
            positions[0] = len(prompt)
            uncached_decode = runner.decode_logits(tokens, positions)[0].clone()

        with GalaxyDirectRunner(handle.model) as runner:
            cached = runner.prefill_chunked(prompt, slot=0, chunk_length=128).clone()
            tokens[0] = int(torch.argmax(cached[0]))
            cached_decode = runner.decode_logits(tokens, positions)[0].clone()

        for name, expected, actual in (
            ("prefill", uncached[0], cached[0]),
            ("decode", uncached_decode, cached_decode),
        ):
            passed, message = _pcc(expected, actual, _PAGED_VS_CONTIGUOUS_PCC)
            assert passed, f"chunked {name} did not match uncached: {message}"
    finally:
        _close(handle)


# ---------------------------------------------------------------------------
# Area 4: device sampling
# ---------------------------------------------------------------------------


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_device_greedy_sampling_equals_host_argmax(mesh_device: ttnn.MeshDevice):
    """Every slot, not just slot 0: the column selector must gather correctly."""

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    handle = _load(mesh_device, paged_attention_config=_paged_config(context=2048, active_slots=32))
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            for slot, row in enumerate(rows):
                runner.prefill_row(row, slot=slot)
            tokens = [1] * GALAXY_PHYSICAL_BATCH
            positions = [128] * GALAXY_PHYSICAL_BATCH

            host_logits = runner.decode_logits(tokens, positions)
            expected = torch.argmax(host_logits[:, : handle.model.vocab_size], dim=-1)
            sampled = runner.decode_sampled(tokens, positions, _GREEDY)

        mismatched = [slot for slot in range(GALAXY_PHYSICAL_BATCH) if int(sampled[slot]) != int(expected[slot])]
        assert not mismatched, f"device greedy disagreed with host argmax in slots {mismatched}"
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(GalaxySamplingPolicy(top_k=1, temperature=0.0, on_device=True), id="greedy"),
        pytest.param(GalaxySamplingPolicy(top_k=32, top_p=1.0, temperature=1.5, seed=7, on_device=True), id="t1.5"),
        pytest.param(GalaxySamplingPolicy(top_k=8, top_p=0.9, temperature=0.5, seed=7, on_device=True), id="t0.5"),
    ],
)
@torch.no_grad()
def test_llama_no_padded_vocabulary_id_is_ever_sampled(mesh_device: ttnn.MeshDevice, policy):
    """Llama pads 128256 up to 129024; those 768 ids are not tokens.

    Added by `mb-coverage` attempt 2. The file was written asserting the
    opposite - that Llama has no padding at all - and at this tree that is
    false: ``galaxy_padded_vocab_size`` rounds the *per-device* width up to a
    whole number of 24-core ring rows (D-B19), which 128256 // 8 = 16032 is not.
    An invalid id winning is a correctness bug, not a rounding issue, so this
    runs at three policies. ``T == 1.0`` is skipped on purpose - it is its own
    reciprocal and would hide a temperature inversion (defect D4).
    """

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    handle = _load(mesh_device, paged_attention_config=_paged_config(context=2048, active_slots=32))
    try:
        vocab_size = handle.model.vocab_size
        padded_vocab_size = handle.model.padded_vocab_size
        assert padded_vocab_size > vocab_size, "this test is only meaningful for a padded vocabulary"
        print(f"[vocab] llama vocab={vocab_size} padded={padded_vocab_size} invalid={padded_vocab_size - vocab_size}")

        with GalaxyDirectRunner(handle.model) as runner:
            for slot, row in enumerate(rows):
                runner.prefill_row(row, slot=slot)
            tokens = [1] * GALAXY_PHYSICAL_BATCH
            positions = [128] * GALAXY_PHYSICAL_BATCH
            sampled = runner.decode_sampled(tokens, positions, policy)

        invalid = [(slot, int(value)) for slot, value in enumerate(sampled) if int(value) >= vocab_size]
        assert not invalid, f"padded vocabulary ids were sampled: {invalid}"
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_a_near_zero_temperature_collapses_onto_the_host_argmax(mesh_device: ttnn.MeshDevice):
    """The one device check for defect D4's temperature inversion.

    ``ttnn.sampling``'s ``temp`` argument is the **reciprocal** temperature and
    ``Sampling2D`` writes ``1 / T``; ``direct_runner.sample_host`` divides by
    ``T``. The step-7 brief asks for that pairing to be verified on device rather
    than assumed, and notes why the obvious test cannot do it: ``T == 1.0`` is its
    own reciprocal, and every seeded comparison against the host is confounded
    because ``_device_seed`` and ``_host_seed`` are deliberately different
    digests, so device and host streams are not expected to match token for token.

    This case avoids both traps. Sample **stochastically** - ``top_k=32``, no
    forced argmax - at ``T = 0.02``. Under the correct convention the device gets
    ``temp = 50`` and the distribution is so peaked that every slot must land on
    its argmax. Under the inverted convention it would get ``temp = 0.02``, which
    flattens the distribution to near-uniform over 32 candidates, and 32 slots
    agreeing with argmax would be a ``1 / 32**32`` event. So the assertion below
    fails loudly on an inversion and cannot pass by luck.

    ``T = 2.0`` is run too and its agreement count is **printed, not asserted**:
    it is the complementary evidence (a flatter distribution should disagree with
    argmax sometimes), and turning "sometimes" into an assertion would be a flaky
    test. Added by `mb-coverage` attempt 2.
    """

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    handle = _load(mesh_device, paged_attention_config=_paged_config(context=2048, active_slots=32))
    try:
        vocab_size = handle.model.vocab_size
        with GalaxyDirectRunner(handle.model) as runner:
            for slot, row in enumerate(rows):
                runner.prefill_row(row, slot=slot)
            tokens = [1] * GALAXY_PHYSICAL_BATCH
            positions = [128] * GALAXY_PHYSICAL_BATCH
            expected = torch.argmax(runner.decode_logits(tokens, positions)[:, :vocab_size], dim=-1)

            cold = runner.decode_sampled(
                tokens, positions, GalaxySamplingPolicy(top_k=32, top_p=1.0, temperature=0.02, seed=11, on_device=True)
            )
            hot = runner.decode_sampled(
                tokens, positions, GalaxySamplingPolicy(top_k=32, top_p=1.0, temperature=2.0, seed=11, on_device=True)
            )

        agree_cold = sum(1 for slot in range(GALAXY_PHYSICAL_BATCH) if int(cold[slot]) == int(expected[slot]))
        agree_hot = sum(1 for slot in range(GALAXY_PHYSICAL_BATCH) if int(hot[slot]) == int(expected[slot]))
        print(f"[temperature] T=0.02 agrees with host argmax in {agree_cold}/32 slots; T=2.0 in {agree_hot}/32")
        mismatched = [slot for slot in range(GALAXY_PHYSICAL_BATCH) if int(cold[slot]) != int(expected[slot])]
        assert not mismatched, (
            f"at T=0.02 the device sampled off-argmax in slots {mismatched}; a reciprocal-temperature "
            f"inversion (defect D4) is the first thing to check"
        )
        assert torch.all(cold < vocab_size)
        assert torch.all(hot < vocab_size)
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_a_seeded_slot_repeats_across_runs(mesh_device: ttnn.MeshDevice):
    """Slot stability, first sense: same seed, same slot, same token every time.

    The second sense - that moving a request to another slot keeps its stream -
    is **not** asserted: the seed digest mixes the slot in on purpose. See
    ``models/common/tests/models/galaxy/test_step7_sampling.py`` and REPORT.md
    finding D-C2.
    """

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    policy = GalaxySamplingPolicy(top_k=16, top_p=0.9, temperature=0.8, seed=20260827, on_device=True)
    handle = _load(mesh_device, paged_attention_config=_paged_config(context=2048, active_slots=32))
    try:
        observed = []
        for _ in range(3):
            with GalaxyDirectRunner(handle.model) as runner:
                for slot, row in enumerate(rows):
                    runner.prefill_row(row, slot=slot)
                tokens = [1] * GALAXY_PHYSICAL_BATCH
                positions = [128] * GALAXY_PHYSICAL_BATCH
                observed.append(runner.decode_sampled(tokens, positions, policy).clone())

        assert torch.equal(observed[0], observed[1]), "a seeded stochastic decode did not repeat"
        assert torch.equal(observed[1], observed[2]), "a seeded stochastic decode did not repeat"
        assert len(set(int(value) for value in observed[0])) > 1, "one seed collapsed every slot onto one token"
        assert torch.all(observed[0] < handle.model.vocab_size)
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_llama_per_slot_heterogeneous_sampling_controls(mesh_device: ttnn.MeshDevice):
    """Serving mixes top-k, top-p and temperature; the greedy slots must stay greedy.

    Slots 0, 8, 16 and 24 - one per mesh column - are forced greedy while their
    neighbours sample. A greedy slot whose token is not the host argmax means
    the per-slot buffers did not land on the slots they were written for.
    """

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    greedy_slots = (0, 8, 16, 24)
    top_k = [1 if slot in greedy_slots else 8 + (slot % 4) for slot in range(GALAXY_PHYSICAL_BATCH)]
    top_p = [1.0 if slot in greedy_slots else 0.5 + 0.1 * (slot % 4) for slot in range(GALAXY_PHYSICAL_BATCH)]
    temperature = [0.0 if slot in greedy_slots else 0.6 + 0.2 * (slot % 4) for slot in range(32)]

    handle = _load(mesh_device, paged_attention_config=_paged_config(context=2048, active_slots=32))
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            for slot, row in enumerate(rows):
                runner.prefill_row(row, slot=slot)
            tokens = [1] * GALAXY_PHYSICAL_BATCH
            positions = [128] * GALAXY_PHYSICAL_BATCH
            host_logits = runner.decode_logits(tokens, positions)
            expected = torch.argmax(host_logits[:, : handle.model.vocab_size], dim=-1)

            sampled = handle.model.sample_decode(
                runner._decode_device_logits(tokens, positions),
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=[20260827 + slot for slot in range(GALAXY_PHYSICAL_BATCH)],
                slot_ids=list(range(GALAXY_PHYSICAL_BATCH)),
            )
            from models.common.auto_compose import to_torch_auto_compose

            chosen = to_torch_auto_compose(sampled).reshape(-1)[:GALAXY_PHYSICAL_BATCH].to(torch.int64)

        for slot in greedy_slots:
            assert int(chosen[slot]) == int(expected[slot]), f"greedy slot {slot} did not take the host argmax"
        assert torch.all(chosen < handle.model.vocab_size)
    finally:
        _close(handle)


# ---------------------------------------------------------------------------
# Area 1, across two processes: the paged-pool comparison the in-process case
# cannot reach.
#
# `test_llama_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable`
# builds both pools in one process and dies before it can compare them:
# limitation L1. `mb-coverage` attempt 3 measured that on silicon
# (`a3_q_two_pools`) - the *second* model's `activate("decode")` cannot create
# its global circular buffer because the first model's L1 was not returned by
# `close()`, with 923776 of 1393472 bytes per bank still allocated. Finding
# D-C7.
#
# So the comparison is split: one process per pool records its logits, and a
# host-only case compares them. Same claim, same PCC gate, no second model in
# any one process. The artifact directory defaults to a fixed path under the
# system temp dir so the two device processes and the comparison find each other
# with no environment set up; override with STEP7_ARTIFACT_DIR.
# ---------------------------------------------------------------------------

#: One pool per process. The value is only a flag: `None` takes whatever
#: `from_pretrained` installs by default (2048 blocks - D-C4), anything else
#: asks for an explicit 4096-block pool, which changes every slot's run of
#: block ids.
_CROSS_PROCESS_POOLS = {"default2048": None, "explicit4096": "explicit"}


def _artifact_dir() -> str:
    import tempfile

    return os.environ.get("STEP7_ARTIFACT_DIR", os.path.join(tempfile.gettempdir(), "tttv2_step7_artifacts"))


def _artifact_path(pool: str) -> str:
    return os.path.join(_artifact_dir(), "llama33_70b_galaxy_" + pool + ".pt")


@_PARAMS
@_GALAXY
@pytest.mark.parametrize("pool", sorted(_CROSS_PROCESS_POOLS), ids=sorted(_CROSS_PROCESS_POOLS))
@torch.no_grad()
def test_llama_paged_pool_logits_are_recorded_for_cross_process_comparison(mesh_device: ttnn.MeshDevice, pool: str):
    """Record one pool's prefill and decode logits, for the comparison below.

    Exactly one model is built here, so nothing in this case depends on L1 being
    returned. The recording is not itself the gate - the gate is
    ``test_llama_two_paged_pools_agree_across_processes``, which fails loudly
    if either recording is absent.
    """

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    wants_explicit = _CROSS_PROCESS_POOLS[pool] is not None
    paged = _paged_config(context=4096, active_slots=GALAXY_PHYSICAL_BATCH) if wants_explicit else None
    handle = _load(mesh_device, max_seq_len=4096 if wants_explicit else 2048, paged_attention_config=paged)
    try:
        configs = {spec.paged_attention_config for spec in handle.model.kv_specs}
        assert len(configs) == 1
        resolved = next(iter(configs))
        assert resolved is not None, (
            "D-C4 has been fixed: from_pretrained now builds a contiguous cache. "
            "Restore the paged-versus-contiguous comparison this case was written for."
        )
        print(f"[pool] {pool}: block_size={resolved.block_size} max_num_blocks={resolved.max_num_blocks}", flush=True)
        with GalaxyDirectRunner(handle.model) as runner:
            prefill = torch.cat([runner.prefill_row(row, slot=slot) for slot, row in enumerate(rows)], dim=0)
            tokens = [1] * GALAXY_PHYSICAL_BATCH
            positions = [128] * GALAXY_PHYSICAL_BATCH
            decode = runner.decode_logits(tokens, positions).clone()
        os.makedirs(_artifact_dir(), exist_ok=True)
        target = _artifact_path(pool)
        torch.save(
            {
                "pool": pool,
                "block_size": int(resolved.block_size),
                "max_num_blocks": int(resolved.max_num_blocks),
                "prefill": prefill.clone().float(),
                "decode": decode.float(),
                "vocab_size": int(handle.model.vocab_size),
            },
            target,
        )
        print(f"[pool] wrote {target} prefill={tuple(prefill.shape)} decode={tuple(decode.shape)}", flush=True)
    finally:
        _close(handle)


@torch.no_grad()
def test_llama_two_paged_pools_agree_across_processes():
    """The area-1 gate: two different paging geometries, the same logits.

    Host only - it reads what the two recorder processes wrote. A missing
    artifact is a **failure**, never a skip: a skip counted as green is exactly
    the failure mode this project distrusts.
    """

    loaded = {}
    missing = []
    for pool in sorted(_CROSS_PROCESS_POOLS):
        path = _artifact_path(pool)
        if os.path.exists(path):
            loaded[pool] = torch.load(path, weights_only=False)
        else:
            missing.append(path)
    assert not missing, (
        "the cross-process pool recordings are not both on disk: "
        + ", ".join(missing)
        + ". Run test_llama_paged_pool_logits_are_recorded_for_cross_process_comparison "
        "for each pool id first, one node id per process."
    )

    first, second = loaded["default2048"], loaded["explicit4096"]
    assert first["max_num_blocks"] != second["max_num_blocks"], (
        "both recordings used the same pool geometry, so this compares a pool against itself: "
        f"{first['max_num_blocks']} vs {second['max_num_blocks']}"
    )
    print(
        f"[pool] comparing {first['max_num_blocks']}-block against {second['max_num_blocks']}-block, "
        f"block_size {first['block_size']}",
        flush=True,
    )
    for stage in ("prefill", "decode"):
        for slot in range(GALAXY_PHYSICAL_BATCH):
            passed, message = _pcc(first[stage][slot], second[stage][slot], _PAGED_VS_CONTIGUOUS_PCC)
            assert passed, f"{stage} slot {slot}, 2048-block pool vs 4096-block pool: {message}"
    print(f"[pool] all 32 slots agree at PCC >= {_PAGED_VS_CONTIGUOUS_PCC} for prefill and decode", flush=True)
