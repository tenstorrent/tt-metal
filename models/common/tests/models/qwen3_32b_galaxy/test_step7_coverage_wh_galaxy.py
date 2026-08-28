# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step-7 device coverage for Galaxy Qwen3-32B: the gaps.

The Llama counterpart is
``models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py``
and it carries the reasoning for each case. This file covers the same gaps for
Qwen plus the one claim that is Qwen-only:

* **the padded vocabulary can never be sampled.** Qwen's 151936 pads to
  ``galaxy_padded_vocab_size`` **153600** - the ring-exact width, 1664 invalid
  ids, not the 128 of the minimal alignment (D-B19). The claim that Llama needs
  no such case was **wrong**: 128256 pads to 129024, 768 invalid ids, for the
  same ring-exactness reason. `mb-coverage` attempt 2 added the Llama twin.

**The two blockers in this banner are gone.** Both were true when the file was
written on 2026-08-27 and neither is true now:

* the Qwen3-32B checkpoint **is** on this machine, under
  ``HF_HOME=/localdev/ctr-apbernal/hf_data`` - not ``/proj_sw/user_dev/hf_data``,
  which holds Llama only. `mb-qwen` attempt 2 qualified the whole model on
  silicon from it;
* the mesh is whole: ``ls /sys/class/tenstorrent | wc -l`` is 32 and an 8x4
  cluster opens.

This file was **first executed on 2026-08-27/28 by `mb-coverage` attempt 2**;
what each case measured is in
``tttv2_milestone_b_evidence/coverage/REPORT.md`` §A2.

Run::

    pytest models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py -v
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
from models.common.models.qwen3_32b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, from_pretrained
from models.common.models.qwen3_32b_galaxy.model import DEFAULT_HF_REVISION
from models.common.tests.models.galaxy.galaxy_hardware import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    hf_config_or_skip,
    load_reference_tokens,
)

_REFERENCE_NAME = "Qwen3-32B"
_BLOCK_SIZE = 32
_PAGED_VS_CONTIGUOUS_PCC = 0.99
_GREEDY = GalaxySamplingPolicy(top_k=1, temperature=0.0)

#: Ascending. 1024 and 2048 are deliberately omitted until 128-512 have been
#: seen to pass once: each entry is another resolved recipe and another set of
#: collective resources, and a 32x2048 concatenated stream is 65536 tokens.
_BATCHED_LENGTHS = (128, 256, 512)


def _hf_model() -> str:
    return os.getenv("QWEN3_32B_HF_MODEL", DEFAULT_HF_MODEL)


def _hf_revision() -> str | None:
    return DEFAULT_HF_REVISION if _hf_model() == DEFAULT_HF_MODEL else None


def _layers() -> int | None:
    value = os.getenv("QWEN3_32B_GALAXY_TEST_LAYERS")
    return int(value) if value else None


def _paged_config(*, context: int, active_slots: int) -> GalaxyPagedAttentionConfig:
    blocks_per_user = -(-context // _BLOCK_SIZE)
    sinks = GALAXY_PHYSICAL_BATCH - active_slots
    return GalaxyPagedAttentionConfig(block_size=_BLOCK_SIZE, max_num_blocks=blocks_per_user * active_slots + sinks)


def _load(mesh_device: ttnn.MeshDevice, **overrides: Any):
    hf_model = _hf_model()
    hf_config_or_skip(hf_model, revision=_hf_revision())
    kwargs: dict[str, Any] = dict(
        hf_model=hf_model,
        hf_revision=_hf_revision(),
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
def test_qwen_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable(mesh_device: ttnn.MeshDevice):
    """The Llama twin carries the reasoning; see finding D-C4.

    ``from_pretrained(paged_attention_config=None)`` substitutes
    ``default_paged_attention_config``, so "paged versus contiguous" is not
    expressible through the adaptor and the case as written compared a pool
    against itself. This compares the default 2048-block pool against an
    explicit 4096-block one, which changes every slot's run of block ids.
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
            configs = {spec.paged_attention_config for spec in handle.model.kv_specs}
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
def test_qwen_paged_capacity_resolved_after_construction_serves_a_request(mesh_device: ttnn.MeshDevice, expect_error):
    prompt = _prompt(128)
    handle = _load(mesh_device, paged_attention_config=None)
    try:
        model = handle.model
        # D-C4: `paged_attention_config=None` does not mean "contiguous"; the
        # adaptor substitutes `default_paged_attention_config`. The reachable
        # claim is that the geometry can still be replaced before anything is
        # bound. Corrected by `mb-coverage` attempt 2.
        installed = {spec.paged_attention_config for spec in model.kv_specs}
        assert len(installed) == 1 and next(iter(installed)) is not None
        default_pool = next(iter(installed))
        print(f"[pool] as constructed: {default_pool}")

        pool = _paged_config(context=4096, active_slots=32)
        assert pool != default_pool, "the replacement pool must differ from the default to prove anything"
        model.configure_paged_attention(block_size=pool.block_size, max_num_blocks=pool.max_num_blocks)

        with GalaxyDirectRunner(model) as runner:
            with expect_error(RuntimeError, "cannot be reconfigured"):
                model.configure_paged_attention(block_size=32, max_num_blocks=pool.max_num_blocks + 32)
            logits = runner.prefill_row(prompt, slot=0)
            assert 0 <= int(torch.argmax(logits[0])) < model.vocab_size
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_qwen_a_write_for_one_user_never_appears_in_another_users_blocks(mesh_device: ttnn.MeshDevice):
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
    finally:
        _close(handle)


# ---------------------------------------------------------------------------
# Area 2: concat-32 physical prefill
# ---------------------------------------------------------------------------


@_PARAMS
@_GALAXY
@pytest.mark.parametrize("length", _BATCHED_LENGTHS, ids=[f"len{value}" for value in _BATCHED_LENGTHS])
@torch.no_grad()
def test_qwen_concat32_matches_sequential_prefill_at_each_length(mesh_device: ttnn.MeshDevice, length: int):
    rows = _distinct_rows(length, GALAXY_PHYSICAL_BATCH)
    handle = _load(
        mesh_device,
        prefill_sequence_lengths=(length,),
        batched_prefill_sequence_lengths=(length,),
        paged_attention_config=_paged_config(context=2048, active_slots=32),
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
def test_qwen_concat32_padded_rows_change_no_active_rows_logits(mesh_device: ttnn.MeshDevice, active: int):
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
# Area 3: prefix-cached and chunked prefill
# ---------------------------------------------------------------------------


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_qwen_chunked_prefill_matches_a_single_uncached_prefill(mesh_device: ttnn.MeshDevice):
    prompt = _prompt(256)
    handle = _load(
        mesh_device,
        prefill_sequence_lengths=(128, 256),
        chunked_prefill_sequence_lengths=(128,),
        paged_attention_config=_paged_config(context=2048, active_slots=32),
    )
    try:
        tokens = [0] * GALAXY_PHYSICAL_BATCH
        positions = [0] * GALAXY_PHYSICAL_BATCH
        positions[0] = len(prompt)

        with GalaxyDirectRunner(handle.model) as runner:
            uncached = runner.prefill_row(prompt, slot=0, sequence_length=256).clone()
            tokens[0] = int(torch.argmax(uncached[0]))
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


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_qwen_a_prefix_cached_request_then_a_normal_one(mesh_device: ttnn.MeshDevice):
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
def test_qwen_prefix_cached_and_plain_requests_mixed_across_slots(mesh_device: ttnn.MeshDevice):
    """A mix of both in one batch, then one decode over all of them.

    The Llama twin carries the reasoning. `mb-coverage` attempt 3 added this
    case: attempt 2 left the Qwen half of the step-7 brief's "a mix of both in
    the same batch" unwritten, so the claim had no Qwen measurement at all.
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
            for slot in range(GALAXY_PHYSICAL_BATCH):
                if slot % 2:
                    runner.prefill_chunked(chunked_prompt, slot=slot, chunk_length=128)
                else:
                    runner.prefill_row(plain_prompt, slot=slot)

            tokens = [1] * GALAXY_PHYSICAL_BATCH
            positions = [
                len(plain_prompt) if slot % 2 == 0 else len(chunked_prompt) for slot in range(GALAXY_PHYSICAL_BATCH)
            ]
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


# ---------------------------------------------------------------------------
# Area 4: device sampling, including the padded vocabulary
# ---------------------------------------------------------------------------


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_qwen_device_greedy_sampling_equals_host_argmax(mesh_device: ttnn.MeshDevice):
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
def test_qwen_no_padded_vocabulary_id_is_ever_sampled(mesh_device: ttnn.MeshDevice, policy):
    """Qwen pads 151936 up to 153600; those 1664 ids are not tokens.

    An invalid id winning is a correctness bug, not a rounding issue, so this
    runs at three policies: greedy, and two temperatures either side of 1.0.
    ``T == 1.0`` is skipped on purpose - it is its own reciprocal and would hide
    a temperature inversion (defect D4).
    """

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    handle = _load(mesh_device, paged_attention_config=_paged_config(context=2048, active_slots=32))
    try:
        vocab_size = handle.model.vocab_size
        padded_vocab_size = handle.model.padded_vocab_size
        assert padded_vocab_size > vocab_size, "this test is only meaningful for a padded vocabulary"

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
def test_qwen_a_near_zero_temperature_collapses_onto_the_host_argmax(mesh_device: ttnn.MeshDevice):
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
def test_qwen_a_seeded_slot_repeats_across_runs(mesh_device: ttnn.MeshDevice):
    """Slot stability, first sense only; see the Llama file and REPORT.md D-C2."""

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

        assert torch.equal(observed[0], observed[1])
        assert torch.equal(observed[1], observed[2])
        assert torch.all(observed[0] < handle.model.vocab_size)
    finally:
        _close(handle)


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_qwen_per_slot_heterogeneous_sampling_controls(mesh_device: ttnn.MeshDevice):
    """Serving mixes top-k, top-p and temperature; the greedy slots must stay greedy.

    The Llama twin carries the reasoning. `mb-coverage` attempt 3 added this
    case: the step-7 brief asks for per-slot heterogeneous controls for both
    models and attempt 2 wrote only the Llama half.
    """

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    greedy_slots = (0, 8, 16, 24)
    top_k = [1 if slot in greedy_slots else 8 + (slot % 4) for slot in range(GALAXY_PHYSICAL_BATCH)]
    top_p = [1.0 if slot in greedy_slots else 0.5 + 0.1 * (slot % 4) for slot in range(GALAXY_PHYSICAL_BATCH)]
    temperature = [0.0 if slot in greedy_slots else 0.6 + 0.2 * (slot % 4) for slot in range(GALAXY_PHYSICAL_BATCH)]

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
# `test_qwen_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable`
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
    return os.path.join(_artifact_dir(), "qwen3_32b_galaxy_" + pool + ".pt")


@_PARAMS
@_GALAXY
@pytest.mark.parametrize("pool", sorted(_CROSS_PROCESS_POOLS), ids=sorted(_CROSS_PROCESS_POOLS))
@torch.no_grad()
def test_qwen_paged_pool_logits_are_recorded_for_cross_process_comparison(mesh_device: ttnn.MeshDevice, pool: str):
    """Record one pool's prefill and decode logits, for the comparison below.

    Exactly one model is built here, so nothing in this case depends on L1 being
    returned. The recording is not itself the gate - the gate is
    ``test_qwen_two_paged_pools_agree_across_processes``, which fails loudly
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
def test_qwen_two_paged_pools_agree_across_processes():
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
        + ". Run test_qwen_paged_pool_logits_are_recorded_for_cross_process_comparison "
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


# ---------------------------------------------------------------------------
# Area 4, behind D-C5: the four sampling claims, with the one-line fix applied
# at the call site.
#
# Every area-4 case above dies in `GalaxyColumnUserSelector.__call__`
# (`collectives.py:445`), a bare `ttnn.matmul` whose default multi-core program
# config requires an INTERLEAVED input B, on decode logits that
# `recipes.lm_head_output_memcfg` makes WIDTH_SHARDED for **both** models.
# Measured: `a3_q_greedy` and `a3_l_greedy`, same TT_FATAL, same frame.
#
# This case is a *diagnostic*, not a substitute gate. It does what the selector
# would have to do - one `ttnn.sharded_to_interleaved`, the same op
# `collectives._relocate_sharded` already uses two hundred lines above, chosen
# there because it runs on its input's own `shard_spec.grid` and so stays
# worker-confined under a loaded sub-device manager - and then asks whether
# anything *else* is wrong behind D-C5.
#
# That question is not rhetorical in this project. D-C5 was invisible for Llama
# because the L1 address clash arrived first; two independent faults, one hiding
# the other. So area 4 stays reported as BLOCKED by D-C5 whatever this case says,
# and what this case adds is either "D-C5 is the only blocker, and here is the
# hardware evidence that the one-line fix is sufficient" or "there is a second
# one, and here it is".
#
# Assertions are all at the end, after every measurement is printed, so a log
# carries the whole picture even when one claim fails.
# ---------------------------------------------------------------------------


@_PARAMS
@_GALAXY
@torch.no_grad()
def test_qwen_device_sampling_claims_behind_dc5_with_interleaved_logits(mesh_device: ttnn.MeshDevice):
    """Greedy, padded vocabulary, D4's reciprocal temperature, seed stability and
    per-slot heterogeneous controls - all in one process, all through
    ``model.sample_decode``, with the decode logits relocated to interleaved DRAM
    first because the selector does not do it for itself."""

    from models.common.auto_compose import to_torch_auto_compose

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    handle = _load(mesh_device, paged_attention_config=_paged_config(context=2048, active_slots=32))
    try:
        vocab_size = handle.model.vocab_size
        padded_width = None
        results = {}
        with GalaxyDirectRunner(handle.model) as runner:
            for slot, row in enumerate(rows):
                runner.prefill_row(row, slot=slot)
            tokens = [1] * GALAXY_PHYSICAL_BATCH
            positions = [128] * GALAXY_PHYSICAL_BATCH
            expected = torch.argmax(runner.decode_logits(tokens, positions)[:, :vocab_size], dim=-1)

            def sample(label, **kwargs):
                nonlocal padded_width
                device_logits = runner._decode_device_logits(tokens, positions)
                memcfg = device_logits.memory_config()
                layout = memcfg.memory_layout
                padded_width = int(device_logits.shape[-1])
                # Identity when the logits are already interleaved: if D-C5 is
                # ever fixed by relocating inside the selector, this case must
                # keep measuring the same claims rather than start failing on a
                # redundant relocation.
                relocated = (
                    ttnn.sharded_to_interleaved(device_logits, ttnn.DRAM_MEMORY_CONFIG)
                    if memcfg.is_sharded()
                    else device_logits
                )
                print(
                    f"[dc5] {label}: decode logits were {layout}, width {padded_width}; "
                    f"relocated to {relocated.memory_config().memory_layout}",
                    flush=True,
                )
                try:
                    sampled = handle.model.sample_decode(
                        relocated, slot_ids=list(range(GALAXY_PHYSICAL_BATCH)), **kwargs
                    )
                finally:
                    if relocated is not device_logits:
                        ttnn.deallocate(relocated)
                return to_torch_auto_compose(sampled).reshape(-1)[:GALAXY_PHYSICAL_BATCH].to(torch.int64)

            results["greedy"] = sample("greedy", top_k=1, temperature=0.0)
            results["cold"] = sample("T=0.02", top_k=32, top_p=1.0, temperature=0.02, seed=11)
            results["hot"] = sample("T=2.0", top_k=32, top_p=1.0, temperature=2.0, seed=11)
            seeds = [20260828 + slot for slot in range(GALAXY_PHYSICAL_BATCH)]
            results["seeded_a"] = sample("seeded pass 1", top_k=32, top_p=1.0, temperature=0.8, seed=seeds)
            results["seeded_b"] = sample("seeded pass 2", top_k=32, top_p=1.0, temperature=0.8, seed=seeds)
            greedy_slots = list(range(0, GALAXY_PHYSICAL_BATCH, 4))
            results["heterogeneous"] = sample(
                "per-slot controls",
                top_k=[1 if slot in greedy_slots else 4 + (slot % 8) for slot in range(GALAXY_PHYSICAL_BATCH)],
                top_p=[
                    1.0 if slot in greedy_slots else 0.5 + 0.1 * (slot % 4) for slot in range(GALAXY_PHYSICAL_BATCH)
                ],
                temperature=[
                    0.0 if slot in greedy_slots else 0.6 + 0.2 * (slot % 4) for slot in range(GALAXY_PHYSICAL_BATCH)
                ],
                seed=seeds,
            )

        def agreement(name):
            return sum(1 for slot in range(GALAXY_PHYSICAL_BATCH) if int(results[name][slot]) == int(expected[slot]))

        print(f"[dc5] vocab_size={vocab_size} padded logits width={padded_width}", flush=True)
        print(f"[dc5] greedy agrees with host argmax in {agreement('greedy')}/32 slots", flush=True)
        print(f"[dc5] T=0.02 agrees in {agreement('cold')}/32, T=2.0 agrees in {agreement('hot')}/32", flush=True)
        repeats = sum(
            1
            for slot in range(GALAXY_PHYSICAL_BATCH)
            if int(results["seeded_a"][slot]) == int(results["seeded_b"][slot])
        )
        print(f"[dc5] the same seed in the same slot repeated in {repeats}/32 slots", flush=True)
        print(f"[dc5] heterogeneous greedy slots {greedy_slots}", flush=True)
        for name, chosen in sorted(results.items()):
            over = [slot for slot in range(GALAXY_PHYSICAL_BATCH) if int(chosen[slot]) >= vocab_size]
            print(f"[dc5] {name}: padded ids sampled in slots {over}", flush=True)
        print(f"[dc5] greedy tokens: {[int(value) for value in results['greedy']]}", flush=True)
        print(f"[dc5] seeded tokens: {[int(value) for value in results['seeded_a']]}", flush=True)

        # The claims, asserted only now that every measurement is in the log.
        for name, chosen in sorted(results.items()):
            over = [slot for slot in range(GALAXY_PHYSICAL_BATCH) if int(chosen[slot]) >= vocab_size]
            assert not over, f"{name} sampled a padded vocabulary id in slots {over}; vocab_size={vocab_size}"
        greedy_mismatch = [
            slot for slot in range(GALAXY_PHYSICAL_BATCH) if int(results["greedy"][slot]) != int(expected[slot])
        ]
        assert not greedy_mismatch, f"device greedy disagreed with the host argmax in slots {greedy_mismatch}"
        cold_mismatch = [
            slot for slot in range(GALAXY_PHYSICAL_BATCH) if int(results["cold"][slot]) != int(expected[slot])
        ]
        assert not cold_mismatch, (
            f"at T=0.02 the device sampled off-argmax in slots {cold_mismatch}; a reciprocal-temperature "
            "inversion (defect D4) is the first thing to check"
        )
        assert (
            repeats == GALAXY_PHYSICAL_BATCH
        ), f"the same seed in the same slot did not repeat: only {repeats}/32 slots matched"
        for slot in greedy_slots:
            assert int(results["heterogeneous"][slot]) == int(
                expected[slot]
            ), f"per-slot controls: slot {slot} was configured greedy and did not take the host argmax"
    finally:
        _close(handle)
