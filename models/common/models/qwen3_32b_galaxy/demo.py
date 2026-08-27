# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B direct demo for the Galaxy Qwen3-32B reconstruction.

"Direct" means the model graph is driven straight through
:class:`GalaxyDirectRunner`: there is no model-owned executor, no trace, and no
vLLM boundary yet — those are Milestone C. What this file proves is that the
reconstructed tensor model can prefill, decode, sample, and detokenize real
prompts on a WH Galaxy `(8, 4)` mesh at batch 1 and at the full physical batch
32, with no cross-slot contamination.

Run::

    pytest models/common/models/qwen3_32b_galaxy/demo.py -v

Environment knobs:

- ``QWEN3_32B_HF_MODEL`` — another copy of the same checkpoint geometry;
- ``QWEN3_32B_GALAXY_DEMO_LAYERS`` — build a layer subset for fast iteration
  (output is meaningless, but every placement and collective still runs);
- ``QWEN3_32B_GALAXY_DEMO_TOKENS`` — decode length, default 16.

**This file has never been executed.** It was written without a Galaxy mesh.
"""

from __future__ import annotations

import gc
import os
from typing import Any

import pytest
import torch

import ttnn
from models.common.models.galaxy.direct_demo import DEFAULT_DEMO_PROMPTS, fill_prompt_slots, run_direct_demo
from models.common.models.galaxy.direct_runner import GalaxySamplingPolicy
from models.common.models.qwen3_32b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, DEFAULT_HF_REVISION, from_pretrained

_MESH_SHAPE = (8, 4)
_PHYSICAL_BATCH = 32
_PREFILL_LENGTH = 128

_DEVICE_PARAMS = {
    "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
}


def _local_files_only() -> bool:
    return any(
        os.getenv(name, "").lower() in {"1", "true", "yes"} for name in ("CI", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _hf_model() -> str:
    return os.getenv("QWEN3_32B_HF_MODEL", DEFAULT_HF_MODEL)


def _hf_revision() -> str | None:
    """A user-supplied checkpoint copy is not the pinned revision."""

    return DEFAULT_HF_REVISION if _hf_model() == DEFAULT_HF_MODEL else None


def _demo_layers() -> int | None:
    value = os.getenv("QWEN3_32B_GALAXY_DEMO_LAYERS")
    return int(value) if value else None


def _demo_tokens() -> int:
    return int(os.getenv("QWEN3_32B_GALAXY_DEMO_TOKENS", "16"))


def _skip_unless_checkpoint_resolves(hf_model: str) -> None:
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(hf_model, revision=_hf_revision(), local_files_only=_local_files_only())
    except BaseException as error:  # noqa: BLE001 - any resolution failure is a skip, not a defect
        pytest.skip(f"Qwen3-32B checkpoint {hf_model!r} is unavailable: {error}")


def _load(mesh_device: Any, **overrides: Any):
    hf_model = _hf_model()
    _skip_unless_checkpoint_resolves(hf_model)
    kwargs: dict[str, Any] = dict(
        hf_model=hf_model,
        hf_revision=_hf_revision(),
        max_seq_len=2048,
        prefill_sequence_lengths=(_PREFILL_LENGTH,),
        n_layers=_demo_layers(),
    )
    kwargs.update(overrides)
    # A layer subset reads only the shards it needs instead of materialising the
    # whole 64-layer checkpoint. Only for iteration: every gate in this file
    # ignores `_demo_layers()` and runs all 64.
    layers = kwargs.get("n_layers")
    if layers:
        from models.common.tests.models.galaxy.galaxy_checkpoint import load_layer_subset_causal_lm

        kwargs["load_hf_model"] = lambda: load_layer_subset_causal_lm(
            hf_model, layer_indices=tuple(range(layers)), revision=_hf_revision()
        )
    return from_pretrained(mesh_device, **kwargs)


def _close(handle: Any) -> None:
    try:
        handle.close()
    finally:
        del handle
        gc.collect()


def _assert_valid(results, *, vocab_size: int, expected_slots: int) -> None:
    assert len(results) == expected_slots
    for result in results:
        # Print the text before asserting anything. "The full model producing
        # coherent demo output" is a claim a human has to read, and a test that
        # only checks token ranges cannot make it - so the evidence log has to
        # carry the actual continuation.
        print(f"[demo] slot {result.slot} prompt: {result.prompt!r}", flush=True)
        print(f"[demo] slot {result.slot} text  : {result.text!r}", flush=True)
        print(f"[demo] slot {result.slot} tokens: {list(result.tokens)}", flush=True)
    for result in results:
        assert result.tokens, f"slot {result.slot} produced no tokens"
        assert all(0 <= token < vocab_size for token in result.tokens), f"slot {result.slot} sampled outside the vocab"


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_direct_demo_batch1(mesh_device: ttnn.MeshDevice):
    """One prompt, greedy host sampling, single-row prefill."""

    handle = _load(mesh_device)
    try:
        results = run_direct_demo(
            handle,
            prompts=DEFAULT_DEMO_PROMPTS[:1],
            max_new_tokens=_demo_tokens(),
            policy=GalaxySamplingPolicy(top_k=1, temperature=0.0),
        )
        _assert_valid(results, vocab_size=handle.model.vocab_size, expected_slots=1)
    finally:
        _close(handle)


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination(mesh_device: ttnn.MeshDevice):
    """Every slot decodes its own prompt.

    The 32 slots are eight distinct prompts repeated four times. Greedy decoding
    is deterministic, so slots holding the same prompt must produce identical
    tokens and slots holding different prompts must not — the first check catches
    a slot reading another slot's KV blocks, the second catches every slot
    collapsing onto one user's cache.
    """

    prompts = fill_prompt_slots(DEFAULT_DEMO_PROMPTS, _PHYSICAL_BATCH)
    handle = _load(mesh_device)
    try:
        results = run_direct_demo(
            handle,
            prompts=prompts,
            max_new_tokens=_demo_tokens(),
            policy=GalaxySamplingPolicy(top_k=1, temperature=0.0),
        )
        _assert_valid(results, vocab_size=handle.model.vocab_size, expected_slots=_PHYSICAL_BATCH)
        distinct = len(DEFAULT_DEMO_PROMPTS)
        for slot in range(distinct, _PHYSICAL_BATCH):
            message = f"slot {slot} disagrees with slot {slot % distinct} on the same prompt"
            assert results[slot].tokens == results[slot % distinct].tokens, message
        unique = {result.tokens for result in results[:distinct]}
        assert len(unique) > 1, "every distinct prompt produced the same continuation"
    finally:
        _close(handle)


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_direct_demo_concat32_prefill_matches_sequential(mesh_device: ttnn.MeshDevice):
    """Concatenated physical-batch-32 prefill agrees with sequential prefill.

    Both paths run the same 32 prompts through the same weights; only the
    prefill row mode differs, so greedy continuations must match.
    """

    prompts = fill_prompt_slots(DEFAULT_DEMO_PROMPTS, _PHYSICAL_BATCH)
    policy = GalaxySamplingPolicy(top_k=1, temperature=0.0)
    handle = _load(mesh_device, batched_prefill_sequence_lengths=(_PREFILL_LENGTH,))
    try:
        sequential = run_direct_demo(handle, prompts=prompts, max_new_tokens=_demo_tokens(), policy=policy)
        batched = run_direct_demo(
            handle, prompts=prompts, max_new_tokens=_demo_tokens(), policy=policy, batched_prefill=True
        )
        _assert_valid(batched, vocab_size=handle.model.vocab_size, expected_slots=_PHYSICAL_BATCH)
        mismatched = [slot for slot in range(_PHYSICAL_BATCH) if sequential[slot].tokens != batched[slot].tokens]
        assert not mismatched, f"concat-32 prefill diverged from sequential prefill in slots {mismatched}"
    finally:
        _close(handle)


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_direct_demo_device_sampling_matches_host_greedy(mesh_device: ttnn.MeshDevice):
    """Greedy device sampling selects the same tokens as host argmax.

    This is the first exercise of the column user selector plus ``Sampling2D``
    on a real model, and the reason greedy is used: a forced argmax has one
    correct answer, so any disagreement is a defect rather than a seed.
    """

    prompts = fill_prompt_slots(DEFAULT_DEMO_PROMPTS, _PHYSICAL_BATCH)
    handle = _load(mesh_device)
    try:
        host = run_direct_demo(
            handle,
            prompts=prompts,
            max_new_tokens=_demo_tokens(),
            policy=GalaxySamplingPolicy(top_k=1, temperature=0.0, on_device=False),
        )
        device = run_direct_demo(
            handle,
            prompts=prompts,
            max_new_tokens=_demo_tokens(),
            policy=GalaxySamplingPolicy(top_k=1, temperature=0.0, on_device=True),
        )
        mismatched = [slot for slot in range(_PHYSICAL_BATCH) if host[slot].tokens != device[slot].tokens]
        assert not mismatched, f"device sampling disagreed with host argmax in slots {mismatched}"
    finally:
        _close(handle)
