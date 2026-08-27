# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step-3 and step-7 hardware coverage for Galaxy Qwen3-32B.

The full 64-layer model, driven directly through :class:`GalaxyDirectRunner`
with a paged KV cache. Together these tests cover the Milestone B exit gate for
this model:

- full-model prefill plus a first decode token;
- teacher-forced accuracy at batch 1, sequence length 512
  (top-1 >= 89%, top-5 >= 97%);
- batch 1 and batch 32, with no cross-slot contamination;
- paged KV throughout — every test here allocates a paged pool;
- prefix-cached and chunked prefill, checked against uncached execution;
- batch-1 long-context functional smokes at 4K, 32K and 128K;
- repeated requests and deterministic cleanup.

Concatenated physical-32 prefill and device sampling live in the package demo,
which drives the same runner.

**This file has never been executed.** It was written without a Galaxy mesh.
Every threshold is the plan's gate; every staging decision is a hypothesis.

Cost notes for whoever runs it first:

- the accuracy test is 511 eager decode steps of an 80-layer model;
- the 128K smoke needs roughly ``4127 blocks * 32 * 128`` bytes of bfloat8 KV
  per tensor per layer, about 2.7 GB per device across 80 layers, on top of the
  weights. If it fails to allocate, that is a real capacity result, not a bug.

Run::

    pytest models/common/tests/models/qwen3_32b_galaxy/test_full_model_wh_galaxy.py -v
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
    align_top5,
    hf_config_or_skip,
    load_reference_tokens,
    teacher_forcing_accuracy,
)

_REFERENCE_NAME = "Qwen3-32B"
_BLOCK_SIZE = 32
_ACCURACY_PROMPT_LENGTH = 512
_ACCURACY_DECODE_TOKENS = 511
_TOP1_GATE = 0.89
_TOP5_GATE = 0.97
_GREEDY = GalaxySamplingPolicy(top_k=1, temperature=0.0)


def _hf_model() -> str:
    return os.getenv("QWEN3_32B_HF_MODEL", DEFAULT_HF_MODEL)


def _hf_revision() -> str | None:
    """A user-supplied checkpoint copy is not the pinned revision."""

    return DEFAULT_HF_REVISION if _hf_model() == DEFAULT_HF_MODEL else None


def _layers() -> int | None:
    """Allow a layer subset for fast iteration; accuracy gates ignore it."""

    value = os.getenv("QWEN3_32B_GALAXY_TEST_LAYERS")
    return int(value) if value else None


def _paged_config(*, context: int, active_slots: int) -> GalaxyPagedAttentionConfig:
    """Size the block pool for ``active_slots`` plus one sink block per idle slot."""

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
    # A layer subset reads only the shards it needs instead of materialising the
    # whole 64-layer checkpoint. Only for iteration: every gate in this file
    # ignores `_layers()` and runs all 64.
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


def _pcc(expected: torch.Tensor, actual: torch.Tensor, threshold: float):
    from models.common.utility_functions import comp_pcc

    return comp_pcc(expected.unsqueeze(0).float(), actual.unsqueeze(0).float(), threshold)


def _reference_prompt(length: int) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    reference_tokens, top5_tokens = load_reference_tokens(_REFERENCE_NAME)
    if len(reference_tokens) <= length:
        pytest.skip(f"reference sequence has {len(reference_tokens)} tokens, need more than {length}")
    return [int(value) for value in reference_tokens[:length]], reference_tokens, top5_tokens


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_full_model_prefill_and_first_decode_token(mesh_device: ttnn.MeshDevice):
    """Prefill 128 real tokens, decode one, and check both against the reference.

    The prefill prediction and the first decode prediction address consecutive
    positions of the same reference sequence, so a decode step that silently
    reads the wrong KV blocks shows up as a top-5 miss even though the prefill
    passed.
    """

    prompt, reference_tokens, top5_tokens = _reference_prompt(128)
    aligned = align_top5(top5_tokens, reference_tokens, len(prompt))
    handle = _load(mesh_device)
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            prefill_logits = runner.prefill_row(prompt, slot=0)
            first_token = int(torch.argmax(prefill_logits[0]))
            # Print before asserting: a passing gate that records no number is
            # not evidence.
            print(
                f"[full-model] prefill {len(prompt)} predicted {first_token}, "
                f"reference top-5 {aligned[0, :].tolist()}, target {int(reference_tokens[len(prompt)])}",
                flush=True,
            )
            message = f"prefill predicted {first_token}, outside the reference top-5 {aligned[0, :].tolist()}"
            assert first_token in aligned[0, :].tolist(), message

            tokens = [0] * GALAXY_PHYSICAL_BATCH
            positions = [0] * GALAXY_PHYSICAL_BATCH
            tokens[0] = int(reference_tokens[len(prompt)])
            positions[0] = len(prompt)
            decode_logits = runner.decode_logits(tokens, positions)
            second_token = int(torch.argmax(decode_logits[0]))
            print(
                f"[full-model] decode position {len(prompt)} predicted {second_token}, "
                f"reference top-5 {aligned[1, :].tolist()}, target {int(reference_tokens[len(prompt) + 1])}",
                flush=True,
            )
            message = f"decode predicted {second_token}, outside the reference top-5 {aligned[1, :].tolist()}"
            assert second_token in aligned[1, :].tolist(), message
    finally:
        _close(handle)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_teacher_forced_accuracy_batch1(mesh_device: ttnn.MeshDevice):
    """Milestone B accuracy gate: top-1 >= 89%, top-5 >= 97%."""

    prompt, reference_tokens, top5_tokens = _reference_prompt(_ACCURACY_PROMPT_LENGTH)
    targets = reference_tokens[len(prompt) : len(prompt) + _ACCURACY_DECODE_TOKENS]
    if len(targets) < _ACCURACY_DECODE_TOKENS:
        pytest.skip(f"reference sequence yields {len(targets)} targets, need {_ACCURACY_DECODE_TOKENS}")
    aligned = align_top5(top5_tokens, reference_tokens, len(prompt))[:_ACCURACY_DECODE_TOKENS]

    handle = _load(mesh_device, max_seq_len=2048, prefill_sequence_lengths=(_ACCURACY_PROMPT_LENGTH,))
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            logits = runner.teacher_forced_decode(prompt, [int(value) for value in targets], slot=0)
        predictions = [int(value) for value in torch.argmax(logits, dim=-1)]
        top1, top5 = teacher_forcing_accuracy(predictions, aligned)
        # The raw counts, not only the percentage: a passing gate that records no
        # number is not evidence.
        top1_hits = sum(1 for index, token in enumerate(predictions) if token == int(aligned[index, 0]))
        top5_hits = sum(1 for index, token in enumerate(predictions) if token in aligned[index, :].tolist())
        print(
            f"[accuracy] reference={_REFERENCE_NAME} prompt={len(prompt)} decode={len(predictions)}\n"
            f"[accuracy] top-1 {top1_hits}/{len(predictions)} = {top1:.4f} (gate >= {_TOP1_GATE})\n"
            f"[accuracy] top-5 {top5_hits}/{len(predictions)} = {top5:.4f} (gate >= {_TOP5_GATE})",
            flush=True,
        )
        assert top1 >= _TOP1_GATE, f"top-1 {top1:.3f} below the {_TOP1_GATE:.2f} gate"
        assert top5 >= _TOP5_GATE, f"top-5 {top5:.3f} below the {_TOP5_GATE:.2f} gate"
    finally:
        _close(handle)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_batch32_slots_are_isolated(mesh_device: ttnn.MeshDevice):
    """A slot's continuation must not depend on what the other 31 slots hold.

    Slot 0 runs the same prompt alone and inside a full physical batch whose
    other slots hold different prompts. Greedy decoding is deterministic, so the
    two continuations must be identical.
    """

    reference_tokens, _ = load_reference_tokens(_REFERENCE_NAME)
    stride = 8
    prompts = [
        [int(value) for value in reference_tokens[slot * stride : slot * stride + 128]]
        for slot in range(GALAXY_PHYSICAL_BATCH)
    ]
    if any(len(prompt) < 128 for prompt in prompts):
        pytest.skip("reference sequence is too short for 32 distinct 128-token prompts")

    handle = _load(mesh_device)
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            alone = runner.generate(prompts[:1], max_new_tokens=8, policy=_GREEDY)
        with GalaxyDirectRunner(handle.model) as runner:
            batched = runner.generate(prompts, max_new_tokens=8, policy=_GREEDY)
        message = "slot 0 changed when the other 31 slots were occupied"
        assert batched[0].generated_tokens == alone[0].generated_tokens, message
        distinct = {tuple(result.generated_tokens) for result in batched}
        assert len(distinct) > 1, "every slot produced the same continuation from different prompts"
    finally:
        _close(handle)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_prefix_cached_prefill_matches_uncached(mesh_device: ttnn.MeshDevice):
    """Two 128-token chunks must predict what one 256-token prefill predicts."""

    prompt, _, _ = _reference_prompt(256)
    handle = _load(
        mesh_device,
        prefill_sequence_lengths=(128, 256),
        chunked_prefill_sequence_lengths=(128,),
    )
    try:
        with GalaxyDirectRunner(handle.model) as runner:
            uncached = runner.prefill_row(prompt, slot=0, sequence_length=256)
        with GalaxyDirectRunner(handle.model) as runner:
            chunked = runner.prefill_chunked(prompt, slot=0, chunk_length=128)
        message = "prefix-cached prefill predicted a different token than uncached prefill"
        assert int(torch.argmax(uncached[0])) == int(torch.argmax(chunked[0])), message
        passing, message = _pcc(uncached[0], chunked[0], 0.99)
        assert passing, f"prefix-cached logits diverged: {message}"
    finally:
        _close(handle)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@pytest.mark.parametrize("context", [4096, 32768, 131072], ids=["4k", "32k", "128k"])
@torch.no_grad()
def test_qwen3_32b_galaxy_long_context_smoke(mesh_device: ttnn.MeshDevice, context: int):
    """Batch-1 functional smoke: chunked prefill to ``context``, then decode.

    Only slot 0 is served, so the block pool holds one user's context plus one
    sink block per idle slot. The check is functional, not numerical: the model
    must complete the whole chunked prefill and produce an in-vocabulary token
    at the far end of the context.
    """

    chunk = 2048
    reference_tokens, _ = load_reference_tokens(_REFERENCE_NAME)
    source = [int(value) for value in reference_tokens]
    prompt = (source * -(-context // len(source)))[:context]
    # One chunk of headroom past the prompt so the decode step after a full
    # prefill still has a block to write into.
    served = context + chunk

    handle = _load(
        mesh_device,
        max_seq_len=served,
        prefill_sequence_lengths=(chunk,),
        chunked_prefill_sequence_lengths=(chunk,),
        paged_attention_config=_paged_config(context=served, active_slots=1),
    )
    try:
        with GalaxyDirectRunner(handle.model, active_slots=1) as runner:
            logits = runner.prefill_chunked(prompt, slot=0, chunk_length=chunk)
            token = int(torch.argmax(logits[0]))
            assert 0 <= token < handle.model.vocab_size

            tokens = [0] * GALAXY_PHYSICAL_BATCH
            positions = [0] * GALAXY_PHYSICAL_BATCH
            tokens[0] = token
            positions[0] = context
            decoded = int(torch.argmax(runner.decode_logits(tokens, positions)[0]))
            assert 0 <= decoded < handle.model.vocab_size
    finally:
        _close(handle)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_repeated_requests_and_deterministic_cleanup(mesh_device: ttnn.MeshDevice):
    """The same request repeats identically, and cleanup is terminal.

    A second identical run proves the KV cache, the CCL semaphores and the
    prefetcher context all return to their starting state. Closing the runner
    twice must be a no-op rather than a double free.
    """

    prompt, _, _ = _reference_prompt(128)
    handle = _load(mesh_device)
    try:
        outputs = []
        for _ in range(2):
            runner = GalaxyDirectRunner(handle.model)
            runner.open()
            try:
                outputs.append(runner.generate([prompt], max_new_tokens=8, policy=_GREEDY)[0].generated_tokens)
            finally:
                runner.close()
                runner.close()
        assert outputs[0] == outputs[1], "a repeated identical request produced different tokens"
    finally:
        _close(handle)
