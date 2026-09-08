# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decode demo for the whole 43-layer stack on a 32-chip Galaxy,
with every non-expert decoder weight resident in L1.

The 8-chip demo (``test_full_model_decode_demo.py``) streams each layer's weights
from DRAM through the prefetcher's shared GCB, one page at a time, every token.
This one does no weight DRAM traffic at all: the ``galaxy32`` profile turns on
``decode.packed_l1_weights``, so each chip's layers have their non-expert matmul
weights fused into a single bf4 height-sharded L1 tensor (``tt/l1_placement.py``
for the per-zone placement, ``tt/l1_weights.py`` for the packing), and every
matmul reads its slab straight out of L1 where it already sits. Only the routed
experts stay in DRAM -- they are far too large to be resident and are gathered
per token by ``fused_experts`` anyway.

Why 22 chips and not 32
-----------------------
The placement fits at most TWO layers per chip, so 43 layers need
``ceil(43/2) = 22``: chips 0-20 hold a contiguous pair and chip 21 holds the odd
layer 42. ``pipeline.max_devices: 22`` pins that and leaves the remaining 10
chips idle on purpose -- spreading the same 43 layers over all 32 would hold no
more per chip while adding 10 more cross-chip socket hops to every token (31
instead of 21), which is pure added latency. Putting those chips to work needs
tensor/expert parallelism inside a stage, which does not exist yet.

The mesh is still opened at its full 32 chips: that is what makes the ``galaxy32``
profile match on device count, and the profile is pinned by name here besides, so
a machine that reports a different count cannot silently swap in ``p150x8``.

Run it (ttnn venv, on the Galaxy)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache DEEPSEEK_V4_MAX_NEW_TOKENS=64 \\
    pytest -s models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo_galaxy32.py

The first run pays for packing every chip's fused weight tensor; set
``DEEPSEEK_V4_CACHE_DIR`` so later runs read the packed tiles back instead of
re-reading the checkpoint.

``DEEPSEEK_V4_DECODE_LAYERS=N`` caps the stack for a smaller bring-up run, but
note that the placement only doubles up once the layers outnumber the chips: 8
layers over 22 chips is 8 chips holding one layer each, which never exercises the
two-layer packing. Pair it with ``DEEPSEEK_V4_PIPELINE_MAX_DEVICES=ceil(N/2)`` to
keep the 2-per-chip shape this profile is built around.
"""

from __future__ import annotations

import contextlib
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.tt.l1_placement import (
    L1_PER_CORE,
    RESIDENT_LAYERS,
    ZONES,
    budget_report,
)
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config

from models.experimental.deepseek_v4_flash.tests.test_full_model_decode_demo import (
    _DEFAULT_TEXT,
    _build_and_prefill,
    _checkpoint_available,
    _region,
)

_PROFILE = "galaxy32"
# Read at import so the mesh is opened with exactly the fabric / queue settings the
# profile asks for, rather than a hardcoded copy of them that can drift.
_GALAXY32 = load_system_config(profile=_PROFILE)


def _log_packed_l1_budget(model, config) -> None:
    """Log each chip's resident layers and check their L1 budget on the host.

    The packed tensor is one allocation per chip, so an overflowing pair shows up
    on device as a circular-buffer/L1 clash from whichever op happens to be built
    next -- far from the actual cause. ``budget_report`` is pure arithmetic over
    the same placement the packing uses, so run it first and fail here instead.
    """
    by_device: dict[int, list[int]] = {}
    for li, dev in enumerate(model.layer_submesh_ids):
        by_device.setdefault(dev, []).append(li)

    logger.info(f"packed L1 weights: {len(by_device)} chips, {model.num_layers} layers")
    worst = 0
    for dev, layers in sorted(by_device.items()):
        kinds = tuple(config.layer_types[li] for li in layers)
        note = ""
        if len(kinds) == RESIDENT_LAYERS:
            report = budget_report(kinds)
            per_core = max(z["per_core_bytes"] for z in report.values())
            worst = max(worst, per_core)
            tightest = min(report.items(), key=lambda kv: kv[1]["free_bytes"])
            note = (
                f"  peak {per_core / 1024:.0f} KB/core, "
                f"tightest {tightest[0]} free {tightest[1]['free_bytes'] / 1024:.0f} KB"
            )
        logger.info(f"  chip {dev:2d}: layers {layers} {'+'.join(k.split('_')[0] for k in kinds)}{note}")
    if worst:
        logger.info(
            f"worst chip holds {worst / 1024:.0f} KB/core of resident weights, "
            f"{(L1_PER_CORE - worst) / 1024:.0f} KB/core left for activations and CBs "
            f"(zones: {', '.join(f'{z}={c}c' for z, (_, c) in ZONES.items())})"
        )


@pytest.mark.skipif(not _checkpoint_available(), reason="V4-Flash checkpoint not found")
@pytest.mark.timeout(21600)  # heavy: packs 22 chips' fused weights + every expert in bf4
@torch.no_grad()
@pytest.mark.parametrize("device_params", [_GALAXY32.device.device_params()], indirect=True, ids=["galaxy32"])
@pytest.mark.parametrize("mesh_device", [32], indirect=True, ids=["32chip"])
@pytest.mark.parametrize("text", (_DEFAULT_TEXT,))
def test_galaxy32_packed_l1_decode_demo(mesh_device, reset_seeds, text: str) -> None:
    """Full-stack decode on 22 of the Galaxy's 32 chips, weights resident in L1."""
    system_config = load_system_config(profile=_PROFILE, mesh_device=mesh_device).log()
    # Pinned by name above, so these are assertions about the profile's contents
    # rather than about profile selection -- an edit that turned packing off (or
    # the prefetcher back on) would otherwise quietly run the streamed path here.
    assert system_config.decode.packed_l1_weights, "the galaxy32 profile must enable packed L1 weights"
    assert system_config.prefetcher.enabled is False, "packed L1 weights and the prefetcher are exclusive"
    assert system_config.decode.ttnn_weight_dtype == ttnn.bfloat4_b, "the packed placement is sized for bf4"

    with contextlib.ExitStack() as prefetcher:
        state = _build_and_prefill(mesh_device, text, prefetcher, system_config=system_config)
        model, lm_head, tokenizer = state["model"], state["lm_head"], state["tokenizer"]
        real_len, max_seq = state["real_len"], state["max_seq"]
        max_new_tokens, eos_id = state["max_new_tokens"], state["eos_id"]
        traced, next_id = state["traced"], state["next_id"]

        assert model.use_packed_l1_weights, "model did not take the packed L1 weight path"
        _log_packed_l1_budget(model, state["config"])
        logger.info(
            f"pipeline: {model.pipeline_stages} stages over {model.mesh_devices} chips, "
            f"{len(model.pipeline_edges)} socket hops per token"
        )

        generated: list[int] = [next_id]
        decode_tokens = 0
        decode_time = 0.0
        for step in range(1, max_new_tokens):
            if next_id == eos_id:
                logger.info("hit EOS; stopping")
                break
            pos = real_len + step - 1
            if pos >= max_seq:
                logger.warning(f"hit max RoPE length {max_seq}; stopping at {len(generated)} tokens")
                break
            t0 = time.perf_counter()
            if traced:
                logits = model.decode_traced(next_id, pos).reshape(1, -1).float()
            else:
                hidden = model.decode(next_id, pos, state["rope"])
                with _region("LM_HEAD"):
                    logits = ttnn.to_torch(lm_head(hidden)).reshape(1, -1).float()
            next_id = int(logits[0].argmax().item())
            decode_time += time.perf_counter() - t0
            decode_tokens += 1
            generated.append(next_id)
            logger.info(f"step {step:3d} (pos {pos:4d}): token id {next_id} {tokenizer.decode([next_id])!r}")

            if decode_tokens % 10 == 0:
                logger.info(
                    f"decode throughput: {decode_tokens / decode_time:.2f} tok/s/user "
                    f"({decode_tokens} tokens in {decode_time:.2f}s)"
                )
                decode_tokens = 0
                decode_time = 0.0

    if decode_tokens:
        logger.info(
            f"decode throughput (final): {decode_tokens / decode_time:.2f} tok/s/user "
            f"({decode_tokens} tokens in {decode_time:.2f}s)"
        )

    assert generated, "no tokens were generated"
    logger.info(f"PROMPT    : {tokenizer.decode(state['prompt_ids'])!r}")
    logger.info(f"GENERATED : {tokenizer.decode(generated)!r}  ({len(generated)} tokens)")
