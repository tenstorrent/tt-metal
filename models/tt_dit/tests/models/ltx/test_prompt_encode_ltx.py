# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Prompt-conditioning guard for the Gemma text encode.

An encoder that ignores its prompt does not fail: the DiT reads an all-zero context, denoises
unconditionally, and returns a plausible clip — the same clip for every prompt. Nothing downstream
can tell that apart from a working render, so it has to be asserted here.

Two invariants the existing tests could not see:

- **Distinct prompts must give distinct embeddings.** ``test_gemma_full`` encodes one prompt and
  PCC-checks it; a prompt-independent encoder passes that unchanged.
- **The weights must come from the cache the *pipeline* uses.** The encoder-side cache key is
  derived from ``checkpoint_name`` (``encoder_pair._load_connector``), so a test that builds the
  pair with ``checkpoint_name=None`` loads a *different* ``ltx-connectors`` cache and never touches
  the one the pipeline reads. A stale/poisoned pipeline cache stays invisible to it. This test
  therefore passes the real checkpoint, so it loads exactly what the pipeline loads.

Encoder-only (no DiT/VAE), so it is cheap enough to gate on.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.ltx.pipeline_ltx import LTXPipeline
from ....utils.ltx import DEFAULT_LTX_PROMPT, default_ltx_checkpoint, default_ltx_gemma

# Shares no subject, setting, or palette with DEFAULT_LTX_PROMPT (a woman with an acoustic
# guitar), so an identical embedding can only mean the encoder ignored its input.
CONTRAST_PROMPT = (
    "A red vintage sports car speeds along a coastal cliff road at sunset. Waves crash against "
    "the rocks far below as the camera tracks the car from the side, low golden light flaring "
    "across the windshield."
)


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param((4, 8), {"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="4x8")],
    indirect=["mesh_device", "device_params"],
)
def test_prompt_conditioning(mesh_device):
    """Two different prompts must encode to two different, finite, non-zero embeddings."""
    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    gemma = default_ltx_gemma()

    # checkpoint_name=ckpt (not None): keys the feature-extractor/connector caches exactly as the
    # pipeline does, so this loads the same weights the pipeline will. The transformer/VAE stay
    # unloaded — they are built lazily, so this is still encoder-only.
    # dynamic_load=False keeps the encoder resident, which is what turns the whole-encode trace on.
    pipe = LTXPipeline.create_pipeline(
        mesh_device, checkpoint_name=ckpt, gemma_path=gemma, mode="av", dynamic_load=False
    )
    pair = pipe.gemma_encoder_pair
    pair.ensure_loaded()
    assert pair._encoder_trace, "resident encoder must trace, else this guards nothing"

    # The pipeline's exact order: "warmup" captures the trace, the real prompts replay it.
    # use_cache=False so this measures the encoder, not the embedding disk cache.
    warm = pipe.encode_prompts(["warmup"], use_cache=False)[0][0]
    traced_a = pipe.encode_prompts([DEFAULT_LTX_PROMPT], use_cache=False)[0][0]
    traced_b = pipe.encode_prompts([CONTRAST_PROMPT], use_cache=False)[0][0]

    # Untraced references from the same weights: release the capture, then re-encode eagerly.
    type(pair)._encode_device._tracers[pair].release_trace()
    pair._encoder_trace = False
    eager_a = pipe.encode_prompts([DEFAULT_LTX_PROMPT], use_cache=False)[0][0]

    for name, t in (("warmup", warm), ("A", traced_a), ("B", traced_b)):
        logger.info(f"{name:6s} |max|={t.abs().max().item():.4f} finite={bool(torch.isfinite(t).all())}")
    for name, d in (
        ("traced  A vs B", _maxdiff(traced_a, traced_b)),
        ("traced  A vs warmup", _maxdiff(traced_a, warm)),
        ("traced  B vs warmup", _maxdiff(traced_b, warm)),
        ("traced A vs eager A", _maxdiff(traced_a, eager_a)),
    ):
        logger.info(f"{name:22s}: max|Δ| = {d:.6f}")

    # Corrupt aggregate weights overflow the 188160-deep projection to inf, which the connector
    # then collapses to an all-zero embedding — the DiT's unconditional input.
    for name, t in (("warmup", warm), ("A", traced_a), ("B", traced_b)):
        assert torch.isfinite(t).all(), f"{name}: embedding has inf/nan — check the encoder weight cache"
        assert torch.any(t), f"{name}: embedding is all zeros — the DiT would ignore the prompt"

    assert _maxdiff(traced_a, traced_b) > 0.0, "different prompts gave identical embeddings"
    assert _maxdiff(traced_a, warm) > 0.0, "prompt A got the warmup embedding (stale traced input?)"
    assert _maxdiff(traced_b, warm) > 0.0, "prompt B got the warmup embedding (stale traced input?)"
    # Same graph, same weights, deterministic device math: the replay must reproduce the eager encode.
    assert _maxdiff(traced_a, eager_a) == 0.0, "traced replay disagrees with the untraced encode"
