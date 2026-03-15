# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""test_demo.py — CI test matrix for the Kimi K2.5 demo.

Test cases
----------

smoke_random_weights
    Random-weights 2-layer smoke test.  No real checkpoint required.
    Validates that the hardware pipeline (mesh init, forward pass, decode loop)
    runs without errors.  Suitable for per-commit gate checks on any TG system.

tg_light
    Light TG demo: 32 prompts, 5 layers, 32 new tokens, trace enabled.
    Verifies the full model pipeline with real weights on a single TG pass.
    Used for post-merge sanity on main.

tg_full
    Full TG demo: 56 prompts, all 61 layers, 128 new tokens, repeated 2×.
    Stress test for throughput and stability.
    Used for nightly benchmarking and release validation.

Running locally::

    # Random-weights smoke (no real weights needed):
    MESH_DEVICE=TG pytest models/demos/kimi_k25/demo/test_demo.py \\
        -k smoke_random_weights -v

    # Light TG test with real weights:
    MESH_DEVICE=TG KIMI_HF_MODEL=/workspace/extra/Kimi-K2.5 \\
      KIMI_CACHE=/workspace/extra/kimi_cache \\
      pytest models/demos/kimi_k25/demo/test_demo.py \\
        -k tg_light -v

    # Full TG test:
    MESH_DEVICE=TG KIMI_HF_MODEL=/workspace/extra/Kimi-K2.5 \\
      KIMI_CACHE=/workspace/extra/kimi_cache \\
      pytest models/demos/kimi_k25/demo/test_demo.py \\
        -k tg_full -v --timeout=3600
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.demos.kimi_k25.demo.demo import load_prompts_from_json, run_demo

# ---------------------------------------------------------------------------
# Model/cache path resolution (env vars override defaults)
# ---------------------------------------------------------------------------

_MODEL_PATH = Path(os.getenv("KIMI_HF_MODEL", "/workspace/extra/Kimi-K2.5"))
_CACHE_DIR = Path(os.getenv("KIMI_CACHE", "/workspace/extra/kimi_cache"))
_PROMPTS_FILE = Path(__file__).parent / "test_prompts.json"


# ---------------------------------------------------------------------------
# Parametrise test cases
# ---------------------------------------------------------------------------


def _case(
    *,
    random_weights: bool,
    override_num_layers: int | None,
    max_new_tokens: int,
    repeat_batches: int,
    enable_trace: bool,
    max_prompts: int | None,
    case_id: str,
    marks=None,
) -> pytest.param:
    return pytest.param(
        {
            "random_weights": random_weights,
            "override_num_layers": override_num_layers,
            "max_new_tokens": max_new_tokens,
            "repeat_batches": repeat_batches,
            "enable_trace": enable_trace,
            "max_prompts": max_prompts,
        },
        id=case_id,
        marks=marks or [],
    )


# Test matrix:
#
# | id                   | random_weights | layers | new_tokens | repeat | trace | max_prompts |
# |----------------------|----------------|--------|------------|--------|-------|-------------|
# | smoke_random_weights | True           | 2      | 4          | 1      | True  | N/A         |
# | tg_light             | False          | 5      | 32         | 1      | True  | 8           |
# | tg_full              | False          | None   | 128        | 2      | True  | 56          |


@pytest.mark.parametrize(
    "case",
    [
        _case(
            random_weights=True,
            override_num_layers=2,
            max_new_tokens=4,
            repeat_batches=1,
            enable_trace=True,
            max_prompts=None,
            case_id="smoke_random_weights",
            marks=pytest.mark.requires_device(["TG"]),
        ),
        _case(
            random_weights=False,
            override_num_layers=5,
            max_new_tokens=32,
            repeat_batches=1,
            enable_trace=True,
            max_prompts=8,
            case_id="tg_light",
            marks=[
                pytest.mark.requires_device(["TG"]),
                pytest.mark.timeout(600),
            ],
        ),
        _case(
            random_weights=False,
            override_num_layers=None,
            max_new_tokens=128,
            repeat_batches=2,
            enable_trace=True,
            max_prompts=56,
            case_id="tg_full",
            marks=[
                pytest.mark.requires_device(["TG"]),
                pytest.mark.timeout(3600),
            ],
        ),
    ],
)
def test_kimi_demo(case: dict) -> None:
    """Run the Kimi K2.5 demo for the parametrised case and verify outputs."""
    random_weights: bool = case["random_weights"]
    override_num_layers: int | None = case["override_num_layers"]
    max_new_tokens: int = case["max_new_tokens"]
    repeat_batches: int = case["repeat_batches"]
    enable_trace: bool = case["enable_trace"]
    max_prompts: int | None = case["max_prompts"]

    # Validate model path exists before spending time opening a mesh device
    if not random_weights:
        if not _MODEL_PATH.exists():
            pytest.skip(
                f"Kimi K2.5 model not found at '{_MODEL_PATH}'. "
                f"Set KIMI_HF_MODEL to the checkpoint directory."
            )

    # Load prompts
    if random_weights:
        prompts = None
    elif _PROMPTS_FILE.exists():
        prompts = load_prompts_from_json(str(_PROMPTS_FILE), max_prompts=max_prompts)
    else:
        # Fallback: minimal prompts for testing
        prompts = [
            "What is Tenstorrent?",
            "Explain MoE routing in one sentence.",
            "Write a haiku about silicon.",
            "What is 1337 in hex?",
        ]
        if max_prompts is not None:
            prompts = prompts[:max_prompts]

    results = run_demo(
        prompts,
        model_path=_MODEL_PATH,
        max_new_tokens=max_new_tokens,
        cache_dir=_CACHE_DIR,
        random_weights=random_weights,
        override_num_layers=override_num_layers,
        enable_trace=enable_trace,
        repeat_batches=repeat_batches,
        sample_on_device=True,
    )

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert "generations" in results, "run_demo must return a 'generations' key"
    assert len(results["generations"]) > 0, "At least one generation expected"

    for i, gen_result in enumerate(results["generations"]):
        assert "tokens" in gen_result, f"generation[{i}] must have 'tokens'"
        tokens = gen_result["tokens"]
        assert isinstance(tokens, list), f"generation[{i}]['tokens'] must be a list"
        assert len(tokens) > 0, f"generation[{i}] produced no tokens"

        # In full-model mode the tokenizer should decode to a non-empty string
        if not random_weights and gen_result.get("text") is not None:
            assert isinstance(gen_result["text"], str), (
                f"generation[{i}]['text'] must be str, got {type(gen_result['text'])}"
            )
            # Decoded text may be empty for short generation runs — only check it's not None
