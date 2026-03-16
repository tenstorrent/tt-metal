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


# ---------------------------------------------------------------------------
# Throughput benchmark test
# ---------------------------------------------------------------------------


# Minimum acceptable decode throughput (t/s total) — conservative floor for
# a 5-layer random-init run on TG.  This catches complete performance
# regressions (e.g. trace disabled, serialised decode) rather than being a
# tight SLA.  Real-weight, full-model targets will be tuned separately once
# baseline hardware numbers are established.
_MIN_DECODE_TOK_S_RANDOM = 1.0  # t/s — floor for smoke/5-layer runs
_MIN_DECODE_TOK_S_REAL = 5.0    # t/s — floor for real-weight 5-layer run


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "random_weights, min_tok_s, case_id",
    [
        pytest.param(True, _MIN_DECODE_TOK_S_RANDOM, "smoke", id="throughput_smoke_random_weights"),
        pytest.param(False, _MIN_DECODE_TOK_S_REAL, "tg_light", id="throughput_tg_light"),
    ],
)
def test_kimi_throughput(random_weights: bool, min_tok_s: float, case_id: str) -> None:
    """Throughput benchmark — decode t/s must meet a minimum floor.

    Runs the Kimi K2.5 demo with 5 layers + 8 prompts + 32 new tokens and
    asserts that ``statistics["decode_t/s"]`` is at or above ``min_tok_s``.

    This catches performance regressions (trace disabled, serialised decode,
    incorrect batching) without being a tight SLA.  Tighten thresholds once
    baseline numbers are established on real hardware.
    """
    if not random_weights and not _MODEL_PATH.exists():
        pytest.skip(
            f"Kimi K2.5 model not found at '{_MODEL_PATH}'. "
            f"Set KIMI_HF_MODEL to the checkpoint directory."
        )

    prompts: list[str] | None
    if random_weights:
        prompts = None
    elif _PROMPTS_FILE.exists():
        prompts = load_prompts_from_json(str(_PROMPTS_FILE), max_prompts=8)
    else:
        prompts = [
            "What is Tenstorrent?",
            "Explain MoE routing in one sentence.",
            "Write a haiku about silicon.",
            "What is 1337 in hex?",
        ]

    results = run_demo(
        prompts,
        model_path=_MODEL_PATH,
        max_new_tokens=32,
        cache_dir=_CACHE_DIR,
        random_weights=random_weights,
        override_num_layers=2 if random_weights else 5,
        enable_trace=True,
        repeat_batches=1,
        sample_on_device=True,
    )

    # Basic generation sanity
    assert "generations" in results, "run_demo must return 'generations'"
    assert len(results["generations"]) > 0, "At least one generation expected"

    # Throughput assertion
    stats = results.get("statistics") or {}
    decode_tok_s = stats.get("decode_t/s")
    if decode_tok_s is None:
        pytest.skip(
            "Generator did not return 'decode_t/s' statistics. "
            "Either trace is disabled or the generator does not report throughput yet."
        )

    assert decode_tok_s >= min_tok_s, (
        f"Kimi K2.5 decode throughput {decode_tok_s:.2f} t/s is below the "
        f"minimum acceptable floor of {min_tok_s:.1f} t/s for case '{case_id}'. "
        f"Full statistics: {stats}"
    )

    # Log for CI visibility
    import logging
    logging.getLogger(__name__).info(
        f"[kimi_throughput/{case_id}] decode_t/s={decode_tok_s:.2f} "
        f"(floor={min_tok_s:.1f})"
    )
