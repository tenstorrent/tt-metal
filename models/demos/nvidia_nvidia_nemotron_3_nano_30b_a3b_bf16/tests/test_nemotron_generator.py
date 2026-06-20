# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validates the NemotronHForCausalLM Generator class (vLLM / tt-inference-server interface).

Tests the full lifecycle exercised by the tt-inference-server:
  initialize_vllm_model → allocate_kv_cache → warmup_model_prefill →
  warmup_model_decode → prefill_forward → decode_forward → reset_state →
  prefill_forward (second request, trace still valid after reset).

Key regressions caught:
  - prefill_forward must use nemotron_h_prefill_stateful (bulk path), not the
    S=1 token loop.  The S=1 loop is ~200× slower at ISL=256 and does not fill
    the dense-attention KV cache from paged_fill_cache.
  - reset_state must call reset_inplace(), not allocate_decoder_state().
    Reallocation invalidates the decode trace's DRAM buffer references.
  - warmup_model_prefill must compile the chunked-prefill kernels, not only the
    S=1 path.  Missing warmup causes a multi-second compile pause on the first
    real request.

Usage:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=ttnn:tools:.
    export LD_LIBRARY_PATH=build_Release/lib:/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
    pytest models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_nemotron_generator.py -v -s --noconftest
"""

import os
import sys
import time

os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
_root = os.environ["TT_METAL_HOME"]
for p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch

VOCAB_SIZE = 131_072
PROMPT_LEN = 32  # short prompt to keep test runtime manageable
DECODE_STEPS = 5  # number of decode tokens to generate per request


@pytest.fixture(scope="module")
def mesh_device():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4

    dev = open_device_tp4()
    yield dev
    close_device_tp4(dev)


@pytest.fixture(scope="module")
def generator(mesh_device):
    """Full lifecycle setup: initialize → allocate → warmup_prefill → warmup_decode."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.nemotron_generator import NemotronHForCausalLM

    gen = NemotronHForCausalLM.initialize_vllm_model(
        hf_config=None,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=4096,
    )
    kv_cache = gen.allocate_kv_cache(max_batch_size=1, max_seq_len=4096)

    print("\n[test] Running warmup_model_prefill...", flush=True)
    gen.warmup_model_prefill(kv_cache)

    print("[test] Running warmup_model_decode...", flush=True)
    gen.warmup_model_decode(kv_cache, enable_trace=True)

    yield gen


def _dummy_prompt(length: int = PROMPT_LEN) -> torch.Tensor:
    """Random token ids in [1, 1000] — avoids padding token 0."""
    return torch.randint(1, 1000, (1, length), dtype=torch.int64)


class TestGeneratorLifecycle:
    def test_prefill_forward_returns_correct_shape(self, generator):
        """prefill_forward must return [1, vocab_size] for B=1."""
        tokens = _dummy_prompt(PROMPT_LEN)
        generator.reset_state()

        logits = generator.prefill_forward(tokens)

        assert logits.shape == (1, VOCAB_SIZE), f"Expected prefill logits shape (1, {VOCAB_SIZE}), got {logits.shape}"
        assert logits.dtype == torch.bfloat16, f"Expected bfloat16, got {logits.dtype}"

    def test_prefill_forward_finite_logits(self, generator):
        """Bulk prefill must not produce NaN or Inf."""
        tokens = _dummy_prompt(PROMPT_LEN)
        generator.reset_state()

        logits = generator.prefill_forward(tokens)

        assert torch.isfinite(logits).all(), "prefill_forward produced NaN or Inf logits"

    def test_decode_forward_returns_correct_shape(self, generator):
        """decode_forward must return [1, 1, vocab_size] for B=1."""
        tokens = _dummy_prompt(PROMPT_LEN)
        generator.reset_state()
        generator.prefill_forward(tokens)

        next_token = torch.randint(1, 1000, (1, 1), dtype=torch.int64)
        current_pos = torch.tensor([PROMPT_LEN], dtype=torch.int64)
        logits = generator.decode_forward(next_token, current_pos=current_pos)

        assert logits.shape == (
            1,
            1,
            VOCAB_SIZE,
        ), f"Expected decode logits shape (1, 1, {VOCAB_SIZE}), got {logits.shape}"

    def test_decode_forward_finite_logits(self, generator):
        """Traced decode must not produce NaN or Inf."""
        tokens = _dummy_prompt(PROMPT_LEN)
        generator.reset_state()
        generator.prefill_forward(tokens)

        for step in range(DECODE_STEPS):
            next_tok = torch.randint(1, 1000, (1, 1), dtype=torch.int64)
            pos = torch.tensor([PROMPT_LEN + step], dtype=torch.int64)
            logits = generator.decode_forward(next_tok, current_pos=pos)
            assert torch.isfinite(logits).all(), f"NaN/Inf at decode step {step}"

    def test_decode_positions_advance(self, generator):
        """decode_pos must increment correctly after each decode step."""
        tokens = _dummy_prompt(PROMPT_LEN)
        generator.reset_state()
        generator.prefill_forward(tokens)

        assert generator._decode_pos == PROMPT_LEN, (
            f"After prefill of {PROMPT_LEN} tokens, _decode_pos should be {PROMPT_LEN}, " f"got {generator._decode_pos}"
        )

        for step in range(DECODE_STEPS):
            next_tok = torch.randint(1, 1000, (1, 1), dtype=torch.int64)
            generator.decode_forward(next_tok)
            expected_pos = PROMPT_LEN + step + 1
            assert (
                generator._decode_pos == expected_pos
            ), f"At step {step}: expected _decode_pos={expected_pos}, got {generator._decode_pos}"

    def test_reset_state_does_not_invalidate_trace(self, generator):
        """reset_state() must preserve trace validity — trace must work on second request."""
        # First request
        tokens1 = _dummy_prompt(PROMPT_LEN)
        generator.reset_state()
        generator.prefill_forward(tokens1)
        first_tok = torch.randint(1, 1000, (1, 1), dtype=torch.int64)
        logits1 = generator.decode_forward(first_tok)

        # reset_state() — previously used allocate_decoder_state() which broke trace
        generator.reset_state()
        assert generator._prefill_pos == 0, "reset_state must zero _prefill_pos"
        assert generator._decode_pos == 0, "reset_state must zero _decode_pos"
        assert generator._trace_id is not None, "reset_state must NOT release the trace"

        # Second request — trace must still be valid
        tokens2 = _dummy_prompt(PROMPT_LEN)
        logits2_prefill = generator.prefill_forward(tokens2)
        assert torch.isfinite(
            logits2_prefill
        ).all(), "Second request prefill has NaN/Inf — state corruption after reset_state"

        second_tok = torch.randint(1, 1000, (1, 1), dtype=torch.int64)
        logits2 = generator.decode_forward(second_tok)
        assert torch.isfinite(logits2).all(), "Second request decode has NaN/Inf — trace broken by reset_state"

    def test_reset_state_produces_different_output_from_different_input(self, generator):
        """Two different prompts should produce different logits after reset."""
        tokens_a = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
        tokens_b = torch.tensor([[100, 200, 300, 400, 500]], dtype=torch.int64)

        generator.reset_state()
        logits_a = generator.prefill_forward(tokens_a)

        generator.reset_state()
        logits_b = generator.prefill_forward(tokens_b)

        assert not torch.allclose(
            logits_a, logits_b, atol=1e-2
        ), "Different prompts produced identical logits — reset_state may not be zeroing state"

    def test_multiple_reset_cycles(self, generator):
        """Three full prefill+decode+reset cycles must all produce finite logits."""
        for cycle in range(3):
            generator.reset_state()
            tokens = _dummy_prompt(PROMPT_LEN + cycle * 8)
            prefill_logits = generator.prefill_forward(tokens)
            assert torch.isfinite(prefill_logits).all(), f"cycle {cycle}: NaN in prefill"

            for step in range(3):
                tok = torch.randint(1, 1000, (1, 1), dtype=torch.int64)
                decode_logits = generator.decode_forward(tok)
                assert torch.isfinite(decode_logits).all(), f"cycle {cycle} step {step}: NaN in decode"

    def test_prefill_throughput_regression(self, generator):
        """Bulk prefill must be meaningfully faster than a S=1 loop would be.

        For a 256-token prompt on QB TP=4, the chunked SSD scan completes in
        ~5 s total (all 52 layers + compile).  The S=1 loop would take ~200 s
        (256 tokens × ~0.8 s/token eager).  We gate on < 120 s to catch a
        regression to the S=1 loop while leaving room for slower machines.
        """
        TIMING_PROMPT_LEN = 256
        tokens = _dummy_prompt(TIMING_PROMPT_LEN)
        generator.reset_state()

        t0 = time.perf_counter()
        generator.prefill_forward(tokens)
        elapsed = time.perf_counter() - t0

        print(
            f"\n[timing] prefill {TIMING_PROMPT_LEN} tokens: {elapsed:.1f}s ({elapsed/TIMING_PROMPT_LEN*1000:.0f} ms/tok)"
        )
        assert elapsed < 120.0, (
            f"prefill of {TIMING_PROMPT_LEN} tokens took {elapsed:.1f}s — "
            f"likely regressed to S=1 token loop (should be <120s on QB TP=4)"
        )
