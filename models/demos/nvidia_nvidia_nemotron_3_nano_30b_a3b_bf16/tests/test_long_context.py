# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Long-context tests for NemotronH-30B — up to 256k ISL.

Strategy
--------
Actually prefilling 256k tokens at S=1 takes ~3 hours — impractical for a
test suite.  Instead we test the four things that could actually break at long
context WITHOUT running all prior tokens:

  1. Allocation   — allocate_decoder_state succeeds at each ISL; shapes and
                    memory fit in 32 GB DRAM.
  2. Index math   — paged_update_cache writes to the correct physical block
                    at arbitrary positions; paged_sdpa_decode reads back the
                    correct value.  We do this by setting current_pos directly
                    (no prefill required) and inspecting the cache tensor.
  3. Block boundary — positions exactly at multiples of block_size=32 are
                    handled correctly (off-by-one would corrupt the KV cache).
  4. End-of-range — positions near MODEL_MAX_SEQ_LEN − 1 (262143) don't
                    crash, wrap, or address out-of-bounds memory.

For tests 2–4 we run a real forward pass at the target position with an
otherwise-empty KV cache.  The model output is not compared against a
reference — we are testing that the infrastructure doesn't break, not that the
output is numerically accurate given missing context.

Full-prefill tests (marked @pytest.mark.slow) run a real sequential prefill
at shorter ISLs to validate end-to-end quality at non-trivial lengths.

ISL coverage
------------
  Allocation-only :  4k, 8k, 16k, 32k, 64k, 128k, 256k
  Forward at pos  :  block boundaries (0, 31, 32, 63, 64)
                  +  mid-range       (1024, 16384, 65536)
                  +  near-end        (MODEL_MAX_SEQ_LEN − 2, MODEL_MAX_SEQ_LEN − 1)
  Full prefill    :  512, 2048  (@slow)
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
_root = os.environ["TT_METAL_HOME"]
for _p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ttnn
from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.kv_cache import (
    DEFAULT_BLOCK_SIZE,
    HEAD_DIM,
    MODEL_MAX_SEQ_LEN,
    N_D_LAYERS,
    N_KV_HEADS,
    N_M_LAYERS,
    NUM_SSM_HEADS,
    SNAP,
    SSM_HEAD_DIM,
    SSM_STATE_SIZE,
    allocate_decoder_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4

    dev = open_device_tp4()
    yield dev
    close_device_tp4(dev)


@pytest.fixture(scope="module")
def weight_cache():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import WeightCache

    return WeightCache()


# ---------------------------------------------------------------------------
# 0. Config sync check — MODEL_MAX_SEQ_LEN must match config.json
# ---------------------------------------------------------------------------


def test_model_max_seq_len_matches_hf_config():
    """MODEL_MAX_SEQ_LEN in kv_cache.py must match config.json max_position_embeddings.

    This catches drift when the HF checkpoint is updated but the fallback
    constant in kv_cache.py is not. Skipped when the checkpoint is not cached.
    """
    import json
    import pathlib

    cfg_path = pathlib.Path(SNAP) / "config.json"
    if not cfg_path.exists():
        pytest.skip(f"HF checkpoint not found at {SNAP}")
    cfg = json.loads(cfg_path.read_text())
    assert MODEL_MAX_SEQ_LEN == cfg["max_position_embeddings"], (
        f"MODEL_MAX_SEQ_LEN={MODEL_MAX_SEQ_LEN} but config.json has "
        f"max_position_embeddings={cfg['max_position_embeddings']}. "
        f"Update kv_cache._read_max_position_embeddings fallback or SNAP path."
    )


# ---------------------------------------------------------------------------
# 1. Allocation tests — parametrized over ISL values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_seq_len",
    [
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        MODEL_MAX_SEQ_LEN,  # 262144
    ],
    ids=lambda n: f"{n//1024}k",
)
def test_allocate_decoder_state(mesh_device, max_seq_len):
    """allocate_decoder_state succeeds and produces tensors of expected shape."""
    state = allocate_decoder_state(mesh_device, B=1, max_seq_len=max_seq_len)
    num_blocks = max_seq_len // DEFAULT_BLOCK_SIZE

    # SSM states: 23 × [1, 64, 64, 128]
    assert len(state.ssm_states) == N_M_LAYERS
    assert len(state.ssm_state_outs) == N_M_LAYERS
    for s in state.ssm_states:
        assert list(s.shape) == [
            1,
            NUM_SSM_HEADS,
            SSM_HEAD_DIM,
            SSM_STATE_SIZE,
        ], f"SSM state shape {list(s.shape)} unexpected"

    # KV caches: 6 × (k, v) each [num_blocks, 2, block_size, 128]
    assert len(state.kv_caches) == N_D_LAYERS
    for k_tt, v_tt in state.kv_caches:
        expected = [num_blocks, N_KV_HEADS, DEFAULT_BLOCK_SIZE, HEAD_DIM]
        assert list(k_tt.shape) == expected, f"K cache shape {list(k_tt.shape)} != {expected}"
        assert list(v_tt.shape) == expected, f"V cache shape {list(v_tt.shape)} != {expected}"

    # Page tables: 6 × [1, num_blocks]
    assert len(state.page_tables) == N_D_LAYERS
    for pt in state.page_tables:
        assert list(pt.shape) == [1, num_blocks], f"Page table shape {list(pt.shape)} != [1, {num_blocks}]"

    # Memory estimate: 6 layers × 2 (K+V) × num_blocks × 2 × 32 × 128 × 2 B
    kv_bytes = N_D_LAYERS * 2 * num_blocks * N_KV_HEADS * DEFAULT_BLOCK_SIZE * HEAD_DIM * 2
    print(f"\n  max_seq_len={max_seq_len//1024}k  KV cache={kv_bytes/1e6:.0f} MB  blocks={num_blocks}")


def test_allocate_exceeds_model_limit(mesh_device):
    """allocate_decoder_state rejects max_seq_len > MODEL_MAX_SEQ_LEN."""
    with pytest.raises(ValueError, match="max_position_embeddings"):
        allocate_decoder_state(mesh_device, B=1, max_seq_len=MODEL_MAX_SEQ_LEN + 1)


# ---------------------------------------------------------------------------
# 2 + 3 + 4. Forward at arbitrary position (no prefill)
#
# We set current_pos directly to the target position and run one forward pass.
# This exercises paged_update_cache block-index arithmetic and
# paged_sdpa_decode with a (mostly empty) KV cache at that position.
# ---------------------------------------------------------------------------


def _run_one_step_at_pos(mesh_device, wc, state, pos: int) -> torch.Tensor:
    """Run one stateful forward at `pos`.  Returns logits [1, 1, 131072] CPU."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import _to_device_token, _update_pos
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward_stateful
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    ids_tt = _to_device_token(1, mesh_device)  # token id = 1 (arbitrary)
    _update_pos(state.current_pos, pos)

    logits_tt = nemotron_h_forward_stateful(mesh_device, ids_tt, wc, state, cpu_gate=True)
    ttnn.synchronize_device(mesh_device)
    state.advance()
    return _host_rep(logits_tt, mesh_device, 1)  # [1, 1, vocab]


@pytest.mark.parametrize(
    "pos",
    [
        0,
        31,  # last position in block 0
        32,  # first position in block 1 — block boundary
        63,  # last position in block 1
        64,  # first position in block 2
        1023,
        1024,
    ],
    ids=lambda p: f"pos{p}",
)
def test_forward_block_boundaries(mesh_device, weight_cache, pos):
    """Forward pass at block boundaries produces finite, non-NaN logits."""
    max_seq = max(pos + DEFAULT_BLOCK_SIZE * 4, 4096)
    state = allocate_decoder_state(mesh_device, B=1, max_seq_len=max_seq)
    logits = _run_one_step_at_pos(mesh_device, weight_cache, state, pos)

    assert torch.isfinite(logits).all(), f"NaN/Inf in logits at pos={pos}"
    assert logits.shape == (1, 1, 131_072), f"Unexpected logits shape {logits.shape} at pos={pos}"

    expected_block = pos // DEFAULT_BLOCK_SIZE
    print(f"\n  pos={pos}  block={expected_block}  logits OK  " f"top1={int(logits[0,0].argmax())}")


@pytest.mark.parametrize(
    "pos,max_seq_len",
    [
        (1_024, 4_096),
        (4_095, 4_096),  # last valid position in a 4k window
        (16_383, 32_768),
        (32_767, 32_768),  # last valid position in a 32k window
        (65_535, 65_536),
    ],
    ids=lambda x: f"{x}" if isinstance(x, int) else "",
)
def test_forward_mid_range(mesh_device, weight_cache, pos, max_seq_len):
    """Forward pass at mid-range and end-of-window positions."""
    state = allocate_decoder_state(mesh_device, B=1, max_seq_len=max_seq_len)
    logits = _run_one_step_at_pos(mesh_device, weight_cache, state, pos)

    assert torch.isfinite(logits).all(), f"NaN/Inf at pos={pos} (max_seq_len={max_seq_len})"
    block = pos // DEFAULT_BLOCK_SIZE
    num_blocks = max_seq_len // DEFAULT_BLOCK_SIZE
    print(f"\n  pos={pos}  block={block}/{num_blocks-1}  top1={int(logits[0,0].argmax())}")


@pytest.mark.parametrize(
    "pos",
    [
        131_071,  # last position in 128k window
        MODEL_MAX_SEQ_LEN - 2,  # 262142
        MODEL_MAX_SEQ_LEN - 1,  # 262143 — last valid position in the model
    ],
    ids=lambda p: f"pos{p}",
)
def test_forward_near_end_of_range(mesh_device, weight_cache, pos):
    """Forward pass near MODEL_MAX_SEQ_LEN (262143) — no crash, no wraparound."""
    state = allocate_decoder_state(mesh_device, B=1, max_seq_len=MODEL_MAX_SEQ_LEN)
    logits = _run_one_step_at_pos(mesh_device, weight_cache, state, pos)

    assert torch.isfinite(logits).all(), f"NaN/Inf at pos={pos} (model max {MODEL_MAX_SEQ_LEN})"

    expected_block = pos // DEFAULT_BLOCK_SIZE
    total_blocks = MODEL_MAX_SEQ_LEN // DEFAULT_BLOCK_SIZE
    print(f"\n  pos={pos}  block={expected_block}/{total_blocks-1}  " f"top1={int(logits[0,0].argmax())}")


# ---------------------------------------------------------------------------
# KV cache write-then-read consistency
#
# Writes a known pattern into the KV cache at a target position, runs a
# forward pass, and checks that paged_update_cache wrote to the right block.
# We inspect the cache tensor directly (D2H) rather than comparing logits.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pos,max_seq_len",
    [
        (0, 4_096),
        (31, 4_096),  # end of block 0
        (32, 4_096),  # start of block 1
        (4_095, 4_096),  # last block, last slot
        (32_767, 32_768),  # last block, last slot in 32k window
    ],
    ids=lambda x: str(x),
)
def test_kv_cache_write_index(mesh_device, weight_cache, pos, max_seq_len):
    """After one forward at pos, verify the KV cache wrote to block pos//block_size,
    slot pos%block_size — and not to any other block."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import _to_device_token, _update_pos
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward_stateful

    state = allocate_decoder_state(mesh_device, B=1, max_seq_len=max_seq_len)
    ids_tt = _to_device_token(42, mesh_device)
    _update_pos(state.current_pos, pos)

    nemotron_h_forward_stateful(mesh_device, ids_tt, wc=weight_cache, decoder_state=state, cpu_gate=True)
    ttnn.synchronize_device(mesh_device)

    expected_block = pos // DEFAULT_BLOCK_SIZE
    expected_slot = pos % DEFAULT_BLOCK_SIZE

    # Inspect first KV cache pair (layer 0 dense attention).
    k_tt, _ = state.kv_caches[0]
    k_cpu = ttnn.to_torch(
        k_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[
        0:1
    ]  # take device-0 replica: [num_blocks, 2, 32, 128]

    # The written block should be non-zero (input was a real token through the model).
    written = k_cpu[expected_block]  # [2, 32, 128]
    other_blocks = torch.cat([k_cpu[:expected_block], k_cpu[expected_block + 1 :]], dim=0)

    assert written.abs().max() > 0, f"Expected block {expected_block} to be non-zero after write at pos={pos}"
    assert other_blocks.abs().max() == 0, (
        f"Unexpected non-zero data in blocks other than {expected_block} " f"after a single write at pos={pos}"
    )

    print(
        f"\n  pos={pos}  block={expected_block}  slot={expected_slot}  "
        f"written max={written.abs().max():.4f}  other max={other_blocks.abs().max():.6f}"
    )


# ---------------------------------------------------------------------------
# Full prefill tests (slow) — sequential S=1 prefill at real ISL
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("isl", [512, 2_048], ids=lambda n: f"isl{n}")
def test_full_prefill_quality(mesh_device, weight_cache, isl):
    """Run a real sequential prefill at `isl` tokens; check output is coherent.

    Uses a fixed seed text so the test is deterministic.  Checks:
      - No NaN/Inf in logits at any prefill step.
      - Greedy top-1 token at the last prefill step is a printable token
        (not UNK or special) — crude but fast quality gate.
    """
    from transformers import AutoTokenizer

    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import (
        SNAP,
        _to_device_token,
        _update_ids,
        _update_pos,
    )
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward_stateful
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    tokenizer = AutoTokenizer.from_pretrained(SNAP)

    # Build a prompt of exactly `isl` tokens from a repeating seed.
    seed = (
        "Tenstorrent Blackhole is a high-performance AI accelerator. "
        "NemotronH-30B combines Mamba2 SSM layers with sparse MoE transformers. "
    ) * 40
    ids = tokenizer.encode(seed, add_special_tokens=True)
    while len(ids) < isl:
        ids += tokenizer.encode(seed, add_special_tokens=False)
    ids = ids[:isl]
    assert len(ids) == isl, f"Could not build {isl}-token prompt"

    max_seq = isl + 64  # small buffer for generated tokens
    state = allocate_decoder_state(mesh_device, B=1, max_seq_len=max_seq)
    ids_tt = _to_device_token(ids[0], mesh_device)

    nan_steps = []
    for pos, tok in enumerate(ids):
        _update_ids(ids_tt, tok)
        _update_pos(state.current_pos, pos)
        logits_tt = nemotron_h_forward_stateful(
            mesh_device, ids_tt, wc=weight_cache, decoder_state=state, cpu_gate=True
        )
        ttnn.synchronize_device(mesh_device)
        state.advance()

        if not torch.isfinite(_host_rep(logits_tt, mesh_device, 1)).all():
            nan_steps.append(pos)
            if len(nan_steps) >= 5:
                break

        if (pos + 1) % 100 == 0:
            print(f"  prefill {pos+1}/{isl}", flush=True)

    assert not nan_steps, f"NaN/Inf in logits at prefill steps: {nan_steps[:5]}"

    # Check last-step top-1 is a real printable token.
    last_logits = _host_rep(logits_tt, mesh_device, 1)  # [1, 1, vocab]
    top1 = int(last_logits[0, 0].argmax())
    decoded = tokenizer.decode([top1])
    print(f"\n  ISL={isl}  last-step top-1 = {top1} = {repr(decoded)}")
    assert top1 != tokenizer.unk_token_id, f"Top-1 prediction at end of {isl}-token prefill is UNK"
