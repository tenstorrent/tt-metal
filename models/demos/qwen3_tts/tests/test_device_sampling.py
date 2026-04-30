# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Smoke test the on-device top-k sampling pipeline for Qwen3-TTS decode.

Verifies that:
  ttnn.topk(logits, k) → (values, indices)
  ttnn.sampling(values, indices_rm, k_tensor, p_tensor, temp_tensor, seed)
returns a valid token id within the vocab range, and that it sits inside the
expected top-k set of the input logits.

    pytest -q models/demos/qwen3_tts/tests/test_device_sampling.py
"""
import pytest
import torch

import ttnn


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def test_device_sampling_cp_decode_shape(device):
    torch.manual_seed(42)
    vocab = 2048  # CP lm_head vocab — power of 2, no padding needed
    top_k_actual = 50
    max_top_k = 64  # ttnn.topk inner dim must be a multiple of 32; round up.
    temp = 0.9

    # Synthetic logits with one clear high-prob token to make sampling deterministic-ish.
    logits = torch.randn(1, 1, 1, vocab, dtype=torch.bfloat16) * 0.1
    # Bias one token strongly so the top-k always contains it.
    target_idx = 1234
    logits[0, 0, 0, target_idx] = 100.0

    # ttnn.sampling kernel requires exactly 32 users (input_shape[0]*[1]*[2] == 32).
    # We're batch=1 so replicate the row 32× before topk; the sampler's per-user output
    # at user[0] is what we want.
    logits_padded = logits.expand(1, 1, 32, vocab).contiguous()
    logits_tt = ttnn.from_torch(
        logits_padded,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Top-k values + indices
    topk_values_tt, topk_indices_tt = ttnn.topk(logits_tt, k=max_top_k, dim=-1, largest=True, sorted=True)
    print(f"topk_values shape={topk_values_tt.shape} dtype={topk_values_tt.dtype} layout={topk_values_tt.layout}")
    print(f"topk_indices shape={topk_indices_tt.shape} dtype={topk_indices_tt.dtype} layout={topk_indices_tt.layout}")

    # ttnn.sampling needs indices in INT32/UINT32 ROW_MAJOR.
    indices_rm = ttnn.to_layout(topk_indices_tt, ttnn.ROW_MAJOR_LAYOUT)
    indices_rm = ttnn.typecast(indices_rm, ttnn.int32)

    # 32-user param tensors (kernel reads per-user k/p/temp).
    k_tensor = ttnn.from_torch(
        torch.full((32,), top_k_actual, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    p_tensor = ttnn.from_torch(
        torch.full((32,), 1.0, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    temp_tensor = ttnn.from_torch(
        torch.full((32,), temp, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    out_tok_tt = ttnn.sampling(
        topk_values_tt,
        indices_rm,
        k=k_tensor,
        p=p_tensor,
        temp=temp_tensor,
        seed=12345,
    )
    print(f"out_tok shape={out_tok_tt.shape} dtype={out_tok_tt.dtype} layout={out_tok_tt.layout}")
    out = ttnn.to_torch(out_tok_tt).flatten()
    print(f"sampled token: {int(out[0].item())}")

    sampled = int(out[0].item())
    # Verify token is within vocab and in the actual top-k of the cpu reference.
    cpu_vals, cpu_idx = torch.topk(logits.float().squeeze(), k=top_k_actual)
    assert sampled in cpu_idx.tolist(), f"sampled token {sampled} not in CPU top-{top_k_actual}={cpu_idx.tolist()}"
    # And the strongly-biased token (target_idx) should have ~100% prob with temp=0.9.
    assert sampled == target_idx, f"expected biased token {target_idx}, got {sampled}"
