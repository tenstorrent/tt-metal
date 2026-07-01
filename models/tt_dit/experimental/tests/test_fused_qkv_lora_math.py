# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-only numerical correctness test for the fused-QKV LoRA reconstruction.

Wan attention fuses the separate Q/K/V (self-attn) and K/V (cross-attn)
projections into a single ``to_qkv`` / ``to_kv`` Linear whose output dim is
**head-interleaved** across tensor-parallel devices (see
``WanAttention._interleave_heads`` in
``models/tt_dit/models/transformers/wan2_2/attention_wan.py``). LoRA adapters
ship separate q/k/v pairs, so ``adapter_loader._register_fused_qkv`` hand-builds
a combined adapter (``A_fused`` stacked on rank, zero-padded block-diagonal
``B_fused`` head-interleaved via ``_head_interleave_lora_B``).

The fused LoRA delta added to the fused base weight MUST equal the per-source
deltas interleaved with the SAME permutation the base weight uses:

    scale·(B_fused @ A_fused)  ==  interleave([ scale·B_q@A_q, scale·B_k@A_k, scale·B_v@A_v ])

This is pure linear algebra — no device needed. Crucially we sweep ``n_dev`` so
the multi-device interleave path (production runs at TP=4) is exercised; the
existing structural test only runs at TP=1 where the interleave is near-trivial.

Run:
    ./python_env/bin/python3 -m pytest -xvs \
        models/tt_dit/experimental/tests/test_fused_qkv_lora_math.py
"""
from __future__ import annotations

import pytest
import torch

from models.tt_dit.experimental.lora.adapter_loader import _head_interleave_lora_B


def ref_interleave_heads(
    tensors: list[torch.Tensor],
    *,
    n_dev: int,
    n_local_heads: int,
    head_dim: int,
    num_heads: int,
) -> torch.Tensor:
    """Verbatim transcription of WanAttention._interleave_heads
    (attention_wan.py:186-206) — the authoritative base-weight out-dim layout.

    Inputs are [out=num_heads*head_dim, in] (PyTorch [out, in]). Returns the
    merged [len(tensors)*num_heads*head_dim, in] tensor.
    """
    # Transpose to [in, out]
    tensors = [t.T for t in tensors]
    # Reshape out dim to [in, n_dev, n_local_heads, head_dim]
    tensors = [t.reshape(t.shape[0], n_dev, n_local_heads, head_dim) for t in tensors]
    # Concatenate on the heads dim so each device shard gets its own heads from every tensor
    merged = torch.cat(tensors, dim=2)  # [in, n_dev, len(tensors)*n_local_heads, head_dim]
    merged = merged.reshape(merged.shape[0], len(tensors) * num_heads * head_dim)
    # Transpose back to [out, in] PyTorch convention
    return merged.T


def _loader_fused_delta(
    A_per: list[torch.Tensor],
    B_per: list[torch.Tensor],
    *,
    scale: float,
    n_dev: int,
    n_local_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reproduce adapter_loader._register_fused_qkv's fused A/B construction and
    return the resulting weight delta in PyTorch [out, in] convention.

    Mirrors lines 300-339 of adapter_loader.py. The device adds
    ``scale·(B_fused @ A_fused)`` (in [out, in]) into the fused base weight
    (see lora.py:_apply_delta / _upload_ab_for_w_sharding).
    """
    n = len(A_per)
    r = A_per[0].shape[0]
    out_per = n_dev * n_local_heads * head_dim

    # A stacked on rank dim → [n*r, in]
    A_fused = torch.cat(A_per, dim=0)

    # B: embed each per-source [out_per, r] into a [out_per, n*r] block-diagonal tile
    B_per_padded: list[torch.Tensor] = []
    for i, B in enumerate(B_per):
        pad = torch.zeros(out_per, n * r, dtype=B.dtype)
        pad[:, i * r : (i + 1) * r] = B
        B_per_padded.append(pad)

    B_fused = _head_interleave_lora_B(B_per_padded, n_dev=n_dev, n_local_heads=n_local_heads, head_dim=head_dim)

    # Device computes delta in [out, in] = scale * (B_fused @ A_fused)
    return scale * (B_fused @ A_fused)


# Wan2.2-T2V-A14B geometry: dim=5120, num_heads=40, head_dim=128. Use the real
# head_dim/num_heads so per-head reshapes match production; in_dim shrunk for speed.
HEAD_DIM = 128
NUM_HEADS = 40
IN_DIM = 256
RANK = 64
SCALE = 0.125  # alpha(8)/rank(64)


@pytest.mark.parametrize("n_dev", [1, 2, 4])
@pytest.mark.parametrize("case", ["self_attn_qkv", "cross_attn_kv"])
def test_fused_qkv_lora_delta_matches_interleaved_reference(n_dev: int, case: str) -> None:
    torch.manual_seed(0)
    dtype = torch.float32  # exact math; bf16 rounding is covered by the device test

    n = 3 if case == "self_attn_qkv" else 2
    dim = NUM_HEADS * HEAD_DIM
    n_local_heads = NUM_HEADS // n_dev
    assert NUM_HEADS % n_dev == 0

    # Per-source LoRA pairs: A_i [rank, in], B_i [out=dim, rank]
    A_per = [torch.randn(RANK, IN_DIM, dtype=dtype) for _ in range(n)]
    B_per = [torch.randn(dim, RANK, dtype=dtype) for _ in range(n)]

    # Loader path (the code under test)
    delta_fused = _loader_fused_delta(
        A_per, B_per, scale=SCALE, n_dev=n_dev, n_local_heads=n_local_heads, head_dim=HEAD_DIM
    )

    # Reference: per-source deltas interleaved exactly like the base weight
    per_source_delta = [SCALE * (B @ A) for A, B in zip(A_per, B_per)]  # each [dim, in]
    delta_ref = ref_interleave_heads(
        per_source_delta,
        n_dev=n_dev,
        n_local_heads=n_local_heads,
        head_dim=HEAD_DIM,
        num_heads=NUM_HEADS,
    )

    assert delta_fused.shape == delta_ref.shape == (n * dim, IN_DIM)

    # Localize any mismatch to specific output heads to make a failure actionable.
    if not torch.allclose(delta_fused, delta_ref, atol=1e-4, rtol=1e-4):
        out_rows = delta_fused.shape[0]
        bad = []
        # Compare in head_dim-sized row chunks; report first few mismatching chunks.
        for h in range(out_rows // HEAD_DIM):
            sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            if not torch.allclose(delta_fused[sl], delta_ref[sl], atol=1e-4, rtol=1e-4):
                bad.append(h)
        pytest.fail(
            f"[{case} n_dev={n_dev}] fused LoRA delta != interleaved reference. "
            f"{len(bad)}/{out_rows // HEAD_DIM} head-rows differ (first 10: {bad[:10]}). "
            f"max abs err={ (delta_fused - delta_ref).abs().max().item():.3e }"
        )
