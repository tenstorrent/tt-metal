# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-device confirmation of the fused-QKV LoRA path across tensor-parallel shards.

The host test (``test_fused_qkv_lora_math.py``) proves the *unsharded*
construction is correct. This test confirms the remaining device-only links:
the head-interleaved fused weight + fused LoRA ``B`` are **column-sharded**
across devices, the fuse-merge runs on-device (``_apply_delta``), and the
matmul reconstructs the right output — at the production layout (TP=4).

It exercises the exact production-critical module (``LoRAColParallelLinear`` —
the class ``WanAttention.to_qkv`` is) loaded with a head-interleaved fused
weight, rather than the full attention forward (which would need 8+ devices
for the real 14B model). Parametrized over a 1x1 mesh (trivial) and a 1x4 mesh
(TP=4, matching p300x2's n_dev).

Run (needs hardware; the tt-metal server must NOT be holding the devices):
    ./python_env/bin/python3 -m pytest -xvs \
        models/tt_dit/experimental/tests/test_fused_qkv_lora_device.py
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.experimental.lora.adapter_loader import _head_interleave_lora_B
from models.tt_dit.layers.linear import LoRAColParallelLinear
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor

HEAD_DIM = 32
NUM_HEADS = 8
IN_DIM = 256
RANK = 8
SCALE = 0.125


def _interleave_heads(tensors, *, n_dev, n_local_heads, head_dim, num_heads):
    """Verbatim transcription of WanAttention._interleave_heads
    (attention_wan.py:186-206) — authoritative fused out-dim layout."""
    tensors = [t.T for t in tensors]
    tensors = [t.reshape(t.shape[0], n_dev, n_local_heads, head_dim) for t in tensors]
    merged = torch.cat(tensors, dim=2)
    merged = merged.reshape(merged.shape[0], len(tensors) * num_heads * head_dim)
    return merged.T


def _build_fused_ab(A_per, B_per, *, n_dev, n_local_heads, head_dim):
    """Mirror adapter_loader._register_fused_qkv's A_fused / B_fused build."""
    n = len(A_per)
    r = A_per[0].shape[0]
    out_per = n_dev * n_local_heads * head_dim
    A_fused = torch.cat(A_per, dim=0)
    B_per_padded = []
    for i, B in enumerate(B_per):
        pad = torch.zeros(out_per, n * r, dtype=B.dtype)
        pad[:, i * r : (i + 1) * r] = B
        B_per_padded.append(pad)
    B_fused = _head_interleave_lora_B(B_per_padded, n_dev=n_dev, n_local_heads=n_local_heads, head_dim=head_dim)
    return A_fused, B_fused


def _readback(out, mesh_device, tp_axis):
    """Concat the column-sharded output back to a full torch tensor."""
    concat_dims = [None, None]
    concat_dims[tp_axis] = -1
    concat_dims[1 - tp_axis] = 0  # other axis has size 1 in these configs
    return ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((1, 1), id="1x1_tp1"), pytest.param((1, 4), id="1x4_tp4")],
    indirect=True,
)
@pytest.mark.parametrize("case", ["self_attn_qkv", "cross_attn_kv"])
def test_fused_qkv_lora_device(mesh_device: ttnn.MeshDevice, case: str) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tp_axis = 1
    n_dev = tuple(mesh_device.shape)[tp_axis]
    n_local_heads = NUM_HEADS // n_dev
    assert NUM_HEADS % n_dev == 0

    n = 3 if case == "self_attn_qkv" else 2
    dim = NUM_HEADS * HEAD_DIM
    out_dim = n * dim
    seq_len = 128

    # Separate per-source base weights → interleaved fused weight (the real layout).
    W_per = [torch.randn(dim, IN_DIM, dtype=dtype) * 0.02 for _ in range(n)]
    W_fused = _interleave_heads(W_per, n_dev=n_dev, n_local_heads=n_local_heads, head_dim=HEAD_DIM, num_heads=NUM_HEADS)
    torch_lin = torch.nn.Linear(IN_DIM, out_dim, bias=False).to(dtype=dtype)
    with torch.no_grad():
        torch_lin.weight.copy_(W_fused)

    # Per-source LoRA pairs and the fused A/B the loader would build.
    A_per = [torch.randn(RANK, IN_DIM, dtype=dtype) * 0.05 for _ in range(n)]
    B_per = [torch.randn(dim, RANK, dtype=dtype) * 0.05 for _ in range(n)]
    A_fused, B_fused = _build_fused_ab(A_per, B_per, n_dev=n_dev, n_local_heads=n_local_heads, head_dim=HEAD_DIM)

    # Reference (float32): base, and base + interleaved per-source deltas.
    x_t = torch.randn(1, 1, seq_len, IN_DIM, dtype=dtype)
    per_src_delta = [SCALE * (B.float() @ A.float()) for A, B in zip(A_per, B_per)]
    delta_ref = _interleave_heads(
        per_src_delta, n_dev=n_dev, n_local_heads=n_local_heads, head_dim=HEAD_DIM, num_heads=NUM_HEADS
    )
    with torch.no_grad():
        y_base_ref = torch.nn.functional.linear(x_t.float(), W_fused.float())
        y_lora_ref = torch.nn.functional.linear(x_t.float(), W_fused.float() + delta_ref)

    # Device module: the production fused-projection class.
    tt = LoRAColParallelLinear(
        IN_DIM, out_dim, bias=False, mesh_device=mesh_device, mesh_axis=tp_axis, lora_mode="fuse"
    )
    tt.load_torch_state_dict(torch_lin.state_dict())
    idx = tt.register_lora(A_fused, B_fused, scale=SCALE)
    x_tt = bf16_tensor(x_t, device=mesh_device)

    # 1) base (no LoRA bound)
    assert_quality(y_base_ref, _readback(tt(x_tt), mesh_device, tp_axis), pcc=0.999)

    # 2) bound → fused LoRA delta merged on-device and column-sharded
    tt.bind_active(idx, scale=SCALE)
    assert_quality(y_lora_ref, _readback(tt(x_tt), mesh_device, tp_axis), pcc=0.99)

    # 3) unbind restores base
    tt.unbind_active()
    assert_quality(y_base_ref, _readback(tt(x_tt), mesh_device, tp_axis), pcc=0.999)

    tt.deallocate_lora()
