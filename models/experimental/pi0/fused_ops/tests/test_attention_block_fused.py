"""Fused attention_block kernel — first increment test (LN1 + residual).

Validates the multi-Op chaining mechanic: a single TRISC dispatch that
calls LN1 then residual_add in sequence, with L1 CB chaining between
them (no host round-trip). Math: out = LN1(x; gamma, beta) + x.

PCC target ≥ 0.999. Future increments add QKV → 16× SDPA → O-proj.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
from golden_fc1 import pcc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attention_block"))
from op import (  # noqa: E402
    SigLIPAttentionBlockFused,
    build_tensors_for_fused_attention_block,
)

M, D = 256, 1152
EPS = 1e-6


def make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.5
    gamma = torch.ones(D, dtype=torch.bfloat16) + torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.1
    beta = torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.05
    return x, gamma, beta


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_ln_plus_residual(device):
    x, gamma, beta = make_inputs(seed=42)

    # Golden: LN1(x) + x in fp32, output bf16.
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    y_golden = (ln_out_golden + x.float()).to(torch.bfloat16)

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta, num_cores=8)
    SigLIPAttentionBlockFused.op(*tensors, num_cores=8, eps=EPS)

    import ttnn as _ttnn

    final_out_tt = tensors[-1]
    y_device = _ttnn.to_torch(final_out_tt)

    p = pcc(y_golden, y_device)
    print(f"\nPCC (fused LN1+residual vs torch fp32) = {p:.6f}")
    print(f"  shape={tuple(y_device.shape)}, dtype={y_device.dtype}")
    assert p >= 0.999, f"PCC {p} below 0.999 gate"
