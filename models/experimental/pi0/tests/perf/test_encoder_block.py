"""SigLIP encoder block: attention sub-block then MLP sub-block.

End-to-end orchestrator chaining test_attention_block + test_mlp_block:

  x  →  attn_sub(x; LN1, QKV, MHA-16-head, O-proj)  →  x1 = x + OProj(MHA(LN1(x)))
  x1 →  mlp_sub(x1; LN2, FC1, GELU, FC2)            →  x2 = x1 + FC2(GELU(FC1(LN2(x1))))

This is the SigLIP-So400m encoder layer 0 shape (synthetic weights for now).
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import pcc  # noqa: E402

# Composition delegates to existing orchestrators.
from test_attention_block import torch_attention_subblock, device_attention_subblock  # noqa: E402
from test_mlp_block import torch_mlp_subblock, device_mlp_subblock  # noqa: E402

M = 256
D = 1152
INTERMEDIATE_TRUE = 4304


def make_layer_weights(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.5

    # Attention sub-block weights
    ln1_w = torch.ones(D, dtype=torch.bfloat16) + torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.1
    ln1_b = torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.05
    qkv_w = torch.randn(D, 3 * D, generator=g, dtype=torch.bfloat16) * (1.0 / math.sqrt(D))
    o_w = torch.randn(D, D, generator=g, dtype=torch.bfloat16) * (1.0 / math.sqrt(D))

    # MLP sub-block weights
    ln2_w = torch.ones(D, dtype=torch.bfloat16) + torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.1
    ln2_b = torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.05
    fc1_w = torch.randn(D, INTERMEDIATE_TRUE, generator=g, dtype=torch.bfloat16) * (1.0 / math.sqrt(D))
    fc2_w = torch.randn(INTERMEDIATE_TRUE, D, generator=g, dtype=torch.bfloat16) * (1.0 / math.sqrt(INTERMEDIATE_TRUE))

    return x, ln1_w, ln1_b, qkv_w, o_w, ln2_w, ln2_b, fc1_w, fc2_w


def torch_encoder_block(x, ln1_w, ln1_b, qkv_w, o_w, ln2_w, ln2_b, fc1_w, fc2_w):
    x1 = torch_attention_subblock(x, ln1_w, ln1_b, qkv_w, o_w)
    x2 = torch_mlp_subblock(x1, ln2_w, ln2_b, fc1_w, fc2_w)
    return x2


def device_encoder_block(device, x, ln1_w, ln1_b, qkv_w, o_w, ln2_w, ln2_b, fc1_w, fc2_w):
    x1 = device_attention_subblock(device, x, ln1_w, ln1_b, qkv_w, o_w)
    x2 = device_mlp_subblock(device, x1, ln2_w, ln2_b, fc1_w, fc2_w)
    return x2


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_encoder_block_synthetic(device):
    """End-to-end SigLIP encoder block on synthetic weights.

    Chains attention sub-block + MLP sub-block. All primitives are K1.1
    PCC-validated; this validates the chained data flow.
    """
    inputs = make_layer_weights(seed=42)
    x = inputs[0]

    y_golden = torch_encoder_block(*inputs)
    y_device = device_encoder_block(device, *inputs)

    p = pcc(y_golden, y_device)
    print(f"\nPCC (full encoder block end-to-end) = {p:.6f}")
    print(f"  attention sub-block + MLP sub-block, M={M}, D={D}")
    assert p >= 0.99, f"Encoder block PCC {p} below 0.99 gate"
