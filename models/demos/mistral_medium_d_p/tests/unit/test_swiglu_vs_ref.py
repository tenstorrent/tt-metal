# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE (1 chip): the fused SwiGLU activation op vs torch.

``tt/mlp.py`` computes ``silu(gate) * up`` as a SINGLE device op —
``ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`` — the idiom
``llama3_70b_galaxy/tt/llama_mlp.py:259`` uses. This test isolates that op so an activation
regression is not diagnosed as an MLP sharding bug.

Mistral is **plain** SwiGLU (``hidden_act: "silu"``), NOT the clamped ``swigluoai`` (alpha 1.702,
limit 7.0) that gpt-oss, MiniMax-M3 and Kimi ship — which is why none of their expert kernels are
reusable here. The last test pins that difference.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_swiglu_vs_ref.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ..test_factory import parametrize_mesh_with_fabric, replicate
from .shapes import FFN, per_chip


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [512, 2048], ids=["s512", "s2k"])
def test_fused_silu_mul_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """silu(gate) * up, fused, vs torch — at the per-chip FFN width."""
    torch.manual_seed(0)
    width = per_chip(4)["ffn"]  # 7168 — the width this op actually sees on the target
    gate = torch.randn(1, 1, seq_len, width)
    up = torch.randn(1, 1, seq_len, width)

    ref = torch.nn.functional.silu(gate) * up

    gate_tt, up_tt = replicate(gate, mesh_device), replicate(up, mesh_device)
    out_tt = ttnn.mul(gate_tt, up_tt, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0])

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"fused silu*mul vs ref (w={width}, s={seq_len}): {pcc}")
    assert passing, f"fused SwiGLU PCC fail: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_fused_activation_is_silu_not_clamped(mesh_device, device_params, reset_seeds):
    """Guard against silently running gpt-oss/M3's clamped swigluoai here.

    Clamped SwiGLU is ``x*sigmoid(alpha*x)`` with the input clamped to +/-limit, so the two diverge
    hard on large-magnitude inputs. Feed values past the clamp limit (7.0) and require plain silu.
    """
    torch.manual_seed(0)
    big = torch.linspace(-20, 20, 32 * 64).reshape(1, 1, 32, 64)
    ones = torch.ones_like(big)

    plain_silu = torch.nn.functional.silu(big) * ones
    clamped = big.clamp(-7.0, 7.0) * torch.sigmoid(1.702 * big.clamp(-7.0, 7.0)) * ones
    assert not torch.allclose(plain_silu, clamped, atol=1e-2), "test inputs do not distinguish the two activations"

    out_tt = ttnn.mul(
        replicate(big, mesh_device),
        replicate(ones, mesh_device),
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat16,
    )
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0])

    passing_silu, pcc_silu = comp_pcc(plain_silu, out, 0.99)
    logger.info(f"activation identity: vs plain silu {pcc_silu}")
    assert passing_silu, f"fused activation is not plain silu: {pcc_silu}"


def test_ffn_width_divides_the_tp_axis():
    """Host-side: the FFN sharding the MLP relies on. 28672/4 = 7168 = 224 tiles, no padding."""
    pc = per_chip(4)
    assert FFN % 4 == 0 and pc["ffn"] == 7168
    assert pc["ffn"] % 32 == 0, "per-chip FFN must be tile-aligned or the fused gate|up slice misaligns"
    assert pc["gate_up"] == 2 * pc["ffn"] == 14336
