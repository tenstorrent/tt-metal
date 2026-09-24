# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small decoder blocks vs torch at real dims on the 8x4 mesh, random weights:

* RMSNorm (zero-centred, ``(1+w)`` fold)         pattern: minimax_m3/tests/unit/test_norm_vs_ref.py
* QK-norm over head_dim 256                        pattern: minimax_m3/tests/unit/test_qk_norm_vs_ref.py
* SwiGLU activation (plain SiLU gate)              pattern: minimax_m3/tests/unit/test_swiglu_vs_ref.py
* partial RoPE (64 of 256, HF half-split)          (no separate minimax row; attention depends on it)
* dense MLP, TP column/row parallel                pattern: minimax_m3/tests/unit/test_dense_mlp_vs_ref.py
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.qwen_3_8_27b.config import QWEN38, PrefillSpec, ttnn_dtype
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref
from models.demos.qwen_3_8_27b.tests.common import assert_pcc, from_sp, to_sp
from models.demos.qwen_3_8_27b.tt.mlp import TtMLP
from models.demos.qwen_3_8_27b.tt.rms_norm import TtRMSNorm
from models.demos.qwen_3_8_27b.tt.rope import TtRope

H = QWEN38.hidden_size


def _rand(*shape, seed=0, scale=1.0):
    return (torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale).to(torch.bfloat16)


@pytest.mark.parametrize("which", ["input_layernorm", "final_norm"])
def test_norm_vs_ref(mesh_config, which):
    norm = ref.RMSNorm(H, QWEN38.rms_norm_eps)
    with torch.no_grad():
        norm.weight.copy_(_rand(H, seed=1, scale=0.2).float())
    x = _rand(1, 1, 2048, H, seed=2, scale=3.0)
    want = norm.to(torch.bfloat16)(x)
    tt = TtRMSNorm(mesh_config, norm.weight.detach(), QWEN38.rms_norm_eps)
    assert_pcc(f"rmsnorm_{which}", from_sp(tt(to_sp(x, mesh_config)), mesh_config), want)


def test_qk_norm_vs_ref(mesh_config):
    hd = QWEN38.head_dim
    norm = ref.RMSNorm(hd, QWEN38.rms_norm_eps)
    with torch.no_grad():
        norm.weight.copy_(_rand(hd, seed=3, scale=0.2).float())
    x = _rand(1, 6, 2048, hd, seed=4, scale=2.0)
    want = norm.to(torch.bfloat16)(x)
    tt = TtRMSNorm(mesh_config, norm.weight.detach(), QWEN38.rms_norm_eps)
    assert_pcc("qk_norm", from_sp(tt(to_sp(x, mesh_config)), mesh_config), want)


def test_swiglu_vs_ref(mesh_config):
    g, u = _rand(1, 1, 2048, 4352, seed=5, scale=3.0), _rand(1, 1, 2048, 4352, seed=6)
    want = F.silu(g.float()) * u.float()
    out = ttnn.multiply(
        to_sp(g, mesh_config), to_sp(u, mesh_config), input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
    )
    assert_pcc("swiglu", from_sp(out, mesh_config), want)


@pytest.mark.parametrize("start", [0, 5120])
def test_partial_rope_vs_ref(mesh_config, start):
    S = 5120
    x = _rand(1, 6, S, QWEN38.head_dim, seed=7)
    cos, sin = ref.rope_cos_sin(QWEN38, torch.arange(start, start + S), dtype=torch.float32)
    want = ref.apply_partial_rope(x.float(), cos, sin)
    rope = TtRope(mesh_config, QWEN38)
    c, s = rope.tables(start, S // mesh_config.sp)
    got = from_sp(rope.apply(to_sp(x, mesh_config), c, s), mesh_config)
    assert_pcc(f"partial_rope_start{start}", got, want)


@pytest.mark.parametrize("seq", [2048, 10240], ids=["2k", "10k"])
def test_dense_mlp_vs_ref(mesh_config, seq):
    spec = PrefillSpec.load()
    mlp = ref.init_random_(ref.MLP(QWEN38), seed=8).to(torch.bfloat16)
    x = _rand(1, 1, seq, H, seed=9)
    with torch.no_grad():
        want = mlp(x)
    tt = TtMLP(
        mesh_config, mlp.state_dict(), dtypes={w: ttnn_dtype(spec.weight_dtype_mlp(w)) for w in ("gate", "up", "down")}
    )
    assert_pcc(f"dense_mlp_{seq}", from_sp(tt(to_sp(x, mesh_config)), mesh_config), want)
