# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decoder suite rows 2 and 10: the SwiGLU activation and the dense MLP.

Recipe rows ``test_swiglu_vs_ref.py`` and ``test_dense_mlp_vs_ref.py``. They are one file here
because for this model the activation *is* ``ttnn.silu`` — there is no clamped/limited variant to
build out of primitives the way minimax-m3's ``swigluoai`` needs, so the activation row is a
constants check rather than a module of its own. Keeping them adjacent makes the relationship
obvious: if ``test_swiglu_activation_vs_ref`` passes and ``test_dense_mlp_vs_ref`` fails, the
fault is in the projections or the all-reduce, not the non-linearity.

Real dims on the target mesh: 12288 -> 28672 -> 12288, intermediate sharded 7168 per chip.
"""

import torch
import torch.nn.functional as F

import ttnn
from models.demos.mistral_medium_3_5_128b.reference.modeling import REF_DTYPE, MistralMLP
from models.demos.mistral_medium_3_5_128b.tests.device_utils import assert_pcc, from_mesh_sp, to_mesh
from models.demos.mistral_medium_3_5_128b.tt.mlp import MLP

SEQ = 2048


def test_swiglu_activation_vs_ref(galaxy, cfg):
    """``ttnn.silu`` against ``F.silu`` at the model's exact variant.

    ``hidden_act`` is plain ``"silu"`` with no ``swiglu_limit`` / ``swiglu_alpha`` clamp, so the
    gate is ``x * sigmoid(x)`` unmodified. The input spans +-8 because silu's interesting region
    (and the range where a clamped variant would diverge) is near zero and in the negative tail.
    """
    assert cfg.hidden_act == "silu", f"config says {cfg.hidden_act}; this test pins silu"

    torch.manual_seed(0)
    x = (torch.rand(1, 1, 256, cfg.intermediate_size) * 16 - 8).to(REF_DTYPE)
    gated = F.silu(x.float()).to(REF_DTYPE) * x  # the SwiGLU product shape, with up == gate

    tt_x = to_mesh(galaxy, x, dims=[-2, None])
    tt_out = ttnn.mul(ttnn.silu(tt_x), tt_x)
    out = from_mesh_sp(galaxy, tt_out)

    assert_pcc("swiglu_activation", gated, out)


def test_dense_mlp_vs_ref(galaxy, mesh_config, ccl, cfg):
    """The whole dense MLP at real dims, same random weights on both sides."""
    torch.manual_seed(1)
    ref = MistralMLP(cfg, REF_DTYPE).eval()
    x = (torch.randn(1, SEQ, cfg.hidden_size) * 0.1).to(REF_DTYPE)
    with torch.no_grad():
        ref_out = ref(x).unsqueeze(0)  # [1, 1, SEQ, hidden]

    state_dict = {
        "gate_proj.weight": ref.gate_proj.weight.detach(),
        "up_proj.weight": ref.up_proj.weight.detach(),
        "down_proj.weight": ref.down_proj.weight.detach(),
    }
    mlp = MLP(galaxy, cfg, state_dict, mesh_config, ccl)

    tt_x = to_mesh(galaxy, x.unsqueeze(0), dims=[-2, None])
    tt_out = mlp(tt_x)
    out = from_mesh_sp(galaxy, tt_out)

    assert_pcc("dense_mlp", ref_out, out)
