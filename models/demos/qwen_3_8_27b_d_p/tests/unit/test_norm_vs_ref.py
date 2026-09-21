# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm (Gemma ``(1 + w)`` fold) and the per-head QK-norm vs the torch reference.

Both are the same TT class at different widths: the decoder/final norms at ``hidden_size``, and
the QK-norm at ``head_dim`` on a head-split tensor. The fold is applied at load, so this also
pins that the folded gain and the reference's ``1 + weight`` agree.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.qwen_3_8_27b_d_p.reference.modeling import REF_DTYPE, Qwen35RMSNorm
from models.demos.qwen_3_8_27b_d_p.tt.rms_norm import RMSNorm

from ..test_factory import mesh_setup, parametrize_mesh, unit_test_config
from .helpers import check_pcc, from_sp_sharded, randn, to_sp_sharded


@parametrize_mesh()
@pytest.mark.parametrize("width_name", ["hidden", "head_dim"])
def test_rms_norm_vs_ref(mesh, submesh_shape, device_params, width_name):
    cfg = unit_test_config()
    mesh_config, _ccl = mesh_setup(mesh)
    width = cfg.hidden_size if width_name == "hidden" else cfg.head_dim
    tokens = 128 * mesh_config.sp

    ref = Qwen35RMSNorm(width, cfg.rms_norm_eps).to(REF_DTYPE)
    with torch.no_grad():
        ref.weight.copy_(randn(width, seed=1, scale=0.1))
    x = randn(1, 1, tokens, width, seed=2)
    with torch.no_grad():
        expected = ref(x)

    norm = RMSNorm(
        mesh,
        width,
        cfg.rms_norm_eps,
        {"weight": ref.weight.detach()},
        mesh_config=mesh_config,
    )
    tt_x = to_sp_sharded(x, mesh, mesh_config)
    got = from_sp_sharded(norm(tt_x), mesh_config)
    check_pcc(f"rms_norm[{width_name}]", expected, got, shape=(1, 1, tokens, width))


@parametrize_mesh()
def test_qk_norm_per_head_vs_ref(mesh, submesh_shape, device_params):
    """The same norm on a head-split ``[1, n_heads, S, head_dim]`` tensor.

    head_dim is not TP-sharded, so this must be exact per chip with no collective — a failure here
    means the gain is being broadcast over the wrong axis.
    """
    cfg = unit_test_config()
    mesh_config, _ccl = mesh_setup(mesh)
    n_heads_local = cfg.num_attention_heads // mesh_config.tp
    tokens = 128

    ref = Qwen35RMSNorm(cfg.head_dim, cfg.rms_norm_eps).to(REF_DTYPE)
    with torch.no_grad():
        ref.weight.copy_(randn(cfg.head_dim, seed=3, scale=0.1))
    x = randn(1, n_heads_local, tokens, cfg.head_dim, seed=4)
    with torch.no_grad():
        expected = ref(x)

    norm = RMSNorm(mesh, cfg.head_dim, cfg.rms_norm_eps, {"weight": ref.weight.detach()}, mesh_config=mesh_config)
    tt_x = ttnn.from_torch(
        x.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.replicate(mesh),
    )
    got = ttnn.to_torch(ttnn.get_device_tensors(norm(tt_x))[0])
    check_pcc("qk_norm_per_head", expected, got, shape=(1, n_heads_local, tokens, cfg.head_dim))


@parametrize_mesh()
def test_gemma_fold_is_not_a_plain_gain(mesh, submesh_shape, device_params):
    """Guard-rail: ``x_normed * (1 + w)`` and ``x_normed * w`` must be distinguishable here, so a
    regression that drops the fold cannot pass the test above by luck."""
    cfg = unit_test_config()
    mesh_config, _ccl = mesh_setup(mesh)
    width = cfg.hidden_size
    weight = randn(width, seed=5, scale=0.1)
    x = randn(1, 1, 32, width, seed=6)

    folded = Qwen35RMSNorm(width, cfg.rms_norm_eps).to(REF_DTYPE)
    with torch.no_grad():
        folded.weight.copy_(weight)
        gemma = folded(x).float()
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
        plain = (h * weight.float()).float()
    from models.common.utility_functions import comp_pcc

    _, pcc = comp_pcc(gemma, plain, 0.0)
    assert float(pcc) < 0.99, f"gemma and plain norms are indistinguishable at PCC {pcc} — bad test"
