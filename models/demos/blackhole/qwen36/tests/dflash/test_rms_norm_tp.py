# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1/M2: plain RMSNorm and per-head QK norm, vs ``Qwen3RMSNorm``, real weights.

Activations are real too: the hidden-norm cases run on the captured fixture's
``noise_embedding`` (the target's own embedding of ``[anchor, MASK x15]``), and the per-head
cases run on real ``q_proj``/``k_proj`` output. Nothing here is graded on random noise.

Run:
    MESH_DEVICE=T3K pytest models/demos/blackhole/qwen36/tests/dflash/test_rms_norm_tp.py -v -s
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.dflash.conftest import load_fixture
from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.rms_norm import rms_norm
from models.demos.blackhole.qwen36.tt.dflash.weights import read_state_dict

# The four hidden-width norms in the checkpoint, all [5120] with the same semantics.
HIDDEN_NORM_KEYS = [
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "hidden_norm.weight",
    "norm.weight",
]


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Stock ``Qwen3RMSNorm`` with these weights -- the oracle, not a reimplementation."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    ref = Qwen3RMSNorm(weight.shape[0], eps=eps)
    with torch.no_grad():
        ref.weight.copy_(weight)
    return ref(x)


def _replicate(x: torch.Tensor, mesh) -> ttnn.Tensor:
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _read_replicated(t: ttnn.Tensor, mesh) -> torch.Tensor:
    """Read a replicated tensor back, asserting every device agrees.

    The agreement check is not ceremony: a mis-mapped weight (sharded where it should be
    replicated) still produces plausible output on device 0, and this is what catches it.
    """
    stacked = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    n = mesh.get_num_devices()
    per = stacked.shape[0] // n
    first = stacked[:per]
    for d in range(1, n):
        assert torch.equal(stacked[d * per : (d + 1) * per], first), f"device {d} disagrees with device 0"
    return first


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize(
    "weight_key", HIDDEN_NORM_KEYS, ids=lambda k: k.replace(".weight", "").replace("layers.0.", "")
)
def test_rms_norm_tp(mesh_device, weight_key, reset_seeds, ensure_gc, request, drafter_cfg):
    """M1: hidden-width RMSNorm on the fixture's real noise embedding."""
    fx = load_fixture(512)
    x = fx["noise_embedding"].float()  # [1, 16, 5120], the target's real embeddings
    weight = read_state_dict(keys=[weight_key])[weight_key]

    expected = _reference(x, weight.float(), drafter_cfg.rms_norm_eps)

    tt_w = _replicate(weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), mesh_device)
    tt_x = _replicate(x.to(torch.bfloat16), mesh_device)
    tt_out = rms_norm(tt_x, tt_w, drafter_cfg.rms_norm_eps)
    actual = _read_replicated(tt_out, mesh_device).float()

    assert actual.shape == expected.shape, f"{actual.shape} != {expected.shape}"
    passing, pcc = comp_pcc(expected, actual, get_pcc_threshold(request))
    logger.info(f"{weight_key}: PCC {pcc}")
    assert passing, f"{weight_key} PCC {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("proj", ["q", "k"])
def test_qk_head_norm_tp(mesh_device, proj, reset_seeds, ensure_gc, request, drafter_cfg):
    """M2: per-head RMSNorm over head_dim, on real projection output.

    Mirrors the reference exactly: project, view as ``[b, seq, heads, head_dim]``, then
    normalise the last dim. Heads are sharded under TP but ``head_dim`` is not, so the
    statistics stay device-local -- the shard is on dim -2 here, and each device's slice is
    normalised independently.
    """
    fx = load_fixture(512)
    x = fx["noise_embedding"].float()  # [1, 16, 5120]

    keys = [f"layers.0.self_attn.{proj}_proj.weight", f"layers.0.self_attn.{proj}_norm.weight"]
    sd = read_state_dict(keys=keys)
    proj_w = sd[keys[0]].float()
    norm_w = sd[keys[1]]

    heads = drafter_cfg.num_attention_heads if proj == "q" else drafter_cfg.num_key_value_heads
    projected = (x @ proj_w.T).view(1, x.shape[1], heads, drafter_cfg.head_dim)
    expected = _reference(projected, norm_w.float(), drafter_cfg.rms_norm_eps)

    tt_w = _replicate(norm_w.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), mesh_device)
    # Shard the head axis (dim -2), matching how q/k arrive from a column-parallel projection.
    tt_x = ttnn.from_torch(
        projected.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-2),
    )
    tt_out = rms_norm(tt_x, tt_w, drafter_cfg.rms_norm_eps)
    actual = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-2)).float()

    assert actual.shape == expected.shape, f"{actual.shape} != {expected.shape}"
    passing, pcc = comp_pcc(expected, actual, get_pcc_threshold(request))
    logger.info(f"{proj}_norm per-head: PCC {pcc}")
    assert passing, f"{proj}_norm PCC {pcc}"
