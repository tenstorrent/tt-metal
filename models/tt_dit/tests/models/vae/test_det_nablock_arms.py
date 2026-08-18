# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate and block-level timing for the deterministic NABlock's optional fast paths.

Each ``DIFFVAE_DET_*`` flag changes how one block computes without changing what it computes, so
every arm is checked against the unflagged path on identical weights and input. The flags are read
in ``NeighborhoodAttention.__init__``/``SwiGLU.__init__``, so they are set before construction.

The timing test is the instrument these paths were tuned with: the changes move ops inside one
block, which is invisible against a whole decode. Geometry is the s34x60 decode's, stage by stage.
Stage 1 is absent on purpose -- its W=60 does not divide the size-8 mesh axis, so it runs replicated
on the gather backend and none of these flags reach it.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.layers.na3d import build_device_plan, plan_na3d
from models.tt_dit.models.vae.diffvae_ltx import NABlock, default_rope_dim_split, rope_tables
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality

HEAD_DIM = 64
SP_AXIS, TP_AXIS = 1, 0

FLAGS = (
    "DIFFVAE_DET_FUSED_QKV",
    "DIFFVAE_DET_COLPAR_QKV",
    "DIFFVAE_DET_FUSED_ROPE",
    "DIFFVAE_DET_FLAT_SEQ",
    "DIFFVAE_DET_FUSED_SWIGLU",
    "DIFFVAE_DET_TP_MLP",
)

ARMS = {
    "fused_qkv": ("DIFFVAE_DET_FUSED_QKV",),
    "colpar_qkv": ("DIFFVAE_DET_COLPAR_QKV",),
    "fused_swiglu": ("DIFFVAE_DET_FUSED_SWIGLU",),
    "tp_mlp": ("DIFFVAE_DET_TP_MLP",),
    "recommended": ("DIFFVAE_DET_COLPAR_QKV", "DIFFVAE_DET_FUSED_ROPE", "DIFFVAE_DET_FUSED_SWIGLU"),
    "flat_seq": (
        "DIFFVAE_DET_COLPAR_QKV",
        "DIFFVAE_DET_FUSED_ROPE",
        "DIFFVAE_DET_FUSED_SWIGLU",
        "DIFFVAE_DET_FLAT_SEQ",
    ),
}

#: (label, dim, kernel, full dims, blocks in that stage) for the W-sharded deterministic stages.
STAGES = [
    ("stage2", 1024, (3, 7, 7), (6, 68, 120), 6),
    ("stage3", 512, (3, 5, 5), (11, 68, 120), 4),
    ("stage4", 512, (3, 5, 5), (21, 136, 240), 2),
]


def _state(dim: int, seed: int) -> dict[str, torch.Tensor]:
    """Checkpoint-shaped weights: fused ``attn.qkv`` and split ``mlp.w_gate``/``w_up``.

    Every arm loads this same dict; only ``_prepare_torch_state`` differs between them, so a
    divergence is the arm's fault and not the input's.
    """
    g = torch.Generator().manual_seed(seed)
    hidden = (int(dim * 4.0) + 15) // 16 * 16

    def rn(*shape):
        return torch.randn(*shape, generator=g) * (shape[-1] ** -0.5)

    return {
        "norm1.weight": 1.0 + 0.05 * torch.randn(dim, generator=g),
        "norm2.weight": 1.0 + 0.05 * torch.randn(dim, generator=g),
        "attn.qkv.weight": rn(3 * dim, dim),
        "attn.qkv.bias": 0.02 * torch.randn(3 * dim, generator=g),
        "attn.proj.weight": rn(dim, dim),
        "attn.proj.bias": 0.02 * torch.randn(dim, generator=g),
        "attn.q_norm.weight": 1.0 + 0.05 * torch.randn(HEAD_DIM, generator=g),
        "attn.k_norm.weight": 1.0 + 0.05 * torch.randn(HEAD_DIM, generator=g),
        "mlp.w_gate.weight": rn(hidden, dim),
        "mlp.w_up.weight": rn(hidden, dim),
        "mlp.w_down.weight": rn(dim, hidden),
    }


def _build(mesh, dim, kernel, enabled: tuple[str, ...]):
    """An NABlock with exactly ``enabled`` set, asserting the flags actually took."""
    for flag in FLAGS:
        os.environ[flag] = "1" if flag in enabled else "0"
    block = NABlock(
        dim,
        kernel,
        head_dim=HEAD_DIM,
        mesh_device=mesh,
        na3d_backend="op_sp_w_sharded",
        ccl_manager=CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear),
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
    )
    colpar = "DIFFVAE_DET_COLPAR_QKV" in enabled
    tp_mlp = "DIFFVAE_DET_TP_MLP" in enabled
    assert block.attn.colpar_qkv is colpar
    assert block.attn.fused_qkv is (colpar or "DIFFVAE_DET_FUSED_QKV" in enabled)
    assert block.attn.fused_rope is ("DIFFVAE_DET_FUSED_ROPE" in enabled)
    assert block.attn.flat_seq is ("DIFFVAE_DET_FLAT_SEQ" in enabled)
    assert block.mlp.tp_mlp is tp_mlp
    assert block.mlp.fused is (tp_mlp or "DIFFVAE_DET_FUSED_SWIGLU" in enabled)
    return block


def _inputs(mesh, dim, dims):
    sp = int(list(mesh.shape)[SP_AXIS])
    t, h, w = dims
    tokens = t * h * (w // sp)
    cos, sin = rope_tables(dims, default_rope_dim_split(HEAD_DIM), mesh_device=mesh)
    cos = ttnn.mesh_partition(cos, dim=3, cluster_axis=SP_AXIS)
    sin = ttnn.mesh_partition(sin, dim=3, cluster_axis=SP_AXIS)
    return tokens, (t, h, w // sp), cos, sin


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["1d"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("arm", list(ARMS), ids=list(ARMS))
@pytest.mark.parametrize("stage", STAGES, ids=[s[0] for s in STAGES])
def test_det_nablock_arm_matches_baseline(*, mesh_device, device_params, arm, stage):
    """Every fast path reproduces the unflagged block bit-for-bit on the same weights."""
    _, dim, kernel, dims, _ = stage
    state = _state(dim, seed=11)
    tokens, local, cos, sin = _inputs(mesh_device, dim, dims)
    x_t = torch.randn(tokens, dim, generator=torch.Generator().manual_seed(5))

    def run(enabled):
        block = _build(mesh_device, dim, kernel, enabled)
        block.load_torch_state_dict(dict(state))
        x = ttnn.from_torch(x_t, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = block(x, dims=local, cos=cos, sin=sin, device_plan=None)
        # Chip 0's W-band suffices: the arms differ only in per-chip head bookkeeping.
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()

    reference = run(())
    assert_quality(reference, run(ARMS[arm]), pcc=0.999)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["1d"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("arm", ["baseline", *ARMS], ids=["baseline", *ARMS])
def test_det_nablock_arm_timing(*, mesh_device, device_params, arm):
    """Per-block device time for one arm, summed over the W-sharded stages. Run with ``-s``."""
    iters = int(os.environ.get("ITERS", 10))
    enabled = () if arm == "baseline" else ARMS[arm]
    total = 0.0
    for label, dim, kernel, dims, depth in STAGES:
        block = _build(mesh_device, dim, kernel, enabled)
        for name, param in _named_params(block):
            param.load_torch_tensor(_seeded(name, tuple(param.total_shape)))
        tokens, local, cos, sin = _inputs(mesh_device, dim, dims)
        x = ttnn.from_torch(
            torch.randn(tokens, dim, generator=torch.Generator().manual_seed(3)),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        for _ in range(2):  # warm the program cache
            x = block(x, dims=local, cos=cos, sin=sin, device_plan=None)
        ttnn.synchronize_device(mesh_device)

        t0 = time.perf_counter()
        for _ in range(iters):
            x = block(x, dims=local, cos=cos, sin=sin, device_plan=None)
        ttnn.synchronize_device(mesh_device)
        per_block = (time.perf_counter() - t0) / iters * 1000
        total += per_block * depth
        print(f"\n[{arm}/{label}] {per_block:8.2f} ms/block  x{depth} = {per_block * depth:8.1f} ms", flush=True)
        ttnn.deallocate(x)
    print(f"\n[{arm}] W-sharded det blocks total: {total:8.1f} ms\n", flush=True)


def _named_params(module, prefix: str = ""):
    """``named_parameters`` is not recursive; walk the children."""
    for name, param in module.named_parameters():
        yield f"{prefix}{name}", param
    for name, child in module.named_children():
        yield from _named_params(child, f"{prefix}{name}.")


def _seeded(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    if "norm" in name:
        return (1.0 + 0.05 * torch.randn(*shape, generator=g)).to(torch.float32)
    scale = shape[-2] ** -0.5 if len(shape) > 1 else 0.02
    return (torch.randn(*shape, generator=g) * scale).to(torch.float32)


#: Stage 1 of the same decode: (dim, kernel, dims, blocks). Its W=60 does not divide the size-8 axis
#: so it runs replicated on the gather backend, which is why it needs its own case -- and why only
#: the arms that do not depend on a TP axis or on the W-sharded attention can reach it.
STAGE1 = (2048, (3, 7, 7), (6, 34, 60), 4)

STAGE1_ARMS = {
    "swiglu": ("DIFFVAE_DET_FUSED_SWIGLU",),
    "qkv": ("DIFFVAE_DET_FUSED_QKV",),
    "qkv_rope": ("DIFFVAE_DET_FUSED_QKV", "DIFFVAE_DET_FUSED_ROPE"),
    "recommended": ("DIFFVAE_DET_FUSED_QKV", "DIFFVAE_DET_FUSED_ROPE", "DIFFVAE_DET_FUSED_SWIGLU"),
}


def _build_stage1(mesh, enabled: tuple[str, ...]):
    """A replicated stage-1 block, asserting the flags took and that the unreachable ones did not."""
    for flag in FLAGS:
        os.environ[flag] = "1" if flag in enabled else "0"
    dim, kernel, _, _ = STAGE1
    block = NABlock(
        dim,
        kernel,
        head_dim=HEAD_DIM,
        mesh_device=mesh,
        na3d_backend="gather",
        ccl_manager=None,
        sp_axis=None,
        tp_axis=None,
    )
    assert block.attn.tp == 1
    assert block.attn.fused_qkv is ("DIFFVAE_DET_FUSED_QKV" in enabled)
    assert block.attn.fused_rope is ("DIFFVAE_DET_FUSED_ROPE" in enabled)
    assert block.attn.colpar_qkv is False, "colpar needs a tp_axis to shard the weight over"
    assert block.attn.flat_seq is False, "flat_seq exists only in the W-sharded attention"
    assert block.mlp.fused is ("DIFFVAE_DET_FUSED_SWIGLU" in enabled)
    return block


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["1d"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("arm", list(STAGE1_ARMS), ids=list(STAGE1_ARMS))
def test_det_stage1_arm_matches_baseline(*, mesh_device, device_params, arm):
    """Every stage-1 fast path reproduces the unflagged replicated block on the same weights."""
    dim, kernel, dims, _ = STAGE1
    state = _state(dim, seed=11)
    t, h, w = dims
    tokens = t * h * w  # replicated: no W shard
    cos, sin = rope_tables(dims, default_rope_dim_split(HEAD_DIM), mesh_device=mesh_device)
    plan = build_device_plan(
        plan_na3d(dims, kernel),
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear),
    )
    x_t = torch.randn(tokens, dim, generator=torch.Generator().manual_seed(5))

    def run(enabled):
        block = _build_stage1(mesh_device, enabled)
        block.load_torch_state_dict(dict(state))
        x = ttnn.from_torch(x_t, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = block(x, dims=dims, cos=cos, sin=sin, device_plan=plan)
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()

    reference = run(())
    assert_quality(reference, run(STAGE1_ARMS[arm]), pcc=0.999)
