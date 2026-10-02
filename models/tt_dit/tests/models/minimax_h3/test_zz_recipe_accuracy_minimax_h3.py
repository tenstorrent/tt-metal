# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bench only (not for merge): MiniMax-H3 attention accuracy per SDPA recipe at production-like lengths.

One `MiniMaxH3Attention` (random weights, real config) on the Galaxy against diffusers' torch attention in FP32
on a 16:9 768P latent grid (24 x 42 per frame). The torch reference is computed once per length and cached in
H3_ACC_CACHE; the recipe comes from H3_SDPA_RECIPE / H3_SDPA_KV (bench_recipe_env). Metrics go to H3_ACC_OUT.
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor_2dshard
from .bench_recipe_env import recipe_from_env, recipe_tag
from .common import GALAXY_RING, ROPE_FREQ_DIM, ROPE_THETA, randomize_norm_weights, upload_rope
from .test_transformer_minimax_h3 import (
    ATTENTION_HEAD_DIM,
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    QK_NORM_EPS,
    MiniMaxH3RotaryPosEmbed,
    TorchMiniMaxH3Attention,
    logical_length_tensor,
)

GRID_H, GRID_W = 24, 42  # 768P 16:9 latent patches per frame


def _reference(frames: int):
    cache = Path(os.environ.get("H3_ACC_CACHE", str(Path.home() / "h3_acc_cache"))) / f"attn_f{frames}.pt"
    if cache.exists():
        return torch.load(cache)
    torch.manual_seed(1234)
    coords = torch.meshgrid(torch.arange(frames), torch.arange(GRID_H), torch.arange(GRID_W), indexing="ij")
    position_ids = torch.stack([c.reshape(-1) for c in coords], dim=-1)
    model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE, heads=NUM_ATTENTION_HEADS, dim_head=ATTENTION_HEAD_DIM, qk_norm_eps=QK_NORM_EPS
    ).to(torch.float32)
    randomize_norm_weights(model)
    model.eval()
    rope_cos, rope_sin = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)(position_ids)
    spatial = torch.randn((1, position_ids.shape[0], HIDDEN_SIZE), dtype=torch.float32)
    # The device sees BF16 weights and inputs; the reference runs FP32 on those same rounded values.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
        spatial = spatial.to(torch.bfloat16).to(torch.float32)
        logger.info(f"torch reference: {position_ids.shape[0]} tokens (computed once, cached at {cache})")
        out = model(hidden_states=spatial, rotary_emb=(rope_cos, rope_sin), attention_mask=None)
    ref = {"state_dict": model.state_dict(), "spatial": spatial, "rope_cos": rope_cos, "rope_sin": rope_sin, "out": out}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, cache)
    return ref


@GALAXY_RING
@pytest.mark.parametrize("frames", [pytest.param(48, id="f48_s48384"), pytest.param(96, id="f96_s96768")])
def test_minimax_h3_attention_recipe_accuracy(
    mesh_device, sp_axis, tp_axis, num_links, frames, is_fsdp, topology, reset_seeds
):
    torch.set_num_threads(os.cpu_count())
    ref = _reference(frames)
    seq_len = ref["spatial"].shape[1]
    sp_factor, tp_factor = tuple(mesh_device.shape)[sp_axis], tuple(mesh_device.shape)[tp_axis]
    assert seq_len % (sp_factor * ttnn.TILE_SIZE) == 0

    precision, kv = recipe_from_env()
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    tt_model = MiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_ATTENTION_HEADS,
        head_dim=ATTENTION_HEAD_DIM,
        rotary_dim=ref["rope_cos"].shape[-1],
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        sdpa_precision=precision,
        sdpa_kv_dtype=kv,
    )
    tt_model.load_torch_state_dict(ref["state_dict"])
    cos_t, sin_t = prepare_rope_tables(ref["rope_cos"], ref["rope_sin"], ATTENTION_HEAD_DIM)
    tt_cos, tt_sin = upload_rope(cos_t, sin_t, mesh_device=mesh_device, sp_axis=sp_axis)
    tt_spatial = bf16_tensor_2dshard(ref["spatial"].unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_out = tt_model(tt_spatial, logical_n=logical_length_tensor(mesh_device, seq_len), rope_cos=tt_cos, rope_sin=tt_sin)
    dims = [None, None]
    dims[sp_axis], dims[tp_axis] = 2, 3
    out = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )[:, :, :seq_len, :].double().reshape(-1)
    golden = ref["out"].double().reshape(-1)
    diff = out - golden
    pcc = torch.corrcoef(torch.stack([out, golden]))[0, 1].item()
    rmse = diff.pow(2).mean().sqrt().item()
    rel_l2 = (diff.norm() / golden.norm()).item() * 100
    line = f"ACC recipe={recipe_tag()} tokens={seq_len} pcc={pcc:.6f} rmse={rmse:.6f} rel_l2={rel_l2:.4f}% golden_rms={golden.pow(2).mean().sqrt().item():.4f}"
    logger.info(line)
    if os.environ.get("H3_ACC_OUT"):
        with open(os.environ["H3_ACC_OUT"], "a") as handle:
            handle.write(line + "\n")
    assert torch.isfinite(out).all()
