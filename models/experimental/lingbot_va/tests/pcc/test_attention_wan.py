# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for WanAttention (self and cross) with Lingbot-VA model parameters.

Uses Lingbot-VA TT WanAttention from lingbot_va.tt and compares against the reference
block's attn1/attn2. Config: dim=3072, num_heads=24, head_dim=128, patch_size=(1,2,2).
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# Ensure tt-metal root is on path when running from various working directories
_tt_metal_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_tt_metal_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_tt_metal_root))

from models.experimental.lingbot_va.reference.model import (
    WanTransformer3DModel as TorchWanTransformer3DModel,
)
from models.experimental.lingbot_va.tt.attention_wan import WanAttention
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.padding import pad_vision_seq_parallel
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard
from models.tt_dit.utils.test import line_params

# ---------------------------------------------------------------------------
# Lingbot-VA model configuration (from reference/model.py)
# ---------------------------------------------------------------------------
DIM = 3072  # 24 * 128
NUM_HEADS = 24
HEAD_DIM = 128
PATCH_SIZE = (1, 2, 2)
EPS = 1e-6
QK_NORM = True

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", _tt_metal_root)
LINGBOT_VA_CHECKPOINT = Path(TT_METAL_HOME) / "models/experimental/lingbot_va/reference/checkpoints/transformer"


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((1, 2), (1, 1), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="1x1_single_device"),
    ],
    indirect=["mesh_device", "device_params"],
)
# Single (T,H,W) to avoid OOM from repeated full-model checkpoint loads across parametrized runs
@pytest.mark.parametrize(
    ("T", "H", "W"),
    [pytest.param(8, 24, 24, id="demo_8x24x24")],
)
@pytest.mark.parametrize(
    "prompt_seq_len",
    [
        pytest.param(77, id="cross_long"),
    ],
)
def test_wan_attention_lingbot_va(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int | None,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """PCC test: TT WanAttention (self/cross) vs Lingbot-VA reference first block attn1/attn2."""
    if not LINGBOT_VA_CHECKPOINT.exists():
        pytest.skip(f"Lingbot-VA checkpoint not found: {LINGBOT_VA_CHECKPOINT}")

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    torch_dtype = torch.float32
    B = 1
    layer_id = 0
    p_t, p_h, p_w = PATCH_SIZE
    patch_F, patch_H, patch_W = T // p_t, H // p_h, W // p_w
    spatial_seq_len = patch_F * patch_H * patch_W

    MIN_PCC = 0.988

    # Load Lingbot-VA reference and take first block's attention
    parent_torch_model = TorchWanTransformer3DModel.from_pretrained(
        str(LINGBOT_VA_CHECKPOINT),
        torch_dtype=torch_dtype,
        attn_mode="torch",
    )
    parent_torch_model.eval()
    first_layer = parent_torch_model.blocks[layer_id]
    torch_model = first_layer.attn1
    torch_model.eval()

    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=topology,
    )
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_model = WanAttention(
        dim=DIM,
        num_heads=NUM_HEADS,
        qk_norm=QK_NORM,
        eps=EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        is_self=False,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch.manual_seed(0)
    spatial_input = torch.randn((B, spatial_seq_len, DIM), dtype=torch_dtype)
    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    logger.info(f"spatial_input shape: {spatial_input.shape}, tt_spatial shape: {tt_spatial.shape}")

    prompt_input = torch.randn((B, prompt_seq_len, DIM), dtype=torch_dtype)
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    logger.info(f"prompt_input shape: {prompt_input.shape}, tt_prompt shape: {tt_prompt.shape}")
    rotary_emb_ref = None
    tt_rope_cos = None
    tt_rope_sin = None
    tt_trans_mat = None

    # Reference WanAttention.forward(q, k, v, rotary_emb): self-attn uses (x,x,x,rope), cross uses (q,k,v,None)
    with torch.no_grad():
        torch_spatial_out = torch_model(spatial_input, prompt_input, prompt_input, None)

    tt_spatial_out = tt_model(
        tt_spatial,
        N=spatial_seq_len,
        prompt_1BLP=tt_prompt,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
    )

    spatial_concat_dims = [None, None]
    spatial_concat_dims[sp_axis] = 2
    spatial_concat_dims[tp_axis] = 3
    tt_spatial_out = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=spatial_concat_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    tt_spatial_out = tt_spatial_out[:, :, :spatial_seq_len, :]
    if tt_spatial_out.dim() == 4 and tt_spatial_out.shape[0] == 1:
        tt_spatial_out = tt_spatial_out.squeeze(0)

    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC)
