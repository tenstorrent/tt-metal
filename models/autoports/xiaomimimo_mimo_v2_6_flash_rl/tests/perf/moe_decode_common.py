# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the MiMo MoE decode perf / correctness scripts (one MoE layer on Galaxy row 0)."""

import os

import torch

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference import model as ref
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference.checkpoint import MiMoCheckpoint
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference.config import MiMoConfig
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.common import MeshPlan
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.moe import TtMiMoMoE

LAYER = 1
USERS = 4
MODEL_PATH = os.environ.get("MIMO_MODEL_PATH", "/home/zni/models/MiMo-V2.6-Flash-RL")
CACHE_ROOT = os.environ.get("MIMO_TT_CACHE", "/home/zni/models/mimo-tt-cache")
GOLDEN = os.environ.get("MIMO_GOLDEN", "/home/zni/models/mimo-golden/golden_48l.pt")


def open_row(trace_region_size=0):
    """Open the 4x8 Galaxy with the ring fabric and return (parent mesh, row-0 1x8 submesh)."""
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536, trace_region_size=trace_region_size)
    row = mesh.create_submeshes(ttnn.MeshShape(1, 8))[0]
    return mesh, row


def close(mesh, row):
    ttnn.close_mesh_device(row)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def config():
    return MiMoConfig.from_model_path(MODEL_PATH)


def build_moe(row, cfg, cache_tag, expert_dtype, ckpt=None):
    """MoE layer LAYER from the tensor cache (``ckpt`` fills a cold cache)."""
    router = experts = None
    if ckpt is not None:
        router = ckpt.router(LAYER)
        experts = ref.LazyExperts(ckpt, LAYER, torch.bfloat16)
    moe = TtMiMoMoE(
        MeshPlan(row),
        cfg,
        LAYER,
        router,
        experts,
        cache_dir=os.path.join(CACHE_ROOT, cache_tag),
        expert_dtype=expert_dtype,
    )
    moe.set_decode_users(USERS)
    return moe


def device_inputs(row, x_rows):
    """x_rows [USERS, hidden] -> (x [1, 1, 32, hidden] bf16, token mask [1, 1, 32, 1]) on the row."""
    x = torch.zeros(32, x_rows.shape[-1])
    x[: x_rows.shape[0]] = x_rows
    mask = (torch.arange(32) < x_rows.shape[0]).to(torch.bfloat16).reshape(1, 1, 32, 1)
    rep = ttnn.ReplicateTensorToMesh(row)
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, 32, -1).bfloat16(), device=row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep
    )
    m_tt = ttnn.from_torch(mask, device=row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
    return x_tt, m_tt


def golden_moe_inputs(cfg, ckpt):
    """Real post-attention-norm activations of the golden prompts at LAYER (fp32 rows, bf16-exact)."""
    golden = torch.load(GOLDEN)
    w = dict(ckpt.norms(LAYER))
    w["attn"] = ckpt.attention(LAYER)
    dims = cfg.attention_dims(LAYER)
    rows = []
    with torch.no_grad():
        for hidden in golden["hidden"]:
            x = hidden[LAYER]
            pos = torch.arange(x.shape[0])
            h = ref.rms_norm(x, w["input_layernorm"], cfg.layernorm_epsilon)
            x = x + ref.attention(h, w["attn"], dims, cfg.attention_value_scale, pos)
            rows.append(ref.rms_norm(x, w["post_attention_layernorm"], cfg.layernorm_epsilon))
    return torch.cat(rows).bfloat16().float()


def checkpoint(cfg):
    return MiMoCheckpoint(MODEL_PATH, cfg)
