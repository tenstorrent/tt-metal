# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the MiMo MoE decode perf / correctness scripts (one MoE layer on Galaxy row 0)."""

import os

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.moe import TtMiMoMoE

LAYER = 1
USERS = 4
MODEL_PATH = default_model_path()
CKPT_NAME = os.path.basename(os.path.normpath(MODEL_PATH))
# Converted weights, goldens and saved base outputs are kept per checkpoint (MOPD and RL share names, not values).
CACHE_ROOT = os.path.join(os.environ.get("MIMO_TT_CACHE", "/home/zni/models/mimo-tt-cache"), CKPT_NAME)
GOLDEN_DIR = os.path.join(os.environ.get("MIMO_GOLDEN_ROOT", "/home/zni/models/mimo-golden"), CKPT_NAME)
GOLDEN = os.environ.get("MIMO_GOLDEN", os.path.join(GOLDEN_DIR, "golden_48l.pt"))


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


def build_moe(row, cfg, cache_tag, expert_dtype, ckpt=None, users=USERS):
    """MoE layer LAYER from the tensor cache (``ckpt`` fills a cold cache), serving ``users`` decode slots."""
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
    moe.set_decode_users(users)
    return moe


def device_inputs(row, x_rows, active=None):
    """x_rows [users, hidden] -> (x [1, 1, 32, hidden] bf16, token mask [1, 1, 32, 1]) on the row; ``active`` (bool
    [users]): the slots that hold a token (default: every row), the others idle (mask 0)."""
    x = torch.zeros(32, x_rows.shape[-1])
    x[: x_rows.shape[0]] = x_rows
    on = torch.zeros(32, dtype=torch.bool)
    on[: x_rows.shape[0]] = True if active is None else torch.as_tensor(active)
    mask = on.to(torch.bfloat16).reshape(1, 1, 32, 1)
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


def prompt_starts(cfg):
    """Row offset of each golden prompt in golden_moe_inputs' rows, and the prompts' lengths."""
    golden = torch.load(GOLDEN)
    lengths = [h.shape[1] for h in golden["hidden"]]
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    return starts, lengths


def verify_steps(cfg, steps):
    """Speculative-verify shaped steps of 8 decode slots over the golden rows: (name, row indices [8] (-1: idle)).

    - one_request: 4 consecutive tokens of one prompt in slots 0-3 (a request with 3 drafts), slots 4-7 idle;
    - two_requests: 4 consecutive tokens of two prompts in slots 0-3 and 4-7;
    - plain4 / plain8: 4 / 8 tokens of different prompts (plain decode of distinct requests).
    Consecutive tokens share experts (4 select ~23 distinct, 8 of two requests ~37)."""
    starts, lengths = prompt_starts(cfg)
    windows = [(p, o) for p, n in enumerate(lengths) for o in range(0, n - 3, 4)]
    out = []
    for s in range(steps):
        (pa, oa), (pb, ob) = windows[s % len(windows)], windows[(s * 5 + 3) % len(windows)]
        a = [starts[pa] + oa + j for j in range(4)]
        b = [starts[pb] + ob + j for j in range(4)]
        distinct = [starts[p] + (3 * s + 5 * j + 1) % lengths[p] for j, p in enumerate((s + k) % 8 for k in range(8))]
        out.append(("one_request", a + [-1] * 4))
        out.append(("two_requests", a + b))
        out.append(("plain4", distinct[:4] + [-1] * 4))
        out.append(("plain8", distinct))
    return out


def step_inputs(row, inputs, idx):
    """(x, mask) of one verify_steps step: rows ``idx`` of ``inputs``, -1 slots idle (zero rows, mask 0)."""
    x_rows = torch.stack([inputs[i] if i >= 0 else torch.zeros(inputs.shape[-1]) for i in idx])
    return device_inputs(row, x_rows, [i >= 0 for i in idx])
