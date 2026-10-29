# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ttnn DFlash2 drafter (models/demos/blackhole/qwen36/tt/dflash2_tp.py, DFlash2DrafterTP) on the resident
model's mesh.

The drafter runs as ordinary ttnn ops (and traces) on CQ0 between resident launches. It cannot read the resident
lm_head (BankLayout tiles with the final norm folded in), so it gets its own vocab-sharded bf8 copy of the
checkpoint's lm_head ([H, V / n] per chip, the drafter applies its own final norm), a TT_CCL for its all-reduces
and the target's token embedding on the host. Its context taps are the target's residual after layers
cfg["taps"], dim-fractured: chip c holds columns [c H / n, (c + 1) H / n) of every tap row.
"""
import hashlib
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.dflash2 import DEFAULT_WEIGHTS, H, resolve_weights_dir
from models.demos.blackhole.qwen36.tt.dflash2_tp import DFlash2DrafterTP
from models.tt_transformers.tt.ccl import TT_CCL
from qwen36_resident import weights as Q


def ccl_topology():
    """Ring on the boxes whose chips form one (ModelArgs.ccl_topology), else Linear"""
    ring = (ttnn.cluster.ClusterType.P300_X2, ttnn.cluster.ClusterType.P150_X4, ttnn.cluster.ClusterType.P150_X8)
    return ttnn.Topology.Ring if ttnn.cluster.get_cluster_type() in ring else ttnn.Topology.Linear


def cache_root():
    root = os.environ.get("TT_CACHE_PATH")
    return f"{root}/resident_dflash" if root else None


def lm_head_vshard(mesh, ck, cache_dir=None):
    """the checkpoint's lm_head as [H, V / n] bf8 per chip (vocab-sharded, no norm folded)"""
    name = f"{cache_dir}/lm_head.vshard" if cache_dir else None
    w = ck.get("lm_head.weight")  # [V, H]
    return ttnn.as_tensor(
        w,
        preprocess=lambda t: t.T.contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=name,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )


def build_drafter(mesh, ck, weights_dir=None, embed=None, block=None):
    """DFlash2DrafterTP on `mesh` for the target checkpoint ck; weights cached under $TT_CACHE_PATH/resident_dflash"""
    weights_dir = weights_dir or os.environ.get("DFLASH_WEIGHTS", DEFAULT_WEIGHTS)
    real = os.path.realpath(resolve_weights_dir(weights_dir))
    root = cache_root()
    if root:
        os.makedirs(root, exist_ok=True)
    cache = f"{root}/{os.path.basename(real)}-{hashlib.sha1(real.encode()).hexdigest()[:10]}" if root else None
    emb = (Q.embedding(ck) if embed is None else embed).to(torch.bfloat16)
    lm = lm_head_vshard(mesh, ck, root)
    drafter = DFlash2DrafterTP(
        mesh,
        weights_dir,
        emb,
        TT_CCL(mesh),
        ccl_topology(),
        lm,
        lm_vocab_sharded=True,
        block=block,
        cache_dir=cache,
    )
    logger.info(
        f"[resident-dflash] drafter {real}: block {drafter.block}, taps {drafter.taps}, windows {drafter.windows}, "
        f"selector {drafter.has_selector}, topology {ccl_topology()}"
    )
    return drafter


def tap_buffers(mesh, ntaps, rows, values=None):
    """`ntaps` persistent dim-fractured [1, 1, rows, H / n] bf16 tap tensors (zeros, or values [ntaps, rows, H])"""
    out = []
    for i in range(ntaps):
        x = torch.zeros(1, 1, rows, H) if values is None else values[i].reshape(1, 1, rows, H)
        out.append(
            ttnn.from_torch(
                x.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )
        )
    return out
