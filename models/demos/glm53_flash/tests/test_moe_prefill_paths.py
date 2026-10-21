# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefill routed-expert paths of ``tt/moe.py`` (``GLM53_PREFILL_EXPERTS``: blocks | batched | dispatch | regions) on one chip
or a ``GLM53_MESH_SHAPE`` block (the hybrid layout: the rows' tokens gathered, expert slices per chip), real layer
weights, realistic MoE inputs (``test_moe.realistic_mlp_input``): every path vs the HF ``Glm5NextTextMoE`` and vs the
stage-1 block path, at ``GLM53_MOE_PREFILL_T`` tokens per row (default 2048), plus the warmed wall time of each.
The dispatch path needs a multi-row mesh (its dispatch / combine ops are fabric ops along the row axis); the regions
path (the same d_p expert kernel on a region buffer built with plain ttnn ops) runs anywhere.

    TT_VISIBLE_DEVICES=25 GLM53_MESH_SHAPE=1,1 GLM53_MOE_PREFILL_PATHS=blocks,batched pytest \\
        models/demos/glm53_flash/tests/test_moe_prefill_paths.py -s --timeout 1500
    <block env> GLM53_MESH_SHAPE=2,4 GLM53_MOE_PREFILL_PATHS=blocks,dispatch pytest ... (the per-row tokens gathered: 2 x T)

The paths are selected per ``TtMoE`` instance through the module knob (read at construction), so each path builds its
own layer (the dispatch path slices the stacked experts into per-expert tensors). Reduced experts
(``GLM53_MOE_PREFILL_EXPERTS``, default all 288) are NOT used: the routing must see the real expert set.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import load_config
from models.demos.glm53_flash.tests.test_moe import LAYER, _hf_moe, realistic_mlp_input
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt import moe as moe_mod
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params
from models.demos.glm53_flash.tt.moe import TtMoE

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
T = int(os.environ.get("GLM53_MOE_PREFILL_T", "2048"))  # tokens per mesh row
PATHS = os.environ.get("GLM53_MOE_PREFILL_PATHS", "blocks,regions").split(",")
PCC_BAR = 0.995


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_moe_prefill_paths(mesh_device):
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf, sd = _hf_moe(text, {k: v for k, v in raw.items() if k.startswith("mlp.")})
    mesh = GLMMesh.from_device(mesh_device)
    rows = mesh.rows
    x = realistic_mlp_input(snapshot, text, raw, T * rows)  # [1, rows*T, 4096] bf16: row r holds rows [rT, (r+1)T)
    with torch.no_grad():
        y_hf = hf(x.float().to(torch.bfloat16)).float()
    x_tt = mesh.upload(x.reshape(1, 1, T * rows, -1), ttnn.bfloat16, rows=2 if rows > 1 else None)
    TT = T * rows
    # the experts already stacked by _hf_moe (test_moe's method: no second 14 GB stack per path)
    mlp_sd = {
        k[len("mlp.") :]: v
        for k, v in raw.items()
        if k.startswith("mlp.") and ("experts." not in k or "shared_experts" in k)
    }
    mlp_sd["experts.gate_up_proj"] = sd["experts.gate_up_proj"]
    mlp_sd["experts.down_proj"] = sd["experts.down_proj"]
    outs, walls = {}, {}
    for path in PATHS:
        if path == "dispatch" and rows < 2:
            logger.info("path dispatch needs a multi-row mesh: skipped")
            continue
        moe_mod.PREFILL_EXPERTS = path
        t0 = time.perf_counter()
        moe = TtMoE(mesh_device, cfg, mlp_sd, mesh=mesh)
        logger.info(f"path {path}: layer {LAYER} MoE built in {time.perf_counter() - t0:.0f} s")
        y = moe(x_tt)  # compile
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(y)
        ws = []
        for _ in range(2):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            y = moe(x_tt)
            ttnn.synchronize_device(mesh_device)
            ws.append(time.perf_counter() - t0)
            if _ == 0:
                ttnn.deallocate(y)
        walls[path] = min(ws)
        outs[path] = mesh.to_host(y, rows=2 if rows > 1 else None).float().reshape(TT, -1)
        ttnn.deallocate(y)
        p = pcc(y_hf.reshape(TT, -1), outs[path])
        logger.info(
            f"path {path}: {TT} tokens ({T} per row) on {MESH}: warmed wall {walls[path] * 1e3:.1f} ms "
            f"({TT / walls[path]:.0f} tok/s), pcc vs HF {p:.6f}"
        )
        # release the layer's weights before the next path's build
        from models.demos.glm53_flash.tt.functional_decoder import device_tensors_of

        for t in device_tensors_of(moe):
            if t.is_allocated():
                ttnn.deallocate(t)
        del moe
    ref = outs.get("blocks")
    lines = ["| path | wall ms | tok/s | pcc vs HF | pcc vs blocks |", "|---|---|---|---|---|"]
    for path in outs:
        p_hf = pcc(y_hf.reshape(TT, -1), outs[path])
        p_b = pcc(ref, outs[path]) if ref is not None else float("nan")
        lines.append(f"| {path} | {walls[path] * 1e3:.1f} | {TT / walls[path]:.0f} | {p_hf:.6f} | {p_b:.6f} |")
        assert p_hf > PCC_BAR, (path, p_hf)
    logger.info("MoE prefill paths, layer %d, %d tokens on %s:\n%s" % (LAYER, TT, MESH, "\n".join(lines)))


def test_region_bookkeeping_host():
    """Host golden of ``_forward_regions``' routing math (no device): random top-8 routing of T tokens over E experts
    with E_loc local experts; the slots computed as the device path does (one-hot x mask, cumsum ranks, tile-aligned
    exclusive cumsum offsets, dummy slot for the non-local picks) must be unique inside every region, lie in
    ``[offset_e, offset_e + count_e)``, keep the regions tile-aligned and disjoint, and never reach the dummy slot."""
    T, E, E_loc, k, TILE = 8192, 288, 72, 8, 32
    g = torch.Generator().manual_seed(0)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])  # [T, k] distinct experts per token
    dense = torch.zeros(T, E)
    dense.scatter_(1, idx, torch.rand(T, k, generator=g) + 0.01)
    mask = torch.zeros(E)
    mask[:E_loc] = 1.0
    onehot = (dense != 0).float() * mask
    counts = onehot.sum(0)
    rank = onehot.cumsum(0)
    aligned = torch.floor((counts + TILE - 1) / TILE) * TILE
    offsets = aligned.cumsum(0) - aligned
    max_buf = T * k + E_loc * TILE
    pos = (rank - 1 + offsets) * onehot + (1 - onehot) * (max_buf - 1)
    pos_tk = pos.gather(1, idx)
    loc_tk = onehot.gather(1, idx)
    local = loc_tk.bool()
    slots = pos_tk[local].long()
    assert slots.numel() == int(counts.sum()) and slots.unique().numel() == slots.numel(), "slots not unique"
    assert (pos_tk[~local] == max_buf - 1).all(), "non-local picks must land on the dummy slot"
    for e in range(E_loc):
        sl = pos[:, e][onehot[:, e] > 0].long()
        assert (
            sl.numel() == int(counts[e]) and sl.min() == int(offsets[e]) and sl.max() == int(offsets[e] + counts[e] - 1)
        )
        assert int(offsets[e]) % TILE == 0
    assert int(offsets[E_loc - 1] + aligned[E_loc - 1]) <= max_buf - TILE, "regions must stay below the dummy slot"
    logger.info(
        f"region bookkeeping: {int(counts.sum())} routed rows over {E_loc} local experts (max {int(counts.max())} per expert), "
        f"regions end at {int(offsets[E_loc - 1] + aligned[E_loc - 1])} of max_buf {max_buf}"
    )
