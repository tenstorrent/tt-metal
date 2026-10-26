# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-blaze G53F-0175: does the sequence-parallel ttKDA prefill of a RAGGED chunk depend on what DRAM held before?

The 1M prompt's last chunk is 7804 tokens padded to 7936 = 8 rows x 992 (31 tile rows per chip); its history-dependent
rows were mesh row 0's local rows 864..990 -- the positions that are padding on mesh row 7. This runs one KDA layer
(the model's ``TtKDA``) on such a chunk twice, with DRAM filled with different junk before each run (allocate + free a
large tensor of a constant), and reports which output rows differ between the runs, by mesh row and local row, plus the
PCC vs HF on the real rows. ``GLM53_RAGGED_S`` = the real rows (default 7804), ``GLM53_RAGGED_T`` = the padded rows
(default 7936); S = T is the control (no padding).

    GLM53_MESH_SHAPE=8,4 pytest models/demos/glm53_flash/tests/test_kda_ragged_history.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.kda import TtKDA
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
T = int(os.environ.get("GLM53_RAGGED_T", "7936"))
S = int(os.environ.get("GLM53_RAGGED_S", "7804"))
JUNK = [float(v) for v in os.environ.get("GLM53_RAGGED_JUNK", "1e4,-3e3,0").split(",")]
JUNK_ROWS = int(os.environ.get("GLM53_RAGGED_JUNK_ROWS", "65536"))  # x 4096 bf16 per chip = 512 MiB
LAYER = int(os.environ.get("GLM53_KDA_LAYER", "0"))
# the HF reference needs a transformers that knows glm5_next (not every pod host has it): GLM53_RAGGED_HF=1
HF = os.environ.get("GLM53_RAGGED_HF", "0") == "1"
# G53F-0175 carry-in mode: GLM53_RAGGED_PRE=N runs an N-row full chunk at position 0 first (N % (rows*32) == 0) and the
# ragged chunk at start N with that chunk's state -- the model's last chunk at depth (start > 0, a carried-in state)
PRE = int(os.environ.get("GLM53_RAGGED_PRE", "0"))


def _poison(mesh_device, value: float):
    junk = ttnn.full([1, 1, JUNK_ROWS, 4096], value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(junk)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_kda_ragged_history(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=lambda n: n.startswith("self_attn."))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    mesh = GLMMesh.from_device(mesh_device)
    rows = mesh.rows
    assert T % (rows * 32) == 0 and S <= T, (T, S, rows)
    T_loc = T // rows
    tt = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, mesh=mesh)
    torch.manual_seed(0)
    x = (torch.randn(1, T, cfg.hidden_size) * 0.8).to(torch.bfloat16)
    x[:, S:] = 0  # the model's padding rows are the embedding of the pad id; zeros here (they are past actual_end)
    x_pre = (torch.randn(1, PRE, cfg.hidden_size) * 0.8).to(torch.bfloat16) if PRE else None
    outs = []
    for junk in JUNK:
        state = tt.allocate_state(1)
        if PRE:  # the carried-in state of a prompt's earlier chunks (re-made per run: same input, same state)
            assert PRE % (rows * 32) == 0, PRE
            p_dev = mesh.upload(x_pre, ttnn.bfloat16, rows=1)
            y_pre, state = tt.prefill_forward(p_dev, state, tt.scalar(0), None)
            ttnn.deallocate(y_pre)
            ttnn.deallocate(p_dev)
        _poison(mesh_device, junk)
        x_dev = mesh.upload(x, ttnn.bfloat16, rows=1)
        end = tt.scalar(PRE + S) if (S != T or PRE) else None
        y, new_state = tt.prefill_forward(x_dev, state, tt.scalar(PRE), end)
        out = mesh.to_host(y, rows=1).float()[0]  # [T, 4096] in slab order
        rec = ttnn.to_torch(mesh.device_tensors(new_state.recurrent)[0]).float()
        outs.append((junk, out, rec))
        ttnn.deallocate(y)
        ttnn.deallocate(x_dev)
        logger.info(f"[ragged] junk {junk:g}: output rows {out.shape[0]}, state sum {rec.abs().sum():.6g}")
    y_hf = None
    if HF:
        from transformers import DynamicCache

        from models.demos.glm53_flash.reference.checkpoint import load_config
        from models.demos.glm53_flash.tests.test_kda import _hf_linear_attention

        _, text = load_config(snapshot)
        hf = _hf_linear_attention(snapshot, text, raw)
        with torch.no_grad():
            y_hf = hf(
                x[:, :S], cache_params=DynamicCache(config=text), attention_mask=torch.ones(1, S, dtype=torch.bool)
            )
    ref_junk, ref, ref_rec = outs[0]
    for junk, out, rec in outs:
        diff = (out[:S] != ref[:S]).any(-1).nonzero().flatten()
        by_row = {}
        for g in diff.tolist():
            r, i = divmod(g, T_loc)
            lo, hi, n = by_row.get(r, (i, i, 0))
            by_row[r] = (min(lo, i), max(hi, i), n + 1)
        p = pcc(y_hf[0].float(), out[:S]) if y_hf is not None else float("nan")
        logger.info(
            f"[ragged] T={T} S={S} junk {junk:g} vs junk {ref_junk:g}: {len(diff)} of {S} real rows differ "
            f"{ {r: f'local {lo}..{hi} ({n} rows)' for r, (lo, hi, n) in sorted(by_row.items())} }; final state "
            f"{'EQUAL' if torch.equal(rec, ref_rec) else 'DIFF'}; pcc vs HF {p:.6f}"
        )
    if y_hf is None:
        return
    worst = []
    for r in range(rows):
        lo, hi = r * T_loc, min(S, (r + 1) * T_loc)
        if lo < hi:
            worst.append(round(pcc(y_hf[0, lo:hi].float(), outs[0][1][lo:hi]), 6))
    logger.info(f"[ragged] per mesh row pcc vs HF (junk {ref_junk:g}): {worst}")
