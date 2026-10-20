# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel prefill v1 on a (rows, 4) block through the FunctionalDecoder API: the prompt's rows
split over the mesh rows (KDA sequence-parallel, MoE hybrid over the gathered slabs, DSA over the
gathered full chunk), output vs the HF prefill rows of the R4097 dump, then HF-live decode steps.

    TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 pytest models/demos/glm53_flash/tests/test_mesh_sp_prefill.py -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import build_layer, load_config
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)
from models.demos.glm53_flash.tt.mesh import mesh_device_params

DUMP = Path("bringup/references/glm53_flash/R4097")
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
S = int(os.environ.get("GLM53_SP_S", "1000"))
N_DECODE = 2


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3, 4])
def test_sp_prefill_then_decode_vs_hf(mesh_device, layer_idx):
    from transformers import DynamicCache

    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]  # causal: rows [:S] of x_out are the HF output of the S-token prefix
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=1216
    )
    mesh = dec.mesh
    x_dev = host_streams_to_device(x_in[:, :S], mesh_device, mesh=mesh)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    y = dec.prefill_forward(x_dev, user=0, seq_len=S)
    ttnn.synchronize_device(mesh_device)
    wall = time.perf_counter() - t0
    out = device_streams_to_host(y, 4, S, mesh=mesh)
    p = pcc(x_out[:, :S], out)
    rows = mesh.rows
    per_row = [
        pcc(x_out[:, r * S // rows : (r + 1) * S // rows], out[:, r * S // rows : (r + 1) * S // rows])
        for r in range(rows)
    ]
    logger.info(
        f"SP prefill layer {layer_idx} on {MESH} S={S}: pcc vs HF {p:.6f} per row-slab {['%.6f' % v for v in per_row]} "
        f"(first-call wall {wall * 1e3:.0f} ms)"
    )
    assert p > 0.995
    # decode steps vs HF live (the prefilled slot 0 is replicated on every row; compare row 0)
    hf, _ = build_layer(snapshot, text, layer_idx, headers)
    cache = DynamicCache(config=text)
    with torch.no_grad():
        hf(x_in[:, :S], attention_mask=torch.ones(1, S, dtype=torch.bool), past_key_values=cache, use_cache=True)
    dp = dec.make_decode_positions([S] * dec.total_users)
    for t in range(N_DECODE):
        xt = x_in[:, S + t : S + t + 1]
        with torch.no_grad():
            y_hf, _ = hf(xt, attention_mask=torch.ones(1, 1, dtype=torch.bool), past_key_values=cache, use_cache=True)
        dec.update_decode_positions(dp, [S + t] * dec.total_users)
        xin = xt.repeat(1, dec.total_users, 1, 1)  # the same token for every row's slot 0
        yd = dec.decode_forward(
            host_streams_to_device(xin, mesh_device, pad_to=dec.total_users, mesh=mesh), positions=dp
        )
        outd = device_streams_to_host(yd, 4, dec.total_users, mesh=mesh)
        ps = [pcc(y_hf, outd[:, r : r + 1]) for r in range(dec.total_users)]
        logger.info(f"  decode at position {S + t}: pcc vs HF per row {['%.6f' % v for v in ps]}")
        assert min(ps) > 0.995
