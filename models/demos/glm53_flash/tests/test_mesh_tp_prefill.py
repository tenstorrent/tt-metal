# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2: the decoder layers tensor-parallel over a single mesh row (1, 4): prefill (ttKDA TP /
DSA with head-sharded projections / hybrid MoE with 4 expert groups) and decode vs HF with real
weights, through the stage-1 FunctionalDecoder API unchanged. The outputs must be replicated across
the four chips.

    TT_VISIBLE_DEVICES=25,29,21,17 TT_MESH_GRAPH_DESC_PATH=~/g4cov/p150_x4_galaxy_2ch.textproto MESH_DEVICE=P150x4 \
    pytest models/demos/glm53_flash/tests/test_mesh_tp_prefill.py -s
"""

from __future__ import annotations

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
ROWS, COLS = 1, 4
S = 200  # non-aligned prompt
N_DECODE = 3


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3, 4])
def test_tp4_prefill_decode_vs_hf(mesh_device, layer_idx):
    from transformers import DynamicCache

    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf, _ = build_layer(snapshot, text, layer_idx, headers)
    base = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")["x_in"]
    x_prompt, x_next = base[:, 100 : 100 + S], base[:, 100 + S : 100 + S + N_DECODE]
    cache = DynamicCache(config=text)
    with torch.no_grad():
        y_hf, _ = hf(x_prompt, attention_mask=torch.ones(1, S, dtype=torch.bool), past_key_values=cache, use_cache=True)
        y_hf_dec = []
        for t in range(N_DECODE):
            yt, _ = hf(
                x_next[:, t : t + 1],
                attention_mask=torch.ones(1, 1, dtype=torch.bool),
                past_key_values=cache,
                use_cache=True,
            )
            y_hf_dec.append(yt)

    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=256
    )
    mesh = dec.mesh
    assert (mesh.rows, mesh.cols) == (ROWS, COLS)
    y = dec.prefill_forward(host_streams_to_device(x_prompt, mesh_device), user=0, seq_len=S)
    shards = [ttnn.to_torch(d) for d in mesh.device_tensors(y)]
    for c in range(1, COLS):
        assert torch.equal(shards[c][:, :, :S], shards[0][:, :, :S]), c
    out = device_streams_to_host(y, 4, S, mesh=mesh)
    p = pcc(y_hf, out)
    logger.info(f"layer {layer_idx} TP4 prefill S={S}: pcc vs HF {p:.6f} (replicated across the 4 chips)")
    assert p > 0.995
    dp = dec.make_decode_positions([S])
    for t in range(N_DECODE):
        dec.update_decode_positions(dp, [S + t])
        yd = dec.decode_forward(host_streams_to_device(x_next[:, t : t + 1], mesh_device, pad_to=1), positions=dp)
        p_d = pcc(y_hf_dec[t], device_streams_to_host(yd, 4, 1, mesh=mesh))
        logger.info(f"layer {layer_idx} TP4 decode at position {S + t}: pcc vs HF {p_d:.6f}")
        assert p_d > 0.995
