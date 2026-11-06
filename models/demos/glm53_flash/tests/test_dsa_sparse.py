# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA sparse path (context > index_topk = 2048): pooled-key indexer, top-k, sparse_sdpa.

Uses the 4097-token host reference dump of layer 3 (prefill + 4 decode steps at positions
4097..4100), i.e. the full decoder layer including mHC and MoE, so a selection mismatch shows up
directly in the layer output PCC.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_dsa_sparse.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)

DUMP = Path("bringup/references/glm53_flash/T4097")
LAYER = 3


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_layer3_sparse_prefill_decode_vs_dump(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(DUMP / f"layer{LAYER}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    assert S > cfg.index_topk
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=1, max_seq_len=S + 64
    )

    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=S)
    y_h = device_streams_to_host(y, 4, S)
    p_all = pcc(x_out, y_h)
    p_dense = pcc(x_out[:, : cfg.index_topk], y_h[:, : cfg.index_topk])
    p_sparse = pcc(x_out[:, cfg.index_topk :], y_h[:, cfg.index_topk :])
    logger.info(
        f"layer 3 sparse prefill S={S}: pcc all {p_all:.6f}  rows<2048 {p_dense:.6f}  rows>=2048 {p_sparse:.6f}"
    )
    assert p_all > 0.995 and p_sparse > 0.995

    for step in range(4):
        d = torch.load(DUMP / f"layer{LAYER}_decode{step}.pt")
        pos = d["position"]
        dp = dec.make_decode_positions([pos])
        yd = dec.decode_forward(host_streams_to_device(d["x_in"], mesh_device, pad_to=1), positions=dp)
        p_d = pcc(d["x_out"], device_streams_to_host(yd, 4, 1))
        logger.info(f"  sparse decode step {step} (pos {pos}): pcc {p_d:.6f}")
        assert p_d > 0.995
