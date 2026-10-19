# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Repeated identical inputs give identical device outputs (prefill and traced decode), and no
host/torch call happens inside a decode or prefill pass (the pass runs with the ttnn fallback
guard raised).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_determinism.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)

DUMP = Path("bringup/references/glm53_flash/T256")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3])
def test_repeat_determinism_and_no_fallback(mesh_device, layer_idx):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, layer_idx, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    x_in = pre["x_in"]
    S = x_in.shape[1]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=S + 64
    )
    x_dev = host_streams_to_device(x_in, mesh_device)

    outs = []
    with ttnn.manage_config("throw_exception_on_fallback", True):
        for _ in range(2):
            y = dec.prefill_forward(x_dev, user=0, seq_len=S)
            outs.append(device_streams_to_host(y, 4, S))
    assert torch.equal(outs[0], outs[1]), "prefill is not bit-deterministic across repeats"
    logger.info(f"layer {layer_idx}: two identical prefills (S={S}) are bit-identical; no host fallback raised")

    d0 = torch.load(DUMP / f"layer{layer_idx}_decode0.pt")
    streams_dev = host_streams_to_device(d0["x_in"], mesh_device, pad_to=1)
    pos_dev = ttnn.from_torch(
        torch.tensor([d0["position"]], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        trace_id, out_dev, _ = dec.capture_decode_trace(streams_dev, pos_dev)
    # replaying the same token twice from the same state: reset the state by re-prefilling, then replay
    results = []
    for _ in range(2):
        dec.prefill_forward(x_dev, user=0, seq_len=S)
        dec.replay_decode(trace_id)
        results.append(device_streams_to_host(out_dev, 4, 1).clone())
    assert torch.equal(results[0], results[1]), "traced decode is not bit-deterministic across repeats"
    logger.info(f"layer {layer_idx}: two traced decode replays from the same state are bit-identical")
    ttnn.release_trace(mesh_device, trace_id)
