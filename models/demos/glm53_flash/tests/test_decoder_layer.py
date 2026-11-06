# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full decoder layers (mHC + attention + MLP) vs the HF reference, real weights, one chip.

Layer kinds: 0 = KDA + dense, 3 = DSA + MoE, 4 = KDA + MoE. The 256-token case replays the host
reference dump (``bringup/references/glm53_flash/T256``: prefill streams in/out and 4 decode
steps); the non-aligned case builds the HF layer on the fly.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_decoder_layer.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import build_layer, load_config
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)

DUMP = Path("bringup/references/glm53_flash/T256")


def _decoder(mesh_device, snapshot, layer_idx, raw, max_seq_len):
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    return FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=max_seq_len
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3, 4])
def test_layer_prefill_decode_vs_dump(mesh_device, layer_idx):
    snapshot = default_snapshot_dir()
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]  # [1,S,4,4096] bf16
    S = x_in.shape[1]
    dec = _decoder(mesh_device, snapshot, layer_idx, raw, max_seq_len=S + 64)

    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=S)
    y_h = device_streams_to_host(y, 4, S)
    p = pcc(x_out, y_h)
    logger.info(
        f"layer {layer_idx} ({'KDA' if dec.layer.is_kda else 'DSA'}+{'MoE' if dec.layer.is_moe else 'dense'}) prefill S={S}: pcc {p:.6f}"
    )
    assert p > 0.995

    for step in range(4):
        d = torch.load(DUMP / f"layer{layer_idx}_decode{step}.pt")
        dp = dec.make_decode_positions([d["position"]])
        yd = dec.decode_forward(host_streams_to_device(d["x_in"], mesh_device, pad_to=1), positions=dp)
        p_d = pcc(d["x_out"], device_streams_to_host(yd, 4, 1))
        logger.info(f"  decode step {step} (pos {d['position']}): pcc {p_d:.6f}")
        assert p_d > 0.995


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3])
@pytest.mark.parametrize("seq_len", [200])
def test_layer_nonaligned_vs_hf(mesh_device, layer_idx, seq_len):
    """Non-tile-aligned prefill length followed by decode, HF layer built on the fly."""
    from transformers import DynamicCache

    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    hf, _ = build_layer(snapshot, text, layer_idx, headers)
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    x = pre["x_in"][:, : seq_len + 2]  # realistic streams
    cache = DynamicCache(config=text)
    with torch.no_grad():
        y_hf, _ = hf(
            x[:, :seq_len],
            attention_mask=torch.ones(1, seq_len, dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
        )
    dec = _decoder(mesh_device, snapshot, layer_idx, raw, max_seq_len=seq_len + 64)
    y = dec.prefill_forward(host_streams_to_device(x[:, :seq_len], mesh_device), user=0, seq_len=seq_len)
    p = pcc(y_hf, device_streams_to_host(y, 4, seq_len))
    logger.info(f"layer {layer_idx} prefill S={seq_len} (non-aligned): pcc {p:.6f}")
    assert p > 0.995
    for step in range(2):
        xt = x[:, seq_len + step : seq_len + step + 1]
        with torch.no_grad():
            y_hf_d, _ = hf(xt, attention_mask=torch.ones(1, 1, dtype=torch.bool), past_key_values=cache, use_cache=True)
        dp = dec.make_decode_positions([seq_len + step])
        yd = dec.decode_forward(host_streams_to_device(xt, mesh_device, pad_to=1), positions=dp)
        p_d = pcc(y_hf_d, device_streams_to_host(yd, 4, 1))
        logger.info(f"  decode step {step} (pos {seq_len + step}): pcc {p_d:.6f}")
        assert p_d > 0.995
