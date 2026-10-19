# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced decode (capture once, replay per token) and multi-user decode vs the HF reference.

Capture semantics: the first decode step runs eagerly (program compile) and is also captured;
the capture step itself consumes the SAME inputs, so the state advances twice for that token.
To keep the comparison exact the test captures on a throw-away user (user 1) ... no: simpler,
it captures on the real inputs of decode step 0, so after capture the state has seen token 0
twice. Instead this test prefills, captures the trace on a dummy token at position S (advancing
the state by two dummy tokens), then re-prefills to reset the state and replays the real steps.
That exercises exactly what serving does: capture once, then only replays.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_decode_trace.py -s
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

DUMP = Path("bringup/references/glm53_flash/T256")


def _packed_host(streams: torch.Tensor, n_rows: int) -> torch.Tensor:
    _, S, n, C = streams.shape
    out = torch.zeros(1, 1, n_rows, n * C, dtype=torch.float32)
    out[0, 0, :S] = streams.reshape(S, n * C).float()
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3, 4])
def test_traced_decode_vs_dump(mesh_device, layer_idx):
    snapshot = default_snapshot_dir()
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    x_in = pre["x_in"]
    S = x_in.shape[1]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=S + 64
    )

    # persistent decode inputs
    streams_dev = ttnn.from_torch(
        _packed_host(x_in[:, :1], 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    dp = dec.make_decode_positions([S])

    # 1) prefill, capture the decode trace on a dummy token (state advances by 2 dummy tokens)
    dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=S)
    trace_id, out_dev, _ = dec.capture_decode_trace(streams_dev, dp)

    # 2) reset the state by re-prefilling (cache pages are rewritten, KDA state replaced in place)
    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=S)
    p = pcc(pre["x_out"], device_streams_to_host(y, 4, S))
    logger.info(f"layer {layer_idx} prefill S={S}: pcc {p:.6f}")
    assert p > 0.995

    # 3) replay the captured decode for the 4 reference steps, updating only the inputs
    for step in range(4):
        d = torch.load(DUMP / f"layer{layer_idx}_decode{step}.pt")
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(_packed_host(d["x_in"], 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), streams_dev
        )
        dec.update_decode_positions(dp, [d["position"]])
        dec.replay_decode(trace_id)
        p_d = pcc(d["x_out"], device_streams_to_host(out_dev, 4, 1))
        logger.info(f"  traced decode step {step} (pos {d['position']}): pcc {p_d:.6f}")
        assert p_d > 0.995
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3])
def test_two_users_decode_vs_hf(mesh_device, layer_idx):
    """Two users with different prompt lengths prefilled one at a time, then decoded as a batch."""
    from transformers import DynamicCache

    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf, _ = build_layer(snapshot, text, layer_idx, headers)
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    lens = [100, 37]
    prompts = [pre["x_in"][:, :100], pre["x_in"][:, 120 : 120 + 37]]
    n_decode = 2
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=2, max_seq_len=256
    )

    caches, ys = [], []
    for u, (L, x) in enumerate(zip(lens, prompts)):
        cache = DynamicCache(config=text)
        with torch.no_grad():
            y_hf, _ = hf(x, attention_mask=torch.ones(1, L, dtype=torch.bool), past_key_values=cache, use_cache=True)
        y = dec.prefill_forward(host_streams_to_device(x, mesh_device), user=u, seq_len=L)
        p = pcc(y_hf, device_streams_to_host(y, 4, L))
        logger.info(f"layer {layer_idx} user {u} prefill S={L}: pcc {p:.6f}")
        assert p > 0.995
        caches.append(cache)

    for step in range(n_decode):
        toks = [pre["x_in"][:, 200 + 2 * u + step : 201 + 2 * u + step] for u in range(2)]
        y_hf = []
        for u in range(2):
            with torch.no_grad():
                yu, _ = hf(
                    toks[u],
                    attention_mask=torch.ones(1, 1, dtype=torch.bool),
                    past_key_values=caches[u],
                    use_cache=True,
                )
            y_hf.append(yu)
        xin = torch.cat(toks, dim=1)  # [1, B=2, 4, 4096]: row b = user b
        dp = dec.make_decode_positions([L + step for L in lens])
        yd = dec.decode_forward(host_streams_to_device(xin, mesh_device, pad_to=2), positions=dp)
        out = device_streams_to_host(yd, 4, 2)  # [1, 2, 4, 4096]
        for u in range(2):
            p_d = pcc(y_hf[u], out[:, u : u + 1])
            logger.info(f"  decode step {step} user {u} (pos {lens[u] + step}): pcc {p_d:.6f}")
            assert p_d > 0.995
