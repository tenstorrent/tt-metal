# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2: the decoder layers on a (rows, 4) chip block — TP4 over the columns, one DP user group
per row, hybrid EP(cols) x TP(rows) experts — against HF with real weights.

The mesh prefill is not wired yet, so every user's prompt is fed through the decode step token by
token (positions 0..S-1): HF's prefill output row t is the exact reference of the decode output at
step t (the recurrence / causal attention are the same computation). Then HF decode steps with a
live cache are compared with further decode steps, eager and trace-replayed. Each row decodes its own
users (different prompts per row), the outputs must be replicated across the four columns of a row.

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 pytest models/demos/glm53_flash/tests/test_mesh_decoder.py -s
"""

from __future__ import annotations

import os
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
ROWS, COLS = 2, 4
B = int(os.environ.get("GLM53_MESH_B", "2"))  # users per row (design: 4 and 32)
S = int(os.environ.get("GLM53_MESH_S", "48"))  # prompt tokens fed through the decode step
N_DECODE = 3  # HF-cache decode steps after the prompt, all through the captured trace


def _hf_prefill_and_decode(hf, text, x_prompt, x_next):
    from transformers import DynamicCache

    cache = DynamicCache(config=text)
    Sp = x_prompt.shape[1]
    with torch.no_grad():
        y_p, _ = hf(x_prompt, attention_mask=torch.ones(1, Sp, dtype=torch.bool), past_key_values=cache, use_cache=True)
        y_d = []
        for t in range(x_next.shape[1]):
            yt, _ = hf(
                x_next[:, t : t + 1],
                attention_mask=torch.ones(1, 1, dtype=torch.bool),
                past_key_values=cache,
                use_cache=True,
            )
            y_d.append(yt)
    return y_p, torch.cat(y_d, dim=1)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=60_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3, 4])
def test_mesh_decode_vs_hf(mesh_device, layer_idx):
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf, _ = build_layer(snapshot, text, layer_idx, headers)
    base = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")["x_in"]  # [1, 4097, 4, 4096] random-token streams
    U = ROWS * B
    offs = [100 + (3900 // U) * u for u in range(U)]  # distinct prompt windows inside the 4097-token dump
    prompts = [base[:, o : o + S] for o in offs]
    nexts = [base[:, o + S : o + S + N_DECODE] for o in offs]
    refs = [_hf_prefill_and_decode(hf, text, p, n) for p, n in zip(prompts, nexts)]

    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=B, max_seq_len=256
    )
    mesh = dec.mesh
    assert (mesh.rows, mesh.cols) == (ROWS, COLS) and dec.total_users == U

    def token(t_prompt: int | None, t_next: int | None = None) -> torch.Tensor:
        src = (
            [p[:, t_prompt : t_prompt + 1] for p in prompts]
            if t_next is None
            else [n[:, t_next : t_next + 1] for n in nexts]
        )
        return torch.cat(src, dim=1)  # [1, U, 4, 4096]

    def load(x_dev, xin):
        packed = xin.reshape(1, 1, U, -1).float()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh.shard_rows(2)), x_dev
        )

    def reset_kda():
        if dec.kda_batched is None:
            return
        for t in (dec.kda_batched.recurrent, dec.kda_batched.convolution):
            z = ttnn.zeros(
                tuple(t.shape),
                dtype=t.dtype,
                layout=t.layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.copy(z, t)
            ttnn.deallocate(z)

    # persistent inputs; warm-up + capture at position 0 with the real first token (the warm-up output
    # is the eager step 0; the capture pass advances the state a second time -> reset the KDA state,
    # the DSA cache/index rows at position 0 were simply rewritten with the same values)
    x_dev = host_streams_to_device(token(0), mesh_device, pad_to=U, mesh=mesh)
    dp = dec.make_decode_positions([0] * U)
    trace_id, y_tr, warm = dec.capture_decode_trace(x_dev, dp)
    out0 = device_streams_to_host(warm, 4, U, mesh=mesh)
    p0 = [pcc(refs[u][0][:, 0:1], out0[:, u : u + 1]) for u in range(U)]
    logger.info(f"layer {layer_idx} eager step 0 (warm-up) per-user pcc vs HF row 0: {['%.6f' % p for p in p0]}")
    reset_kda()

    def step(pos: int, xin: torch.Tensor, eager: bool = False) -> torch.Tensor:
        load(x_dev, xin)
        dec.update_decode_positions(dp, [pos] * U)
        if eager:
            y = dec.decode_forward(x_dev, positions=dp)
        else:
            dec.replay_decode(trace_id)
            y = y_tr
        if pos == 0:  # the TP all-reduce must leave every column of a row with the same tensor
            shards = [ttnn.to_torch(d) for d in mesh.device_tensors(y)]
            for r in range(ROWS):
                for c in range(1, COLS):
                    assert torch.equal(shards[r * COLS + c], shards[r * COLS]), (r, c)
        return device_streams_to_host(y, 4, U, mesh=mesh)  # [1, U, 4, 4096]

    # the prompt, token by token through the captured graph (one eager step in the middle)
    worst_step = 1.0
    outs = [[] for _ in range(U)]
    for t in range(S):
        out = step(t, token(t), eager=(t == S // 2))
        for u in range(U):
            outs[u].append(out[:, u : u + 1])
            worst_step = min(worst_step, pcc(refs[u][0][:, t : t + 1], out[:, u : u + 1]))
    per_user = sorted(pcc(refs[u][0], torch.cat(outs[u], dim=1)) for u in range(U))
    logger.info(
        f"layer {layer_idx} mesh {ROWS}x{COLS}: {U} users x {S} prompt tokens through the traced decode step "
        f"(step {S // 2} eager): per-user pcc vs HF prefill min {per_user[0]:.6f} p10 {per_user[U // 10]:.6f} "
        f"median {per_user[U // 2]:.6f} max {per_user[-1]:.6f}; worst single-token row {worst_step:.6f}"
    )
    # single-row PCCs are noisy samples (the position-0 row of a layer-3 user is ~0.98-0.997 on one chip too)
    assert per_user[0] > 0.995 and worst_step > 0.97
    # HF-cache decode steps through the captured graph
    for t in range(N_DECODE):
        out = step(S + t, token(None, t))
        p_users = sorted(pcc(refs[u][1][:, t : t + 1], out[:, u : u + 1]) for u in range(U))
        logger.info(
            f"layer {layer_idx} traced decode at position {S + t}: per-user pcc vs HF min {p_users[0]:.6f} "
            f"median {p_users[U // 2]:.6f} max {p_users[-1]:.6f}"
        )
        assert p_users[0] > 0.995
    ttnn.release_trace(mesh_device, trace_id)
