# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The traced multi-user sparse decode of the DSA layer (OPT_SPARSE_1M deliverable 2): every user slot of the
chip, dense or sparse, attended inside ONE trace-replayable decode graph through the per-user metadata
tensors (``SparseDecodeMeta``), the per-user indexer scoring + top-k, the pool gather into a compact cache
and the batched flash-MLA decode over it (no host chunk geometry, no whole-prefix copy).

* ``test_two_users_dense_sparse_traced``: layer 3, two user slots prefilled from two different reference
  streams (user 0: the 2040-token random prompt R2040, user 1: the 4097-token R4097), then 12 lockstep decode
  steps: user 0 walks 2040..2051 across the dense / sparse forms (2047) and the HF-exact dense bound (2050),
  user 1 is sparse throughout. Steps 0-1 eager (step 0 also through the eager reference path, new vs eager
  compared), the trace is captured on step 2 and replayed for steps 3..11 with only the inputs and the
  metadata refreshed; every (user, step) layer output is held to PCC > 0.995 vs its HF dump.
* ``test_selected_pools_vs_hf``: the traced path's selected pools (expanded to token ids, plus the tail) vs
  the HF indexer's ``index_ids`` of the R4097 decode steps (common ids >= 97 %, the test_dsa_long bar).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_dsa_sparse_traced.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import attention_input, pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)

LAYER = 3
R4097 = Path("bringup/references/glm53_flash/R4097")
R2040 = Path("bringup/references/glm53_flash/R2040")


def _packed(streams: torch.Tensor, n_rows: int) -> torch.Tensor:
    _, S, n, C = streams.shape
    out = torch.zeros(1, 1, n_rows, n * C, dtype=torch.float32)
    out[0, 0, :S] = streams.reshape(S, n * C).float()
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 64 * 1024 * 1024, "l1_small_size": 32768}], indirect=True
)
def test_two_users_dense_sparse_traced(mesh_device, monkeypatch):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre0, pre1 = torch.load(R2040 / f"layer{LAYER}_prefill.pt"), torch.load(R4097 / f"layer{LAYER}_prefill.pt")
    S0, S1 = pre0["x_in"].shape[1], pre1["x_in"].shape[1]
    n_steps = 12
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=2, max_seq_len=S1 + 64
    )
    # permuted page table: the gather translates logical pools through it
    bps = dec.cache.blocks_per_seq
    perm = torch.randperm(2 * bps, generator=torch.Generator().manual_seed(7)).to(torch.int32).reshape(2, bps)
    dec.cache.set_page_table(perm)
    for u, (pre, S) in enumerate(((pre0, S0), (pre1, S1))):
        y = dec.prefill_forward(host_streams_to_device(pre["x_in"], mesh_device), user=u, seq_len=S)
        p = pcc(pre["x_out"], device_streams_to_host(y, 4, S))
        logger.info(f"layer {LAYER} user {u} prefill S={S}: pcc {p:.6f}")
        assert p > 0.995
        ttnn.deallocate(y)

    steps = [
        (torch.load(R2040 / f"layer{LAYER}_decode{s}.pt"), torch.load(R4097 / f"layer{LAYER}_decode{s}.pt"))
        for s in range(n_steps)
    ]
    streams_dev = ttnn.from_torch(
        _packed(torch.cat([steps[0][0]["x_in"], steps[0][1]["x_in"]], dim=1), 2),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    positions = [steps[0][0]["position"], steps[0][1]["position"]]
    assert positions == [S0, S1]
    dp = dec.make_decode_positions(positions)
    results = []  # (step, user, pos, pcc, mode)

    def check(out_dev, s, mode):
        out = device_streams_to_host(out_dev, 4, 2)
        for u in range(2):
            d = steps[s][u]
            p = pcc(d["x_out"], out[:, u : u + 1])
            results.append((s, u, d["position"], p, mode))
            logger.info(f"  step {s} user {u} pos {d['position']} ({mode}): pcc {p:.6f}")
        return out

    def load_step(s):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                _packed(torch.cat([steps[s][0]["x_in"], steps[s][1]["x_in"]], dim=1), 2),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            ),
            streams_dev,
        )
        dec.update_decode_positions(dp, [steps[s][0]["position"], steps[s][1]["position"]])

    # step 0: the eager reference path (dense flash-MLA + per-user sparse_sdpa rows) and the new path on the
    # same token (the state writes are idempotent at a fixed position)
    monkeypatch.setenv("GLM53_SPARSE_DECODE", "eager")
    y_e = dec.decode_forward(streams_dev, positions=dp)
    out_e = check(y_e, 0, "eager reference")
    ttnn.deallocate(y_e)
    monkeypatch.setenv("GLM53_SPARSE_DECODE", "traced")
    y_n = dec.decode_forward(streams_dev, positions=dp)
    out_n = check(y_n, 0, "new path, eager run")
    ttnn.deallocate(y_n)
    for u in range(2):
        p = pcc(out_e[:, u : u + 1], out_n[:, u : u + 1])
        same = torch.equal(out_e[:, u : u + 1], out_n[:, u : u + 1])
        logger.info(f"  step 0 user {u}: new path vs eager reference pcc {p:.7f} (bit-equal {same})")
        assert p > 0.999
    # step 1 eager through the new path, step 2 = trace capture (warm-up output checked), 3.. replays
    load_step(1)
    y = dec.decode_forward(streams_dev, positions=dp)
    check(y, 1, "new path, eager run")
    ttnn.deallocate(y)
    load_step(2)
    trace_id, out_dev, warm = dec.capture_decode_trace(streams_dev, dp)
    check(warm, 2, "new path, capture warm-up")
    for s in range(3, n_steps):
        load_step(s)
        dec.replay_decode(trace_id)
        check(out_dev, s, "trace replay")
    ttnn.release_trace(mesh_device, trace_id)
    worst = min(r[3] for r in results)
    logger.info(f"two users through {n_steps} steps: worst pcc {worst:.6f} over {len(results)} (user, step) samples")
    assert all(r[3] > 0.995 for r in results), [r for r in results if r[3] <= 0.995]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_selected_pools_vs_hf(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    P, K = cfg.index_kpool, cfg.index_select_pools
    pre = torch.load(R4097 / f"layer{LAYER}_prefill.pt")
    S = pre["x_in"].shape[1]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=1, max_seq_len=S + 64
    )
    dec.prefill_forward(host_streams_to_device(pre["x_in"], mesh_device), user=0, seq_len=S)
    for step in range(12):
        d = torch.load(R4097 / f"layer{LAYER}_decode{step}.pt")
        a = d["position"]
        x = attention_input(raw, cfg, d["x_in"]).reshape(1, 1, 1, -1)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        # the step's own token is not yet in the cache (decode appends first): the HF indexer saw it, so
        # run the decode step first (writes the latent / index rows), then select on the updated index
        dp = dec.make_decode_positions([a])
        y = dec.decode_forward(host_streams_to_device(d["x_in"], mesh_device, pad_to=1), positions=dp)
        p_d = pcc(d["x_out"], device_streams_to_host(y, 4, 1))
        pools = dec.layer.attn.selected_pools(x_tt, dec.cache, 0, a)
        n, t = (a + 1) // P, (a + 1) % P
        assert len(pools) == K and all(0 <= p < n for p in pools), (a, n, pools[:8])
        tt_ids = {P * p + j for p in pools for j in range(P)} | {P * n + j for j in range(t)}
        hf_ids = {int(i) for i in d["index_ids"][0, 0].tolist() if i >= 0}
        inter = len(hf_ids & tt_ids)
        logger.info(
            f"  step {step} pos {a}: layer pcc {p_d:.6f}; ids HF {len(hf_ids)} TT {len(tt_ids)} common {inter} ({inter / len(hf_ids):.4f})"
        )
        assert p_d > 0.995 and inter / len(hf_ids) > 0.97
        ttnn.deallocate(x_tt)
