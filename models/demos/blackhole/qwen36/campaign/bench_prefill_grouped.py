# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped-vs-serial prefill TTFT bench for the concurrent-short-prompt serving target.

Measures ONE B-user prefill wave two ways on the same model instance:

* grouped — a single ``prefill_paged_grouped`` pass (one hybrid forward for the whole
  group, opted past the defensive long-bucket cap via ``QWEN36_GDN_MAX_BG``); every
  user's TTFT is the wave's wall time.
* serial  — B sequential per-user prefills (the ``prefill_paged_peruser`` loop,
  instrumented so user u's TTFT is the cumulative wall time when ITS prefill lands).

Emits one ``BENCH_JSON {...}`` line per case on stdout; set the ``BENCH_JSON`` env var
to a file path to also append the line there.

Knobs (env): ``QWEN36_BENCH_LAYERS`` truncates the model for smoke runs (default: full
model — TTFT numbers are only meaningful at full depth).

Run:
    MESH_DEVICE=P150x8 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest -svq models/demos/blackhole/qwen36/campaign/bench_prefill_grouped.py
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BLOCK_SIZE = 64


def _emit(result):
    line = "BENCH_JSON " + json.dumps(result)
    print(line, flush=True)
    path = os.environ.get("BENCH_JSON")
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def _serial_prefill_instrumented(model, mesh, token_list, page_table, lens):
    """prefill_paged_peruser's exact loop, with a device-synced timestamp after each user."""
    for layer in model.layers:
        if not layer.is_full_attention:
            layer.attention._pending = []
    t0 = time.time()
    ttfts = []
    for u, toks in enumerate(token_list):
        model._prefill_paged_tp(toks, page_table[u : u + 1], valid_len=lens[u], gdn_collect=True)
        ttnn.synchronize_device(mesh)
        ttfts.append(time.time() - t0)
    for layer in model.layers:
        if not layer.is_full_attention:
            layer.attention.finalize_pending()
    ttnn.synchronize_device(mesh)
    return time.time() - t0, ttfts


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B, isl", [(8, 1024), (8, 2048)], ids=["b8-isl1k", "b8-isl2k"])
def test_bench_prefill_grouped(mesh_device, B, isl, monkeypatch, reset_seeds, ensure_gc):
    nd = mesh_device.get_num_devices()
    assert nd > 1, "grouped prefill is the TP (num_devices>1) path"
    n_layers = int(os.environ["QWEN36_BENCH_LAYERS"]) if os.environ.get("QWEN36_BENCH_LAYERS") else None

    bpu = max(8, -(-(isl + 8) // BLOCK_SIZE))
    bpu = ((bpu + 7) // 8) * 8
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=bpu * BLOCK_SIZE, n_layers=n_layers)
    args = model.args
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    kv_shape = (B * bpu, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)

    torch.manual_seed(0)
    lens = [isl] * B
    token_list = [
        torch.tensor([torch.randint(0, args.vocab_size, (isl,)).tolist()], dtype=torch.long) for _ in range(B)
    ]

    monkeypatch.setenv("QWEN36_GDN_MAX_BG", str(B))

    # Compile warmup for both paths (program compilation is a one-time server startup cost,
    # not TTFT — mirrors the demo's outside-the-timer warmups).
    model.prefill_paged_grouped(token_list, page_table, valid_lens=lens, group_size=B)
    _serial_prefill_instrumented(model, mesh_device, token_list, page_table, lens)

    # ---- grouped: one wave, everyone's TTFT is the wave wall time ----
    t0 = time.time()
    model.prefill_paged_grouped(token_list, page_table, valid_lens=lens, group_size=B)
    ttnn.synchronize_device(mesh_device)
    grouped_wall = time.time() - t0

    # ---- serial: B sequential per-user prefills, cumulative per-user TTFT ----
    serial_wall, serial_ttfts = _serial_prefill_instrumented(model, mesh_device, token_list, page_table, lens)

    result = {
        "bench": "prefill_grouped",
        "mesh_devices": nd,
        "n_layers": n_layers or len(model.layers),
        "B": B,
        "isl": isl,
        "grouped": {"wall_s": round(grouped_wall, 4), "ttft_per_user_s": [round(grouped_wall, 4)] * B},
        "serial": {"wall_s": round(serial_wall, 4), "ttft_per_user_s": [round(t, 4) for t in serial_ttfts]},
        "speedup": round(serial_wall / grouped_wall, 3),
    }
    _emit(result)
    logger.info(
        f"grouped B={B} isl={isl}: wave={grouped_wall:.3f}s vs serial={serial_wall:.3f}s "
        f"(worst serial TTFT {serial_ttfts[-1]:.3f}s) — speedup x{result['speedup']}"
    )
