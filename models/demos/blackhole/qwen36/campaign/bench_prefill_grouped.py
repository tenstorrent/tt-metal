# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped-vs-serial prefill TTFT bench for the concurrent-short-prompt serving target.

Measures ONE B-user prefill wave two ways on the same model instance:

* serial  — B sequential per-user prefills (the ``prefill_paged_peruser`` loop,
  instrumented so user u's TTFT is the cumulative wall time when ITS prefill lands).
  Runs FIRST and emits its numbers before the grouped pass starts, so a grouped-path
  hang still leaves the baseline in the log.
* grouped — a single ``prefill_paged_grouped`` pass (batched GDN over the whole group,
  opted past the defensive long-bucket cap via ``QWEN36_GDN_MAX_BG``); every user's
  TTFT is the wave's wall time.

Emits one ``BENCH_JSON {...}`` line per case on stdout (plus a partial one after the
serial phase); set the ``BENCH_JSON`` env var to a file path to also append the lines
there. Every phase boundary prints a ``[bench-grouped] ...`` marker to stdout — the
"grouped" token keeps the markers visible through log pipelines that grep for it.

Hang self-triage: a device-op timeout is armed by default (a hung dispatch raises and
dumps inspector state instead of freezing until walltime). The env is applied here via
setdefault before devices open; the recipe below makes it explicit, and
TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE can attach tt-triage on top.

Knobs (env): ``QWEN36_BENCH_LAYERS`` truncates the model for smoke runs (default: full
model — TTFT numbers are only meaningful at full depth). ``QWEN36_BENCH_OP_TIMEOUT_S``
overrides the armed device-op timeout (default 180; "0" disables).

Run:
    MESH_DEVICE=P150x8 HF_MODEL=Qwen/Qwen3.6-27B \
    TT_METAL_OPERATION_TIMEOUT_SECONDS=180 \
    TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1 \
      pytest -svq models/demos/blackhole/qwen36/campaign/bench_prefill_grouped.py
"""

import json
import os
import time

# Armed before ttnn opens devices (rtoptions reads env at device init): a hung dispatch
# must raise + serialize inspector state, not freeze the job until walltime.
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", os.environ.get("QWEN36_BENCH_OP_TIMEOUT_S", "180"))
os.environ.setdefault("TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT", "1")

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BLOCK_SIZE = 64


def _mark(msg):
    """Stdout phase marker. The [bench-grouped] tag carries the "grouped" token so the
    marker survives keyword-grep log filters; flush so a hang can't swallow it."""
    print(f"[bench-grouped] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _emit(result):
    line = "BENCH_JSON " + json.dumps(result)
    print(line, flush=True)
    path = os.environ.get("BENCH_JSON")
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def _serial_prefill_instrumented(model, mesh, token_list, page_table, lens, tag):
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
        _mark(f"{tag} serial user {u} done at +{ttfts[-1]:.2f}s")
    for layer in model.layers:
        if not layer.is_full_attention:
            layer.attention.finalize_pending()
    ttnn.synchronize_device(mesh)
    return time.time() - t0, ttfts


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize(
    "B, isl", [(8, 1024), (8, 2048), (8, "ragged")], ids=["b8-isl1k", "b8-isl2k", "b8-ragged"]
)
def test_bench_prefill_grouped(mesh_device, B, isl, monkeypatch, reset_seeds, ensure_gc):
    # "ragged": mixed per-user lengths (alternating 1k/2k) — the serving-realistic case.
    ragged = isl == "ragged"
    if ragged:
        isl = 2048  # bucket/KV sizing uses the max
    nd = mesh_device.get_num_devices()
    assert nd > 1, "grouped prefill is the TP (num_devices>1) path"
    n_layers = int(os.environ["QWEN36_BENCH_LAYERS"]) if os.environ.get("QWEN36_BENCH_LAYERS") else None

    bpu = max(8, -(-(isl + 8) // BLOCK_SIZE))
    bpu = ((bpu + 7) // 8) * 8
    _mark(f"model load start (B={B} isl={isl} layers={n_layers or 'full'})")
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=bpu * BLOCK_SIZE, n_layers=n_layers)
    args = model.args
    grid = mesh_device.compute_with_storage_grid_size()
    grid_cap = (grid.x * grid.y) // args.gdn_nv_tp
    assert grid_cap >= B, f"fused GDN grid ceiling {grid_cap} < B={B}; grouped wave would silently degrade"
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    kv_shape = (B * bpu, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)
    _mark(f"model ready ({len(model.layers)} layers, grid_cap={grid_cap})")

    torch.manual_seed(0)
    lens = [1024 if u % 2 == 0 else 2048 for u in range(B)] if ragged else [isl] * B
    token_list = [
        torch.tensor([torch.randint(0, args.vocab_size, (L,)).tolist()], dtype=torch.long) for L in lens
    ]

    monkeypatch.setenv("QWEN36_GDN_MAX_BG", str(B))
    base = {"bench": "prefill_grouped", "mesh_devices": nd, "n_layers": len(model.layers), "B": B, "isl": ("ragged-1k2k" if ragged else isl)}

    # ---- serial phase first (silicon-proven path): warmup, timed, EMIT — so the baseline
    # numbers land even if the grouped pass hangs. Compile warmups are a one-time server
    # startup cost, not TTFT (mirrors the demo's outside-the-timer warmups).
    _mark("serial warmup start")
    _serial_prefill_instrumented(model, mesh_device, token_list, page_table, lens, "warmup")
    _mark("serial timed start")
    serial_wall, serial_ttfts = _serial_prefill_instrumented(model, mesh_device, token_list, page_table, lens, "timed")
    _mark(f"serial timed done: wall={serial_wall:.3f}s")
    _emit(
        dict(
            base,
            phase="serial",
            serial={"wall_s": round(serial_wall, 4), "ttft_per_user_s": [round(t, 4) for t in serial_ttfts]},
        )
    )

    # ---- grouped: one wave, everyone's TTFT is the wave wall time ----
    _mark("grouped warmup start")
    model.prefill_paged_grouped(token_list, page_table, valid_lens=lens, group_size=B)
    ttnn.synchronize_device(mesh_device)
    _mark("grouped warmup done; grouped timed start")
    t0 = time.time()
    model.prefill_paged_grouped(token_list, page_table, valid_lens=lens, group_size=B)
    ttnn.synchronize_device(mesh_device)
    grouped_wall = time.time() - t0
    _mark(f"grouped timed done: wall={grouped_wall:.3f}s")

    result = dict(
        base,
        phase="final",
        grouped={"wall_s": round(grouped_wall, 4), "ttft_per_user_s": [round(grouped_wall, 4)] * B},
        serial={"wall_s": round(serial_wall, 4), "ttft_per_user_s": [round(t, 4) for t in serial_ttfts]},
        speedup=round(serial_wall / grouped_wall, 3),
    )
    _emit(result)
    _mark(
        f"B={B} isl={isl}: grouped wave={grouped_wall:.3f}s vs serial={serial_wall:.3f}s "
        f"(worst serial TTFT {serial_ttfts[-1]:.3f}s) — speedup x{result['speedup']}"
    )
