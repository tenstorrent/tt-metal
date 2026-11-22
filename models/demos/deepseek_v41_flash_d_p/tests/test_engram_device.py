# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.87 (ENGRAM_DEVICE=on), op level on the mesh alone (no model): tt/engram_device.EngramLookup against the host
path (tt/engram.TtEngramRows.host_bfp8: the reference hash, the checkpoint rows, the bfloat8_b packing) on real ids
(agentic / prose texts), every chip's [Q, 6144] rows compared value for value (bit-identical bfloat8_b), the
embedding's natural ids, the slot's device id history and the tail it carries into the next chunk:
  * a chunk at 145408 (not chunk-aligned: the server order wraps) after a history uploaded from the host,
  * the next chunk with a padded tail (valid 3000: zero rows past it), its tail from the device history,
  * a chunk at 0 (DEAD tail).
Also the one-time table upload (seconds) and the lookup's time a call (synced loop).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_engram_device.py"
"""

import importlib
import os
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BATCH_SHARED = True
TEXTS = [t for t in os.environ.get("P87_TEXTS", "agentic,prose").split(",") if t]


def _host_tiles_to_float(ttnn, tiles: torch.Tensor, q: int, width: int) -> torch.Tensor:
    t = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(tiles.numpy().tobytes(), [1, 1, q, width])
    return ttnn.to_torch(t).float().reshape(q, width)


@mesh_parametrize(timeout_s=5400)
def test_engram_device(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.common.bringup.reference import texts
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
    from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint, EngramTable

    ed = importlib.import_module("models.demos.deepseek_v41_flash_d_p.tt.engram_device")  # (DS41-311)
    from models.demos.deepseek_v41_flash_d_p.tt.engram import TtEngramRows, engram_hasher
    from models.demos.deepseek_v41_flash_d_p.tt.engram_bfp8 import tile_bytes
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir
    from models.demos.deepseek_v41_flash_d_p.tt.runners.adapter import model_path

    mesh = mesh_device
    path = model_path()
    cfg = Config.from_dir(path)
    ck = Checkpoint(path)
    hasher = engram_hasher(path, cfg)
    layers = [i for i in hasher.layer_ids]
    chunk = int(S.get("target.chunk"))
    n = ed.chips(mesh)
    Q = chunk // n
    R, C = tuple(mesh.shape)
    W = chunk // R
    assert ed.fits(mesh, cfg), "the tables do not fit this mesh (ENGRAM_DEVICE_MAX_GB)"
    t0 = time.time()
    nums = dict(zip(cfg.raw["engram_layer_ids"], cfg.raw["engram_num_embeddings"]))
    tables = {i: ed.load_table(mesh, ck, i, nums[i], expert_cache_dir(S)) for i in layers}
    load_s = time.time() - t0
    print(f"P87 tables ready in {load_s:.1f} s (upload s: {[tables[i].load_s for i in layers]})", flush=True)
    metrics.record("p87_table_load_s", round(max([load_s] + [t.load_s or 0 for t in tables.values()]), 1))
    links = os.environ.get("P87_LINKS")  # (a sweep of the exchange's links a direction; default the Setting)
    lookup = ed.EngramLookup(mesh, hasher, tables, chunk, links=int(links) if links else None)
    host = {i: TtEngramRows(mesh, hasher, EngramTable(ck, i), i) for i in layers}
    vocab = lookup.vocab
    max_seq = 150528 + chunk
    bad = []
    ids, _ = texts.tokens(S, max_seq, TEXTS[0])
    for text in [] if os.environ.get("P87_SKIP_CHECK") else TEXTS:
        ids, info = texts.tokens(S, max_seq, text)
        hist_host = torch.zeros(max_seq, dtype=torch.int64)
        start0 = 145408
        hist_host[:start0] = ids[:start0]
        explicit = text != TEXTS[0]  # (the second text: an empty device history, the tail given explicitly)
        pages = -(-max_seq // ed.HIST_PAGE)
        hh = torch.zeros(pages * ed.HIST_PAGE, dtype=torch.int64)
        hh[:max_seq] = 0 if explicit else hist_host
        hist = ttnn.from_torch(
            hh.to(torch.int32).view(1, 1, pages, ed.HIST_PAGE),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for start, valid in ((start0, chunk), (start0 + chunk, 3000), (0, chunk)):
            nat = torch.full((chunk,), 0xFFFFFFFF, dtype=torch.int64)
            nat[:valid] = ids[start : start + valid]
            tail = hasher.new_tail()
            last = hist_host[max(0, start - 3) : start]
            if last.numel():
                tail[-last.numel() :] = hasher.token_map[last]
            ids_in = ed.engine_ids(mesh, nat, start, chunk)
            tl = ed.new_tail(mesh, tail) if explicit and start == start0 else None
            outs, emb = lookup(ids_in, hist, start, valid, tl)
            ttnn.synchronize_device(mesh)
            st = lookup.read_status()
            assert all(s == [0] * 8 for s in st), ("lookup status (timeouts / bad meta)", text, start, st)
            for i in layers:
                exps, mags, _ = host[i].host_bfp8(nat, tail, valid, fast=True)
                dev = [ttnn.to_torch(t).float().reshape(Q, -1) for t in ttnn.get_device_tensors(outs[i])]
                for q in range(n):
                    want = _host_tiles_to_float(
                        ttnn, tile_bytes(exps[q * Q : (q + 1) * Q], mags[q * Q : (q + 1) * Q]), Q, lookup.width
                    )
                    if not torch.equal(dev[q], want):
                        diff = dev[q] != want
                        rows = diff.any(-1).nonzero().reshape(-1).tolist()
                        bad.append((text, start, i, q, int(diff.sum()), rows[:8]))
            # the embedding's natural ids (pads clamped) and the slot history
            want_emb = nat.clone()
            want_emb[valid:] = vocab - 1
            want_emb = want_emb.clamp(max=vocab - 1)
            for k, t in enumerate(ttnn.get_device_tensors(emb)):
                r = k // C
                got = ttnn.to_torch(t).reshape(-1).to(torch.int64) & 0xFFFFFFFF
                if not torch.equal(got, want_emb[r * W : (r + 1) * W]):
                    bad.append((text, start, "emb", k, int((got != want_emb[r * W : (r + 1) * W]).sum())))
            hist_host[start : start + valid] = nat[:valid]
            for k, t in enumerate(ttnn.get_device_tensors(hist)):
                got = ttnn.to_torch(t).reshape(-1).to(torch.int64)[:max_seq] & 0xFFFFFFFF
                lo = start0 if explicit and start >= start0 else 0
                if not torch.equal(got[lo : start + valid], hist_host[lo : start + valid]):
                    bad.append((text, start, "hist", k))
                    break
            print(f"P87 {text} start {start} valid {valid}: {len(bad)} mismatches so far", flush=True)
            assert not bad, f"device engram != host path (stopped at the first wrong call): {bad[:20]}"
        ttnn.deallocate(hist)
    assert not bad, f"device engram != host path: {bad[:20]}"
    # time a call (synced loop of the same chunk; the program reads the same history positions)
    hist = ed.new_history(mesh, max_seq)
    nat = ids[:chunk].clone()
    ids_in = ed.engine_ids(mesh, nat, 0, chunk)
    for _ in range(3):
        lookup(ids_in, hist, 0, chunk)
    ttnn.synchronize_device(mesh)
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        lookup(ids_in, hist, 0, chunk)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) * 1e3 / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        lookup(ids_in, hist, 0, chunk)
    host_ms = (time.perf_counter() - t0) * 1e3 / reps
    ttnn.synchronize_device(mesh)
    print(
        f"P87 lookup: {ms:.3f} ms a call (synced), host enqueue {host_ms:.3f} ms; cores a chip {lookup.cores}",
        flush=True,
    )
    metrics.record("p87_lookup_ms", round(ms, 3))
    metrics.record("p87_lookup_host_ms", round(host_ms, 3))
