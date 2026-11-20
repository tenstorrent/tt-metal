# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The paged store end to end with the prefill stack (no decode): ``engine.Engine`` drives real storage on the
8x4 mesh. The prefill stack is built in process through its adapter exactly as
``models/demos/deepseek_v3_d_p/tests/hy4/test_hy4_agentx_append.py`` builds it, with ``1 + HY4_KV_REFS`` ordinary
user slots of ``HY4_KV_BENCH`` tokens: slot 0 is the workbench, the others are reference / fixed slots. Prefill
passes run in trace mode (one capture for everything).

``HY4_KV_SCENARIO=interleave`` (correctness): five conversations of random tokens interleaved through the one
workbench: first turns, appends that restart inside a frame, growth, a compaction, an explicit eviction and a
re-prefill, LRU evictions by the manager. At two checkpoints every live conversation is prefilled ALONE into an
ordinary slot with the same passes, and its pool frames are compared with that slot on the device (every layer,
both kinds, every chip, positions [0, ceil32(N))); ``HY4_KV_HOST_CHECK`` conversations are also read to the host
and compared there.

``HY4_KV_SCENARIO=replay``: the first ``HY4_KV_REQUESTS`` requests of the recorded AgentX hour in recorded order,
prompt lengths capped at the workbench, random tokens with the recorded shared prefixes, one request at a time
(prefill only; decode is a stub). Per request: copy-in, prefill passes, copy-out, table rewrite; the
``HY4_KV_REFS`` busiest conversations also run on fixed slots (the same requests without the store).

Env: HY4_KV_LAYERS (6), HY4_KV_BENCH (32768), HY4_KV_POOL_FRAMES (9), HY4_KV_REFS (1), HY4_KV_TRACE (1),
HY4_KV_HOST_CHECK (number of conversations checked on the host per checkpoint: 99), HY4_KV_REQUESTS (120),
HY4_KV_REPORT (JSON path).

Mock Galaxy: the stack is not built; the engine runs against tensors of its own and a prefill stub (programs are
compiled, the bookkeeping runs, nothing is compared)."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.experimental.hy4_preview.fused.kv_store import device as dv
from models.experimental.hy4_preview.fused.kv_store import frames as fr
from models.experimental.hy4_preview.fused.kv_store.engine import Engine, StackPrefill, dram
from models.experimental.hy4_preview.fused.kv_store.manager import KvStore

MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
MESH = (8, 4)
LAYERS = int(os.environ.get("HY4_KV_LAYERS", "6"))
BENCH = int(os.environ.get("HY4_KV_BENCH", "32768"))
POOL_FRAMES = int(os.environ.get("HY4_KV_POOL_FRAMES", "9"))
REFS = int(os.environ.get("HY4_KV_REFS", "1"))
TRACE = os.environ.get("HY4_KV_TRACE", "1") == "1"
SCENARIO = os.environ.get("HY4_KV_SCENARIO", "interleave")
HOST_CHECK = int(os.environ.get("HY4_KV_HOST_CHECK", "99"))
REQUESTS = int(os.environ.get("HY4_KV_REQUESTS", "120"))
REPORT = os.environ.get("HY4_KV_REPORT")
TRACE_REGION = 256 * 1024 * 1024
USERS = 1 + REFS
_PARAMS = dict(fabric_payload_size=moe_fabric_payload(GLM53Config), l1_small_size=1536)
if TRACE:
    _PARAMS["trace_region_size"] = TRACE_REGION


def _write(report):
    if REPORT and not MOCK:
        Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
        Path(REPORT).write_text(json.dumps(report, indent=1, default=str))


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(MESH, torus_xy_device_params(**_PARAMS), id="torus-xy-8x4")],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_engine(mesh_device, device_params, monkeypatch):
    report = {
        "layers": LAYERS,
        "bench_tokens": BENCH,
        "pool_frames": POOL_FRAMES,
        "users": USERS,
        "scenario": SCENARIO,
        "trace": TRACE,
    }
    store = KvStore(POOL_FRAMES, rows=8)
    if MOCK:
        lat = fr.Kind("latent", LAYERS, 656, fr.LATENT.shard_bytes, 704)
        idx = fr.Kind("index", 3, 128, fr.INDEX.shard_bytes, 0)
        bench = [
            dv.allocate(mesh_device, fr.Geometry.contiguous(k, BENCH // fr.FRAME_TOKENS), USERS) for k in (lat, idx)
        ]
        engine = Engine(
            mesh_device,
            store,
            bench[0],
            bench[1],
            lambda slot, s, e, ids: {"prep_s": 0.0, "chunk_s": 0.0},
            bench_users=USERS,
        )
        engine.warm()
        SCENARIOS[SCENARIO](engine, mesh_device, None, None, 1000, report)
        return

    from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
    from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, os.environ.get(hy4_layout.LAYOUT_ENV) or hy4_layout.DEV_LAYOUT)
    monkeypatch.setenv("PREFILL_MAX_SEQ_LEN", str(BENCH))
    adapter = get_adapter("hy4_preview")
    hf_config = adapter.load_hf_config()
    hf_config.max_seq_len = BENCH
    params = PrefillRunParams(
        mesh_shape=MESH,
        num_layers=LAYERS,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=BENCH,
        chunk_size=4096,
        num_users=USERS,
        capacity_factor=int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8)),
        num_links=2,
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=True,
        weight_cache_path=adapter.weight_cache_path(MESH, sp_axis=0),
        tp_shard_kv=adapter.tp_shard_kv_default,
        sp_axis=0,
        tp_axis=1,
        sparse_kv_cache_format=MlaKvCacheFormat.SCALED_FP8,
        use_trace=TRACE,
        overlap_shared_expert_with_dispatch=True,
    )
    t0 = time.perf_counter()
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    report["build_runtime_s"] = time.perf_counter() - t0
    report["dram_after_weights"] = dram(mesh_device)
    kv_caches = None
    try:
        kv_caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
        report["dram_after_caches"] = dram(mesh_device)
        # everything of the store that lives on the device is allocated and launched once BEFORE the capture
        engine = Engine(mesh_device, store, kv_caches.kvpe.storage, kv_caches.index, None, bench_users=USERS)
        report["dram_after_pool"] = dram(mesh_device)
        report["pool_gb_per_chip"] = (
            sum(g.bytes_per_bank(store.num_frames) * fr.BANKS for g in engine.pool_g.values()) / 1e9
        )
        t0 = time.perf_counter()
        runtime.compile(kv_caches)
        report["compile_s"] = time.perf_counter() - t0
        engine.warm()
        report["dram_after_compile"] = dram(mesh_device)
        if TRACE:
            t0 = time.perf_counter()
            runtime.capture_trace(kv_caches)
            ttnn.synchronize_device(mesh_device)
            report["capture_s"] = time.perf_counter() - t0
            report["trace_segments"] = runtime._controller.num_segments
            report["trace_bytes"] = runtime._controller.trace_bytes()
            report["dram_after_capture"] = dram(mesh_device)
        engine.prefill = StackPrefill(runtime, kv_caches, mesh_device, MESH, trace=TRACE)
        logger.info(
            f"HY4_MK10 engine {LAYERS} layers, {USERS} user slots of {BENCH} tokens (slot 0 the workbench), pool "
            f"{POOL_FRAMES} frames = {report['pool_gb_per_chip']:.2f} GB per chip, trace {TRACE}: build "
            f"{report['build_runtime_s']:.0f} s, compile {report['compile_s']:.0f} s, capture {report.get('capture_s', 0):.1f} s, "
            f"DRAM per chip: weights {report['dram_after_weights']['allocated_gb']:.2f} GB, + slots "
            f"{report['dram_after_caches']['allocated_gb']:.2f}, + pool {report['dram_after_pool']['allocated_gb']:.2f}, "
            f"after compile and capture {report.get('dram_after_capture', report['dram_after_compile'])['allocated_gb']:.2f}"
        )
        _write(report)
        SCENARIOS[SCENARIO](engine, mesh_device, runtime, kv_caches, int(hf_config.vocab_size), report)
    finally:
        _write(report)
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import clear_tt_ccl_cache

        for step in (runtime.release_trace, clear_tt_ccl_cache):
            try:
                step()
            except Exception as e:
                logger.warning(f"HY4_MK10 engine teardown step {step.__name__} failed: {e}")
        ttnn.synchronize_device(mesh_device)
    runtime.model = None
    del runtime
    gc.collect()
    assert not report.get("errors"), report["errors"]


# ---------------------------------------------------------------------------------------------------------------
# Host comparison of one conversation: pool frames against an ordinary slot
# ---------------------------------------------------------------------------------------------------------------
def host_check(engine, mesh_device, runtime, kv_caches, key, slot, layers):
    """Byte / value comparison on the host of the conversation's pool rows with user slot ``slot``, for the
    given layers, on all chips, positions [0, ceil32(N)); also counts the rows that lie beyond them in the
    conversation's last pool frame and are not zero (stale rows of whoever used the workbench before)."""
    from models.demos.deepseek_v3_d_p.tt.hy4.kv_cache import read_chip_slabs

    stripe = runtime.kv_stripe
    conv = engine.store.conversation(key)
    n32 = -(-len(engine.prompts[key]) // 32) * 32
    whole = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    full_layers = [l for l in runtime.kv_layout(kv_caches)["full_indexer_layers"] if l < engine.latent.layers]
    out = {"latent_rows": 0, "latent_bad": 0, "index_rows": 0, "index_bad": 0, "stale_rows_beyond": 0}
    for kind, tensor in ((engine.latent, kv_caches.kvpe.storage), (engine.index, kv_caches.index)):
        slabs = read_chip_slabs(tensor, mesh_device, stripe)  # [chips, slots, rows, width]
        pool = ttnn.to_torch(engine.pool[kind], mesh_composer=whole)
        chips = slabs.shape[0]
        pool = pool.reshape(chips, engine.store.num_frames * kind.layers, fr.FRAME_ROWS, -1)
        if kind is engine.latent:
            slabs, pool = slabs.contiguous().view(torch.uint8), pool.contiguous().view(torch.uint8)
            slabs, pool = slabs.reshape(chips, -1, slabs.shape[2], 656), pool.reshape(chips, -1, fr.FRAME_ROWS, 656)
        which = layers if kind is engine.latent else [full_layers.index(l) for l in layers if l in full_layers]
        for chip in range(chips):
            visible = fr.visible_rows(n32, chip)
            for layer in which:
                for k, f in enumerate(conv.frames):
                    rows = min(max(visible - k * fr.FRAME_ROWS, 0), fr.FRAME_ROWS)
                    mine = pool[chip, f * kind.layers + layer]
                    ref = slabs[chip, slot * kind.layers + layer, k * fr.FRAME_ROWS : (k + 1) * fr.FRAME_ROWS]
                    bad = int((mine[:rows] != ref[:rows]).any(-1).sum())
                    out[f"{kind.name}_rows"] += rows
                    out[f"{kind.name}_bad"] += bad
                    if kind is engine.latent:
                        out["stale_rows_beyond"] += int((mine[rows:] != 0).any(-1).sum())
    return out


def checkpoint(engine, mesh_device, runtime, kv_caches, keys, report, name, slot=1):
    """Every conversation of ``keys``: alone into ordinary slot ``slot``, then pool against slot."""
    results, previous = [], None
    spread = sorted(
        {0, 1, 2, 5, engine.latent.layers // 2, engine.latent.layers - 2, engine.latent.layers - 1}
        & set(range(engine.latent.layers))
    )
    for i, key in enumerate(keys):
        if MOCK:
            continue
        if previous is not None:  # the slot still holds the previous conversation: this one must NOT match it
            r = engine.compare(key, slot)
            assert (
                r["latent"]["bad"] > 0 and r["index"]["bad"] > 0
            ), f"{name}: {key} equals {previous}'s reference: the compare proves nothing"
        passes = engine.reference(key, slot)
        r = engine.compare(key, slot)
        row = {
            "checkpoint": name,
            "key": key,
            "prompt": len(engine.prompts[key]),
            "frames": engine.store.conversation(key).frames,
            "reference_passes": passes,
            **r,
        }
        if i < HOST_CHECK:
            row["host"] = host_check(engine, mesh_device, runtime, kv_caches, key, slot, spread)
        results.append(row)
        logger.info(
            f"HY4_MK10 engine {name} conversation {key}: {row['prompt']} tokens in frames {row['frames']}; against a prefill alone in "
            f"slot {slot} ({passes} passes): latent {r['latent']['rows']} rows, {r['latent']['bad']} differ; index keys "
            f"{r['index']['rows']} blocks, {r['index']['bad']} differ (device, all layers, {mesh_device.get_num_devices()} chips)"
            + (f"; host, layers {spread}: {row['host']}" if "host" in row else "")
        )
        if r["latent"]["bad"] or r["index"]["bad"]:
            report.setdefault("errors", []).append(f"{name} {key}: pool differs from the reference: {r}")
        if "host" in row and (row["host"]["latent_bad"] or row["host"]["index_bad"]):
            report.setdefault("errors", []).append(f"{name} {key}: host comparison differs: {row['host']}")
        assert MOCK or (r["latent"]["rows"] > 0 and r["index"]["rows"] > 0)
        previous = key
    report.setdefault("checkpoints", []).extend(results)
    _write(report)


# ---------------------------------------------------------------------------------------------------------------
# Scenario 1: conversations interleaved through the one workbench
# ---------------------------------------------------------------------------------------------------------------
def interleave(engine, mesh_device, runtime, kv_caches, vocab, report):
    assert (
        engine.capacity_frames >= 3 and engine.store.num_frames == 9
    ), "the scenario is written for a pool of 9 frames"
    gen = torch.Generator().manual_seed(0)
    stream = {name: torch.randint(0, vocab, (24000,), generator=gen).numpy() for name in "ABCDEX"}
    steps = report["steps"] = []

    def turn(key, tokens, note):
        rec = engine.begin(key, tokens)
        assert rec is not None
        engine.end(key)
        engine.store.check()
        steps.append(
            {
                "note": note,
                **{k: v for k, v in rec.items() if k != "passes"},
                "num_passes": len(rec["passes"]),
                "frames_now": list(engine.store.conversation(key).frames),
            }
        )
        logger.info(
            f"HY4_MK10 engine step {note}: {key} {rec['prompt']} tokens, restart {rec['restart']}, frames {steps[-1]['frames_now']}; "
            f"copy-in {rec['copy_in_frames']} frames {1e3 * rec['copy_in_s']:.2f} ms, {len(rec['passes'])} passes "
            f"{rec['prefill_s']:.3f} s, copy-out {rec['copy_out_frames']} frames {1e3 * rec['copy_out_s']:.2f} ms"
        )
        return rec

    turn("A", stream["A"][:9000], "A turn 1")
    turn("B", stream["B"][:13500], "B turn 1")
    r = turn("A", stream["A"][:9700], "A turn 2 (append, restart 8192)")
    assert r["restart"] == 8192 and r["copy_in_frames"] == 1
    turn("C", stream["C"][:4100], "C turn 1")
    r = turn("A", stream["A"][:17000], "A turn 3 (grows by a frame)")
    assert r["restart"] == 8192 and r["new_frames"] == 1
    engine.evict("B")
    r = turn("B", stream["B"][:14500], "B again after an explicit eviction (from scratch)")
    assert r["restart"] == 0 and r["copy_in_frames"] == 0
    turn("D", stream["D"][:20000], "D turn 1 (the pool is full)")
    r = turn("C", stream["C"][:4400], "C turn 2 (restart 4096)")
    assert r["restart"] == 4096
    r = turn(
        "A",
        np.concatenate([stream["A"][:5000], stream["X"][:3000]]),
        "A turn 4 (compaction: 5000 shared, shrinks to one frame)",
    )
    assert r["restart"] == 4096 and r["frames"] == 1
    r = turn("B", stream["B"][:16500], "B turn 3 (restart 12288, inside frame 1)")
    assert r["restart"] == 12288 and r["copy_in_frames"] == 2
    checkpoint(engine, mesh_device, runtime, kv_caches, ["A", "B", "C", "D"], report, "checkpoint 1")
    evictions = engine.store.stats.evictions
    r = turn("E", stream["E"][:20000], "E turn 1 (the manager evicts the least recently used: D)")
    assert engine.store.stats.evictions == evictions + 1 and not engine.store.conversation("D").frames
    r = turn("D", stream["D"][:20500], "D again after the LRU eviction (from scratch; evicts C and A)")
    assert r["restart"] == 0 and r["reprefill"] == 20000
    r = turn("B", stream["B"][:18000], "B turn 4")
    checkpoint(engine, mesh_device, runtime, kv_caches, ["B", "D", "E"], report, "checkpoint 2")
    st = engine.store.stats
    report["store"] = {
        "evictions": st.evictions,
        "frames_allocated": st.frames_allocated,
        "reprefill_tokens": st.reprefill_tokens,
    }
    logger.info(
        f"HY4_MK10 engine interleave done: {len(steps)} turns, {st.evictions} evictions, errors: {report.get('errors')}"
    )


# ---------------------------------------------------------------------------------------------------------------
# Scenario 2: a slice of the recorded hour
# ---------------------------------------------------------------------------------------------------------------
def replay(engine, mesh_device, runtime, kv_caches, vocab, report):
    model = "generated/hy4_preview/agentx_model"
    if model not in sys.path:
        sys.path.insert(0, model)
    import workload

    reqs, _ = workload.load()
    reqs = sorted(reqs, key=lambda q: q.rec_start)[:REQUESTS]
    cap = engine.capacity_frames * fr.FRAME_TOKENS - 4096
    count = {}
    for q in reqs:
        count[q.stream] = count.get(q.stream, 0) + 1
    fixed = {
        key: 1 + i for i, (key, _) in enumerate(sorted(count.items(), key=lambda kv: -kv[1])[:REFS])
    }  # key -> slot
    names = {key: f"c{i:02d}" for i, key in enumerate(count)}
    gen = torch.Generator().manual_seed(1)
    prompts, fixed_prompt, records = {}, {}, []
    report.update(requests=len(reqs), conversations=len(count), cap=cap, fixed={names[k]: s for k, s in fixed.items()})
    row_cycle = 0
    for i, q in enumerate(reqs):
        isl = min(q.isl, cap)
        prev = prompts.get(q.stream)
        lcp = 0 if prev is None else (min(len(prev), isl) if q.lcp >= len(prev) else min(q.lcp, isl))
        fresh = torch.randint(0, vocab, (isl - lcp,), generator=gen).numpy()
        tokens = fresh if prev is None else np.concatenate([prev[:lcp], fresh])
        prompts[q.stream] = tokens
        evictions = engine.store.stats.evictions
        rec = engine.begin(q.stream, tokens)
        assert rec is not None
        # decode stub: the conversation gets a decode row, the tables are rewritten, the row is given back
        row = row_cycle % engine.store.rows
        row_cycle += 1
        engine.store.bind(row, q.stream)
        rec["table_s"] = engine.sync_tables()
        engine.store.unbind(row)
        engine.end(q.stream)
        rec.update(
            i=i,
            key=names[q.stream],
            isl_recorded=q.isl,
            num_passes=len(rec["passes"]),
            evictions=engine.store.stats.evictions - evictions,
        )
        if q.stream in fixed:  # the same request on a fixed slot of the stack: passes only
            slot = fixed[q.stream]
            was = fixed_prompt.get(slot)
            shared = 0
            if was is not None:
                n = min(len(was), len(tokens))
                differ = np.nonzero(was[:n] != tokens[:n])[0]
                shared = int(differ[0]) if differ.size else n
            start = shared // 4096 * 4096
            t0 = time.perf_counter()
            f = (
                engine.prefill_slot(slot, start, len(tokens), tokens)
                if len(tokens) > start
                else {"passes": [], "prefill_s": 0.0, "prep_s": 0.0}
            )
            rec.update(
                fixed_total_s=time.perf_counter() - t0,
                fixed_prefill_s=f["prefill_s"],
                fixed_passes=len(f["passes"]),
                fixed_restart=start,
            )
            fixed_prompt[slot] = tokens
        rec.pop("passes")
        records.append(rec)
        if i % 10 == 0 or rec["evictions"]:
            logger.info(
                f"HY4_MK10 engine replay {i}: {rec['key']} prompt {rec['prompt']} (recorded {q.isl}), restart {rec['restart']}, "
                f"copy-in {rec['copy_in_frames']} frames {1e3 * rec['copy_in_s']:.2f} ms, {rec['num_passes']} passes {rec['prefill_s']:.2f} s, "
                f"copy-out {rec['copy_out_frames']} frames {1e3 * rec['copy_out_s']:.2f} ms, tables {1e3 * rec['table_s']:.2f} ms, evictions {rec['evictions']}"
            )
        report["records"] = records
        if i % 20 == 0:
            _write(report)
    engine.store.check()
    # the fixed conversations: pool against their fixed slot, on the device
    for key, slot in fixed.items():
        if engine.store.conversation(key).frames and not MOCK:
            r = engine.compare(key, slot)
            report.setdefault("fixed_compare", []).append({"key": names[key], "slot": slot, **r})
            logger.info(
                f"HY4_MK10 engine replay conversation {names[key]} ({count[key]} requests, {len(prompts[key])} tokens): pool against "
                f"its fixed slot {slot}: latent {r['latent']['rows']} rows, {r['latent']['bad']} differ; index keys {r['index']['rows']} blocks, {r['index']['bad']} differ"
            )
            if r["latent"]["bad"] or r["index"]["bad"]:
                report.setdefault("errors", []).append(f"replay {names[key]}: pool differs from its fixed slot: {r}")
    summarize(engine, records, report)


def summarize(engine, records, report):
    def q(vals, p):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * p))] if vals else 0.0

    work = [r for r in records if r["num_passes"]]
    tot = lambda k, rs=records: sum(r[k] for r in rs)
    st = engine.store.stats
    s = report["summary"] = {
        "requests": len(records),
        "with_passes": len(work),
        "passes": tot("num_passes"),
        "prefill_s": tot("prefill_s"),
        "prep_s": tot("prep_s"),
        "copy_in_s": tot("copy_in_s"),
        "copy_out_s": tot("copy_out_s"),
        "table_s": tot("table_s"),
        "store_s": tot("store_s"),
        "total_s": tot("total_s"),
        "plan_write_s": tot("copy_in_plan_s") + tot("copy_out_plan_s"),
        "copy_in_frames": tot("copy_in_frames"),
        "copy_out_frames": tot("copy_out_frames"),
        "per_request_ms": {
            name: {
                "mean": 1e3 * tot(k, work) / max(len(work), 1),
                "p50": 1e3 * q([r[k] for r in work], 0.5),
                "max": 1e3 * max([r[k] for r in work], default=0.0),
            }
            for name, k in (
                ("copy_in", "copy_in_s"),
                ("copy_out", "copy_out_s"),
                ("table", "table_s"),
                ("store_total", "store_s"),
                ("prefill", "prefill_s"),
            )
        },
        "evictions": st.evictions,
        "reprefill_tokens": st.reprefill_tokens,
        "frames_allocated": st.frames_allocated,
        "peak_frames": st.peak_frames,
    }
    fx = [r for r in records if "fixed_total_s" in r]
    same = [r for r in fx if r["fixed_passes"] == r["num_passes"]]
    s["fixed"] = {
        "requests": len(fx),
        "same_passes": len(same),
        "store_path_s": sum(r["total_s"] + r["table_s"] for r in same),
        "fixed_slot_s": sum(r["fixed_total_s"] for r in same),
        "store_passes": sum(r["num_passes"] for r in fx),
        "fixed_passes": sum(r["fixed_passes"] for r in fx),
    }
    pr = s["per_request_ms"]
    logger.info(
        f"HY4_MK10 engine replay summary: {s['requests']} requests ({s['with_passes']} with new tokens), {report.get('conversations')} conversations, "
        f"{s['passes']} passes {s['prefill_s']:.1f} s (+ {s['prep_s']:.1f} s input upload); store: copy-in {s['copy_in_frames']} frames {s['copy_in_s']:.3f} s, "
        f"copy-out {s['copy_out_frames']} frames {s['copy_out_s']:.3f} s (of both, {s['plan_write_s']:.3f} s is the host writing the plans), tables {s['table_s']:.3f} s; per request with new tokens: copy-in "
        f"{pr['copy_in']['mean']:.2f} ms (p50 {pr['copy_in']['p50']:.2f}, max {pr['copy_in']['max']:.2f}), copy-out {pr['copy_out']['mean']:.2f} ms "
        f"(p50 {pr['copy_out']['p50']:.2f}, max {pr['copy_out']['max']:.2f}), tables {pr['table']['mean']:.2f} ms, everything the store adds "
        f"{pr['store_total']['mean']:.2f} ms (p50 {pr['store_total']['p50']:.2f}, max {pr['store_total']['max']:.2f}) beside {pr['prefill']['mean']:.0f} ms of passes; "
        f"evictions {s['evictions']}, prefilled again {s['reprefill_tokens']} tokens, peak {s['peak_frames']} frames"
    )
    logger.info(
        f"HY4_MK10 engine replay fixed slots: {s['fixed']['requests']} requests of the fixed conversations, {s['fixed']['same_passes']} with the same passes on "
        f"both paths: store path {s['fixed']['store_path_s']:.2f} s, fixed slot {s['fixed']['fixed_slot_s']:.2f} s; passes store {s['fixed']['store_passes']}, fixed {s['fixed']['fixed_passes']}"
    )
    _write(report)


SCENARIOS = {"interleave": interleave, "replay": replay}
