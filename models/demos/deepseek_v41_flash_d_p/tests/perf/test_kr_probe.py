# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.KR (OWNER 2026-11-21) first probes, before building the owned index-key gather + local scoring.

(a) test_kr_attn_ops: the per-op device profile (real-time profiler, op rows) of ONE chip-aligned 4 x 2k chunk and ONE
    single 8192 chunk at KR_POS (1024000), steady (the real previous chunk untimed first, PB.9C's method), after warm
    runs. Printed and written (KR_OUT): per section the slowest chip's ms, per (section, op) the slowest chip's summed
    ms, calls, programs, timeline gap / slot ms; the attention phase's rows single vs 4 x 2k side by side.
(b) test_kr_score: op level on the held model's plain indexer objects (no layer runs), a 1M slot's capacity, the
    model's dtypes and configs, random data, at KR_POS:
      ranges   today's glued one-call ring (lane_ranges, q chunk 32): four chip-aligned 2048-row segments, each its
               own key buffer (every buffer gathered over the SP ring to every chip)
      single   the single-request ring (one key buffer, the model's q chunk)
      local    the non-ring ttnn.bringup.indexer_score_dsa on keys already on the chip (natural order, the same q
               chunk 32 config, same compute config): the compute-time proxy of a local scorer on pre-gathered keys
      send     the owned send of one source's decoded keys (bfloat8_b tiles, one page a tile) to the chips each segment
               owns (tt/owned_gather.OwnedSend, column groups, 4 segments x 8 chips), on the gather column
    ms a call: KR_REPS calls back to back between syncs, median of KR_ITERS.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "<the walls env> \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_kr_probe.py"
"""

from __future__ import annotations

import importlib
import json
import os
import statistics
import time
from collections import defaultdict

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden, warm_up
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request)

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("KR_POS", "KR_ITERS", "KR_REPS", "KR_OUT", "KR_PARTS")
BATCH_POS_VARS = ("KR_POS",)
BATCH_LAYERS = "executed"  # (the ladder's load: layers 0-20 under decode)
CHUNK = _pc.CHUNK
SHIFT = 4096
LENS = (2048, 2048, 2048, 2048)
POS = 1024000
RANGE_STRIDE = 41


def _knob(name: str, default):
    v = os.environ.get(name, "").strip()
    return type(default)(v) if v else default


def _parts() -> set:
    return {x.strip() for x in os.environ.get("KR_PARTS", "ops,score").split(",") if x.strip()}


def _last_layer(model, layers) -> int:
    return max(j for j in layers if j not in getattr(model, "kv_only", ()))


def _runner(runtime, kv, mesh, layers, text, p: int, path: str):
    """run(prev, rot, prof) -> (wall ms, profile or None) for ``path`` single8192 / 4x2k at history p."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as PD
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment

    model = runtime.model
    n = len(LENS)
    states = [runtime.state(kv, k) for k in range(n)]

    def run(prev: bool, rot: int, prof: bool = False):
        if path == "single8192":
            s = p - (CHUNK if prev else 0)
            desc = ChunkDescriptor.single(CHUNK, 0, s, s + CHUNK)
            ids, sts = [text[s : s + CHUNK].long()], [states[0]]
        else:
            order = [(k + rot) % n for k in range(n)]
            segs, ids, sts = [], [], []
            for k in order:
                s = p - (LENS[k] if prev else 0)
                segs.append(Segment(k, s, s + LENS[k], None))
                ids.append(text[k * SHIFT + s : k * SHIFT + s + LENS[k]].long())
                sts.append(states[k])
            desc = ChunkDescriptor.packed(CHUNK, segs)
        inp = PD.make_input(mesh, desc, ids)
        model.sync()
        if prof:
            profiler.enable(mesh, rt=True, ops=True)
            profiler.set_layer(None)
        res = None
        t0 = time.time()
        try:
            nat = PD.ids_from_input(mesh, inp, desc)
            PD.run_chunk(model, desc, nat, sts, layers, last=False, inp=inp)
            model.sync()
            wall = (time.time() - t0) * 1e3
            if prof:
                profiler.signpost("end")
                profiler.finish_rt()
                res = profiler.result()
        finally:
            if prof:
                profiler.disable()
            ttnn.deallocate(inp)
        return wall, res

    return run


def _summary(prof: dict) -> dict:
    """{"sections": {sec: slowest-chip ms}, "ops": {sec: [{op, shape, calls, programs, ms, gap_ms, slot_ms}]},
    "device": slowest chip ms}."""
    per_chip = defaultdict(float)
    for d in prof["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    secs = {k: round(v / 1e6, 3) for k, v in sorted(prof["kernel_ns"].items(), key=lambda kv: -kv[1])}
    ops = {}
    for sec, rows in prof["op_ns"].items():
        agg = {}
        for r in rows:
            k = (r["op"], r["shape"])
            a = agg.setdefault(
                k, {"op": r["op"], "shape": r["shape"], "calls": 0, "programs": 0, "dev": defaultdict(float)}
            )
            a["calls"] += r["calls"]
            a["programs"] += r["programs"]
            for c, v in r["ns_dev"].items():
                a["dev"][c] += v
            for f in ("gap_ns", "slot_ns"):
                a[f] = a.get(f, 0.0) + r.get(f, 0.0)
        out = []
        for a in agg.values():
            dv = a.pop("dev")
            a["ms"] = round(max(dv.values()) / 1e6, 4) if dv else 0.0
            a["ms_min_chip"] = round(min(dv.values()) / 1e6, 4) if dv else 0.0
            a["gap_ms"] = round(a.pop("gap_ns", 0.0) / 1e6, 4)
            a["slot_ms"] = round(a.pop("slot_ns", 0.0) / 1e6, 4)
            out.append(a)
        ops[sec] = sorted(out, key=lambda r: -r["ms"])
    return {"device": round(max(per_chip.values()) / 1e6, 3) if per_chip else 0.0, "sections": secs, "ops": ops}


def _free_states(runtime, kv):
    import ttnn

    for key in [k for k in runtime._states if k[0] == id(kv)]:
        try:
            runtime._states.pop(key).free()
        except Exception as e:  # noqa: BLE001
            print(f"state free: {e}")
    for t in kv.contract.by_config.values():
        if t.is_allocated():
            ttnn.deallocate(t)


def _write(key: str, val) -> None:
    dst = os.environ.get("KR_OUT", "/tmp/pb_kr_probe.json")
    try:
        with open(dst) as f:
            cur = json.load(f)
    except Exception:  # noqa: BLE001
        cur = {}
    cur[key] = val
    with open(dst, "w") as f:
        json.dump(cur, f, indent=1)
    print(f"KR probe: {key} written to {dst}")


@mesh_parametrize(timeout_s=4 * 3600)
def test_kr_attn_ops(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    if "ops" not in _parts():
        pytest.skip("KR_PARTS")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192"
    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode"
    from contextlib import ExitStack

    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP

    p = _knob("KR_POS", POS)
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = -(-deploy // CHUNK) * CHUNK
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(len(LENS)))
    runtime, adapter, hf, params = SP._runtime(mesh_device, layers, CHUNK, want)
    model = runtime.model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    text = PositionGolden(S, (len(LENS) - 1) * SHIFT + p + CHUNK, CHUNK).tokens()
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
    res = {}
    try:
        for path in ("single8192", "4x2k"):
            ctx = ExitStack()
            if path == "single8192":
                ctx.enter_context(_pc.packed("off"))
            else:
                ctx.enter_context(_pc.packed("on"))
                ctx.enter_context(_pc.switch("PACKED_GLUED", "on"))
            with ctx:
                run = _runner(runtime, kv, mesh_device, layers, text, p, path)
                rot = [0]

                def one_run():
                    run(True, rot[0])
                    w = run(False, rot[0])[0]
                    rot[0] += 1
                    return round(w, 1)

                w = warm_up(mesh_device, one_run)
                walls = [one_run() for _ in range(3)]
                run(True, rot[0])
                _, prof = run(False, rot[0], prof=True)
                sm = _summary(prof)
                sm["walls"] = walls
                sm["warm"] = bool(w.get("warm"))
                res[path] = sm
                print(
                    f"KR ops {path} p{p}: walls {walls}, device {sm['device']} ms; sections {json.dumps(sm['sections'])}"
                )
                metrics.record(f"kr_ops_{path}_device_ms_p{p}", sm["device"])
    finally:
        _free_states(runtime, kv)
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    # the attention phase side by side
    a, b = res["single8192"], res["4x2k"]
    for sec in sorted(set(a["sections"]) | set(b["sections"])):
        print(f"KR section {sec}: single {a['sections'].get(sec, 0.0)} 4x2k {b['sections'].get(sec, 0.0)}")
    for sec in sorted(set(a["ops"]) | set(b["ops"])):
        if not sec.startswith("attn") and not sec.startswith("history") and sec != "attention":
            continue
        ra = {(r["op"], r["shape"]): r for r in a["ops"].get(sec, [])}
        rb = {(r["op"], r["shape"]): r for r in b["ops"].get(sec, [])}
        for k in sorted(set(ra) | set(rb), key=lambda k: -(rb.get(k, {}).get("ms", 0) + ra.get(k, {}).get("ms", 0))):
            x, y = ra.get(k, {}), rb.get(k, {})
            print(
                f"KR op [{sec}] {k[0]} {k[1]}: single {x.get('ms', 0.0)} ms ({x.get('calls', 0)} calls, gap "
                f"{x.get('gap_ms', 0.0)}) | 4x2k {y.get('ms', 0.0)} ms ({y.get('calls', 0)} calls, gap {y.get('gap_ms', 0.0)}, "
                f"min chip {y.get('ms_min_chip', 0.0)})"
            )
    _write(f"ops_p{p}", res)


@mesh_parametrize(timeout_s=3600)
def test_kr_score(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    if "score" not in _parts():
        pytest.skip("KR_PARTS")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import owned_gather
    from models.demos.deepseek_v41_flash_d_p.tt.common import SP_AXIS, TP_AXIS
    from models.demos.deepseek_v41_flash_d_p.tt.gather_hide import Split
    from models.demos.deepseek_v41_flash_d_p.tt.indexer import K_CHUNK

    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    mesh = model.mesh
    sp, tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
    n = sp * tp
    ix = next(b.indexer for b in model.blocks.values() if type(getattr(b, "indexer", None)).__name__ == "TtIndexer")
    R, nh, hd, ratio = ix.sq, ix.nh, ix.hd, ix.ratio
    T = 1048576 // ratio
    iters = _knob("KR_ITERS", 3)
    reps = _knob("KR_REPS", 4)
    h = _knob("KR_POS", POS)
    q_dtype = ttnn.bfloat4_b if ix.q_bfp4 else ttnn.bfloat16

    def up(t, dtype, dims):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=dims),
        )

    def zeros_rep(shape, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    k_locals = [up(torch.randn(sp, 1, T // sp, hd), ix.k_dtype, (0, None)) for _ in range(4)]
    kbufs = [zeros_rep([1, 1, T, hd], ix.k_dtype) for _ in range(4)]
    qd = up(torch.randn(n, nh, R, hd).view(sp, tp * nh, R, hd), q_dtype, (0, 1))
    wd = up(torch.randn(n, 1, R, nh).view(sp, tp, R, nh), ttnn.bfloat16, (0, 1))
    cfg32 = ttnn.bringup.IndexerScoreProgramConfig(
        q_chunk_size=32,
        k_chunk_size=K_CHUNK,
        head_group_size=0,
        max_grid_x=ix.score_cfg.max_grid_x,
        max_grid_y=ix.score_cfg.max_grid_y,
    )
    common = dict(
        cluster_axis=SP_AXIS,
        topology=ix.topology,
        num_links=ix.num_links,
        key_compression_ratio=ratio,
        compute_kernel_config=ix.score_ckc,
        seq_subshard_axis=TP_AXIS if tp > 1 else None,
        block_cyclic_sp_axis=SP_AXIS,
        block_cyclic_chunk_local=ix.key_rows,
        natural_query_blocks=True,
        natural_query_rows=0 if ix.key_rows == ix.w_rows else ix.w_rows,
    )

    def timed(fn, free=True) -> float:
        r = fn()
        if free and r is not None:
            ttnn.deallocate(r)
        ts = []
        for _ in range(iters):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(reps):
                r = fn()
                if free and r is not None:
                    ttnn.deallocate(r)
            ttnn.synchronize_device(mesh)
            ts.append((time.perf_counter() - t0) * 1e3 / reps)
        return statistics.median(ts)

    start = min(h, T * ratio - 8192)
    kv = min(T, -(-((start + model.chunk) // ratio) // 32) * 32)
    chips_per = n // 4
    words = []
    for q in range(n):
        j = q // chips_per
        words += [1, 0, R, j, start + (q % chips_per) * R, kv] + [0] * (RANGE_STRIDE - 6)

    def ranges():
        return ttnn.bringup.ring_indexer_score_dsa(
            qd,
            kbufs[0],
            wd,
            k_locals[0],
            ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            chunk_start_idx=0,
            program_config=cfg32,
            kv_len=kv,
            k_extra=kbufs[1:4],
            k_local_extra=k_locals[1:4],
            lane_ranges=words,
            **common,
        )

    def single():
        return ttnn.bringup.ring_indexer_score_dsa(
            qd,
            kbufs[0],
            wd,
            k_locals[0],
            ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            chunk_start_idx=start,
            program_config=ix.score_cfg,
            kv_len=kv,
            **common,
        )

    def local(cfg=cfg32):
        return ttnn.bringup.indexer_score_dsa(
            qd,
            kbufs[0],
            wd,
            chunk_start_idx=start + 7 * R,
            key_compression_ratio=ratio,
            program_config=cfg,
            compute_kernel_config=ix.score_ckc,
            kv_len=kv,
        )

    res = {}
    for name, fn in (
        ("ranges", ranges),
        ("single", single),
        ("local", local),
        ("local_qmodel", lambda: local(ix.score_cfg)),
    ):
        try:
            ms = timed(fn)
        except Exception as e:  # noqa: BLE001
            print(f"KR score {name}: FAILED {e}")
            ms = -1.0
        res[name] = round(ms, 3)
        metrics.record(f"kr_score_{name}_ms_p{h}", round(ms, 3))
        print(f"KR score {name} p{h}: {ms:.3f} ms a call (kv_len {kv}, T {T})")
    # the owned send of one source's keys (pages = bfp8 tiles) to every segment's chips, column groups
    try:
        page = int(ttnn.tile_size(ix.k_dtype))
    except Exception:  # noqa: BLE001
        page = 1088
    Rp = (T // sp) // 32 * (hd // 32)  # pages an owner (SP rank) of a key buffer
    hp = (kv // sp) // 32 * (hd // 32)  # pages an owner holds below the chunk at this history (upper bound)
    snd = owned_gather.sender(mesh)
    dest = [q // chips_per for q in range(n)]
    out = zeros_rep([1, 1, T, hd], ix.k_dtype)
    split = Split.of(mesh) if hasattr(Split, "of") else None

    def send():
        entries = [(k_locals[j], 0, hp) for j in range(4)]
        snd(out, Rp, 1, entries, dest, page=page)

    def send_on_column():
        mesh.load_sub_device_manager(split.mgr)
        try:
            send()
        finally:
            mesh.clear_loaded_sub_device_manager()

    for name, fn in (("send_whole", send), ("send_column", send_on_column if split is not None else None)):
        if fn is None:
            continue
        try:
            ms = timed(lambda fn=fn: fn(), free=False)
        except Exception as e:  # noqa: BLE001
            print(f"KR score {name}: FAILED {e}")
            ms = -1.0
        res[name] = round(ms, 3)
        metrics.record(f"kr_{name}_ms_p{h}", round(ms, 3))
        print(f"KR {name} p{h}: {ms:.3f} ms a call ({hp} pages of {page} B an owner, {sp - 1} owners a chip received)")
    # check the send (64 rows at each end of every owner's block): every chip of segment j holds k_locals[j]'s rows of
    # every owner at owner-major rows
    try:
        comp = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(sp, tp), dims=(0, 1))
        rows = hp // (hd // 32) * 32
        bad = 0
        for a0 in (0, rows - 64):
            refs = []
            for j in range(4):
                sl = ttnn.slice(k_locals[j], [0, 0, a0, 0], [1, 1, a0 + 64, hd])
                refs.append(ttnn.to_torch(sl, mesh_composer=comp))
                ttnn.deallocate(sl)
            for o in range(sp):
                b0 = o * (T // sp) + a0
                sl = ttnn.slice(out, [0, 0, b0, 0], [1, 1, b0 + 64, hd])
                got = ttnn.to_torch(sl, mesh_composer=comp)
                ttnn.deallocate(sl)
                for q in range(n):
                    r_, c_ = divmod(q, tp)
                    bad += int(not torch.equal(got[r_, c_].float(), refs[dest[q]][o, c_].float()))
        res["send_bad_blocks"] = bad
        print(f"KR send check: {bad} (chip, owner, end) blocks differ")
    except Exception as e:  # noqa: BLE001
        print(f"KR send check failed: {e}")
    # the local scorer itself (ring_indexer_score_dsa keys_gathered on the owned-sent keys): bit-identical scores to the
    # ranges ring, and its time
    if "keys_gathered" in (getattr(ttnn.bringup.ring_indexer_score_dsa, "__doc__", "") or ""):
        send_all = [(k_locals[j], 0, Rp) for j in range(4)]
        snd(out, Rp, 1, send_all, dest, page=page)
        words0 = []
        for q in range(n):
            words0 += [1, 0, R, 0, start + (q % chips_per) * R, kv] + [0] * (RANGE_STRIDE - 6)

        def owned():
            return ttnn.bringup.ring_indexer_score_dsa(
                qd,
                out,
                wd,
                k_locals[0],
                ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
                chunk_start_idx=0,
                program_config=cfg32,
                kv_len=kv,
                lane_ranges=words0,
                keys_gathered=True,
                **common,
            )

        a_ = ranges()
        b_ = owned()
        worst = {}
        for nm, (c0, c1) in (("valid", (0, kv)), ("past", (kv, T))):
            if c1 <= c0:
                continue
            sa = ttnn.slice(a_, [0, 0, 0, c0], [1, 1, R, c1])
            sb = ttnn.slice(b_, [0, 0, 0, c0], [1, 1, R, c1])
            ta, tb_ = ttnn.to_layout(sa, ttnn.TILE_LAYOUT), ttnn.to_layout(sb, ttnn.TILE_LAYOUT)
            ne = ttnn.ne(ta, tb_)
            mx = ttnn.max(ne)
            worst[nm] = max(float(ttnn.to_torch(t).float().max()) for t in ttnn.get_device_tensors(mx))
            for t in (sa, sb, ta, tb_, ne, mx):
                ttnn.deallocate(t)
        print(f"KR owned vs ranges any-diff by column span: {worst}")
        worst = worst.get("valid", -1.0)
        cols = [(0, 2048), (kv - 2048, kv)]
        host_bad = 0
        for c0, c1 in cols:
            sa = ttnn.slice(a_, [0, 0, 0, c0], [1, 1, 64, c1])
            sb = ttnn.slice(b_, [0, 0, 0, c0], [1, 1, 64, c1])
            xa = ttnn.to_torch(sa, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            xb = ttnn.to_torch(sb, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            host_bad += int((xa.view(torch.int16) != xb.view(torch.int16)).sum())
            ttnn.deallocate(sa)
            ttnn.deallocate(sb)
        ttnn.deallocate(a_)
        ttnn.deallocate(b_)
        res["owned_vs_ranges_any_diff"] = worst
        res["owned_vs_ranges_host_bad"] = host_bad
        metrics.record(f"kr_owned_vs_ranges_any_diff_p{h}", worst)
        print(f"KR owned scorer vs ranges ring: device any-diff {worst}, host slices differing {host_bad}")
        ms = timed(owned)
        res["owned"] = round(ms, 3)
        metrics.record(f"kr_score_owned_ms_p{h}", round(ms, 3))
        print(f"KR score owned (keys_gathered) p{h}: {ms:.3f} ms a call")
    for t in k_locals + kbufs + [qd, wd, out]:
        ttnn.deallocate(t)
    _write(f"score_p{h}", res)
