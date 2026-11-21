# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.2 (OWNER 2026-11-20, rule 20: serving never rebuilds): PERMANENT check that the single-request serving path
builds no program after its warm-up and uploads the chunk's ids only once (the engine's own H2D).

The serving runtime exactly as prefill_runner builds it (adapter.build_runtime over the gate's executed layers 0-20,
DECODER_SKIP=decode; through the model's one load path, adapter.load_model: in a batch the held model), the engine's
KV cache (adapter.allocate_kv_cache, NO_REBUILD_SLOTS slots at the deployment context), ``runtime.compile`` (the
engine's warm-up at startup) and NOTHING else before the count. Then fresh chunks, every one at positions, slots and
ends never seen before (AgentX history draws, seeds fixed), each request's real previous chunk first:

    aligned    a chunk at a chunk-aligned start
    steady     a chunk at a 64-aligned start (two chunk periods)
    resumed    a resumed request: global_write_start = start + 128 (the encoder replay)
    partial    a request's last chunk, end = start + m (m from AgentX new tokens), its cap below end
    pulled     a pulled-back last chunk: start = ceil32(end - chunk) below the previous chunk's end, global_write_start
               = that end (the replay offset a per-request value)
    acks_only  a last chunk with no position below its cap (only the snapshots at the cap and the acks)

each through ``runtime.prefill_chunk`` with the engine's payload made before the count (C.payload_tensor). Metrics
(gated): program_builds_after_warmup_serving (program-cache entries added + ttnn.bringup.prepare_program calls in the
fresh chunks; == 0) and ids_uploads_per_chunk (host -> device writes of a chunk's worth of integer ids inside
prefill_chunk, a chunk; == 0 under POS_DYNAMIC=on: the ids are reordered on the device). On a build, the BuildProbe
(tests/fresh_common.py) names the op, its call site and the argument that changed. Also (recorded, asserted): the
pulled-back chunk run again in two fresh slots with POS_DYNAMIC off and on gives bit-identical final streams
(no_rebuild_on_off_max_abs_diff == 0).

PB.9C (OWNER 2026-11-21: the no-rebuild check per path, PASS / FAIL only): at BRINGUP_CHUNK=8192 with PACKED=on it
also checks the packed paths through ``runtime.prefill_packed``, every chunk at fresh slots (a pool of PACKED_SLOTS
slots no single draw and no warm-up used; each draw's slots a new subset in a new order) and fresh histories (AgentX
draws, 64-aligned), each request's real previous chunk first, kinds steady / resumed (write_start = start + 128) /
steady (PB.9G: chip-aligned lanes, every chunk through the one-call path; a resumed segment of fewer than 192 tokens
stays steady):
    lanes4x2k     four 2048-token segments (8 chips each)
    lanes4x1600s  four 1600-token segments (7 chips each) + a 1024-row split piece (4 chips)
    lanes6mix     segments of 200 (1 chip), 1 (1 token, its chip idle), 2048, 777, 40 (cap <= start: acks only), 1500
    one4k         PB.FX (DS41-373): ONE 4096-token segment (a lone segment runs today's single path in prefill_packed)
                  at a continuing, non-chunk-aligned start after a partial chunk of its slot: the window carry read
                  from the slot's ring (attention._ring_carry_pages)
    one1600       the same, 1600 tokens (a partial chunk end)
    lanes2x3k     the sibling kind: two 3072-token segments (12 chips each) at such starts
Metrics: no_rebuild_pass_<path> (1 / 0; also single<chunk> for the single-request check above) and
program_builds_after_warmup_<path> (program-cache entries + prepare_program calls in the path's fresh chunks),
packed_ids_uploads_<path>, no_rebuild_rows_<path>, no_rebuild_probe_<path> (the ops that built).

PB.NR (OWNER 2026-11-21): asserted always: the single-request path builds nothing (no_rebuild_pass_single<chunk> == 1).
The packed paths (PACKED_PATHS) are RECORDED (PASS / FAIL, their build count and the probe's ops, and
packed_fallback_chunks_<path>: chunks that did not run the one call, 0 by construction since PB.9G), and asserted only
with NO_REBUILD_PACKED_STRICT=1 (then every packed path must build nothing); a packed path that raises on a fresh chunk
is a recorded FAIL unless strict (a hang always raises).

WHY COLD (F297, BATCH_COLD = True): this test counts program-cache entries and prepare_program calls after the
engine's own warm-up (runtime.compile). Its draws are fixed-seed, so in a held process (the resident runner, a batch
after test_skip_positions / walls, or this test's own earlier run) every program its fresh chunks need may already be
cached by EARLIER traffic, and the count reads 0 although serving, which only gets runtime.compile, would build:
PB.9C saw 153 / 16 builds in a fresh process and 0 in the held one on the same code. So the batch runs it in a fresh
process (a cold program cache: only runtime.compile before the count), first after the determinism check.

    PYTHONPATH=$PWD DS41_ENGRAM_TABLE=resident DS41_POS_DYNAMIC=on DS41_DECODER_SKIP=decode \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_no_rebuild.py
"""

from __future__ import annotations

import functools
import os
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import cache_entries
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29: takes the model of a bring-up batch process (core/batch.shareable)
BATCH_LAYERS = "executed"  # F285: the layers a decode chunk executes (0-20), as test_skip_positions
# F279 / F283: the chunk this run measures at (spec target.chunk, F271's BRINGUP_CHUNK when set). Declared per-run so
# the gate's unit (no BRINGUP_CHUNK: the spec's 5120) and test_skip_positions' (BRINGUP_CHUNK=5120) share the one
# executed-layers load; a run at another chunk still gets its own model (testing/batch.shared_model compares the
# load's chunk: a fresh process, or a failure under one_load), never another chunk's model.
# PB.NR: NO_REBUILD_PACKED_STRICT=1 asserts the packed paths too (default: recorded only); a per-run test switch
BATCH_ITEM_VARS = ("BRINGUP_CHUNK", "NO_REBUILD_PACKED_STRICT")
# F297 (PB.NR): a build counter needs a cold program cache (module docstring, WHY COLD): a fresh process, first in its
# batch after the determinism check
BATCH_COLD = True
REPLAY = 128
SLOTS = 11  # (slot 0: compile's; 1-8: the fresh draws; 9, 10: the on / off comparison, never written before it)
KINDS = ("aligned", "steady", "resumed", "partial", "pulled", "acks_only", "steady", "partial")
# PB.9C: the packed paths' pool (slots SLOTS .. SLOTS + PACKED_SLOTS - 1: compile's glued warm-up takes 0..5, the single
# draws 1..8, the on / off comparison 9, 10)
PACKED_SLOTS = 9
# PB.9G (chip-aligned lanes, PACKED_CHIPS=on): one path a layout; every chunk runs the one-call glued path
PACKED_PATHS = (
    ("lanes4x2k", (2048,) * 4),  # four 2048-token requests, 8 chips each
    ("lanes4x1600s", (1600,) * 4 + (1024,)),  # four 1600 (7 chips each) + a 1024-row split piece (4 chips)
    ("lanes6mix", (200, 1, 2048, 777, 40, 1500)),  # 1-chip, 1-token, idle (cap <= start) and padded segments
    # PB.FX (DS41-373): one segment at a continuing, non-chunk-aligned start after a partial chunk of its slot (today's
    # single path inside prefill_packed, its window carry from the slot's ring), whole and partial; and the sibling
    # two-segment kind at such starts
    ("one4k", (4096,)),
    ("one1600", (1600,)),
    ("lanes2x3k", (3072, 3072)),
)
PACKED_KINDS = ("steady", "resumed", "steady")


class Uploads:
    """Host -> device writes of a chunk's worth of integer ids (ttnn.from_torch of an int torch tensor of ``chunk``
    values; copy_host_to_device_tensor of an int32 / uint32 host tensor of that volume) while installed."""

    def __init__(self, chunk: int):
        self.chunk, self.n, self.undo = int(chunk), 0, []

    def _ids(self, t) -> bool:
        if isinstance(t, torch.Tensor):
            return not t.is_floating_point() and t.numel() == self.chunk
        import ttnn

        try:
            return t.dtype in (ttnn.uint32, ttnn.int32) and int(t.logical_volume()) == self.chunk
        except Exception:  # noqa: BLE001
            return False

    def __enter__(self):
        import ttnn

        me = self
        for name in ("from_torch", "copy_host_to_device_tensor"):
            f = getattr(ttnn, name)

            def g(*a, _f=f, **k):
                if a and me._ids(a[0]):
                    me.n += 1
                return _f(*a, **k)

            functools.update_wrapper(g, f)
            setattr(ttnn, name, g)
            self.undo.append((name, f))
        return self

    def __exit__(self, *exc):
        import ttnn

        for name, f in reversed(self.undo):
            setattr(ttnn, name, f)
        self.undo.clear()
        return False


def _draws(chunk: int, hi: int) -> list:
    """The fresh chunks (module docstring): [(kind, slot, [(start, end, write_start), ...] the request's chunks)]."""
    from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as F

    out = []
    for n, kind in enumerate(KINDS):
        d = F.Draw(424200 + n)
        p = d.history(lo=chunk, hi=hi)
        slot = 1 + n % (SLOTS - 3)  # (slot 0: compile's; every slot new to this test until it wraps)
        if kind == "aligned":
            p = max(chunk, p // chunk * chunk)
        if kind in ("aligned", "steady"):
            chunks = [(p - chunk, p, None), (p, p + chunk, None)]
        elif kind == "resumed":
            chunks = [(p - chunk, p, None), (p, p + chunk, p + REPLAY)]
        elif kind == "partial":
            m = max(64, (d.new_tokens() - 1) % chunk + 1)
            chunks = [(p - chunk, p, None), (p, p + m, None)]
        elif kind == "pulled":
            m = max(64, (d.new_tokens() - 1) % chunk + 1)
            if m % 32 == 0:
                m += 7  # (an end off the 32 grid: a start strictly inside the previous chunk)
            e = p + m
            s = -(-(e - chunk) // 32) * 32
            chunks = [(p - chunk, p, None), (s, e, p)]
        else:  # acks_only: no position below the cap (cap = end // 64 x 64 = start)
            p = p // 64 * 64
            chunks = [(p - chunk, p, None), (p, p + 20, None)]
        out.append((kind, slot, chunks))
    return out


def _payload(mesh, text, s: int, e: int, chunk: int, sp: int):
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

    return C.payload_tensor(mesh, R.server_payload(text[s:e].tolist(), s, e, chunk=chunk, sp=sp))


def _final_streams(runtime, inp, kv, slot: int, s: int, e: int, ws) -> torch.Tensor:
    """prefill_chunk with the last executed layer's output streams kept (a device clone a layer, the previous freed)
    -> host, every chip's rows stacked."""
    import ttnn

    model = runtime.model
    keep = {}
    layer = model.layer

    def wrapped(i, h, *a, **k):
        out = layer(i, h, *a, **k)
        if getattr(out, "streams", None) is not None:
            old = keep.pop("t", None)
            if old is not None:
                ttnn.deallocate(old)
            keep["t"] = ttnn.clone(out.streams, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    model.layer = wrapped
    try:
        runtime.prefill_chunk(inp, kv, slot_id=slot, actual_start=s, actual_end=e, global_write_start=ws)
    finally:
        del model.layer
    t = keep["t"]
    host = torch.stack([ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)])
    ttnn.deallocate(t)
    return host


@mesh_parametrize(timeout_s=3600)
def test_no_rebuild(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as F
    from models.demos.deepseek_v41_flash_d_p.tests.test_skip_positions import _runtime

    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode (the gate's serving mode)"
    # PB.8R (OV-61, F283): the shipped ENGRAM_TABLE as test_skip_positions sets it, so the gate's perf units share one
    # load (unset, a test process reads F222's test value mmap and the one-load rule refused this unit)
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    print(f"ENGRAM_TABLE={settings.get('ENGRAM_TABLE')}")
    pos = settings.get("POS_DYNAMIC")
    print(f"POS_DYNAMIC={pos}")
    chunk = int(S.get("target.chunk"))
    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = max(-(-deploy // chunk) * chunk, int(S.data["target"]["seq"]))  # (test_skip_positions' fresh load)
    monkeypatch.setitem(S.data["target"], "seq", cap)
    packed_check = chunk == 8192 and settings.get("PACKED") == "on"  # PB.9C (module docstring)
    strict = os.environ.get("NO_REBUILD_PACKED_STRICT", "0") == "1"  # PB.NR: assert the packed paths too
    print(f"NO_REBUILD_PACKED_STRICT={int(strict)}")
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(SLOTS + (PACKED_SLOTS if packed_check else 0)))
    t0 = time.time()
    runtime, adapter, hf, params = _runtime(mesh_device, layers, chunk, cap)
    model = runtime.model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    print(f"model ready in {time.time() - t0:.1f} s")
    sp = int(tuple(mesh_device.shape)[0])
    hi = min(cap, model.max_seq) - chunk
    text = torch.randint(0, int(model.vocab), (hi + chunk,), generator=torch.Generator().manual_seed(7))
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
    try:
        t0 = time.time()
        runtime.compile(kv)  # (the engine's warm-up at startup: the only warm-up before the count)
        model.sync()
        print(f"runtime.compile in {time.time() - t0:.1f} s")
        draws = _draws(chunk, hi)
        for _, slot, _ in draws:  # (a serving process holds every slot's state after its first request: untimed)
            runtime.state(kv, slot)
        uploads = Uploads(chunk)
        pc = F.PrepCount()
        n_chunks, rows = 0, []
        e0 = cache_entries(mesh_device)
        with F.BuildProbe(mesh_device) as pb:
            for kind, slot, chunks in draws:
                b0, p0, u0 = cache_entries(mesh_device), pc.n, uploads.n
                for s, e, ws in chunks:
                    inp = _payload(mesh_device, text, s, e, chunk, sp)  # (the engine's H2D: before the count)
                    with pc, uploads:
                        runtime.prefill_chunk(
                            inp, kv, slot_id=slot, actual_start=s, actual_end=e, global_write_start=ws
                        )
                    model.sync()
                    ttnn.deallocate(inp)
                    n_chunks += 1
                row = {
                    "kind": kind,
                    "slot": slot,
                    "chunks": chunks,
                    "builds": cache_entries(mesh_device) - b0,
                    "prepared": pc.n - p0,
                    "ids_uploads": uploads.n - u0,
                }
                rows.append(row)
                print(f"NO_REBUILD {row}", flush=True)
        built = cache_entries(mesh_device) - e0
        pb.print("no_rebuild")
        builds = built + pc.n
        F._record_gate_name("program_builds_after_warmup_serving", int(builds))
        F._record_gate_name(f"no_rebuild_pass_single{chunk}", int(builds == 0))
        F._record_gate_name(f"program_builds_after_warmup_single{chunk}", int(builds))
        F._record_gate_name("ids_uploads_per_chunk", round(uploads.n / max(1, n_chunks), 3))
        metrics.record("no_rebuild_cache_entries", int(built))
        metrics.record("no_rebuild_prepares", int(pc.n))
        metrics.record("no_rebuild_chunks", int(n_chunks))
        metrics.record("no_rebuild_rows", rows)
        metrics.record("no_rebuild_probe", pb.summary(20))

        packed = {}
        if packed_check:  # PB.9C / PB.9G: the packed paths (one a layout)
            for path, lens in PACKED_PATHS:
                try:
                    packed[path] = _packed_path(runtime, kv, mesh_device, text, path, lens, hi)
                except Exception as e:  # noqa: BLE001 (a path that fails on fresh chunks: recorded FAIL, not hidden)
                    if F.is_hang(e) or strict:
                        raise
                    msg = f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"[:300]
                    print(f"NO_REBUILD {path}: FAIL (error on a fresh chunk: {msg})", flush=True)
                    F._record_gate_name(f"no_rebuild_pass_{path}", 0)
                    metrics.record(f"no_rebuild_error_{path}", msg)
                    packed[path] = {"builds": -1, "pass": 0, "error": msg}

        # bit-identity of the two paths on the pulled-back chunk: two slots nothing wrote before (the same zero history
        # below the request's chunks), its previous chunk first
        kind, _, chunks = next(d for d in draws if d[0] == "pulled")
        got = {}
        snap = settings.snap
        for side, slot in (("off", SLOTS - 1), ("on", SLOTS - 2)):
            if snap is not None:
                snap["POS_DYNAMIC"] = side
            else:
                monkeypatch.setenv(settings.env_name("POS_DYNAMIC"), side)
            try:
                for j, (s, e, ws) in enumerate(chunks):
                    inp = _payload(mesh_device, text, s, e, chunk, sp)
                    if j + 1 < len(chunks):
                        runtime.prefill_chunk(
                            inp, kv, slot_id=slot, actual_start=s, actual_end=e, global_write_start=ws
                        )
                    else:
                        got[side] = _final_streams(runtime, inp, kv, slot, s, e, ws)
                    ttnn.deallocate(inp)
            finally:
                if snap is not None:
                    snap["POS_DYNAMIC"] = pos
        diff = float((got["on"] - got["off"]).abs().max())
        metrics.record("no_rebuild_on_off_max_abs_diff", diff)
        print(f"NO_REBUILD on vs off (pulled-back chunk {chunks[-1]}): max |diff| {diff}", flush=True)
    finally:
        for key in [k for k in runtime._states if k[0] == id(kv)]:
            try:
                runtime._states.pop(key).free()
            except Exception as e:  # noqa: BLE001
                print(f"state free: {e}")
        for t in kv.contract.by_config.values():
            if t.is_allocated():
                ttnn.deallocate(t)
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    for path, v in packed.items():  # PB.NR: the per-path verdict, recorded above; asserted only when strict
        print(f"NO_REBUILD VERDICT {path}: {'PASS' if v['pass'] else 'FAIL'} ({v['builds']} builds)", flush=True)
    print(f"NO_REBUILD VERDICT single{chunk}: {'PASS' if builds == 0 else 'FAIL'} ({builds} builds)", flush=True)
    assert builds == 0, f"{builds} programs built after warm-up on fresh serving chunks: {pb.summary(8)}"
    if strict:
        bad = {k: v for k, v in packed.items() if v["builds"] != 0}
        assert not bad, f"packed paths built programs after warm-up on fresh chunks: {bad}"
    if pos == "on":
        assert uploads.n == 0, f"{uploads.n} ids uploads in {n_chunks} chunks (POS_DYNAMIC=on reorders on the device)"
    assert diff == 0, f"POS_DYNAMIC on vs off differ: max |diff| {diff}"


def _packed_path(runtime, kv, mesh, text, path: str, lens, hi: int) -> dict:
    """PB.9C: one packed path's fresh chunks through runtime.prefill_packed (module docstring) -> {"builds", "pass",
    "rows"}; records no_rebuild_pass_<path>, program_builds_after_warmup_<path>, packed_ids_uploads_<path>."""
    import random

    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as F
    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import switch
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as PD
    from models.demos.deepseek_v41_flash_d_p.tt.packed import GluedChunk, Segment

    model = runtime.model
    chunk = int(model.chunk)
    pool = list(range(SLOTS, SLOTS + PACKED_SLOTS))
    n = len(lens)
    draws = []
    for j, kind in enumerate(PACKED_KINDS):
        d = F.Draw(525200 + 100 * j + 10 * n + sum(lens) % 7)
        rnd = random.Random(d.seed)
        slots = rnd.sample(pool, n)  # (a new subset of the pool in a new order every draw)
        hs = [d.history(lo=max(lens), hi=hi - max(lens)) for _ in range(n)]
        draws.append((kind, slots, hs))
    uploads = Uploads(chunk)
    pc = F.PrepCount()
    rows = []
    PD.STATS["fallback"] = 0
    with switch("PACKED_GLUED", "on"):
        e0, p0, u0, nch = cache_entries(mesh), pc.n, uploads.n, 0
        with F.BuildProbe(mesh) as pb:
            for kind, slots, hs in draws:
                b0, q0, v0 = cache_entries(mesh), pc.n, uploads.n
                for prev in (True, False):
                    segs, ids = [], []
                    for k in range(n):
                        m = -(-lens[k] // 64) * 64 if prev else lens[k]  # (the previous chunk: whole kv blocks)
                        s = hs[k] - (m if prev else 0)
                        ws = s + REPLAY if (kind == "resumed" and not prev and lens[k] >= REPLAY + 64) else None
                        segs.append(Segment(slots[k], s, s + m, ws))
                        ids.append(text[s : s + m].long())
                    desc = GluedChunk(chunk, tuple(segs))
                    inp = runtime.make_packed_input(desc, ids)  # (the engine's H2D: before the count)
                    with pc, uploads:
                        runtime.prefill_packed(inp, kv, desc)
                    model.sync()
                    ttnn.deallocate(inp)
                    nch += 1
                row = {
                    "kind": kind,
                    "slots": slots,
                    "histories": hs,
                    "builds": cache_entries(mesh) - b0,
                    "prepared": pc.n - q0,
                    "ids_uploads": uploads.n - v0,
                }
                rows.append(row)
                print(f"NO_REBUILD {path} (PACKED_CHIPS={settings.get('PACKED_CHIPS')}) {row}", flush=True)
        pb.print(f"no_rebuild {path}")
        builds = (cache_entries(mesh) - e0) + (pc.n - p0)
    ok = int(builds == 0)
    F._record_gate_name(f"no_rebuild_pass_{path}", ok)
    F._record_gate_name(f"program_builds_after_warmup_{path}", int(builds))
    metrics.record(f"packed_ids_uploads_{path}", round((uploads.n - u0) / max(1, nch), 3))
    F._record_gate_name(f"packed_fallback_chunks_{path}", int(PD.STATS["fallback"]))
    metrics.record(f"no_rebuild_rows_{path}", rows)
    metrics.record(f"no_rebuild_probe_{path}", pb.summary(20))
    print(
        f"NO_REBUILD {path}: {'PASS' if ok else 'FAIL'} ({builds} builds after warm-up in {nch} fresh chunks; "
        f"PACKED_CHIPS={settings.get('PACKED_CHIPS')}); probe {pb.summary(8)}",
        flush=True,
    )
    return {"builds": int(builds), "pass": ok, "rows": rows}
