# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mistral-Large-4-1T-A52B on a Blackhole Galaxy: the device model over a contiguous run of layers (all 72 by
default), end to end against the CPU reference golden (cpu_golden.py; no GPU golden of this model exists).

One prefill of the golden's tokens (its seq in chunks of its chunk) through the device model: the dumped layers'
outputs and the final norm of every chunk, and the last chunk's last rows of logits (PCC, top-1). A second, warm
prefill with no read-back is timed per chunk. Results go to ML4_E2E_OUT (JSON).

    ML4_E2E_GALAXY=1 ML4_GOLDEN=<cpu_golden.py --out dir> \
    python -m pytest models/demos/deepseek_v3_d_p/tests/mistral_large_4/test_e2e_galaxy.py

Skipped unless ML4_E2E_GALAXY=1: it needs a Blackhole Galaxy, the checkpoint and a golden. It opens the 8x4 mesh
itself (FABRIC_1D_RING), so it needs no device fixture. The layers are the golden's; ML4_E2E_WARM=0 skips the timing.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import statistics
import time
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import sub_device
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.hooks import device_model
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mistral_large_4 import CHECKPOINT_DIR

CKPT = os.environ.get("ML4_CKPT") or os.environ.get("PREFILL_HF_MODEL") or CHECKPOINT_DIR
GOLDEN = Path(os.environ.get("ML4_GOLDEN", ""))
OUT = Path(os.environ.get("ML4_E2E_OUT", "generated/ml4_e2e/result.json"))
PCC_TARGET, PCC_LOWER = 0.99, 0.85
CAPACITY = 56320  # the served slot capacity: the model is built as it serves, whatever the golden's length


class _Spec:
    """What the device model's builder (tt/mistral_large_4/hooks.py) reads off a spec."""

    def __init__(self, hf_dir: str, seq: int, chunk: int):
        self.hf_dir = Path(hf_dir)
        self._v = {"target.chunk": chunk, "target.seq": seq, "reference.weight_dtype": "bfloat16"}

    def get(self, key, default=None):
        return self._v.get(key, default)


REPLAY_ISSUE_MS: list = []  # host enqueue ms of every traced replay


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _top1_detail(lg: torch.Tensor, g: torch.Tensor) -> dict:
    """How far a last-position top-1 flip is. Each side's rank of the other's top-1 (0 = its own top-1), the
    logit gap to it on each side, and the top-5 overlap, for the last row; top-5 containment over all rows."""
    a, b = lg[-1].double(), g[-1].double()
    ta, tb = int(a.argmax()), int(b.argmax())
    gold_in_top5 = (lg.topk(5, dim=-1).indices == g.argmax(-1, keepdim=True)).any(-1)
    return {
        "golden_rank_in_ours": int((a > a[tb]).sum()),
        "our_rank_in_golden": int((b > b[ta]).sum()),
        "our_gap": round(float(a[ta] - a[tb]), 4),  # our logit of our top-1 minus ours of the golden's
        "golden_gap": round(float(b[tb] - b[ta]), 4),  # the golden's margin over our top-1
        "last_top5_overlap": len(set(a.topk(5).indices.tolist()) & set(b.topk(5).indices.tolist())),
        "golden_top1_in_our_top5_rows": int(gold_in_top5.sum()),
    }


def _cmp(dev: torch.Tensor, gold: torch.Tensor) -> dict:
    return {"pcc": round(_pcc(dev, gold), 6), "rel": round(_rel(dev, gold), 6), "finite": bool(dev.isfinite().all())}


def _meta() -> dict:
    """The golden's settings: its meta.json, or, while cpu_golden.py is still writing it (the device model loads in the
    meantime), the same settings from the environment: ML4_SEQ, ML4_LAYERS ("0-71"), ML4_DUMP ("0,35,71"),
    ML4_PROMPT."""
    p = GOLDEN / "meta.json"
    if p.exists():
        return json.loads(p.read_text())
    a, _, b = os.environ.get("ML4_LAYERS", "0-71").partition("-")
    return {
        "seq": int(os.environ["ML4_SEQ"]),
        "chunk": 5120,
        "dump": [int(x) for x in os.environ["ML4_DUMP"].split(",")],
        "logits_rows": 32,
        "layers": [int(a), int(b or a)],
        "prompt": os.environ["ML4_PROMPT"],
    }


class _Gold:
    """chunk_<c>.safetensors of the golden, read when first compared; a chunk cpu_golden.py has not finished (the next
    chunk's file or meta.json not there yet) is waited for, up to ML4_GOLDEN_WAIT_S."""

    def __init__(self, chunks: int, load):
        self.chunks, self.load, self.got = chunks, load, {}

    def __getitem__(self, c: int) -> dict:
        if c not in self.got:
            done = GOLDEN / ("meta.json" if c == self.chunks - 1 else f"chunk_{c + 1:02d}.safetensors")
            t0 = time.time()
            while not done.exists():
                assert time.time() - t0 < float(os.environ.get("ML4_GOLDEN_WAIT_S", "36000")), f"golden chunk {c}"
                time.sleep(30)
            self.got[c] = self.load(str(GOLDEN / f"chunk_{c:02d}.safetensors"))
        return self.got[c]


@pytest.fixture
def mesh_device():
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import fabric

    fabric.set_from_env()  # ML4_FABRIC / ML4_FABRIC_PAYLOAD; default 1D ring
    # ML4_TRACE_REGION_MB reserves a trace region; 0 (default) = traces allocated from DRAM as captured
    region = int(os.environ.get("ML4_TRACE_REGION_MB", "0")) * 1024 * 1024
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), **({"trace_region_size": region} if region else {}))
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


_LAST_STATE = None
KV_LAYERS = (0, 1, 3, 35, 39, 71)


def _kv_vs_golden(model, state, seq: int, layers) -> dict | None:
    """The served caches after a prefill against the golden's state (cpu_golden.py --state): each owned cache of
    KV_LAYERS, index keys rotated as the cache holds them, and layer 39's copy of layer 35's KV (KV_MIRRORS)."""
    from safetensors.torch import load_file

    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.indexer import hadamard_matrix
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

    d = GOLDEN / "kv_cache"
    if not d.exists():
        return None
    out = {}
    for i in (j for j in KV_LAYERS if j in layers):
        g = load_file(str(d / f"layer_{i}.safetensors"))
        got = state.to_torch(i, seq)
        for n in ("kv_latent", "k_rope", "index_key"):
            gt = g.get(f"{n}_cache_layer_{i}")
            if gt is None or not gt.shape[-1] or not got[n].shape[-1]:
                continue
            gt = gt[:seq].float()
            if n == "index_key" and settings.get("INDEXER_HADAMARD"):
                gt = gt @ hadamard_matrix(gt.shape[-1])
            out[f"L{i}.{n}"] = _cmp(got[n].float(), gt)
    m = model
    if 39 in m.kv_region and 35 in layers and not m.cfg.owns_kv(39):
        kv = state._read(m.kvpe, len(m.kv_region), m.kv_region[39], seq)
        g = load_file(str(d / "layer_35.safetensors"))
        out["L39.mirror_of_L35.kv_latent"] = _cmp(
            kv[:, : m.cfg.kv_lora].float(), g["kv_latent_cache_layer_35"][:seq].float()
        )
    return out


def _prefill(model, tokens, seq: int, chunk: int, layers, watch=None):
    """One prefill of tokens[:seq]; watch(c, name, host tensor) gets the dumped layers' outputs, the final norm and
    the last chunk's final norm device tensor (name "final_dev"). Returns the per-chunk seconds."""
    global _LAST_STATE
    state = _LAST_STATE = model.new_state(seq)
    times = []
    for c in range(seq // chunk):
        start = c * chunk
        t0 = time.time()
        h = model.embed(tokens[start : start + chunk], start=start)
        for i in layers:
            t1 = time.time()
            out = model.layer(i, h, start, state, end=start + chunk)
            model.free(h)
            h = out
            if watch is not None:
                ttnn.synchronize_device(model.mesh)
                print(f"chunk {c} layer {i}: {time.time() - t1:.1f} s", flush=True)
            if watch is not None and watch.wants(i):
                watch(c, f"L{i}", model.to_host(h)[:chunk])
        final = model.final_norm(h)
        model.free(h)
        ttnn.synchronize_device(model.mesh)
        times.append(round(time.time() - t0, 3))
        if watch is not None:
            watch(c, "final_norm", model.to_host(final)[:chunk])
            if c == seq // chunk - 1:
                watch(c, "final_dev", final)
        model.free(final)
    return times


def _dram_mb(mesh) -> float:
    """DRAM allocated on a chip (all banks), MB: before and after a capture, the trace's size."""
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return round(v.total_bytes_allocated_per_bank * v.num_banks / 2**20, 1)


def _dram_free_mb(mesh) -> float:
    """DRAM still free on a chip (all banks), MB: the headroom left for traces and longer contexts."""
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return round(v.total_bytes_free_per_bank * v.num_banks / 2**20, 1)


def _depth_fit(times):
    """chunk time = a + b * chunk index (least squares, as tt-metal's chunked prefill tests report): a is the cost every
    chunk pays, b the extra per 5120 tokens of KV depth already cached. Returns ms."""
    n = len(times)
    if n < 3:
        return None
    xb, yb = (n - 1) / 2, statistics.mean(times)
    b = sum((i - xb) * (t - yb) for i, t in enumerate(times)) / sum((i - xb) ** 2 for i in range(n))
    return {"a_ms": round((yb - b * xb) * 1000, 1), "b_ms_per_chunk": round(b * 1000, 2)}


def _prefill_traced(model, tokens, seq: int, chunk: int, layers, repeats: int = 1):
    """Each chunk's layers and final norm captured as a ttnn trace, replayed (timed) and released before the next
    chunk; one trace at a time fits next to the weights. The embedding runs eagerly and is copied into the trace's
    persistent input. Returns per-chunk replay seconds, capture seconds, trace MB of DRAM, and {chunk: final norm host}.
    """
    mesh = model.mesh
    state = model.new_state(seq)  # caches zeroed; each replay writes its chunk as an eager prefill does
    h_in = model.embed(tokens[:chunk], start=0)
    times, captures, sizes, out = [], [], [], {}
    issues = REPLAY_ISSUE_MS
    try:
        for c in range(seq // chunk):
            start = c * chunk
            h = model.embed(tokens[start : start + chunk], start=start)
            ttnn.copy(h, h_in)
            model.free(h)
            ttnn.synchronize_device(mesh)
            dram0 = _dram_mb(mesh)
            t0 = time.time()
            tid = sub_device.begin_capture(mesh)
            try:
                h = h_in
                for i in layers:
                    o = model.layer(i, h, start, state, end=start + chunk)
                    if h is not h_in:
                        model.free(h)
                    h = o
                final = model.final_norm(h)
                model.free(h)
            finally:
                sub_device.end_capture(mesh, tid)
            captures.append(round(time.time() - t0, 3))
            sizes.append(round(_dram_mb(mesh) - dram0, 1))
            try:
                # a replay rewrites this chunk's own KV rows with the same values, so repeats are idempotent
                runs = []
                for _ in range(max(1, repeats)):
                    t1 = time.time()
                    sub_device.execute(mesh, tid)
                    issues.append(round((time.time() - t1) * 1e3, 2))  # host time to enqueue the replay's segments
                    ttnn.synchronize_device(mesh)
                    runs.append(time.time() - t1)
                times.append([round(t, 4) for t in runs])
                out[c] = model.to_host(final)[:chunk]
            finally:
                sub_device.release(mesh, tid)
                model.free(final)
    finally:
        model.free(h_in)
    return times, captures, sizes, out


def _same(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Bit-identity of two host tensors (the traced path against eager), and how far apart they are if not."""
    a, b = a.float(), b.float()
    return {"equal": bool(torch.equal(a, b)), "max_abs": float((a - b).abs().max()), "pcc": round(_pcc(a, b), 7)}


def _real_rows(model, start: int, end: int) -> torch.Tensor:
    """Mask of the chunk rows (to_host's SP-row order, model.embed's ``order``) that hold real tokens [start, end)."""
    g = torch.arange(start, start + model.chunk)
    order = sorted(range(model.chunk), key=lambda k: ((int(g[k]) // model.rows) % model.sp, k))
    return torch.tensor([k < end - start for k in order])


class _Eager:
    """The model's eager (host-scalar) path while the metadata path is set up: model.meta unset inside."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.saved, self.model.meta = self.model.meta, None

    def __exit__(self, *a):
        self.model.meta = self.saved


def _kv_snapshot(model, state, slot: int, length: int, layers) -> dict:
    state.slot = slot
    return {i: state.to_torch(i, length) for i in layers}


def _kv_same(a: dict, b: dict) -> dict:
    out = {}
    for i in a:
        for k in a[i]:
            if a[i][k].numel():
                out[f"L{i}.{k}"] = bool(torch.equal(a[i][k], b[i][k]))
    return {"all_equal": all(out.values()), "per": out}


def _meta_suite(model, tokens, seq: int, chunk: int, layers, eager_final: dict, gold, rows: int, repeats: int) -> dict:
    """The trace-safe (metadata) forward (tt/mistral_large_4/chunk_meta.py). (1) its exact fp32 table selection vs the eager slices;
    (2) an eager prefill on the metadata path vs the eager one (bit-identical expected); (3) serving-like chunks (slot
    1, a start inside a chunk, a partial end) eager vs metadata, outputs and KV; (4) ONE captured trace of the
    metadata forward replayed for every chunk of the prompt and the serving-like chunks, against eager."""
    mesh = model.mesh
    res = {}
    meta = model.enable_meta()
    state = model.new_state(seq)
    full = [i for i in layers if model.cfg.is_full(i)]
    rope_full = [i for i in full if model.cfg.use_rope(i)]

    # (1) split tables (rope.SplitTables) vs RowTables at a few starts
    sel = {}
    mod = model.mods[rope_full[0]]["indexer"] if rope_full else None
    if mod is not None:
        mod._split_tables()
        comp = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1))
        for start in (0, 4480, 51200):
            meta.write(0, start, start + chunk)
            meta.unpack()
            for name, rt, n, sub in (
                ("cos", mod.k_rope[0], mod.rows, False),
                ("sin", mod.k_rope[1], mod.rows, False),
                ("cos_q", mod.q_rope[0], mod.q_rows, True),
                ("f", mod.f_rows, mod.q_rows, True),
            ):
                if rt is None:
                    continue
                got = mod._selected(meta, name.replace("_q", ""), n, sub)
                ref = rt.rows(start)
                a, b = ttnn.to_torch(got, mesh_composer=comp), ttnn.to_torch(ref, mesh_composer=comp)
                sel[f"{name}@{start}"] = _same(a, b)
                if start % chunk:
                    ttnn.deallocate(ref)
        meta.free_derived()
    res["split_tables_vs_rowtables"] = sel

    # (2) eager prefill on the metadata path
    finals, times = {}, []
    for c in range(seq // chunk):
        start = c * chunk
        meta.write(0, start, start + chunk)
        t0 = time.time()
        h = model.embed(tokens[start : start + chunk], start=start)
        f = model.forward_meta(h, state, layers)
        model.free(h)
        ttnn.synchronize_device(mesh)
        times.append(round(time.time() - t0, 3))
        finals[c] = model.to_host(f)[:chunk]
        model.free(f)
        print(f"meta-eager chunk {c}: {times[-1]} s, equal {torch.equal(finals[c], eager_final[c])}", flush=True)
    res["meta_eager_chunk_seconds"] = times
    res["meta_eager_vs_eager"] = {c: _same(finals[c], eager_final[c]) for c in finals}
    res["meta_eager_bit_identical"] = all(v["equal"] for v in res["meta_eager_vs_eager"].values())

    # (3) serving-like chunks: (slot, start, end); eager, then the metadata path, outputs (real rows) and KV
    # a start on a block boundary (prefix reuse at 4480), one inside a block with a partial end, a short last chunk
    cases = [(1, 4480, 4480 + chunk), (1, 9024, 9024 + chunk - 151), (0, seq - chunk, seq - 320)]
    kv_layers = [i for i in (0, 1, 3, 35, 39, 71) if i in layers]
    case_eager, case_res = {}, {}
    for n, (slot, start, end) in enumerate(cases):
        toks = tokens[start : start + chunk]
        # rows the chunk writes or zeroes ([start, ceil32(end))) and the prefix; rows past them are other chunks' (the
        # replays below run after every case's eager runs)
        kv_len = -(-end // 32) * 32
        with _Eager(model):
            state.slot = slot
            h = model.embed(toks, start=start)
            for i in layers:
                o = model.layer(i, h, start, state, end=end)
                model.free(h)
                h = o
            f = model.final_norm(h)
            model.free(h)
            ef = model.to_host(f)[:chunk]
            model.free(f)
        kv_e = _kv_snapshot(model, state, slot, kv_len, kv_layers)
        meta.write(slot, start, end)
        h = model.embed(toks, start=start)
        f = model.forward_meta(h, state, layers)
        model.free(h)
        mf = model.to_host(f)[:chunk]
        model.free(f)
        kv_m = _kv_snapshot(model, state, slot, kv_len, kv_layers)
        real = _real_rows(model, start, end)
        case_eager[n] = (ef, kv_e, real)
        case_res[n] = {
            "case": [slot, start, end],
            "meta_vs_eager_real_rows": _same(mf[real], ef[real]),
            "meta_vs_eager_all_rows": _same(mf, ef),
            "kv_meta_vs_eager": _kv_same(kv_e, kv_m),
        }
        print(f"case {n} {cases[n]}: {case_res[n]['meta_vs_eager_real_rows']}", flush=True)
    res["cases"] = case_res

    # (4) one trace for every chunk
    meta.write(0, 0, chunk)
    h_in = model.embed(tokens[:chunk], start=0)
    ttnn.synchronize_device(mesh)
    dram0 = _dram_mb(mesh)
    t0 = time.time()
    tid = sub_device.begin_capture(mesh)
    try:
        final = model.forward_meta(h_in, state, layers)
    finally:
        sub_device.end_capture(mesh, tid)
    res["capture_seconds"] = round(time.time() - t0, 2)
    res["trace_dram_mb"] = round(_dram_mb(mesh) - dram0, 1)

    def replay(slot, start, end, toks, reps):
        t0 = time.time()
        meta.write(slot, start, end)
        h = model.embed(toks, start=start)
        ttnn.copy(h, h_in)
        model.free(h)
        runs = []
        for _ in range(reps):
            t1 = time.time()
            sub_device.execute(mesh, tid)
            ttnn.synchronize_device(mesh)
            runs.append(round(time.time() - t1, 4))
        return runs, round(time.time() - t0, 4)

    try:
        runs, walls, cmp_e, cmp_g = [], [], {}, {}
        for c in range(seq // chunk):
            start = c * chunk
            r, w = replay(0, start, start + chunk, tokens[start : start + chunk], max(1, repeats))
            runs.append(r)
            walls.append(w)
            host = model.to_host(final)[:chunk]
            cmp_e[c] = _same(host, eager_final[c])
            cmp_g[c] = _cmp(host, gold[c]["final_norm"].float())
            print(f"traced chunk {c}: {r} (wall {w}), equal {cmp_e[c]['equal']}", flush=True)
        c = seq // chunk - 1
        lg = model.logits(final, list(range(chunk - rows, chunk)))
        g = gold[c]["logits"].float()
        res["logits_traced"] = {
            **_cmp(lg, g),
            "top1_match_rows": int((lg.argmax(-1) == g.argmax(-1)).sum()),
            "last_top1": [int(lg[-1].argmax()), int(g[-1].argmax())],
        }
        med = [statistics.median(r) for r in runs]
        res["chunk_seconds_traced"] = [round(t, 4) for t in med]
        res["chunk_seconds_traced_runs"] = runs
        res["chunk_seconds_traced_wall"] = walls  # meta.write + embed + copy + every repeat
        res["prefill_seconds_traced"] = round(sum(med), 3)
        res["depth_fit_traced"] = _depth_fit(med)
        res["traced_vs_eager"] = cmp_e
        res["traced_vs_golden"] = cmp_g
        res["traced_bit_identical"] = all(v["equal"] for v in cmp_e.values())
        for n, (slot, start, end) in enumerate(cases):
            r, w = replay(slot, start, end, tokens[start : start + chunk], 1)
            host = model.to_host(final)[:chunk]
            ef, kv_e, real = case_eager[n]
            kv_len = -(-end // 32) * 32
            case_res[n]["traced_seconds"] = r[0]
            case_res[n]["traced_vs_eager_real_rows"] = _same(host[real], ef[real])
            case_res[n]["kv_traced_vs_eager"] = _kv_same(kv_e, _kv_snapshot(model, state, slot, kv_len, kv_layers))
            print(f"traced case {n}: {case_res[n]['traced_vs_eager_real_rows']}", flush=True)
    finally:
        sub_device.release(mesh, tid)
        model.free(h_in)
    return res


@pytest.mark.timeout(25200)  # weight load plus waiting on the CPU golden, which runs alongside
@pytest.mark.skipif(os.environ.get("ML4_E2E_GALAXY") != "1", reason="needs a Blackhole Galaxy (ML4_E2E_GALAXY=1)")
def test_e2e_full(mesh_device):
    from safetensors.torch import load_file

    meta = _meta()
    seq, chunk, dump, rows = meta["seq"], meta["chunk"], set(meta["dump"]), meta["logits_rows"]
    layers = list(range(meta["layers"][0], meta["layers"][1] + 1))
    tokens = json.loads(Path(meta["prompt"]).read_text())["token_ids"][:seq]
    gold = _Gold(seq // chunk, load_file)
    result = {"golden": str(GOLDEN), "layers": meta["layers"], "seq": seq, "chunk": chunk, "compare": {}}
    result["node"] = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    result["fabric"] = os.environ.get("ML4_FABRIC", "ring1d")
    result["fabric_payload"] = int(os.environ.get("ML4_FABRIC_PAYLOAD", "12288")) or None
    result["partition"] = os.environ.get("SLURM_JOB_PARTITION")

    t0 = time.time()
    model = device_model(mesh_device, _Spec(CKPT, max(seq, CAPACITY), chunk), layers, lm_head=True)
    result["build_seconds"] = round(time.time() - t0, 1)

    eager_final = {}

    class Watch:
        def wants(self, i):
            return i in dump

        def __call__(self, c, name, t):
            if name == "final_norm":
                eager_final[c] = t
            if name == "final_dev":
                logits = model.logits(t, list(range(chunk - rows, chunk)))
                g = gold[c]["logits"].float()
                result["logits"] = {
                    **_cmp(logits, g),
                    "last_row_pcc": round(_pcc(logits[-1], g[-1]), 6),
                    "top1_match_rows": int((logits.argmax(-1) == g.argmax(-1)).sum()),
                    "rows": rows,
                    "last_top1": [int(logits[-1].argmax()), int(g[-1].argmax())],
                    **_top1_detail(logits, g),
                }
                return
            result["compare"].setdefault(name, {})[c] = _cmp(t.float(), gold[c][name].float())

    result["chunk_seconds_first"] = _prefill(model, tokens, seq, chunk, layers, Watch())
    result["load_seconds"] = round(model.load_seconds, 1)
    result["kv_vs_golden"] = _kv_vs_golden(model, _LAST_STATE, seq, layers)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")  # the accuracy results survive a later failure
    if os.environ.get("ML4_E2E_WARM", "1") == "1":
        result["chunk_seconds_warm"] = _prefill(model, tokens, seq, chunk, layers)
        result["prefill_seconds_warm"] = round(sum(result["chunk_seconds_warm"]), 2)
        result["depth_fit_warm"] = _depth_fit(result["chunk_seconds_warm"])
    if os.environ.get("ML4_E2E_TRACE") == "meta":
        # the trace-safe forward: one capture replayed for every chunk
        try:
            repeats = int(os.environ.get("ML4_E2E_TRACE_REPEATS", "3"))
            result["dram_free_mb_before_trace"] = _dram_free_mb(model.mesh)
            result["meta"] = _meta_suite(model, tokens, seq, chunk, layers, eager_final, gold, rows, repeats)
        except Exception as e:
            import traceback

            traceback.print_exc()
            result["trace_error"] = f"{type(e).__name__}: {e}"[:2000]
    if os.environ.get("ML4_E2E_TRACE") == "1":
        try:
            repeats = int(os.environ.get("ML4_E2E_TRACE_REPEATS", "5"))
            result["dram_free_mb_before_trace"] = _dram_free_mb(model.mesh)
            runs, capture, sizes, finals = _prefill_traced(model, tokens, seq, chunk, layers, repeats)
            med = [round(statistics.median(r), 4) for r in runs]
            result["chunk_seconds_traced"] = med  # per-chunk median over the repeats
            result["chunk_seconds_traced_stdev"] = [round(statistics.stdev(r), 4) if len(r) > 1 else 0.0 for r in runs]
            result["chunk_seconds_traced_runs"] = runs
            result["capture_seconds"], result["trace_dram_mb"] = capture, sizes
            result["replay_issue_ms"] = list(REPLAY_ISSUE_MS)  # host time to enqueue each replay
            result["trace_segments"] = sub_device.last_segments
            result["prefill_seconds_traced"] = round(sum(med), 2)
            result["tokens_per_second_traced"] = round(seq / sum(med))
            result["depth_fit_traced"] = _depth_fit(med)
            result["compare_traced"] = {c: _cmp(t.float(), gold[c]["final_norm"].float()) for c, t in finals.items()}
        except Exception as e:  # the eager results above still get written
            result["trace_error"] = f"{type(e).__name__}: {e}"[:2000]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))

    finals = [v["pcc"] for v in result["compare"]["final_norm"].values()]
    assert all(v["finite"] for d in result["compare"].values() for v in d.values()), "non-finite device output"
    assert min(finals) >= PCC_LOWER, f"final norm PCC {min(finals)} < {PCC_LOWER}"
    assert result["logits"]["last_top1"][0] == result["logits"]["last_top1"][1], "last position's top-1 differs"


# ---- serving soak: the metadata path's trace replayed many times (trace calls go through sub_device.py)
def _boot_check() -> None:
    """ML4_BOOT_CHECK_LOG (the job's own log): fail before the weight load when the mesh came up with the real-time
    profiler clock sync failing ("sync failed - not enough samples"), which precedes hangs during the load."""
    log = os.environ.get("ML4_BOOT_CHECK_LOG")
    if not log:
        return
    time.sleep(float(os.environ.get("ML4_BOOT_CHECK_WAIT_S", "30")))  # the sync messages follow the mesh open
    text = Path(log).read_text(errors="replace") if Path(log).exists() else ""
    n = text.count("sync failed")
    if n:
        pytest.fail(f"bad boot: {n} real-time profiler sync failures at mesh open (resubmit on another node)")


@pytest.mark.skipif(os.environ.get("ML4_SOAK") != "1", reason="needs a Blackhole Galaxy (ML4_SOAK=1)")
def test_trace_soak(mesh_device):
    """Serving reliability: the traced forward (one capture of the metadata path) replayed ML4_SOAK_N times (default
    600), cycling the chunk start over the first ML4_SOAK_CHUNKS positions (default 11: a 56320-token prompt) and the
    slot over 2, as a serving engine drives it; one line per replay (time, slot, start, host enqueue
    ms, replay ms). A hang shows as the dispatch timeout (TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE runs tt-triage).
    """
    _boot_check()
    max_seq, slots = int(os.environ.get("ML4_SOAK_MAX_SEQ", CAPACITY)), int(os.environ.get("ML4_SOAK_SLOTS", "2"))
    n_rep = int(os.environ.get("ML4_SOAK_N", "600"))
    n_pos = int(os.environ.get("ML4_SOAK_CHUNKS", "11"))
    meta_g = _meta()
    chunk = meta_g["chunk"]
    layers = list(range(meta_g["layers"][0], meta_g["layers"][1] + 1))
    prompt = json.loads(Path(meta_g["prompt"]).read_text())["token_ids"]
    out_path = Path(os.environ.get("ML4_SOAK_OUT", str(OUT)))
    res = {
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "max_seq": max_seq,
        "slots": slots,
        "replays": n_rep,
        "positions": n_pos,
        "settings": {k: v for k, v in os.environ.items() if k.startswith(("MISTRAL_LARGE_4_", "ML4_FABRIC"))},
    }

    def dump():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=1) + "\n")

    t0 = time.time()
    model = device_model(mesh_device, _Spec(CKPT, max_seq, chunk), layers, lm_head=False, num_users=slots)
    mesh = model.mesh
    meta = model.enable_meta()
    state = model.new_state(max_seq)
    toks = [torch.as_tensor(prompt[c * chunk : (c + 1) * chunk]) for c in range(n_pos)]
    meta.write(0, 0, chunk)
    h = model.embed(toks[0], start=0)
    model.free(model.forward_meta(h, state, layers))  # builds every layer, compiles
    model.free(h)
    ttnn.synchronize_device(mesh)
    res["build_seconds"] = round(time.time() - t0, 1)
    res["dram_free_mb"] = _dram_free_mb(mesh)
    meta.write(0, 0, chunk)
    h_in = model.embed(toks[0], start=0)
    ttnn.synchronize_device(mesh)
    tid = sub_device.begin_capture(mesh)
    try:
        final = model.forward_meta(h_in, state, layers)
    finally:
        sub_device.end_capture(mesh, tid)
    res["trace_segments"] = sub_device.last_segments
    print(
        f"{time.strftime('%H:%M:%S')} soak: built in {res['build_seconds']} s, {res['dram_free_mb']} MB free, "
        f"{res['trace_segments']} trace segments",
        flush=True,
    )
    dump()
    ms, enq = [], []
    try:
        for k in range(n_rep):
            c, slot = k % n_pos, (k // n_pos) % slots
            start = c * chunk
            meta.write(slot, start, start + chunk)
            hc = model.embed(toks[c], start=start)
            ttnn.copy(hc, h_in)
            model.free(hc)
            t = time.time()
            sub_device.execute(mesh, tid)
            enq.append(round((time.time() - t) * 1e3, 2))
            ttnn.synchronize_device(mesh)
            ms.append(round((time.time() - t) * 1e3, 2))
            print(
                f"{time.strftime('%H:%M:%S')} replay {k} slot {slot} start {start}: enqueue {enq[-1]} ms, "
                f"{ms[-1]} ms",
                flush=True,
            )
            if k % 50 == 0:
                res["done"], res["ms"], res["enqueue_ms"] = k + 1, ms, enq
                dump()
        fn = model.to_host(final).float()
        res["final_finite"] = bool(fn.isfinite().all())
    finally:
        res["done"], res["ms"], res["enqueue_ms"] = len(ms), ms, enq
        dump()
        sub_device.release(mesh, tid)
        model.free(h_in)
    assert len(ms) == n_rep and res["final_finite"]


# ---- context-length sweep: one load, one capture, every ISL replayed through it
SWEEP_ISL = (5120, 10240, 20480, 40960, 56320, 102400, 204800, 409600, 655360, 880640, 1044480)


def _golden_logits(head: torch.Tensor, x: torch.Tensor, softcap: float) -> torch.Tensor:
    """The reference's logits of final-norm rows x [n, H] (fp32, the checkpoint's LM head)."""
    out = []
    for a in range(0, head.shape[0], 16384):
        out.append(x.float() @ head[a : a + 16384].float().t())
    lg = torch.cat(out, dim=-1)
    return torch.tanh(lg / softcap) * softcap


@pytest.mark.skipif(os.environ.get("ML4_CTX_SWEEP") != "1", reason="needs a Blackhole Galaxy (ML4_CTX_SWEEP=1)")
def test_ctx_sweep(mesh_device):
    """Prefill time against context length on one Galaxy. One model load at the longest ISL (ML4_CTX_ISL, default
    SWEEP_ISL; 1 slot, SEQ_BUFFERS=shared, SSMAX_TABLE=shared_log), one eager
    metadata-path chunk (builds every layer, compiles), ONE capture of the trace-safe forward, then each ISL as a prefill from position 0 replayed chunk by chunk through that capture
    (meta.write, the embedded chunk copied into the persistent input, execute, synchronize: the chunk's ms).

    Tokens: the golden prompt (56320), then the prompt again (token p = prompt[p % 56320]). Accuracy: up to 56320, the
    final norm of every chunk vs the CPU golden, and the last 32 rows' logits vs the reference's (the golden's logits on
    its last chunk; on earlier chunks the reference's LM head on the golden's final norm); beyond, finite outputs and the
    last 32 rows' top-1 vs the next prompt token. Writes ML4_CTX_OUT (commit, node, points)."""
    from safetensors.torch import load_file

    _boot_check()
    os.environ.setdefault("MISTRAL_LARGE_4_SEQ_BUFFERS", "shared")
    os.environ.setdefault("MISTRAL_LARGE_4_SSMAX_TABLE", "shared_log")
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4._bringup_interface import weight_dtype
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import checkpoint, load_dense

    meta_g = _meta()
    gseq, chunk, rows = meta_g["seq"], meta_g["chunk"], meta_g["logits_rows"]
    layers = list(range(meta_g["layers"][0], meta_g["layers"][1] + 1))
    prompt = torch.as_tensor(json.loads(Path(meta_g["prompt"]).read_text())["token_ids"][:gseq])
    gold = _Gold(gseq // chunk, load_file)
    isls = [int(x) for x in os.environ.get("ML4_CTX_ISL", ",".join(map(str, SWEEP_ISL))).split(",")]
    assert all(i % chunk == 0 for i in isls), isls
    max_seq = max(isls)
    out_path = Path(os.environ.get("ML4_CTX_OUT", str(OUT)))
    res = {
        "commit": os.environ.get("ML4_COMMIT", ""),
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "build": {
            "max_seq": max_seq,
            "slots": 1,
            "seq_buffers": os.environ["MISTRAL_LARGE_4_SEQ_BUFFERS"],
            "ssmax_table": os.environ["MISTRAL_LARGE_4_SSMAX_TABLE"],
            "trace_region_mb": int(os.environ.get("ML4_TRACE_REGION_MB", "0")),
        },
        "points": [],
        "breakdown": [],
        "served": [],
    }

    def dump():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=1) + "\n")

    def toks(start):
        return prompt[torch.arange(start, start + chunk) % gseq]

    t0 = time.time()
    model = device_model(mesh_device, _Spec(CKPT, max_seq, chunk), layers, lm_head=True, num_users=1)
    mesh = model.mesh
    meta = model.enable_meta()
    state = model.new_state(max_seq)
    # the eager metadata-path chunk: builds every layer (the weight load), compiles, builds the split tables
    meta.write(0, 0, chunk)
    h = model.embed(toks(0), start=0)
    model.free(model.forward_meta(h, state, layers))
    model.free(h)
    ttnn.synchronize_device(mesh)
    res["build"]["seconds"] = round(time.time() - t0, 1)
    res["build"]["dram_free_mb"] = _dram_free_mb(mesh)
    print(f"built: {res['build']}", flush=True)
    dump()

    cfg = Config(CKPT)
    head = load_dense(checkpoint(Path(CKPT)), {"w": "output.weight"}, weight_dtype(_Spec(CKPT, max_seq, chunk)))["w"]
    g10 = gold[gseq // chunk - 1]
    check = _golden_logits(head, g10["final_norm"][chunk - rows : chunk], cfg.softcap)
    res["golden_logits_check_pcc"] = round(_pcc(check, g10["logits"].float()), 6)  # host LM head vs the golden's logits

    meta.write(0, 0, chunk)
    h_in = model.embed(toks(0), start=0)
    ttnn.synchronize_device(mesh)
    t1 = time.time()
    tid = sub_device.begin_capture(mesh)
    try:
        final = model.forward_meta(h_in, state, layers)
    finally:
        sub_device.end_capture(mesh, tid)
    res["build"]["capture_seconds"] = round(time.time() - t1, 1)
    res["build"]["trace_segments"] = sub_device.last_segments  # MOE_OVERLAP splits the one capture
    res["build"]["dram_free_mb_after_capture"] = _dram_free_mb(mesh)
    dump()
    try:
        for isl in isls:
            n = isl // chunk
            pt = {"isl": isl, "chunk_ms": [], "dram_free_mb": res["build"]["dram_free_mb"]}
            norms, finite = [], True
            digest = (
                hashlib.sha256()
            )  # the bytes of every final norm read back + the last logits: run-to-run bit identity
            every = os.environ.get("ML4_CTX_HASH") == "all"
            for c in range(n):
                start = c * chunk
                meta.write(0, start, start + chunk)
                hc = model.embed(toks(start), start=start)
                ttnn.copy(hc, h_in)
                model.free(hc)
                ttnn.synchronize_device(mesh)
                t = time.time()
                sub_device.execute(mesh, tid)
                ttnn.synchronize_device(mesh)
                pt["chunk_ms"].append(round((time.time() - t) * 1e3, 2))
                if c % 16 == 0:  # progress, to locate a hang
                    print(f"{time.strftime('%H:%M:%S')} ISL {isl} chunk {c}/{n}: {pt['chunk_ms'][-1]} ms", flush=True)
                golden_chunk = c < gseq // chunk
                if golden_chunk or c == n - 1 or c % 16 == 0 or every:
                    fn = model.to_host(final)[:chunk].float()
                    digest.update(c.to_bytes(4, "little") + fn.numpy().tobytes())
                    finite &= bool(fn.isfinite().all())
                    if golden_chunk:
                        norms.append(_pcc(fn, gold[c]["final_norm"].float()))
                if c == n - 1:
                    lg = model.logits(final, list(range(chunk - rows, chunk)))
                    finite &= bool(lg.isfinite().all())
                    digest.update(lg.float().numpy().tobytes())
                    nxt = prompt[(torch.arange(start + chunk - rows, start + chunk) + 1) % gseq]
                    pt["next_token_rows"] = f"{int((lg.argmax(-1) == nxt).sum())}/{rows}"
                    if golden_chunk:
                        g = gold[c]
                        gl = (
                            g["logits"].float()
                            if "logits" in g
                            else _golden_logits(head, g["final_norm"][chunk - rows : chunk], cfg.softcap)
                        )
                        pt["logits_pcc"] = round(_pcc(lg, gl), 6)
                        pt["top1_rows"] = f"{int((lg.argmax(-1) == gl.argmax(-1)).sum())}/{rows}"
            ms = pt["chunk_ms"]
            pt["prefill_s"] = round(sum(ms) / 1e3, 3)
            pt["tok_s"] = round(isl / (sum(ms) / 1e3))
            fit = _depth_fit([m / 1e3 for m in ms])
            if fit is not None:
                pt["a_ms"], pt["b_ms"] = fit["a_ms"], fit["b_ms_per_chunk"]
            else:
                pt["a_ms"], pt["b_ms"] = ms[0], (round(ms[1] - ms[0], 2) if n == 2 else None)
            pt["final_norm_pcc_min"] = round(min(norms), 6) if norms else None
            pt.setdefault("logits_pcc", None)
            pt.setdefault("top1_rows", None)
            if (
                isl <= gseq
            ):  # the golden's own text: the top-1 rows vs the golden are the check; keep this for reference
                pt["next_token_rows_natural"] = pt.pop("next_token_rows", None)
                pt["next_token_rows"] = None
            pt["finite"] = finite
            pt["out_sha256"] = digest.hexdigest()[:16]
            pt["hashed_chunks"] = "all" if every else "golden+every16+last"
            res["points"].append(pt)
            print(
                f"ISL {isl}: {pt['prefill_s']} s, {pt['tok_s']} tok/s, a {pt['a_ms']} b {pt['b_ms']}, "
                f"norm {pt['final_norm_pcc_min']}, logits {pt['logits_pcc']} {pt['top1_rows']}, "
                f"next {pt['next_token_rows']}, finite {finite}",
                flush=True,
            )
            dump()
    finally:
        sub_device.release(mesh, tid)
        model.free(h_in)
    assert all(p["finite"] for p in res["points"]), "non-finite output"
    pccs = [p["final_norm_pcc_min"] for p in res["points"] if p["final_norm_pcc_min"] is not None]
    assert not pccs or min(pccs) >= 0.96, pccs


# ---- run-to-run determinism of the long build
@pytest.mark.skipif(os.environ.get("ML4_NDET") != "1", reason="needs a Blackhole Galaxy (ML4_NDET=1)")
def test_ndet(mesh_device):
    """Where the long build stops being deterministic. One load (ML4_NDET_MAX_SEQ, default 1044480: 1 slot,
    SEQ_BUFFERS=shared, SSMAX_TABLE=shared_log), then ML4_NDET_REPS (default 3) passes of chunks 0..ML4_NDET_CHUNKS-1
    (default 11) from position 0 through the metadata path, uncaptured (the traced forward's ops, one at a time), as the
    sweep's ISL passes. Every layer's output is hashed (blake2b-8 of each device shard, host side); from chunk
    ML4_NDET_DETAIL_FROM (default 4) every step boundary of every layer too (mistral_large_4/model.py run_block's
    rec). Writes ML4_NDET_OUT; prints the first (chunk, layer, step) any pass differs
    from pass 0 at. Across jobs: compare the JSONs (same node, other node, other payload)."""
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import model as tm

    _boot_check()
    max_seq = int(os.environ.get("ML4_NDET_MAX_SEQ", "1044480"))
    if max_seq > CAPACITY:
        os.environ.setdefault("MISTRAL_LARGE_4_SEQ_BUFFERS", "shared")
        os.environ.setdefault("MISTRAL_LARGE_4_SSMAX_TABLE", "shared_log")
    reps = int(os.environ.get("ML4_NDET_REPS", "3"))
    n = int(os.environ.get("ML4_NDET_CHUNKS", "11"))
    detail_from = int(os.environ.get("ML4_NDET_DETAIL_FROM", "4"))
    meta_g = _meta()
    chunk, gseq = meta_g["chunk"], meta_g["seq"]
    layers = list(range(meta_g["layers"][0], meta_g["layers"][1] + 1))
    prompt = torch.as_tensor(json.loads(Path(meta_g["prompt"]).read_text())["token_ids"][:gseq])
    out_path = Path(os.environ.get("ML4_NDET_OUT", str(OUT)))
    res = {
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "max_seq": max_seq,
        "reps": reps,
        "chunks": n,
        "detail_from": detail_from,
        "payload": int(os.environ.get("ML4_FABRIC_PAYLOAD", "12288")),
        "settings": {k: v for k, v in os.environ.items() if k.startswith("MISTRAL_LARGE_4_")},
        "passes": [],
    }

    def dump():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res) + "\n")

    t0 = time.time()
    model = device_model(mesh_device, _Spec(CKPT, max_seq, chunk), layers, lm_head=False, num_users=1)
    mesh = model.mesh
    meta = model.enable_meta()
    state = model.new_state(max_seq)
    res["build_seconds"] = round(time.time() - t0, 1)
    composer = ttnn.ConcatMeshToTensor(mesh, dim=0)

    def h8(t):
        if isinstance(t, (list, tuple)):
            return [h8(x) for x in t]
        if not isinstance(t, ttnn.Tensor):
            return None
        x = ttnn.to_torch(t, mesh_composer=composer)
        x = x.float() if x.is_floating_point() else x.long()
        return hashlib.blake2b(x.contiguous().numpy().tobytes(), digest_size=8).hexdigest()

    steps = []  # this layer's step boundaries, when detailed
    detail = [False]
    run_block = tm.run_block

    def hashed_run_block(graph, component, ctx, h, rec=None, **kw):
        def rec2(name, t):
            if detail[0]:
                steps.append([name, h8(t)])
            if rec is not None:
                rec(name, t)

        return run_block(graph, component, ctx, h, rec=rec2, **kw)

    tm.run_block = hashed_run_block
    try:
        for r in range(reps):
            p = []
            res["passes"].append(p)
            for c in range(n):
                start = c * chunk
                meta.write(0, start, start + chunk)
                h = model.embed(prompt[torch.arange(start, start + chunk) % gseq], start=start)
                meta.unpack()
                state.slot = 0
                rec_c = {"layers": []}
                t = time.time()
                for i in layers:
                    detail[0] = c >= detail_from
                    steps.clear()
                    out = model.layer(i, h, start, state, end=start + chunk)
                    model.free(h)
                    h = out
                    rec_c["layers"].append({"i": i, "out": h8(h), "steps": list(steps)})
                fn = model.final_norm(h)
                model.free(h)
                rec_c["final"] = h8(fn)
                model.free(fn)
                p.append(rec_c)
                print(
                    f"{time.strftime('%H:%M:%S')} pass {r} chunk {c}: final {rec_c['final']} "
                    f"({time.time() - t:.1f} s)",
                    flush=True,
                )
            dump()
    finally:
        tm.run_block = run_block
    # the first difference from pass 0
    first = []
    for r in range(1, reps):
        for c in range(n):
            a, b = res["passes"][0][c], res["passes"][r][c]
            hit = None
            for la, lb in zip(a["layers"], b["layers"]):
                for sa, sb in zip(la["steps"], lb["steps"]):
                    if sa != sb:
                        hit = {"pass": r, "chunk": c, "layer": la["i"], "step": sa[0]}
                        break
                if hit is None and la["out"] != lb["out"]:
                    hit = {"pass": r, "chunk": c, "layer": la["i"], "step": "out"}
                if hit:
                    break
            if hit:
                first.append(hit)
                break
    res["first_difference"] = first
    dump()
    print(f"first differences from pass 0: {first or 'none (all passes identical)'}", flush=True)
