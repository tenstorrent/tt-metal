# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the long-context chunked prefill spends its time (tt-blaze G53F-0172).

Builds the seed server's model (the same env: ``seed_server42.sh`` long geometry) and runs ``PROF_RUNS``
passes of ``prefill_chunked`` over the first ``PROF_N`` ids of ``PROF_IDS``. A pass is ``<chunk>:<instr>[:<n>[:<regions_t>]]``
(``n``: this pass uses only the first n ids; ``regions_t``: the MoE regions sub-chunk for this pass; then optionally
``:<rank>`` (cumsum | mm), ``:<buf>`` (full | fit, the region buffer rows, G53F-0183) and ``:<gather>`` (embedding |
row, the regions path's row gathers, G53F-0186)):
``instr=1`` wraps every layer call, its attention core, its FFN core (MoE: router / experts / shared) with
device synchronizations and installs the DSA ``StageTimer`` (the sync points change the overlap, so the
instrumented total is an upper bound); ``instr=0`` times only whole chunks (the real cost). Rows go to
``PROF_OUT`` (jsonl): one per chunk with every component's seconds, plus a summary per pass.

    PROF_IDS=.../needle_260000_d10.ids PROF_RUNS=16384:1,16384:0 PROF_OUT=prof.jsonl \
      pytest models/demos/glm53_flash/tests/test_prefill_profile.py -s
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt import moe as moe_mod
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import StageTimer
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
MAX_SEQ = int(os.environ.get("GLM53F_PREFILL_MAXSEQ", "1048576"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/mnt/tt-data/sdawle/glm53f/prefill/mesh_cache")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
IDS = os.environ.get("PROF_IDS", "/mnt/tt-data/sdawle/glm53f/needle/needle_260000_d10.ids")
N = int(os.environ.get("PROF_N", "0"))
RUNS = [r.split(":") for r in os.environ.get("PROF_RUNS", "16384:1,16384:0").split(",") if r.strip()]
OUT = os.environ.get("PROF_OUT", "/mnt/tt-data/sdawle/glm53f/prefill/logs/prof.jsonl")
TRACE_AB = int(os.environ.get("PROF_TRACE_AB", "0"))  # decoder index whose FFN is A/B'd eager vs trace (0 = off)
TRACE_T = int(os.environ.get("PROF_TRACE_T", "16384"))
DUMP = os.environ.get("PROF_DUMP", "")
OPS_AB = int(os.environ.get("PROF_OPS", "0"))
#: G53F-0174 gate: "rt:rank,rt:rank,..." -- prefill the first PROF_EXPORT_N ids under each setting and sha256 every layer's
#: exported seed state (the server's export helpers); bitwise-identical exports across settings = a pure-perf change
EXPORT_AB = os.environ.get("PROF_EXPORT_AB", "")
EXPORT_ONLY_N = int(
    os.environ.get("PROF_EXPORT_ONLY_N", "0")
)  # G53F-0182: export-only timing of the last pass's caches
EXPORT_N = int(
    os.environ.get("PROF_EXPORT_N", "4092")
)  # decoder index whose routed experts get an op-by-op device breakdown  # path prefix: save each pass's last-token logits


class Prof:
    """Synchronized timers around wrapped callables; ``on`` gates them (off = pass-through)."""

    def __init__(self, device):
        self.device = device
        self.on = False
        self.acc: dict[str, float] = defaultdict(float)
        self.dsa_timer = StageTimer(device)

    def wrap(self, obj, name: str, key: str, dsa_start: bool = False):
        fn = getattr(obj, name)

        def timed(*a, **k):
            if not self.on:
                return fn(*a, **k)
            ttnn.synchronize_device(self.device)
            t = time.perf_counter()
            if dsa_start:
                self.dsa_timer._t = t
            r = fn(*a, **k)
            ttnn.synchronize_device(self.device)
            self.acc[key] += time.perf_counter() - t
            return r

        setattr(obj, name, timed)

    def take(self) -> dict[str, float]:
        out = dict(self.acc)
        for k, v in self.dsa_timer.times.items():
            out[f"dsa.{k}"] = v
        self.acc = defaultdict(float)
        self.dsa_timer.reset()
        return out


def _instrument(model, prof: Prof):
    for k, dec in enumerate(model.layers):
        layer = dec.layer
        kind = "dsa" if dec.cache is not None else "kda"
        ffn = "moe" if layer.is_moe else "dense"
        prof.wrap(dec, "prefill_forward", f"layer.{kind}.{ffn}")
        prof.wrap(layer.attn, "prefill_forward", f"attn.{kind}", dsa_start=(kind == "dsa"))
        if kind == "dsa":
            layer.attn.set_stage_timer(prof.dsa_timer)
        prof.wrap(layer.mlp, "forward", f"ffn.{ffn}")
        if layer.is_moe:
            for sub in ("router", "experts", "shared"):
                prof.wrap(getattr(layer.mlp, sub), "forward", f"ffn.moe.{sub}")


def _set_dsa_timer(model, prof: Prof, on: bool):
    for dec in model.layers:
        if dec.cache is not None:
            dec.layer.attn.set_stage_timer(prof.dsa_timer if on else None)


@pytest.mark.timeout(86400)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_prefill_profile(mesh_device):
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    t0 = time.time()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        list(range(45)),
        max_batch=1,
        max_seq_len=MAX_SEQ,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    logger.info(f"[prof] model built in {time.time() - t0:.0f} s")
    ids = [int(v) for v in Path(IDS).read_text().replace("\n", ",").split(",") if v.strip()]
    if N:
        ids = ids[:N]
    prompt = torch.tensor(ids, dtype=torch.int64)
    prof = Prof(mesh_device)
    _instrument(model, prof)
    f = open(OUT, "a")
    full = prompt
    for p, spec in enumerate(RUNS):
        chunk, instr, marks = int(spec[0]), spec[1] in ("1", "2"), spec[1] == "1"
        prompt = full[: int(spec[2])] if len(spec) > 2 and spec[2] else full
        if len(spec) > 3 and spec[3]:  # regions sub-chunk (tt/moe.py REGIONS_T is read at call time)
            moe_mod.REGIONS_T = int(spec[3])
        if len(spec) > 4 and spec[4]:  # regions rank mode (cumsum | mm, G53F-0174)
            moe_mod.REGIONS_RANK = spec[4]
        if len(spec) > 5 and spec[5]:  # regions buffer rows (full | fit, G53F-0183)
            moe_mod.REGIONS_BUF = spec[5]
        if len(spec) > 6 and spec[6]:  # regions row gathers (embedding | row, G53F-0186)
            moe_mod.REGIONS_GATHER = spec[6]
        if len(spec) > 8 and spec[8]:  # regions combine (reduce | fused, G53F-0190)
            moe_mod.REGIONS_COMBINE = spec[8]
        if len(spec) > 7 and spec[7]:  # prefill mHC rows (wrap | fast | trace, G53F-0187 / 0188; tt/mhc.py)
            from models.demos.glm53_flash.tt import mhc as mhc_mod

            if spec[7] in ("wrap", "trace"):  # the composite needs the TtMHCSplit build (G53F-0188 default: fast only)
                assert mhc_mod.MHC_PREFILL_IMPL == "wrap", "mhc mode wrap/trace needs GLM53_MHC_PREFILL_IMPL=wrap"
            mhc_mod.MHC_PREFILL_RUNTIME = "" if spec[7] in ("wrap", "trace") else spec[7]
            os.environ[mhc_mod.MHC_PREFILL_TRACE_ENV] = "1" if spec[7] == "trace" else "0"
        logger.info(
            f"[prof] run {p}: chunk {chunk} instr {spec[1]} n {int(prompt.numel())} regions_t {moe_mod.REGIONS_T}"
        )
        prof.on = instr
        _set_dsa_timer(model, prof, marks)
        rows = []

        def on_chunk(s0, e, secs, _rows=rows):
            row = dict(
                kind="chunk",
                run=p,
                chunk=chunk,
                instr=int(spec[1]),
                regions_t=moe_mod.REGIONS_T,
                start=s0,
                end=e,
                seconds=round(secs, 4),
            )
            if instr:
                row["parts"] = {k: round(v, 4) for k, v in prof.take().items()}
            _rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            logger.info(f"[prof] run {p} chunk {s0}-{e}: {secs:.3f} s")

        ttnn.synchronize_device(mesh_device)
        tw = time.perf_counter()
        try:
            out = model.prefill_chunked(prompt, user=0, chunk=chunk, on_chunk=on_chunk)
        except Exception as ex:  # noqa: BLE001 -- a geometry that does not fit is a result
            logger.warning(f"[prof] run {p} chunk {chunk} failed: {ex!r}")
            f.write(json.dumps(dict(kind="fail", run=p, chunk=chunk, error=repr(ex)[:400])) + "\n")
            f.flush()
            continue
        ttnn.synchronize_device(mesh_device)
        wall = time.perf_counter() - tw
        first = model.first_token(out[1], owner_row=out[2])
        if DUMP:  # the last chunk's last-token logits (owner row's first chip) for a bitwise A/B across processes
            lg = ttnn.to_torch(model.mesh.device_tensors(out[1])[out[2] * model.mesh.cols])
            torch.save(lg, f"{DUMP}_run{p}_n{int(prompt.numel())}.pt")
        ttnn.deallocate(out[0])
        if out[1] is not None:
            ttnn.deallocate(out[1])
        warm = [r for r in rows[1:-1]] or rows
        summ = dict(
            kind="summary",
            run=p,
            chunk=chunk,
            instr=int(spec[1]),
            regions_t=moe_mod.REGIONS_T,
            rank=moe_mod.REGIONS_RANK,
            buf=moe_mod.REGIONS_BUF,
            gather=moe_mod.REGIONS_GATHER,
            mhc=__import__("models.demos.glm53_flash.tt.mhc", fromlist=["x"]).MHC_PREFILL_RUNTIME
            or ("trace" if os.environ.get("GLM53_MHC_PREFILL_TRACE", "0") not in ("", "0") else "wrap"),
            n=int(prompt.numel()),
            wall_s=round(wall, 3),
            tok_s=round(int(prompt.numel()) / wall, 1),
            first_token=first,
            warm_chunk_s=round(sum(r["seconds"] for r in warm) / len(warm), 4),
            **_dram_view(mesh_device),
            regions=dict(moe_mod.REGIONS_STATS),  # what the regions calls ran (buffer rows, combine path)
        )
        moe_mod.REGIONS_STATS.clear()
        if instr:
            tot: dict[str, float] = defaultdict(float)
            for r in warm:
                for k, v in r["parts"].items():
                    tot[k] += v / len(warm)
            summ["warm_parts"] = {k: round(v, 4) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}
        f.write(json.dumps(summ) + "\n")
        f.flush()
        logger.info(f"[prof] summary {json.dumps(summ)}")
    if EXPORT_AB:
        prof.on = False
        _set_dsa_timer(model, prof, False)
        _export_ab(model, full, f)
    if EXPORT_ONLY_N:
        _export_only(model, f)
    if OPS_AB:
        prof.on = False
        _set_dsa_timer(model, prof, False)
        for spec_ in os.environ.get("PROF_OPS_RT", "4096").split(","):
            rt, _, rk = spec_.partition(":")
            if rk:
                moe_mod.REGIONS_RANK = rk
            moe_mod.REGIONS_T = int(rt)  # the cached triangular matrix is sized by REGIONS_T
            _regions_ops(model, mesh_device, full, f, int(rt))
    if TRACE_AB:
        prof.on = False  # no synchronizations inside a trace capture
        _set_dsa_timer(model, prof, False)
        _trace_ab(model, mesh_device, full, f)
    f.close()


def _dram_view(dev) -> dict:
    """Per-bank DRAM allocator state in MB after a pass (the seed server's reply fields): the headroom a lever that holds
    device memory (G53F-0188's mHC traces) leaves at the 1M geometry."""
    try:
        v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        mb = lambda b: round(b / 1e6, 1)
        return dict(
            dram_alloc_mb=mb(v.total_bytes_allocated_per_bank),
            dram_free_mb=mb(v.total_bytes_free_per_bank),
            dram_largest_free_mb=mb(v.largest_contiguous_bytes_free_per_bank),
        )
    except Exception as e:  # noqa: BLE001 -- diagnosis only
        return dict(dram_view_error=repr(e))


def _trace_ab(model, dev, prompt: torch.Tensor, f):
    """Eager vs captured-trace replay of one layer's FFN (``PROF_TRACE_AB`` = decoder index) on a real 16 K chunk's
    input: replay = device time only, so eager - replay = the host dispatch cost of that call."""
    k = TRACE_AB
    mlp = model.layers[k].layer.mlp
    kw = dict(prefill=True) if model.layers[k].layer.is_moe else {}
    grab = {}
    fwd = mlp.forward

    def keep(x, *a, **kk):
        if "x" not in grab:
            grab["x"] = ttnn.clone(x)
        return fwd(x, *a, **kk)

    mlp.forward = keep
    out = model.prefill(prompt[:TRACE_T], user=0, start=0)
    for t in out[:2]:
        if t is not None:
            ttnn.deallocate(t)
    mlp.forward = fwd
    x = grab["x"]

    def timed(fn, n=5):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(n):
            fn()
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t) / n

    def eager():
        ttnn.deallocate(fwd(x, **kw))

    for _ in range(2):
        eager()
    t_eager = timed(eager)
    y_ref = fwd(x, **kw)
    ref = ttnn.to_torch(model.mesh.device_tensors(y_ref)[0])
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    y_tr = fwd(x, **kw)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    t_trace = timed(lambda: ttnn.execute_trace(dev, tid, cq_id=0, blocking=False))
    got = ttnn.to_torch(model.mesh.device_tensors(y_tr)[0])
    ttnn.release_trace(dev, tid)
    row = dict(
        kind="trace_ab",
        layer=k,
        T_loc=int(x.shape[-2]),
        eager_s=round(t_eager, 5),
        trace_s=round(t_trace, 5),
        host_share=round(1 - t_trace / t_eager, 3),
        bitwise=bool(torch.equal(ref, got)),
    )
    f.write(json.dumps(row) + "\n")
    f.flush()
    logger.info(f"[prof] trace_ab {json.dumps(row)}")


def _regions_ops(model, dev, prompt: torch.Tensor, f, regions_t: int):
    """Op-by-op wall time (synchronized after every op; the MoE is device-bound, G53F-0173 trace A/B) of decoder
    ``PROF_OPS``'s routed-expert call on a real 16 K chunk's routed inputs: a transcription of
    ``TtRoutedExperts.forward`` (row gathers) -> ``_forward_regions_chunked`` -> ``_forward_regions`` -> reduce-scatter.
    Rows: ``kind=regions_ops`` with seconds per op name summed over the sub-chunks."""
    k = OPS_AB
    ex = model.layers[k].layer.mlp.experts
    grab = {}
    fwd = ex.forward

    def keep(x, dense_w, idx=None, w=None):
        if "x" not in grab:
            grab.update(x=ttnn.clone(x), dense_w=ttnn.clone(dense_w), idx=ttnn.clone(idx), w=ttnn.clone(w))
        return fwd(x, dense_w, idx=idx, w=w)

    ex.forward = keep
    out = model.prefill(prompt[:TRACE_T], user=0, start=0)
    for t in out[:2]:
        if t is not None:
            ttnn.deallocate(t)
    ex.forward = fwd
    acc: dict[str, float] = defaultdict(float)
    last = [0.0]

    def tick(name):
        ttnn.synchronize_device(dev)
        now = time.perf_counter()
        acc[name] += now - last[0]
        last[0] = now

    TILE = 32
    mesh = ex.mesh
    ttnn.synchronize_device(dev)
    t_all = last[0] = time.perf_counter()
    x = mesh.all_gather_rows(grab["x"], dim=2)
    tick("ag_x")
    dense_w = mesh.all_gather_rows(grab["dense_w"], dim=2)
    idx = mesh.all_gather_rows(grab["idx"], dim=2)
    w = mesh.all_gather_rows(grab["w"], dim=2)
    tick("ag_routing")
    d = ex._dispatch
    E, E_loc, H = ex.E, ex.E_local, ex.H
    T_all = x.shape[-2]
    outs = []
    for t0 in range(0, T_all, regions_t):
        t1 = min(t0 + regions_t, T_all)
        xs, dws, ids, ws = [ttnn.slice(t, [0, 0, t0, 0], [1, 1, t1, t.shape[-1]]) for t in (x, dense_w, idx, w)]
        tick("slice_in")
        kk, T = ids.shape[-1], xs.shape[-2]
        onehot = ttnn.multiply(ttnn.typecast(ttnn.ne(dws, 0.0), ttnn.float32), d["mask"])
        tick("onehot")
        if t0 == 0 and not grab.get("rank_checked"):  # G53F-0174: matmul ranks vs ttnn.cumsum on every chip, bitwise
            grab["rank_checked"] = True
            r_c = ttnn.cumsum(onehot, dim=-2)
            mode = moe_mod.REGIONS_RANK
            moe_mod.REGIONS_RANK = "mm"
            r_m = ex._rank(onehot)
            moe_mod.REGIONS_RANK = mode
            same = [
                torch.equal(ttnn.to_torch(a_), ttnn.to_torch(b_))
                for a_, b_ in zip(mesh.device_tensors(r_c), mesh.device_tensors(r_m))
            ]
            f.write(json.dumps(dict(kind="rank_check", T=int(T), chips=len(same), bitwise_chips=int(sum(same)))) + "\n")
            f.flush()
            logger.info(f"[prof] rank_check T={T}: {sum(same)}/{len(same)} chips bitwise")
            ttnn.deallocate(r_c)
            ttnn.deallocate(r_m)
            for Ts in [int(v) for v in os.environ.get("PROF_RANK_TS", "").split(",") if v.strip() and int(v) < T]:
                # (Ts == T would slice to the same tensor and the deallocate below would free onehot)
                oh_s = ttnn.slice(onehot, [0, 0, 0, 0], [1, 1, Ts, onehot.shape[-1]])
                r_c = ttnn.cumsum(oh_s, dim=-2)
                moe_mod.REGIONS_RANK = "mm"
                r_m = ex._rank(oh_s)
                moe_mod.REGIONS_RANK = mode
                cc, mm_ = [ttnn.to_torch(t_) for t_ in mesh.device_tensors(r_c)], [
                    ttnn.to_torch(t_) for t_ in mesh.device_tensors(r_m)
                ]
                same = [torch.equal(a_, b_) for a_, b_ in zip(cc, mm_)]
                ref = torch.cumsum(ttnn.to_torch(mesh.device_tensors(oh_s)[0]).double(), dim=-2)
                row = dict(
                    kind="rank_check",
                    T=Ts,
                    chips=len(same),
                    bitwise_chips=int(sum(same)),
                    cumsum_vs_torch=float((cc[0].double() - ref).abs().max()),
                    mm_vs_torch=float((mm_[0].double() - ref).abs().max()),
                )
                f.write(json.dumps(row) + "\n")
                f.flush()
                logger.info(f"[prof] rank_check {json.dumps(row)}")
                for t_ in (oh_s, r_c, r_m):
                    ttnn.deallocate(t_)
            tick("rank_check")
        rank = ex._rank(onehot)
        tick("rank")
        if moe_mod.REGIONS_RANK == "mm":
            counts_f = ttnn.slice(rank, [0, 0, T - 1, 0], [1, 1, T, onehot.shape[-1]])
        else:
            counts_f = ttnn.sum(onehot, dim=-2, keepdim=True)
        tick("counts")
        aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(counts_f, TILE - 1), 1.0 / TILE)), float(TILE))
        incl = ttnn.cumsum(aligned, dim=-1)
        offsets_f = ttnn.subtract(incl, aligned)
        tick("offsets")
        max_buf = ex.dispatch_max_tokens or (-(-(T * kk + E_loc * TILE) // TILE) * TILE)
        pos_te = ttnn.add(
            ttnn.multiply(ttnn.add(ttnn.subtract(rank, 1.0), offsets_f), onehot),
            ttnn.multiply(ttnn.rsub(onehot, 1.0), float(max_buf - 1)),
        )
        tick("pos_te")
        for t in (rank, aligned, incl):
            ttnn.deallocate(t)
        idx_u32 = ttnn.typecast(ids, ttnn.uint32)
        pos_tk = ttnn.gather(pos_te, -1, idx_u32)
        loc_tk = ttnn.gather(onehot, -1, idx_u32)
        tick("gather_pick")
        ttnn.deallocate(pos_te)
        ttnn.deallocate(onehot)
        ttnn.deallocate(idx_u32)
        pos_rm = ttnn.reshape(ttnn.to_layout(ttnn.typecast(pos_tk, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, T * kk])
        ttnn.deallocate(pos_tk)
        tick("pos_rm")
        slot_src = ttnn.scatter(ex._zero_u32_row(max_buf), -1, pos_rm, ex._iota_rows(T, kk))
        tick("scatter_slots")
        x_rm = ttnn.reshape(ttnn.to_layout(xs, ttnn.ROW_MAJOR_LAYOUT), [T, H])
        tick("x_to_rm")
        buf = ttnn.embedding(slot_src, x_rm, layout=ttnn.ROW_MAJOR_LAYOUT)
        tick("embed_regions")
        ttnn.deallocate(slot_src)
        ttnn.deallocate(x_rm)
        counts = ttnn.reshape(ttnn.to_layout(ttnn.typecast(counts_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
        offsets = ttnn.reshape(ttnn.to_layout(ttnn.typecast(offsets_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
        tick("counts_offsets_rm")
        ttnn.deallocate(counts_f)
        ttnn.deallocate(offsets_f)
        o = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            ttnn.reshape(buf, [max_buf, H]),
            offsets,
            counts,
            d["gidx_rm"],
            d["gate"],
            d["up"],
            d["down"],
            ex.dispatch_max_tokens or T,
            compute_kernel_config=ex.compute,
            activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        )
        tick("expert_kernel")
        ttnn.deallocate(buf)
        out16 = ttnn.typecast(o, ttnn.bfloat16)
        ttnn.deallocate(o)
        tick("out_typecast")
        y_rm = ttnn.to_layout(out16, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(out16)
        tick("out_to_rm")
        comb = ttnn.embedding(pos_rm, y_rm, layout=ttnn.ROW_MAJOR_LAYOUT)
        tick("embed_combine")
        ttnn.deallocate(y_rm)
        ttnn.deallocate(pos_rm)
        comb5 = ttnn.reshape(comb, [1, 1, T, kk, H])
        w_loc = ttnn.multiply(ws, loc_tk)
        ttnn.deallocate(loc_tk)
        w16 = ttnn.typecast(w_loc, ttnn.bfloat16)
        w_rm = ttnn.unsqueeze(ttnn.to_layout(w16, ttnn.ROW_MAJOR_LAYOUT), -1)
        ttnn.deallocate(w_loc)
        ttnn.deallocate(w16)
        tick("weights_rm")
        y = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb5, w_rm, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tick("post_combine_reduce")
        for t in (comb, w_rm, counts, offsets):
            ttnn.deallocate(t)
        if y.dtype != ex.activation_dtype:
            y = ttnn.typecast(y, ex.activation_dtype)
        tick("y_typecast")
        outs.append(y)
        for t in (xs, dws, ids, ws):
            ttnn.deallocate(t)
    y = ttnn.concat(outs, dim=-2) if len(outs) > 1 else outs[0]
    tick("concat")
    y = mesh.reduce_scatter_rows(y, dim=2)
    tick("reduce_scatter")
    total = time.perf_counter() - t_all
    row = dict(
        kind="regions_ops",
        layer=k,
        regions_t=regions_t,
        rank=moe_mod.REGIONS_RANK,
        T=int(T_all),
        total_s=round(total, 4),
        ops={n: round(v, 4) for n, v in sorted(acc.items(), key=lambda kv: -kv[1])},
    )
    f.write(json.dumps(row) + "\n")
    f.flush()
    logger.info(f"[prof] regions_ops {json.dumps(row)}")


def _export_only(model, f):
    """G53F-0182: time the export of the FIRST PROF_EXPORT_ONLY_N rows of the caches the last pass left (no
    prefill), one-chip read vs mesh_partition + device untilize (GLM53F_EXPORT_PARTITION), with all-layer digests."""
    import hashlib

    from models.demos.glm53_flash.tests.test_prefill_vs_blaze_golden import _dsa_state_host, _kda_state_host

    n = EXPORT_ONLY_N
    res = {}
    for mode in ("0", "1", "0"):
        os.environ["GLM53F_EXPORT_PARTITION"] = mode
        ttnn.synchronize_device(model.mesh.device)
        t0 = time.perf_counter()
        digests, dsa_s = [], 0.0
        for dec in model.layers:
            t1 = time.perf_counter()
            st = _dsa_state_host(model, dec, n) if dec.cache is not None else _kda_state_host(model, dec)
            if dec.cache is not None:
                dsa_s += time.perf_counter() - t1
            h = hashlib.sha256()
            for key in sorted(st):
                h.update(key.encode())
                h.update(st[key].contiguous().view(torch.uint8).numpy().tobytes())
            digests.append(h.hexdigest()[:16])
        wall = time.perf_counter() - t0
        ref = res.setdefault("ref", digests)
        row = dict(
            kind="export_only",
            partition=mode,
            n=n,
            wall_s=round(wall, 3),
            dsa_s=round(dsa_s, 3),
            layers_equal=sum(a == b for a, b in zip(digests, ref)),
            layers=len(digests),
        )
        f.write(json.dumps(row) + "\n")
        f.flush()
        logger.info(f"[prof] export_only {json.dumps(row)}")
    os.environ.pop("GLM53F_EXPORT_PARTITION", None)


def _export_ab(model, prompt: torch.Tensor, f):
    import hashlib

    from models.demos.glm53_flash.tests.test_prefill_vs_blaze_golden import _dsa_state_host, _kda_state_host

    digests = {}
    for spec_ in EXPORT_AB.split(","):
        # "rt[:rank[:buf[:export]]]": REGIONS_T, the rank mode, the region-buffer rows (full | fit, G53F-0183) and the
        # export read (one = chip 0's replica | part = mesh_partition + one mesh read, GLM53F_EXPORT_PARTITION, G53F-0182)
        fields = spec_.split(":")
        moe_mod.REGIONS_T = int(fields[0])
        if len(fields) > 1 and fields[1]:
            moe_mod.REGIONS_RANK = fields[1]
        if len(fields) > 2 and fields[2]:
            moe_mod.REGIONS_BUF = fields[2]
        if len(fields) > 3 and fields[3]:
            os.environ["GLM53F_EXPORT_PARTITION"] = "1" if fields[3] == "part" else "0"
        n = EXPORT_N
        out = model.prefill(prompt[:n], user=0, start=0)
        first = model.first_token(out[1], owner_row=out[2])
        for t in out[:2]:
            if t is not None:
                ttnn.deallocate(t)
        per_layer = []
        ttnn.synchronize_device(model.mesh.device)
        t_exp = time.perf_counter()
        for dec in model.layers:
            st = _dsa_state_host(model, dec, n) if dec.cache is not None else _kda_state_host(model, dec)
            h = hashlib.sha256()
            for key in sorted(st):
                h.update(key.encode())
                h.update(st[key].contiguous().view(torch.uint8).numpy().tobytes())
            per_layer.append(h.hexdigest()[:16])
        export_s = time.perf_counter() - t_exp
        digests[spec_] = (first, per_layer)
        logger.info(f"[prof] export {spec_}: export + hash {export_s:.3f} s")
        logger.info(f"[prof] export {spec_}: first {first}, layer digests {per_layer[:3]}...")
    ref_key = next(iter(digests))
    ref = digests[ref_key]
    for key, (first, per_layer) in digests.items():
        row = dict(
            kind="export_ab",
            setting=key,
            ref=ref_key,
            n=EXPORT_N,
            first_token=first,
            first_equal=first == ref[0],
            layers_equal=sum(a == b for a, b in zip(per_layer, ref[1])),
            layers=len(per_layer),
            digests=per_layer,  # all 45 (cross-process gates compare these, G53F-0187)
        )
        f.write(json.dumps(row) + "\n")
        f.flush()
        logger.info(f"[prof] export_ab {json.dumps(row)}")
