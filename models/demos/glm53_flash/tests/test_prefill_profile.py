# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the long-context chunked prefill spends its time (tt-blaze G53F-0172).

Builds the seed server's model (the same env: ``seed_server42.sh`` long geometry) and runs ``PROF_RUNS``
passes of ``prefill_chunked`` over the first ``PROF_N`` ids of ``PROF_IDS``. A pass is ``<chunk>:<instr>[:<n>[:<regions_t>]]``
(``n``: this pass uses only the first n ids; ``regions_t``: the MoE regions sub-chunk for this pass):
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
DUMP = os.environ.get("PROF_DUMP", "")  # path prefix: save each pass's last-token logits


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
            n=int(prompt.numel()),
            wall_s=round(wall, 3),
            tok_s=round(int(prompt.numel()) / wall, 1),
            first_token=first,
            warm_chunk_s=round(sum(r["seconds"] for r in warm) / len(warm), 4),
        )
        if instr:
            tot: dict[str, float] = defaultdict(float)
            for r in warm:
                for k, v in r["parts"].items():
                    tot[k] += v / len(warm)
            summ["warm_parts"] = {k: round(v, 4) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}
        f.write(json.dumps(summ) + "\n")
        f.flush()
        logger.info(f"[prof] summary {json.dumps(summ)}")
    if TRACE_AB:
        prof.on = False  # no synchronizations inside a trace capture
        _set_dsa_timer(model, prof, False)
        _trace_ab(model, mesh_device, full, f)
    f.close()


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
