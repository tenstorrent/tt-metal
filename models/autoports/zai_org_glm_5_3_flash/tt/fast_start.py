# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 12 fast start: weight-source switch, startup timeline, warm-up instrumentation and parallel capture.

* `build_model_backend_fast(model_dir, mesh, **kw)`: drop-in for `tt/model.py::build_model_backend` that honours
  GLM53_WEIGHT_SOURCE=image|tensors (default tensors). "image" = `tt/weight_image.py` (DRAM-image pull + allocation
  replay) with fallback to the tensor loader on any mismatch.
* `build_generator(model_dir, mesh, **kw)`: the vLLM generator factory (GLM53_VLLM_GENERATOR_FACTORY=
  "models.autoports.zai_org_glm_5_3_flash.tt.fast_start:build_generator"): same contract as
  `tt/generator.py::build_generator`, backend from `build_model_backend_fast`, startup timeline printed.
* `Timeline` / `instrument_warmup()`: per-phase wall time, JIT compile / cache-hit counts (tt-metal build telemetry via
  `ttnn._ttnn.dram_image.jit_stats`), program-cache entries per chip, trace-capture time per chip; printed as
  `[fast-start] timeline ...` lines and written to $GLM53_TIMELINE_PATH (JSON) when set.
* GLM53_PARALLEL_CAPTURE=1: the prefill / decode trace captures run one thread per chip (each chip's captures only
  touch that chip's command queue; ttnn ops release the GIL in C++).
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Optional

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.weight_image import (
    ImageMismatch,
    chip_of,
    code_identity,
    find_image,
    log,
    replay_load,
    weights_identity,
)

WEIGHT_SOURCE_ENV = "GLM53_WEIGHT_SOURCE"
GEN_KEYS = ("max_batch", "max_seq_len", "host_sampling", "stop_token_ids", "sampler_seed")


# ================================================================================================ timeline
class Timeline:
    """Startup phases (wall seconds since the timeline started) + per-phase JIT / program-cache counters."""

    def __init__(self, name: str = "startup"):
        self.name = name
        self.t0 = time.perf_counter()
        self.utc0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.phases: list = []
        self.info: dict = {}
        self.lock = threading.Lock()
        self._depth = threading.local()

    @contextmanager
    def phase(self, name: str, meshes: Optional[dict] = None, **extra):
        j0 = jit_stats()
        p0 = program_cache_entries(meshes) if meshes else None
        t = time.perf_counter()
        row = {"phase": name, "start_s": round(t - self.t0, 3), **extra}
        stack = getattr(self._depth, "stack", None)
        if stack is None:
            stack = self._depth.stack = []
        if stack:
            row["parent"] = stack[-1]
            # inside a phase that is itself counted in a bucket: not counted again
            row["nested"] = any(self.BUCKETS.get(x) for x in stack)
        stack.append(name)
        try:
            yield row
        finally:
            stack.pop()
            row["seconds"] = round(time.perf_counter() - t, 3)
            j1 = jit_stats()
            row["jit_compiled"] = j1["compiled"] - j0["compiled"]
            row["jit_cache_hits"] = j1["cache_hits"] - j0["cache_hits"]
            if meshes:
                p1 = program_cache_entries(meshes)
                row["program_cache_new"] = {c: p1[c] - p0.get(c, 0) for c in p1}
            with self.lock:
                self.phases.append(row)
            log(
                f"timeline {self.name} {name}: {row['seconds']:.2f} s (jit compiled {row['jit_compiled']}, "
                f"cache hits {row['jit_cache_hits']})"
            )
            self.write()

    # phase name -> Stage 9 gauge bucket (glm53_startup_seconds{phase})
    BUCKETS = {
        "weights_image": "load",
        "weights_tensors": "load",
        "image_identity": "load",
        "allocate_kv_cache": "kv_cache",
        "allocate_decode_io": "kv_cache",
        "prepare_prefill": "warmup",
        "prepare_decode": "warmup",
        "capture_decode": "capture",
        "capture_prefill": "capture",
        "capture_sampling": "capture",
    }

    def summary(self) -> dict:
        totals: dict = {}
        for p in self.phases:
            b = self.BUCKETS.get(p["phase"])
            if b and not p.get("nested"):
                totals[b] = round(totals.get(b, 0.0) + p["seconds"], 3)
        return {
            "name": self.name,
            "utc0": self.utc0,
            "total_s": round(time.perf_counter() - self.t0, 3),
            **{f"{k}_s": v for k, v in totals.items()},
            "jit_compiled": sum(p.get("jit_compiled", 0) for p in self.phases if not p.get("nested")),
            "phases": self.phases,
            **self.info,
        }

    def write(self, path: Optional[str] = None) -> None:
        """Rewrite $GLM53_TIMELINE_PATH (one JSON object, the current summary) atomically."""
        path = path or os.environ.get("GLM53_TIMELINE_PATH")
        if not path:
            return
        try:
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(self.summary(), f, default=str)
            os.replace(tmp, path)
        except OSError as e:
            log(f"timeline write to {path} failed: {e}")

    def dump(self, path: Optional[str] = None) -> None:
        """Log the summary line and append it to `path` (JSONL history; default: none)."""
        s = self.summary()
        log(
            f"timeline {self.name} total {s['total_s']:.2f} s: "
            + ", ".join(f"{p['phase']} {p['seconds']:.1f}" for p in self.phases)
        )
        if path:
            with open(path, "a") as f:
                f.write(json.dumps(s, default=str) + "\n")


def jit_stats() -> dict:
    try:
        return dict(ttnn._ttnn.dram_image.jit_stats())
    except Exception:  # noqa: BLE001  (older build without the binding)
        return {"compiled": 0, "cache_hits": 0, "srcs": 0}


def program_cache_entries(meshes: dict) -> dict:
    out = {}
    for c, m in meshes.items():
        try:
            out[c] = int(m.num_program_cache_entries())
        except Exception:  # noqa: BLE001
            out[c] = -1
    return out


TIMELINE: Optional[Timeline] = None


def timeline() -> Timeline:
    global TIMELINE
    if TIMELINE is None:
        TIMELINE = Timeline()
    return TIMELINE


# ================================================================================================ backend
def stage_submeshes(mesh, stages) -> dict:
    """coord -> 1x1 submesh, created in the order MultichipDecoder.__init__ would create them (one per distinct coord,
    stage order), so the normal and the image load see identical submesh / allocator histories."""
    subs: dict = {}
    for p in stages:
        c = tuple(p.coord)
        if c not in subs:
            subs[c] = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(*c))
    return subs


def _teardown(be) -> None:
    try:
        be.m.close_links()
    except Exception:  # noqa: BLE001
        pass


def build_model_backend_fast(model_dir, mesh_device, **kw):
    """`tt/model.py::build_model_backend` with GLM53_WEIGHT_SOURCE=image|tensors (see module docstring). Extra kw:
    weight_source (overrides the env), record (a weight_image.WeightIO("record") to log the tensor load into)."""
    from models.autoports.zai_org_glm_5_3_flash.tt import model as M

    tl = timeline()
    source = kw.pop("weight_source", None) or os.environ.get(WEIGHT_SOURCE_ENV, "tensors")
    record = kw.pop("record", None)
    if source not in ("image", "tensors"):
        raise ValueError(f"{WEIGHT_SOURCE_ENV}={source!r}: expected image | tensors")
    stages = kw.pop("stages", None) or M.plan_stages(
        mesh_device,
        partition=kw.pop("partition", None),
        partition_key=kw.pop("partition_key", None),
        units=kw.pop("units", None),
    )
    subs = kw.pop("submeshes", None) or stage_submeshes(mesh_device, stages)
    chips = {chip_of(s): s for s in subs.values()}
    tl.info["chips"] = sorted(chips)
    if source == "image":
        try:
            with tl.phase("image_identity") as row:
                code, weights = code_identity(), weights_identity()
                forced = os.environ.get("GLM53_IMAGE_DIR")
                if forced:  # TEST ONLY: an image built by another code id; the allocation-map / hash checks still apply
                    with open(os.path.join(forced, "manifest.json")) as f:
                        found = (forced, json.load(f))
                    log(f"GLM53_IMAGE_DIR={forced}: code identity check SKIPPED (image code {found[1]['code']['id']})")
                    row["forced_image_dir"] = forced
                else:
                    found = find_image(code, weights)
                row.update(code=code["id"], weights=weights["id"])
                if found is None:
                    raise ImageMismatch(f"no image for {code['id']}-{weights['id']}")
            d, man = found
            sub = {}
            with tl.phase("weights_image", chips) as row:
                be = replay_load(
                    lambda: M.build_model_backend(
                        model_dir, mesh_device, stages=stages, submeshes=subs, prefetch=False, **kw
                    ),
                    d,
                    man,
                    chips,
                    teardown=_teardown,
                    timeline=sub,
                )
                row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sub.items() if k != "pull"})
                row["pull"] = {
                    str(c): {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
                    for c, s in sub.get("pull", {}).items()
                    if not str(c).startswith("_")
                }
                row["image_dir"] = d
            be.weight_source = "image"
            return be
        except ImageMismatch as e:
            log(f"image unusable ({e}); falling back to the tensor loader")
            tl.info["image_fallback"] = str(e)[:500]
    with tl.phase("weights_tensors", chips):
        if record is not None:
            with record:
                be = M.build_model_backend(model_dir, mesh_device, stages=stages, submeshes=subs, **kw)
        else:
            be = M.build_model_backend(model_dir, mesh_device, stages=stages, submeshes=subs, **kw)
    be.weight_source = "tensors"
    return be


def build_generator(model_dir, mesh_device, **kwargs):
    """vLLM generator factory (GLM53_VLLM_GENERATOR_FACTORY): tt/generator.py::build_generator with the backend built
    by build_model_backend_fast and the warm-up instrumented (startup timeline)."""
    from models.autoports.zai_org_glm_5_3_flash.tt import generator as G

    tl = timeline()
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import _check_device_mesh

        _check_device_mesh(mesh_device)  # vLLM skips this check when a factory override is set
    except ImportError:
        pass
    gen_kw = {k: kwargs.pop(k) for k in GEN_KEYS + ("tokenizer",) if k in kwargs}
    backend = kwargs.pop("backend", None)
    if backend is None:  # every other kwarg (decoder=, ocfg=, units=, ...) belongs to build_model_backend
        backend = build_model_backend_fast(model_dir, mesh_device, **kwargs)
    gen = G.build_generator(model_dir, mesh_device, backend=backend, **gen_kw)
    instrument_warmup(gen)
    tl.info["weight_source"] = getattr(backend, "weight_source", None)
    tl.write()
    return gen


# ================================================================================================ warm-up
_WRAPPED = set()


def _wrap(obj, name: str, phase: str, meshes_of=None):
    """Wrap obj.name (bound on the instance) in a timeline phase (once per object)."""
    key = (id(obj), name)
    if key in _WRAPPED or not hasattr(obj, name):
        return
    _WRAPPED.add(key)
    fn = getattr(obj, name)

    def wrapped(*a, **k):
        with timeline().phase(phase, meshes_of() if meshes_of else None):
            return fn(*a, **k)

    setattr(obj, name, wrapped)


def chip_meshes(be) -> dict:
    out = {}
    for st in be.m.stages:
        out.setdefault(chip_of(st.mesh), st.mesh)
    return out


def enable_parallel_prepare(be) -> None:
    """Glm53PipelineBackend.prepare_prefill with one host thread per chip: each thread issues exactly the op sequence the
    serial pass issues on its chip (per length: [chip of stage 0: embedding] -> for each of its stages: [recv from the
    previous stage] -> the stage's units -> [send to the next stage | last stage: heads]), so the programs and the
    per-chip command order are the serial pass's; only the host dispatch of different chips overlaps (ttnn ops release
    the GIL; a socket recv blocks the device, not the host). Then per chip: padded-chunk masks, samplers (chip 0), the
    KDA scratch-state reads; reset + synchronize in the main thread."""
    import torch

    from models.autoports.zai_org_glm_5_3_flash.tt import model as M

    m = be.m
    if getattr(be, "_parallel_prepare", False) or m.handoff == "host" or len(m.stages) < 2:
        return
    be._parallel_prepare = True
    be._prepare_prefill_serial = be.prepare_prefill

    def prepare_prefill_parallel(kv_cache):
        g = kv_cache.group
        lens = sorted({b for b in m.buckets() if b % M.MODEL_ALIGN == 0} | {20, 40})
        pad = torch.zeros(max(lens), dtype=torch.int64)
        first = m.stages[0]
        cov = be._covered_tokens(0, g)
        plans = {}
        for n in lens:  # _run_prefill's chunk plan (slot 0, start 0)
            plan = M.prefill_chunk_plan(n, m.buckets(), be.chunk_policy)
            if any(off + phys > cov for off, _, phys in plan):
                plan = M.model_chunk_plan(n, m.buckets())
            plans[n] = plan
        stages_of: dict = {}
        for k, st in enumerate(m.stages):
            stages_of.setdefault(chip_of(st.mesh), []).append(k)
        head_chip = chip_of(first.mesh)
        last = len(m.stages) - 1

        def heads(off, n, phys, y):  # = prepare_prefill's on_output
            h = be._select_row(y, phys, n - 1, 0)
            lg = be.term.logits_decode(h)
            ttnn.deallocate(h)
            ttnn.copy(lg, be.plogits)
            ttnn.deallocate(lg)
            lg = be.term.logits_prefill(y)
            ttnn.deallocate(lg)

        def chip_passes(ks):
            def fn():
                for n in lens:
                    for off, nn, phys in plans[n]:
                        for k in ks:
                            st = m.stages[k]
                            if k == 0:
                                be._embed_chunk(pad[off : off + nn], nn, phys, first.pbufs[phys][0], tail=nn < phys)
                            else:
                                st.link_in.recv(st.pbufs[phys][0])
                            y = m._stage_prefill_chunk(
                                k, g, phys=phys, n_valid=nn, chunk_start=off, user_slot=0, eager=True
                            )
                            if k < last:
                                st.link_out.send(st.pbufs[phys][len(st.plan.units) % 2])
                            else:
                                heads(off, nn, phys, y)

            return fn

        t0 = time.perf_counter()
        secs = _run_per_chip({c: chip_passes(ks) for c, ks in stages_of.items()}, "prepare_prefill")

        def chip_rest(chip, ks):
            def fn():
                if chip == head_chip:
                    for p in sorted(b for b in m.buckets() if b % M.MODEL_ALIGN == 0):
                        be._embed_chunk(pad[: p - 1], p - 1, p, first.pbufs[p][0], tail=True)
                st0 = m.stages[ks[0]]
                for p in st0.pmask:
                    if p in st0.pbufs:
                        m._mask_rows(st0, st0.pbufs[p][0], p, p - 1)
                if chip == head_chip:
                    be._prefill_sample(0, M.SamplingParams())
                    be._prefill_sample(0, M.SamplingParams(temperature=1.0, top_p=0.95))
                for k in ks:
                    st = m.stages[k]
                    gg = st.groups[g]
                    for l in st.state_layers:
                        dec, cache = st.decoders[l], gg.caches[l]
                        if dec.is_kda:
                            for sl in range(cache.max_batch):
                                dec._scratch_state_in(cache, sl)
                            ttnn.fill_cache(cache.kda.S, cache.kda.prefill_S, 0)

            return fn

        secs2 = _run_per_chip({c: chip_rest(c, ks) for c, ks in stages_of.items()}, "prepare_prefill_rest")
        m.reset_slot(0, group=g)
        m.synchronize()
        timeline().info["prepare_prefill_chip_s"] = {str(c): secs[c] + secs2.get(c, 0.0) for c in secs}
        log(
            f"parallel prepare_prefill: {time.perf_counter() - t0:.2f} s (per chip {timeline().info['prepare_prefill_chip_s']})"
        )

    be.prepare_prefill = prepare_prefill_parallel


def instrument_warmup(gen) -> None:
    """Time the warm-up phases of a generator over the pipeline backend; optionally parallel captures."""
    be = getattr(gen, "backend", None) or getattr(gen, "b", None)
    if be is None or not hasattr(be, "m"):
        return
    meshes = lambda: chip_meshes(be)  # noqa: E731
    if os.environ.get("GLM53_PARALLEL_CAPTURE", "0") == "1":
        enable_parallel_capture(be)  # before the timers wrap the (replaced) capture methods
    if os.environ.get("GLM53_PARALLEL_WARMUP", "0") == "1":
        enable_parallel_prepare(be)
    for name in (
        "allocate_kv_cache",
        "allocate_decode_io",
        "prepare_prefill",
        "prepare_decode",
        "capture_decode",
        "capture_sampling",
    ):
        _wrap(be, name, name, meshes)
    _wrap(be.m, "_capture_prefill", "capture_prefill", meshes)
    install_capture_probe()


# ---- per-chip capture time (wraps ttnn.begin/end_trace_capture process-wide; cheap)
CAPTURE = {"by_chip": {}, "count": {}}
_CAPTURE_LOCK = threading.Lock()
_PROBE = {"on": False}


def install_capture_probe() -> None:
    if _PROBE["on"]:
        return
    _PROBE["on"] = True
    begin, end = ttnn.begin_trace_capture, ttnn.end_trace_capture
    open_t: dict = {}

    def begin_trace_capture(mesh, *a, **k):
        tid = begin(mesh, *a, **k)
        open_t[(id(mesh), threading.get_ident())] = time.perf_counter()
        return tid

    def end_trace_capture(mesh, *a, **k):
        r = end(mesh, *a, **k)
        t0 = open_t.pop((id(mesh), threading.get_ident()), None)
        if t0 is not None:
            try:
                c = chip_of(mesh)
            except Exception:  # noqa: BLE001
                c = -1
            with _CAPTURE_LOCK:
                CAPTURE["by_chip"][c] = CAPTURE["by_chip"].get(c, 0.0) + time.perf_counter() - t0
                CAPTURE["count"][c] = CAPTURE["count"].get(c, 0) + 1
        return r

    ttnn.begin_trace_capture, ttnn.end_trace_capture = begin_trace_capture, end_trace_capture


# ---- parallel captures (one thread per chip)
def _run_per_chip(work: dict, what: str) -> dict:
    """work: chip -> callable. Runs each in its own thread; re-raises the first error after all joined."""
    errors, secs = [], {}

    def run(c, fn):
        t = time.perf_counter()
        try:
            fn()
        except BaseException as e:  # noqa: BLE001
            errors.append((c, e))
        secs[c] = round(time.perf_counter() - t, 3)

    th = [threading.Thread(target=run, args=kv, name=f"{what}-chip{kv[0]}") for kv in work.items()]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise RuntimeError(f"{what}: chip {errors[0][0]} failed") from errors[0][1]
    return secs


def enable_parallel_capture(be) -> None:
    """Replace the stack's prefill capture with a per-chip-thread version (same body per stage as
    MultichipDecoder._capture_prefill: one trace per (group, unit, bucket); a chip's stages run in order in its thread).
    """
    m = be.m
    if getattr(m, "_parallel_capture", False):
        return
    m._parallel_capture = True
    orig = m._capture_prefill

    def capture_prefill_parallel():
        by_chip: dict = {}
        for st in m.stages:
            by_chip.setdefault(chip_of(st.mesh), []).append(st)

        def chip_work(stages):
            def fn():
                for st in stages:
                    _capture_stage_prefill(m, st)

            return fn

        secs = _run_per_chip({c: chip_work(s) for c, s in by_chip.items()}, "capture_prefill")
        m.prefill_captured = True
        timeline().info.setdefault("capture_prefill_chip_s", {}).update({str(c): s for c, s in secs.items()})

    m._capture_prefill_serial = orig
    m._capture_prefill = capture_prefill_parallel

    if not hasattr(be, "captured_stack"):
        return
    from models.autoports.zai_org_glm_5_3_flash.tt.model import _Trace

    def capture_decode_parallel(io, kv_cache, width):
        """Glm53PipelineBackend.capture_decode with the per-stage decode captures run one thread per chip (a chip's
        stages -- chip 0 holds the head and the tail stage -- in pipeline order in its thread), then the prefill
        captures (parallel too)."""
        gi = kv_cache.group
        if not be.captured_stack:
            by_chip: dict = {}
            for k, st in enumerate(m.stages):
                by_chip.setdefault(chip_of(st.mesh), []).append(k)

            def chip_work(ks):
                def fn():
                    for k in ks:
                        st = m.stages[k]
                        g = st.groups[gi]
                        g.trace_id = ttnn.begin_trace_capture(st.mesh, cq_id=0)
                        try:
                            m._stage_decode(k, gi)
                        finally:
                            ttnn.end_trace_capture(st.mesh, g.trace_id, cq_id=0)

                return fn

            secs = _run_per_chip({c: chip_work(ks) for c, ks in by_chip.items()}, "capture_decode")
            timeline().info.setdefault("capture_decode_chip_s", {}).update({str(c): s for c, s in secs.items()})
            m.captured = True
            if m.traced_prefill:
                m._capture_prefill()
            be.captured_stack = True
        tr = _Trace("decode", width, [(st.mesh, st.groups[gi].trace_id) for st in m.stages])
        be.traces.append(tr)
        return tr

    be._capture_decode_serial = be.capture_decode
    be.capture_decode = capture_decode_parallel


def _capture_stage_prefill(m, st) -> None:
    """MultichipDecoder._capture_prefill's body for one stage (kept in step with tt/multichip_decoder.py)."""
    for g in st.groups:
        for i, u in enumerate(st.plan.units):
            dec = st.decoders[u.layer]
            cache = g.caches.get(u.layer)
            traces = m._unit_traces(st, g, i, u)
            if u.part == "full":
                dec.capture_prefill_traces(cache, traces)
                continue
            for t in traces.values():
                if cache is not None:
                    dec._prime_chunk(cache, t.bucket)
                tid = ttnn.begin_trace_capture(st.mesh, cq_id=0)
                try:
                    m._unit_body(dec, u, t, cache)
                except BaseException:
                    ttnn.end_trace_capture(st.mesh, tid, cq_id=0)
                    ttnn.release_trace(st.mesh, tid)
                    raise
                ttnn.end_trace_capture(st.mesh, tid, cq_id=0)
                t.trace_id = tid
            if cache is not None:
                dec.reset_slot(cache, 0)
    ttnn.synchronize_device(st.mesh)
