# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 probe launcher: the stock prefill runner with checks added (tt_metal_tracker-q171.2.9, q171.2.10).

Use with ``STAGE_PROBE_MODEL=glm54 STAGE_PROBE_RUNNER_MODULE=models.demos.deepseek_v3_d_p.scripts.stage_probe.
trunk_probe_runner run_stage_probe.sh <name> <first_pass> <num_passes>``: all 106 layer passes on one rank (0 106),
or one stage alone (G3 = 82 24). With PREFILL_MTP_LEVELS set the last rank runs GLM-5.3's MTP after the stage.
Nothing here changes what the runner computes or when; every check only observes.

* Pass-2 push reuse (q171.3.8 review m1), on a stage that runs loop pass 2 (else ``N/A``). Every loop pass-2 ttMLA logs whether its push output reuses the dense
  gather buffer (``_pass2_push_reuses_dense``), every ``_pass2_push_output`` call records whether it returned that
  buffer and whether it ran inside trace capture, and an own ``loop_pass2`` push buffer being created is recorded.
  Before the trace is released one ``TRUNK_ASSERT push_reuse PASS|FAIL`` line states the verdict: PASS needs every
  pass-2 layer to choose reuse, the trace to be captured with every pass-2 push writing the dense buffer, and no
  own pass-2 buffer. A FAIL is logged, not raised, so the run's timing survives; the probe's checker reads it.
* KV digest (finite, deterministic). After the request, before the trace is released, user 0's KVPE rows of every
  trunk cache slot of the stage (78 on the whole trunk; a loop layer's pass-1 rows) are read for each chunk in
  ``TRUNK_DIGEST_CHUNKS`` (default the probe chunks 0,9,19,50,101,203), plus, on a stage that runs pass 2, the loop
  layers' pass-2 window ring (G2's own KV, the last chunks' pass 2), with ``read_padded_kv_cache`` (the G1->G2
  handoff's reader), and logged per chunk as ``TRUNK_KV_DIGEST chunk=<c> sha256=<hex> finite=<bool> nonzero_rows=<n>
  rows=<n>``. With MTP its K slots (past the trunk's) are digested per chunk as ``chunk=<c>mtp``: nonzero rows show
  MTP wrote its KV there. The cache starts zero-filled and two runs feed identical tokens, so equal digests across
  runs mean bit-identical KV; the last chunk's layer-77 rows depend on every pass of every chunk.
  ``TRUNK_DIGEST_SLOT_RANGES`` (e.g. ``0-53,54-76,77-77``) adds a digest per stage-slot range (``chunk=<c>s<lo>_<hi>``).
  ``TRUNK_KV_DUMP_DIR`` also saves those rows per cache slot as logical bf16 (``_kv_dump``), to compare runs by PCC.
* MTP wiring (q171.2.10), with MTP on. Every forward records the blocks it runs, the trunk norm's input and output, and
  the tensor MTP is seeded with. ``TRUNK_ASSERT mtp_input PASS|FAIL`` (logged with the checks above) needs, in every
  forward that ran MTP: MTP ran once, after every pass-schedule step of the stage (``model.steps``, in order: 106 on
  the whole trunk), with the last step the coda's last layer (L77, not a loop pass), and its seed the trunk norm's
  output of that step's output (h^0 = norm(trunk output), as GLM-5.3). ``TRUNK_MTP_LAYOUT`` logs the cache slots
  (trunk + MTP), the MTP slot and layer, the index-cache slots, and this rank's layer-ack ids.
"""

from __future__ import annotations

import hashlib
import os
import runpy

from loguru import logger

_PASS2 = {}  # layer -> (reuses_dense, push spec rows, dense rows)
_CALLS = {"capture": [], "other": []}  # (layer, returned the dense buffer)
_OWN_BUFFERS = []
_STATE = {"in_capture": False, "captured": False, "use_trace": None, "caches": None, "params": None, "runtime": None}
_FWD = {"blocks": [], "norm": None, "mtp_seeds": [], "results": []}  # per forward: MTP wiring observations


def _patch_mla() -> None:
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPlanner

    build = ttMLA._build_two_bank_index
    push_output = ttMLA._pass2_push_output
    push_buffer = SparseKvPushPlanner.push_buffer

    def build_two_bank_index(self):
        out = build(self)
        dense = self._sparse_kv_gather_buffer
        spec_rows = self._push.buffer_spec(dense, **self._pass2_push_geometry).shape[2]
        _PASS2[self.layer_idx] = (bool(self._pass2_push_reuses_dense), int(spec_rows), int(dense.shape[2]))
        logger.info(
            f"TRUNK_PUSH_REUSE layer={self.layer_idx} reuses_dense={self._pass2_push_reuses_dense} "
            f"spec_rows={spec_rows} dense_rows={dense.shape[2]}"
        )
        return out

    def pass2_push_output(self):
        out = push_output(self)
        _CALLS["capture" if _STATE["in_capture"] else "other"].append(
            (self.layer_idx, out is self._sparse_kv_gather_buffer)
        )
        return out

    def push_buffer_recording(self, parity, like, **kwargs):
        if parity == "loop_pass2":
            _OWN_BUFFERS.append(kwargs)
            logger.warning(f"TRUNK_PUSH_OWN_BUFFER created: {kwargs}")
        return push_buffer(self, parity, like, **kwargs)

    ttMLA._build_two_bank_index = build_two_bank_index
    ttMLA._pass2_push_output = pass2_push_output
    SparseKvPushPlanner.push_buffer = push_buffer_recording


def _push_reuse_verdict() -> None:
    model = getattr(_STATE["runtime"], "model", None)
    if model is not None and not getattr(model, "_pass2_layers", None) and not _PASS2:
        logger.info("TRUNK_ASSERT push_reuse N/A: this stage runs no loop pass-2 layer")
        return
    reuse = sorted(l for l, (r, _, _) in _PASS2.items() if r)
    no_reuse = sorted(l for l, (r, _, _) in _PASS2.items() if not r)
    cap = _CALLS["capture"]
    cap_layers = sorted({l for l, _ in cap})
    cap_not_dense = sorted({l for l, dense in cap if not dense})
    other_not_dense = sorted({l for l, dense in _CALLS["other"] if not dense})
    ok = (
        len(_PASS2) > 0
        and not no_reuse
        and _STATE["use_trace"]
        and _STATE["captured"]
        and cap_layers == sorted(_PASS2)
        and not cap_not_dense
        and not other_not_dense
        and not _OWN_BUFFERS
    )
    logger.info(
        f"TRUNK_ASSERT push_reuse {'PASS' if ok else 'FAIL'}: pass2_layers={len(_PASS2)} "
        f"({min(_PASS2, default=None)}-{max(_PASS2, default=None)}) reuse={len(reuse)} no_reuse={no_reuse} "
        f"use_trace={_STATE['use_trace']} captured={_STATE['captured']} "
        f"capture_calls={len(cap)} capture_layers={len(cap_layers)} capture_not_dense={cap_not_dense} "
        f"eager_calls={len(_CALLS['other'])} eager_not_dense={other_not_dense} own_pass2_buffers={len(_OWN_BUFFERS)}"
    )


def _digest_rows(runtime, kv, start: int, first: int, count: int) -> tuple:
    """sha256, finite, nonzero rows and rows of user 0's KVPE rows of one chunk at cache row ``start``, cache slots
    [first, first + count) of the stage, from every chip."""
    import torch

    import ttnn
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, unpack_scaled_fp8_kv_cache

    storage = kv.storage
    chunk = runtime.config.chunk_size
    meta = [runtime._meta1_dev(v) for v in (0, start, start + chunk)]
    rows = ttnn.experimental.deepseek_prefill.read_padded_kv_cache(
        storage,
        meta[0],
        meta[1],
        first_layer=first,
        num_read_layers=count,
        num_layers=_cache_slots(),
        chunk_size_global=chunk,
        cluster_axis=0,
        tp_axis=1,
    )
    host = ttnn.to_torch(rows, mesh_composer=ttnn.ConcatMeshToTensor(storage.device(), dim=0))
    for t in (rows, *meta):
        ttnn.deallocate(t)
    raw = host.contiguous().view(torch.uint8)
    flat = raw.reshape(-1, raw.shape[-1])
    digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
    if kv.format == MlaKvCacheFormat.SCALED_FP8:
        latent, scales, rope = unpack_scaled_fp8_kv_cache(flat.view(torch.float8_e4m3fn), kv.geometry)
        finite = all(bool(torch.isfinite(t).all()) for t in (latent, scales, rope))
    else:
        finite = bool(torch.isfinite(flat.view(torch.bfloat16).float()).all())
    return digest, finite, int(flat.any(dim=-1).sum()), int(flat.shape[0]), flat


def _mtp_slots() -> int:
    """KVPE slots MTP writes past the trunk's on this rank (the cache allocator's count)."""
    params = _STATE["params"]
    return params.mtp_levels if params.is_last_rank else 0


def _cache_slots() -> int:
    """KVPE cache slots per user: the stage's blocks, then MTP's."""
    return _STATE["params"].num_layers + _mtp_slots()


def _kv_digest(runtime) -> None:
    from models.demos.deepseek_v3_d_p.tt.mla.two_bank import RING_CHUNKS

    kv = _STATE["caches"].kvpe if _STATE["caches"] is not None else None
    if kv is None:
        logger.error("TRUNK_KV_DIGEST skipped: no KV caches recorded")
        return
    chunk, max_seq = runtime.config.chunk_size, runtime.config.max_seq_len
    chunks = [int(x) for x in os.environ.get("TRUNK_DIGEST_CHUNKS", "0,9,19,50,101,203").split(",") if x]
    trunk_slots, mtp_slots = _STATE["params"].num_layers, _mtp_slots()
    # The loop layers that run pass 2 on this stage hold the window ring past max_seq_len: one contiguous slot range.
    model = runtime.model
    ring_slots = sorted(layer - model.first_layer_idx for layer in model._pass2_layers)
    assert ring_slots == list(range(ring_slots[0], ring_slots[-1] + 1)) if ring_slots else True, ring_slots
    logger.info(
        f"TRUNK_KV_DIGEST storage shape={list(kv.storage.shape)} format={kv.format} chunks={chunks} "
        f"slots={trunk_slots}+{mtp_slots} mtp ring={RING_CHUNKS if ring_slots else 0} chunks at row {max_seq} "
        f"over slots {ring_slots[:1] + ring_slots[-1:]}"
    )
    reads = [(f"chunk={c}", c * chunk, 0, trunk_slots) for c in chunks]
    # Optional per-slot-range digests (TRUNK_DIGEST_SLOT_RANGES="0-53,54-76,77-77"): where two runs' KV differ.
    for spec in filter(None, os.environ.get("TRUNK_DIGEST_SLOT_RANGES", "").split(",")):
        lo, hi = (int(x) for x in spec.split("-"))
        reads += [(f"chunk={c}s{lo}_{hi}", c * chunk, lo, hi - lo + 1) for c in chunks]
    if mtp_slots:
        reads += [(f"chunk={c}mtp", c * chunk, trunk_slots, mtp_slots) for c in chunks]
    if ring_slots:
        reads += [(f"chunk=ring{j}", max_seq + j * chunk, ring_slots[0], len(ring_slots)) for j in range(RING_CHUNKS)]
    for label, start, first, count in reads:
        try:
            digest, finite, nonzero, n, _ = _digest_rows(runtime, kv, start, first, count)
            logger.info(f"TRUNK_KV_DIGEST {label} sha256={digest} finite={finite} nonzero_rows={nonzero} rows={n}")
        except Exception as exc:  # report, never fail the timing run
            logger.error(f"TRUNK_KV_DIGEST {label} failed: {type(exc).__name__}: {exc}")
    dump_dir = os.environ.get("TRUNK_KV_DUMP_DIR")
    if dump_dir:
        _kv_dump(runtime, kv, dump_dir, chunks, trunk_slots, ring_slots)


def _kv_dump(runtime, kv, dump_dir: str, chunks: list, trunk_slots: int, ring_slots: list) -> None:
    """``TRUNK_KV_DUMP_DIR``: user 0's logical (bf16) KVPE rows per cache slot, for each digest chunk of every trunk
    slot and each ring chunk of every ring slot, saved as ``<dir>/chunk<c>.pt`` / ``ring<j>.pt`` = {slot: rows}, so two
    runs can be compared layer by layer (``TRUNK_KV_DUMP`` lines)."""
    import torch

    from models.demos.deepseek_v3_d_p.tt.mla.two_bank import RING_CHUNKS
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

    os.makedirs(dump_dir, exist_ok=True)
    chunk, max_seq = runtime.config.chunk_size, runtime.config.max_seq_len
    reads = [(f"chunk{c}", c * chunk, range(trunk_slots)) for c in chunks]
    reads += [(f"ring{j}", max_seq + j * chunk, ring_slots) for j in range(RING_CHUNKS if ring_slots else 0)]
    for label, start, slots in reads:
        try:
            out = {}
            for slot in slots:
                flat = _digest_rows(runtime, kv, start, slot, 1)[4]
                if kv.format == MlaKvCacheFormat.SCALED_FP8:
                    flat = flat.view(torch.float8_e4m3fn)
                out[slot] = kv.unpack_host(flat).to(torch.bfloat16)
            path = os.path.join(dump_dir, f"{label}.pt")
            torch.save(out, path)
            logger.info(f"TRUNK_KV_DUMP {label} slots={len(out)} -> {path}")
        except Exception as exc:  # report, never fail the timing run
            logger.error(f"TRUNK_KV_DUMP {label} failed: {type(exc).__name__}: {exc}")


def _first(out):
    return out[0] if isinstance(out, tuple) else out


def _patch_mtp_wiring() -> None:
    """Record, per transformer forward, the blocks run (global layer, loop pass 2?, output id), the trunk norm's input
    and output ids, and every MTP seed (the h_normed run_mtp / run_mtp_traced receive)."""
    import models.demos.common.prefill.runners.layer_completion_layout as lcl
    from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer

    fwd, block_fwd, norm_fwd = TtPrefillTransformer.forward, TtPrefillBlock.forward, TtDistributedRmsNorm.forward
    run_mtp, run_mtp_traced, layout = (
        TtPrefillTransformer.run_mtp,
        TtPrefillTransformer.run_mtp_traced,
        lcl.layer_completion_layout,
    )

    def forward(self, *args, **kwargs):
        _FWD.update(model=self, blocks=[], norm=None, mtp_seeds=[])
        out = fwd(self, *args, **kwargs)
        if _FWD["mtp_seeds"]:
            _FWD["results"].append(_check_forward(self))
        _FWD["model"] = None
        return out

    def block_forward(self, *args, **kwargs):
        out = block_fwd(self, *args, **kwargs)
        if _FWD.get("model") is not None:
            _FWD["blocks"].append((self.mla.layer_idx, bool(kwargs.get("loop_pass2")), id(_first(out))))
        return out

    def norm_forward(self, x):
        out = norm_fwd(self, x)
        model = _FWD.get("model")
        if model is not None and self is model.norm:
            _FWD["norm"] = (len(_FWD["blocks"]), id(x), id(out))
        return out

    def seed(h_normed):
        _FWD["mtp_seeds"].append((len(_FWD["blocks"]), id(h_normed), _FWD["norm"]))

    def mtp(self, h_normed, *args, **kwargs):
        seed(h_normed)
        return run_mtp(self, h_normed, *args, **kwargs)

    def mtp_traced(self, h_normed, *args, **kwargs):
        seed(h_normed)
        return run_mtp_traced(self, h_normed, *args, **kwargs)

    def layer_completion_layout(*args, **kwargs):
        out = layout(*args, **kwargs)
        logger.info(
            f"TRUNK_MTP_LAYOUT ack ids: completions_per_chunk={out.completions_per_chunk} "
            f"rank_record_count={out.rank_record_count} rank_layer_ids={out.rank_layer_ids}"
        )
        return out

    TtPrefillTransformer.forward = forward
    TtPrefillBlock.forward = block_forward
    TtDistributedRmsNorm.forward = norm_forward
    TtPrefillTransformer.run_mtp = mtp
    TtPrefillTransformer.run_mtp_traced = mtp_traced
    lcl.layer_completion_layout = layer_completion_layout


def _check_forward(model) -> str:
    """'' when this forward seeded MTP once, after every pass-schedule step in order ending on the coda's last layer,
    with norm(that step's output); else what differs."""
    expected = [(s.layer, s.loop_pass >= 2) for s in model.steps if s.kind == "layer"]
    blocks, seeds = _FWD["blocks"], _FWD["mtp_seeds"]
    if len(seeds) != 1:
        return f"{len(seeds)} MTP seeds"
    n_before, seed_id, norm = seeds[0]
    ran = [(layer, pass2) for layer, pass2, _ in blocks[:n_before]]
    if ran != expected:
        return f"blocks before MTP {ran[:3]}..{ran[-3:]} ({len(ran)}) != steps {expected[:3]}..{expected[-3:]} ({len(expected)})"
    last_layer, last_pass2, last_out = blocks[n_before - 1]
    if (last_layer, last_pass2) != (model._num_model_layers - 1, False):
        return f"last step L{last_layer} pass2={last_pass2}, not the coda's last layer"
    if norm is None or norm[0] != n_before or norm[1] != last_out or norm[2] != seed_id:
        return f"seed is not norm(last step output): norm={norm} last_out={last_out} seed={seed_id}"
    return ""


def _mtp_input_verdict() -> None:
    model = _STATE["runtime"].model
    if model.mtp_predictor is None:
        logger.info("TRUNK_ASSERT mtp_input N/A: MTP off")
        return
    results = _FWD["results"]
    bad = sorted({r for r in results if r})
    steps = [s for s in model.steps if s.kind == "layer"]
    params, kv, index = _STATE["params"], _STATE["caches"].kvpe, _STATE["caches"].index
    pred = model.mtp_predictor
    logger.info(
        f"TRUNK_MTP_LAYOUT caches: trunk_slots={params.num_layers} mtp_slots={_mtp_slots()} "
        f"model.num_kvpe_cache_layers={model.num_kvpe_cache_layers} mtp first_cache_slot={pred.first_cache_slot} "
        f"layer_idx={pred.layer_idx} levels={pred.num_levels} mtp_block layer_num="
        f"{getattr(pred.module.layer.mla, 'layer_num', None)} kvpe storage={list(kv.storage.shape)} "
        f"index={list(index.shape) if index is not None else None} ack_base={model.first_layer_idx + pred.first_cache_slot}"
    )
    ok = bool(results) and not bad
    logger.info(
        f"TRUNK_ASSERT mtp_input {'PASS' if ok else 'FAIL'}: forwards_with_mtp={len(results)} failed={len(bad)} "
        f"steps={len(steps)} ({steps[0].layer}..{steps[-1].layer}, loop pass-2 steps="
        f"{sum(s.loop_pass >= 2 for s in steps)}) seed=norm(L{steps[-1].layer} output) {bad[:3]}"
    )


def _profile_window(runtime, run_chunk, label: str, out_dir: str):
    """Run one chunk inside a real-time profiler window and save every program record of every chip.

    The device is synchronized before the window opens (so only this chunk's programs land in it) and after the
    chunk, then records are drained until none arrives for 0.5 s. Saved per record: runtime_id, chip_id, raw start
    and end ticks, frequency, and an index into a table of kernel-source sets (the op identity)."""
    import gzip
    import pickle
    import threading
    import time

    import ttnn

    mesh = runtime.mesh_device
    lock = threading.Lock()
    rows, sources, src_of_rid, dropped = [], [], {}, [0]

    def collect(batch):
        with lock:
            dropped[0] += int(batch.dropped)
            for r in batch.records:
                rid = int(r.runtime_id)
                key = src_of_rid.get(rid)
                if key is None:
                    key = len(sources)
                    sources.append(tuple(str(s) for s in r.kernel_sources))
                    src_of_rid[rid] = key
                rows.append(
                    (rid, int(r.chip_id), int(r.start_timestamp), int(r.end_timestamp), float(r.frequency), key)
                )

    ttnn.synchronize_device(mesh)
    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    t0 = time.monotonic()
    try:
        out = run_chunk()
        ttnn.synchronize_device(mesh)
        wall_ms = (time.monotonic() - t0) * 1e3
        last, last_change = -1, time.monotonic()
        while time.monotonic() - last_change < 0.5:
            time.sleep(0.05)
            with lock:
                n = len(rows)
            if n != last:
                last, last_change = n, time.monotonic()
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    path = os.path.join(out_dir, f"rtprof-{label}.pkl.gz")
    with gzip.open(path, "wb") as f:
        pickle.dump({"label": label, "rows": rows, "sources": sources, "dropped": dropped[0], "wall_ms": wall_ms}, f)
    logger.info(
        f"TRUNK_PROFILE {label} records={len(rows)} programs={len(sources)} dropped={dropped[0]} "
        f"sync_wall_ms={wall_ms:.1f} file={path}"
    )
    return out


def _patch_profile() -> None:
    """Env-gated real-time profiler windows (q171.3.15): TRUNK_PROFILE_CHUNKS lists measured-request chunk indices
    (e.g. "1,101"), TRUNK_PROFILE_WARMUP the producer's warm-up chunk count (they reuse starts 0..warmup-1 chunks),
    TRUNK_PROFILE_DIR the output directory. Each listed chunk runs synchronized inside its own window; all other
    chunks run as before."""
    wanted = {int(x) for x in os.environ.get("TRUNK_PROFILE_CHUNKS", "").split(",") if x.strip()}
    if not wanted:
        return
    import ttnn
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    warmup = int(os.environ.get("TRUNK_PROFILE_WARMUP", "10"))
    out_dir = os.environ["TRUNK_PROFILE_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    seen = {}
    prefill_chunk = TtPrefillRuntime.prefill_chunk

    def profiled_prefill_chunk(self, input_tensor, kv_caches, slot_id, actual_start, actual_end, *args, **kwargs):
        run = lambda: prefill_chunk(self, input_tensor, kv_caches, slot_id, actual_start, actual_end, *args, **kwargs)
        if actual_start is None:
            return run()
        c = int(actual_start) // self.config.chunk_size
        seen[c] = seen.get(c, 0) + 1
        measured = seen[c] == (2 if c < warmup else 1)
        if not (measured and c in wanted):
            return run()
        if not _STATE.get("rt_logged"):
            _STATE["rt_logged"] = True
            logger.info(f"TRUNK_PROFILE realtime_profiler_active={ttnn.device.IsProgramRealtimeProfilerActive()}")
        return _profile_window(self, run, f"chunk{c}", out_dir)

    TtPrefillRuntime.prefill_chunk = profiled_prefill_chunk


def _patch_runtime() -> None:
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_4 import GLM54Adapter
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    capture = TtPrefillRuntime.capture_trace
    release = TtPrefillRuntime.release_trace
    allocate = GLM54Adapter.allocate_kv_cache

    def capture_trace(self, kv_caches):
        _STATE["use_trace"] = bool(self.config.use_trace)
        _STATE["in_capture"] = True
        try:
            capture(self, kv_caches)
        finally:
            _STATE["in_capture"] = False
        _STATE["captured"] = bool(getattr(self, "_trace_captured", False))

    def release_trace(self):
        if not _STATE.get("checked"):
            _STATE["checked"] = True
            _STATE["runtime"] = self
            _push_reuse_verdict()
            _mtp_input_verdict()
            _kv_digest(self)
        return release(self)

    def allocate_kv_cache(self, *, params, **kwargs):
        caches = allocate(self, params=params, **kwargs)
        _STATE["caches"], _STATE["params"] = caches, params
        return caches

    TtPrefillRuntime.capture_trace = capture_trace
    TtPrefillRuntime.release_trace = release_trace
    GLM54Adapter.allocate_kv_cache = allocate_kv_cache


def main() -> None:
    manifest = os.environ.get("PREFILL_MANIFEST", "")
    assert manifest.endswith("glm54.json"), f"the trunk probe runs GLM-5.4 (STAGE_PROBE_MODEL=glm54), got {manifest!r}"
    _patch_mla()
    _patch_runtime()
    _patch_mtp_wiring()
    _patch_profile()
    runpy.run_module("models.demos.common.prefill.runners.prefill_runner", run_name="__main__")


if __name__ == "__main__":
    main()
