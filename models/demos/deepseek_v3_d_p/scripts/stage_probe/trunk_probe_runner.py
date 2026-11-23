# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 whole-trunk probe launcher: the stock prefill runner with two checks added (tt_metal_tracker-q171.2.9).

Use with ``STAGE_PROBE_MODEL=glm54 STAGE_PROBE_RUNNER_MODULE=models.demos.deepseek_v3_d_p.scripts.stage_probe.
trunk_probe_runner run_stage_probe.sh <name> 0 106``: all 106 layer passes on one rank. Nothing here changes what
the runner computes or when; both checks only observe.

* Pass-2 push reuse (q171.3.8 review m1). Every loop pass-2 ttMLA logs whether its push output reuses the dense
  gather buffer (``_pass2_push_reuses_dense``), every ``_pass2_push_output`` call records whether it returned that
  buffer and whether it ran inside trace capture, and an own ``loop_pass2`` push buffer being created is recorded.
  Before the trace is released one ``TRUNK_ASSERT push_reuse PASS|FAIL`` line states the verdict: PASS needs every
  pass-2 layer to choose reuse, the trace to be captured with every pass-2 push writing the dense buffer, and no
  own pass-2 buffer. A FAIL is logged, not raised, so the run's timing survives; the probe's checker reads it.
* KV digest (finite, deterministic). After the request, before the trace is released, user 0's KVPE rows of every
  layer (78 cache layers; a loop layer's pass-1 rows) are read for each chunk in ``TRUNK_DIGEST_CHUNKS`` (default the
  probe chunks 0,9,19,50,101,203), plus the loop layers' pass-2 window ring (G2's own KV, the last chunks' pass 2),
  with ``read_padded_kv_cache`` (the G1->G2 handoff's reader), and logged per chunk as ``TRUNK_KV_DIGEST chunk=<c>
  sha256=<hex> finite=<bool> nonzero_rows=<n> rows=<n>``. The cache starts zero-filled and two runs feed identical
  tokens, so equal digests across runs mean bit-identical KV; the last chunk's layer-77 rows depend on every pass of
  every chunk.
"""

from __future__ import annotations

import hashlib
import os
import runpy

from loguru import logger

_PASS2 = {}  # layer -> (reuses_dense, push spec rows, dense rows)
_CALLS = {"capture": [], "other": []}  # (layer, returned the dense buffer)
_OWN_BUFFERS = []
_STATE = {"in_capture": False, "captured": False, "use_trace": None, "caches": None, "num_layers": None}


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


def _digest_rows(runtime, kv, start: int, ring: bool) -> tuple:
    """sha256, finite, nonzero rows and rows of user 0's KVPE rows of one chunk at cache row ``start``, every layer
    (``ring``: only the loop layers, whose slots hold the window ring), from every chip."""
    import torch

    import ttnn
    from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import GLM54Config
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, unpack_scaled_fp8_kv_cache

    storage = kv.storage
    num_layers = _STATE["num_layers"]
    first, count = (GLM54Config.LOOP_FIRST_LAYER, len(GLM54Config.loop_layers())) if ring else (0, num_layers)
    chunk = runtime.config.chunk_size
    meta = [runtime._meta1_dev(v) for v in (0, start, start + chunk)]
    rows = ttnn.experimental.deepseek_prefill.read_padded_kv_cache(
        storage,
        meta[0],
        meta[1],
        first_layer=first,
        num_read_layers=count,
        num_layers=num_layers,
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
    return digest, finite, int(flat.any(dim=-1).sum()), int(flat.shape[0])


def _kv_digest(runtime) -> None:
    from models.demos.deepseek_v3_d_p.tt.mla.two_bank import RING_CHUNKS

    kv = _STATE["caches"].kvpe if _STATE["caches"] is not None else None
    if kv is None:
        logger.error("TRUNK_KV_DIGEST skipped: no KV caches recorded")
        return
    chunk, max_seq = runtime.config.chunk_size, runtime.config.max_seq_len
    chunks = [int(x) for x in os.environ.get("TRUNK_DIGEST_CHUNKS", "0,9,19,50,101,203").split(",") if x]
    logger.info(
        f"TRUNK_KV_DIGEST storage shape={list(kv.storage.shape)} format={kv.format} chunks={chunks} "
        f"ring={RING_CHUNKS} chunks at row {max_seq}"
    )
    reads = [(f"chunk={c}", c * chunk, False) for c in chunks]
    reads += [(f"chunk=ring{j}", max_seq + j * chunk, True) for j in range(RING_CHUNKS)]
    for label, start, ring in reads:
        try:
            digest, finite, nonzero, n = _digest_rows(runtime, kv, start, ring)
            logger.info(f"TRUNK_KV_DIGEST {label} sha256={digest} finite={finite} nonzero_rows={nonzero} rows={n}")
        except Exception as exc:  # report, never fail the timing run
            logger.error(f"TRUNK_KV_DIGEST {label} failed: {type(exc).__name__}: {exc}")


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
            _push_reuse_verdict()
            _kv_digest(self)
        return release(self)

    def allocate_kv_cache(self, *, params, **kwargs):
        caches = allocate(self, params=params, **kwargs)
        _STATE["caches"], _STATE["num_layers"] = caches, params.num_layers
        return caches

    TtPrefillRuntime.capture_trace = capture_trace
    TtPrefillRuntime.release_trace = release_trace
    GLM54Adapter.allocate_kv_cache = allocate_kv_cache


def main() -> None:
    manifest = os.environ.get("PREFILL_MANIFEST", "")
    assert manifest.endswith("glm54.json"), f"the trunk probe runs GLM-5.4 (STAGE_PROBE_MODEL=glm54), got {manifest!r}"
    _patch_mla()
    _patch_runtime()
    runpy.run_module("models.demos.common.prefill.runners.prefill_runner", run_name="__main__")


if __name__ == "__main__":
    main()
