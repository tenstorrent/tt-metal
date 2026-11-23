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
* KV digest (finite, deterministic). After the request, before the trace is released, the KVPE cache slots of user 0
  for ``TRUNK_DIGEST_LAYERS`` (default 25,53,77: G0's, G1's and G3's last layers; a loop layer's slot also holds its
  pass-2 window ring, i.e. G2's KV) are read whole from every chip and logged as ``TRUNK_KV_DIGEST layer=<l>
  sha256=<hex> finite=<bool> nonzero_rows=<n>``. The cache starts zero-filled and two runs feed identical tokens, so
  equal digests across runs mean bit-identical KV; layer 77's rows depend on every pass of every chunk.
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


def _kv_digest() -> None:
    import torch

    import ttnn
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, unpack_scaled_fp8_kv_cache

    caches, num_layers = _STATE["caches"], _STATE["num_layers"]
    if caches is None:
        logger.error("TRUNK_KV_DIGEST skipped: no KV caches recorded")
        return
    kv = caches.kvpe
    storage = kv.storage
    mesh = storage.device()
    shape = list(storage.shape)
    layers = [int(x) for x in os.environ.get("TRUNK_DIGEST_LAYERS", "25,53,77").split(",") if x]
    logger.info(f"TRUNK_KV_DIGEST storage shape={shape} format={kv.format} num_layers={num_layers} layers={layers}")
    for layer in layers:
        slot = layer  # user 0: slot = user * num_layers + layer
        try:
            piece = ttnn.slice(storage, [slot, 0, 0, 0], [slot + 1, *shape[1:]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            host = ttnn.to_torch(piece, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            ttnn.deallocate(piece)
            raw = host.contiguous().view(torch.uint8)
            digest = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
            rows = raw.reshape(-1, raw.shape[-1])
            nonzero = int(rows.any(dim=-1).sum())
            if kv.format == MlaKvCacheFormat.SCALED_FP8:
                latent, scales, rope = unpack_scaled_fp8_kv_cache(host, kv.geometry)
                finite = all(bool(torch.isfinite(t).all()) for t in (latent, scales, rope))
            else:
                finite = bool(torch.isfinite(host.float()).all())
            logger.info(
                f"TRUNK_KV_DIGEST layer={layer} sha256={digest} finite={finite} nonzero_rows={nonzero} "
                f"rows={rows.shape[0]} chips={mesh.get_num_devices()}"
            )
        except Exception as exc:  # report, never fail the timing run
            logger.error(f"TRUNK_KV_DIGEST layer={layer} failed: {type(exc).__name__}: {exc}")


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
            _kv_digest()
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
