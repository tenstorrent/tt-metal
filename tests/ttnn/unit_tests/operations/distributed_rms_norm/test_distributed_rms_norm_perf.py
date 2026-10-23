# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device perf measurement for the LOOSE perf cases (1x4 ring, G = 4, 2 links, bf16 + bf16 gamma).

Measurement only (prints, no perf assertion); correctness is checked by PCC on the last call.
Durations come from the realtime program profiler (one record per chip program). Two numbers per case:
  * untraced: one dispatch; the chips launch skewed and each waits at the Fabric fence for the last-launched
    one, so the MIN over chips (the last-launched chip) is the op cost;
  * traced: the call replayed from a trace on every chip at once (no host skew); MAX over chips is reported.
Run: scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_perf.py -s
"""

from __future__ import annotations

import threading
import time

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.topology import ring_mock_meshes
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_drn_acceptance_perf", Path(__file__).with_name("test_distributed_rms_norm.py")
)
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)
run_case = _acc.run_case

import os
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as _pd

# knob overrides for A/B runs: DRN_KNOBS="MIN_ROUNDS=2,MIN_BLOCK_W=8"
for _kv in filter(None, os.environ.get("DRN_KNOBS", "").split(",")):
    _k, _v = _kv.split("=")
    setattr(_pd, _k, eval(_v))
    print(f"PERF knob {_k}={_v}")

# x / gamma dtype override for A/B runs: DRN_DTYPE=bfloat8_b (torch side stays bf16)
DTYPE = getattr(ttnn, os.environ.get("DRN_DTYPE", "bfloat16"))
GAMMA_DTYPE = ttnn.bfloat16 if DTYPE == ttnn.bfloat8_b else DTYPE
_acc.TORCH_DTYPE.setdefault(ttnn.bfloat8_b, torch.bfloat16)  # the correctness call's torch-side input

_RING = [(s, a) for s, a in ring_mock_meshes() if s == (1, 4)]


class _Records:
    def __init__(self):
        self.lock = threading.Lock()
        self.recs = []

    def __call__(self, batch):
        with self.lock:
            for r in batch.records:
                try:
                    self.recs.append(
                        (int(r.runtime_id), (r.end_timestamp - r.start_timestamp) / r.frequency, int(r.core_count))
                    )
                except Exception:
                    pass

    def take(self):
        time.sleep(0.3)
        with self.lock:
            out, self.recs = self.recs, []
        return out


def _summary(recs, min_cores=1):
    # keep the op's programs (the big grid); drop readback / small helper programs
    durs = sorted(d for _, d, c in recs if c >= min_cores)
    return [round(d / 1000.0, 2) for d in durs]


@pytest.mark.skipif(not _RING, reason="needs a 1x4 ring stand-in")
@pytest.mark.parametrize("mesh_shape,cluster_axis", [pytest.param(s, a, id="ring1x4") for s, a in _RING])
@pytest.mark.parametrize("num_links", [2, 1], ids=["2link", "1link"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 640, 1792),  # LOOSE perf focus
        (1, 1, 32, 1792),  # LOOSE decode
        (1, 1, 2048, 1280),  # LOOSE box prefill
        (1, 1, 640, 1536),
        (1, 1, 2048, 2048),
        (1, 1, 4096, 1024),
        (4, 1, 512, 512),
    ],
    ids=lambda s: "x".join(map(str, s)),
)
@pytest.mark.parametrize("use_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_perf(shape, use_gamma, num_links, mesh_shape, cluster_axis, mesh_device):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("realtime profiler inactive")
    rec = _Records()
    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(rec)
    try:
        # compile + correctness
        run_case(
            mesh_device,
            shape,
            ttnn.bfloat16,
            cluster_axis,
            use_gamma=use_gamma,
            topology=ttnn.Topology.Ring,
            num_links=num_links,
        )
        rows, cols = tuple(mesh_device.shape)
        torch.manual_seed(0)
        run_case(
            mesh_device,
            shape,
            DTYPE,
            cluster_axis,
            use_gamma=use_gamma,
            gamma_dtype=GAMMA_DTYPE,
            topology=ttnn.Topology.Ring,
            num_links=num_links,
        )
        rows, cols = tuple(mesh_device.shape)
        torch.manual_seed(0)
        x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(torch.bfloat16)
        tx = _acc.create_ttnn_input_tensor(x, mesh_device, dtype=DTYPE, layout=ttnn.TILE_LAYOUT)
        tg = None
        if use_gamma:
            g = torch.randn((rows, cols, 1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)
            tg = _acc.create_ttnn_gamma(g, mesh_device, dtype=GAMMA_DTYPE, layout=ttnn.TILE_LAYOUT)
        kw = dict(cluster_axis=cluster_axis, gamma=tg, epsilon=1e-6, topology=ttnn.Topology.Ring, num_links=num_links)
        out = _acc.distributed_rms_norm(tx, **kw)
        ttnn.synchronize_device(mesh_device)
        rec.take()
        # untraced: last-launched chip (min over chips) per call
        untraced = []
        for _ in range(3):
            out = _acc.distributed_rms_norm(tx, **kw)
            ttnn.synchronize_device(mesh_device)
            s = _summary(rec.take())
            untraced.append(s[0] if s else None)
        print(f"PERF {shape} {DTYPE} gamma={use_gamma} links={num_links} untraced last-chip us: {untraced}")
        # traced: every chip replays at once; max over chips
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = _acc.distributed_rms_norm(tx, **kw)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        rec.take()
        traced = []
        for _ in range(3):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            s = _summary(rec.take())
            print(f"PERF traced all-chips {s}")
            traced.append(s[-1] if s else None)
        ttnn.release_trace(mesh_device, tid)
        print(f"PERF {shape} traced max-chip us: {traced}")
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
