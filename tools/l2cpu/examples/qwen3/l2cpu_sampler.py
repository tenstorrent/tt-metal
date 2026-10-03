# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""x280 samplers for the decode harness (decode_harness.py, interface in dh_samplers.py).

L2cpuHostSampler (Plan A, host-mediated), per decode step:
  1. push op (Tensix, kernels/x280_push_logits.cpp): the `batch` row-major bf16 logits rows -> arena landing
     zone (batch 1: cached zone through the coherent Memory Port; batch > 1: uncached zone through the System Port
     alias), then ttnn.synchronize_device;
  2. the HOST writes step_seq_base, bumps req_seq and rings the MSI doorbell through ttnn.cluster;
  3. the host polls done_seq and reads the tokens from the arena (next_tokens lines).

The firmware session (arena + running firmware + L2cpuOps) must exist before the model's traces are captured:
the harness calls l2cpu_bootstrap(mesh) right after opening the device (`--x280`), from a fresh chip reset.
"""
from __future__ import annotations

import atexit
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import deps  # noqa: E402,F401  (bindings to the other l2cpu components, see deps.py)

from l2cpu.sampling import layout as A  # noqa: E402
from dh_samplers import Sampler  # noqa: E402

CORE_MHZ = 1750.0
_SESSION = {}


def l2cpu_bootstrap(device, image=None, log=print):
    """Allocate the arena, boot the firmware, create the device ops. Call before any trace capture."""
    from l2cpu.sampling.fw import DEFAULT_IMAGE, boot as boot_firmware
    from l2cpu_ops import L2cpuOps

    fw, arena, info = boot_firmware(device, image or DEFAULT_IMAGE, log=log)
    ops = L2cpuOps(device, arena)
    _SESSION.update(fw=fw, arena=arena, ops=ops, info=info)
    log(
        "x280 firmware READY: arena PA 0x%x, layout %d, bring-up %.2f s"
        % (fw.base, A.L2S_LAYOUT_VERSION, info["boot_s"])
    )
    return fw


def session():
    if "fw" not in _SESSION:
        raise RuntimeError("x280 sampler: no firmware session; run the harness with --x280 (from a fresh chip reset)")
    return _SESSION


class L2cpuHostSampler(Sampler):
    name = "x280"

    def bind(self, device, logits_tensor, batch, vocab, params_per_user, seeds, reader=None):
        import ttnn

        from l2cpu.sampling.fw import local_desc

        super().bind(device, logits_tensor, batch, vocab, params_per_user, seeds, reader=reader)
        s = session()
        self.ttnn = ttnn
        self.fw, self.ops = s["fw"], s["ops"]
        vpad = list(logits_tensor.shape)[-1]
        page = logits_tensor.buffer_aligned_page_size()
        assert logits_tensor.dtype == ttnn.bfloat16 and logits_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        assert page == vpad * 2 and page % 64 == 0, (page, vpad)
        self.uncached = batch > 1
        self.fw.set_logits_desc(local_desc(A.L2S_DTYPE_BF16, vpad, rows=32, uncached=self.uncached, row_stride=page))
        self.fw.set_params([(p["temperature"], p["top_k"], p["top_p"], sd) for p, sd in zip(self.params, self.seeds)])
        self.fw.set_ctrl(batch, vocab, vpad, flags=0)
        self.ops.hw = self.fw.hw
        self.ops.set_push_source(logits_tensor, row_bytes=page)  # source table in the arena (after READY)
        self.push = self.ops.push_program(batch, row_bytes=page, uncached=self.uncached)
        self.stats = []
        if not getattr(self, "_atexit", False):
            atexit.register(self.print_summary)
            self._atexit = True

    def sample(self, step):
        fw = self.fw
        t0 = time.perf_counter()
        self.ops.run(self.push)
        self.ttnn.synchronize_device(self.device)
        t1 = time.perf_counter()
        fw.set_step(step)  # firmware step = req_seq - step_seq_base = step
        fw.issue()
        fw.wait_done(timeout=10.0)
        toks = fw.next_tokens(self.batch)
        t2 = time.perf_counter()
        tm = fw.timing_last()
        if tm["req_seq"] != fw.req:
            raise RuntimeError("x280 timing entry %d != req %d" % (tm["req_seq"], fw.req))
        self.stats.append((t1 - t0, t2 - t1, tm))
        self.last = {"read": t1 - t0, "sample": t2 - t1}
        return [int(t) for t in toks]

    def summary(self):
        if not self.stats:
            return {}
        med = statistics.median
        tms = [t for _, _, t in self.stats]
        us = lambda c: c / CORE_MHZ  # noqa: E731
        return {
            "steps": len(self.stats),
            "push_sync_ms": med(p for p, _, _ in self.stats) * 1e3,
            "host_round_trip_ms": med(r for _, r, _ in self.stats) * 1e3,
            "x280_total_us": us(med(t["cyc_total"] for t in tms)),
            "x280_read_us": us(med(t["cyc_read"] for t in tms)),
            "x280_sample_us": us(med(t["cyc_sample"] for t in tms)),
            "x280_write_us": us(med(t["cyc_write"] for t in tms)),
            "x280_wait_workers_us": us(med(t["cyc_wait"] for t in tms)),
            "x280_publish_other_us": us(
                med(t["cyc_total"] - t["cyc_read"] - t["cyc_sample"] - t["cyc_write"] - t["cyc_wait"] for t in tms)
            ),
        }

    def print_summary(self):
        try:
            print("X280 PLAN A SUMMARY", self.summary(), flush=True)
        except Exception as e:  # device may be closed already
            print("X280 PLAN A SUMMARY unavailable:", e)
