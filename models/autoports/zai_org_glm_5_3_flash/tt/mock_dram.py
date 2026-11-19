# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM headroom of the image mock's warm-up and capture phases (host-only: GLM53_MOCK=1; a no-op on device).

The mock cluster runs no kernels but allocates every buffer, so a phase whose allocations do not fit device DRAM fails
on the mock too -- if the mock holds as much DRAM as the device. It holds less: int27c steady state 30.872 GiB/chip on
the mock vs 30.888 GiB/chip on the device (the same 543 programs), ~2 MiB/bank that only the device allocates. Stage 9's
int27m28c2 passed its mock and died on the device inside the prefill warm (prefill_tp warm via capture_decode: a
143.7 MB EpMoe buffer, 0.41 MB/bank short). Two knobs, both mock-only:

- GLM53_MOCK_DRAM_PROBE=1: during each watched phase read the DRAM memory view after every ttnn op (FastOperation
  call) that returns a DRAM tensor of >= GLM53_MOCK_DRAM_PROBE_MIN_ELEMS elements (default 16M: 4 MiB/bank in bf16 --
  the m28c2 failure was one 143.7 MB buffer) and after every 256th op (the view costs ~2 ms per read with 2,000
  buffers: every op would multiply the mock's warm-up); record the smallest largest-free-block per bank, the op it
  followed and the allocated bytes then. One `[mock-dram]` line per phase, appended as JSON to GLM53_MOCK_DRAM_OUT.
- GLM53_MOCK_DRAM_BALLAST_MIB=N: hold N MiB per bank of DRAM for the phase (one ROW_MAJOR bf16 buffer across all
  banks), so every allocation of the phase -- op outputs AND op-internal transients -- must fit beside N MiB/bank of
  device-only usage; an allocation that does not fit fails the mock at its call site like on the device."""

from __future__ import annotations

import contextlib
import json
import os
import time

import ttnn

PROBE_ENV = "GLM53_MOCK_DRAM_PROBE"
BALLAST_ENV = "GLM53_MOCK_DRAM_BALLAST_MIB"
MIN_ELEMS_ENV = "GLM53_MOCK_DRAM_PROBE_MIN_ELEMS"
EVERY = 256
OUT_ENV = "GLM53_MOCK_DRAM_OUT"
RESULTS: dict = {}


def mock() -> bool:
    return os.environ.get("GLM53_MOCK") == "1"


def _big_dram(out, min_elems: int) -> bool:
    """the op returned a DRAM tensor of >= min_elems elements (a list / tuple: any of them)"""
    for t in out if isinstance(out, (list, tuple)) else (out,):
        if not isinstance(t, ttnn.Tensor):
            continue
        try:
            vol = t.volume() if callable(t.volume) else t.volume  # padded elements (this build has no physical_volume)
            if t.memory_config().buffer_type == ttnn.BufferType.DRAM and int(vol) >= min_elems:
                return True
        except Exception:  # a host tensor / no memory config: not an allocation of this phase
            continue
    return False


def _view(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)  # lock-step allocation: chip 0 = every chip
    return int(v.largest_contiguous_bytes_free_per_bank), int(v.total_bytes_allocated_per_bank), int(v.num_banks)


@contextlib.contextmanager
def watch(mesh, label: str):
    """the probe + ballast around one phase (see the module doc); yields the phase's record (None when inactive)"""
    probe = mock() and os.environ.get(PROBE_ENV, "0") == "1"
    mib = int(os.environ.get(BALLAST_ENV, "0") or 0) if mock() else 0
    if mesh is None or not (probe or mib > 0):
        yield None
        return
    lf, al, banks = _view(mesh)
    st = dict(label=label, ballast_mib_per_bank=mib, ops=0, start_alloc_per_bank=al, start_largest_free_per_bank=lf,
              min_largest_free_per_bank=lf, min_at_op="(start)", min_alloc_per_bank=al, max_alloc_per_bank=al)  # fmt: skip
    ballast = None
    if mib > 0:  # ROW_MAJOR interleaved pages of 2048 B spread round-robin over the banks: N MiB on every bank
        rows = mib * 2**20 * banks // 2048
        ballast = ttnn.empty((rows, 1024), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        st["ballast_bytes_per_bank"] = int(_view(mesh)[1]) - al
    restore = None
    if probe:
        from ttnn.decorators import FastOperation

        orig = FastOperation.__call__
        min_elems = int(os.environ.get(MIN_ELEMS_ENV, str(16 * 2**20)))
        st.update(samples=0, probe_s=0.0, probe_min_elems=min_elems)

        def sampled(self, *a, **k):
            out = orig(self, *a, **k)
            st["ops"] += 1
            if st["ops"] % EVERY and not _big_dram(out, min_elems):
                return out
            t_ = time.perf_counter()
            lf_, al_, _ = _view(mesh)
            st["samples"] += 1
            st["probe_s"] += time.perf_counter() - t_
            if lf_ < st["min_largest_free_per_bank"]:
                st.update(min_largest_free_per_bank=lf_, min_at_op=self.python_fully_qualified_name,
                          min_alloc_per_bank=al_)  # fmt: skip
            if al_ > st["max_alloc_per_bank"]:
                st["max_alloc_per_bank"] = al_
            return out

        FastOperation.__call__ = sampled
        restore = (FastOperation, orig)
    t0 = time.perf_counter()
    err = None
    try:
        yield st
    except BaseException as e:
        err = e
        st["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        raise
    finally:
        if restore is not None:
            restore[0].__call__ = restore[1]
        if ballast is not None:
            ttnn.deallocate(ballast)
        st["wall_s"] = round(time.perf_counter() - t0, 2)
        if "probe_s" in st:
            st["probe_s"] = round(st["probe_s"], 2)
        st["end_alloc_per_bank"] = _view(mesh)[1]
        st["min_largest_free_mib_per_bank"] = round(st["min_largest_free_per_bank"] / 2**20, 2)
        st["ok"] = err is None
        RESULTS[label] = st
        print(f"[mock-dram] {json.dumps(st)}", flush=True)
        out = os.environ.get(OUT_ENV)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(st) + "\n")


def point(mesh, label: str):
    """one [mock-dram-point] row (mock only, GLM53_MOCK_DRAM_PROBE=1; a no-op on device): the largest free block and the
    allocated bytes per bank at this point (coordinator 2026-10-25T20:26:11Z: the cf3h-j margin on record). Printed to the
    server log only -- not to GLM53_MOCK_DRAM_OUT, whose rows are the per-phase records the install's DRAM floor reads; a
    point inside a watched phase includes that phase's ballast"""
    if mesh is None or not mock() or os.environ.get(PROBE_ENV, "0") != "1":
        return None
    lf, al, banks = _view(mesh)
    st = dict(label=label, largest_free_mib_per_bank=round(lf / 2**20, 2), alloc_mib_per_bank=round(al / 2**20, 2),
              banks=banks)  # fmt: skip
    print(f"[mock-dram-point] {json.dumps(st)}", flush=True)
    return st


def capture_phase() -> bool:
    """GLM53_MOCK_CAPTURE=1 (mock only): the capture paths run their forwards ONCE on the mock (no trace: the mock has
    no device to record on) so the capture phase's allocations -- what each captured forward allocates and keeps --
    happen in the device's order instead of being skipped"""
    return mock() and os.environ.get("GLM53_MOCK_CAPTURE", "0") == "1"
