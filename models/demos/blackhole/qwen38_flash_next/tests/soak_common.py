# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared trace soak loop (k2_soak): one trace of N back-to-back calls replayed many times, then a bitwise check of the
last replay against N eager calls.  Temporary file: delete with the soak tests (see R/CODE_REPORT.md)."""

import time

import torch

import ttnn

N_CALLS, REPLAYS, SYNC_EVERY, REPORT_EVERY, REFRESH_EVERY, VARIANTS = 20, 1000, 50, 100, 10, 3


def host_tensor(tensor, dtype, layout, mesh=None):
    """A host tensor (no device) that ``ttnn.copy_host_to_device_tensor`` can write into a resident one (mesh: replicated)."""
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if mesh is not None else {}
    return ttnn.from_torch(tensor.contiguous(), dtype=dtype, layout=layout, **kw)


def copy_in(host, device_tensor):
    ttnn.copy_host_to_device_tensor(host, device_tensor)


def raw(t):
    """Raw bits of a torch tensor as int64 (bf16 / fp32 by bit pattern, integers by value)."""
    t = t.contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16).to(torch.int64) & 0xFFFF
    if t.dtype == torch.float32:
        return t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return t.to(torch.int64) & 0xFFFFFFFF


def dies(tensor):
    """Raw bits of every die's copy (a single-device tensor gives one entry)."""
    try:
        parts = ttnn.get_device_tensors(tensor)
    except Exception:
        parts = [tensor]
    return [raw(ttnn.to_torch(p)) for p in parts]


def soak(mesh, name, rows, call, prepare, observe, n_calls=N_CALLS, replays=REPLAYS):
    """call(keep) runs one op (keep=True: leave its outputs alive for observe()); prepare(i) writes the inputs / state of
    variant i into the resident tensors (copy_host_to_device); observe() -> list of torch tensors (raw bits)."""
    tag = f"SOAK name={name} rows={rows}"

    def run_n():
        for _ in range(n_calls - 1):
            call(False)
        call(True)

    prepare(0)
    run_n()  # eager: compile + warm the program cache
    ttnn.synchronize_device(mesh)
    prepare(0)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    run_n()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.synchronize_device(mesh)
    print(f"{tag} replays_done=0 (trace captured, {n_calls} calls)", flush=True)
    t0 = time.perf_counter()
    try:
        for k in range(1, replays + 1):
            if k % REFRESH_EVERY == 0:
                prepare((k // REFRESH_EVERY) % VARIANTS)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            if k % SYNC_EVERY == 0:
                ttnn.synchronize_device(mesh)
            if k % REPORT_EVERY == 0:
                print(f"{tag} replays_done={k} elapsed_s={time.perf_counter() - t0:.1f}", flush=True)
        ttnn.synchronize_device(mesh)
        # corruption check: the same inputs / state, one replay against n_calls eager calls
        prepare(1)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        from_trace = observe()
        prepare(1)
        run_n()
        ttnn.synchronize_device(mesh)
        from_eager = observe()
    finally:
        ttnn.release_trace(mesh, trace)
    bad = [i for i, (a, b) in enumerate(zip(from_trace, from_eager)) if a.shape != b.shape or not torch.equal(a, b)]
    assert len(from_trace) == len(from_eager)
    print(f"{tag} FINAL replays_done={replays} compare_tensors={len(from_trace)} mismatched={bad}", flush=True)
    assert not bad, f"{tag}: last replay differs from the eager run in tensors {bad}"
