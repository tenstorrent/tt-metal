"""Engine L1 guard (rc1d robustness, integ/stage1.L1Guard): single-chip device test.

A traced program with a static CB region is replayed after an L1 buffer was allocated below that region's end (the
d27u08 hang hypothesis 2: a post-capture L1 allocation under the engine's CB region). tt-metal validates the CB region
only at capture, so the replay silently overwrites the buffer. The test shows that hazard on the device, and that
L1Guard.check() (run before every verify replay in the model) raises before such a replay. With the rc1d ttnn binding
ExperimentalReserveL1Bottom it also checks the opt-in reservation: the same allocation then fails at its source,
small allocations land above the CB end, and the trace / eager program still run bit-identically.

One chip, no fabric:  TT_VISIBLE_DEVICES=<chip> pytest --noconftest -s models/demos/blackhole/qwen36/tests/test_mk_l1guard.py
"""
import importlib.util
import os
import time

import pytest
import torch

import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_MK = os.path.join(_HERE, "..", "tt", "megakernel")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_MK, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


stage1 = _load("mk_stage1_l1guard", "integ/stage1.py")
CBP = _load("mk_cbplan_l1guard", "engine/cbplan.py")
KERNEL = os.path.join(_MK, "engine", "kernels", "mk_ident_copy.cpp")
PAGE, BATCH = 2048, 8
DIM = 5120


def _program(x_in, x_out, core, filler):
    """mk_ident_copy (x_in -> x_out through CB 24, one core) behind a static filler CB 0 of ``filler`` bytes, so CB 24
    (the CB the kernel writes) sits at L1 base + filler: the top of the program's CB region."""
    n_pages = DIM // 32  # one tile row of [32, 5120] bf16
    rng = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))])
    cta = ttnn.TensorAccessorArgs(x_in).get_compile_time_args() + ttnn.TensorAccessorArgs(x_out).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    rt[core[0]][core[1]] = [x_in.buffer_address(), x_out.buffer_address(), 0, n_pages, PAGE, BATCH]
    cbs = [ttnn.CBDescriptor(total_size=filler, core_ranges=rng,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=PAGE)]),
           ttnn.CBDescriptor(total_size=PAGE * BATCH, core_ranges=rng,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=24, data_format=ttnn.bfloat16, page_size=PAGE)])]
    k = ttnn.KernelDescriptor(kernel_source=KERNEL, core_ranges=rng, compile_time_args=cta, runtime_args=rt,
                              config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs), cbs


def test_mk_l1guard(monkeypatch):
    monkeypatch.setenv("QWEN36_R16_MK_L1GUARD_EVERY", "1")  # check every replay on either query path (cost printed)
    torch.manual_seed(0)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=8 << 20)
    try:
        info = ttnn._ttnn.reports.get_device_info(mesh)
        base, top = int(info.address_at_first_l1_cb_buffer), int(info.worker_l1_size)
        core = (0, 0)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))])
        mc_big = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                   ttnn.ShardSpec(crs, (128, 1024), ttnn.ShardOrientation.ROW_MAJOR))  # 256 KB on core (0,0)
        mc_small = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                     ttnn.ShardSpec(crs, (32, 1024), ttnn.ShardOrientation.ROW_MAJOR))  # 64 KB
        pat = (torch.arange(128 * 1024) % 251).reshape(1, 1, 128, 1024).to(torch.bfloat16)
        big = lambda: ttnn.from_torch(pat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc_big)
        # where a 256 KB L1 buffer lands once nothing else is in L1 (the allocator is top-down and deterministic)
        t0_ = big()
        addr_t = t0_.buffer_address()
        ttnn.deallocate(t0_)
        filler = -(-(addr_t - base + 4096) // PAGE) * PAGE  # CB 24 starts ~4 KB inside that buffer's range
        x = torch.randn(1, 1, 32, DIM).to(torch.bfloat16)
        x_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_out = ttnn.from_torch(torch.zeros_like(x), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        desc, cbs = _program(x_in, x_out, core, filler)
        plan = CBP.plan(cbs, base)
        cb24 = CBP.address_of(plan, 24, core)
        cb_end = max(CBP.end_per_core(plan).values())
        print(f"[l1guard-test] L1 base {base} top {top}; 256 KB buffer lands at {addr_t}; CB 24 at {cb24}, CB end {cb_end}")
        assert addr_t <= cb24 < cb_end <= addr_t + 128 * 1024 * 2, (addr_t, cb24, cb_end)

        ttnn.generic_op([x_in, x_out], desc)  # eager (compile + CB validation: no L1 buffer yet)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        ttnn.generic_op([x_in, x_out], desc)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)

        def replay_ok():
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.zeros_like(x), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)), x_out)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            return torch.equal(ttnn.to_torch(x_out), x)

        assert replay_ok(), "baseline replay not bit-identical"

        guard = stage1.L1Guard(mesh, cb_end, mode="1")
        guard.check()  # arms: lowest live L1 >= CB end
        assert guard.armed
        n = 5000
        t0 = time.perf_counter()
        for _ in range(n):
            guard.check()
        us = (time.perf_counter() - t0) / n * 1e6
        print(f"[l1guard-test] guard armed via '{guard.kind}', check() {us:.2f} us per replay ({n} calls)")

        # a transient L1 buffer freed before the replay is fine (never seen)
        tr = big()
        ttnn.deallocate(tr)
        guard.check()
        assert replay_ok()

        # the hazard: a buffer allocated AFTER the capture below the CB end, live across a replay
        t = big()
        assert t.buffer_address() == addr_t < cb_end, (t.buffer_address(), addr_t, cb_end)
        for _ in range(2):  # every replay while it is live, not only the first
            with pytest.raises(RuntimeError, match=r"\[mk-l1guard\]"):
                guard.check()
        print("[l1guard-test] guard RAISED on the post-capture L1 buffer below the CB end (expected)")
        # what the guard prevents: without it the replay overwrites that buffer (tt-metal does not re-check at replay)
        before = ttnn.to_torch(t)
        assert torch.equal(before, pat)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        after = ttnn.to_torch(t)
        nbad = int((after != before).sum())
        print(f"[l1guard-test] unguarded replay overwrote {nbad} of {before.numel()} values of the post-capture buffer")
        assert nbad > 0, "expected the replay to overwrite the buffer under the CB region"
        ttnn.deallocate(t)
        guard.check()  # freed again: back above the CB end
        assert replay_ok()

        rsv = getattr(ttnn._ttnn.device, "ExperimentalReserveL1Bottom", None)
        if rsv is None:
            print("[l1guard-test] reservation: SKIPPED (this ttnn build has no ExperimentalReserveL1Bottom)")
            return
        rsv(mesh, cb_end)
        with pytest.raises(Exception):
            big()  # no room for 256 KB above the CB end any more
        s_ = ttnn.from_torch(pat[:, :, :32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc_small)
        assert s_.buffer_address() >= cb_end, (s_.buffer_address(), cb_end)
        guard.check()
        assert replay_ok(), "replay under the reservation not bit-identical"
        ttnn.generic_op([x_in, x_out], desc)  # eager enqueue: CB validation unaffected by the reservation
        ttnn.synchronize_device(mesh)
        assert torch.equal(ttnn.to_torch(x_out), x)
        ttnn.deallocate(s_)
        rsv(mesh, 0)  # release
        t = big()
        assert t.buffer_address() == addr_t, "release did not restore the allocator range"
        ttnn.deallocate(t)
        print("[l1guard-test] reservation: 256 KB refused, 64 KB placed at/above the CB end, replay + eager bit-identical, release OK")
        ttnn.release_trace(mesh, tid)
    finally:
        ttnn.close_mesh_device(mesh)
