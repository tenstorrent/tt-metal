# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TraceL1Guard on device, random weights.

A B = 1 model's trace is captured; then:
  rows build   a B = 8 rows model (more L1, lower addresses) is built beside it. When its buffers stay above the B = 1
               programs' CB extent the replay is safe: the guard lets it run, and the B = 1 traced step must give the
               same x as before the build (the state reset in between);
  filler       an L1 buffer placed below the B = 1 programs' CB extent: the guard must raise, and a replay that
               bypasses the guard must overwrite the filler (it is in the traced programs' CB region);
and the cost of one allocator memory-view walk (l1_low_water).

usage: python -m qwen36_resident.tests.trace_guard
"""
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident.model import Dims, ResidentModel, State, is_attn, l1_layout, l1_low_water, random_weights
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh


def main():
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    mesh = open_mesh()
    ok = True
    try:
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x)
        layers, interval = 4, 4
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        w = random_weights(d, layers - n_attn, n_attn, layers)
        st = lambda: State(d, layers - n_attn, n_attn, 40, 256, seed=1)
        a = ResidentModel(mesh, d, w, st(), layers, interval)
        tok = a.token_state(torch.randn(5120), 40)
        a.step(tok)
        ttnn.synchronize_device(mesh)
        a.capture_trace()
        base = ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1)
        cb_top = base + a._cb_top
        print(f"allocator base {base}, B=1 CB extent top {cb_top}, B=1 L1 buffers from {min(a.l1_addrs)}", flush=True)

        def traced_x():
            a.reset()
            a.step(tok)
            ttnn.synchronize_device(mesh)
            return a.x().clone()

        x0 = traced_x()
        t0 = time.time()
        for _ in range(100):
            l1_low_water(mesh)
        print(f"l1_low_water: {(time.time() - t0) * 10:.3f} ms per call", flush=True)

        b = ResidentModel(mesh, d, w, [st() for _ in range(8)], layers, interval, rows=True)
        low = min(l1_layout())
        print(
            f"rows build: L1 buffers from {low} ({'above' if low >= cb_top else 'BELOW'} the B=1 CB extent)", flush=True
        )
        try:
            x1 = traced_x()
            same = torch.equal(x0, x1)
            print(f"rows build: guard let the B=1 replay run; x bit-identical to before the build: {same}", flush=True)
            ok &= same and low >= cb_top
        except RuntimeError as e:
            print(f"rows build: guard raised ({str(e)[:80]}); buffers below the CB extent: {low < cb_top}", flush=True)
            ok &= low < cb_top

        # a buffer that does reach below the B=1 programs' CB extent: one core, shard rows of 64 B
        free_top = low - base
        rows = -(-(low - cb_top + 64 * 1024) // 64)
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        pattern = torch.full((rows, 32), 7.0)
        filler = ttnn.from_torch(
            pattern,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core, [rows, 32], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        ttnn.synchronize_device(mesh)
        print(
            f"filler at {filler.buffer_address()} ({rows * 64} B; below the CB extent top {cb_top}: "
            f"{filler.buffer_address() < cb_top}; allocator room was {free_top} B above base)",
            flush=True,
        )
        try:
            a._trace_guard.check(full=True)
            print("filler: guard did NOT raise", flush=True)
            ok = False
        except RuntimeError as e:
            print(f"filler: guard raised ({str(e)[:80]})", flush=True)
        # the replay the guard prevents: does it overwrite the filler?
        a.reset()
        ttnn.copy_host_to_device_tensor(tok, a.subs[0]["tok"])
        ttnn.execute_trace(mesh, a.trace_id, cq_id=0, blocking=True)
        after = ttnn.to_torch(ttnn.get_device_tensors(filler)[0]).float()
        clobbered = not torch.equal(after, pattern)
        print(f"filler: a raw replay (guard bypassed) overwrote it: {clobbered}", flush=True)
        a.release_trace()
        del a, b, filler
    finally:
        close_mesh(mesh)
    print(f"trace guard: {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
