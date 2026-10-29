# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TraceL1Guard on device, random weights: a B = 1 model's traced steps run; after a B = 8 rows model (more L1, so
lower addresses) is built beside it, the B = 1 model's next traced step must raise; and the cost of one allocator
memory-view walk (l1_low_water).

usage: python -m qwen36_resident.tests.trace_guard
"""
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident.model import Dims, ResidentModel, State, is_attn, l1_low_water, random_weights
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh


def main():
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    mesh = open_mesh()
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
        for _ in range(3):
            a.step(tok)
        ttnn.synchronize_device(mesh)
        print("B=1 traced steps before the second build: ok", flush=True)
        t0 = time.time()
        for _ in range(100):
            l1_low_water(mesh)
        print(f"l1_low_water: {(time.time() - t0) * 10:.3f} ms per call", flush=True)
        b = ResidentModel(mesh, d, w, [st() for _ in range(8)], layers, interval, rows=True)
        try:
            a.step(tok)
            ttnn.synchronize_device(mesh)
            print("trace guard: FAIL (the B=1 trace replayed beside a later, lower model)", flush=True)
        except RuntimeError as e:
            print(f"trace guard: PASS (raised: {str(e)[:120]})", flush=True)
        a.release_trace()
        del a, b
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
