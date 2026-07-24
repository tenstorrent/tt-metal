# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full 40-layer smoke: build the whole model, run a non-aligned prefill + traced free-running
decode, confirm memory fit / finiteness, and print rough TTFT + decode t/s/u. Not an accuracy gate."""
from __future__ import annotations

import time

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1_500_000_000)
    try:
        t0 = time.perf_counter()
        gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=4096)  # all 40 layers
        print(f"BUILD 40 layers: {time.perf_counter()-t0:.1f}s")

        torch.manual_seed(1)
        P = 100  # non-aligned prompt length
        prompt = torch.randint(0, gen.vocab, (P,), dtype=torch.int64).tolist()
        N = 32

        # device traced free-run
        gen.reset()
        t1 = time.perf_counter()
        seq = gen.generate(prompt, N, next_input=None, enable_trace=True)
        dt = time.perf_counter() - t1
        print("tokens:", seq[:16], "...")
        print("counters:", dict(gen.counters))
        print(f"generate {N} tok wall: {dt*1000:.1f} ms  (~{N/dt:.1f} tok/s incl prefill+readback)")

        # host-argmax control on the first few tokens (correctness of device greedy)
        gen.host_sampling = True
        gen.reset()
        seq_h = gen.generate(prompt, 8, next_input=None, enable_trace=False)
        gen.host_sampling = False
        match = sum(int(a == b) for a, b in zip(seq, seq_h))
        print(f"device-vs-host greedy first-8 match: {match}/8  host={seq_h}")
        print("PROBE_SMOKE_DONE")
    finally:
        try:
            gen.teardown()
        except Exception:
            pass
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
