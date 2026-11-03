# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MTP decode eh_proj as a BankStreamMatmul (tt/mtp.py _eh_bank_stream) against ttnn.linear and torch on Galaxy
row 0: random BFP8 [2 * hidden, hidden] column-parallel weight, x with 8 user rows. Prints EH_STREAM_PASS when both
match the torch product (PCC >= 0.999 on the user rows) and agree with each other, and each one's mean time."""

import time

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan, compute_config
from models.experimental.mimo_v2_6_flash.tt.proj_stream import BankStreamMatmul

HIDDEN, USERS, CALLS = 4096, 8, 50


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def main():
    torch.manual_seed(0)
    mesh, row = common.open_row()
    try:
        plan = MeshPlan(row)
        cfg = compute_config(ttnn.MathFidelity.HiFi2)
        w = torch.randn(2 * HIDDEN, HIDDEN) / 64
        x = torch.zeros(32, 2 * HIDDEN)
        x[:USERS] = torch.randn(USERS, 2 * HIDDEN)
        w_tt = ttnn.from_torch(
            w,
            device=row,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=plan.shard(-1),
        )
        x_tt = ttnn.from_torch(
            x.reshape(1, 1, 32, -1).bfloat16(),
            device=row,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(row),
        )
        stream = None
        for group in (4, 2, 1):
            try:
                stream = BankStreamMatmul(plan, 2 * HIDDEN, HIDDEN // plan.tp, cfg, group=group)
                break
            except AssertionError:
                continue
        assert stream is not None
        print(f"group {stream.group}, {len(stream.cores)} cores")

        def run_stream():
            return stream(x_tt, w_tt, rows=USERS)

        def run_linear():
            return ttnn.linear(
                x_tt, w_tt, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg
            )

        outs, times = {}, {}
        for name, fn in (("stream", run_stream), ("linear", run_linear)):
            out = fn()
            outs[name] = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(row, dim=-1)).float()[0, 0, :USERS]
            out.deallocate(True)
            ttnn.synchronize_device(row)
            t0 = time.perf_counter()
            for _ in range(CALLS):
                fn().deallocate(True)
            ttnn.synchronize_device(row)
            times[name] = (time.perf_counter() - t0) / CALLS * 1e6
        ref = x[:USERS].bfloat16().float() @ w
        p_stream, p_linear, p_both = (
            pcc(outs["stream"], ref),
            pcc(outs["linear"], ref),
            pcc(outs["stream"], outs["linear"]),
        )
        print(f"PCC stream {p_stream:.6f} linear {p_linear:.6f} stream-vs-linear {p_both:.6f}")
        print(
            f"host-timed per call (dispatch included): stream {times['stream']:.1f} us, linear {times['linear']:.1f} us"
        )
        if p_stream >= 0.999 and p_linear >= 0.999 and p_both >= 0.9999:
            print("EH_STREAM_PASS")
    finally:
        common.close(mesh, row)


if __name__ == "__main__":
    main()
