# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-row softmax-denominator error vs. S, flag off vs. on.

With V = all ones every output element is exactly 1.0 (a softmax row sums to 1), and both arms share the same
numerator matmul, so (out - 1) isolates the denominator's rounding error: the pack thread's per-position L1 sums
vs. the math thread's single 16-bit DEST accumulator.

    python bench_denom.py [S ...]
"""
import json
import sys

import torch

import ttnn


def main():
    seqs = [int(x) for x in sys.argv[1:]] or [256, 512, 1024, 2048, 4096]
    CK = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    D = ttnn.open_device(device_id=0, l1_small_size=32768)
    out = []
    try:
        for S in seqs:
            torch.manual_seed(0)
            B, nqh, nkh, d = 1, 8, 8, 128
            q_h = torch.randn(B, nqh, S, d)
            k_h = torch.randn(B, nkh, S, d)
            v_h = torch.ones(B, nkh, S, d)
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=128,
                k_chunk_size=S,
                exp_approx_mode=True,
            )
            q = ttnn.from_torch(q_h, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D)
            k = ttnn.from_torch(k_h, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D)
            v = ttnn.from_torch(v_h, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D)
            row = {"S": S}
            err = {}
            try:
                for flag in (False, True):
                    t = ttnn.transformer.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=False,
                        scale=d**-0.5,
                        program_config=cfg,
                        compute_kernel_config=CK,
                        math_thread_row_sums=flag,
                    )
                    e = (ttnn.to_torch(t).double() - 1.0).flatten()
                    ttnn.deallocate(t)
                    tag = "on" if flag else "off"
                    err[tag] = e
                    row[f"std_{tag}"] = float(e.std())
                    row[f"max_{tag}"] = float(e.abs().max())
                # Both arms share the numerator matmul exactly, so (e_on - e_off) is purely the difference between
                # the two denominators; the shared numerator error, which dominates each e on its own, cancels.
                d = err["on"] - err["off"]
                row["std_denom_diff"] = float(d.std())
                row["max_denom_diff"] = float(d.abs().max())
            except Exception as ex:
                row["error"] = str(ex).splitlines()[0][:160]
            for t in (q, k, v):
                ttnn.deallocate(t)
            out.append(row)
            print("DEN " + json.dumps(row), flush=True)
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    main()
