# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GroupNorm is a third of the flow estimator. Is the native op faster, and by how much?

The traced per-block profile (`probe_flow_ops_traced.py`) found the estimator's largest
single item, and it is not a matmul or a convolution:

    inside one resnet block, Blackhole, traced
      conv1d k3 256->256 @141    0.0320 ms
      groupnorm(8) @141          0.2197 ms      <- 6.9x the convolution it follows
      mish @141                  0.0076 ms

Two GroupNorms per ResNet block puts them at ~78% of the block, and ResNet blocks are ~43%
of an Euler step -- so **GroupNorm is roughly a third of the whole estimator**. Untraced it
looked ordinary (0.44 ms against the conv's 0.43), which is why it went unexamined: the
dispatch cost of both swamped the difference.

`TtGroupNorm` reaches the statistic through a shape change:

    [B, T, C] -> [B, T, G, C/G] -> permute(0,2,1,3) -> [B, G, T, C/G] -> [B, G, T*C/G]
    -> layer_norm -> the same three steps back -> per-channel affine

Under `TILE_LAYOUT` those two permutes swap the tiled row axis, which is a real re-tiling
shuffle rather than a view -- and after the first reshape the tiled face is `G x C/G` =
`8 x 32`, one tile carrying 8 useful rows out of 32.

`estimator.py` records why the native `ttnn.group_norm` was not used, and every reason is
about **accuracy**:

    native DRAM group_norm       PCC 0.9999231835
    native, use_welford=True     PCC 0.9998651553
    this, permute + layer_norm   PCC 0.9999931119

No speed number is attached to that decision. If native is several times faster traced, a
PCC of 0.99992 against a 0.99 gate is a different trade entirely. This probe measures both
halves at the shapes the estimator actually uses, so the decision can be made on the pair.

    python3 models/demos/cosyvoice/scripts/probe_groupnorm.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

BATCH, GROUPS = 2, 8
SHAPES = ((141, 256), (282, 256))  # the mid/down stages, and the outer stages


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()

    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:

        def traced_ms(body) -> float:
            for _ in range(2):
                body()
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            try:
                for _ in range(args.reps):
                    body()
            finally:
                ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            best = 1e9
            for _ in range(args.iters):
                t0 = time.perf_counter()
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                best = min(best, time.perf_counter() - t0)
            ttnn.release_trace(device, tid)
            return best * 1e3 / args.reps

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(f"  GroupNorm(G={GROUPS}), batch {BATCH}, traced replay, {args.reps} calls/trace\n")
        print(
            f"  {'shape':<15}{'permute+LN':>12}{'matmul':>10}{'native':>9}{'speedup':>10}"
            f"{'PCC permute':>14}{'PCC matmul':>13}"
        )
        print("  " + "-" * 84)

        for T, C in SHAPES:
            cg = C // GROUPS
            torch.manual_seed(0)
            xt = torch.randn(BATCH, T, C)
            wt, bt = torch.randn(C) * 0.5 + 1.0, torch.randn(C) * 0.1
            ref = torch.nn.functional.group_norm(xt.permute(0, 2, 1), GROUPS, wt, bt, eps=1e-5).permute(0, 2, 1)

            x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            w = ttnn.from_torch(wt.reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            b = ttnn.from_torch(bt.reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            def permute_form():
                h = ttnn.reshape(x, (BATCH, T, GROUPS, cg))
                h = ttnn.permute(h, (0, 2, 1, 3))
                h = ttnn.reshape(h, (BATCH, GROUPS, T * cg))
                n = ttnn.layer_norm(h, epsilon=1e-5)
                ttnn.deallocate(h)
                n = ttnn.reshape(n, (BATCH, GROUPS, T, cg))
                n = ttnn.permute(n, (0, 2, 1, 3))
                n = ttnn.reshape(n, (BATCH, T, C))
                s = ttnn.multiply(n, w)
                ttnn.deallocate(n)
                o = ttnn.add(s, b)
                ttnn.deallocate(s)
                return o

            got_p = ttnn.to_torch(permute_form()).float()

            # --- the permute-free formulation ---------------------------------
            # Each group's sum over channels is a matmul against a [C, G] indicator, and
            # the remaining reduction is over T -- an axis that needs no re-tiling. The
            # statistics come back as [B, 1, G], go back to [B, 1, C] through the same
            # indicator transposed, and the whole normalise-then-affine collapses into one
            # multiply and one add:
            #
            #     out = x * (inv*w) + (b - mean*inv*w)
            #
            # so the shape never changes and nothing is permuted.
            idx = torch.arange(C) // cg
            M = torch.zeros(C, GROUPS)
            M[torch.arange(C), idx] = 1.0
            m_dev = ttnn.from_torch(
                M.reshape(1, C, GROUPS), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            mt_dev = ttnn.from_torch(
                M.t().contiguous().reshape(1, GROUPS, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            n_elem = float(T * cg)

            def matmul_form():
                sq = ttnn.multiply(x, x)
                s1 = ttnn.matmul(x, m_dev)  # [B, T, G]
                s2 = ttnn.matmul(sq, m_dev)
                ttnn.deallocate(sq)
                S1 = ttnn.sum(s1, dim=1, keepdim=True)  # [B, 1, G]
                S2 = ttnn.sum(s2, dim=1, keepdim=True)
                ttnn.deallocate(s1)
                ttnn.deallocate(s2)
                mean = ttnn.multiply(S1, 1.0 / n_elem)
                ttnn.deallocate(S1)
                ex2 = ttnn.multiply(S2, 1.0 / n_elem)
                ttnn.deallocate(S2)
                var_raw = ttnn.subtract(ex2, ttnn.multiply(mean, mean))
                ttnn.deallocate(ex2)
                # Matches tt/flow/estimator.py's TtGroupNorm exactly: E[x^2]-E[x]^2 is
                # catastrophic-cancellation-prone and can go negative under bfloat16
                # rounding, and rsqrt of that is Inf with no exception -- see
                # PERF.md's "The matmul form needed a variance clamp" note.
                var = ttnn.relu(var_raw)
                ttnn.deallocate(var_raw)
                inv = ttnn.rsqrt(ttnn.add(var, 1e-5))
                ttnn.deallocate(var)
                mean_c = ttnn.matmul(mean, mt_dev)  # [B, 1, C]
                inv_c = ttnn.matmul(inv, mt_dev)
                ttnn.deallocate(mean)
                ttnn.deallocate(inv)
                scale = ttnn.multiply(inv_c, w)  # fold the affine in
                shift = ttnn.subtract(b, ttnn.multiply(mean_c, scale))
                ttnn.deallocate(mean_c)
                ttnn.deallocate(inv_c)
                o = ttnn.add(ttnn.multiply(x, scale), shift)
                ttnn.deallocate(scale)
                ttnn.deallocate(shift)
                return o

            try:
                got_m = ttnn.to_torch(matmul_form()).float()
                mm_ms = traced_ms(lambda: ttnn.deallocate(matmul_form()))
                mm_pcc = pcc(got_m, ref)
            except Exception as exc:  # noqa: BLE001
                mm_ms, mm_pcc = float("nan"), float("nan")
                print(f"  matmul form failed at [{BATCH},{T},{C}]: {str(exc)[:80]}")

            # The native op wants channels last and its own weight layout; it also wants a
            # core grid, which is the negotiation `estimator.py` calls out. Ask the device
            # for one rather than hardcoding, so this runs on both architectures.
            grid = device.compute_with_storage_grid_size()
            native_ok, got_n, native_ms = True, None, float("nan")
            try:
                wn = ttnn.create_group_norm_weight_bias_rm(wt, C, GROUPS)
                bn = ttnn.create_group_norm_weight_bias_rm(bt, C, GROUPS)
                w_n = ttnn.from_torch(wn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                b_n = ttnn.from_torch(bn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                x4 = ttnn.reshape(x, (BATCH, 1, T, C))

                def native_form():
                    return ttnn.group_norm(x4, num_groups=GROUPS, weight=w_n, bias=b_n, epsilon=1e-5)

                got_n = ttnn.to_torch(native_form()).float().reshape(BATCH, T, C)
                native_ms = traced_ms(lambda: ttnn.deallocate(native_form()))
            except Exception as exc:  # noqa: BLE001
                native_ok = False
                print(f"  native group_norm unavailable at [{BATCH},{T},{C}] (grid {grid}): {str(exc)[:70]}")

            perm_ms = traced_ms(lambda: ttnn.deallocate(permute_form()))
            nat = f"{native_ms:.4f}" if native_ok else "-"
            sp = f"{perm_ms / mm_ms:.2f}x" if mm_ms == mm_ms else "-"
            print(
                f"  [{BATCH},{T},{C}]".ljust(17)
                + f"{perm_ms:>10.4f}{mm_ms:>10.4f}{nat:>9}{sp:>10}{pcc(got_p, ref):>14.9f}{mm_pcc:>13.9f}"
            )
            ttnn.deallocate(x)

        print("\n  `speedup` is permute+LN over the matmul form. The accuracy question the")
        print("  original decision turned on is the last two columns.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
