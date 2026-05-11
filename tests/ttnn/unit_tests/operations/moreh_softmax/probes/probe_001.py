# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# REG A (moreh_softmax) — Cross-check against Reg B root cause (moreh.sum 1D bug)
#
# Run with: scripts/tt-probe.sh moreh_softmax --dev
#
# Reg B finding (commit 41de24a2cff): moreh.sum returns inf for 1D inputs.
# This probe asks: does the moreh_softmax / moreh_softmax_backward failure
# share that root cause?
#
# Empirical preconditions:
#   1. Neither moreh_softmax nor moreh_softmax_backward host code calls
#      ttnn.operations.moreh.sum  (grep confirms: 0 hits in the op trees).
#   2. The failing test shapes are 4D, NOT 1D:
#        test_softmax_for_dim_nc:                  [15,109,64,64], dim=0
#        test_softmax_backward_for_dim_hw:         [10,20,96,160] dim=3, etc.
#        test_softmax_backward_large_algorithm:    [2,3,128,160]  dim=3, dim=2
#        test_softmax_backward_not_multiple_of_32: [1,1,10,15]    dim=3, etc.
#
# So even if moreh.sum WERE called transitively, it would be on 2D+/4D padded
# tensors, NOT on 1D inputs. This probe verifies moreh.sum is correct for
# exactly those failing-test shapes — if it is, Reg A has a DIFFERENT root cause
# (matches the original Reg A hypothesis in commit 7d24535ef9c: chain WaitNoPop
# semantics changed cb_wait_front cumulative behavior in commit 7dd7cf3824a's
# moreh_unary_chain_rt / moreh_mul_neg_chain_rt migration of
# moreh_softmax_backward_w.cpp T1.07 / T1.08).

import torch
import ttnn


def main():
    device = ttnn.open_device(device_id=0)

    print("=" * 70)
    print("REG A: cross-check moreh.sum on failing-test shapes")
    print("=" * 70)
    print()
    print("Hypothesis to falsify: 'Reg A shares Reg B root cause (moreh.sum bug)'")
    print()

    # 1D control — confirms Reg B bug still present on this branch (sanity check)
    print("--- 1D control (Reg B repro; expect BROKEN) ---")
    for shape in [[5], [32]]:
        fake = torch.ones(shape, dtype=torch.float32)
        tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
        val = ttnn.to_torch(r).reshape([-1])[0].item()
        expected = float(torch.prod(torch.tensor(shape)).item())
        ok = "OK" if abs(val - expected) < 0.5 else "BROKEN"
        print(f"  shape={str(shape):<14s} moreh.sum={val:>20.4g} expected={expected:>10.1f} [{ok}]")

    # Failing-test shapes for moreh_softmax / moreh_softmax_backward
    # All are 4D, so moreh.sum should be CORRECT (Reg B doesn't apply)
    print()
    print("--- moreh_softmax FAILING test shapes (4D; expect ALL OK) ---")
    softmax_shapes = [
        # test_softmax_for_dim_nc failing case
        [15, 109, 64, 64],
        # test_softmax_backward_for_dim_hw cases (sample of failing shapes)
        [32, 32],
        [3, 32, 32 * 5],
        [5, 6, 32, 32],
        [10, 20, 32 * 3, 32 * 5],
        [3, 32 * 5, 32],
        # test_softmax_backward_large_algorithmfor_dim_hw
        [2, 3, 32 * 4, 32 * 5],
        # test_softmax_backward_not_multiple_of_32_for_dim_hw — non-32-aligned
        [1, 1, 10, 15],
        [1, 1, 10, 32 * 2 + 10],
        [1, 1, 15, 10],
        [1, 1, 32 * 2 + 10, 32],
    ]

    n_ok = 0
    n_bad = 0
    for shape in softmax_shapes:
        fake = torch.ones(shape, dtype=torch.float32)
        tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
        val = ttnn.to_torch(r).reshape([-1])[0].item()
        expected = float(torch.prod(torch.tensor(shape)).item())
        # Relative tolerance — bfloat16 of huge sums (e.g., 8M) loses precision
        rel = abs(val - expected) / max(expected, 1.0)
        ok = "OK" if rel < 0.05 else "BROKEN"
        if ok == "OK":
            n_ok += 1
        else:
            n_bad += 1
        print(
            f"  shape={str(shape):<28s} padded={str(list(tt_in.padded_shape)):<20s} "
            f"moreh.sum={val:>14.6g} expected={expected:>10.1f} [{ok}]"
        )

    print()
    print(f"  Summary: {n_ok} OK / {n_bad} BROKEN out of {len(softmax_shapes)}")
    print()
    if n_bad == 0:
        print("VERDICT: moreh.sum is CORRECT for ALL Reg A failing shapes.")
        print("         Reg A does NOT share Reg B root cause.")
        print("         Reg A root cause remains: 7dd7cf3824a chain semantics regression.")
    else:
        print(f"VERDICT: moreh.sum is BROKEN on {n_bad} Reg A shape(s).")
        print("         Reg A MAY share Reg B root cause for those shapes.")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
