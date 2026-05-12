# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# REG C (moreh_adam) — Cross-check against Reg B root cause (moreh.sum 1D bug)
#
# Run with: scripts/tt-probe.sh moreh_adam --dev
#
# Reg B finding (commit 41de24a2cff): moreh.sum returns inf for 1D inputs.
# This probe asks: does the moreh_adam failure share that root cause?
#
# Empirical preconditions:
#   1. moreh_adam host code does NOT call ttnn.operations.moreh.sum
#      (grep confirms: 0 hits in moreh_adam/ and moreh_adamw/).
#   2. The failing test shapes are 2D and 8D, NOT 1D:
#        test_moreh_adam param shapes: [32,32], [2,2,2,2,2,2,64,64]
#
# So even if moreh.sum WERE called transitively, it would be on
# non-1D tensors. This probe verifies moreh.sum is correct for
# the test's input shapes and the internal per-tile shape
# (moreh_adam.cpp T1.32 / T1.33 act on a single 32x32 tile of
# cb_scalar_args via CopyTile<...,Pinned>{beta1_tile/beta2_tile} +
# Power<>{step}). If moreh.sum works for those shapes, Reg C has
# a DIFFERENT root cause (matches original Reg C hypothesis from
# commit de83edc4d43: Power<>.exec(uint32_t) member-exec dispatch
# is the first production callsite — eltwise_chain.inl:888 path).

import torch
import ttnn


def main():
    device = ttnn.open_device(device_id=0)

    print("=" * 70)
    print("REG C: cross-check moreh.sum on failing-test shapes")
    print("=" * 70)
    print()
    print("Hypothesis to falsify: 'Reg C shares Reg B root cause (moreh.sum bug)'")
    print()

    # 1D control — confirms Reg B bug still present (sanity)
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

    # Failing-test shapes for moreh_adam — full parametric grid is on these 2 shapes.
    # T1.32 / T1.33 in moreh_adam.cpp operate on cb_scalar_args (a single 32x32
    # tile holding 5 packed scalars), so internally adam never sums over the
    # full input shape — but we still probe the user-input shapes for safety.
    print()
    print("--- moreh_adam FAILING test shapes (expect ALL OK) ---")
    adam_shapes = [
        # Both shapes from test_moreh_adam parametrize block
        [32, 32],
        [2, 2, 2, 2, 2, 2, 64, 64],
    ]

    n_ok = 0
    n_bad = 0
    for shape in adam_shapes:
        fake = torch.ones(shape, dtype=torch.float32)
        tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
        val = ttnn.to_torch(r).reshape([-1])[0].item()
        expected = float(torch.prod(torch.tensor(shape)).item())
        rel = abs(val - expected) / max(expected, 1.0)
        ok = "OK" if rel < 0.05 else "BROKEN"
        if ok == "OK":
            n_ok += 1
        else:
            n_bad += 1
        print(
            f"  shape={str(shape):<28s} padded={str(list(tt_in.padded_shape)):<28s} "
            f"moreh.sum={val:>14.6g} expected={expected:>12.1f} [{ok}]"
        )

    # Also probe the single-tile cb_scalar_args shape used internally by
    # moreh_adam.cpp T1.32/T1.33 — Power<>{step} acts on a single 32x32 tile.
    print()
    print("--- moreh_adam INTERNAL per-tile shape (Power<>.exec on cb_scalar_args) ---")
    fake = torch.ones([32, 32], dtype=torch.float32)
    tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
    val = ttnn.to_torch(r).reshape([-1])[0].item()
    expected = 1024.0
    rel = abs(val - expected) / expected
    ok = "OK" if rel < 0.05 else "BROKEN"
    print(f"  single-tile [32,32] moreh.sum={val:>10.4g} expected={expected:.1f} [{ok}]")

    print()
    print(f"  Summary: {n_ok} OK / {n_bad} BROKEN out of {len(adam_shapes)}")
    print()
    if n_bad == 0:
        print("VERDICT: moreh.sum is CORRECT for ALL Reg C failing shapes.")
        print("         Reg C does NOT share Reg B root cause.")
        print("         Reg C root cause remains: 1711213980e Power<>.exec(uint32_t)")
        print("         member-exec dispatch — first production callsite of")
        print("         eltwise_chain.inl:888 with runtime exponent member field.")
    else:
        print(f"VERDICT: moreh.sum is BROKEN on {n_bad} Reg C shape(s).")
        print("         Reg C MAY share Reg B root cause for those shapes.")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
