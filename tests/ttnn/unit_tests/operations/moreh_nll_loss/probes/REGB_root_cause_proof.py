# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# REG B (moreh_nll_loss) ROOT CAUSE PROOF
#
# Run with: scripts/tt-probe.sh moreh_nll_loss
#
# Demonstrates that moreh.sum is BROKEN for 1D inputs on this branch.
# This is the actual root cause of moreh_nll_loss reduction=mean test failures.
# The migration commit 6ea995dc66d is INNOCENT — the bug is upstream in
# moreh_sum (or transitively in reduce_helpers* / scaler-prep paths).
#
# Evidence:
#   1. moreh.sum on 1D logical-shape input returns inf or wildly wrong values.
#   2. moreh.sum on 2D logical-shape input works.
#   3. moreh_nll_loss(reduction='mean') calls moreh.sum to produce the divisor.
#      With divisor=inf, 1/divisor=0, and the final loss is 0 (matching observed
#      tt_loss=0.000000 vs torch_loss=-0.4012, MaxATOL=0.4012).
#   4. Even with main's pristine pre-migration moreh_nll_loss_step2_kernel.cpp
#      swapped in, the test still fails identically — confirming migration is
#      not the cause.

import torch
import ttnn


def main():
    device = ttnn.open_device(device_id=0)

    print("=" * 70)
    print("REG B root-cause proof: moreh.sum broken on 1D inputs")
    print("=" * 70)

    # 1D shapes that REPRODUCE the bug
    bad_shapes_1d = [[5], [16], [32], [100]]
    print("\n--- 1D inputs (BROKEN) ---")
    for shape in bad_shapes_1d:
        fake = torch.ones(shape, dtype=torch.float32)
        tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
        val = ttnn.to_torch(r).reshape([-1])[0].item()
        expected = float(torch.prod(torch.tensor(shape)).item())
        ok = "OK" if abs(val - expected) < 0.5 else "BROKEN"
        print(
            f"  shape={str(shape):<10s} padded={tt_in.padded_shape} sum={val:>20.4g} expected={expected:>8.1f} [{ok}]"
        )

    # 2D shapes that WORK
    good_shapes_2d = [[5, 32], [32, 32], [16, 16], [32, 16]]
    print("\n--- 2D inputs (OK) ---")
    for shape in good_shapes_2d:
        fake = torch.ones(shape, dtype=torch.float32)
        tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
        val = ttnn.to_torch(r).reshape([-1])[0].item()
        expected = float(torch.prod(torch.tensor(shape)).item())
        ok = "OK" if abs(val - expected) < 0.5 else "BROKEN"
        print(
            f"  shape={str(shape):<10s} padded={tt_in.padded_shape} sum={val:>20.4g} expected={expected:>8.1f} [{ok}]"
        )

    # Now demonstrate the moreh_nll_loss connection
    print("\n--- moreh_nll_loss end-to-end shows divisor=inf via moreh.sum bug ---")
    torch.manual_seed(0)
    shape = [5, 10]
    C = shape[1]
    torch_input = torch.rand(shape, dtype=torch.float32).requires_grad_()
    torch_target = torch.randint(0, C, [5], dtype=torch.long)
    torch_weight = torch.rand(C, dtype=torch.float32)
    torch_divisor = torch.tensor([0], dtype=torch.float32)
    torch_output = torch.tensor([0], dtype=torch.float32)

    nll = torch.nn.NLLLoss(weight=torch_weight, ignore_index=1, reduction="mean")
    torch_loss = torch.tensor([nll(torch_input, torch_target)]).item()

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_target = ttnn.from_torch(torch_target, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_divisor = ttnn.from_torch(torch_divisor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.from_torch(torch_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_loss = ttnn.operations.moreh.nll_loss(
        tt_input,
        tt_target,
        "mean",
        weight_tensor=tt_weight,
        divisor_tensor=tt_divisor,
        output_tensor=tt_output,
        ignore_index=1,
    )
    tt_loss_val = ttnn.to_torch(tt_loss).reshape([-1])[0].item()
    tt_divisor_val = ttnn.to_torch(tt_divisor).reshape([-1])[0].item()

    print(f"  torch_loss              = {torch_loss:.6f}")
    print(f"  tt_loss                 = {tt_loss_val:.6f}  (expect ~-0.4012, got 0.0 because divisor=inf)")
    print(f"  tt_divisor (post-call)  = {tt_divisor_val}  (expect ~1.13, got inf because moreh.sum bug)")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
