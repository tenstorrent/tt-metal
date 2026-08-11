# SPDX-License-Identifier: Apache-2.0
"""Does moreh_sum_w / moreh_mean_w survive poisoned padding today?

The W reduce in these ops is a matmul against a scaler column vector, and the ragged tail is handled by
masking the input tile to zero. Replacing that mask with a partial scaler (zeroing the scaler's rows
instead of the input's columns) turns `0 * garbage` into `garbage * 0`, which is only equivalent if the
padding cannot be inf/NaN -- or if the hardware doesn't produce NaN there.

This probe establishes the *current* behaviour, which is the baseline any migration has to preserve.
"""
import math
import torch
import ttnn

W_RAGGED = 95  # 2 full tiles + 31 valid columns
SHAPE = [1, 1, 64, W_RAGGED]


def run(device, pad_value, label):
    torch_x = torch.rand(size=SHAPE, dtype=torch.bfloat16) + 1.0
    x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if pad_value is not None:
        x = ttnn.fill_implicit_tile_padding(x, pad_value)

    got_sum = ttnn.to_torch(ttnn.operations.moreh.sum(x, 3, keepdim=True)).to(torch.float32)
    got_mean = ttnn.to_torch(ttnn.operations.moreh.mean(x, dim=3, keepdim=True)).to(torch.float32)
    exp_sum = torch_x.to(torch.float32).sum(dim=3, keepdim=True)
    exp_mean = torch_x.to(torch.float32).mean(dim=3, keepdim=True)

    for name, got, exp in (("sum", got_sum, exp_sum), ("mean", got_mean, exp_mean)):
        finite = bool(torch.isfinite(got).all())
        err = float((got - exp).abs().max()) if finite else float("nan")
        rel = err / float(exp.abs().max())
        print(f"{label:22s} {name:5s} finite={finite!s:5s} max_abs_err={err:.4f} rel={rel:.5f}")


if __name__ == "__main__":
    d = ttnn.open_device(device_id=0)
    try:
        run(d, None, "padding untouched")
        run(d, 0.0, "padding = 0")
        run(d, 1000.0, "padding = 1000")
        run(d, float("inf"), "padding = +inf")
        run(d, float("nan"), "padding = NaN")
    finally:
        ttnn.close_device(d)
