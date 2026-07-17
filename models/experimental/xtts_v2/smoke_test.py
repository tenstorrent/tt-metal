#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
XTTS-v2 TTNN smoke test — a deliberately trivial one-layer model.

Purpose: establish and verify the change -> run loop for TTNN model development
in this repo BEFORE any real XTTS bringup. It does one linear layer on device and
checks the result against a PyTorch golden via PCC.

Run (from the tt-metal repo root, with the venv active and env vars set):

    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    python models/experimental/xtts_v2/smoke_test.py

Expected output ends with:  SMOKE TEST PASSED (pcc=...)
"""

import torch
import ttnn

from models.common.utility_functions import comp_pcc, comp_allclose


class OneLayer:
    """A one-layer TTNN 'model': y = x @ W + b, all on device.

    Weight convention matches ttnn.linear: input [.., in_features] @ weight
    [in_features, out_features] -> [.., out_features]. (Note this is NOT
    torch.nn.Linear's x @ W^T convention — we keep W as [in, out] to avoid a
    transpose and keep the golden unambiguous.)
    """

    def __init__(self, device, weight: torch.Tensor, bias: torch.Tensor):
        self.device = device
        # Preprocess weights once: host torch -> device ttnn tensors (tiled, bf16).
        self.weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.bias = ttnn.from_torch(bias.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight, bias=self.bias)


def main() -> bool:
    torch.manual_seed(0)

    in_features, out_features, batch = 128, 256, 32

    # Golden (PyTorch, fp32).
    x = torch.randn(batch, in_features)
    weight = torch.randn(in_features, out_features)
    bias = torch.randn(out_features)
    golden = x @ weight + bias

    # Device run.
    device = ttnn.open_device(device_id=0)
    try:
        device.enable_program_cache()

        model = OneLayer(device, weight, bias)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        y_tt = model(x_tt)
        y = ttnn.to_torch(y_tt)
    finally:
        ttnn.close_device(device)

    # Compare. bf16 matmul over 128 accum elements -> use a realistic threshold.
    passed, pcc_msg = comp_pcc(golden, y, pcc=0.99)
    _, allclose_msg = comp_allclose(golden, y)
    print(f"output shape: torch={tuple(golden.shape)} ttnn={tuple(y.shape)}")
    print(f"pcc: {pcc_msg}")
    print(f"allclose: {allclose_msg}")

    if passed:
        print(f"SMOKE TEST PASSED ({pcc_msg})")
    else:
        print(f"SMOKE TEST FAILED ({pcc_msg})")
    return passed


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
