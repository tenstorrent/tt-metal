# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: ttnn.silu after another unary op in the same process (arg: none|gelu|gelu_approx|exp)."""
import sys, torch, ttnn

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 64, 1024)
tx = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
xb = x.bfloat16().float()
pcc = lambda a, b: torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()
pre = {
    "gelu": lambda: ttnn.gelu(tx),
    "gelu_approx": lambda: ttnn.gelu(tx, fast_and_approximate_mode=True),
    "exp": lambda: ttnn.exp(tx),
    "none": lambda: None,
}[sys.argv[1]]
pre()
print(
    f"after {sys.argv[1]:12s} silu pcc={pcc(ttnn.to_torch(ttnn.silu(tx)).float(), torch.nn.functional.silu(xb)):.5f}",
    flush=True,
)
ttnn.close_device(dev)
