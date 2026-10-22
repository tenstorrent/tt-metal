# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: which add broadcast variants run, mainline ttnn.add vs ttnn.experimental.quasar.add (bias add is [1,1,256,3072] + [3072])."""
import torch
import ttnn


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    a = torch.randn(1, 1, 64, 3072)
    cases = {
        "same shape": torch.randn(1, 1, 64, 3072),
        "row bcast [1,1,1,3072]": torch.randn(1, 1, 1, 3072),
        "rank-1 [3072]": torch.randn(3072),
        "scalar-ish [1,1,1,1]": torch.randn(1, 1, 1, 1),
        "col bcast [1,1,64,1]": torch.randn(1, 1, 64, 1),
    }
    ta = ttnn.from_torch(a.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    for name, b in [(f"{op_name}: {n}", b) for op_name in ("ttnn.add", "quasar.add") for n, b in cases.items()]:
        op = ttnn.add if name.startswith("ttnn.add") else ttnn.experimental.quasar.add
        try:
            tb = ttnn.from_torch(b.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
            got = ttnn.to_torch(op(ta, tb)).float()
            ref = a.bfloat16().float() + b.bfloat16().float()
            p = torch.corrcoef(torch.stack([got.flatten().double(), ref.flatten().double()]))[0, 1].item()
            print(f"{name:36s} pcc={p:.5f}", flush=True)
        except Exception as e:
            print(f"{name:36s} FAIL: {str(e).splitlines()[0][:110]}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
