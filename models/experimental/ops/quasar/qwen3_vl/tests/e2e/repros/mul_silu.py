# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: text MLP gate, silu(w1) * w3 via ttnn.mul(input_tensor_a_activations=[SILU]); base vs ttnn.experimental.quasar."""
import torch
import ttnn


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"grid={g.x}x{g.y}", flush=True)
    a, b = torch.randn(1, 1, 128, 9728), torch.randn(1, 1, 128, 9728)
    ta, tb = (ttnn.from_torch(t.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev) for t in (a, b))
    ab, bb = a.bfloat16().float(), b.bfloat16().float()
    ref = torch.nn.functional.silu(ab) * bb
    silu = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    ops = {
        "base mul": ttnn.mul,
        "quasar multiply": ttnn.experimental.quasar.multiply,
    }
    for name, op in ops.items():
        for label, kw in (("plain", {}), ("silu(a)*b", {"input_tensor_a_activations": [silu]})):
            r = ab * bb if label == "plain" else ref
            try:
                out = ttnn.to_torch(
                    op(ta, tb, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
                ).float()
                p = torch.corrcoef(torch.stack([out.flatten().double(), r.flatten().double()]))[0, 1].item()
                print(f"{name:16s} {label:10s} pcc={p:.5f}", flush=True)
            except Exception as e:
                msg = next((ln.strip() for ln in str(e).splitlines() if "not supported" in ln), str(e).splitlines()[0])
                print(f"{name:16s} {label:10s} FAIL: {msg[:110]}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
