# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: prefill SDPA as the model calls it, base ttnn.transformer vs ttnn.experimental.quasar.transformer."""
import time

import torch
import ttnn

CASES = {  # name: (q heads, kv heads, seq, head_dim, causal)
    "vision": (16, 16, 256, 64, False),
    "text": (32, 8, 128, 128, True),
}


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"grid={g.x}x{g.y}", flush=True)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=64, k_chunk_size=64
    )
    ops = {
        "base": ttnn.transformer.scaled_dot_product_attention,
        "experimental.quasar": ttnn.experimental.quasar.transformer.scaled_dot_product_attention,
    }
    for name, (nq, nkv, s, d, causal) in CASES.items():
        q, k, v = (torch.randn(1, n, s, d).bfloat16().float() for n in (nq, nkv, nkv))
        rep = nq // nkv
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=causal
        )
        tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for t in (q, k, v))
        for op_name, op in ops.items():
            t0 = time.time()
            try:
                out = op(tq, tk, tv, is_causal=causal, scale=d**-0.5, compute_kernel_config=ckc, program_config=prog)
                print(
                    f"{name:6s} {op_name:20s} pcc={pcc(ttnn.to_torch(out).float(), ref):.5f} {time.time() - t0:.1f}s",
                    flush=True,
                )
            except Exception as e:
                msg = next((ln.strip() for ln in str(e).splitlines() if "not supported" in ln), str(e).splitlines()[0])
                print(f"{name:6s} {op_name:20s} FAIL: {msg[:110]}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
