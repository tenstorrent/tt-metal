# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: the start of vision block 0 as two ops: interleaved layer_norm (HiFi4, the model's call) and then the QKV
linear with the model's 2x1-grid program config. The linear runs first on its own as a control, then after the norm."""
import torch
import ttnn

from models.experimental.ops.quasar.qwen3_vl.tests.e2e.repros.linear_kblocks import pcc, program_config


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    x = torch.randn(1, 1, 256, 1024)
    gamma, beta = torch.randn(1024), torch.randn(1024)
    w = torch.randn(1024, 3072) / 32
    up = lambda t: ttnn.from_torch(t.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    tx, tw = up(x), up(w)
    # The model uploads gamma/beta expanded to [1, 32, 1024] (32 identical rows), tiled (tt/vision_layernorm.py).
    tg, tb = (up(v.view(1, 1, 1024).expand(1, 32, 1024)) for v in (gamma, beta))
    norm_ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    mm_ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    def linear(inp):
        out = ttnn.linear(
            inp,
            tw,
            compute_kernel_config=mm_ckc,
            program_config=program_config(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        got = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        return got

    xb = x.bfloat16().float()
    print(f"linear alone       pcc={pcc(linear(tx), xb @ w.bfloat16().float()):.5f}", flush=True)

    tn = ttnn.layer_norm(tx, weight=tg, bias=tb, epsilon=1e-6, compute_kernel_config=norm_ckc)
    norm_dev = ttnn.to_torch(tn).float()
    norm_ref = torch.nn.functional.layer_norm(xb, (1024,), gamma.bfloat16().float(), beta.bfloat16().float(), eps=1e-6)
    print(f"layer_norm         pcc={pcc(norm_dev, norm_ref):.5f}", flush=True)
    # Reference for the linear: the device norm output, so only the linear is judged.
    print(f"linear after norm  pcc={pcc(linear(tn), norm_dev @ w.bfloat16().float()):.5f}", flush=True)
    print(f"linear again       pcc={pcc(linear(tn), norm_dev @ w.bfloat16().float()):.5f}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
