# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: vision block 0 QKV linear with the model's real cached weight and a captured input.

usage: cached_qkv_linear.py CACHE_DIR RUN_DIR
  CACHE_DIR  tensor cache holding visual.blocks.0.attention.wqkv_dtype_BFLOAT16_layout_TILE.tensorbin
  RUN_DIR    run folder saved with --qwen-capture-ops vision.block0 (op 0 is the layer_norm output)
Checks the weight survives a device round trip, then runs the linear with the model's 2x1-grid program config."""
import sys
from pathlib import Path

import torch
import ttnn

from models.experimental.ops.quasar.qwen3_vl.tests.e2e.repros.linear_kblocks import pcc, program_config


def main():
    cache, run = Path(sys.argv[1]), Path(sys.argv[2])
    wfile = str(cache / "visual.blocks.0.attention.wqkv_dtype_BFLOAT16_layout_TILE.tensorbin")
    x = torch.load(run / "ops.pt", weights_only=False)[0][3].float()
    w_host = ttnn.to_torch(ttnn.load_tensor(wfile)).float()
    print(f"weight {tuple(w_host.shape)} input {tuple(x.shape)}", flush=True)
    dev = ttnn.open_device(device_id=0)
    w_dev = ttnn.load_tensor(wfile, device=dev)
    back = ttnn.to_torch(w_dev).float()
    bad = (back != w_host).sum().item()
    print(f"weight round trip exact={bad == 0} mismatches={bad}/{back.numel()}", flush=True)
    tx = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    out = ttnn.linear(
        tx, w_dev, compute_kernel_config=ckc, program_config=program_config(1), memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(out).float()
    ref = x.reshape(-1, x.shape[-1]) @ w_host.reshape(-1, w_host.shape[-1])
    print(f"cached-weight linear pcc={pcc(got.reshape(ref.shape), ref):.5f}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
