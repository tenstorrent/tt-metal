# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy probe: one act_width_sharded_linear (emb_layers-shaped) inside signposts."""

import sys

import torch
import ttnn

ROOT = "/home/iguser/proj_vox/tt-metal"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.experimental.hunyuan_image_3_0.tt.matmul_utils import act_width_sharded_linear

try:
    from tracy import signpost
except ImportError:

    def signpost(header="", message=None):
        pass


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        x = ttnn.from_torch(
            torch.randn(1, 1, 2, 4096).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w = ttnn.from_torch(
            torch.randn(8192, 4096).bfloat16().transpose(0, 1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            torch.randn(1, 8192).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        signpost("start")
        out = act_width_sharded_linear(x, w, bias=b, batch_rows=2, compute_kernel_config=ck)
        signpost("stop")
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        ttnn.deallocate(b)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
