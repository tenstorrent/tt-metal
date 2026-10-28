# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""ttsim WH (QUASAR_GAPS S1): a bf16 linear with K = 4096 and fp32 dest accumulation. ttnn's default program splits K
into blocks; each block after the first reloads the fp32 partials into DEST, and the WH LLK's TEN-3868 dummy SrcA
unpack that follows aborts ttsim WH (`unpack_to_dst=0 in_data_format=0 out_data_format=0`). With bf16 dest
accumulation the same linear passes.

usage: fp32_dest_acc_linear.py [0|1]   1 (default): fp32_dest_acc_en=True; 0: bf16 dest accumulation (control)"""
import sys

import torch
import ttnn

M, K, N = 32, 4096, 64


def main():
    fp32_dest_acc = sys.argv[1:] != ["0"]
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()
    up = lambda t: ttnn.from_torch(t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    compute = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc, packer_l1_acc=False
    )
    out = ttnn.to_torch(ttnn.linear(up(a), up(b), compute_kernel_config=compute))
    ref = a.float() @ b.float()
    p = torch.corrcoef(torch.stack([out.flatten().double(), ref.flatten().double()]))[0, 1].item()
    print(f"fp32_dest_acc_en={fp32_dest_acc}: pcc={p:.5f}")
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
