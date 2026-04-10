# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import ttnn
import torch
import utils

def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten().to(torch.float64), y.flatten().to(torch.float64)
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return ((vx @ vy) / denom).item()

def test_sort():
    device = utils.DeviceGetter.get_device((1, 1))
    mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )

    # 1. Provide the exact distribution that caused the network graph to fail!
    cpu_logits = torch.load("topk_logits_level_0.pt", map_location="cpu").to(torch.bfloat16)
    
    # 2. Transfer exactly to TTNN via standard layout primitives
    tt_logits = ttnn.from_torch(
        cpu_logits,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device
    )

    # TT inference: ttnn.sort
    tt_vals, tt_indices = ttnn.sort(
        tt_logits, 1, True, False, memory_config=mem_cfg
    )
    tt_vals_torch = ttnn.to_torch(tt_vals)
    tt_indices_torch = ttnn.to_torch(tt_indices)

    # CPU inference: torch.sort on the explicitly identical tensor
    cpu_vals, cpu_indices = torch.sort(cpu_logits, dim=1, descending=True, stable=False)

    # Comparison
    vals_pcc = compute_pcc(cpu_vals, tt_vals_torch)
    indices_pcc = compute_pcc(cpu_indices, tt_indices_torch)

    print("\n" + "=" * 60)
    print("  SORT STABILITY PROOF (NPU VS CPU)")
    print("=" * 60)
    print(f"  {'Input shape':<25}: {cpu_logits.shape}")
    print(f"  {'Input dtype':<25}: {cpu_logits.dtype}")
    print("-" * 60)
    print(f"  {'CPU indices dtype':<25}: {cpu_indices.dtype}")
    print(f"  {'TT  indices dtype':<25}: {tt_indices_torch.dtype}")
    print("-" * 60)
    print(f"  {'Values PCC':<25}: {vals_pcc}")
    print(f"  {'Indices PCC':<25}: {indices_pcc}")
    print(f"  {'Indices torch.equal':<25}: {torch.equal(cpu_indices, tt_indices_torch.to(cpu_indices.dtype))}")
    print("-" * 60)
    print(f"  CPU indices (First 20): {cpu_indices[0, :20].tolist()}")
    print(f"  TT  indices (First 20): {tt_indices_torch[0, :20].to(torch.int32).tolist()}")
    print("=" * 60)

if __name__ == "__main__":
    test_sort()
