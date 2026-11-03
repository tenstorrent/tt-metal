# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate native split-half rotary on irregular batch shard grids."""
import json

import torch

import ttnn

from .run_decoder import ROOT, pcc

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0])
try:
    torch.manual_seed(124)
    rows = []
    for b in (1, 2, 11, 12, 13, 16, 32):
        x = torch.randn(1, b, 32, 128).bfloat16()
        angles = torch.randn(1, b, 1, 64).repeat(1, 1, 1, 2)
        cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
        expected = x * cos + torch.cat([-x[..., 64:], x[..., :64]], -1) * sin
        grid = ttnn.num_cores_to_corerangeset(b, mesh.compute_with_storage_grid_size(), True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        inputs = [
            ttnn.from_torch(v, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mem)
            for v in (x, cos, sin)
        ]
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        out = ttnn.experimental.rotary_embedding_hf(*inputs, is_decode_mode=True, compute_kernel_config=compute)
        actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        row = dict(
            batch=b, pcc=pcc(actual, expected), per_user_pcc=[pcc(actual[:, u], expected[:, u]) for u in range(b)]
        )
        print(row, flush=True)
        rows.append(row)
    (ROOT / "doc/fused_decoder/rope_probe.json").write_text(json.dumps(rows, indent=2))
finally:
    ttnn.close_mesh_device(mesh)
