# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Discriminate 2D output blocking versus DRAM weight layout with real tensors."""

import json

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.prepare_optimized_inputs import load_state
from models.autoports.meta_models_muse_glimmer_30b.tests.run_fused import ROOT, pcc, read

torch.set_num_threads(8)
state = load_state(0)
x = torch.load(ROOT / "doc/optimized_decoder/inputs/layer_0.pt", weights_only=True)[:, :1024].float()
x = (
    x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * (1 + state["input_layernorm.weight"].float())
).bfloat16()[:, None]
w = torch.cat([state[f"self_attn.{n}_proj.weight"] for n in ("q", "k", "v")]).t().contiguous()
expected = x @ w
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
rows = []
try:
    a = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    dg = mesh.dram_grid_size()
    wm = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))}),
            [6656, 4608 // 8],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    for layout, mem in [("interleaved", ttnn.DRAM_MEMORY_CONFIG), ("dram_sharded", wm)]:
        b = ttnn.from_torch(w, device=mesh, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, memory_config=mem)
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for bh, bw, sh, sw in [(1, 18, 1, 3), (4, 18, 1, 3), (4, 18, 4, 1), (1, 6, 1, 3), (4, 6, 1, 3), (4, 6, 4, 1)]:
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=4,
                out_subblock_h=sh,
                out_subblock_w=sw,
                out_block_h=bh,
                out_block_w=bw,
                per_core_M=4,
                per_core_N=18,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            row = dict(layout=layout, out_block=[bh, bw], subblock=[sh, sw])
            try:
                y = ttnn.linear(
                    a,
                    b,
                    compute_kernel_config=ck,
                    program_config=pc,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                got = read(y)
                row.update(
                    pcc=pcc(expected, got),
                    row_pcc=[pcc(expected[:, :, i : i + 32], got[:, :, i : i + 32]) for i in range(0, 1024, 32)],
                )
                del y
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)
            (ROOT / "doc/optimized_decoder/prefill_block_repro.json").write_text(json.dumps(rows, indent=2))
            print(row, flush=True)
        del b
finally:
    ttnn.close_mesh_device(mesh)
