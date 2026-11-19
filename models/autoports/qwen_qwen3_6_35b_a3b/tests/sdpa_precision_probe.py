# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolate long-context paged SDPA accumulation using the target head geometry."""

import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import pcc


def main():
    torch.set_num_threads(8)
    torch.manual_seed(192)
    n = 262144
    q = torch.randn(1, 1, 16, 256).bfloat16()
    k = torch.randn(n // 32, 2, 32, 256).bfloat16()
    v = torch.randn_like(k)
    keys = k.permute(1, 0, 2, 3).reshape(2, n, 256).repeat_interleave(8, dim=0).float()
    values = v.permute(1, 0, 2, 3).reshape(2, n, 256).repeat_interleave(8, dim=0).float()
    scores = torch.matmul(q.reshape(16, 1, 256).float(), keys.transpose(-1, -2)) / 16
    scores[:, :, -1] = -torch.inf
    expected = torch.matmul(scores.softmax(-1), values).reshape(1, 1, 16, 256)
    del keys, values, scores
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:

        def upload(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
            return ttnn.from_torch(x, device=mesh, layout=layout, dtype=dtype)

        tk, tv = upload(k), upload(v)
        table = upload(torch.arange(n // 32).int().reshape(1, -1), ttnn.ROW_MAJOR_LAYOUT, ttnn.int32)
        pos = upload(torch.tensor([n - 2]).int(), ttnn.ROW_MAJOR_LAYOUT, ttnn.int32)
        shard = ttnn.create_sharded_memory_config(
            (32, 256),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tq = ttnn.to_memory_config(upload(q), shard)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        for grid, chunk in (((1, 1), 32), ((8, 8), 32), ((1, 1), 256), ((8, 8), 256)):
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tq,
                tk,
                tv,
                cur_pos_tensor=pos,
                page_table_tensor=table,
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid,
                    q_chunk_size=32,
                    k_chunk_size=chunk,
                ),
                compute_kernel_config=compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            got = ttnn.to_torch(out).reshape(expected.shape).float()
            row = dict(
                grid=grid,
                k_chunk=chunk,
                pcc=pcc(expected, got),
                relative_rms=float((got - expected).square().mean().sqrt() / expected.square().mean().sqrt()),
                rms_ratio=float(got.square().mean().sqrt() / expected.square().mean().sqrt()),
            )
            rows.append(row)
            print("SDPA_PROBE", row, flush=True)
            del out
    finally:
        ttnn.close_mesh_device(mesh)
    destination = Path(__file__).parents[1] / "doc/functional_decoder/sdpa_precision_probe.json"
    destination.write_text(json.dumps(dict(length=n - 1, rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
