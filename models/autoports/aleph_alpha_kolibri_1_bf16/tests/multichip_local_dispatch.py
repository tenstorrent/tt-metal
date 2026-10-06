# SPDX-License-Identifier: Apache-2.0
"""Independent per-column singleton-axis routing and cached-program regression."""

import json
from pathlib import Path

import torch

import ttnn

ROOT = Path(__file__).resolve().parents[1]


def run():
    torch.set_num_threads(8)
    torch.manual_seed(194)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    rows = []

    def tt(x, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            x.contiguous(),
            device=mesh,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def read(x):
        return [ttnn.to_torch(v) for v in ttnn.get_device_tensors(x)]

    try:
        for tokens in (32, 128):
            for repeat in range(3):
                x = torch.randn(4, tokens, 2560).bfloat16()
                ids = torch.stack([torch.arange(6).repeat(tokens, 1) + 13 * repeat + 41 * r for r in range(4)])
                ids[:, -1, 0] += 6
                counts = torch.stack([torch.bincount(v.flatten(), minlength=384) for v in ids]).int()
                aligned = (counts + 31) // 32 * 32
                offsets = torch.cumsum(aligned, 1).int() - aligned
                capacity = ((6 * tokens + 31 * 384 + 31) // 32) * 32
                dispatched, metadata = ttnn.experimental.deepseek_prefill.dispatch(
                    tt(x, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                    tt(ids, ttnn.uint16),
                    tt(offsets, ttnn.int32),
                    tt(torch.zeros(4, 384, dtype=torch.int32), ttnn.int32),
                    dispatch_group_size=1,
                    experts_per_chip=384,
                    num_routed_experts=384,
                    num_experts_per_tok=6,
                    metadata_len=3,
                    max_dispatch_buffer_token_size=capacity,
                    cluster_axis=0,
                    **({"num_links": 0} if repeat == 2 else {}),
                )
                for rank, (data, meta) in enumerate(zip(read(dispatched), read(metadata))):
                    data = data.reshape(capacity, 2560)
                    meta = meta.reshape(capacity, 3).long()
                    for expert in torch.nonzero(counts[rank]).flatten().tolist():
                        start, count = int(offsets[rank, expert]), int(counts[rank, expert])
                        for slot in range(start, start + count):
                            chip, token, route = meta[slot].tolist()
                            assert chip == rank and ids[rank, token, route] == expert
                            assert torch.equal(data[slot], x[rank, token])
                for layout in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
                    combined = ttnn.experimental.deepseek_prefill.combine(
                        ttnn.to_layout(dispatched, layout),
                        metadata,
                        tt(counts[:, None], ttnn.int32),
                        tt(offsets[:, None], ttnn.int32),
                        dispatch_group_size=1,
                        experts_per_chip=384,
                        num_experts_per_tok=6,
                        seq_len_per_chip=tokens,
                        cluster_axis=0,
                        **({"num_links": 0} if repeat == 2 else {}),
                    )
                    for rank, actual in enumerate(read(combined)):
                        actual = actual.reshape(tokens, 6, 2560)
                        assert torch.equal(actual, x[rank, :, None].expand_as(actual))
                row = dict(
                    tokens=tokens, repeat=repeat, ranks=4, exact=True, zero_count_experts=True, partial_batches=True
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
        (ROOT / "doc/multichip_decoder/local_dispatch_correctness.json").write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    run()
