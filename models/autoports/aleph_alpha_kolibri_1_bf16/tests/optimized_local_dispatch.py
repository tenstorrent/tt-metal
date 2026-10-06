# SPDX-License-Identifier: Apache-2.0
"""Singleton dispatch/combine repair: exact routing and cached-call regression."""
import json

import torch

import ttnn

from .optimized_coverage import ROOT
from .optimized_provenance import provenance


def run():
    torch.set_num_threads(8)
    torch.manual_seed(19)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0])
    rows = []

    def device(x, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(x, device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    try:
        for tokens in (32, 128):
            for repeat in range(3):
                x = torch.randn(1, tokens, 2560).bfloat16()
                ids = torch.arange(6).repeat(tokens, 1)
                ids[-1, 0] = 6
                ids = (ids + 13 * repeat) % 384
                counts = torch.bincount(ids.flatten(), minlength=384).to(torch.int32)
                aligned = ((counts + 31) // 32) * 32
                offsets = torch.cumsum(aligned, 0).int() - aligned
                capacity = ((6 * tokens + 31 * 384 + 31) // 32) * 32
                dispatched, metadata = ttnn.experimental.deepseek_prefill.dispatch(
                    device(x, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                    device(ids[None], ttnn.uint16),
                    device(offsets[None], ttnn.int32),
                    device(torch.zeros(1, 384, dtype=torch.int32), ttnn.int32),
                    dispatch_group_size=1,
                    experts_per_chip=384,
                    num_routed_experts=384,
                    num_experts_per_tok=6,
                    metadata_len=3,
                    max_dispatch_buffer_token_size=capacity,
                    cluster_axis=0,
                    **({"num_links": 0} if repeat == 2 else {}),
                )
                data = ttnn.to_torch(dispatched).reshape(capacity, 2560)
                meta = ttnn.to_torch(metadata).reshape(capacity, 3).long()
                for expert in torch.nonzero(counts).flatten().tolist():
                    start, count = int(offsets[expert]), int(counts[expert])
                    for slot in range(start, start + count):
                        chip, token, rank = meta[slot].tolist()
                        assert chip == 0 and ids[token, rank] == expert
                        assert torch.equal(data[slot], x[0, token])
                for layout in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
                    combined = ttnn.experimental.deepseek_prefill.combine(
                        ttnn.to_layout(dispatched, layout),
                        metadata,
                        device(counts[None, None], ttnn.int32),
                        device(offsets[None, None], ttnn.int32),
                        dispatch_group_size=1,
                        experts_per_chip=384,
                        num_experts_per_tok=6,
                        seq_len_per_chip=tokens,
                        cluster_axis=0,
                    )
                    actual = ttnn.to_torch(combined).reshape(tokens, 6, 2560)
                    assert torch.equal(actual, x[0, :, None].expand_as(actual))
                row = dict(
                    tokens=tokens,
                    repeat=repeat,
                    exact=True,
                    active_experts=int((counts > 0).sum()),
                    min_count=int(counts[counts > 0].min()),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
        (ROOT / "doc/optimized_decoder/local_dispatch_correctness.json").write_text(
            json.dumps(dict(provenance=provenance(), rows=rows), indent=2) + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    run()
