# SPDX-License-Identifier: Apache-2.0
"""Prove preparation preserves an externally owned paged cache exactly."""

import torch

import ttnn

from ..tt.generator import GraniteGenerator


def test_external_cache_preserved():
    torch.set_num_threads(4)
    torch.manual_seed(814)
    from ..tt.generator_vllm import GraniteForCausalLM

    ttnn.set_fabric_config(**GraniteForCausalLM.model_capabilities["fabric_config"])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        cache = [
            [
                ttnn.zeros([4096, 2, 32, 128], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh)
                for _ in range(2)
            ]
        ]
        pages = ttnn.from_torch(
            torch.arange(32, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for t in cache[0]:
            data = ttnn.from_torch(
                torch.randn(1, 2, 1024, 128),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.experimental.paged_fill_cache(t, data, pages, batch_idx=0)
            del data

        def snapshot():
            values = []
            for t in cache[0]:
                prefix = ttnn.slice(t, [0, 0, 0, 0], [32, 2, 32, 128])
                values.append([ttnn.to_torch(x) for x in ttnn.get_device_tensors(prefix)])
                del prefix
            return values

        before = snapshot()
        gen = GraniteGenerator(mesh, override_num_layers=1, kv_cache=cache, batch_buckets=(1, 8, 16))
        gen.prepare()
        after = snapshot()
        assert all(torch.equal(a, b) for pa, pb in zip(before, after) for a, b in zip(pa, pb))
        gen.reset()
        reset = snapshot()
        assert all(torch.equal(a, b) for pa, pb in zip(before, reset) for a, b in zip(pa, pb))
        # Replaying after diagnostic reads proves no sliced temporary survives.
        ids = torch.tensor([[100283, 1234, 5678]])
        table = torch.arange(4096, dtype=torch.int32).reshape(1, -1)
        out = gen.prefill_forward(ids, page_table=table, kv_cache=cache, prompt_lens=[3])
        out = gen.decode_forward(out.reshape(1, 1), torch.tensor([3]), page_table=table, kv_cache=cache)
        assert out.numel() == 1
    finally:
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)
