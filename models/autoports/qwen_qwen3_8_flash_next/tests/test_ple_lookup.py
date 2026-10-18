"""Compressed lookup tests; these do not claim full decoder correctness."""
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.ple_lookup import lookup_rows


def test_changed_indices_trace():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        torch.manual_seed(64)
        tables = [
            ttnn.from_torch(torch.randn(76, 160), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh)
            for _ in range(3)
        ]
        # Decode each shard separately so the control preserves padding at shard boundaries.
        reference = torch.cat([ttnn.to_torch(t).to(torch.bfloat16) for t in tables])
        ids = torch.arange(16, dtype=torch.int32).reshape(1, 16)
        indices = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        output = ttnn.allocate_tensor_on_device(
            (1, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        lookup_rows(tables, indices, output=output)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        lookup_rows(tables, indices, output=output)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for seed in (1, 2, 1):
            torch.manual_seed(seed)
            ids = torch.randint(0, 228, (1, 16), dtype=torch.int32)
            host = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, indices)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(output)
            expected = reference[ids.long()].reshape(1, 2560)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print("PLE_LOOKUP_CHANGED_INPUT_REPEATED_TRACE_BIT_EXACT")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
