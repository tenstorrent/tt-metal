import torch, ttnn, collections
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

cache = _FabricMeshCache()
mesh = cache.get(None)


def mk(shape, mc):
    x = torch.randn((2, 2, *shape))
    g = torch.cat([torch.cat(list(x[r]), dim=1) for r in range(2)], dim=0)
    return ttnn.from_torch(
        g,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mc,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(2, 2)),
    )


for run in range(2):
    tx = mk((1, 1, 64, 256), ttnn.DRAM_MEMORY_CONFIG)
    mesh.disable_and_clear_program_cache()
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = distributed_rms_norm(tx, cluster_axis=1)
    trace = ttnn.graph._cpp_end_graph_capture()
    mesh.enable_program_cache()
    usage = ttnn.graph.extract_resource_usage_per_core(trace)
    kinds = collections.Counter(n.get("node_type") for n in trace)
    print("PROBE run", run, "peak_total", usage.peak_total, "nodes", dict(kinds))
    cbs = [n for n in trace if "circular" in str(n.get("node_type"))]
    print("PROBE cb nodes", len(cbs), cbs[:2])
    bufs = [n for n in trace if n.get("node_type") in ("buffer_allocate",)]
    print("PROBE buffer allocs", [(n.get("params", {}).get("type"), n.get("params", {}).get("size")) for n in bufs][:6])
cache.close()
