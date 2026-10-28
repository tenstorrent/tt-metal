import torch, ttnn, collections
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

cache = _FabricMeshCache()
mesh = cache.get((2, 2))
x = ttnn.from_torch(
    torch.randn(2, 2 * 1792 * 0 + 1 * 32, 2 * 1792).reshape(1, 1, 64, 3584) if False else torch.randn(64, 3584),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(2, 2)),
)


def cap(tag):
    mesh.disable_and_clear_program_cache()
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = distributed_rms_norm(x, cluster_axis=1, num_links=1)
    trace = ttnn.graph._cpp_end_graph_capture()
    mesh.enable_program_cache()
    usage = ttnn.graph.extract_resource_usage_per_core(trace)
    kinds = collections.Counter(n["node_type"] for n in trace)
    print("PROBE", tag, "peak_total", usage.peak_total, "peak_cb", getattr(usage, "peak_cb", None), dict(kinds))
    for n in trace:
        if n["node_type"] in ("circular_buffer_allocate",):
            print("PROBE   cb", n.get("params", {}))
            break


cap("cold")
cap("warm")
ttnn.synchronize_device(mesh)
cache.close()
