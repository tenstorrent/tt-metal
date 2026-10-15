import torch, ttnn, json
from tests.ttnn.unit_tests.operations.matmul_reduce_scatter.test_matmul_reduce_scatter import (
    _open_mesh,
    _close_mesh,
    _stacked,
    _to_mesh,
)
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter
from eval.golden_tests.matmul_reduce_scatter.topology import system_mesh_shape

mesh = _open_mesh(system_mesh_shape(), ttnn.FabricConfig.FABRIC_2D)
try:
    a = _to_mesh(_stacked(mesh, (1, 1, 256, 512)), mesh, ttnn.bfloat16)
    w = _to_mesh(_stacked(mesh, (512, 1024)), mesh, ttnn.bfloat16)
    for label, fn in (
        ("add", lambda: ttnn.add(a, a)),
        ("mrs", lambda: matmul_reduce_scatter(a, w, cluster_axis=1, scatter_dim=-1, num_links=1)),
        ("mrs2", lambda: matmul_reduce_scatter(a, w, cluster_axis=1, scatter_dim=-1, num_links=1)),
    ):
        mesh.disable_and_clear_program_cache()
        ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = fn()
        trace = ttnn.graph._cpp_end_graph_capture()
        usage = ttnn.graph.extract_resource_usage_per_core(trace)
        kinds = {}
        for n in trace:
            kinds[n.get("node_type")] = kinds.get(n.get("node_type"), 0) + 1
        print(
            "PROBE", label, "peak_total", usage.peak_total, "peak_cb", getattr(usage, "peak_cb", None), "kinds", kinds
        )
        mesh.enable_program_cache()
finally:
    _close_mesh(mesh)
