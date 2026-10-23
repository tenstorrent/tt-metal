# Verifier probe (not collected: run with `scripts/run_safe_pytest.sh --dev <this file>`; pytest collects it when named explicitly).
import collections
import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401
from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_input_tensor
from ttnn.operations.distributed_rms_norm import distributed_rms_norm


def cap(mesh_device, fn, tag):
    mesh_device.disable_and_clear_program_cache()
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    fn()
    trace = ttnn.graph._cpp_end_graph_capture()
    mesh_device.enable_program_cache()
    types = collections.Counter(n.get("node_type") for n in trace)
    print("CAPTURE", tag, dict(types), "PEAK", ttnn.graph.extract_resource_usage_per_core(trace).peak_total)


def test_probe(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, 1, 1, 96, 384)).to(torch.bfloat16)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cap(mesh_device, lambda: distributed_rms_norm(tx, cluster_axis=1), "first-call-new-shape")
    cap(mesh_device, lambda: distributed_rms_norm(tx, cluster_axis=1), "second-call")
    cap(mesh_device, lambda: ttnn.add(tx, tx), "ttnn.add")
