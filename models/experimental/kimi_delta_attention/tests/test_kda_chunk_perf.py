# SPDX-License-Identifier: Apache-2.0
# Perf loop analysis: recurrent (token-loop) vs chunked op — device time + kernel count. Real KDA dims.
import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.kimi_delta_attention.torch_functional import kda_ops as ref
from models.experimental.kimi_delta_attention.tt.ttnn_kda_ops import recurrent_kda_ttnn
from models.experimental.kimi_delta_attention.tt.ttnn_kda_chunk import chunk_kda_ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

torch.manual_seed(4)
_F2D = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
}


def _profile(mesh, run_fn):
    ttnn.synchronize_device(mesh)
    _, records = profile_realtime_program(mesh, run_fn, collect_all=True)
    per = {}
    for r in records:
        rid = r["runtime_id"]
        if rid:
            per[rid] = max(per.get(rid, 0.0), float(r["duration_ns"]))
    return sum(per.values()) / 1e3, len(per)  # (total device us, num programs)


@pytest.mark.parametrize("device_params", [_F2D], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize("T", [256, 640])
def test_recur_vs_chunk(mesh_device, T):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("realtime profiler not active")
    HV, K, V, C = 8, 128, 128, 64
    import torch.nn.functional as F
    q = ref.l2norm(torch.randn(1, T, HV, K)); k = ref.l2norm(torch.randn(1, T, HV, K))
    v = torch.randn(1, T, HV, V); g = -F.softplus(torch.randn(1, T, HV, K)); beta = torch.sigmoid(torch.randn(1, T, HV))

    def up(x):
        return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    qd, kd, vd, gd, bd = up(q), up(k), up(v), up(g), up(beta)

    recurrent_kda_ttnn(qd, kd, vd, gd, bd, device=mesh_device)      # warmup
    chunk_kda_ttnn(qd, kd, vd, gd, bd, device=mesh_device, chunk_size=C)
    us_r, n_r = _profile(mesh_device, lambda: recurrent_kda_ttnn(qd, kd, vd, gd, bd, device=mesh_device))
    us_c, n_c = _profile(mesh_device, lambda: chunk_kda_ttnn(qd, kd, vd, gd, bd, device=mesh_device, chunk_size=C))
    logger.info(
        f"[recur_vs_chunk] T={T}: recurrent {us_r:.0f}us/{n_r}k  chunk {us_c:.0f}us/{n_c}k  "
        f"=> {us_r/us_c:.1f}x faster, {n_r/max(n_c,1):.1f}x fewer kernels"
    )
