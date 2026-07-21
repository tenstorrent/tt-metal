# SPDX-License-Identifier: Apache-2.0
# Perf: C++ fused chunk_kda vs composed chunk_kda_ttnn — device time + kernel count.
import pytest, torch, ttnn
import torch.nn.functional as F
from loguru import logger
from models.experimental.kimi_delta_attention.torch_functional import kda_ops as ref
from models.experimental.kimi_delta_attention.tt.ttnn_kda_chunk import chunk_kda_ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
torch.manual_seed(6)
_F2D = {"fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT}

def _prof(mesh, fn):
    ttnn.synchronize_device(mesh); _, recs = profile_realtime_program(mesh, fn, collect_all=True)
    per={}
    for r in recs:
        if r["runtime_id"]: per[r["runtime_id"]]=max(per.get(r["runtime_id"],0.0),float(r["duration_ns"]))
    return sum(per.values())/1e3, len(per)

@pytest.mark.parametrize("device_params", [_F2D], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8,1)], indirect=True)
@pytest.mark.parametrize("T", [256, 640])
def test_cpp_vs_composed(mesh_device, T):
    if not ttnn.device.IsProgramRealtimeProfilerActive(): pytest.fail("rt profiler off")
    HV,K,V,C = 8,128,128,32
    q=ref.l2norm(torch.randn(1,T,HV,K)); k=ref.l2norm(torch.randn(1,T,HV,K))
    v=torch.randn(1,T,HV,V); g=-F.softplus(torch.randn(1,T,HV,K)); beta=torch.sigmoid(torch.randn(1,T,HV))
    up=lambda x: ttnn.from_torch(x,dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=mesh_device,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    qd,kd,vd,gd,bd=up(q),up(k),up(v),up(g),up(beta)
    cpp=lambda: ttnn.transformer.chunk_kda(qd,kd,vd,gd,bd,scale=K**-0.5,chunk_size=C)
    comp=lambda: chunk_kda_ttnn(qd,kd,vd,gd,bd,device=mesh_device,chunk_size=64)
    cpp(); comp()  # warmup
    us_x,n_x=_prof(mesh_device,cpp); us_c,n_c=_prof(mesh_device,comp)
    logger.info(f"[cpp_vs_composed] T={T}: C++ {us_x:.0f}us/{n_x}k  composed {us_c:.0f}us/{n_c}k => {us_c/us_x:.1f}x faster, {n_c/max(n_x,1):.0f}x fewer kernels")
