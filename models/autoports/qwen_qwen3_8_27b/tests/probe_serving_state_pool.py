# SPDX-License-Identifier: Apache-2.0
"""Exact snapshot DMA and allocation-tracked replay probe; no model weights."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import LinearState
from models.autoports.qwen_qwen3_8_27b.tt.serving_state import ServingStatePool

root=Path('models/autoports/qwen_qwen3_8_27b/readiness_vllm')
result={'scope':'snapshot copy primitive on1x4 mesh, synthetic distinct linear states; no model metric'}
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=1048576)
trace=None
try:
    torch.manual_seed(917)
    def put(x,dtype,layout):
        return ttnn.from_torch(x,dtype=dtype,layout=layout,device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    def read(t):return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()
    def upload(value,target):
        host=ttnn.from_torch(value,dtype=target.dtype,layout=target.layout,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        ttnn.copy_host_to_device_tensor(host,target)
    original=[dict(conv=torch.randn(2,3,2560).bfloat16(),recurrent=torch.randn(2,12,128,128)) for _ in range(2)]
    active=[LinearState(put(row['conv'],ttnn.bfloat16,ttnn.ROW_MAJOR_LAYOUT),
                        put(row['recurrent'],ttnn.float32,ttnn.TILE_LAYOUT)) for row in original]
    model=SimpleNamespace(mesh=mesh,layer_indices=[0,7],layers=[SimpleNamespace(layer_type='linear_attention')]*2)
    pool=ServingStatePool(model,active,{0:5,7:5})
    scratch=ttnn.zeros_like(active[0].conv)
    pool.warmup()
    for row,state in zip(original,active):
        assert torch.equal(read(state.conv),row['conv'])
        assert torch.equal(read(state.recurrent),row['recurrent'])
    ttnn.copy(active[0].conv,scratch)
    ttnn.synchronize_device(mesh)
    trace=ttnn.begin_trace_capture(mesh,cq_id=0)
    ttnn.copy(active[0].conv,scratch)
    ttnn.end_trace_capture(mesh,trace,cq_id=0)
    pool.save(0,{0:2,7:4})
    saved=[{name:read(getattr(pool.pools[index],name)) for name in ('conv','recurrent')} for index in (0,7)]
    for row,state in zip(original,active):
        upload(row['conv']+10,state.conv)
        upload(row['recurrent']+10,state.recurrent)
    pool.restore(1,{0:2,7:4})
    exact=[]
    for row,state in zip(original,active):
        for name in ('conv','recurrent'):
            actual=read(getattr(state,name))
            assert torch.equal(actual[1],row[name][0]),name
            assert torch.equal(actual[0],(row[name]+10)[0]),name
            exact.append(dict(field=name,restored_exact=True,peer_unchanged=True))
    # Different physical IDs share exactly the same prepared execution signature.
    pool.save(1,{0:3,7:1})
    pool.restore(0,{0:3,7:1})
    ttnn.execute_trace(mesh,trace,cq_id=0,blocking=False)
    assert torch.equal(read(scratch)[0],original[0]['conv'][0])
    for ordinal,index in enumerate((0,7)):
        old_block=2 if index==0 else 4
        for name in ('conv','recurrent'):
            assert torch.equal(read(getattr(pool.pools[index],name))[old_block],saved[ordinal][name][old_block]),name
    result.update(status='pass',checks=exact,older_snapshot_immutable=True,
                  trace_id=str(trace),captures=1,retirements_during_workload=0,
                  counters=pool.counters,allocated_bytes_per_device=pool.allocated_bytes_per_device)
finally:
    if trace is not None:
        ttnn.synchronize_device(mesh)
        ttnn.release_trace(mesh,trace)
    ttnn.close_mesh_device(mesh)
    (root/'state_pool_probe.json').write_text(json.dumps(result,indent=2)+'\n')
