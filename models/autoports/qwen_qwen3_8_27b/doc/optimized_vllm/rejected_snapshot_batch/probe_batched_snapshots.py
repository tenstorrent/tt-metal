# SPDX-License-Identifier: Apache-2.0
"""Exact batched snapshot DMA, dynamic row counts and live trace safety."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import LinearState
from models.autoports.qwen_qwen3_8_27b.tt.serving_state import ServingStatePool

root=Path('models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm')
assert os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')=='1'
assert os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0')=='0'
result=dict(scope='Synthetic byte-exact snapshot primitive, two linear layers, B32/TP4; not model quality/performance',status='incomplete',cases=[])
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=1048576)
trace=None
try:
    torch.manual_seed(917)
    def put(x,dtype,layout):
        return ttnn.from_torch(x,dtype=dtype,layout=layout,device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    def upload(value,target):
        host=ttnn.from_torch(value,dtype=target.dtype,layout=target.layout,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        ttnn.copy_host_to_device_tensor(host,target)
    def exact(tensor,expected):
        return all(torch.equal(ttnn.to_torch(t),expected) for t in ttnn.get_device_tensors(tensor))
    original=[dict(conv=torch.randn(32,3,2560).bfloat16(),recurrent=torch.randn(32,12,128,128)) for _ in range(2)]
    active=[LinearState(put(row['conv'],ttnn.bfloat16,ttnn.ROW_MAJOR_LAYOUT),
                        put(row['recurrent'],ttnn.float32,ttnn.TILE_LAYOUT)) for row in original]
    model=SimpleNamespace(mesh=mesh,layer_indices=[0,7],layers=[SimpleNamespace(layer_type='linear_attention')]*2)
    pool=ServingStatePool(model,active,{0:64,7:64})
    scratch=ttnn.zeros_like(active[0].conv)
    pool.warmup()
    for expected,state in zip(original,active):
        assert all(exact(getattr(state,name),value) for name,value in expected.items())
    ttnn.copy(active[0].conv,scratch)
    ttnn.synchronize_device(mesh)
    trace=ttnn.begin_trace_capture(mesh,cq_id=0)
    ttnn.copy(active[0].conv,scratch)
    ttnn.end_trace_capture(mesh,trace,cq_id=0)
    programs=mesh.num_program_cache_entries()
    for slots in ([31],[7,0,31],list(reversed(range(32))),[0,31]):
        for row,state in zip(original,active):
            for name,value in row.items():upload(value,getattr(state,name))
        transfers={slot:{0:slot+1,7:32-slot} for slot in slots}
        before=pool.counters['metadata_uploads']
        pool.save_many(transfers)
        assert pool.counters['metadata_uploads']==before+1
        # Swap destinations while restoring; snapshot IDs retain source ownership.
        destination=list(reversed(slots))
        restore={target:transfers[source] for source,target in zip(slots,destination)}
        for row,state in zip(original,active):
            for name,value in row.items():upload(value+10,getattr(state,name))
        pool.restore_many(restore)
        assert pool.counters['metadata_uploads']==before+2
        for row,state in zip(original,active):
            for name,value in row.items():
                expected=value+10
                for source,target in zip(slots,destination):expected[target]=value[source]
                assert exact(getattr(state,name),expected),(slots,name)
        # No new compile or allocation exemption is needed for a new row count.
        ttnn.execute_trace(mesh,trace,cq_id=0,blocking=False)
        expected=original[0]['conv']+10
        for source,target in zip(slots,destination):expected[target]=original[0]['conv'][source]
        assert exact(scratch,expected)
        assert mesh.num_program_cache_entries()==programs
        result['cases'].append(dict(rows=len(slots),order=slots,all_four_replicas_exact=True,metadata_uploads=2,programs=programs))
    before=dict(pool.counters)
    pool.save_many({});pool.restore_many({})
    assert pool.counters==before
    try:pool.save_many({32:{0:1,7:1}})
    except ValueError:pass
    else:raise AssertionError('Invalid slot accepted')
    assert pool.counters==before
    result.update(status='pass',trace_id=str(trace),captures=1,retirements_during_workload=0,
                  counters=pool.counters,programs=programs,tracking=True,program_cache_included=True,
                  metadata_shape=list(pool.host_ids.shape),metadata_bytes_per_device=pool.host_ids.numel()*4)
finally:
    if trace is not None:
        ttnn.synchronize_device(mesh);ttnn.release_trace(mesh,trace)
    ttnn.close_mesh_device(mesh)
    (root/'snapshot_batch_contract.json').write_text(json.dumps(result,indent=2)+'\n')
