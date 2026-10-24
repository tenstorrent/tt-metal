# SPDX-License-Identifier: Apache-2.0
"""Exact-shape TP4 scratch lifetime test; synthetic tags test protocol, not precision policy."""
import json
import os
import argparse
import hashlib
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.decode_workspace import DecodeAllReduceWorkspace

assert os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')=='1'
assert os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0')=='0'
base=Path('models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder')
p=argparse.ArgumentParser();p.add_argument('--output',default=str(base/'workspace_lifetime.json'));args=p.parse_args()
assert not Path(args.output).exists(), 'Preserve existing evidence'
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
traces=[]
try:
    pool=DecodeAllReduceWorkspace(mesh,ttnn.bfloat8_b)
    try:
        pool.borrow('out')
        raise AssertionError('Partial invocation was not rejected')
    except RuntimeError:
        pass
    memory=ttnn.create_sharded_memory_config([32,640],ttnn.CoreGrid(x=8,y=1),
        ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
    slots=[]
    for epoch,batch in enumerate([1,3,8,16,32]*3):
        users=torch.arange(batch).reshape(1,1,batch,1)%8
        tensors=[(users+rank+1+4*(epoch%4)).expand(1,1,batch,5120).contiguous().float() for rank in range(4)]
        inp=ttnn.from_torch(torch.cat(tensors,dim=0),device=mesh,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,
            memory_config=memory,mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0))
        out=ttnn.empty([1,1,batch,5120],device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        golden=4*sum(tensors)
        slots.append((inp,out,golden,batch))
    addresses={role:[t.buffer_address() for t in ttnn.get_device_tensors(buffer)] for role,(buffer,_) in pool.slots.items()}
    assert all(a!=b for a,b in zip(addresses['out'],addresses['down']))

    def forward(inp,out):
        pool.begin()
        value=inp
        try:
            for role in ('out','down'):
                scratch,semaphore=pool.borrow(role)
                if value.dtype!=pool.dtype: value=ttnn.typecast(value,pool.dtype)
                value=ttnn.experimental.all_reduce_async(value,scratch,cluster_axis=1,mesh_device=mesh,
                    multi_device_global_semaphore=semaphore,memory_config=memory,dtype=ttnn.bfloat16,
                    topology=ttnn.Topology.Ring,num_links=2)
            ttnn.copy(ttnn.to_memory_config(value,ttnn.DRAM_MEMORY_CONFIG),out)
            pool.finish()
        except BaseException:
            pool.abort()
            raise

    # Every input, retained output, scratch allocation, semaphore and program
    # exists before the first capture. Captures retain no new output tensor.
    for inp,out,_,_ in slots: forward(inp,out)
    ttnn.synchronize_device(mesh)
    for inp,out,_,_ in slots:
        tid=ttnn.begin_trace_capture(mesh,cq_id=0)
        forward(inp,out)
        ttnn.end_trace_capture(mesh,tid,cq_id=0)
        traces.append(tid)
    trace_ids=[str(tid) for tid in traces]
    for _ in range(100):
        for tid in traces: ttnn.execute_trace(mesh,tid,cq_id=0,blocking=False)
    ttnn.synchronize_device(mesh)
    ttnn.release_trace(mesh,traces[0]);traces[0]=None
    # Retiring one trace must not retire the shared buffers or the other traces.
    for tid in traces[1:]: ttnn.execute_trace(mesh,tid,cq_id=0,blocking=False)
    ttnn.synchronize_device(mesh)
    checked=0
    for _,out,golden,_ in slots:
        for device_out in ttnn.get_device_tensors(out):
            assert torch.equal(ttnn.to_torch(device_out).float(),golden)
            checked+=1
    result={'mesh':[1,4],'dtype':'bfloat8_b','batch_sequence':[s[3] for s in slots],
        'captures_before_traffic':15,'trace_ids':trace_ids,'nonblocking_replays':1514,
        'per_replay_host_syncs':0,'checked_outputs_all_devices':checked,'exact':True,
        'scratch_allocations':2,'scratch_bytes_per_device':pool.scratch_bytes_per_device,
        'scratch_addresses':addresses,'partial_invocation_rejected':True,'surviving_trace_after_retirement':True,
        'mid_traffic_releases':0,'post_traffic_release_before_survivor_check':1,
        'allocation_tracking':True,'program_cache_tracking':True,'acknowledgments':[],
        'scope':'Synthetic rank/epoch-tagged collective protocol; not a target-model precision or performance result.'}
    result['workspace_sha256']=hashlib.sha256((base.parents[1]/'tt/decode_workspace.py').read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('WORKSPACE_LIFETIME_PASS',flush=True)
finally:
    ttnn.synchronize_device(mesh)
    for tid in traces:
        if tid is not None: ttnn.release_trace(mesh,tid)
    ttnn.close_mesh_device(mesh)
