# SPDX-License-Identifier: Apache-2.0
"""Real TP4 layers sharing prefill/decode scratch across live trace signatures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import ttnn
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import LinearState
from models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder import audit_runtime

p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
assert not Path(args.output).exists()
assert os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')=='1'
assert os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0')=='0'
root=Path(__file__).resolve().parents[1]
cfg=AutoConfig.from_pretrained(str(root/'doc/functional_decoder/hf_config.json')).text_config
rope=Qwen3_5TextRotaryEmbedding(cfg)
torch.set_num_threads(8);torch.manual_seed(983)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
traces=[];contexts=[];models=[]
try:
    def put(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,sharded=False):
        mapper=ttnn.ShardTensorToMesh(mesh,dim=2) if sharded else ttnn.ReplicateTensorToMesh(mesh)
        return ttnn.from_torch(t.contiguous(),device=mesh,dtype=dtype,layout=layout,mesh_mapper=mapper)
    def restore(ctx):
        src,dst=ctx['backup'],ctx['state']
        for a,b in ((src.conv,dst.conv),(src.recurrent,dst.recurrent)) if isinstance(src,LinearState) else ((src.key,dst.key),(src.value,dst.value)):
            ttnn.copy(a,b)
    def forward(ctx):
        restore(ctx)
        kw=dict(state=ctx['state'],page_table=ctx['table'],cos=ctx['cos'],sin=ctx['sin'])
        if ctx['phase']=='prefill':y=ctx['model'].prefill_forward(ctx['input'],**kw)
        else:y=ctx['model'].decode_forward(ctx['input'],current_pos=ctx['position'],**kw)
        ttnn.copy(ttnn.to_memory_config(y,ttnn.DRAM_MEMORY_CONFIG),ctx['output'])
    def read(ctx):
        values=[ttnn.to_torch(t) for t in ttnn.get_device_tensors(ctx['output'])]
        if ctx['sharded']:return torch.cat(values,dim=-1)
        assert all(torch.equal(values[0],v) for v in values[1:])
        return values[0]
    def sink(batch,length,width):
        return ttnn.empty([batch,length,width],device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for layer in (0,3):
        weights=load_file(f'.cache/qwen38-weights/layer_{layer}.safetensors')
        recorded=torch.load(f'.cache/qwen38-weights/recorded_layer_{layer}.pt',weights_only=True)
        model=MultichipDecoder.from_state_dict(weights,hf_config=cfg,layer_idx=layer,mesh_device=mesh,
            ccl_workspace=models[0].ar_workspace if models else None)
        models.append(model)
        model.prefill_forward=audit_runtime(model.prefill_forward);model.decode_forward=audit_runtime(model.decode_forward)
        x=recorded[:,torch.arange(4098)%recorded.shape[1]]
        with torch.no_grad():c,s=rope(x,torch.arange(4098).expand(3,1,-1))
        state=model.allocate_state(batch_size=1,context_length=8192);empty=model.snapshot(state)
        table=put(torch.randperm(256,dtype=torch.int32)[None],ttnn.int32,ttnn.ROW_MAJOR_LAYOUT)
        for length in (4095,4096,4097):
            contexts.append(dict(model=model,layer=layer,phase='prefill',batch=1,length=length,sharded=True,
                state=state,backup=empty,table=table,input=put(x[:,:length],sharded=True),
                cos=put(c[:,None,:length]),sin=put(s[:,None,:length]),output=sink(1,length,1280)))
        # Establish a filled4096 state for the independent decode replay snapshot.
        forward(contexts[-2]);filled=model.snapshot(state)
        contexts.append(dict(model=model,layer=layer,phase='decode',batch=1,length=1,sharded=False,
            state=state,backup=filled,table=table,input=put(x[:,4096:4097]),
            cos=put(c[:,None,4096:4097]),sin=put(s[:,None,4096:4097]),output=sink(1,1,5120),
            position=put(torch.tensor([4096],dtype=torch.int32),ttnn.int32,ttnn.ROW_MAJOR_LAYOUT)))
        idx=(torch.arange(130)[None,:]+17*torch.arange(32)[:,None])%recorded.shape[1];xb=recorded[0,idx]
        with torch.no_grad():cb,sb=rope(xb,torch.arange(130).expand(3,32,-1))
        bs=model.allocate_state(batch_size=32,context_length=256)
        bt=put(torch.randperm(256,dtype=torch.int32).reshape(32,8),ttnn.int32,ttnn.ROW_MAJOR_LAYOUT)
        if layer==3:
            contexts.append(dict(model=model,layer=layer,phase='prefill',batch=32,length=129,sharded=True,
                state=bs,backup=model.snapshot(bs),table=bt,input=put(xb[:,:129],sharded=True),
                cos=put(cb[:,None,:129]),sin=put(sb[:,None,:129]),output=sink(32,129,1280)))
        out=model.prefill_forward(put(xb[:,:129],sharded=True),state=bs,page_table=bt,cos=put(cb[:,None,:129]),sin=put(sb[:,None,:129]));del out
        contexts.append(dict(model=model,layer=layer,phase='decode',batch=32,length=1,sharded=True,
            state=bs,backup=model.snapshot(bs),table=bt,input=put(xb[:,129:],sharded=True),
            cos=put(cb[:,None,129:]),sin=put(sb[:,None,129:]),output=sink(32,1,1280),
            position=put(torch.full((32,),129,dtype=torch.int32),ttnn.int32,ttnn.ROW_MAJOR_LAYOUT)))
    owner=models[0].ar_workspace
    assert all(m.ar_workspace is owner and m.ccl is owner.ccl for m in models)
    def addresses():return {role:[t.buffer_address() for t in pair] for role,pair in owner.prefill_rs_buffers.items()}
    before=addresses()
    for ctx in contexts:
        forward(ctx);ctx['golden']=read(ctx).clone()
    ttnn.synchronize_device(mesh)
    for ctx in contexts:
        tid=ttnn.begin_trace_capture(mesh,cq_id=0);forward(ctx);ttnn.end_trace_capture(mesh,tid,cq_id=0);traces.append(tid)
    ids=[str(t) for t in traces]
    for repeat in range(32):
        order=range(len(traces)) if repeat%2 else reversed(range(len(traces)))
        for i in order:ttnn.execute_trace(mesh,traces[i],cq_id=0,blocking=False)
    ttnn.synchronize_device(mesh)
    assert all(torch.equal(read(ctx),ctx['golden']) for ctx in contexts)
    assert addresses()==before
    ttnn.release_trace(mesh,traces[0]);traces[0]=None
    for tid in traces[1:]:ttnn.execute_trace(mesh,tid,cq_id=0,blocking=False)
    ttnn.synchronize_device(mesh)
    assert all(torch.equal(read(ctx),ctx['golden']) for ctx in contexts[1:]) and addresses()==before
    sources=['tt/multichip_decoder.py','tt/optimized_decoder.py','tt/decode_workspace.py','tt/fused_decode.py','tt/fused_prefill.py',str(Path(__file__).relative_to(root))]
    result=dict(mesh=[1,4],layers=[0,3],shared_owners=1,signatures=[{k:ctx[k] for k in ('layer','phase','batch','length')} for ctx in contexts],
        captures_before_traffic=len(traces),trace_ids=ids,nonblocking_replays=32*len(traces)+len(traces)-1,
        exact=True,stable_persistent_addresses=True,prefill_rs_bytes_per_device=55705600,
        allocation_tracking=True,program_cache_tracking=True,acknowledgments=[],retired_trace_survivors=len(traces)-1,
        provenance={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
        scope='Real-weight lifetime/replay check with restored state, not performance or accuracy-policy evidence.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print('MIXED_PREFILL_DECODE_TRACES_PASS',flush=True)
finally:
    ttnn.synchronize_device(mesh)
    for tid in traces:
        if tid is not None:ttnn.release_trace(mesh,tid)
    ttnn.close_mesh_device(mesh)
