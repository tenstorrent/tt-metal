# SPDX-License-Identifier: Apache-2.0
"""Real-weight mixed-batch trace ownership; no performance/accuracy-policy ranking."""
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

p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--output',required=True);args=p.parse_args()
assert not Path(args.output).exists()
assert os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')=='1'
assert os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0')=='0'
torch.set_num_threads(8);torch.manual_seed(982)
root=Path(__file__).resolve().parents[1]
cfg=AutoConfig.from_pretrained(str(root/'doc/functional_decoder/hf_config.json')).text_config
weights=load_file(f'.cache/qwen38-weights/layer_{args.layer}.safetensors')
recorded=torch.load(f'.cache/qwen38-weights/recorded_layer_{args.layer}.pt',weights_only=True)
rope=Qwen3_5TextRotaryEmbedding(cfg)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
traces=[]
try:
    model=MultichipDecoder.from_state_dict(weights,hf_config=cfg,layer_idx=args.layer,mesh_device=mesh)
    model.prefill_forward=audit_runtime(model.prefill_forward)
    model.decode_forward=audit_runtime(model.decode_forward)
    contexts=[]
    def put(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,sharded=False):
        mapper=ttnn.ShardTensorToMesh(mesh,dim=2) if sharded else ttnn.ReplicateTensorToMesh(mesh)
        return ttnn.from_torch(t.contiguous(),device=mesh,dtype=dtype,layout=layout,mesh_mapper=mapper)
    def restore(ctx):
        source,dest=ctx['backup'],ctx['state']
        for a,b in ((source.conv,dest.conv),(source.recurrent,dest.recurrent)) if isinstance(source,LinearState) else ((source.key,dest.key),(source.value,dest.value)):
            ttnn.copy(a,b)
    def forward(ctx):
        restore(ctx)
        y=model.decode_forward(ctx['input'],state=ctx['state'],page_table=ctx['table'],current_pos=ctx['position'],cos=ctx['cos'],sin=ctx['sin'])
        ttnn.copy(ttnn.to_memory_config(y,ttnn.DRAM_MEMORY_CONFIG),ctx['output'])
    def read(ctx):
        values=[ttnn.to_torch(t) for t in ttnn.get_device_tensors(ctx['output'])]
        if ctx['sharded']:return torch.cat(values,dim=-1)
        assert all(torch.equal(values[0],v) for v in values[1:])
        return values[0]
    for batch in (1,4,8,16,32):
        idx=(torch.arange(130)[None,:]+17*torch.arange(batch)[:,None])%recorded.shape[1]
        x=recorded[0,idx]
        with torch.no_grad():c,s=rope(x,torch.arange(130).expand(3,batch,-1))
        state=model.allocate_state(batch_size=batch,context_length=256)
        table=put(torch.randperm(batch*8,dtype=torch.int32).reshape(batch,8),ttnn.int32,ttnn.ROW_MAJOR_LAYOUT)
        pre=model.prefill_forward(put(x[:,:129],sharded=True),state=state,page_table=table,cos=put(c[:,None,:129]),sin=put(s[:,None,:129]))
        del pre
        sharded=model.decode_residual_layout(batch)=='sharded'
        ctx=dict(batch=batch,sharded=sharded,state=state,backup=model.snapshot(state),table=table,
            input=put(x[:,129:],sharded=sharded),cos=put(c[:,None,129:]),sin=put(s[:,None,129:]),
            position=put(torch.full((batch,),129,dtype=torch.int32),ttnn.int32,ttnn.ROW_MAJOR_LAYOUT),
            output=ttnn.empty([batch,1,1280 if sharded else 5120],device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG))
        contexts.append(ctx)
    # All state, inputs, outputs, persistent collectives and programs are ready
    # before any capture. No fresh captured output needs an acknowledgment.
    for ctx in contexts:
        forward(ctx);ctx['golden']=read(ctx).clone()
    ttnn.synchronize_device(mesh)
    for ctx in contexts:
        tid=ttnn.begin_trace_capture(mesh,cq_id=0)
        forward(ctx)
        ttnn.end_trace_capture(mesh,tid,cq_id=0)
        traces.append(tid)
    ids=[str(t) for t in traces]
    for repeat in range(64):
        for index in ((0,4,1,3,2) if repeat%2 else (4,0,2,1,3)):
            ttnn.execute_trace(mesh,traces[index],cq_id=0,blocking=False)
    ttnn.synchronize_device(mesh)
    assert all(torch.equal(read(ctx),ctx['golden']) for ctx in contexts)
    ttnn.release_trace(mesh,traces[0]);traces[0]=None
    for tid in traces[1:]:ttnn.execute_trace(mesh,tid,cq_id=0,blocking=False)
    ttnn.synchronize_device(mesh)
    assert all(torch.equal(read(ctx),ctx['golden']) for ctx in contexts[1:])
    sources=['tt/multichip_decoder.py','tt/optimized_decoder.py','tt/decode_workspace.py',str(Path(__file__).relative_to(root))]
    if (root/'tt/fused_decode.py').exists():sources.append('tt/fused_decode.py')
    if (root/'tt/fused_prefill.py').exists():sources.append('tt/fused_prefill.py')
    result=dict(layer_type=cfg.layer_types[args.layer],mesh=[1,4],batches=[1,4,8,16,32],logical_prefix_tokens=129,
        decode_layouts=[model.decode_residual_layout(b) for b in (1,4,8,16,32)],trace_ids=ids,
        captures_before_traffic=5,nonblocking_replays=324,per_replay_host_syncs=0,exact=True,
        distinct_real_user_inputs=True,allocation_tracking=True,program_cache_tracking=True,acknowledgments=[],
        retired_trace_survivors=4,shared_weights=True,runtime_fallback_audit=True,
        provenance={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
        scope='Single real decoder; state restored on device inside each trace. Exact eager/replay protocol check, not performance or model-accuracy evidence.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('MIXED_DECODER_TRACES_PASS',flush=True)
finally:
    ttnn.synchronize_device(mesh)
    for tid in traces:
        if tid is not None:ttnn.release_trace(mesh,tid)
    ttnn.close_mesh_device(mesh)
