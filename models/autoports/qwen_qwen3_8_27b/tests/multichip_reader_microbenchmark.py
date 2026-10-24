# SPDX-License-Identifier: Apache-2.0
"""Alternating reader-count microbenchmarks on real-weight, recorded-input roles.

The traced role includes its input/output layout boundaries. Device windows are
signposted for Tracy; host times are labeled separately. Never run concurrently
with another hardware job.
"""
import argparse, json, time, os, copy
from pathlib import Path
import torch
import ttnn
from safetensors.torch import load_file
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import DEFAULT_POLICY, OptimizedDecoder

p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true');args=p.parse_args()
if args.profile:from tracy import signpost
torch.set_num_threads(8)
root=Path('.cache/qwen38-weights');cfg=AutoConfig.from_pretrained(str(root)).text_config;cfg._attn_implementation='eager'
w=load_file(str(root/f'layer_{args.layer}.safetensors'));prefix=f'model.language_model.layers.{args.layer}.'
ref=Qwen3_5DecoderLayer(cfg,args.layer).to(torch.bfloat16).eval();ref.load_state_dict({k.removeprefix(prefix):v for k,v in w.items()})
x=torch.load(root/f'recorded_layer_{args.layer}.pt',weights_only=True)[:,:130]
rope=Qwen3_5TextRotaryEmbedding(cfg);cs=rope(x,torch.arange(130).expand(3,1,-1));cache=DynamicCache(config=cfg)
inputs={}
mods={'mix':ref.linear_attn.in_proj_qkv,'out':ref.linear_attn.out_proj} if args.layer==0 else {'mix':ref.self_attn.q_proj,'out':ref.self_attn.o_proj}
mods.update(gateup=ref.mlp.gate_proj,down=ref.mlp.down_proj)
handles=[m.register_forward_pre_hook(lambda mod,a,r=r:inputs.__setitem__(r,a[0].detach().clone())) for r,m in mods.items()]
with torch.no_grad():
 mask=torch.where(torch.arange(129)[None,:]>torch.arange(129)[:,None],float('-inf'),0.).to(torch.bfloat16)[None,None]
 ref(x[:,:129],position_embeddings=(cs[0][:,:129],cs[1][:,:129]),past_key_values=cache,attention_mask=mask if args.layer==3 else None)
 ref(x[:,129:],position_embeddings=(cs[0][:,129:],cs[1][:,129:]),past_key_values=cache)
for h in handles:h.remove()
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
 models={}
 workspace=None
 for readers in (1,2,3):
  policy=dict(dram_sharded=True,fidelity='LoFi',mlp_dtype='bfloat4_b',attention_dtype='bfloat4_b',down_dtype='bfloat4_b',
              packed_mlp=True,carry_residual=False,l1_mlp=False,
              mix={'cores':40,'block':4},out={'cores':16,'block':3},down={'cores':17,'block':8},gateup={'cores':80,'block':2})
  policy.update(copy.deepcopy(json.loads(os.environ.get('QWEN_OPT_POLICY','{}'))))
  policy["sharded_decode_min_batch"]=0  # Isolated projection roles do not instantiate the batched fused view.
  for role in ("mix","out","gateup","gate","up","down"):
   policy.setdefault(role,{})["readers"]=readers
  models[readers]=MultichipDecoder.from_state_dict(w,hf_config=cfg,layer_idx=args.layer,mesh_device=mesh,policy=policy,ccl_workspace=workspace)
  workspace=models[readers].ar_workspace
 tx={r:ttnn.from_torch(v,device=mesh,layout=ttnn.TILE_LAYOUT,dtype=ttnn.bfloat16,mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=2) if r in ("out","down") else ttnn.ReplicateTensorToMesh(mesh)) for r,v in inputs.items()}
 def read(t):return torch.cat([ttnn.to_torch(v) for v in ttnn.get_device_tensors(t)],dim=-1)
 results=[]
 for role in mods:
  reference_output=None
  for repeat, readers in enumerate((1,2,3,3,2,1)):
   dec=models[readers];weight=getattr(dec,role)
   def operation():return OptimizedDecoder._linear(dec,tx[role],weight,activation='silu' if role=='gate' else None)
   warm=operation();control=read(warm).clone();del warm
   if reference_output is None:reference_output=control.clone()
   agreement=torch.corrcoef(torch.stack([control.double().flatten(),reference_output.double().flatten()]))[0,1].item()
   assert agreement>=.995
   ttnn.synchronize_device(mesh)
   tid=ttnn.begin_trace_capture(mesh,cq_id=0)
   out=operation();ttnn.end_trace_capture(mesh,tid,cq_id=0)
   try:
    ttnn.execute_trace(mesh,tid,cq_id=0,blocking=True)
    actual=read(out)
    assert torch.equal(control,actual)
    if args.profile:
     ttnn.ReadDeviceProfiler(mesh);signpost(f'MICRO_{role}_R{readers}_{repeat}')
    start=time.perf_counter()
    for _ in range(1 if args.profile else 50):ttnn.execute_trace(mesh,tid,cq_id=0,blocking=True)
    us=(time.perf_counter()-start)*1e6/(1 if args.profile else 50)
    if args.profile:
     signpost(f'MICRO_{role}_R{readers}_{repeat}_END');ttnn.ReadDeviceProfiler(mesh)
    k,n=weight.shape
    results.append(dict(role=role,readers=readers,repeat=repeat,host_us=us,k=k,n=n,logical_n=dec.logical_widths[id(weight)],weight_dtype=str(weight.dtype),fidelity='LoFi',exact_replay=True,pcc_vs_one_reader=agreement,config=dec.policy.get(role,{})))
   finally:ttnn.release_trace(mesh,tid)
   del out
 Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
finally:ttnn.close_mesh_device(mesh)
