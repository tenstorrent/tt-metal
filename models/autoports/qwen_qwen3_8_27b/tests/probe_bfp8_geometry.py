# SPDX-License-Identifier: Apache-2.0
"""Two legal BFP8 geometries on real target weights/recorded HF activations."""
import argparse
import json
import time
from pathlib import Path
import torch
import ttnn
from safetensors.torch import load_file
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import OptimizedDecoder

p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(8)
root=Path('.cache/qwen38-weights');cfg=AutoConfig.from_pretrained(root).text_config;cfg._attn_implementation='eager'
w=load_file(str(root/f'layer_{a.layer}.safetensors'));prefix=f'model.language_model.layers.{a.layer}.'
ref=Qwen3_5DecoderLayer(cfg,a.layer).to(torch.bfloat16).eval()
ref.load_state_dict({k.removeprefix(prefix):v for k,v in w.items()})
x=torch.load(root/f'recorded_layer_{a.layer}.pt',weights_only=True)[:,:130]
rope=Qwen3_5TextRotaryEmbedding(cfg);cs=rope(x,torch.arange(130).expand(3,1,-1));cache=DynamicCache(config=cfg)
inputs={}
mods={'mix':ref.linear_attn.in_proj_qkv if a.layer==0 else ref.self_attn.q_proj,'down':ref.mlp.down_proj}
handles=[m.register_forward_pre_hook(lambda module,args,r=r:inputs.__setitem__(r,args[0].detach().clone())) for r,m in mods.items()]
with torch.no_grad():
    mask=torch.where(torch.arange(129)[None,:]>torch.arange(129)[:,None],float('-inf'),0.).to(torch.bfloat16)[None,None]
    ref(x[:,:129],position_embeddings=(cs[0][:,:129],cs[1][:,:129]),past_key_values=cache,attention_mask=mask if a.layer==3 else None)
    ref(x[:,129:],position_embeddings=(cs[0][:,129:],cs[1][:,129:]),past_key_values=cache)
for h in handles:h.remove()
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
rows=[]
try:
    models={};owner=None
    for name,block,readers in [('block10',10,1),('readers2',20,2)]:
        policy=dict(attention_dtype='bfloat8_b',mlp_dtype='bfloat8_b',down_dtype='bfloat8_b',
                    mix=dict(cores=8,block=block,readers=readers,prefill_block=8 if a.layer==0 else 4,weight_pad_readers=2))
        model=MultichipDecoder.from_state_dict(w,hf_config=cfg,layer_idx=a.layer,mesh_device=mesh,policy=policy,ccl_workspace=owner)
        models[name]=model;owner=model.ar_workspace
    tx={role:ttnn.from_torch(value,device=mesh,layout=ttnn.TILE_LAYOUT,dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=2) if role=='down' else ttnn.ReplicateTensorToMesh(mesh)) for role,value in inputs.items()}
    def read(t):return torch.cat([ttnn.to_torch(v) for v in ttnn.get_device_tensors(t)],dim=-1)
    for role in ('mix','down'):
        control=None
        for name in ('block10','readers2','readers2','block10'):
            model=models[name]
            def operation():return OptimizedDecoder._linear(model,tx[role],getattr(model,role))
            warm=operation();expected=read(warm);del warm
            if control is None:control=expected
            pcc=float(torch.corrcoef(torch.stack([expected.double().flatten(),control.double().flatten()]))[0,1])
            assert pcc>=.995,(role,name,pcc)
            tid=ttnn.begin_trace_capture(mesh,cq_id=0);out=operation();ttnn.end_trace_capture(mesh,tid,cq_id=0)
            try:
                ttnn.execute_trace(mesh,tid,cq_id=0,blocking=True);assert torch.equal(expected,read(out))
                start=time.perf_counter()
                for _ in range(50):ttnn.execute_trace(mesh,tid,cq_id=0,blocking=True)
                us=(time.perf_counter()-start)*1e6/50
                rows.append(dict(role=role,variant=name,traced_host_us=us,pcc_same_dtype=pcc,exact_replay=True,
                    policy=model.policy[role],weight_dtype=str(getattr(model,role).dtype),fidelity='LoFi'))
            finally:ttnn.release_trace(mesh,tid)
            del out
    Path(a.output).write_text(json.dumps(dict(layer=a.layer,scope='Reduced real-weight projection geometry check; not full-model performance',rows=rows),indent=2)+'\n')
finally:ttnn.close_mesh_device(mesh)
