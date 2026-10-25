# SPDX-License-Identifier: Apache-2.0
"""Reduced diagnostic of slot invariance; controls never change production policy."""
import argparse,json,types
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
p=argparse.ArgumentParser();p.add_argument('--control',choices=['production','uniform_fp16_rope'],default='production');p.add_argument('--identical-length',type=int,default=33);p.add_argument('--context',type=int,default=512);p.add_argument('--output');a=p.parse_args()
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
result=dict(scope='Reduced real layers0/3, B8; diagnostic, not stage acceptance',control=a.control,context=a.context,identical_prompt_length=a.identical_length)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,checkpoint='.cache/qwen38-probe',
        layer_indices=[0,3],batch_size=8,context_length=a.context,prepare=False)
    if a.control=='uniform_fp16_rope':
        impl=gen.model.layers[1]._prefill_impl
        original=impl._packed_prefill_rope
        def uniform(self,*args,_original=original,**kwargs):
            saved=self.ckc
            self.ckc=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,fp32_dest_acc_en=False,packer_l1_acc=True)
            try:return _original(*args,**kwargs)
            finally:self.ckc=saved
        impl._packed_prefill_rope=types.MethodType(uniform,impl)
    gen.prepare_decode(prefill_lengths=(33,35,65,a.identical_length))
    stages=[]
    def inspect(name,value):
        diffs=[]
        for local in ttnn.get_device_tensors(value):
            host=ttnn.to_torch(local).float()
            diffs.append([float((host[row]-host[0]).abs().max()) for row in range(8)])
        stages.append(dict(name=name,shape=list(value.shape),max_abs_slot_difference=diffs))
        return value
    original_embed=gen.model.embed
    gen.model.embed=lambda *args,**kwargs:inspect('embedding',original_embed(*args,**kwargs))
    saved_layers=[]
    for index,layer in zip(gen.model.layer_indices,gen.model.layers):
        original=layer.prefill_forward;saved_layers.append(original)
        def wrapped(*args,_original=original,_index=index,**kwargs):
            return inspect('layer'+str(_index),_original(*args,**kwargs))
        layer.prefill_forward=wrapped
    gen.reset()
    pref=gen.prefill_forward(torch.full((8,a.identical_length),101,dtype=torch.int64),page_table=gen.page_table,
        kv_cache=gen.kv_cache,prompt_lens=[a.identical_length]*8,return_device=True)
    inspect('terminal',pref)
    result['identical_prompt_stages']=stages
    gen.model.embed=original_embed
    for layer,original in zip(gen.model.layers,saved_layers):layer.prefill_forward=original
    cases=[(101,33),(102,65),(101,33),(103,35),(104,33),(102,65),(103,35),(101,33)]
    def request(order):
        gen.reset();lengths=[cases[i][1] for i in order]
        tokens=torch.zeros(8,max(lengths),dtype=torch.int64)
        for row,i in enumerate(order):tokens[row,:lengths[row]]=cases[i][0]
        pre=gen.prefill_forward(tokens,page_table=gen.page_table,kv_cache=gen.kv_cache,prompt_lens=lengths).reshape(8,-1)
        dec=gen.decode_forward(torch.full((8,1),105,dtype=torch.int64),torch.tensor(lengths,dtype=torch.int32),
            page_table=gen.page_table,kv_cache=gen.kv_cache,return_logits=True).reshape(8,-1)
        return pre,dec
    order=list(range(8));permutation=[7,3,5,0,6,2,4,1];inverse=[permutation.index(i) for i in order]
    first=request(order);repeat=request(order);permuted=request(permutation)
    result['phases']={}
    for phase,x,y,z in zip(('prefill','decode'),first,repeat,permuted):
        result['phases'][phase]=dict(repeat_exact=torch.equal(x,y),permutation_exact=torch.equal(x,z[inverse]),
            permutation_max_abs=float((x-z[inverse]).abs().max()),
            duplicate_pairs=[dict(slots=[i,j],exact=torch.equal(x[i],x[j]),max_abs=float((x[i]-x[j]).abs().max())) for i,j in ((0,2),(0,7),(1,5),(3,6))])
    print(json.dumps(result),flush=True)
finally:
    Path(a.output or f'models/autoports/qwen_qwen3_8_27b/doc/full_model/slot_logits_{a.control}.json').write_text(json.dumps(result,indent=2)+'\n')
    ttnn.close_mesh_device(mesh)
