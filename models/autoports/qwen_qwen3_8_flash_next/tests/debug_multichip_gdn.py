"""Eager first-token intermediate probe; host readback is diagnostic only."""
import argparse, json
from pathlib import Path
import torch, ttnn
from transformers import AutoConfig
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_components import CHECKPOINT
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder, _hidden_parts
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--impl',choices=['single','mesh'],required=True)
    p.add_argument('--fidelity',choices=['LoFi','HiFi2'],default='LoFi');p.add_argument('--tag',default='gdn_intermediates')
    a=p.parse_args();torch.set_num_threads(8);cfg=AutoConfig.from_pretrained(CHECKPOINT,local_files_only=True).text_config
    rng=torch.Generator().manual_seed(2026);prefix=None
    for n in [1,3,31,32,33,127,128,129,259,511,512,513,2053]:
        x=(torch.randn(1,1,n,2560,generator=rng).repeat(1,1,1,4)*.05+torch.randn(1,1,n,10240,generator=rng)*.005).bfloat16()
        torch.randint(cfg.vocab_size,(1,n),generator=rng,dtype=torch.int32)
        if n==1:prefix=x
    dx=(torch.randn(1,1,1,2560,generator=rng).repeat(1,1,1,4)*.05+torch.randn(1,1,1,10240,generator=rng)*.005).bfloat16()
    cache=Path('/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip')
    if a.impl=='mesh':ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4 if a.impl=='mesh' else 1),trace_region_size=0)
    try:
        model=(MultichipDecoder if a.impl=='mesh' else OptimizedDecoder).from_state_dict(CheckpointState(CHECKPOINT,'model.language_model.layers.0.'),hf_config=cfg,layer_idx=0,mesh_device=mesh,batch_size=1,max_requests=2)
        if a.impl=='mesh':model.attention.projection_compute=ttnn.init_device_compute_kernel_config(mesh.arch(),math_fidelity=getattr(ttnn.MathFidelity,a.fidelity),math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
        captured={}
        def wrap_read(hc,name):
            original=hc.read
            def read(x):
                result=original(x);captured[name]=ttnn.clone(result[0]);captured[name+'_inject']=ttnn.clone(result[1]);return result
            hc.read=read
        wrap_read(model.attn_hyper,'attn_input');wrap_read(model.mlp_hyper,'mlp_input')
        original_linear=model.attention._linear
        def linear(x,w):
            out=original_linear(x,w)
            key='projection' if w is model.attention.packed_weight else 'attn_partial'
            captured[key]=out
            if key=='attn_partial':captured['attn_pre_wo']=x
            return out
        model.attention._linear=linear
        orig_topk=ttnn.topk
        def topk(*args,**kw):
            values,ids=orig_topk(*args,**kw);captured['router_input']=args[0];captured['route_ids']=ids;captured['route_values']=values
            return values,ids
        ttnn.topk=topk
        def upload(x,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,residual=False):
            if a.impl=='mesh':return model.ops.shards(_hidden_parts(x),dtype,layout) if residual else model.ops.upload(x,dtype,layout)
            return ttnn.from_torch(x,dtype=dtype,layout=layout,device=mesh)
        slots=torch.zeros(1,32,dtype=torch.int32);slots[0,0]=1;sd=upload(slots,ttnn.uint32,ttnn.ROW_MAJOR_LAYOUT)
        outputs={}
        for step in range(4):
            pos=torch.zeros(1,32,dtype=torch.int32);pos[0,0]=step
            pd=upload(pos,ttnn.uint32,ttnn.ROW_MAJOR_LAYOUT);xd=upload(prefix if step==0 else dx,residual=True)
            out=(model.prefill_forward if step==0 else model.decode_forward)(xd,current_positions=pd,slots=sd)
            captured['output']=out
            rows={key:[ttnn.to_torch(t) for t in ttnn.get_device_tensors(value)] for key,value in captured.items()}
            outputs[str(step)]=rows;captured.clear()
        output=cache/f'{a.tag}_{a.impl}_{a.fidelity}.pt';torch.save(outputs,output)
        if a.impl=='mesh':
            ref=torch.load(cache/f'{a.tag}_single_LoFi.pt',weights_only=True);metrics=[]
            for step,rows in outputs.items():
                for key,parts in rows.items():
                    if key not in ref[step]:continue
                    if key=='output':value=torch.cat([v.reshape(1,1,1,4,640) for v in parts],-1).reshape(1,1,1,10240)
                    elif key=='attn_partial':value=sum(parts)
                    elif key=='attn_pre_wo':value=torch.cat(parts,-1)
                    elif key=='projection':continue
                    else:value=parts[0]
                    if key=='route_ids':
                        same=torch.equal(value,ref[step][key][0]);metrics.append(dict(step=step,key=key,exact_equal=same,mesh=value.flatten().tolist()[:10],single=ref[step][key][0].flatten().tolist()[:10]));continue
                    finite=torch.isfinite(ref[step][key][0]) & torch.isfinite(value)
                    metrics.append(dict(step=step,key=key,nonfinite=int((~finite).sum()),**pcc_metrics(ref[step][key][0][finite],value[finite])))
            path=Path('models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder')/f'{a.tag}_{a.fidelity}.json';path.write_text(json.dumps(metrics,indent=2))
            print(json.dumps(metrics),flush=True)
        print('DIAGNOSTIC_COMPLETE',output,flush=True)
    finally:ttnn.close_mesh_device(mesh)
if __name__=='__main__':main()
