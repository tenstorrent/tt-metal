"""Isolated custom recurrence attribution; no full-model or perf claim."""
import argparse,importlib.util,json
from pathlib import Path
import torch
import ttnn

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--rm',action='store_true');ap.add_argument('--state-bf16',action='store_true');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 spec=importlib.util.spec_from_file_location('candidate',args.candidate);mod=importlib.util.module_from_spec(spec);import sys;sys.modules['candidate']=mod;spec.loader.exec_module(mod)
 torch.set_num_threads(8);torch.manual_seed(415)
 mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
 upload=lambda x:ttnn.from_torch(x,device=mesh,dtype=ttnn.float32 if x.dtype==torch.float32 else ttnn.bfloat16,layout=ttnn.TILE_LAYOUT)
 read=lambda x:ttnn.to_torch(x,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).float()
 rows=[]
 try:
  b=3
  for mode in ['normal','zero_kv','zero_state']:
   q=torch.randn(b,16,128).bfloat16().float();k=torch.randn(b,16,128).bfloat16().float();v=torch.randn(b,48,128).bfloat16().float()
   g=(-torch.rand(b,1,48)*.1).float();beta=torch.rand(b,1,48).bfloat16();s=torch.randn(b,48,128,128)*.01
   if mode=='zero_kv':k.zero_();v.zero_()
   if mode=='zero_state':s.zero_()
   qraw,kraw=q.clone(),k.clone()
   q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)*(128**-.5);k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
   qr=q.repeat_interleave(3,1);kr=k.repeat_interleave(3,1);s=s.bfloat16().float() if args.state_bf16 else s
   sd=s*g.reshape(b,48,1,1).exp();delta=(v-(sd*kr.unsqueeze(-1)).sum(-2))*beta.float().reshape(b,48,1);expected=sd+kr.unsqueeze(-1)*delta.unsqueeze(-2);out=(expected*qr.unsqueeze(-1)).sum(-2)
   for compact in ([True] if args.state_bf16 else [False,True]):
    state=upload(s.reshape(b*48,128,128).bfloat16() if args.state_bf16 else s.reshape(b*48,128,128))
    if compact and args.rm:
     rm=lambda x:ttnn.from_torch(x.bfloat16(),device=mesh,dtype=ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT)
     y=mod._recurrent_sfpu_compact(rm(qraw.reshape(b,1,2048)),rm(kraw.reshape(b,1,2048)),rm(v.reshape(b,1,6144)),upload(g),upload(beta),state,normalized=False)
    elif compact:y=mod._recurrent_sfpu_compact(upload(q),upload(k),upload(v.reshape(b,1,6144).bfloat16()),upload(g),upload(beta),state,normalized=True)
    else:y=mod._recurrent_sfpu(upload(qr.reshape(b*48,128,1)),upload(kr.reshape(b*48,128,1)),upload(v.reshape(b*48,1,128)),upload(g.exp().reshape(b*48,1,1)),upload(beta.float().reshape(b*48,1,1)),state)
    actual=read(y).reshape(b,48,128);actual_s=read(state).reshape_as(expected)
    metrics=lambda a,e:{'max_abs':float((a-e).abs().max()),'relative_rms':float((a-e).square().mean().sqrt()/e.square().mean().sqrt().clamp_min(1e-20))}
    rows.append({'mode':mode,'compact':compact,'output':metrics(actual,out),'state':metrics(actual_s,expected.bfloat16().float() if args.state_bf16 else expected),'head_state_rel':[metrics(actual_s[:,h],expected[:,h].bfloat16().float() if args.state_bf16 else expected[:,h])['relative_rms'] for h in range(48)]})
  args.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='head_state_rel'} for r in rows]))
 finally:ttnn.close_mesh_device(mesh)
if __name__=='__main__':main()
