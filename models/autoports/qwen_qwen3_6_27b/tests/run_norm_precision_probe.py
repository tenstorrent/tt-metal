"""Isolated same-input hidden RMSNorm accuracy; no model/perf claim."""
import argparse,importlib.util,json,sys
from pathlib import Path
import torch
import ttnn

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 spec=importlib.util.spec_from_file_location('norm_candidate',args.candidate);mod=importlib.util.module_from_spec(spec);sys.modules['norm_candidate']=mod;spec.loader.exec_module(mod)
 torch.set_num_threads(8);torch.manual_seed(194)
 mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
 rows=[]
 try:
  for b in [1,3,32]:
   for scale in [0.,.001,1.]:
    x=(torch.randn(1,b,5120)*scale).bfloat16();w=1+torch.randn(1,1,5120).bfloat16().float()*.1
    up=lambda t:ttnn.from_torch(t,device=mesh,dtype=ttnn.bfloat16 if t.dtype==torch.bfloat16 else ttnn.float32,layout=ttnn.TILE_LAYOUT)
    y=mod.hidden_rmsnorm_sfpu(up(x),up(w))
    actual=ttnn.to_torch(y,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).float()
    expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w).bfloat16().float()
    rel=float((actual-expected).norm()/expected.norm().clamp_min(1e-20));err=float((actual-expected).abs().max())
    row={'batch':b,'scale':scale,'max_abs':err,'relative_rms':rel,'equal_fraction':float((actual==expected).float().mean())};rows.append(row);print(row,flush=True)
  args.output.write_text(json.dumps(rows,indent=2)+'\n')
  assert all(r['relative_rms']<1e-4 for r in rows), rows
 finally:ttnn.close_mesh_device(mesh)
if __name__=='__main__':main()
