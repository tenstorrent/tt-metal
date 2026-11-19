"""Device pack-vs-explicit-RNE attribution; standalone arithmetic only."""
import argparse,json
from pathlib import Path
import torch
import ttnn

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(8);torch.manual_seed(11)
 mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
 try:
  x=torch.randn(1,32,5120);y=torch.randn_like(x)*.01
  up=lambda v:ttnn.from_torch(v,device=mesh,dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT)
  tx,ty=up(x),up(y);rows=[]
  for name,operation in [('add',ttnn.add),('multiply',ttnn.multiply)]:
   expected=(x+y if name=='add' else x*y).bfloat16().float()
   for explicit in [False,True]:
    actual=operation(tx,ty,dtype=ttnn.float32 if explicit else ttnn.bfloat16)
    if explicit:actual=ttnn.typecast(actual,ttnn.bfloat16)
    actual=ttnn.to_torch(actual,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).float()
    rows.append({'operation':name,'explicit_rne':explicit,'equal_fraction':float((actual==expected).float().mean()),'gain':float((actual*expected).sum()/expected.square().sum()),'relative_rms':float((actual-expected).norm()/expected.norm())})
  a.output.write_text(json.dumps(rows,indent=2)+'\n');print(rows)
 finally:ttnn.close_mesh_device(mesh)
if __name__=='__main__':main()
