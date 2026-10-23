# SPDX-License-Identifier: Apache-2.0
"""Finite-value discrimination for the AutoDebug peer-grid diagnosis."""
import importlib.util,json
from pathlib import Path
import torch,ttnn
from safetensors.torch import load_file
from transformers import AutoConfig
from ..tt.optimized_decoder import OptimizedDecoder
torch.set_num_threads(8)
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
spec=importlib.util.spec_from_file_location('models.autoports.qwen_qwen3_8_27b.tt.norm_snapshot',d/'candidate5.py')
import sys
snapshot=importlib.util.module_from_spec(spec);sys.modules[spec.name]=snapshot;spec.loader.exec_module(snapshot)
cfg=AutoConfig.from_pretrained('.cache/qwen38-weights').text_config
w=load_file('.cache/qwen38-weights/layer_0.safetensors');x=torch.load('.cache/qwen38-weights/recorded_layer_0.pt',weights_only=True)[:,:512]
gamma=(1+w['model.language_model.layers.0.input_layernorm.weight'].float()).to(torch.bfloat16)
y=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+cfg.rms_norm_eps))*gamma.float()
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
try:
 tx=ttnn.from_torch(x,device=mesh,layout=ttnn.TILE_LAYOUT,dtype=ttnn.bfloat16);results=[]
 for kind,cls in [('original_nonrectangular',snapshot.OptimizedDecoder),('fixed_rectangular',OptimizedDecoder)]:
  for cores in (40,80):
   p={**json.loads((d/'role_control_l0.policy.json').read_text()),'prefill_sharded_norm':True,'prefill_norm_cores':cores}
   dec=cls.from_state_dict(w,hf_config=cfg,layer_idx=0,mesh_device=mesh,policy=p)
   got=ttnn.to_torch(dec._norm(tx,dec.in_norm)).float();finite=bool(torch.isfinite(got).all());std=got.double().std().item()
   pcc=torch.corrcoef(torch.stack([got.double().flatten(),y.double().flatten()]))[0,1].item() if finite and std>0 else None
   results.append(dict(kind=kind,cores=cores,finite=finite,output_std=std,max_abs=got.abs().max().item(),nonfinite_count=int((~torch.isfinite(got)).sum()),pcc=pcc))
   if kind=='fixed_rectangular':assert finite and pcc>.995
   del dec
 (d/'prefill_norm_probe.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
finally:ttnn.close_mesh_device(mesh)
