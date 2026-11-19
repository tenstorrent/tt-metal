# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""First-boundary probe for compact decode head creation, with real weights."""
import json
import sys
import torch
from transformers import AutoConfig
import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL,REVISION,load_layer,pcc
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class,decoder_kwargs,provenance

def main():
 torch.set_num_threads(8);torch.manual_seed(17)
 c=AutoConfig.from_pretrained(MODEL,revision=REVISION,local_files_only=True).text_config
 mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
 try:
  layer=decoder_class().from_state_dict(load_layer(3),hf_config=c,layer_idx=3,mesh_device=mesh,**decoder_kwargs())
  read=lambda t:ttnn.to_torch(t,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0))
  x=layer._norm(layer._upload(torch.randn(32,1,5120,dtype=torch.bfloat16)*.1),'input_layernorm.weight')
  proj=layer._linear(ttnn.reshape(x,(1,32,5120)),'packed_qkv')
  host=read(proj).reshape(32,8192)
  result={'provenance':provenance(),'rows':[]}
  for layout in ['dram','l1','width']:
   src=ttnn.reshape(proj,(1,1,32,8192))
   if layout=='l1':src=ttnn.to_memory_config(src,ttnn.L1_MEMORY_CONFIG)
   elif layout=='width':src=ttnn.to_memory_config(src,ttnn.create_sharded_memory_config((32,1024),ttnn.CoreGrid(x=8,y=1),ttnn.ShardStrategy.WIDTH,use_height_and_width_as_shard_shape=True))
   q,k,v=ttnn.experimental.nlp_create_qkv_heads_decode(src,num_heads=24,num_kv_heads=4,memory_config=layer._compact_height_mem(layer._compact_cores(32),32,256))
   for name,t,start,end,heads in [('q',q,0,6144,24),('k',k,6144,7168,4),('v',v,7168,8192,4)]:
    expected=host[:,start:end].reshape(1,32,heads,256)
    actual=read(t)
    row={'layout':layout,'name':name,'shape':list(t.shape),'pcc':pcc(expected,actual),'bit_equal':torch.equal(expected,actual),'users':[pcc(expected[:,b],actual[:,b]) for b in range(32)]}
    result['rows'].append(row)
  open(sys.argv[1],'w').write(json.dumps(result,indent=2)+'\n')
 finally:ttnn.close_mesh_device(mesh)
if __name__=='__main__':main()
