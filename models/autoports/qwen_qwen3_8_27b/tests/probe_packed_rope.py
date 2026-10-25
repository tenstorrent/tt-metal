# SPDX-License-Identifier: Apache-2.0
"""Exact packed Q/K RoPE including distinct/permuted users and unchanged tails."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import ttnn
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from ..tt.multichip_decoder import MultichipDecoder

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
assert not Path(a.output).exists()
torch.set_num_threads(8);torch.manual_seed(718)
root=Path(__file__).resolve().parents[1]
cfg=AutoConfig.from_pretrained(str(root/'doc/functional_decoder/hf_config.json')).text_config
weights=load_file('.cache/qwen38-weights/layer_3.safetensors')
recorded=torch.load('.cache/qwen38-weights/recorded_layer_3.pt',weights_only=True)
rope=Qwen3_5TextRotaryEmbedding(cfg)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    model=MultichipDecoder.from_state_dict(weights,hf_config=cfg,layer_idx=3,mesh_device=mesh)
    def put(x,hidden=False):return ttnn.from_torch(x.contiguous(),device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=2) if hidden else ttnn.ReplicateTensorToMesh(mesh))
    def read(x):return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(x)]
    rows=[]
    for batch in (4,5,7,8,9,16,17,31,32):
        impl=model.decode_implementation(batch)
        for step,start in enumerate((129,4096,262110)):
            order=torch.randperm(batch)
            idx=(17*torch.arange(batch)+step)%recorded.shape[1]
            x=recorded[0,idx[order]][:,None]
            positions=((7919*torch.arange(batch)+start)%cfg.max_position_embeddings)[order]
            with torch.no_grad():cos,sin=rope(x,positions.reshape(1,batch,1).expand(3,-1,-1))
            ct,st=put(cos[:,None]),put(sin[:,None]);tables=impl._packed_decode_rope_tables(ct,st,batch)
            projected=impl._linear(impl._norm(put(x,impl.policy['residual_layout']=='sharded'),impl.in_norm),impl.mix)
            q,k,_=ttnn.transformer.split_query_key_value_and_split_heads(projected[...,:2048],num_heads=6,num_kv_heads=1,transpose_key=False)
            for kind,value,weight in [('q',q,impl.q_norm),('k',k,impl.k_norm)]:
                normalized=impl._norm(value,weight)
                expected=impl._rope(normalized,ct,st);actual=impl._packed_decode_rope(normalized,tables)
                ar,er,nr=read(actual),read(expected),read(normalized)
                assert all(torch.equal(a,e) for a,e in zip(ar,er)),(batch,step,kind)
                assert all(torch.equal(a[...,64:],n[...,64:]) for a,n in zip(ar,nr)),(batch,step,kind,'tail')
                rows.append(dict(batch=batch,position_round=step,group=kind,users_permuted=True,distinct_positions=True,all_four_devices_exact=True,tail192_exact=True))
    evidence=dict(mesh=[1,4],real_weights=True,recorded_activations=True,exact=True,checks=rows,
        scope='Q/K intermediate equality versus original per-user TTNN loop; three distinct/permuted position assignments. No timing claim.',
        provenance={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),root/'tt/multichip_decoder.py',root/'tt/optimized_decoder.py')})
    Path(a.output).write_text(json.dumps(evidence,indent=2)+'\n')
finally:ttnn.close_mesh_device(mesh)
