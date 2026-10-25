# SPDX-License-Identifier: Apache-2.0
"""Prove prefill user/sequence packing under uniform rotary FP16 accumulation.

The default retains the historical uniform-FP16 packing control. Pass
--requested-config to regress the production HiFi4/FP32 policy across core groups.
"""
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
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--requested-config',action='store_true');a=p.parse_args()
assert not Path(a.output).exists()
torch.set_num_threads(8);torch.manual_seed(719)
root=Path(__file__).resolve().parents[1]
cfg=AutoConfig.from_pretrained(str(root/'doc/functional_decoder/hf_config.json')).text_config
weights=load_file('.cache/qwen38-weights/layer_3.safetensors')
recorded=torch.load('.cache/qwen38-weights/recorded_layer_3.pt',weights_only=True)
rope=Qwen3_5TextRotaryEmbedding(cfg)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    model=MultichipDecoder.from_state_dict(weights,hf_config=cfg,layer_idx=3,mesh_device=mesh)
    impl=model._prefill_impl
    def put(x,hidden=False):return ttnn.from_torch(x.contiguous(),device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=2) if hidden else ttnn.ReplicateTensorToMesh(mesh))
    def read(x):return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(x)]
    rows=[]
    for batch,length in [(b,s) for b in (8,16,32) for s in (32,160)]+[(32,4096)]:
        order=torch.randperm(batch)
        idx=(17*torch.arange(batch)[:,None]+torch.arange(length)[None,:])%recorded.shape[1]
        x=recorded[0,idx[order]]
        positions=((7919*torch.arange(batch)[:,None]+torch.arange(length)[None,:]+129)%cfg.max_position_embeddings)[order]
        with torch.no_grad():cos,sin=rope(x,positions[None].expand(3,-1,-1))
        ct,st=put(cos[:,None]),put(sin[:,None])
        tables=tuple(ttnn.reshape(t,[1,1,batch*length,64]) for t in (ct,st))
        projected=impl._linear(impl._norm(put(x,True),impl.in_norm),impl.mix)
        q,k,_=ttnn.transformer.split_query_key_value_and_split_heads(projected[...,:2048],num_heads=6,num_kv_heads=1,transpose_key=False)
        for kind,value,weight in [('q',q,impl.q_norm),('k',k,impl.k_norm)]:
            normalized=impl._norm(value,weight)
            original=impl.ckc
            if not a.requested_config:
                impl.ckc=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,math_approx_mode=False,fp32_dest_acc_en=False,packer_l1_acc=True)
            try:
                expected=impl._rope(normalized,ct,st)
                actual=impl._packed_prefill_rope(normalized,tables,batch,length)
            finally:impl.ckc=original
            ar,er,nr=read(actual),read(expected),read(normalized)
            assert all(torch.equal(v,e) for v,e in zip(ar,er)),(batch,length,kind)
            assert all(torch.equal(v[...,64:],n[...,64:]) for v,n in zip(ar,nr)),(batch,length,kind,'tail')
            rows.append(dict(batch=batch,working_sequence=length,group=kind,distinct_permuted_user_positions=True,all_four_devices_exact=True,tail192_exact=True))
    evidence=dict(mesh=[1,4],real_weights=True,recorded_activations=True,exact=True,checks=rows,
        scope=('Packing identity under production requested HiFi4/FP32 rotary accumulation.' if a.requested_config else 'Packing identity under explicit uniform FP16 rotary accumulation. Production precision unchanged; original-policy whole-layer PCC is separate evidence.'),
        requested_config=a.requested_config,
        provenance={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),root/'tt/multichip_decoder.py',root/'tt/optimized_decoder.py')})
    Path(a.output).write_text(json.dumps(evidence,indent=2)+'\n')
finally:ttnn.close_mesh_device(mesh)
