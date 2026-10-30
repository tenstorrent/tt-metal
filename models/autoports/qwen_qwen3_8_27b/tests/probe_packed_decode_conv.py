# SPDX-License-Identifier: Apache-2.0
"""Exact per-user Q/K/V and history check against the former TTNN convolution loop."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import ttnn
from safetensors.torch import load_file
from transformers import AutoConfig
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
assert not Path(a.output).exists()
torch.set_num_threads(8)
root=Path(__file__).resolve().parents[1]
cfg=AutoConfig.from_pretrained(str(root/'doc/functional_decoder/hf_config.json')).text_config
weights=load_file('.cache/qwen38-weights/layer_0.safetensors')
recorded=torch.load('.cache/qwen38-weights/recorded_layer_0.pt',weights_only=True)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    model=MultichipDecoder.from_state_dict(weights,hf_config=cfg,layer_idx=0,mesh_device=mesh)
    def put(x):return ttnn.from_torch(x.contiguous(),device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=2))
    def read(x):return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(x)]
    rows=[]
    for batch in (8,9,16,17,31,32):
        impl=model.decode_implementation(batch)
        idx=(torch.arange(132)[None,:]+17*torch.arange(batch)[:,None])%recorded.shape[1]
        x=recorded[0,idx]
        state=model.allocate_state(batch_size=batch,context_length=256)
        model.prefill_forward(put(x[:,:129]),state=state)
        for step in range(3):
            h=impl._norm(put(x[:,129+step:130+step]),impl.in_norm)
            projected=impl._linear(h,impl.mix)
            padded=ttnn.pad(projected,[(0,0),(0,31),(0,0)],value=0)
            expected=[[],[],[]]
            for user in range(batch):
                hist=state.conv[user:user+1]
                outputs=ttnn.experimental.kda.qkv_causal_conv1d_silu(padded[user:user+1],hist,*impl.taps,512,512,1536,
                    actual_start=impl.conv_start,predecessor_carry=hist,
                    program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=False))
                for group,value in zip(expected,outputs):group.append(value[:,:1])
            expected=[ttnn.concat(group,dim=0) for group in expected]
            q,k,v,next_history=impl._packed_decode_convolution(projected,state)
            for actual,control in zip((q,k,v),expected):
                assert all(torch.equal(x,y) for x,y in zip(read(actual),read(control))), (batch,step)
            expected_history=ttnn.concat([state.conv[:,1:],ttnn.to_layout(projected[...,:2560],ttnn.ROW_MAJOR_LAYOUT)],dim=1)
            assert all(torch.equal(x,y) for x,y in zip(read(next_history),read(expected_history)))
            ttnn.copy(next_history,state.conv)
            rows.append(dict(batch=batch,history_update=step+1,all_four_devices_exact=True))
    result=dict(mesh=[1,4],real_weights=True,recorded_activations=True,prefix_tokens=129,checks=rows,
        output_groups=['q','k','v','raw_history'],exact=True,
        decoder_sha256=hashlib.sha256((root/'tt/multichip_decoder.py').read_bytes()).hexdigest(),
        scope='Convolution intermediate semantics only; three advancing history updates. No performance claim.')
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
finally:ttnn.close_mesh_device(mesh)
