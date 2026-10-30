# SPDX-License-Identifier: Apache-2.0
"""Reduced wrapper probe: real layers 0 and 3 and complete terminal path."""
import argparse
import time
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='.cache/qwen38-full')
p.add_argument('--seq',type=int,default=33)
p.add_argument('--trace',action='store_true')
a=p.parse_args()
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    start=time.monotonic()
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,
        prepare=False,checkpoint=a.checkpoint,layer_indices=[0,3],context_length=512)
    print('BUILD_SECONDS',time.monotonic()-start,flush=True)
    if a.trace:
        gen.prepare_decode(prefill_lengths=(a.seq,))
        print('PREPARED',flush=True)
        print(gen.generate([100]*a.seq,4),flush=True)
    else:
        logits=gen.prefill_logits([100]*a.seq)
        print('PREFILL',list(logits.shape),torch.isfinite(logits).all().item(),flush=True)
    print('PROBE_SECONDS',time.monotonic()-start,flush=True)
finally:
    ttnn.close_mesh_device(mesh)
