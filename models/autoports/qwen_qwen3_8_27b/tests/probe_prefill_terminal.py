# SPDX-License-Identifier: Apache-2.0
"""Real reduced prefill: compact terminal users, preserving public output shape."""
import argparse,json,time
from pathlib import Path
import torch,ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=32);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(8);ttnn.CONFIG.throw_exception_on_fallback=True
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,checkpoint='.cache/qwen38-probe',layer_indices=[0,3],batch_size=a.batch,context_length=4352,prepare=False)
    rows=[]
    for length,mixed in [(33,False),(65,True),(4097,False)]:
        prompt=torch.arange(a.batch)[:,None].expand(a.batch,length)+100
        lengths=[length if not mixed or i%2==0 else 0 for i in range(a.batch)]
        def run(packed):
            gen.pack_prefill_terminal=packed
            gen.reset();ttnn.synchronize_device(mesh)
            begin=time.monotonic()
            logits=gen.prefill_forward(prompt,page_table=gen.page_table,kv_cache=gen.kv_cache,prompt_lens=lengths,return_device=True)
            ttnn.synchronize_device(mesh);elapsed=(time.monotonic()-begin)*1000
            host=gen.host_logits(logits)
            return elapsed,host
        for packed in (False,True):run(packed)
        times={False:[],True:[]}
        for _ in range(3):
            baseline,old=run(False);candidate,new=run(True)
            assert torch.equal(old,new),(length,mixed,float((old-new).abs().max()))
            times[False].append(baseline);times[True].append(candidate)
        rows.append(dict(prompt_len=length,mixed_inactive_rows=mixed,exact=True,host_prefill_ms={str(k):v for k,v in times.items()}))
    Path(a.output).write_text(json.dumps(dict(batch=a.batch,scope='Reduced real layers0/3; end-to-end prefill host times, not full model',rows=rows),indent=2)+'\n')
    print('PREFILL_TERMINAL_EXACT_PASS',flush=True)
finally:ttnn.close_mesh_device(mesh)
