# SPDX-License-Identifier: Apache-2.0
"""Compare prepared padding to inherited logical-length wrapper on real weights."""
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

def pcc(a,b):
    a=a.double().reshape(-1);b=b.double().reshape(-1)
    if torch.equal(a,b): return 1.0
    return float(torch.corrcoef(torch.stack([a,b]))[0,1])
def hosts(t): return [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]

torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,prepare=False,checkpoint='.cache/qwen38-probe',layer_indices=[0,3],context_length=4352)
    results=[]
    for length in [1,2,31,32,33,127,128,129,255,256,257,4095,4096,4097]:
        tokens=(torch.arange(length)%97+100).reshape(1,-1)
        gen.reset()
        for start in range(0,length,4096):
            n=min(4096,length-start)
            ids=gen.put(tokens[:,start:start+n].to(torch.int32),ttnn.uint32)
            positions=gen.put(torch.arange(start,start+n,dtype=torch.int32).reshape(1,-1),ttnn.uint32)
            logits=gen.model.prefill_forward(ids,positions=positions,kv_cache=gen.kv_cache,
                page_table=gen.page_table,batch=1,length=n,start_position=start,logits_to_keep=1)
        old=gen.host_logits(logits)
        old_conv=hosts(gen.kv_cache[0].conv);old_rec=hosts(gen.kv_cache[0].recurrent)
        gen.reset()
        new=gen.prefill_forward(tokens,page_table=gen.page_table,kv_cache=gen.kv_cache,prompt_lens=[length])
        conv=hosts(gen.kv_cache[0].conv);rec=hosts(gen.kv_cache[0].recurrent)
        row=dict(length=length,logits_pcc=pcc(old,new),conv_min_pcc=min(pcc(a,b) for a,b in zip(old_conv,conv)),
                 recurrent_min_pcc=min(pcc(a,b) for a,b in zip(old_rec,rec)),
                 conv_exact=all(torch.equal(a,b) for a,b in zip(old_conv,conv)))
        print(json.dumps(row),flush=True)
        assert min(row['logits_pcc'],row['conv_min_pcc'],row['recurrent_min_pcc'])>=.995,row
        results.append(row)
    Path('models/autoports/qwen_qwen3_8_27b/doc/full_model/prefill_bucket_equivalence.json').write_text(json.dumps(results,indent=2)+'\n')
finally:
    ttnn.close_mesh_device(mesh)
