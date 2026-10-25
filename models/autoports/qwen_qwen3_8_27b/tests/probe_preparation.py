# SPDX-License-Identifier: Apache-2.0
"""Finite-domain preparation followed by unseen logical lengths on live traces."""
import json,time
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    begin=time.monotonic()
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,checkpoint='.cache/qwen38-probe',layer_indices=[0,3],context_length=4352)
    print('PREPARED_SECONDS',time.monotonic()-begin,flush=True)
    ids=[str(gen.model_trace),str(gen.sample_trace)]
    cache=mesh.num_program_cache_entries()
    rows=[]
    for length in [1,2,3,31,32,33,127,128,129,255,256,257,4095,4096,4097,4349]:
        tokens=gen.generate([100]*length,3)
        after=mesh.num_program_cache_entries()
        assert after==cache,(length,cache,after)
        assert [str(gen.model_trace),str(gen.sample_trace)]==ids
        rows.append(dict(length=length,tokens=tokens,programs=after))
        print(json.dumps(rows[-1]),flush=True)
    logits=gen.prefill_logits([100]*259)
    assert tuple(logits.shape)==(1,259,248320)
    assert mesh.num_program_cache_entries()==cache
    again=gen.generate([100]*33,3)
    assert again==next(r['tokens'] for r in rows if r['length']==33)
    result=dict(scope='reduced real layers0/3 and full terminal; complete physical bucket domain',
        prepared_buckets=sorted(gen.prepared_buckets),trace_ids=ids,programs=cache,
        cases=rows,all_logits_shape=list(logits.shape),counters=gen.counters,status='pass',
        tracking=True,skip_program_cache=False)
    Path('models/autoports/qwen_qwen3_8_27b/doc/full_model/preparation_contract.json').write_text(json.dumps(result,indent=2)+'\n')
finally:
    ttnn.close_mesh_device(mesh)
