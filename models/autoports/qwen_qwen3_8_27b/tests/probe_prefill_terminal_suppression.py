# SPDX-License-Identifier: Apache-2.0
"""Compare suppressed terminal work with the identical chunked control."""
import json
import os
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.model import Model
from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator

root=Path('models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm')
assert os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')=='1'
assert os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0')=='0'
result=dict(scope='Reduced real layers0/3 B2 context512, correctness only',cases=[],status='incomplete')
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=134217728)
gen=None
try:
    model=Model('.cache/qwen38-full',mesh,layer_indices=[0,3])
    cache=model.allocate_cache(batch_size=2,context_length=512)
    table=torch.arange(32,dtype=torch.int32).reshape(2,16)
    gen=Generator(model,'.cache/qwen38-full',batch_size=2,context_length=512,
                  kv_cache=cache,page_table=table,serving_state=True,prefill_chunk_size=128)
    gen.prepare_decode()
    original=model.terminal
    calls=[]
    def terminal(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    model.terminal=terminal
    def read(value):
        return [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(value)]
    def run(parts,skip):
        gen.reset()
        gen.set_sampling_params(top_k=1)
        gen.reset_sampling_state(seed=[1234,5678],offsets=0)
        tokens=torch.arange(2*sum(parts),dtype=torch.int64).reshape(2,-1)%251+100
        gen.prefill_forward(tokens[:,:33],page_table=table,kv_cache=cache,
                            prompt_lens=[0,33],return_device=True)
        calls.clear()
        start=0
        for i,count in enumerate(parts):
            final=i==len(parts)-1
            logits=gen.prefill_continuation(tokens[:,start:start+count],page_table=table,
                kv_cache=cache,prompt_lens=[count,0],start_position=start,
                compute_logits=not skip or final)
            if skip and not final:assert logits is None
            start+=count
        terminal_calls=len(calls)
        values={'logits':read(logits)}
        sampled=gen.sample_prefill(logits,active_slots=[0])
        values['tokens']=read(sampled)
        # The resident trace must still consume the final prefill token/state.
        for step in range(3):
            output=gen.decode_forward(torch.tensor([[100],[0]]),torch.tensor([sum(parts)+step,-1]),
                page_table=table,kv_cache=cache,return_device=True,reload_inputs=step==0,
                reload_page_table=False)
            values[f'decode{step}']=read(output)
        for index,state in enumerate(cache):
            for name,value in vars(state).items():values[f'{index}:{name}']=read(value)
        return values,terminal_calls
    captures=gen.counters['captures']
    for parts in ([128,128,1],[64,64,65],[123,6],[128,1]):
        expected,old_calls=run(parts,False)
        actual,new_calls=run(parts,True)
        equal={key:all(torch.equal(a,b) for a,b in zip(value,actual[key])) for key,value in expected.items()}
        result['cases'].append(dict(parts=parts,exact=equal,terminal_calls_before=old_calls,
                                    terminal_calls_after=new_calls))
        assert all(equal.values()),result['cases'][-1]
    result.update(status='pass',captures_before=captures,captures_after=gen.counters['captures'],
                  trace_ids=[str(t) for t in gen.decode_traces.values()],tracking=True,program_cache_included=True)
    assert captures==gen.counters['captures']
finally:
    if gen:
        ttnn.synchronize_device(mesh)
        for trace in set(gen.decode_traces.values())|{gen.sample_trace,gen.buffered_sample_trace}:
            if trace is not None:ttnn.release_trace(mesh,trace)
    ttnn.close_mesh_device(mesh)
    (root/'terminal_suppression_device.json').write_text(json.dumps(result,indent=2)+'\n')
