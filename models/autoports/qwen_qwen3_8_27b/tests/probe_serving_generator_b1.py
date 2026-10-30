# SPDX-License-Identifier: Apache-2.0
"""Serial reduced device check; must not run while a server owns the mesh."""
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.model import Model
from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator

root=Path('models/autoports/qwen_qwen3_8_27b/readiness_vllm')
result={'scope':'reduced B1 real layers0/3; not model accuracy or performance'}
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=134217728)
gen=None
try:
    model=Model('.cache/qwen38-full',mesh,layer_indices=[0,3])
    cache=model.allocate_cache(batch_size=1,context_length=512)
    table=torch.arange(16,dtype=torch.int32).reshape(1,16)
    gen=Generator(model,'.cache/qwen38-full',batch_size=1,context_length=512,
                  kv_cache=cache,page_table=table,serving_state=True)
    gen.prepare_decode()
    gen.set_sampling_params(top_k=1)
    final_table=table.clone(); final_table[:,1]=torch.tensor([4])
    prompts=torch.arange(31,dtype=torch.int64).reshape(1,31)+1000
    lengths=[31]
    def read(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()
    def run(resident):
        gen.reset()
        current_table=table if resident else final_table
        logits=gen.prefill_forward(prompts,page_table=current_table,kv_cache=cache,
                                  prompt_lens=lengths,return_device=True)
        gen.reset_sampling_state(seed=1234,offsets=31,slots=[0])
        gen.sample_prefill(logits,active_slots=[0])
        outputs=[read(gen.tokens).reshape(-1).long()]
        counts=[]
        for step in range(4):
            if resident and step:
                # Poisoned values would violate context if accidentally inspected.
                token=torch.full((1,),99999999,dtype=torch.int64)
                position=torch.full((1,),-999,dtype=torch.int64)
            else:
                token=outputs[-1]
                position=torch.tensor(lengths)+step
            out=gen.decode_forward(token,position,
                page_table=(final_table if step or not resident else table),kv_cache=cache,
                return_device=True,reload_inputs=not resident or step==0,
                reload_page_table=resident and step==1)
            outputs.append(read(out).reshape(-1).long())
            counts.append(dict(gen.counters))
        states=[{name:read(t) for name,t in vars(state).items()} for state in cache]
        return torch.stack(outputs),states,counts,read(gen.positions)
    expected,expected_states,_,expected_positions=run(False)
    actual,actual_states,counts,positions=run(True)
    result['tokens_equal']=torch.equal(expected,actual)
    result['positions_equal']=torch.equal(expected_positions,positions)
    result['state_equal']=[{name:torch.equal(value,actual_states[i][name])
                            for name,value in state.items()} for i,state in enumerate(expected_states)]
    result['tokens']=actual.tolist()
    result['steady_refresh_delta']={key:counts[-1][key]-counts[0][key]
        for key in ('token_refreshes','position_refreshes','page_table_refreshes')}
    result['trace_ids']={str(k):str(v) for k,v in gen.decode_traces.items()}
    result['captures']=gen.counters['captures']
    assert result['tokens_equal'] and result['positions_equal'],result
    assert all(all(state.values()) for state in result['state_equal']),result
    assert result['steady_refresh_delta']==dict(token_refreshes=0,position_refreshes=0,page_table_refreshes=1),result
    before_seeds=read(gen.seeds)
    before_params={name:read(getattr(gen,name)) for name in ('k','p','temp')}
    before_states=[{name:read(t) for name,t in vars(state).items()} for state in cache]
    before_counters=dict(gen.counters)
    gen.remap_state_slots([0])
    assert gen._decode_active is None
    assert torch.equal(before_seeds,read(gen.seeds))
    assert before_counters==gen.counters
    assert all(torch.equal(value,read(getattr(gen,name))) for name,value in before_params.items())
    assert all(torch.equal(value,read(getattr(cache[i],name)))
               for i,state in enumerate(before_states) for name,value in state.items())
    result['singleton_identity']={'all_state_and_rng_exact':True,'counter_delta':0}
    result['status']='pass'
finally:
    if gen is not None:
        ttnn.synchronize_device(mesh)
        traces=set(gen.decode_traces.values())
        traces.update(t for t in (gen.sample_trace,gen.buffered_sample_trace) if t is not None)
        for trace in traces:ttnn.release_trace(mesh,trace)
    ttnn.close_mesh_device(mesh)
    (root/'generator_b1_device_contract.json').write_text(json.dumps(result,indent=2)+'\n')
