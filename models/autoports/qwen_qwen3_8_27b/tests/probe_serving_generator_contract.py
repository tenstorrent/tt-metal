# SPDX-License-Identifier: Apache-2.0
"""Serial reduced device check; must not run while a server owns the mesh."""
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.model import Model
from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator

root=Path('models/autoports/qwen_qwen3_8_27b/readiness_vllm')
result={'scope':'reduced real layers0/3; not model accuracy or performance'}
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=134217728)
gen=None
try:
    model=Model('.cache/qwen38-full',mesh,layer_indices=[0,3])
    cache=model.allocate_cache(batch_size=2,context_length=512)
    table=torch.arange(32,dtype=torch.int32).reshape(2,16)
    gen=Generator(model,'.cache/qwen38-full',batch_size=2,context_length=512,
                  kv_cache=cache,page_table=table,serving_state=True)
    gen.prepare_decode()
    gen.set_sampling_params(top_k=1)
    final_table=table.clone(); final_table[:,1]=torch.tensor([4,20])
    prompts=torch.arange(62,dtype=torch.int64).reshape(2,31)+1000
    lengths=[31,30]
    def read(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()
    def run(resident):
        gen.reset()
        current_table=table if resident else final_table
        logits=gen.prefill_forward(prompts,page_table=current_table,kv_cache=cache,
                                  prompt_lens=lengths,return_device=True)
        gen.sample_prefill(logits)
        outputs=[read(gen.tokens).reshape(-1).long()]
        counts=[]
        for step in range(4):
            if resident and step:
                # Poisoned values would violate context if accidentally inspected.
                token=torch.full((2,),99999999,dtype=torch.int64)
                position=torch.full((2,),-999,dtype=torch.int64)
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
    # Exact integer peer preservation after a terminal prefill samples only slot0.
    # Values cross float precision and sign boundaries; every replica must agree.
    seed_values=torch.arange(32,dtype=torch.int64)+1234
    seed_values[:4]=torch.tensor([0x1000001,0x80000011,0xfffffc00,0x7fffffff])
    gen.refresh(seed_values.to(torch.int32),gen.seeds)
    gen.reset_sampling_state(seed=[0x1000001,42],offsets=[7,9],slots=[0])
    expected_seed=seed_values.clone();expected_seed[0]=0x1000008
    def check_seeds(expected):
        for replica in ttnn.get_device_tensors(gen.seeds):
            actual=ttnn.to_torch(replica).reshape(-1).to(torch.int64)&0xffffffff
            assert torch.equal(actual,expected),(actual.tolist(),expected.tolist())
    check_seeds(expected_seed)
    captures=gen.counters['captures']
    gen.sample_prefill(gen.logits,active_slots=[0])
    expected_seed[0]+=1
    check_seeds(expected_seed)
    gen.sample_prefill(gen.logits,active_slots=[1])
    expected_seed[1]+=1
    check_seeds(expected_seed)
    assert captures==gen.counters['captures']
    result['subset_rng']={'all_replicas_exact':True,'inactive_counters_preserved':True,
                          'high_uint32_values':True,'new_captures':0}
    result['status']='pass'
finally:
    if gen is not None:
        ttnn.synchronize_device(mesh)
        traces=set(gen.decode_traces.values())
        traces.update(t for t in (gen.sample_trace,gen.buffered_sample_trace) if t is not None)
        for trace in traces:ttnn.release_trace(mesh,trace)
    ttnn.close_mesh_device(mesh)
    (root/'generator_device_contract.json').write_text(json.dumps(result,indent=2)+'\n')
