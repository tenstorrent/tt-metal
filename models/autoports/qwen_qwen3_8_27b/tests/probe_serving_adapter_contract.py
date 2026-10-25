# SPDX-License-Identifier: Apache-2.0
"""Reduced adapter API check. Run serially with tracking; no performance claim."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.model import Model
from models.autoports.qwen_qwen3_8_27b.tt.generator_vllm import Qwen38ForCausalLM

root=Path('models/autoports/qwen_qwen3_8_27b/readiness_vllm')
result={'scope':'reduced real layers0/3, adapter methods, B2/context512; not advertised capacity or performance'}
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=134217728)
adapter=None
try:
    # Deliberately bypass public construction only to reduce this targeted test's
    # pool. Production initialize_vllm_model continues to require native context.
    adapter=Qwen38ForCausalLM()
    adapter.model=Model('.cache/qwen38-full',mesh,layer_indices=[0,3])
    adapter.mesh=mesh; adapter.checkpoint='.cache/qwen38-full'
    adapter.batch_size=2; adapter.context_length=512
    adapter.request_seeds=[1234,5678]; adapter.generator=None; adapter.kv_cache=None
    adapter.allow_host_sampling=False; adapter.host_logits_reads=0
    adapter.state_pool=None; adapter.scheduler_block_size=32; adapter.slot_ends=[None]*2
    adapter.decode_slots=set(); adapter.trace_retirements=0; adapter.event_path=None
    cache=adapter.allocate_kv_cache((32,1,32,256),torch.bfloat16,1)
    adapter.warmup_model_prefill(kv_cache=cache)
    adapter.warmup_model_decode(kv_cache=cache)
    gen=adapter.generator
    trace_ids={str(k):str(v) for k,v in gen.decode_traces.items()}
    captures=gen.counters['captures']
    table=torch.arange(32,dtype=torch.int32).reshape(2,16)
    final_table=table.clone(); final_table[:,1]=torch.tensor([4,20])
    prompts=torch.arange(62,dtype=torch.int64).reshape(2,31)+1000
    lengths=torch.tensor([31,30])
    params=SimpleNamespace(temperature=[.7,.7],top_k=[4,4],top_p=[1.,1.],seed=[1234,5678])
    def read(t):return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()
    def finish(handle,events):
        for event in events:ttnn.event_synchronize(event)
        return adapter.process_decode_output_host(handle,is_tokens=True)
    def run(resident):
        gen.reset()
        prefill,deltas=adapter.prefill_forward(prompts,torch.zeros(2,dtype=torch.int64),lengths,
            table if resident else final_table,cache,empty_slots=[0,1],sampling_params=params)
        assert deltas.tolist()==[0,0]
        outputs=[prefill.clone()]
        counts=[]; pending=[]
        for step in range(4):
            full=not resident or step==0
            tokens=outputs[-1] if full else torch.full((2,1),99999999,dtype=torch.int64)
            positions=lengths+step if full else torch.full((2,),-999,dtype=torch.int64)
            output=adapter.decode_forward(tokens,positions,
                table if resident and step==0 else final_table,cache,sampling_params=params,
                read_from_device=not resident,reload_inputs=full,
                reload_page_table=resident and step==1,
                reload_sampling_params=full,reset_sampling_state=full,
                rope_deltas_all_users=torch.zeros(2,dtype=torch.int64))
            if resident:
                handle,events=adapter.read_decode_output(output,async_read=True)
                pending.append((handle,events))
                # Keep step N's host transfer alive until N+1 has been submitted.
                # Resident submissions never consult outputs[-1].
                if len(pending)>1:
                    outputs.append(finish(*pending.pop(0)).clone())
            else:
                outputs.append(output.clone())
            counts.append(dict(gen.counters))
        for transfer in pending:outputs.append(finish(*transfer).clone())
        states=[{name:read(t) for name,t in vars(state).items()} for state in cache]
        return torch.stack(outputs),states,counts,read(gen.positions)
    expected,expected_states,_,expected_positions=run(False)
    actual,actual_states,counts,positions=run(True)
    result.update(tokens_equal=torch.equal(expected,actual),positions_equal=torch.equal(expected_positions,positions),
        state_equal=[{name:torch.equal(t,actual_states[i][name]) for name,t in state.items()}
                     for i,state in enumerate(expected_states)],tokens=actual.tolist(),
        steady_refresh_delta={key:counts[-1][key]-counts[0][key]
            for key in ('token_refreshes','position_refreshes','page_table_refreshes')},
        delayed_read_consumed_after_next_submit=True,trace_ids=trace_ids,
        captures_before=captures,captures_after=gen.counters['captures'])
    assert result['tokens_equal'] and result['positions_equal'],result
    assert all(all(state.values()) for state in result['state_equal']),result
    assert result['steady_refresh_delta']==dict(token_refreshes=0,position_refreshes=0,page_table_refreshes=1),result
    assert trace_ids=={str(k):str(v) for k,v in gen.decode_traces.items()}
    assert captures==gen.counters['captures']
    # Synchronous plugin finalization receives raw multi-device sampled IDs.
    before_reads=gen.counters['readbacks']
    direct=adapter.process_decode_output_host(gen.tokens,is_tokens=True)
    direct_reads=gen.counters['readbacks']-before_reads
    handle,events=adapter.read_decode_output(gen.tokens,async_read=True)
    after_async_read=gen.counters['readbacks']
    already_read=finish(handle,events)
    async_finalize_reads=gen.counters['readbacks']-after_async_read
    result['output_finalization']={
        'raw_device_equals_async_host':torch.equal(direct,already_read),
        'raw_device_equals_last_token':torch.equal(direct,actual[-1]),
        'direct_device_readbacks':direct_reads,
        'async_host_finalize_extra_readbacks':async_finalize_reads}
    assert torch.equal(direct,already_read) and torch.equal(direct,actual[-1])
    assert direct_reads==1 and async_finalize_reads==0
    # Exercise the new state-copy operators against real nonzero request state.
    gen.remap_state_slots([1,0])
    for i,state in enumerate(cache):
        for name in ('conv','recurrent'):
            if hasattr(state,name):
                assert torch.equal(read(getattr(state,name)),actual_states[i][name].flip(0)),name
    moved=[{name:read(t) for name,t in vars(state).items()} for state in cache]
    gen.reset_state_slots([0])
    for i,state in enumerate(cache):
        for name in ('conv','recurrent'):
            if hasattr(state,name):
                value=read(getattr(state,name))
                assert value[0].count_nonzero()==0,name
                assert torch.equal(value[1],moved[i][name][1]),name
    result['remap_and_peer_preserving_reset']='exact'
    assert adapter.host_logits_reads==0
    result['host_logits_reads']=adapter.host_logits_reads
    result['status']='pass'
finally:
    if adapter is not None:
        ttnn.synchronize_device(mesh)
        adapter.release_persistent_capture()
    ttnn.close_mesh_device(mesh)
    (root/'adapter_device_contract.json').write_text(json.dumps(result,indent=2)+'\n')
