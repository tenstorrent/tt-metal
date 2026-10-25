# SPDX-License-Identifier: Apache-2.0
"""Reduced real-terminal replay and cross-request ownership checks."""
import argparse,json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

parser=argparse.ArgumentParser();parser.add_argument('--output',default='models/autoports/qwen_qwen3_8_27b/doc/full_model/reduced_trace_contract.json');args=parser.parse_args()
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,prepare=False,checkpoint='.cache/qwen38-probe',
                        layer_indices=[0,3],context_length=512)
    gen.prepare_decode(prefill_lengths=(33,))
    trace_ids=[gen.model_trace,gen.sample_trace,gen.buffered_sample_trace]
    def host(t): return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1).tolist()
    a=gen.generate([100]*33,8)
    assert host(gen.positions)==[40] and host(gen.rope_indices)==[40]
    assert host(gen.tokens)==[a[-1]]
    # Reconcile the host map once after using the externally mutable device API.
    gen.refresh_page_table(gen.page_table_host.clone())
    before=dict(gen.counters)
    gen.refresh_page_table(gen.page_table_host.clone())
    assert gen.counters['page_table_refreshes']==before['page_table_refreshes']
    changed=gen.page_table_host.flip(-1).contiguous()
    gen.refresh_page_table(changed)
    assert gen.counters['page_table_refreshes']==before['page_table_refreshes']+1
    b=gen.generate([100]*33,8)
    assert a==b,(a,b)
    assert gen.counters['token_refreshes']==0 and gen.counters['position_refreshes']==0
    assert [gen.model_trace,gen.sample_trace,gen.buffered_sample_trace]==trace_ids and gen.counters['captures']==3
    # Explicit scheduler override uses the exact captured input buffers.
    gen.decode_forward(torch.tensor([[101]]),torch.tensor([40]),page_table=changed,kv_cache=gen.kv_cache,return_logits=True,return_device=True)
    assert host(gen.tokens)==[101] and host(gen.positions)==[41] and host(gen.rope_indices)==[41]
    ttnn.execute_trace(mesh,gen.sample_trace,cq_id=0,blocking=False)
    first=host(gen.tokens)
    # Next model replay consumes this same token backing allocation without a host refresh.
    token_address=gen.tokens.buffer_address()
    ttnn.execute_trace(mesh,gen.model_trace,cq_id=0,blocking=False)
    assert host(gen.tokens)==first and host(gen.positions)==[42]
    ttnn.execute_trace(mesh,gen.sample_trace,cq_id=0,blocking=False)
    assert gen.tokens.buffer_address()==token_address
    gen.set_sampling_params(top_k=8,top_p=.9,temperature=2,seed=42)
    assert host(gen.seeds)[0]==42
    sampled_a=gen.generate([100]*33,8)
    sampled_b=gen.generate([100]*33,8)
    assert sampled_a==sampled_b,'Reset must restore configured RNG seeds'
    assert [gen.model_trace,gen.sample_trace,gen.buffered_sample_trace]==trace_ids
    gen.set_sampling_params(top_k=1,top_p=0,temperature=1,seed=1234)
    greedy=gen.generate([100]*33,8)
    compatibility=gen.generate([100]*33,8,host_sampling=True)
    assert greedy==compatibility,(greedy,compatibility)
    assert gen.counters['captures']==3
    original_stops=gen.eos_token_ids
    gen.eos_token_ids={220}
    stopped=gen.generate([100]*33,8)
    fixed=gen.generate([100]*33,8,stop_on_eos=False)
    assert stopped==[198,220] and len(fixed)==8,(stopped,fixed)
    teacher_fixed=gen.generate([100]*33,8,next_input=lambda step,token:token)
    assert len(teacher_fixed)==8
    gen.eos_token_ids=original_stops
    high_seed=(1<<32)-1-gen.context_length
    gen.set_sampling_params(top_k=8,top_p=.9,temperature=2,seed=high_seed)
    assert host(gen.seeds)[0]==high_seed
    high_a=gen.generate([100]*33,8)
    high_b=gen.generate([100]*33,8)
    assert high_a==high_b
    result=dict(scope='reduced layers0/3; exact33-token prepared signature only',
                high_seed=high_seed,high_seed_outputs=high_a,
                eos_stopped=stopped,fixed_length_tokens=fixed,eos_ids=sorted(original_stops),
                sampled_outputs=sampled_a,greedy_outputs=greedy,host_compatibility_outputs=compatibility,
                outputs=a,page_remap_outputs=b,trace_ids=[str(x) for x in trace_ids],
                counters=gen.counters,final_positions=host(gen.positions),
                token_feedback='same persistent allocation consumed before next sampler write',
                tracking=True,skip_program_cache=False,status='pass')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
finally:
    ttnn.close_mesh_device(mesh)
