# SPDX-License-Identifier: Apache-2.0
"""Exact streamed/buffered autoregressive comparison with real reduced stack."""
import argparse,json,os,time
from pathlib import Path
import torch,ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=1);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(8)
ttnn.CONFIG.throw_exception_on_fallback=True
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,checkpoint='.cache/qwen38-probe',layer_indices=[0,3],batch_size=a.batch,context_length=1024,prepare=False)
    gen.prepare_decode(prefill_lengths=[1,32,64,96,128])
    traces=[*map(str,gen.decode_traces.values()),str(gen.sample_trace),str(gen.buffered_sample_trace)]
    captures=gen.counters['captures']; rows=[]
    cases=[(1,128,min(a.batch,3)),(8,130,min(a.batch,3)),(1,3,min(a.batch,3))]
    if a.batch>1:
        cases += [(k,8,active) for active in (a.batch,3,a.batch) for k in (1,8)]
    for k,n,active in cases:
        gen.set_sampling_params(top_k=k,top_p=0 if k==1 else .9,seed=42)
        prompts=[[100+i]*(33+32*(i%3)) for i in range(active)]
        expected=gen.generate(prompts,n,stop_on_eos=False,buffered_output=False)
        actual=gen.generate(prompts,n,stop_on_eos=False,buffered_output=True)
        assert actual==expected,(k,n,actual,expected)
        assert captures==gen.counters['captures']
        assert traces==[*map(str,gen.decode_traces.values()),str(gen.sample_trace),str(gen.buffered_sample_trace)]
        rows.append(dict(k=k,outputs=n,active_slots=active,exact=True,metrics=gen.last_generation_metrics))
    # Exercise repeated16-step EOS-aware windows in sampled mode, while an
    # empty EOS set ensures the comparison crosses several window boundaries.
    eos_ids=gen.eos_token_ids
    gen.eos_token_ids=set()
    try:
        gen.set_sampling_params(top_k=8,top_p=.9,seed=42)
        expected=gen.generate(prompts,35,stop_on_eos=True,buffered_output=False)
        actual=gen.generate(prompts,35,stop_on_eos=True)
        assert actual==expected
        assert gen.last_generation_metrics['output_window_steps']==16
        assert gen.last_generation_metrics['output_windows']==3
        assert captures==gen.counters['captures']
        assert traces==[*map(str,gen.decode_traces.values()),str(gen.sample_trace),str(gen.buffered_sample_trace)]
        rows.append(dict(k=8,outputs=35,active_slots=len(prompts),exact=True,eos_set_empty_for_boundary_test=True,metrics=gen.last_generation_metrics))
    finally:
        gen.eos_token_ids=eos_ids
    Path(a.output).write_text(json.dumps(dict(scope='Reduced real layers0/3 plus full terminal and sampler; not full-model performance',batch=a.batch,rows=rows,trace_ids=traces,captures=captures,watcher=os.getenv('TT_METAL_WATCHER'),watcher_noinline=os.getenv('TT_METAL_WATCHER_NOINLINE'),watcher_disable_eth=os.getenv('TT_METAL_WATCHER_DISABLE_ETH'),tracking=os.getenv('TT_METAL_TRACE_ALLOC_TRACKING'),skip_program_cache=os.getenv('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE')),indent=2)+'\n')
    print('BUFFERED_EXACT_PASS',flush=True)
finally:ttnn.close_mesh_device(mesh)
