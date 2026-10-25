# SPDX-License-Identifier: Apache-2.0
"""Reduced arbitrary-offset prefill check with peer isolation and no sampling."""
import json
import math
import os
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.model import Model
from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator

root=Path('models/autoports/qwen_qwen3_8_27b/readiness_vllm')
result=dict(scope='reduced real layers0/3,B2,context512; no model accuracy/performance',
    physical_prefill_chunk_size=128,
    trace_allocation_tracking=os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING'),
    skip_program_cache=os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0'),
    numerical_gate={'pcc_min':.995,'normalized_rmse_max':.1,
        'basis':'Inherited whole-vs-chunk recurrence/logits PCC gate; added scale-sensitive normalized RMSE. BF16/BFP8/BFP4 paths have different reduction order.'},cases=[])
if result['trace_allocation_tracking']!='1' or result['skip_program_cache']!='0':
    raise RuntimeError('Run with allocation tracking and program-cache allocations included')
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
    captures=gen.counters['captures']
    trace_ids={str(k):str(v) for k,v in gen.decode_traces.items()}
    def read(t):return [ttnn.to_torch(local).clone() for local in ttnn.get_device_tensors(t)]
    def snapshot():return [{name:read(t) for name,t in vars(state).items()} for state in cache]
    def peer_equal(before,after):
        for old,new in zip(before,after):
            for name,replicas in old.items():
                for a,b in zip(replicas,new[name]):
                    index=1 if name in ('conv','recurrent') else slice(16,32)
                    if not torch.equal(a[index],b[index]):return False
        return True
    def metrics(a,b):
        a=a.float().flatten(); b=b.float().flatten()
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        exact=torch.equal(a,b)
        if exact:pcc=1.
        elif a.std()==0 or b.std()==0:pcc=0.
        else:pcc=float(torch.corrcoef(torch.stack((a,b)))[0,1])
        rms=float(torch.sqrt(torch.mean((a-b)**2)))
        nrms=rms/max(float(torch.sqrt(torch.mean(a*a))),1e-8)
        return dict(pcc=pcc,normalized_rmse=nrms,max_abs=float((a-b).abs().max()),
                    exact=exact,finite=finite,pass_gate=finite and pcc>=.995 and nrms<=.1)
    def compare(left,right,length):
        comparisons={'logits':metrics(left['logits'][0],right['logits'][0])}
        for layer,(old,new) in enumerate(zip(left['states'],right['states'])):
            for name,replicas in old.items():
                for rank,(a,b) in enumerate(zip(replicas,new[name])):
                    if name in ('conv','recurrent'):
                        a,b=a[0],b[0]
                    else:
                        # Compare only active logical KV entries, not zero pool padding.
                        pages=math.ceil(length/32)
                        a=a[:pages,0].reshape(-1,256)[:length]
                        b=b[:pages,0].reshape(-1,256)[:length]
                    comparisons[f'layer{layer}:{name}:rank{rank}']=metrics(a,b)
        return comparisons
    words=gen.tokenizer.encode('A careful research team compares every observation with its control before deciding what changed. ',add_special_tokens=False)
    def run(parts):
        total=sum(parts)
        tokens=torch.zeros(2,total,dtype=torch.int64)
        tokens[0]=torch.tensor((words*((total+len(words)-1)//len(words)))[:total])
        gen.reset()
        peer=torch.zeros(2,67,dtype=torch.int64)
        peer[1]=torch.tensor((words*10)[:67])
        gen.prefill_forward(peer,page_table=table,kv_cache=cache,prompt_lens=[0,67],return_device=True)
        peer_before=snapshot(); seeds=read(gen.seeds); samplings=gen.counters['sampling_replays']
        chunks=[]; start=0
        for count in parts:
            logits=gen.prefill_continuation(tokens[:,start:start+count],page_table=table,
                kv_cache=cache,prompt_lens=[count,0],start_position=start)
            # Terminal logits are shard-composed only in this correctness probe.
            host_logits=gen.host_logits(logits)
            state=snapshot()
            assert peer_equal(peer_before,state),'Inactive peer state changed'
            assert all(torch.equal(a,b) for a,b in zip(seeds,read(gen.seeds))),'Bridge advanced RNG'
            assert gen.counters['sampling_replays']==samplings,'Bridge sampled an intermediate token'
            start+=count
            chunks.append(dict(end=start,logits=host_logits,states=state))
        return chunks
    failed=False
    for parts in ([64,64,65],[123,6]):
        length=sum(parts)
        whole=run([length])[-1]
        chunks=run(parts)
        comparison=compare(whole,chunks[-1],length)
        bad=[name for name,m in comparison.items() if not m['pass_gate']]
        case=dict(total=length,chunks=parts,starts=[sum(parts[:i]) for i in range(len(parts))],
                  comparisons=comparison,inactive_peer_exact=True,sampling_advance=0,
                  status='pass' if not bad else 'fail',failing_fields=bad)
        if bad:
            # Locate the earliest boundary that differs from a fresh whole-prefix
            # control. Report real measurements; do not widen gates after failure.
            case['first_divergence']=None
            for chunk in chunks:
                control=run([chunk['end']])[-1]
                partial=compare(control,chunk,chunk['end'])
                divergent=[name for name,m in partial.items() if not m['pass_gate']]
                if divergent:
                    case['first_divergence']=dict(end=chunk['end'],fields=divergent,comparisons=partial)
                    break
            failed=True
        result['cases'].append(case)
    result.update(captures_before=captures,captures_after=gen.counters['captures'],trace_ids=trace_ids,
                  status='fail' if failed else 'pass')
    assert captures==gen.counters['captures']
    assert not failed,'Whole-vs-off-grid continuation numerical gate failed; inspect JSON first divergence'
finally:
    if gen is not None:
        ttnn.synchronize_device(mesh)
        traces=set(gen.decode_traces.values())
        traces.update(t for t in (gen.sample_trace,gen.buffered_sample_trace) if t is not None)
        for trace in traces:ttnn.release_trace(mesh,trace)
    ttnn.close_mesh_device(mesh)
    (root/'continuation_probe.json').write_text(json.dumps(result,indent=2)+'\n')
