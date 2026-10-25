# SPDX-License-Identifier: Apache-2.0
"""Allocation-event audit only: no device profiler and no performance claims."""
import gzip
import hashlib
import json
from pathlib import Path
import torch
import ttnn


def allocation_peak(gen, prompt, evidence_path):
    gen.reset()
    gen.refresh(torch.zeros(1,dtype=torch.int32),gen.output_index)
    ttnn.synchronize_device(gen.mesh)
    before=ttnn.get_memory_view(gen.mesh,ttnn.BufferType.DRAM)
    banks=before.num_banks
    baseline=before.total_bytes_allocated_per_bank*banks
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        gen.reset()
        logits=gen.prefill_forward(torch.tensor(prompt,dtype=torch.int64),page_table=gen.page_table,
            kv_cache=gen.kv_cache,prompt_lens=[len(p) for p in prompt],return_device=True)
        del logits
        gen.refresh(torch.full((gen.batch_size,),len(prompt[0]),dtype=torch.int32),gen.positions)
        gen.refresh(torch.full((gen.batch_size,1),len(prompt[0]),dtype=torch.int32),gen.rope_indices)
        gen._decode()
        gen._sample_buffered()
        if gen.batch_size>1:
            gen._decode(masked=True)
            gen._sample_buffered()
        ttnn.synchronize_device(gen.mesh)
    finally:
        graph=ttnn.graph.end_graph_capture()
    events=[]
    live={}
    current={}
    peak={}
    for node in graph:
        kind=node['node_type']
        params=node.get('params',{})
        if kind not in ('buffer_allocate','buffer_deallocate') or params.get('type')!='DRAM':
            continue
        device=int(params['device_id']); address=int(params['address'])
        size=int(params['max_size_per_bank'])*banks
        key=(device,address)
        events.append([kind,device,address,size])
        if kind=='buffer_allocate':
            assert key not in live,('duplicate active allocation',key)
            live[key]=size
            current[device]=current.get(device,0)+size
            peak[device]=max(peak.get(device,0),current[device])
        else:
            assert key in live,('preexisting allocation released during audit',key)
            current[device]-=live.pop(key)
    assert events,'Allocation audit captured no DRAM events'
    raw=json.dumps(events,separators=(',',':')).encode()
    with gzip.open(evidence_path,'wb') as f:
        f.write(raw)
    after=ttnn.get_memory_view(gen.mesh,ttnn.BufferType.DRAM)
    allocator_after=after.total_bytes_allocated_per_bank*banks
    expected_device=gen.mesh.id()
    assert set(peak)=={expected_device},('Unexpected graph allocator domains',expected_device,peak)
    assert allocator_after-baseline==current[expected_device],('Allocation-event reconciliation failed',baseline,allocator_after,current)
    return dict(scope='all64layer request reset, eager prefill, all-active/masked decode and common sampler plus output collector allocation events; not device timing',
        input_tokens_per_user=len(prompt[0]),batch=gen.batch_size,num_banks=banks,
        baseline_allocated_bytes_per_device=baseline,additional_peak_bytes_by_graph_device=peak,
        peak_allocated_bytes_per_device_upper_bound=baseline+max(peak.values()),
        final_additional_live_bytes_by_graph_device=current,
        allocator_after_bytes_per_device=allocator_after,mesh_allocator_id=expected_device,
        event_allocator_reconciliation_exact=True,
        allocation_event_count=len(events),events_file=Path(evidence_path).name,events_sha256=hashlib.sha256(raw).hexdigest(),
        basis='C++ graph buffer_allocate/deallocate events; actual aligned_size_per_bank times DRAM bank count. Add maximum simultaneous new allocations to pre-audit allocator baseline. Trace replay reuses capture storage; eager decode bounds its scratch. No timings from this diagnostic enter performance.')


def release_trace_bytes(gen):
    ttnn.synchronize_device(gen.mesh)
    before=ttnn.get_memory_view(gen.mesh,ttnn.BufferType.DRAM)
    ids=list(gen.decode_traces.values())+[gen.sample_trace,gen.buffered_sample_trace]
    for trace in ids:
        ttnn.release_trace(gen.mesh,trace)
    ttnn.synchronize_device(gen.mesh)
    after=ttnn.get_memory_view(gen.mesh,ttnn.BufferType.DRAM)
    released=(before.total_bytes_allocated_per_bank-after.total_bytes_allocated_per_bank)*before.num_banks
    assert released>0,('No DRAM released by dynamic nonempty traces',ids)
    return dict(trace_ids=[str(t) for t in ids],
        released_trace_bytes_per_device=released,
        basis='Allocator delta immediately before/after releasing every live trace, after all benchmark/audit work. No model/request tensors are released.')
