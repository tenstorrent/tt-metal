#!/usr/bin/env python3
"""Collect complete host phase intervals while the matching server is alive."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import time
import urllib.request


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_events(path):
    ready = json.loads(path.with_suffix('.ready.json').read_text())
    script = ('import os,signal,pathlib; p='+str(ready['pid'])+'; '
              'assert pathlib.Path(f"/proc/{p}/stat").read_text().rsplit(")",1)[1].split()[19]=='
              +repr(ready['start_ticks'])+'; os.kill(p,signal.SIGUSR1)')
    old = path.stat().st_mtime_ns if path.exists() else 0
    subprocess.run(['docker','exec','qwen38-functional-build','python','-c',script],check=True)
    deadline=time.monotonic()+30
    while (not path.exists() or path.stat().st_mtime_ns == old) and time.monotonic()<deadline:
        time.sleep(.1)
    if not path.exists() or path.stat().st_mtime_ns == old:
        raise RuntimeError('Live phase buffer did not flush')
    if json.loads(path.with_suffix('.flush.json').read_text())['pending_records']:
        raise RuntimeError('Unfinished async phase records remain; collect again after natural completion')
    return [json.loads(line) for line in path.read_text().splitlines()]


def partition(events, auxiliary, request_ids, *, window=None):
    """Exclusive scheduler service pieces with explicit deferred host ownership.

    A base step spans schedule entry to actual output-ready (never consumption).
    In an opposite-phase overlap, the earlier step owns time until its observed
    readiness. Actual drain spans own outgoing-phase overhead. Tagged state apply
    and scheduler consumption carve only their own host spans out of that base.
    Uncovered gaps are service only while measured requests are admitted and not
    finished; otherwise they are retained as request-free residuals. Admission
    before the first service step is also a named residual. No HTTP latency is
    used to infer a device completion or phase split.
    """
    if not events:
        raise ValueError('No measured phases')
    phases={e['step']:e for e in events}
    if len(phases)!=len(events):
        raise ValueError('Duplicate phase completion')
    ids=set(request_ids)
    if any(e['phase'] not in ('prefill','decode') or e['end']<=e['start'] for e in events):
        raise ValueError('Invalid complete phase interval')
    if any(any(r['request'] not in ids for r in e['rows']) for e in events):
        raise ValueError('Mixed measured/foreign phase')
    if any(not r['final'] for e in events if e['phase']=='prefill' for r in e['rows']):
        raise ValueError('Intermediate prefill has no exclusive completion boundary')
    admitted={}
    finished={}
    consumed={}
    tails=[]
    drains=[]
    for e in auxiliary:
        if e['event']=='admitted' and e['request'] in ids:
            if e['request'] in admitted:
                raise ValueError('Duplicate admission')
            admitted[e['request']]=e['end']
        if e['event']=='consumed':
            for rid in set(e.get('finished',())) & ids:
                if rid in finished:
                    raise ValueError('Duplicate finished request')
                finished[rid]=e['end']
            if e['step'] in phases:
                if e['step'] in consumed:
                    raise ValueError('Duplicate scheduler consumption')
                consumed[e['step']]=e
                if set(e['requests'])-ids:
                    raise ValueError('Mixed scheduler consumption')
                tails.append((e['start'],e['end'],phases[e['step']]['phase'],2,'consumed',e['step']))
            elif set(e.get('requests',())) & ids:
                raise ValueError('Measured scheduler consumption has no matched phase')
            else:
                tails.append((e['start'],e['end'],'residual',2,'foreign_consumed',e['step']))
        elif e['event']=='state_apply' and e.get('step') in phases:
            if set(e['requests'])-ids or e['phase']!=phases[e['step']]['phase']:
                raise ValueError('Unjoinable or mixed state application')
            tails.append((e['start'],e['end'],e['phase'],3,'state_apply',e['step']))
        elif e['event']=='state_apply':
            if set(e.get('requests',())) & ids:
                raise ValueError('Measured state application has no matched phase')
            # Warmup/probe outputs can be ready yet leave CPU bookkeeping until
            # the first measured build. Exclude only this observed foreign span,
            # never stretch the old request or charge its work to the new phase.
            tails.append((e['start'],e['end'],'residual',3,'foreign_state_apply',e.get('step')))
        elif e['event']=='drain' and e.get('step') in phases:
            origins=e['origin_steps']
            if not origins:
                continue  # empty drain overhead belongs to enclosing base step
            if any(s not in phases for s in origins):
                raise ValueError('Drain mixes measured and foreign pending outputs')
            kinds={phases[s]['phase'] for s in origins}
            if len(kinds)!=1 or set(e['origin_phases'])!=kinds:
                raise ValueError('Ambiguous mixed-phase drain')
            kind=next(iter(kinds))
            if any(phases[s]['end']>e['end'] for s in origins):
                raise ValueError('Drain ends before its actual outgoing completion')
            drains.append((e['start'],e['end'],kind,1,'drain',e['step']))
    if set(admitted)!=ids or set(finished)!=ids:
        raise ValueError('Missing admission/finished request boundaries')
    if set(consumed)!=set(phases):
        raise ValueError('Missing scheduler consumption boundaries')
    if any(finished[r]<admitted[r] for r in ids):
        raise ValueError('Invalid request lifetime')
    first=min(e['start'] for e in events)
    # The client completion limit is only an exclusion window, not a phase timer.
    lo,hi=window if window is not None else (min(admitted.values()),max(e['end'] for e in consumed.values()))
    if first<lo or any(e['end']>hi for e in events):
        raise ValueError('Measured execution escapes the completed client window')
    spans=tails+drains
    # Late state bookkeeping outside generation is retained by caller, not timed.
    spans=[(max(a,lo),min(b,hi),k,p,t,s) for a,b,k,p,t,s in spans if a<hi and b>lo]
    base=[(e['start'],e['end'],e['phase'],e['step']) for e in events]
    foreign=[e for e in auxiliary if e['event']=='phase' and e['step'] not in phases]
    points={lo,hi,first}
    for a,b,_,_ in base:
        points.update((a,b))
    for a,b,*_ in spans:
        points.update((a,b))
    for rid in ids:
        points.update((max(lo,min(hi,admitted[rid])),max(lo,min(hi,finished[rid]))))
    for e in foreign:
        if e['start']<hi and e['end']>lo:
            points.update((max(lo,e['start']),min(hi,e['end'])))
    pieces=[]
    for a,b in zip(sorted(points),sorted(points)[1:]):
        if b<=a:
            continue
        current=[x for x in spans if x[0]<b and x[1]>a]
        active=[x for x in base if x[0]<b and x[1]>a]
        live=[rid for rid in ids if admitted[rid]<b and finished[rid]>a]
        interference=[e for e in foreign if e['start']<b and e['end']>a]
        if interference and (active or live):
            raise ValueError('Foreign execution overlaps measured active service')
        if current:
            priority=max(x[3] for x in current)
            owners={x[2] for x in current if x[3]==priority}
            if len(owners)!=1:
                raise ValueError('Conflicting explicit host-span owners')
            kind=next(iter(owners)); reason='+'.join(sorted({x[4] for x in current if x[3]==priority}))
        elif active:
            # Earlier actual ready boundary decides outgoing ownership; never
            # unconditional prefill priority, and never ready→consume stretching.
            oldest=min(active,key=lambda x:(x[0],x[3]))
            kind=oldest[2]; reason='base_service'
        elif a<first:
            kind='residual'; reason='initial_admission_before_service'
        elif live:
            upcoming=[x for x in base if x[0]>=b and any(r['request'] in live for r in phases[x[3]]['rows'])]
            upcoming += [(x[0],x[1],x[2],x[5]) for x in spans if x[0]>=b and x[5] in phases
                         and any(r['request'] in live for r in phases[x[5]]['rows'])]
            if not upcoming:
                raise ValueError('Active request gap lacks a subsequent owned service boundary')
            kind=min(upcoming,key=lambda x:(x[0],x[3]))[2]; reason='active_request_host_gap'
        else:
            kind='residual'; reason='foreign_service' if interference else 'request_free_http_turnaround'
        pieces.append(dict(start=a,end=b,phase=kind,reason=reason))
    # This conservation check accompanies ownership tests; alone it is not proof.
    if abs(sum(p['end']-p['start'] for p in pieces)-(hi-lo))>1e-6:
        raise ValueError('Timeline does not conserve observed window')
    return pieces

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--concurrency',type=int,required=True)
    p.add_argument('--action',choices=['check','collect'],required=True)
    a=p.parse_args(); root=a.run_dir; n=a.concurrency
    server_path=root/f'perf-b{n}-server.json'
    server=json.loads(server_path.read_text())
    path=root/f'phases-b{n}.jsonl'
    ready=json.loads(path.with_suffix('.ready.json').read_text())
    assert ready['batch']==n and ready['layers']==list(range(64))
    assert ready['precision']['config_id']=='head_bfp8_hifi2'
    from benchmark_work import prefill_work, decode_work, PEAK_FLOPS, PEAK_DRAM, WORK_METHOD as BASIS
    if a.action=='check':
        # Distinct fixed token IDs exercise final prefill and actual async decode
        # readback, without profiling or any additional device synchronization.
        body=dict(model=server['model'],prompt=[5000+i%2000 for i in range(4096)],
                  max_tokens=2,temperature=0,ignore_eos=True)
        request=urllib.request.Request(server['base_url']+'/v1/completions',
            json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=300) as response:
            result=json.load(response)
        (root/f'roofline-b{n}-probe.json').write_text(json.dumps(result,indent=2)+'\n')
        events=read_events(path)
        probe_pattern=re.compile('^'+re.escape(result['id'])+r'-0-[a-f0-9]{8}$')
        probe_ids={e['request'] for e in events if e['event']=='request' and probe_pattern.fullmatch(e['request'])}
        assert len(probe_ids)==1, ('Probe request identity missing/ambiguous',result['id'])
        phases=[e for e in events if e['event']=='phase' and any(r['request'] in probe_ids for r in e['rows'])]
        probe_pieces=partition(phases,events,probe_ids)
        (root/f'roofline-b{n}-probe-partition.json').write_text(json.dumps(
            dict(request_ids=sorted(probe_ids),phases=phases,disjoint_timeline=probe_pieces,
                 raw_events_sha256=sha(path)),indent=2)+'\n')
        assert {'prefill','decode'} <= {e['phase'] for e in phases}
        assert all(e['end']>e['start'] for e in phases)
        assert prefill_work([dict(start=0,length=4096,final=True)],n)>0
        assert decode_work([dict(position=4096)],n)>0
        print('READY: actual prefill and asynchronous decode completion; model/hardware accounting loaded')
        return
    raw_path=root/f'perf-b{n}.json'; raw=json.loads(raw_path.read_text())
    events=read_events(path)
    starts=raw['start_times']; lo=min(starts)
    # These are client perf_counter starts for measured requests, not the
    # process wall window (which also contains its test request).
    hi=max(s+t+sum(gaps) for s,t,gaps in zip(starts,raw['ttfts'],raw['itls']))
    prefix=json.loads((root/f'perf-b{n}-client.json').read_text())['request_id_prefix']
    pattern=re.compile(r'^cmpl-'+re.escape(prefix)+r'(\d+)-0-[a-f0-9]{8}$')
    reqs={e['request']:e for e in events if e['event']=='request' and pattern.fullmatch(e['request'])}
    assert sorted(int(pattern.fullmatch(r)[1]) for r in reqs)==list(range(raw['completed']))
    assert len(reqs)==raw['completed'], (len(reqs),raw['completed'],lo,hi)
    assert all(e['prompt_tokens']==4096 for e in reqs.values())
    assert len({e['prompt_sha256'] for e in reqs.values()})==len(reqs), 'Performance prompts must be distinct'
    input_path=root/f'perf-b{n}-inputs.jsonl'
    client_inputs=[json.loads(line) for line in input_path.read_text().splitlines()]
    assert sorted(row['index'] for row in client_inputs)==list(range(raw['completed']))
    assert all(len(row['tokens'])==4096 and
               hashlib.sha256(json.dumps(row['tokens']).encode()).hexdigest()==row['prompt_sha256']
               for row in client_inputs)
    by_index={row['index']:row for row in client_inputs}
    assert all(by_index[int(pattern.fullmatch(rid)[1])]['prompt_sha256']==event['prompt_sha256']
               for rid,event in reqs.items()), 'Client token arrays differ from actual engine input'
    selected=[e for e in events if e['event']=='phase' and any(r['request'] in reqs for r in e['rows'])]
    assert selected and all(all(r['request'] in reqs for r in e['rows']) for e in selected), 'Mixed probe/measurement step'
    coverage={rid:dict(prefill=0,decode=0) for rid in reqs}
    for e in selected:
        for r in e['rows']:
            coverage[r['request']][e['phase']]+=r['length'] if e['phase']=='prefill' else 1
            if e['phase']=='prefill':
                assert r['final'], 'Intermediate prefill needs a later completion boundary; refusing enqueue time'
    assert all(c['prefill']==4096 and c['decode']>=127 for c in coverage.values()), coverage
    assert all(c['decode']<=129 for c in coverage.values()), 'Unexpected execution/replay count'
    pieces=partition(selected,events,reqs,window=(lo,hi))
    selected_steps={e['step'] for e in selected}
    enqueued={e['step'] for e in events if e['event']=='state_enqueued' and e.get('step') in selected_steps}
    applied={e['step'] for e in events if e['event']=='state_apply' and e.get('step') in selected_steps and e['end']<=hi}
    bookkeeping=dict(enqueued_steps=sorted(enqueued),applied_in_window=sorted(applied),
        not_executed_in_measurement=sorted(enqueued-applied),
        note='Queued terminal state bookkeeping is not output readback; unperformed spans have no invented duration.')
    published={rid:0 for rid in reqs}
    for event in events:
        if event['event']=='consumed' and event['step'] in selected_steps:
            for rid,count in event['published_tokens'].items():
                if rid in published: published[rid]+=count
    assert all(v==128 for v in published.values()), ('Published output count mismatch',published)
    seconds={k:sum(e['end']-e['start'] for e in pieces if e['phase']==k) for k in ('prefill','decode')}
    flops=sum(prefill_work(e['rows'],n) for e in selected if e['phase']=='prefill')
    traffic=sum(decode_work(e['rows'],n) for e in selected if e['phase']=='decode')
    evidence=root/f'phase-accounting-b{n}.json'
    evidence.write_text(json.dumps(dict(ready=ready,request_mapping=reqs,coverage=coverage,
        phases=selected,disjoint_timeline=pieces,seconds=seconds,prefill_flops=flops,decode_dram_bytes=traffic,
        deferred_bookkeeping=bookkeeping,published_tokens=published,
        residual_seconds=sum(e["end"]-e["start"] for e in pieces if e["phase"]=="residual"),
        basis=BASIS,window=dict(client_start_min=lo,client_end_max=hi),
        raw_events_sha256=sha(path)),indent=2)+'\n')
    common=dict(timing_scope='full_phase_wall_time',
        timing_method='Actual output-ready boundaries with outgoing-phase overlap ownership; explicitly tagged drain/state-apply/consumption spans; admitted-to-finished active host gaps; request-free residuals excluded; no added synchronization',
        work_method=BASIS,peak_source='https://docs.tenstorrent.com/aibs/blackhole/p300.html; matmul_device_operation.cpp performance model',
        evidence=evidence.name,evidence_sha256=sha(evidence))
    from benchmark_stage.subsets import digest
    entry=dict(performance_sha256=sha(raw_path),server_identity_sha256=digest(server),
        requests=raw['completed'],input_tokens=raw['total_input_tokens'],output_tokens=raw['total_output_tokens'],
        prefill=dict(common,flops=flops,seconds=seconds['prefill'],peak_flops_per_second=PEAK_FLOPS),
        decode=dict(common,dram_bytes=traffic,seconds=seconds['decode'],peak_dram_bytes_per_second=PEAK_DRAM))
    target=root/'roofline.json'; data=json.loads(target.read_text()) if target.exists() else {}
    data[str(n)]=entry; target.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(batch=n,requests=len(reqs),seconds=seconds,flops=flops,dram_bytes=traffic)))


if __name__=='__main__':
    main()
