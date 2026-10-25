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


def partition(events):
    """Disjoint phase timeline. Prefill work cannot overlap pending TT decode:
    runner drains at that transition. Earlier host build/drain belongs to decode.
    Decode pipeline overlaps are unioned, never summed. Idle gaps within one
    active measurement epoch are charged to the next phase, including dispatch.
    """
    intervals=[]
    for e in events:
        start=e['work_start'] if e['phase']=='prefill' else e['start']
        intervals.append((start,e['end'],e['phase']))
    points=sorted({min(e['start'] for e in events)} | {v for a,b,_ in intervals for v in (a,b)})
    pieces=[]
    for a,b in zip(points,points[1:]):
        active={kind for x,y,kind in intervals if x < b and y > a}
        if len(active)>1:
            # A decode execute may first drain a previous prefill/output. The
            # actual prefill forward owns that overlap, not enqueue timestamps.
            kind='prefill'
        elif active:
            kind=next(iter(active))
        else:
            kind=next(k for x,y,k in sorted(intervals) if x>=b)
        pieces.append(dict(start=a,end=b,phase=kind))
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
        phases=[e for e in events if e['event']=='phase']
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
    selected=[e for e in events if e['event']=='phase' and any(r['request'] in reqs for r in e['rows'])]
    consumed={e['step']:e['end'] for e in events if e['event']=='consumed'}
    assert all(e['step'] in consumed for e in selected), 'Missing scheduler consumption boundaries'
    for e in selected:
        e['output_ready']=e['end']
        e['end']=max(e['end'],consumed[e['step']])
    assert selected and all(all(r['request'] in reqs for r in e['rows']) for e in selected), 'Mixed probe/measurement step'
    coverage={rid:dict(prefill=0,decode=0) for rid in reqs}
    for e in selected:
        for r in e['rows']:
            coverage[r['request']][e['phase']]+=r['length'] if e['phase']=='prefill' else 1
            if e['phase']=='prefill':
                assert r['final'], 'Intermediate prefill needs a later completion boundary; refusing enqueue time'
    assert all(c['prefill']==4096 and c['decode']>=127 for c in coverage.values()), coverage
    assert all(c['decode']<=129 for c in coverage.values()), 'Unexpected execution/replay count'
    pieces=partition(selected)
    seconds={k:sum(e['end']-e['start'] for e in pieces if e['phase']==k) for k in ('prefill','decode')}
    flops=sum(prefill_work(e['rows'],n) for e in selected if e['phase']=='prefill')
    traffic=sum(decode_work(e['rows'],n) for e in selected if e['phase']=='decode')
    evidence=root/f'phase-accounting-b{n}.json'
    evidence.write_text(json.dumps(dict(ready=ready,request_mapping=reqs,coverage=coverage,
        phases=selected,disjoint_timeline=pieces,seconds=seconds,prefill_flops=flops,decode_dram_bytes=traffic,
        basis=BASIS,window=dict(client_start_min=lo,client_end_max=hi),
        raw_events_sha256=sha(path)),indent=2)+'\n')
    common=dict(timing_scope='full_phase_wall_time',
        timing_method='Existing async completion/readback + synchronous final prefill sampling; union of intervals on shared monotonic clock, gaps charged to next phase; no added synchronization',
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
