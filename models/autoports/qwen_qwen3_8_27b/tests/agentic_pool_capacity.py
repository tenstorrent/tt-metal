#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unscored distinct-prefix ten-agent pool experiment; coordinator owns queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[4]
REL=Path('models/autoports/qwen_qwen3_8_27b');TESTS=REPO/REL/'tests'
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,TESTS/file);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
long=module('pool_native_stream','agentic_long_control_driver.py');probe=long.probe;write=probe.write

def common_prefix(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))

def prepare(path):
    from transformers import AutoTokenizer
    _,_,hf,original=long.reference()
    checkpoint=REPO/'.cache/qwen38-full'
    for name,expected in hf['checkpoint_metadata_sha256'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint metadata mismatch: '+name)
    tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    if hashlib.sha256(tok.get_chat_template().encode()).hexdigest()!=original['chat_template_sha256']:raise ValueError('Native chat template mismatch')
    rows=[]
    for i in range(10):
        facts={'agent':i,'port':7310+i,'project':'Cedar'+str(i)}
        def messages(n):
            return [{'role':'user','content':f'Agent unique identifier {i}. Retrieve the facts at the end.\n'+(f'Archive for agent {i}: the quiet garden has blue stones and a silver fence.\n'*n)+'\nAuthoritative facts: '+json.dumps(facts)+'. Return exactly these facts as a JSON object, without explanation.'}]
        lo,hi=0,10000
        while lo<hi:
            n=(lo+hi)//2
            ids=tok.apply_chat_template(messages(n),tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if len(ids)<60000:lo=n+1
            else:hi=n
        msg=messages(lo);ids=tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if not 59500<=len(ids)<=60500:raise ValueError('Prompt size outside pressure band')
        rows.append(dict(agent=i,messages=msg,input_tokens=ids,input_sha256=probe.sha(ids),expected=facts))
    max_shared=max(common_prefix(a['input_tokens'],b['input_tokens']) for i,a in enumerate(rows) for b in rows[i+1:])
    if max_shared>=2048:raise ValueError('Distinct-agent prompts share a complete cache block')
    value=dict(model=long.MODEL,revision=long.REVISION,chat_template_sha256=hashlib.sha256(tok.get_chat_template().encode()).hexdigest(),max_pairwise_common_prefix=max_shared,checkpoint_metadata_sha256=hf['checkpoint_metadata_sha256'],cases=rows)
    value['cases_sha256']=probe.sha(rows);write(path,value)

def cases(path):
    data=json.loads(path.read_text());rows=data['cases']
    if len(rows)!=10 or data['cases_sha256']!=probe.sha(rows) or data['revision']!=long.REVISION:raise ValueError('Invalid frozen cohort')
    for r in rows:
        if not 59500<=len(r['input_tokens'])<=60500 or r['input_sha256']!=probe.sha(r['input_tokens']):raise ValueError('Invalid native input proof')
    if max(common_prefix(a['input_tokens'],b['input_tokens']) for i,a in enumerate(rows) for b in rows[i+1:])>=2048:raise ValueError('Shared complete prefix block')
    return data

def compare_pair(a,b):
    return (a['cases_sha256']==b['cases_sha256'] and len(a['rounds'])==len(b['rounds'])==2 and
        all(len(x['responses'])==len(y['responses'])==10 and all(
            l['request_sha256']==r['request_sha256'] and l['prompt_token_ids']==r['prompt_token_ids'] and
            l['token_ids']==r['token_ids'] and l['status']==r['status']=='completed' and
            l['finish_reason']==r['finish_reason']=='stop'
            for l,r in zip(x['responses'],y['responses'])) for x,y in zip(a['rounds'],b['rounds'])))

def run_probe(args):
    cohort=cases(args.cases);identity=json.loads(args.server_config.read_text());argv=identity['server_command']
    if identity.get('layer_count')!=64 or identity.get('max_num_seqs')!=32 or identity.get('model_revision')!=long.REVISION:raise ValueError('Model identity mismatch')
    for flag,value in (('--block-size','2048'),('--mamba-block-size','2048'),('--num-gpu-blocks-override',str(args.pool_blocks))):
        if argv[argv.index(flag)+1]!=value:raise ValueError('Pool launch identity mismatch')
    result=dict(status='running',scored=False,cases_sha256=cohort['cases_sha256'],pool_blocks=args.pool_blocks,server_identity=identity,rounds=[],max_tokens=81920,request_timeout_seconds=12000)
    write(args.probe/'result.json',result)
    warm=long.stream_request(args.base_url,[{'role':'user','content':'Return only {"ready":true}.'}],args.probe/'warmup')
    if warm['status']!='completed' or warm['finish_reason']!='stop':raise ValueError('Warmup incomplete')
    before=probe.event_snapshot(args.events);result.update(events_before=before,metrics_before=probe.metric_snapshot(args.base_url))
    histories=[copy.deepcopy(r['messages']) for r in cohort['cases']]
    for turn in range(2):
        started=time.monotonic()
        with ThreadPoolExecutor(max_workers=10) as pool:
            rows=list(pool.map(lambda pair:long.stream_request(args.base_url,pair[1],args.probe/f'turn{turn}/agent{pair[0]}'),enumerate(histories)))
        checks=[]
        for i,(row,case) in enumerate(zip(rows,cohort['cases'])):
            try:semantic=json.loads(row['content'])==case['expected']
            except ValueError:semantic=False
            checks.append(dict(agent=i,semantic_exact=semantic,natural_eos=row['finish_reason']=='stop' and bool(row['token_ids']) and row['token_ids'][-1]==248046,
                input_exact=turn!=0 or row['prompt_token_ids']==case['input_tokens']))
        result['rounds'].append(dict(elapsed_seconds=time.monotonic()-started,responses=rows,checks=checks));write(args.probe/'result.json',result)
        if not all(r['status']=='completed' and r['finish_reason']=='stop' for r in rows):break
        for history,row in zip(histories,rows):history.extend([{'role':'assistant','content':row['content']},{'role':'user','content':'Return exactly the same authoritative facts as JSON again.'}])
    after=probe.event_snapshot(args.events);result.update(events_after=after,metrics_after=probe.metric_snapshot(args.base_url))
    for key,counter,path in [('trace_capture_delta','captures','counters'),('trace_retirement_delta','retirement_count',None),('host_logits_read_delta','host_logits_reads',None)]:
        result[key]=probe.counter_delta(before,after,counter,*([path] if path else []))
    prepared=[]
    for line in args.events.read_text().splitlines():
        try:prepared.append(json.loads(line))
        except ValueError:continue
    result['allocation_checkpoint']=next((r.get('allocation') for r in reversed(prepared) if r.get('event')=='prepared'),None)
    result['allocation_scope']='Prepared checkpoint, not peak; actual transient scratch only proven to fit if entire experiment completes.'
    result['complete_native_semantic']=len(result['rounds'])==2 and all(len(r['responses'])==10 and all(row.get('status')=='completed' and row.get('stream_done') is True and isinstance(row.get('usage'),dict) and row['usage'].get('completion_tokens')==len(row.get('token_ids',[])) for row in r['responses']) and all(all(c[k] for k in ('semantic_exact','natural_eos','input_exact')) for c in r['checks']) for r in result['rounds'])
    result['paired_exact']=None if args.reference is None else compare_pair(json.loads(args.reference.read_text()),result)
    result['actual_pool_metrics_match']=any('num_gpu_blocks="'+str(args.pool_blocks)+'"' in line for line in result['metrics_after'])
    result['confirmation_complete']=result['complete_native_semantic'] and result['actual_pool_metrics_match'] and result['allocation_checkpoint'] is not None and all(result[k]==0 for k in ('trace_capture_delta','trace_retirement_delta','host_logits_read_delta')) and (args.reference is None or result['paired_exact'] is True)
    result['status']='review_required' if result['confirmation_complete'] else 'incomplete'
    result['acceptance']='Coordinator reviews actual IDs, preemptions, misses and timing; capacity arithmetic alone is not a measured benefit.'
    write(args.probe/'result.json',result);return 0 if result['status']=='review_required' else 2

def lifecycle(args):
    cohort=cases(args.cases)
    if args.pool_blocks not in (353,385):raise ValueError('Explicit pool353 or385 required')
    if args.pool_blocks == 385 and args.reference is None:raise ValueError('Candidate pressure requires completed353 reference')
    if args.reference:
        baseline=json.loads(args.reference.read_text())
        if baseline.get('pool_blocks')!=353 or baseline.get('cases_sha256')!=cohort['cases_sha256'] or not baseline.get('complete_native_semantic') or not compare_pair(baseline,baseline):
            raise ValueError('Pressure reference is incomplete or not the same353 native cohort')
    if not 0<args.box_hours<=.8:raise ValueError('Box must be positive and <=.8 mesh-hour')
    root=args.output.resolve();root.relative_to(REPO)
    if root.exists() and any(root.iterdir()):raise ValueError('Fresh output required')
    root.mkdir(parents=True);observed=root/'b32/observed';observed.mkdir(parents=True)
    owned=module('pool_owned','benchmark_server.py');helpers=module('pool_queue','agentic_batch_driver.py')
    started=time.monotonic();deadline=started+args.box_hours*3600;state=root/'b32/owned_server.json';relay=None
    summary=dict(status='starting',scored=False,pool_blocks=args.pool_blocks,box_hours=args.box_hours,started_unix=time.time(),errors=[],cases_sha256=cases(args.cases)['cases_sha256'])
    summary['source_sha256']={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),TESTS/'benchmark_server.py',TESTS/'agentic_long_control_driver.py',TESTS/'agentic_batch_probe.py')}
    summary['cases_file_sha256']=hashlib.sha256(args.cases.read_bytes()).hexdigest()
    if args.reference:summary['reference_sha256']=hashlib.sha256(args.reference.read_bytes()).hexdigest()
    def interrupted(signum,frame):raise InterruptedError('Coordinator interruption '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        helpers.check_port_available()
        with (root/'relay.log').open('w') as log:relay=subprocess.Popen([sys.executable,str(TESTS/'benchmark_proxy.py'),str(root/'relay-owner.json')],stdout=log,stderr=subprocess.STDOUT)
        for _ in range(50):
            if relay.poll() is not None:raise RuntimeError('Owned relay exited')
            if (root/'relay-owner.json').exists():break
            time.sleep(.1)
        else:raise TimeoutError('Relay readiness')
        write(root/'summary.json',summary);startup=time.monotonic()
        helpers.checked_run([sys.executable,str(TESTS/'benchmark_server.py'),'--max-num-seqs','32','--scheduler-block-size','2048','--num-gpu-blocks-override',str(args.pool_blocks),'--shutdown-timeout','30','--output',str(observed/'server.json'),'--serving-events',str(observed/'events.jsonl')],deadline,observed/'launch.log')
        summary.update(startup_seconds=time.monotonic()-startup,status='probing');write(root/'summary.json',summary)
        command=[sys.executable,str(Path(__file__)),'--probe',str(root/'b32/probe'),'--server-config',str(observed/'server.json'),'--events',str(observed/'events.jsonl'),'--cases',str(args.cases.resolve()),'--pool-blocks',str(args.pool_blocks)]
        if args.reference:command+=['--reference',str(args.reference.resolve())]
        helpers.checked_run(command,deadline,observed/'probe.log');summary['status']='review_required'
    except BaseException as error:summary.update(status='incomplete',missing_reason=repr(error));summary['errors'].append(repr(error))
    finally:
        if state.exists():
            try:owned.helper('inspect',state);owned.helper('stop',state)
            except BaseException as error:summary['errors'].append('owned cleanup: '+repr(error))
        if relay is not None and relay.poll() is None:
            relay.terminate()
            try:relay.wait(timeout=10)
            except subprocess.TimeoutExpired:relay.kill();relay.wait()
        summary.update(elapsed_seconds=time.monotonic()-started,used_mesh_hours=(time.monotonic()-started)/3600,owned_server_state_remaining=state.exists(),relay_exit=None if relay is None else relay.returncode,
            cutoff_policy='No score. Preserve partial streams; progressing cutoffs require coordinator larger-box re-estimate, never lower release limits.')
        write(root/'summary.json',summary)
    return 0 if summary['status']=='review_required' and not summary['errors'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',type=Path);mode.add_argument('--output',type=Path);mode.add_argument('--probe',type=Path)
    p.add_argument('--cases',type=Path);p.add_argument('--pool-blocks',type=int,choices=[353,385]);p.add_argument('--reference',type=Path)
    p.add_argument('--server-config',type=Path);p.add_argument('--events',type=Path);p.add_argument('--base-url',default='http://127.0.0.1:8000');p.add_argument('--box-hours',type=float,default=.8)
    a=p.parse_args();raise SystemExit(prepare(a.prepare) if a.prepare else run_probe(a) if a.probe else lifecycle(a))
