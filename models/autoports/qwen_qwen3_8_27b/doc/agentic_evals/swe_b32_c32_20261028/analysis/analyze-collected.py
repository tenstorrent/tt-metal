"""Reuse stage12 Harbor/isolation analyzers, adding whole-turn prefix and runtime checks."""
import argparse,collections,hashlib,importlib.util,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('control',type=Path);p.add_argument('output',type=Path);p.add_argument('--allow-workflow-snapshot',action='store_true');a=p.parse_args()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
harbor_path=Path('/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/analyze_agentic_harbor.py')
isolation_path=Path('/home/mvasiljev/agentic-research-data/experiment-26/goal12-publish/metal-swe-b32-bounded/models/autoports/qwen_qwen3_8_27b/tests/audit_swe_isolation.py')
harbor=module('harbor_evidence',harbor_path);isolation=module('isolation_evidence',isolation_path)
engine_path=harbor_path.with_name('analyze_agentic_engine.py');engine=module('engine_evidence',engine_path)
workflow=next(a.directory.glob('workflow_logs*'));exports=next(a.directory.glob('final_audits*'));audit_zip=next(exports.glob('recovered-audits-*.zip'),None)
stable_export=audit_zip is not None
if not stable_export:
 if not a.allow_workflow_snapshot:raise ValueError('Stable owned export unavailable; explicit snapshot mode required')
 audit_zip=a.output.with_name(a.output.stem+'-workflow-snapshot.zip')
 files=list(workflow.rglob('*qwen38*.jsonl'))
 if len(files)!=2:raise ValueError('Snapshot audit files missing or ambiguous')
 with zipfile.ZipFile(audit_zip,'w',compression=zipfile.ZIP_STORED) as out:
  records=[]
  for f in files:
   name=str(f.relative_to(workflow));data=f.read_bytes();out.writestr(name,data);records.append({'archive_path':name,'sha256':hashlib.sha256(data).hexdigest(),'stable_two_reads':False})
  out.writestr('audit-recovery-manifest.json',json.dumps({'image':None,'source_owner':None,'files':records}))
combined=a.output.with_name(a.output.stem+'-input.zip');combined.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(audit_zip) as z:
 manifest=json.loads(z.read('audit-recovery-manifest.json'))
 for r in manifest['files']:assert hashlib.sha256(z.read(r['archive_path'])).hexdigest()==r['sha256'] and (r['stable_two_reads'] or not stable_export)
 with zipfile.ZipFile(combined,'w',compression=zipfile.ZIP_STORED) as out:
  names=set()
  for f in workflow.rglob('*'):
   if f.is_file() and not (f.name.startswith(('qwen38-http-','qwen38-goal12-')) and f.suffix=='.jsonl'):name=str(f.relative_to(workflow));out.write(f,name);names.add(name)
  for name in z.namelist():
   if name.endswith('.jsonl') and name not in names:out.writestr(name,z.read(name))
 result=harbor.analyze(combined)
 http=[json.loads(line) for name in z.namelist() if 'qwen38-http-' in name for line in z.read(name).splitlines() if line.strip()]
 events=[json.loads(line) for name in z.namelist() if 'qwen38-goal12-' in name for line in z.read(name).splitlines() if line.strip()]
control=json.loads(a.control.read_text());expected=control['task_names']['swe_bench_verified'];result['selection_expected']=expected;result['isolation']=isolation.audit(workflow,expected)
rawspec=json.loads(next(workflow.rglob('runtime_model_spec_*.json')).read_text())['runtime_model_spec'];dev=rawspec['device_model_spec'];vargs=dev['vllm_args']
result['resolved_model_spec']={k:rawspec.get(k) for k in ('impl','hf_model_repo','device_type','docker_image','tt_metal_commit','vllm_commit')}
result['resolved_model_spec']['device']={k:dev.get(k) for k in ('max_concurrency','max_context')}
result['resolved_model_spec']['vllm_args']={k:vargs.get(k) for k in ('max_num_seqs','max_model_len','max_num_batched_tokens','long_prefill_token_threshold','block_size','mamba_block_size','prefix_cache_retention_interval','revision','tokenizer_revision','enable_prefix_caching')}
result['resolved_model_spec']['policy_env']={k:v for k,v in rawspec['env_vars'].items() if k.startswith('QWEN38_') or k.startswith('TT_METAL_TRACE_ALLOC')}
harbor_cfg=json.loads(next(workflow.rglob('*harbor_config.json')).read_text());result['resolved_harbor_concurrency']=harbor_cfg['n_concurrent_trials'];result['resolved_harbor_attempts']=harbor_cfg['n_attempts']
result['control_manifest_matches_harbor']=harbor_cfg['n_concurrent_trials']==control['max_concurrent'] and harbor_cfg['n_attempts']==1

prepared=[r for r in events if r['event']=='prepared'];assert len(prepared)==1;prep=prepared[0];block=prep['allocation']['scheduler_block_size'];serving=[r for r in events if r['event']!='shutdown_release']
result['runtime']={'prepared_count':len(prepared),'batch':prep['batch'],'throw_exception_on_fallback':prep.get('throw_exception_on_fallback'),'normal_shutdown_events':sum(r['event']=='shutdown_release' for r in events),'physical_prefill_chunk_size':prep['physical_prefill_chunk_size'],'allocation':prep['allocation'],'preparation_memory':prep['preparation_memory'],'capture_values':sorted({r['counters']['captures'] for r in serving if 'counters' in r}),'max_retirement_count':max(r.get('retirement_count',0) for r in serving),'max_host_logits_reads':max(r.get('host_logits_reads',0) for r in serving),'state_pool_last':next((r.get('state_pool') for r in reversed(serving) if r.get('state_pool')),None)}
epoch=engine.Epoch(str(audit_zip),prepared[0]['pid'],0)
for index,event in enumerate(events,1):epoch.add(event,index)
result['engine_observations']=epoch.summary();result['engine_observation_limit']='Host event intervals are observation coverage, not measured device utilization or per-operator execution time.'
chunks=[r for r in events if r['event']=='prefill_chunk']
logical=sum(r['length']*len(r['slots']) for r in chunks)
physical=sum((1 if r['length']==1 else (r['length']+31)//32*32)*prep['batch'] for r in chunks)
result['prefill_batching_observations']={'chunks':len(chunks),'useful_rows_per_chunk_histogram':dict(collections.Counter(len(r['slots']) for r in chunks)),'logical_row_token_units':logical,'padded_tensor_row_token_units':physical,'logical_over_padded_tensor_ratio':logical/physical if physical else None,'scope':'Physical512 buckets and fixedresidentB32 tensor shape proxy; not measured FLOPs or device utilization. Single-token bucket treated separately.'}
end=[r for r in http if r['event']=='request_end'];by_session=collections.defaultdict(list)
for r in end:by_session[r.get('session_sha256')].append(r)
for ev in result['evals']:
 names=[r['task_name'] for r in ev['tasks']];ev['exact_expected_task_set']=set(names)==set(expected) and len(names)==len(expected)
 for task in ev['tasks']:
  raw=workflow/task['evidence'][0].split('::',1)[1]
  d=json.loads(raw.read_text());traj=raw.parent/'agent/mini-swe-agent.trajectory.json';saved=json.loads(traj.read_text()) if traj.exists() else {};info=saved.get('info',{});kwargs=info.get('config',{}).get('model',{}).get('model_kwargs',{});sid=kwargs.get('extra_headers',{}).get('X-Session-ID');rows=by_session.get(hashlib.sha256(sid.encode()).hexdigest(),[]) if sid else []
  rows=sorted(rows,key=lambda r:r.get('started_unix',0))
  task['first_request_identity']={k:rows[0].get(k) for k in ('messages_sha256','prompt_ids_sha256','prompt_tokens','requested_chat_template_policy')} if rows else None
  task['response_identity_sequence']=[{k:r.get(k) for k in ('messages_sha256','prompt_ids_sha256','generated_ids_sha256','prompt_tokens','completion_tokens','finish_reasons')} for r in rows]
  comparable=[r for r in rows if r.get('previous_turn_lcp_tokens') is not None and r.get('previous_generated_tokens',0)>0]
  reusable=lambda r:min(r['previous_turn_lcp_tokens'],max(0,r['previous_prompt_plus_completion_tokens']-1),max(0,r['prompt_tokens']-1))//block*block
  task['agent_exit_status']=info.get('exit_status');task['actual_model_kwargs']=kwargs;task['stored_session_header']=sid;task['all_returning_turns_full_prefix_preserved']=all(r['previous_turn_lcp_tokens']==r['previous_prompt_plus_completion_tokens'] for r in comparable) if comparable else None
  task['whole_turn_prefix']={'comparable_turns':len(comparable),'full_prefix_preserved_turns':sum(r['previous_turn_lcp_tokens']==r['previous_prompt_plus_completion_tokens'] for r in comparable),'lost_history_tokens':sum(r['previous_prompt_plus_completion_tokens']-r['previous_turn_lcp_tokens'] for r in comparable),'expected_quantized_cache_tokens':sum(min(r['previous_turn_lcp_tokens'],r['prompt_tokens']-1)//block*block for r in comparable),'conservative_reusable_cache_tokens':sum(reusable(r) for r in comparable),'last_sampled_token_boundary_allowance':'A final sampled token may not yet be committed to KV; retain raw quantized-LCP bound separately.','actual_cache_tokens_on_comparable_turns':sum(r.get('cached_tokens',0) for r in comparable),'unexplained_loss_beyond_quantization':sum(max(0,reusable(r)-r.get('cached_tokens',0)) for r in comparable),'generated_prefix_missing_reasons':dict(collections.Counter(r.get('generated_prefix_missing_reason') for r in rows if r.get('generated_prefix_missing_reason')))}
  assistant=[x for x in saved.get('messages',[]) if x.get('role')=='assistant'];task['saved_response_count']=len(assistant);saved_usage=[(x.get('extra',{}).get('response',{}).get('usage',{}) or {}).get('completion_tokens') for x in assistant];task['saved_response_output_tokens']=sum(v for v in saved_usage if isinstance(v,(int,float)) and not isinstance(v,bool));task['saved_response_missing_output_usage']=sum(not isinstance(v,(int,float)) or isinstance(v,bool) for v in saved_usage);task['saved_reasoning_messages']=sum(bool(x.get('reasoning_content')) for x in assistant)
  task['retrieval_candidates']=[c['function']['arguments'] for x in assistant for c in (x.get('tool_calls') or []) if any(k in c.get('function',{}).get('arguments','') for k in ('curl ','wget ','git clone','git show','git log','pip install','urllib','requests.get'))]
  task['raw_agent_result']=d.get('agent_result');task['raw_trial_result']=str(raw)
result['stable_audit_export']={'qualified':stable_export,'basis':'owned stable export' if stable_export else 'workflow artifact snapshot only; final owned export withheld','manifest_image':manifest['image'],'source_owner':manifest['source_owner'],'all_file_hashes_verified':True,'files':manifest['files']};result['analysis_source_hashes']={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (harbor_path,isolation_path,engine_path,Path(__file__),a.control)}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'evals':[{'eval':e['eval'],'tasks':len(e['tasks']),'solved':sum((t['verifier']['rewards'] or {}).get('reward',0) for t in e['tasks']),'exact_task_set':e['exact_expected_task_set'],'harness':e['proof'],'http':e['http_suite_start_window']} for e in result['evals']],'isolation':result['isolation']['status'],'runtime':{k:v for k,v in result['runtime'].items() if k not in ('allocation','preparation_memory')}},indent=2))
