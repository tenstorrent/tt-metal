# SPDX-License-Identifier: Apache-2.0
"""Build compact stage manifests from completed runs; never launches hardware."""
import argparse, hashlib, json
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'doc/optimized_decoder'
def read(name):return json.loads((D/name).read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
p=argparse.ArgumentParser();p.add_argument('--packet-template',type=Path);p.add_argument('--packet-output',type=Path);p.add_argument('--review-status',default='pending',choices=['pending','clean-pass']);p.add_argument('--commit');a=p.parse_args()
required=[f'{prefix}_l{layer}.json' for layer in (0,3) for prefix in ('pytest_contract','watcher','context262144','context262142','final_accepted')]
required += [f'workload_b{batch}_l{layer}.json' for batch in (8,16,32) for layer in (0,3)]
required += ['ragged_decode.json']
manifest={}
selected=read("selected_policy.json")
for name in required:
 r=read(name)
 expected=['decode_pcc','changed_position_pcc'] if name=='ragged_decode.json' else ['prefill_pcc']
 if not name.startswith('context262144') and name!='ragged_decode.json':expected+=['decode_pcc','changed_input_decode_pcc']
 if name.startswith(('watcher','workload','final_accepted')):expected+=['workload_final_decode_pcc']
 if name.startswith('pytest_contract'):
  expected+=['resume_pcc']
  if r['layer_type']=='full_attention':expected+=['remap_pcc']
 for key in expected:assert r.get(key) is not None and r[key]>=.995,(name,key)
 if not name.startswith('final_accepted'):assert r['runtime_fallback_audit'] is True,name
 if name!='ragged_decode.json':
  assert r['runtime_class']=='OptimizedDecoder',name
  assert r['policy']==selected[r['layer_type']],name
  for filename,digest in r['provenance'].items():
   assert hashlib.sha256((B/filename).read_bytes()).hexdigest()==digest,(name,filename,'stale source')
 for key in ('prefill_pcc','decode_pcc','changed_input_decode_pcc','workload_final_decode_pcc','resume_pcc','remap_pcc','changed_position_pcc'):
  if r.get(key) is not None:assert r[key]>=.995,(name,key,r[key])
 if 'deterministic' in r:assert r['deterministic'],name
 if r.get('prefill_deterministic') is not None:assert r['prefill_deterministic'],name
 if 'runtime_class' in r:assert r['runtime_class']=='OptimizedDecoder',name
 manifest[name]={k:r[k] for k in ('layer_type','batch','sequence','prefill_pcc','decode_pcc','workload_final_decode_pcc','resume_pcc','remap_pcc','deterministic','prefill_deterministic','runtime_fallback_audit','prefill_compared_tokens','compared_tokens') if k in r}
write(D/'validation_summary.json',dict(status='validated',review=a.review_status,results=manifest))
context=json.loads((B/'doc/context_contract.json').read_text())
context['stage_evidence']['03-optimized-decoder']=dict(supported_context=262144,capability_reduction=None,
 linear_state_layout='BF16 row-major [B,3,10240]; FP32 tiled [B,48,128,128]',linear_state_bytes_per_request=3207168,
 full_attention_kv_dtype='BFP8',full_attention_kv_bytes_per_token=2176,full_attention_kv_bytes_at_max_context=262144*2176,
 linear_prefill_evidence='optimized_decoder/context262144_l0.json',linear_nonaligned_prefill_and_final_decode_evidence='optimized_decoder/context262142_l0.json',
 full_prefill_evidence='optimized_decoder/context262144_l3.json',full_nonaligned_prefill_and_final_decode_evidence='optimized_decoder/context262142_l3.json',
 allocated_decoder_setup_bytes={kind:read(f'final_accepted_l{layer}.json')['memory']['after_weights']['allocated_bytes'] for layer,kind in ((0,'linear_attention'),(3,'full_attention'))},
 memory_effect='BFP4 weights, BFP8 KV; two gate/up representations for phase-specific matmuls, adding100270080 bytes versus packed-only. Decode MLP packs users in tile rows to avoid per-user padding. No context reduction.',
 scope='Single-layer capacity at B1; B8/B16/B32 measured at4096+128. Long fixtures repeat recorded HF activations and compare final128 prefill outputs using the whole prefix, plus two final-position decodes.')
write(B/'doc/context_contract.json',context)
agentic=json.loads((B/'doc/agentic_serving.json').read_text())
if agentic['stage']!='03-optimized-decoder':
 previous=agentic.pop('previous_stages',{});previous[agentic['stage']]=dict(agentic);agentic['previous_stages']=previous
agentic.update(stage='03-optimized-decoder',stage_scope='Single optimized decoder layer, no serving scheduler/full-model pool.',status='optimized_decoder_'+a.review_status,
 tested_batch_sizes=[1,8,16,32],batch_evidence={str(n):[f'optimized_decoder/'+(f'final_accepted_l{l}.json' if n==1 else f'workload_b{n}_l{l}.json') for l in (0,3)] for n in (1,8,16,32)},
 performance_evidence='optimized_decoder/performance.json',prefix_contract_evidence='optimized_decoder/README.md',
 memory_evidence=['optimized_decoder/final_accepted_l0.json:memory','optimized_decoder/final_accepted_l3.json:memory'],
 trace_allocation_evidence=['optimized_decoder/final_validation.log','optimized_decoder/validation_summary.json'],
 state_memory=dict(linear_bytes_per_request=3207168,full_kv_bytes_per_token=2176,trace_region_size=0,trace_allocation_mode="dynamic DRAM",actual_trace_bytes=None,
 trace_memory_missing_reason="Setup-only snapshots precede capture; zero is configured reservation, not actual trace allocation.",
 weight_representations='One attention/output/down copy; separate and packed gate/up copies for phase-specific latency.',scope='Single layer, not a full-model serving pool.'),
 performance_note='B1 device envelopes; B8/B16/B32 are4096+128 host-wall traced workloads. Host time is never substituted for device rooflines.')
write(B/'doc/agentic_serving.json',agentic)
if a.packet_template:
 assert a.packet_output
 packet=json.loads(a.packet_template.read_text());perf=read('performance.json')
 packet['selected_mesh']={'shape':[1,1],'physical_device_ids':[0],'device':'one p300c Blackhole ASIC'}
 keys=('prefill_device_us','decode_device_us','prefill_flops_pct','decode_dram_pct','prefill_useful_flops','decode_dram_bytes','peak_flops_per_s','peak_dram_bytes_per_s','peak_basis','source','evidence')
 packet['layer_types']={kind:{k:r[k] for k in keys} for kind,r in perf.items()}
 for row in packet['layer_types'].values():
  row['evidence']=[str(Path('models/autoports/qwen_qwen3_8_27b')/item) if item.startswith('tests/') else str(Path('models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder')/item) for item in row['evidence']]
 packet['accuracy']=[]
 for layer,kind in ((0,'linear_attention'),(3,'full_attention')):
  r=read(f'final_accepted_l{layer}.json')
  for key,name,phase,count,total,scope in [('prefill_pcc','HF prefill PCC','prefill',4096,4096,'full'),('decode_pcc','HF initial decode PCC','decode',1,1,'full'),('workload_final_decode_pcc','HF final128th decode PCC','decode',1,128,'subset')]:
   packet['accuracy'].append(dict(name=name,value=r[key],unit='PCC',phase=phase,layer_type=kind,completed_samples=count,dataset_samples=total,scope=scope,evidence=f'models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/final_accepted_l{layer}.json'))
 packet['notes']=['Single layer supplied-hidden-input workload, not full-model token generation. Accuracy sample counts are hidden-token positions.',
 'Whole-layer device denominators include all operations and gaps. Useful dense FLOPs; estimated compulsory physical DRAM bytes including BFP headers/padding, excluding intermediate spills. Not hardware counters.',
 'LoFi chip peak is a theoretical denominator; sensitive norm/attention/recurrence remain HiFi4/FP32. No percentage is clamped.',
 'Context262144 retained. Long tests repeat recorded HF activations; final128 prefill positions and final two decodes are compared.',
 'Independent stage-review: '+a.review_status]
 if a.commit:packet['notes'].append('Local tt-metal commit '+a.commit+'; never pushed.')
 packet['missing_reasons']={};write(a.packet_output,packet)
