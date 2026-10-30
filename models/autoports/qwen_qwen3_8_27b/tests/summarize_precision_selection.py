# SPDX-License-Identifier: Apache-2.0
"""Reconcile selected all-layer token-out/memory evidence; no hardware access."""
import argparse
import collections
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('--packet-template',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
doc=root/'doc';out=doc/'datatype_sweep';benchroot=out/'selected'
config=json.loads((out/'selected_precision_config.json').read_text())
benchmarks={b:json.loads((benchroot/f'benchmark_b{b}_perf.json').read_text()) for b in (1,8,16,32)}
measurements=[];allocations=[]
for batch,data in benchmarks.items():
    assert data['scope']=='all64layers' and data['source']=='autoregressive'
    assert data['precision_config']==config
    assert data['tracking'] in ('0', 0, False)
    assert not data['diagnostics'].get('TT_METAL_WATCHER') and not data['diagnostics'].get('TT_METAL_DEVICE_PROFILER')
    groups=collections.defaultdict(list)
    for row in data['records']:
        assert row['source']=='autoregressive' and row['batch']==batch and row['concurrency']==batch
        assert row['output_tokens']==128 and row['decode_trace']
        assert row['steady_state_counters']['captures']==0 and row['inner_loop_host_refreshes']==0
        groups[tuple(row['input_tokens'])].append(row)
    for lengths,rows in groups.items():
        measurements.append(dict(batch=batch,concurrency=batch,input_tokens=lengths[0],output_tokens=128,
            repetitions=len(rows),source='autoregressive',ttft_ms=statistics.median(r['ttft_ms'] for r in rows),
            decode_tps=statistics.median(r['decode_tps_per_user'] for r in rows),
            evidence=f'datatype_sweep/selected/benchmark_b{batch}_perf.json',
            measurement_regime='warmed token-out; no inner-loop host refresh/readback; EOS disabled'))
    cache=data['cache_inventory']
    assert len({row['layer'] for row in cache})==64
    kv=sum(r['payload_bytes_per_device'] for r in cache if r['field'] in ('key','value'))
    linear=sum(r['payload_bytes_per_device'] for r in cache if r['field'] in ('conv','recurrent'))
    actual_kv_dtypes={r['dtype'] for r in cache if r['field'] in ('key','value')}
    expected='DataType.'+config['kv_cache_dtype'].upper()
    assert actual_kv_dtypes=={expected},(actual_kv_dtypes,expected)
    model=data['weight_inventory']['total_payload_bytes_per_device']
    request=data['request_buffers']['total_payload_bytes_per_device']
    before=data['memory_before_preparation']['allocated_bytes_per_device']
    after=data['memory_after_preparation']['allocated_bytes_per_device']
    workspace=72089600  # Fixed BF8 prefill and BF16 native-row workspace; L1 CCL is separate.
    audit=data['allocation_audit'];trace=data['trace_allocation']
    assert audit['event_allocator_reconciliation_exact']
    capacity=before+data['memory_before_preparation']['free_bytes_per_device']
    assert audit['peak_allocated_bytes_per_device_upper_bound']<=capacity
    allocations.append(dict(batch=batch,context=data['context'],pool_tokens=batch*data['context'],
        model_tensor_payload_bytes_per_device=model,request_tensor_payload_bytes_per_device=request,
        kv_payload_bytes_per_device=kv,linear_state_payload_bytes_per_device=linear,
        kv_bytes_per_token_per_device=kv/(batch*((data['context']+255)//256)*256),
        shared_CCL_DRAM_bytes_per_device=workspace,allocator_before_preparation_bytes_per_device=before,
        allocator_after_preparation_bytes_per_device=after,unitemized_setup_bytes_per_device=before-model-request-kv-linear-workspace,
        warmup_program_cache_trace_increment_bytes_per_device=after-before,
        capacity_bytes_per_device=capacity,allocation_audit=audit,trace_allocation=trace,
        evidence=f'datatype_sweep/selected/benchmark_b{batch}_perf.json'))
families=collections.defaultdict(int)
for tensor in benchmarks[1]['weight_inventory']['tensors']:
    name=tensor['name'].split('.')[2] if tensor['name'].startswith('layers.') else tensor['name']
    families[name]+=tensor['payload_bytes_per_device']
weights=[]
for name,size in families.items():
    row=dict(name=name,bytes_per_device=size,copies=1)
    if name in ('gate','up','gateup'):
        row.update(copies=2,copy_reason='Measured inherited separate prefill and packed decode representations; each named row counts only its own bytes, sum once. See optimized_multichip_decoder/prefill_projection_selection.json.')
    elif name in ('in_norm','post_norm','norm_shards'):
        row.update(copies=2,copy_reason='Inherited replicated/sharded norm representations for phase-specific residual contracts; each named row counts only its own bytes.')
    weights.append(row)
ledger=dict(status='allocator_reconciled',weights=weights,allocations=allocations,
    kv_bytes_per_token_per_device=allocations[0]['kv_bytes_per_token_per_device'],
    linear_state_bytes_per_request_per_device=allocations[0]['linear_state_payload_bytes_per_device'],
    pool_tokens=allocations[-1]['pool_tokens'],
    reserves_bytes_per_device=dict(activation_and_CCL_estimate=12884901888,trace_estimate=1073741824),
    measured_transient_peak_bytes_per_device=max(max(x['allocation_audit']['additional_peak_bytes_by_graph_device'].values()) for x in allocations),
    measured_trace_bytes_per_device=max(x['trace_allocation']['released_trace_bytes_per_device'] for x in allocations),
    reconciliation='Unitemized setup is scalar/helper/semaphore allocations and bank padding. Warmup delta includes program cache, traces and persistent buffers. Shared workspace counted once. Peak audit covers4096 requests; longer prefill reuses the same bounded4096 chunk buffers. No claim of a device-time roofline.',
    weight_scope='Local physical tensor payload; copies describes representations and is not an extra byte multiplier.')
assert ledger['pool_tokens']>=32*32768
native=benchmarks[1]['context_validation']
assert native['exact_max_prefill']==262144 and native['final_decode_input_position']==262143 and native['finite']
capacity=dict(status='selected_precision_native_context_and_batch_pairs_verified',supported_context=262144,
    capability_reduction=None,selected_precision_config=config,ledger=ledger,context_validation=native,
    scope='B1 native262144; batches1/8/16/32 at recorded contexts, including B32 agent32768 workload. Not every batch/native-context Cartesian combination.')
(out/'memory_capacity_plan.json').write_text(json.dumps(capacity,indent=2)+'\n')
(out/'performance.json').write_text(json.dumps(dict(selected_config=config['config_id'],measurements=measurements,
    workload=dict(target_concurrency=32,working_set_tokens=32768,typical_context_tokens=4096,source='planning estimate')),indent=2)+'\n')
contract=json.loads((doc/'context_contract.json').read_text())
contract.update(supported_context=262144,largest_tested_prefill_context=262144,largest_tested_decode_context=262144,
    capability_reduction=None,status='datatype_sweep_selected_native_context_verified',
    comparison_scope='Selected precision all64-layer native capacity/finite checks; AIME24 readiness accuracy is separately measured at203 input tokens.',
    largest_tested_context_scope='Selected precision all64-layer B1 prefill262144, nonaligned262141 plus3 generated tokens, explicit decode input262143; finite and capacity verified, no native-context HF accuracy claim.')
contract.setdefault('stage_evidence',{})['08-datatype-sweep']=dict(selected_config=config['config_id'],
    kv_dtype=config['kv_cache_dtype'],supported_context=262144,largest_tested_prefill_context=262144,
    largest_tested_decode_context=262144,kv_bytes_per_token_per_device=ledger['kv_bytes_per_token_per_device'],
    nonaligned_prefill=262141,capability_reduction=None,
    evidence='datatype_sweep/selected/benchmark_b1_perf.json:context_validation',
    memory_evidence='datatype_sweep/memory_capacity_plan.json',
    construction='tt.precision loads selected_precision_config.json by default; max_context must equal checkpoint262144')
contract.setdefault('precision_candidate_evidence',{})[config['config_id']]=dict(
    kv_dtype=config['kv_cache_dtype'],supported_context=262144,capability_reduction=None,
    kv_bytes_per_token_per_device=ledger['kv_bytes_per_token_per_device'],
    native_kv_payload_bytes_per_device=allocations[0]['kv_payload_bytes_per_device'],
    quality_status='passed',memory_status='passed',evidence='datatype_sweep/memory_capacity_plan.json')
(doc/'context_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
agentic=json.loads((doc/'agentic_serving.json').read_text())
prior={k:v for k,v in agentic.items() if k!='previous_stages'}
if prior['stage'] != '08-datatype-sweep':
    agentic.setdefault('previous_stages',{})[prior['stage']]=prior
agentic.update(stage='08-datatype-sweep',status='selected_precision_full_model_verified',ledger=ledger,
    tested_batch_sizes=[1,8,16,32],performance_status='measured_for_recorded_batches',
    performance_evidence='datatype_sweep/performance.json',memory_evidence=['datatype_sweep/memory_capacity_plan.json'],
    batch_evidence={str(b):[f'datatype_sweep/selected/benchmark_b{b}_perf.json'] for b in benchmarks})
(doc/'agentic_serving.json').write_text(json.dumps(agentic,indent=2)+'\n')
template=Path(a.packet_template);packet=json.loads(template.read_text())
headline=next(r for r in measurements if r['batch']==1 and r['input_tokens']==4096)
assert headline['repetitions']==3
packet['selected_mesh']={'shape':[1,4],'devices':4,'hardware':'Blackhole p300c'}
packet['performance']=dict(source='autoregressive',ttft_ms=headline['ttft_ms'],decode_tps=headline['decode_tps'],
    prefill_device_us=None,decode_device_us=None,prefill_flops_pct=None,decode_dram_pct=None,
    evidence=headline['evidence']+'; all64layers,3 warmed4096/128/B1/one-request repeats; normal selected-config construction')
packet['accuracy']=[]
for phase in ('prefill','teacher_forcing'):
    evidence=out/'selected_validation'/f'{phase}_accuracy.json'
    rows=json.loads(evidence.read_text())
    for metric in ('top1','top5','top100'):
        packet['accuracy'].append(dict(name=metric,value=min(r[metric] for r in rows),unit='fraction',phase=phase,
            completed_samples=len(rows),dataset_samples=None,scope='subset',
            evidence=str(evidence.relative_to(root))+';100 generated positions on one AIME24 chat problem'))
packet['missing_reasons']={'device_metrics':'No matching all64-model device-time window/roofline measurement; host timings and reduced-layer profiles are not substituted.','accuracy.dataset_samples':'Reference contains one AIME24 problem; total source-dataset count is not recorded in this reference artifact.'}
packet['notes']=['Selected '+config['config_id']+'. Headline is post-selection autoregressive token-out; teacher forcing is only the separate sweep ranking metric.',
    'Accuracy is token agreement on one problem, not AIME task accuracy. Native262144 capability preserved.']
target=template.with_name(template.name.replace('.template',''));target.write_text(json.dumps(packet,indent=2)+'\n')
print('Reconciled selected precision, context, agent memory and packet:',target)
