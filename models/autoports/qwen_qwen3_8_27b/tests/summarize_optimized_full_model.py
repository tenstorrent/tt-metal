# SPDX-License-Identifier: Apache-2.0
"""CPU-only evidence reconciliation; never starts devices or benchmarks."""
import collections
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'doc'
out = root / 'optimized_full_model'
def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

measurements = []
allocations = []
benchmarks = {}
for batch in (1, 8, 16, 32):
    path = out / f'benchmark_b{batch}_perf.json'
    if not path.exists():
        path = out / f'benchmark_b{batch}_tracking_final.json'
    if not path.exists():
        path = out / f'benchmark_b{batch}.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    benchmarks[batch] = data
    groups = collections.defaultdict(list)
    for record in data['records']:
        groups[tuple(record['input_tokens'])].append(record)
    for lengths, records in groups.items():
        measurements.append(dict(batch=batch, concurrency=batch, input_tokens_per_user=lengths[0], output_tokens_per_user=128,
            repetitions=len(records), source='autoregressive', ttft_ms=statistics.median(r['ttft_ms'] for r in records),
            decode_tps_per_user=statistics.median(r['decode_tps_per_user'] for r in records),
            decode_ms_per_step=statistics.median(r['decode_seconds']/127*1000 for r in records),
            scope='all64layers; end-to-end token-out; warmed; EOS disabled', tracking=data['tracking'], evidence=path.name,
            deterministic_across_repeats=(len({r['tokens_sha256'] for r in records}) == 1 if len(records)>1 else None),
            steady_state_host_refreshes_zero=all(all(r['steady_state_counters'][k] == 0 for k in ('host_copies_total','token_refreshes','position_refreshes','page_table_refreshes','captures')) for r in records)))
    model = data['weight_inventory']['total_payload_bytes_per_device']
    request = data['request_buffers']['total_payload_bytes_per_device']
    kv = batch * data['context'] * 8704
    recurrent = batch * 38486016
    workspace = 72089600
    before = data['memory_before_preparation']['allocated_bytes_per_device']
    after = data['memory_after_preparation']['allocated_bytes_per_device']
    allocations.append(dict(batch=batch, context=data['context'], pool_tokens=batch*data['context'],
        model_tensor_payload_bytes_per_device=model, request_tensor_payload_bytes_per_device=request,
        kv_payload_bytes_per_device=kv, recurrent_state_payload_bytes_per_device=recurrent,
        shared_CCL_DRAM_bytes_per_device=workspace, allocator_before_preparation_bytes_per_device=before,
        allocator_after_preparation_bytes_per_device=after,
        unitemized_setup_bytes_per_device=before-model-request-kv-recurrent-workspace,
        warmup_program_cache_trace_increment_bytes_per_device=after-before,
        available_after_preparation_bytes_per_device=data['memory_after_preparation']['free_bytes_per_device'],
        allocation_audit=data.get('allocation_audit'),trace_allocation=data.get('trace_allocation'),evidence=path.name))

families = collections.defaultdict(int)
for tensor in benchmarks[1]['weight_inventory']['tensors']:
    name = tensor['name'].split('.')[2] if tensor['name'].startswith('layers.') else tensor['name']
    families[name] += tensor['payload_bytes_per_device']
weights = []
for name, size in families.items():
    row = dict(name=name, bytes_per_device=size, copies=1)
    if name in ('gate', 'up', 'gateup'):
        row.update(copies=2, copy_reason='Inherited measured stage05 separate prefill plus packed decode/one-row weights. Bytes in each row count only that named representation; sum rows once. See optimized_multichip_decoder/prefill_projection_selection.json.')
    elif name in ('in_norm', 'post_norm', 'norm_shards'):
        row.update(copies=2, copy_reason='Inherited full and hidden-sharded norms support measured replicated B1-7 and native-RS B8-32 residual contracts. Bytes in each row count that representation only. See optimized_multichip_decoder/residual_contract.md.')
    weights.append(row)
ledger = dict(status='allocator_reconciled_for_recorded_batches', weights=weights,
    weight_scope='Persistent model tensors, including untied embedding/head, RoPE and recurrence constants; physical tile payload, excluding bank padding. copies denotes representation multiplicity, not an extra byte multiplier.',
    kv_bytes_per_token_per_device=8704, linear_state_bytes_per_request_per_device=38486016,
    pool_tokens=allocations[-1]['pool_tokens'], allocations=allocations,
    reserves_bytes_per_device=dict(activation_and_CCL_estimate=12884901888, trace_estimate=1073741824),
    measured_transient_peak_bytes_per_device=max((max(a['allocation_audit']['additional_peak_bytes_by_graph_device'].values()) for a in allocations if a['allocation_audit']),default=None),
    measured_trace_bytes_per_device=max((a['trace_allocation']['released_trace_bytes_per_device'] for a in allocations if a['trace_allocation']),default=None),
    reconciliation='Unitemized setup is scalar/helper/semaphore allocations and allocator bank padding not enumerated by tensor inventory. Warmup delta combines program-cache, trace and persistent allocations; it is not an isolated trace measurement. Shared DRAM workspace is counted once. Shared AR/native-RS L1 buffers are outside this DRAM ledger.',
    missing_reasons=[])
if ledger['measured_transient_peak_bytes_per_device'] is None:
    ledger['missing_reasons'].append('Allocation-event peak diagnostic pending; reserves remain inherited planning estimates.')
if ledger['measured_trace_bytes_per_device'] is None:
    ledger['missing_reasons'].append('Trace-release allocator delta diagnostic pending.')
ledger['reserve_comparison'] = dict(
    activation_estimate_minus_measured_peak_bytes_per_device=(
        ledger['reserves_bytes_per_device']['activation_and_CCL_estimate']-ledger['measured_transient_peak_bytes_per_device']
        if ledger['measured_transient_peak_bytes_per_device'] is not None else None),
    trace_estimate_minus_measured_trace_bytes_per_device=(
        ledger['reserves_bytes_per_device']['trace_estimate']-ledger['measured_trace_bytes_per_device']
        if ledger['measured_trace_bytes_per_device'] is not None else None),
    basis='Measured request reset/prefill/decode/sampling allocation peaks are additional to prepared allocator baseline, which already includes persistent CCL and traces. Trace bytes are separately isolated by release. Planning reserves are compared, not added again to measured peaks. Preparation checkpoints for external scratch are a separate scope.')
if ledger['pool_tokens'] < 32*32768:
    ledger['pool_shortfall_reason'] = 'Target B32 run remains pending; this interim ledger records only completed allocations.'
external_path=out/'external_prepare_b32.json'
if external_path.exists():
    external=json.loads(external_path.read_text())
    ledger['external_preparation']=dict(status=external['status'],batch=external['batch'],context=external['context'],
        allocator_checkpoints=external.get('preparation_memory'),evidence=external_path.name,
        scope='Full-stack external-pool preparation capacity and preservation probe; checkpoints are not peak measurements.')
# Preserve the previous full-stack external-pool capacity evidence explicitly;
# the changed trace lifecycle has fresh reduced verification in this stage.
if 'external_preparation' not in ledger:
    prior=json.loads((root/'full_model/memory_capacity_plan.json').read_text())['ledger']['external_preparation']
    ledger['external_preparation']={**prior,'inherited':True,
        'evidence':'full_model/external_prepare_b32.json',
        'scope':'Inherited stage06 all64 external-pool capacity and preservation checkpoints; not a new stage07 measurement.'}
reduced_external=out/'external_prepare_reduced_b32.json'
if reduced_external.exists():
    current=json.loads(reduced_external.read_text())
    ledger['external_preparation_trace_revalidation']=dict(status=current['status'],batch=current['batch'],context=current['context'],
        evidence='optimized_full_model/external_prepare_reduced_b32.json',
        scope='Current reduced real layers0/3: external ownership, failure cleanup and new trace lifecycle; not full-stack capacity.')
complete_clean = all(b in benchmarks and benchmarks[b].get('tracking') in (False, 0, '0') and len(benchmarks[b]['records']) >= 3 for b in (1,8,16,32))
capacity = dict(status=('native_B1_preserved_recorded_batch_context_pairs_verified' if complete_clean else 'native_B1_verified_larger_batch_measurements_in_progress'), supported_context=262144, capability_reduction=None,
    capacity_bytes_per_device=34178731008, full_attention_layers=16, linear_attention_layers=48,
    kv_dtype='bfloat8_b', ledger=ledger, context_validation_inherited_from='full_model/benchmark_b1_perf.json', context_validation=benchmarks[1].get('context_validation') or json.loads((root/'full_model/benchmark_b1_perf.json').read_text()).get('context_validation'),
    scope='Native context is preserved. Batch/context pairs are explicit; no claim of every batch x native-context combination. Reserve estimates do not establish a hard physical limit.',
    evidence=['optimized_full_model/'+a['evidence'] for a in allocations])
write(out/'memory_capacity_plan.json', capacity)
# Accepted stage05 ordinary host replay medians; separate from device profiles.
# The rotary core-group correction preserves policy but changes previously ignored
# arithmetic, so this is an inherited estimate, not a strict new-kernel bound.
layer_us={1:(445.78,359.46),8:(777.06,631.97),16:(1251.27,751.57),32:(2000.69,989.93)}
comparisons=[]
for row in measurements:
    if row['input_tokens_per_user']!=4096:
        continue
    linear,full=layer_us[row['batch']]
    stack=(48*linear+16*full)/1000
    comparisons.append(dict(batch=row['batch'],linear_layer_us=linear,full_layer_us=full,
        estimated_stack_ms=stack,token_out_ms=row['decode_ms_per_step'],
        difference_ms=row['decode_ms_per_step']-stack,
        basis='48linear+16full; inherited stage05 host trace medians at4096+128; includes per-layer replay overhead and predates rotary core-group correction; estimate, not directly subtractable device attribution',
        source='../optimized_multichip_decoder/README.md',full_model_source=row['evidence']))
write(out/'layer_stack_comparison.json',dict(rows=comparisons,
    scope='Host-wall comparison only. Negative differences can reflect combined-trace dispatch amortization. Terminal packing and common sampler comparisons are separately measured; reduced profiler gives device attribution.'))
write(out/'performance.json', dict(rows=measurements, device_time=None, missing_reasons=['All-layer device profiling is prohibited in this stage; reduced profiling cannot populate full-model device time or rooflines.'], teacher_forcing_evidence='benchmark_b1_perf.json:teacher_forcing_control', diagnostic_teacher_forcing_evidence='teacher_forcing_performance.json'))
context = json.loads((root/'context_contract.json').read_text())
context['stage_evidence']['07-optimized-full-model'] = capacity
context['status'] = ('full_model_native_context_and_recorded_batch_pairs_verified' if complete_clean else 'full_model_native_context_verified_larger_batch_validation_in_progress')
context['largest_tested_context_scope'] = 'Full all64layer B1 native prefill262144, nonaligned262141 plus3tokens, and explicit decode input262143 passed finite/capacity checks. HF accuracy at native context was not measured in this stage; prior per-layer accuracy is separately scoped.'
write(root/'context_contract.json', context)
agentic = json.loads((root/'agentic_serving.json').read_text())
agentic.update(stage='07-optimized-full-model', memory_evidence=['optimized_full_model/memory_capacity_plan.json'],
    review_evidence=('optimized_full_model/stage_review.md' if (out/'stage_review.md').exists() else None),
    status=('optimized_full_model_measurements_reconciled' if complete_clean else 'implementation_in_progress'), ledger=ledger, tested_batch_sizes=list(benchmarks), batch_evidence={str(a['batch']):['optimized_full_model/'+a['evidence']] for a in allocations}, performance_evidence='optimized_full_model/performance.json', performance_status='measured_for_recorded_batches', performance_note='All-layer autoregressive token-out measurements only. Teacher-forcing and reduced profiler results are separate.')
write(root/'agentic_serving.json', agentic)
print(json.dumps(dict(performance=measurements, allocations=allocations), indent=2))
