"""Host-only closure checks over measured precision-stage artifacts."""
import hashlib
import json
import statistics
from pathlib import Path
from models.autoports.qwen_qwen3_8_flash_next.tests.datatype_policy_check import check

ROOT = Path('models/autoports/qwen_qwen3_8_flash_next')
OUT = ROOT / 'doc/datatype_sweep'
LENGTHS = [31, 32, 33, 127, 128, 129, 203, 511, 512, 513, 2053]


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    policy = read('selected_precision_config.json')
    rows = read('sweep_results.json')
    passing = [r for r in rows if r['status'] == 'pass']
    winner = max(passing, key=lambda r: r['traced_teacher_forcing_tps'])
    assert winner['config_id'] == policy['config_id']
    measured = 0
    for row in rows:
        for key in ('config_id', 'dtype_policy', 'compute_fidelity_policy', 'top1', 'top5', 'top100',
                    'ttft_ms', 'traced_teacher_forcing_tps', 'measurement_regime', 'command', 'hardware', 'mesh', 'status'):
            assert key in row, (row['config_id'], key)
        if row['traced_teacher_forcing_tps'] is None:
            continue
        raw = json.loads(Path(row['trace_evidence']).read_text())
        reps = raw['teacher_forcing_repetitions']
        assert len(reps) == 3 and not raw['args']['final']
        for rep in reps:
            counters = rep['counter_delta']
            assert counters['decode_replays'] == 99
            assert counters.get('captures', 0) == counters.get('releases', 0) == 0
        gate = all(m['top1'] >= .90 and m['top5'] >= .98 and m['top100'] == 1
                   for group in [raw['prefill'], *[r['metrics'] for r in reps]] for m in group)
        assert gate == (row['status'] == 'pass')
        assert row['traced_teacher_forcing_tps'] == statistics.median(r['metrics'][0]['decode_t/s/u'] for r in reps)
        measured += 1
    qualifications = []
    for name, tracking in [('selected_default.json', '0'), ('selected_tracked.json', '1')]:
        report = read(name)
        assert report['status'] == 'pass' and report['args']['final']
        assert report['environment']['TT_METAL_TRACE_ALLOC_TRACKING'] == tracking
        if tracking == '1':
            assert report['environment']['TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE'] == '0'
        for path, digest in report['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        assert [r['length'] for r in report['nonaligned']] == LENGTHS
        assert all(r['traced'] == r['eager'] for r in report['nonaligned'])
        assert len(report['qualitative']) == 6
        qualifications.append(check(report, policy))
    normal, tracked = read('selected_default.json'), read('selected_tracked.json')
    assert normal['teacher_forcing_repetitions'][0]['predicted_tokens'] == tracked['teacher_forcing_repetitions'][0]['predicted_tokens']
    for batch in (3, 32):
        report = read(f'selected_batch{batch}.json')
        assert report['args']['batch'] == batch and report['args']['layers'] == '0,1,3'
        for key in ('repeated_logits_exact', 'batch_lane_logits_exact', 'external_prefill_continuation_exact',
                    'scheduler_activity_retains_slots_and_pages', 'host_compatibility_equal', 'sample_parameter_update_and_seed_repeat'):
            assert report[key], (batch, key)
        for path, digest in report['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    contract = json.loads((ROOT / 'doc/context_contract.json').read_text())['datatype_sweep_validation']
    assert contract['supported_context'] == contract['hf_advertised_context'] == policy['max_context'] == 262144
    assert contract['capability_reduction'] is None and contract['kv_cache_dtype'] == policy['kv_cache_dtype']
    assert contract['actual_free_ordinary_dram_bytes'] > 0
    for name in ('top1_perf_pareto.png', 'top5_perf_pareto.png'):
        assert (OUT / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert len(read('group_comparisons.json')) == 11
    result = dict(status='technical_checks_pass', selected_config_id=policy['config_id'],
                  measured_full_model_configs=measured, qualifications=qualifications,
                  nonaligned_lengths=LENGTHS, reduced_batch_checks=[3, 32],
                  independent_review='Required separately; this artifact does not substitute for stage-review clean-pass.')
    (OUT / 'technical_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
