"""Summarize the selected default run without mixing benchmark regimes."""
import json
import statistics
from pathlib import Path

OUT = Path('models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep')


def main():
    selected = json.loads((OUT / 'selected_precision_config.json').read_text())
    report = json.loads((OUT / 'selected_default.json').read_text())
    selection = json.loads((OUT / 'selection.json').read_text())
    baseline = json.loads((OUT / 'baseline.json').read_text())
    assert report['status'] == 'pass' and report['args']['config'] is None
    assert report['runtime']['policy'] == selected
    assert report['environment']['TT_METAL_TRACE_ALLOC_TRACKING'] == '0'
    runs = report['post_selection_token_out']
    assert len(runs) == 3
    for run in runs:
        assert run['perf']['prompt_len'] == run['perf']['gen_len'] == 128
        counters = run['counter_delta']
        assert counters['decode_replays'] == 127
        assert counters.get('token_refreshes', 0) == 0
        assert counters['token_readbacks'] == counters['history_readbacks'] == 1
        assert counters.get('captures', 0) == counters.get('releases', 0) == 0
    token_out = dict(
        config_id=selected['config_id'], artifact='selected_default.json',
        construction='build_generator with precision_config=None',
        measurement_regime='warmed token-out; B1 input128 output128; retained decode trace; no per-token host feedback/readback',
        input_tokens=report['post_selection_token_out_input_tokens'],
        ttft_ms=1000 * statistics.median(r['perf']['ttft_s'] for r in runs),
        decode_tps=statistics.median(r['perf']['decode_tps'] for r in runs),
        repetitions=[r['perf'] for r in runs],
        runtime_counters=[r['counter_delta'] for r in runs],
        command=report['command'], hardware=report['hardware'], mesh=report['mesh'],
        baseline_artifact='baseline.json', baseline_repetitions=baseline['primary_128_128'],
        future_reporting_rule='Use this token-out result for later serving comparisons; teacher forcing is only the precision selection metric.')
    token_out['baseline_ttft_ms'] = 1000 * statistics.median(r['ttft_s'] for r in baseline['primary_128_128'])
    token_out['baseline_decode_tps'] = statistics.median(r['decode_tps'] for r in baseline['primary_128_128'])
    token_out['decode_speedup_percent'] = 100 * (token_out['decode_tps'] / token_out['baseline_decode_tps'] - 1)
    tf = [r['metrics'][0] for r in report['teacher_forcing_repetitions']]
    summary = dict(
        config_id=selected['config_id'], selection_candidate=selection['selected_candidate'],
        reproduced_default_teacher_forcing=dict(
            artifact='selected_default.json', measurement_regime=report['measurement_regime'],
            top1=min(r['top1'] for r in tf), top5=min(r['top5'] for r in tf),
            top100=min(r['top100'] for r in tf),
            ttft_ms=statistics.median(r['ttft_ms'] for r in tf),
            decode_tps=statistics.median(r['decode_t/s/u'] for r in tf)),
        post_selection_token_out=token_out,
        comparison_caution='Different prompts, lengths and host feedback/readback regimes; do not compare teacher-forcing throughput directly with token-out throughput.')
    for name, value in [('post_selection_token_out.json', token_out), ('performance_summary.json', summary)]:
        (OUT / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(summary['reproduced_default_teacher_forcing'], indent=2))
    print(json.dumps({key:token_out[key] for key in ('config_id', 'ttft_ms', 'decode_tps')}, indent=2))


if __name__ == '__main__':
    main()
