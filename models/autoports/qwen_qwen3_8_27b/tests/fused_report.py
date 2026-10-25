# SPDX-License-Identifier: Apache-2.0
"""Rebuild compact fused-layer performance evidence from complete device windows."""
import argparse
import csv
import json
from pathlib import Path

from .roofline import calculate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--linear-profile', default='history_slice_final')
    parser.add_argument('--conv-row-major', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'doc/fused_decoder'
    result = {'workload': dict(input_tokens=4096, output_tokens=128, batch=1, concurrency=1),
              'scope': 'Single decoder layer with supplied hidden activations; not token generation.',
              'layer_types': {}, 'supplemental_batches': {}, 'operation_audit': {}}
    for kind, label, baseline in [('linear_attention', args.linear_profile, 'linear_b1_complete'),
                                   ('full_attention', 'final_full', 'full_b1')]:
        path = root / 'tracy' / label
        phases = json.loads((path / 'whole_layer_timing.json').read_text())['phases']
        values = calculate(kind, 1, phases['prefill']['device_us'], phases['decode']['device_us'])
        if kind == 'linear_attention':
            # Two 48-column gate projections physically occupy 64 columns each.
            values['decode_dram_bytes'] += 2 * 16 * 5120 * 2
            if args.conv_row_major:
                values['decode_dram_bytes'] -= 2 * (32-3) * 10240 * 2
            values['decode_dram_pct'] = 100 * values['decode_dram_bytes'] / (
                values['peak_dram_bytes_per_s'] * values['decode_device_us'] / 1e6)
            values['traffic_assumptions'] += ' Includes physical padding of the a/b projection columns.'
            if args.conv_row_major:
                values['traffic_assumptions'] = values['traffic_assumptions'].replace('physically padded BF16 convolution state', 'three-row row-major BF16 convolution state')
        values['evidence'] = [f'tracy/{label}/whole_layer_timing.json', f'tracy/{label}/ops.csv', f'{label}.json']
        base = json.loads((root.parent / 'functional_decoder/tracy' / baseline / 'whole_layer_timing.json').read_text())['phases']
        values['before'] = {f'{phase}_device_us': base[phase]['device_us'] for phase in ('prefill', 'decode')}
        values['speedup'] = {phase: base[phase]['device_us']/phases[phase]['device_us'] for phase in ('prefill', 'decode')}
        result['layer_types'][kind] = values
        rows = list(csv.DictReader((path / 'ops.csv').open()))
        audit = {}
        for phase in ('prefill', 'decode'):
            start = next(i for i,r in enumerate(rows) if r['OP CODE'] == f'PERF_{phase.upper()}')
            end = next(i for i,r in enumerate(rows) if i > start and r['OP CODE'] == f'PERF_{phase.upper()}_END')
            counts = {}
            for row in rows[start+1:end]:
                if row['OP TYPE'] != 'tt_dnn_device':
                    continue
                item = counts.setdefault(row['OP CODE'], {'count': 0, 'kernel_us': 0})
                item['count'] += 1
                item['kernel_us'] += float(row['DEVICE KERNEL DURATION [ns]'])/1000
            audit[phase] = counts
        result['operation_audit'][kind] = audit
        layer = 0 if kind == 'linear_attention' else 3
        result['supplemental_batches'][kind] = {}
        for batch in (8,16,32):
            source = root / (f'history_slice_workload_b{batch}.json' if args.conv_row_major and layer == 0 else f'workload_layer{layer}_b{batch}.json')
            if source.exists():
                row = json.loads(source.read_text())
                result['supplemental_batches'][kind][str(batch)] = dict(
                    workload=row.get('performance_workload'),
                    final_step_host_us=row.get('workload_last_decode_host_us'),
                    warmed_trace_screen_host_us=row.get('screen_host_traced_us'),
                    source='host wall time around blocking trace execution; not device time', evidence=source.name)
    (root / 'performance.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['layer_types'], indent=2))


if __name__ == '__main__':
    main()
