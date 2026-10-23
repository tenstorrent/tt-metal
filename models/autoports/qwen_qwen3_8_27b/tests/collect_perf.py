# SPDX-License-Identifier: Apache-2.0
"""CPU-only report extraction. Run in the environment containing tt-perf-report."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--decode-only', action='store_true')
    args = parser.parse_args()
    root = args.directory
    source = sorted(root.glob('reports/*/ops_perf_results*.csv'))[-1]
    copied = root / 'ops.csv'
    shutil.copyfile(source, copied)
    rows = list(csv.DictReader(copied.open()))
    with (root / '.logs/profile_log_device.csv').open() as f:
        header = next(f).strip()
    mhz = float(re.search(r'CHIP_FREQ\[MHz\]: (\d+)', header)[1])
    metrics = {'source': str(source), 'device_header': header, 'clock_mhz': mhz,
               'time_basis': 'Whole-layer firmware timestamp envelope, including all operations and gaps; device cycles / MHz = us.',
               'phases': {}}
    for phase in (('DECODE',) if args.decode_only else ('PREFILL', 'DECODE')):
        start = next(i for i, r in enumerate(rows) if r['OP TYPE'] == 'signpost' and r['OP CODE'] == f'PERF_{phase}')
        end = next(i for i, r in enumerate(rows) if i > start and r['OP TYPE'] == 'signpost' and r['OP CODE'] == f'PERF_{phase}_END')
        window = [r for r in rows[start + 1:end] if r['OP TYPE'] == 'tt_dnn_device']
        assert window and all(r['DEVICE FW START CYCLE'] and r['DEVICE FW END CYCLE'] for r in window)
        first = min(int(r['DEVICE FW START CYCLE']) for r in window)
        last = max(int(r['DEVICE FW END CYCLE']) for r in window)
        metrics['phases'][phase.lower()] = dict(device_us=(last-first)/mhz, first_cycle=first, last_cycle=last,
            device_ops=len(window), summed_kernel_us=sum(float(r['DEVICE KERNEL DURATION [ns]']) for r in window)/1000,
            trace_ids=sorted(set(r['METAL TRACE ID'] for r in window)),
            trace_replay_sessions=sorted(set(r['METAL TRACE REPLAY SESSION ID'] for r in window)))
        common = ['tt-perf-report', str(copied), '--start-signpost', f'PERF_{phase}',
                  '--end-signpost', f'PERF_{phase}_END', '--no-summary', '--no-advice', '--no-color']
        with (root / f'{phase.lower()}_perf_report.console.log').open('w') as f:
            subprocess.run(common + ['--csv', str(root / f'{phase.lower()}_perf_report.csv')], stdout=f, stderr=subprocess.STDOUT, check=True)
        with (root / f'{phase.lower()}_perf_report.txt').open('w') as f:
            subprocess.run(common, stdout=f, stderr=subprocess.STDOUT, check=True)
    (root / 'whole_layer_timing.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
