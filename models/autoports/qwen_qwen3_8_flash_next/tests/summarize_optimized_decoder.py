# SPDX-License-Identifier: Apache-2.0
"""Derive compact tables from retained measurements; never infer gate success."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'doc/optimized_decoder'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='integrated1')
    args = parser.parse_args()
    trials = []
    for path in sorted(EVIDENCE.glob('*_result.json')):
        row = json.loads(path.read_text())
        trial = dict(tag=path.name.removesuffix('_result.json'), returncode=row['returncode'],
                     prefill_ms=row.get('prefill_ms'), decode_ms=row.get('decode_ms'), artifact=path.name)
        for mode in ('prefill', 'decode'):
            comparison = row.get('comparison', {}).get(mode, {})
            trial[mode+'_pcc_vs_control'] = comparison.get('pcc')
            trial[mode+'_exact_vs_control'] = comparison.get('exact')
        trials.append(trial)
    if trials:
        with (EVIDENCE/'trials.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=trials[0]); writer.writeheader(); writer.writerows(trials)
    groups = defaultdict(list)
    logs = []
    for path in sorted(EVIDENCE.glob('*_'+args.suffix+'.log')):
        count = 0
        for line in path.read_text().splitlines():
            if not line.startswith('{'): continue
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            if 'pcc' not in row or 'layer_idx' not in row: continue
            key = (row['layer_idx'], row['batch'], row['mode'])
            groups[key].append(row); count += 1
        if count: logs.append(dict(log=path.name, pcc_records=count))
    summary = [dict(layer=k[0], batch=k[1], mode=k[2], records=len(rows),
                    min_pcc=min(r['pcc'] for r in rows),
                    thresholds=sorted({r['threshold'] for r in rows}),
                    failures=sum(r['pcc'] < r['threshold'] for r in rows),
                    lengths=sorted({r['length'] for r in rows}),
                    max_position=max(r['position'] for r in rows)) for k,rows in sorted(groups.items())]
    gates=[]
    for path in sorted(EVIDENCE.glob('*_'+args.suffix+'_command.json')):
        record=json.loads(path.read_text())
        gates.append(dict(artifact=path.name, returncode=record.get('returncode'),
                          runtime_sha256=record.get('runtime_sha256'), elapsed_seconds=record.get('elapsed_seconds')))
    (EVIDENCE/('correctness_summary_'+args.suffix+'.json')).write_text(
        json.dumps(dict(gates=gates, source_logs=logs, measurements=summary), indent=2)+'\n')
    print(f'{len(trials)} candidate results; {len(summary)} correctness groups; {len(gates)} gate records')

if __name__ == '__main__': main()
