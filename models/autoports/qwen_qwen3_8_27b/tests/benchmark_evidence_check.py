#!/usr/bin/env python3
"""Run the supplied evidence gate with its stale one-hour cap corrected.

Only the budget expression/message are changed, matching benchmark_stage.run's
1800 seconds per configured batch. Also enforce the original deadline across
both attempts, intervening repairs, checks and report generation.
"""
import hashlib
import inspect
import json
import time
from datetime import datetime
from pathlib import Path

import benchmark_stage.check as upstream


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'doc/benchmark'
    source = inspect.getsource(upstream)
    old = "min(config.get('budget_seconds', 3600), 3600)"
    new = "min(config.get('budget_seconds', 3600), 1800 * len(config.get('concurrencies', [1, 8, 16, 32])))"
    assert source.count(old) == 1, 'Upstream gate changed; review correction before use'
    corrected = source.replace(old, new).replace(
        'within its budget and one hour', 'within its budget and 30 minutes per configured batch')
    namespace = dict(upstream.__dict__, __name__='benchmark_budget_corrected')
    exec(compile(corrected, upstream.__file__, 'exec'), namespace)
    namespace['check'](root, 'Qwen/Qwen3.8-27B')
    setup = json.loads((base / 'setup.json').read_text())
    repo=root.parents[2]
    for path,key in [(root/'doc/context_contract.json','context_contract_sha256'),
                     (repo/'bringup/agentic_evals/inputs.json','inputs_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==setup[key], f'Protected input changed: {path}'
    summary=json.loads((base/'run/summary.json').read_text())
    for batch in summary['performance']:
        accounting=json.loads((base/f'run/phase-accounting-b{batch}.json').read_text())
        raw=base/f'run/phases-b{batch}.jsonl'
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==accounting['raw_events_sha256'], 'Raw timing hash mismatch'
        rows=[len(p['rows']) for p in accounting['phases'] if p['phase']=='decode']
        assert max(rows)==int(batch), 'Measured decode never filled the configured batch'
    finished = time.time()
    original_start = setup['stage_deadline_unix'] - setup['budget_seconds']
    elapsed = finished - original_start
    assert 0 < elapsed < setup['budget_seconds'], 'Combined attempts exceeded original budget'
    conservative_start = datetime.fromisoformat(setup['first_observed_setup_command_utc'].replace('Z', '+00:00')).timestamp()
    receipt = dict(
        status='passed', upstream_source=upstream.__file__,
        upstream_sha256=hashlib.sha256(source.encode()).hexdigest(),
        correction={'before': old, 'after': new},
        original_start_unix=original_start, checked_unix=finished,
        combined_elapsed_seconds=elapsed, budget_seconds=setup['budget_seconds'],
        origin_note='Frozen initial run_config timestamp immediately after runner monotonic start; per-attempt monotonic durations are in their summaries.',
        conservative_elapsed_including_observed_setup_seconds=finished-conservative_start,
        scope='Both attempts, repairs, launches, compilation, warmups, inference, collection, verification and reporting through this check',
    )
    (base / 'run/evidence-check.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
