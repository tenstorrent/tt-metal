#!/usr/bin/env python3
"""Resume performance after a rejected warmup, preserving completed accuracy.

Use the installed runner unchanged except for reuse of its completed accuracy
outputs and currently owned/validated baseline server. Preserve failed records
and enforce the original deadline, never a fresh stage allowance.
"""
import hashlib
import inspect
import json
from pathlib import Path
import time
import benchmark_stage.run as upstream


root=Path(__file__).resolve().parents[1]
base=root/'doc/benchmark'
out=base/'run'
old_summary=json.loads((out/'summary.json').read_text())
assert old_summary['status']=='failed' and 'input_lens' in old_summary['error']
archive=out/'rejected-tokenization'
archive.mkdir(exist_ok=True)
for name in ['summary.json','REPORT.md','run_config.json',
             'perf-b32-warmup.json','perf-b32-warmup.log','perf-b32-warmup-client.json']:
    if (archive/name).exists():
        assert (archive/name).read_bytes()==(out/name).read_bytes(), 'Refusing to replace preserved failure evidence'
    else:
        (archive/name).write_bytes((out/name).read_bytes())
config=json.loads((out/'run_config.json').read_text())
setup=json.loads((base/'setup.json').read_text())
config['budget_seconds']=int(setup['stage_deadline_unix']-time.time()-60)
assert config['budget_seconds']>0
source=inspect.getsource(upstream)
assert source.count('output.mkdir(parents=True, exist_ok=False)')==1
source=source.replace('output.mkdir(parents=True, exist_ok=False)',
                      'output.mkdir(parents=True, exist_ok=True)')
before="baseline_server = select_server(accuracy_concurrency)"
after="""baseline_server = json.loads((output / f'perf-b{accuracy_concurrency}-server.json').read_text())
        validate_server(baseline_server, accuracy_concurrency, config['model'], config['base_url'], output=output)"""
assert source.count(before)==1
source=source.replace(before,after)
start=source.index('        for batch in batches:')
end=source.index('        for concurrency in concurrencies:',start)
replacement="""        for task in config['tasks']:
            result = json.loads((output / task / 'results.json').read_text())
            meta = result['benchmark_stage']
            assert meta['responses'] == meta['expected_samples']
            assert meta['generation_overrides'] == config['generation'][task]
            assert meta['concurrency'] == accuracy_concurrency
            assert meta['subset_sha256'] == json.loads((output / 'manifest.json').read_text())['manifest_sha256']
            summary['accuracy'][task] = meta
        summary['continuation'] = {'accuracy_reused': True, 'prior_summary': 'rejected-tokenization/summary.json',
            'reason': 'Correct exact input-token transport after rejected warmup; no accuracy rerun'}
"""
source=source[:start]+replacement+source[end:]
receipt=dict(upstream_file=upstream.__file__,
    upstream_sha256=hashlib.sha256(inspect.getsource(upstream).encode()).hexdigest(),
    previous_elapsed_seconds=old_summary['elapsed_seconds'],
    remaining_budget_seconds=config['budget_seconds'],
    deadline_unix=setup['stage_deadline_unix'],
    modifications=['Reuse existing validated B32 server (same PID; pre-accuracy collector check retained)',
                   'Reuse existing output directory after preserving failed runner/warmup artifacts',
                   'Reuse all completed accuracy evidence unchanged',
                   'Reduce budget to original remaining time minus reporting reserve'])
(out/'continuation.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'continuation-runner.py.txt').write_text(source)
namespace=dict(upstream.__dict__,__name__='benchmark_performance_continuation')
exec(compile(source,upstream.__file__,'exec'),namespace)
config_path=base/'continuation_config.json'
config_path.write_text(json.dumps(config,indent=2)+'\n')
namespace['run'](config_path=config_path,output=out)
