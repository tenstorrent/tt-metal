"""Record an already-running server's command, relevant env and source hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-prefix', required=True)
args = parser.parse_args()
matches = []
for proc in Path('/proc').glob('[0-9]*'):
    try:
        argv = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        if 'vllm.entrypoints.openai.api_server' in argv:
            matches.append((proc, argv))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
if len(matches) != 1:
    raise RuntimeError(f'Expected one API server, found {len(matches)}')
proc, argv = matches[0]
env = dict(x.split('=', 1) for x in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in x)
keys = ('TT_METAL_TRACE_ALLOC_TRACKING', 'TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE',
        'TT_METAL_DEVICE_PROFILER', 'TT_METAL_WATCHER', 'QWEN38_HOST_SAMPLING',
        'QWEN38_SERVING_EVENTS', 'QWEN38_SERVING_LAYERS', 'QWEN38_PRECISION_CONFIG',
        'TT_MODEL_CLASS_OVERRIDES')
files = list(Path('models/autoports/qwen_qwen3_8_27b/tt').rglob('*.py'))
files += list(Path('models/autoports/qwen_qwen3_8_27b/tt/kernels').glob('*'))
files += list(Path('bringup/dependencies/vllm-tt-plugin/src/vllm_tt_plugin').rglob('*.py'))
record = dict(pid=int(proc.name), argv=argv, environment={k: env.get(k) for k in keys},
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(files) if p.is_file()})
prefix = Path(args.output_prefix)
prefix.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
prefix.with_suffix('.txt').write_text(shlex.join(argv) + '\n')
print(json.dumps({'pid': int(proc.name), 'manifest': str(prefix.with_suffix('.json'))}))
