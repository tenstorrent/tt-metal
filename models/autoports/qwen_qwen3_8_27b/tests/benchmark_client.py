#!/usr/bin/env python3
"""Invoke upstream vLLM 0.26 CLI with an auditable per-profile request ID prefix."""
import json
import hashlib
import os
from pathlib import Path
import sys

repo=Path(__file__).resolve().parents[4]
args=sys.argv[1:]
name=args[args.index('--result-filename')+1].removesuffix('.json')
prefix='qwen38-v2-'+name+'-'
result_dir=Path(args[args.index('--result-dir')+1])
argv=[str(repo/'.cache/benchmark-perf/bin/python'),str(Path(__file__).with_name('benchmark_perf_cli.py')),'bench','serve',
      '--tokenizer',str(repo/'.cache/qwen38-full'),'--request-id-prefix',prefix,*args]
(result_dir/(name+'-client.json')).write_text(json.dumps(dict(argv=argv,request_id_prefix=prefix,
    transport_adapter_sha256=hashlib.sha256(Path(__file__).with_name('benchmark_perf_cli.py').read_bytes()).hexdigest(),
    prompt_transport='Exact random token ID arrays; upstream seeded sampling and serving metrics retained'),indent=2)+'\n')
env=dict(os.environ,PYTHONPATH='',VLLM_PLUGINS='')
os.execve(argv[0],argv,env)
