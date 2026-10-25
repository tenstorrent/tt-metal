#!/usr/bin/env python3
"""Invoke upstream vLLM 0.26 CLI with an auditable per-profile request ID prefix."""
import json
import os
from pathlib import Path
import sys

repo=Path(__file__).resolve().parents[4]
args=sys.argv[1:]
name=args[args.index('--result-filename')+1].removesuffix('.json')
prefix='qwen38-'+name+'-'
result_dir=Path(args[args.index('--result-dir')+1])
argv=[str(repo/'.cache/benchmark-perf/bin/python'),'-m','vllm.entrypoints.cli.main','bench','serve',
      '--tokenizer',str(repo/'.cache/qwen38-full'),'--request-id-prefix',prefix,*args]
(result_dir/(name+'-client.json')).write_text(json.dumps(dict(argv=argv,request_id_prefix=prefix),indent=2)+'\n')
env=dict(os.environ,PYTHONPATH='',VLLM_PLUGINS='')
os.execve(argv[0],argv,env)
