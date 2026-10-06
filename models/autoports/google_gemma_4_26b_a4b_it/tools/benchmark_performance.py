# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the upstream vLLM CLI with deterministic request IDs for phase matching."""

import json
import os
import sys
import time
from pathlib import Path

args = sys.argv[1:]
name = args[args.index("--result-filename") + 1]
root = Path(args[args.index("--result-dir") + 1])
prefix = "gemma4-" + name.removesuffix(".json") + "-"
tokenizer = "/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
if not Path(tokenizer, "tokenizer.json").is_file():
    raise FileNotFoundError("Pinned native performance tokenizer is unavailable")
args += ["--tokenizer", tokenizer]
metadata = {"request_id_prefix": prefix, "command": args, "client_dispatch_before_ns": time.perf_counter_ns()}
(root / (name.removesuffix(".json") + "-request-map.json")).write_text(json.dumps(metadata, indent=2) + "\n")
python = "/home/container_app_user/tt-metal/python_env/bin/python"
os.execv(python, [python, "-m", "vllm.entrypoints.cli.main", "bench", "serve", *args, "--request-id-prefix", prefix])
