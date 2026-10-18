# SPDX-License-Identifier: Apache-2.0
"""CPU-only reference page warmup; never import TTNN or access TT devices."""

import ast
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState

if __name__ == "__main__":
    torch.set_num_threads(8)
    source = Path(__file__).with_name("ple_long_context.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "MappedCheckpointTable")
    namespace = dict(torch=torch, os=os, time=time, json=json)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<established CPU reference class>", "exec"), namespace)
    checkpoint = Path(
        os.environ.get(
            "QWEN38_CHECKPOINT",
            "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540",
        )
    )
    table = namespace["MappedCheckpointTable"](CheckpointState(checkpoint))
    result = table.warm_pages()
    result["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    print(json.dumps(result), flush=True)
