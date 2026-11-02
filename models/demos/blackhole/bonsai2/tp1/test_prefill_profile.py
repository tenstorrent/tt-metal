# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op device profile of ONE timed traced_4k-style prefill (Agent C, TTFT breakdown).

Replicates text_demo's traced path for a PROMPT_LEN-token prompt: eager warmup prefill (compiles every
program), chunk-trace capture, one warm prefill_traced_chunked, a profiler flush, then the profiled
prefill_traced_chunked between signposts: one traced 2048-row chunk replay plus the eager tail.
Token values don't change the device work, so the prompt is random ids.
    TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -v --no-web-server --op-support-count 12000 \
      -m pytest models/demos/blackhole/bonsai2/tp1/test_prefill_profile.py -s
"""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PROMPT_LEN = int(os.environ.get("PF_PROMPT_LEN", "2642"))  # the traced_4k prompt
BLOCK_SIZE = 64
NUM_BLOCKS = 72  # text_demo's budget for traced_4k: 4096 + 100 tokens, rounded to 8 blocks
CHUNK = 2048


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
def test_prefill_profile(mesh_device):
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if profiling:
        from tracy import signpost
    else:
        signpost = lambda *_: None  # noqa: E731

    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE, enable_mtp=False
    )
    args = model.args
    model.allocate_kv_caches([NUM_BLOCKS, args.n_kv_heads, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    tokens = torch.randint(1000, 100000, (1, PROMPT_LEN), dtype=torch.long)
    bucket = ((PROMPT_LEN + CHUNK - 1) // CHUNK) * CHUNK
    padded = torch.cat([tokens, tokens[:, -1:].expand(1, bucket - PROMPT_LEN)], dim=1)

    model.prefill(tokens)  # text_demo's _warmup_prefill: compiles the eager programs
    ttnn.synchronize_device(mesh_device)
    model.reset_state()
    model.capture_prefill_trace_chunked(mesh_device, page_table, chunk_size=CHUNK, warmup_masked_buckets=False)

    def run():
        t0 = time.time()
        logits = model.prefill_traced_chunked(padded, page_table, actual_len=PROMPT_LEN)
        ttnn.to_torch(logits)
        return time.time() - t0

    print(f"PFPROF warm prefill {run():.3f} s", flush=True)
    ttnn.synchronize_device(mesh_device)
    if profiling:
        ttnn.ReadDeviceProfiler(mesh_device)
    signpost("prefill_start")
    print(f"PFPROF profiled prefill {run():.3f} s", flush=True)
    ttnn.synchronize_device(mesh_device)
    if profiling:
        ttnn.ReadDeviceProfiler(mesh_device)
    signpost("prefill_end")
