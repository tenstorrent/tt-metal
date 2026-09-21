# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy-friendly Qwen3.5 TP prefill capture: ONE signposted measured prefill.

Everything expensive and uninteresting -- weight load, program compile, the warmup
prefill -- runs BEFORE ``tracy.signpost("start")``, so the ops between the ``start``
and ``stop`` signposts are exactly one chunk-outer prefill of the full model. Filter
``ops_perf_results_*.csv`` to that window to get a per-op breakdown of TTFT alone,
instead of a whole-session capture polluted by compile and 100 decode steps (which
also overflows the profiler DRAM buffers and silently drops markers).

Usage:
    export HF_MODEL=... MESH_DEVICE=P150x8
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
      python -m tracy -p -r -v -m pytest \
      models/demos/blackhole/qwen36/tests/perf/perf_prefill_tracy.py -sv

Then keep the CSV rows between the "start" and "stop" signposts.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

try:
    from tracy import signpost
except ImportError:  # profiler build absent -- harness still runs, just unmarked

    def signpost(*_a, **_k):
        pass


SEQ_LEN = int(os.environ.get("QWEN_PERF_ISL", "4096"))
BLOCK_SIZE = 64


@torch.no_grad()
@parametrize_mesh_tp()
def test_perf_prefill_tracy(mesh_device, reset_seeds, ensure_gc):
    """One signposted chunk-outer prefill at QWEN_PERF_ISL (default 4096)."""
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=SEQ_LEN + 512)
    vocab = model.args.vocab_size
    torch.manual_seed(0)
    tokens = torch.randint(0, vocab, (1, SEQ_LEN), dtype=torch.long)

    num_blocks = (SEQ_LEN // BLOCK_SIZE) + 8
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_shape = (num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    # Warmup OUTSIDE the signposts: compiles every program so the measured pass is steady state.
    model.reset_tp()
    warm = model.prefill_traced_chunked(tokens, page_table, actual_len=SEQ_LEN)
    ttnn.deallocate(warm)
    ttnn.synchronize_device(mesh_device)

    model.reset_tp()
    logger.info(f"measured prefill: ISL={SEQ_LEN}, {mesh_device.get_num_devices()} devices")
    signpost("start")
    out = model.prefill_traced_chunked(tokens, page_table, actual_len=SEQ_LEN)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")

    assert out is not None
    ttnn.deallocate(out)
