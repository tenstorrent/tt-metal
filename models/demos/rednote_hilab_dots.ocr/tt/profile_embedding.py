# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr LM token-embedding leaf block.

Profiles :class:`TtEmbedding` (a single ttnn.embedding gather over a
DRAM-resident table, vocab=151936, hidden=1536) in isolation under metal
trace so the CSV reflects device-kernel time rather than host dispatch.
The embedding gathers one row per token id; a representative production
prompt length is used for the sequence shape.

This is a single gather op with no matmul, no reshape chain, and no
activation -- there is no alternative kernel to shard or fuse, so the block
is expected to be at-ceiling. The harness exists to capture the mandated
traced tracy CSV as evidence.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \
      models/demos/rednote_hilab_dots.ocr/tt/profile_embedding.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv
(and a cpp_device_perf_report.csv under generated/profiler/.logs/).
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("dots_tt_embedding_profile", os.path.join(_TT_DIR, "embedding.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtEmbedding = _mod.TtEmbedding

# Production-representative shape: a token-id sequence at the LM vocab size.
# (Reduced PCC golden is [1,128]; production OCR prompts run longer sequences,
# so 512 tokens is a representative tile through the same per-token gather.)
VOCAB = 151936
HIDDEN = 1536
BATCH = 1
SEQ_LEN = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32)
        emb = TtEmbedding(device=device, weight=weight)

        host_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), dtype=torch.int32)
        input_ids = ttnn.from_torch(
            host_ids,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernel into the program cache.
        for _ in range(3):
            out = emb(input_ids)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = emb(input_ids)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            # One profiled replay.
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            out = emb(input_ids)
            ttnn.synchronize_device(device)

        print("profile_embedding done; out shape", tuple(out.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
