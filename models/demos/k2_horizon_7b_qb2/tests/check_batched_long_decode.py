# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model batched decode beyond the stock bound: flash vs chunked-fallback logits.

Mixed-length real-text prompts are prefilled once per case into one shuffled paged cache. Then one
traced decode step runs per kernel from the same inputs: B32 at capacity 8192 (stock bound 4096),
and B8 at capacity 131072 (stock bound 16384). Both kernels write identical K/V for the new position
before attending, so the comparison is on the same cache. Reports logit PCC, max-abs difference and
top-1 agreement per row.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import build_generator
from ..tt.multichip_decoder import MultichipDecoder
from .benchmark_long_context import MODEL_DIR, prompt_tokens


def pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["32:8192:4000", "8:131072:65000"], help="batch:capacity:base_len")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    results = []
    try:
        gen = build_generator(MODEL_DIR, mesh)
        for case in a.cases:
            batch, capacity, base = map(int, case.split(":"))
            cache, table = gen.model.allocate_cache(batch_size=batch, capacity=capacity)
            torch.manual_seed(37)
            table = torch.randperm(table.numel()).reshape_as(table).int()
            lengths = [base + 37 * i for i in range(batch)]
            assert max(lengths) < capacity
            text = prompt_tokens(gen.tokenizer, max(lengths) + 97 * batch)
            tokens = torch.tensor([text[97 * i : 97 * i + max(lengths)] for i in range(batch)])
            start = time.perf_counter()
            first = gen.prefill_forward(tokens, page_table=table, kv_cache=cache, prompt_lens=lengths)
            prefill_seconds = time.perf_counter() - start
            logits = {}
            for kernel in ("flash", "chunked"):
                MultichipDecoder.accurate_decode_kernel = kernel
                gen._release_traces(drop_state=True, next_batch=batch)
                logits[kernel] = gen.decode_forward(
                    first.reshape(batch, 1).int(),
                    torch.tensor(lengths),
                    page_table=table,
                    kv_cache=cache,
                    sampling_mode="host",
                ).float()
            MultichipDecoder.accurate_decode_kernel = "flash"
            f, c = logits["flash"], logits["chunked"]
            row = {
                "batch": batch,
                "capacity": capacity,
                "lengths": [min(lengths), max(lengths)],
                "prefill_seconds": prefill_seconds,
                "logit_pcc": pcc(f, c),
                "per_row_min_pcc": min(pcc(f[i], c[i]) for i in range(batch)),
                "max_abs_diff": float((f - c).abs().max()),
                "top1_agree": int((f.argmax(-1) == c.argmax(-1)).sum()),
                "rows": batch,
            }
            print("BATCHED_DECODE", json.dumps(row), flush=True)
            results.append(row)
            a.output.write_text(json.dumps(results, indent=1) + "\n")
            gen._release_traces(drop_state=True)
            del cache
    finally:
        a.output.write_text(json.dumps(results, indent=1) + "\n")
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
