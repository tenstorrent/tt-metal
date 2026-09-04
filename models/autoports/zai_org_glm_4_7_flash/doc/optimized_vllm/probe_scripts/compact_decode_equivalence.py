# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced-model equivalence check for the compact (kc-bucketed) decode traces.

Two generators over the same reduced 2-layer model (one dense + one MoE, the
same shape contract the full 47-layer stack has), one with
``moe_decode_compact=False`` (the shipped union decode) and one with it on.
For a range of live-row counts the two must produce the same decode logits, and
the compact one must have picked the bucket its live-row count implies.

    python .../probe_scripts/compact_decode_equivalence.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

import ttnn

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm import GLM47FlashForCausalLM  # noqa: E402

MODEL_DIR = REPO / "models" / "autoports" / "zai_org_glm_4_7_flash"
OUT = MODEL_DIR / "doc" / "optimized_vllm" / "compact_decode_equivalence.json"

MAX_SEQ_LEN = 4096
BLOCK_SIZE = 64
MAX_BATCH = 32
BLOCKS_PER_USER = math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)
NUM_BLOCKS = MAX_BATCH * BLOCKS_PER_USER


def build(dev, compact: bool):
    gen = build_generator(
        MODEL_DIR,
        dev,
        layer_indices=[0, 1],
        max_batch_size=MAX_BATCH,
        max_seq_len=MAX_SEQ_LEN,
        defer_cache_and_traces=True,
        moe_decode_compact=compact,
        progress=lambda m: None,
    )
    model = GLM47FlashForCausalLM(gen)
    kv = model.allocate_kv_cache(
        kv_cache_shape=(NUM_BLOCKS, 1, BLOCK_SIZE, gen.model.layers[0].kvpe_dim),
        dtype=torch.bfloat16,
        num_layers=len(gen.model.layers),
    )
    model.warmup_model_decode(
        kv_cache=kv, max_batch_size=MAX_BATCH, num_blocks=NUM_BLOCKS, can_sample_on_device=True, enable_trace=True
    )
    gen.reset()
    return gen, model, kv


def logits_for(gen, rows, toks, pos, pt):
    gen.refresh_page_table(pt)
    gen.set_decode_tokens(toks + [0] * (MAX_BATCH - len(toks)))
    gen.set_decode_positions(pos + [-1] * (MAX_BATCH - len(pos)))
    gen.replay_decode_trace()
    ttnn.synchronize_device(gen.mesh_device)
    out = ttnn.to_torch(gen._decode_logits).float()[0, 0, : len(toks), : gen.model.vocab_size]
    return out


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.equal(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    """One arm per process: two live generators on one device would make any
    disagreement ambiguous between "the compact path is wrong" and "two sets of
    live traces on one device interfere"."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "compact"
    compact = mode == "compact"
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    report = {"mode": mode}
    try:
        torch.manual_seed(0)
        pt = torch.arange(MAX_BATCH * BLOCKS_PER_USER, dtype=torch.int32).reshape(MAX_BATCH, BLOCKS_PER_USER)
        gen, model, kv = build(dev, compact=compact)
        print(f"{mode} buckets:", gen._decode_kc_buckets, flush=True)
        report["buckets"] = list(gen._decode_kc_buckets)
        report["kc_by_rows_sample"] = {str(r): gen.decode_kc_for_rows(r) for r in (0, 1, 2, 4, 5, 6, 8, 16, 32)}
        cases = {}
        # Every live-row count where the shipped selection table changes bucket,
        # plus the saturated case for each: 1(kc4) 2..4(kc16) 5..6(kc24) 7..8(kc32)
        # and 16/32 (union).
        for rows in (1, 2, 4, 5, 6, 8, 16, 32):
            toks = [(100 + 7 * i) % 100000 for i in range(rows)]
            pos = [64 + i for i in range(rows)]
            lg = logits_for(gen, rows, toks, pos, pt)
            cases[str(rows)] = {
                "kc_used": gen._active_kc,
                "argmax": lg.argmax(dim=-1).tolist(),
                "absmax": round(float(lg.abs().max()), 4),
                "mean": round(float(lg.mean()), 6),
                "row_checksum": [round(float(v), 4) for v in lg.float().sum(dim=-1).tolist()],
            }
            print(
                f"rows={rows}: kc={gen._active_kc} absmax={cases[str(rows)]['absmax']} "
                f"argmax0={cases[str(rows)]['argmax'][0]}",
                flush=True,
            )
        report["cases"] = cases
        report["kc_replays"] = {str(k): v for k, v in gen.kc_replays.items()}
        report["bucket_switches"] = gen.counters["decode_trace_bucket_switches"]
        out = OUT.with_name(f"compact_decode_equivalence_{mode}.json")
        out.write_text(json.dumps(report, indent=2) + "\n")
        print("WROTE", out, flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
