# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Do the kc buckets change the logits at all, on the real 47-layer model?

Every captured decode trace lives in the same generator and reads the same
persistent inputs, so this can force each bucket in turn over identical
token/position/page-table state and diff the logits directly -- no second
process, no second model, no trace interference.

Two things matter:

* Same live rows, different bucket: the compact path sums the expert outputs in
  union-score order while the union path sums them in expert-id order, and
  float addition is not associative, so the two can legitimately differ in the
  last bits even though they compute the same sum. This measures whether they
  actually do, and whether greedy argmax ever moves.
* Bucket-to-bucket among compact widths: the extra slots a wider bucket adds
  carry routing weight exactly zero, so those should be bit-identical.

    python .../probe_scripts/bucket_numerics.py
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
from models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm import (  # noqa: E402
    VLLM_PREFILL_BUCKETS,
    VLLM_PREFILL_CHUNK_SIZE,
    GLM47FlashForCausalLM,
)

MODEL_DIR = REPO / "models" / "autoports" / "zai_org_glm_4_7_flash"
OUT = MODEL_DIR / "doc" / "optimized_vllm" / "bucket_numerics.json"

MAX_SEQ_LEN = 202752
BLOCK_SIZE = 64
MAX_BATCH = 32
NUM_BLOCKS = 7362
BLOCKS_PER_USER = math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    try:
        gen = build_generator(
            MODEL_DIR,
            dev,
            max_batch_size=MAX_BATCH,
            max_seq_len=MAX_SEQ_LEN,
            defer_cache_and_traces=True,
            prefill_chunk_size=VLLM_PREFILL_CHUNK_SIZE,
            prefill_buckets=VLLM_PREFILL_BUCKETS,
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
        print("buckets:", gen._decode_kc_buckets, flush=True)

        pt = torch.zeros((MAX_BATCH, BLOCKS_PER_USER), dtype=torch.int32)
        for r in range(MAX_BATCH):
            pt[r] = torch.arange(r * 16, r * 16 + BLOCKS_PER_USER, dtype=torch.int32) % NUM_BLOCKS
        gen.refresh_page_table(pt)

        def logits_at(kc, live_rows):
            toks = [(1000 + 977 * r) % 150000 for r in range(live_rows)] + [0] * (MAX_BATCH - live_rows)
            pos = [128 + 13 * r for r in range(live_rows)] + [-1] * (MAX_BATCH - live_rows)
            gen.set_decode_tokens(toks)
            gen.set_decode_positions(pos)
            gen._decode_trace_id = gen._decode_traces[kc]
            gen._active_kc = kc
            gen._advance_host_positions()
            ttnn.execute_trace(gen.mesh_device, gen._decode_trace_id, cq_id=0, blocking=True)
            return ttnn.to_torch(gen._decode_logits).float()[0, 0, :live_rows, : gen.model.vocab_size].clone()

        def cmp(a, b):
            d = (a - b).abs()
            return {
                "bitwise_identical": bool(torch.equal(a, b)),
                "max_abs_diff": float(d.max()),
                "argmax_identical": bool(torch.equal(a.argmax(-1), b.argmax(-1))),
                "rows_with_argmax_change": int((a.argmax(-1) != b.argmax(-1)).sum()),
            }

        # One live-row count per shipped compact bucket, chosen as the row count
        # where that bucket's bound is exactly SATURATED (live_rows * top_k ==
        # kc), i.e. its zero-slack case, which is the one that most needs
        # checking. Derived from the shipped table rather than hard-coded, so a
        # bucket cannot be added without this probe covering it.
        from models.autoports.zai_org_glm_4_7_flash.tt.generator import COMPACT_KC_BUCKETS  # noqa: E402

        top_k = gen.model.layers[1].top_k
        bucket_for = {}
        for kc in COMPACT_KC_BUCKETS:
            rows = kc // top_k
            if rows >= 1 and gen.decode_kc_for_rows(rows) == kc:
                bucket_for[rows] = kc
        covered = set(bucket_for.values())
        missing = [kc for kc in COMPACT_KC_BUCKETS if kc not in covered]
        if missing:
            raise RuntimeError(f"no saturated row count selects buckets {missing}; equivalence would be unproven")
        print("saturated row -> bucket:", bucket_for, flush=True)
        res = {}
        for rows in sorted(bucket_for):
            base = logits_at(bucket_for[rows], rows)
            union = logits_at(None, rows)
            res[f"rows{rows}_compactbucket_vs_union"] = cmp(base, union)
            widest = max(COMPACT_KC_BUCKETS)
            if bucket_for[rows] != widest:
                wider = logits_at(widest, rows)
                res[f"rows{rows}_compactbucket_vs_compact{widest}"] = cmp(base, wider)
            # determinism of one bucket across repeated replays
            again = logits_at(bucket_for[rows], rows)
            res[f"rows{rows}_same_bucket_repeat"] = cmp(base, again)
            for k, v in res.items():
                pass
            print(rows, json.dumps({k: v for k, v in res.items() if k.startswith(f"rows{rows}")}), flush=True)

        OUT.write_text(
            json.dumps(
                {
                    "purpose": (
                        "Does the decode kc bucket change the logits on the real 47-layer model? Same generator, "
                        "same persistent inputs, trace forced per bucket."
                    ),
                    "buckets": [b if b is not None else "union" for b in gen._decode_kc_buckets],
                    "saturated_row_per_bucket": {str(r): kc for r, kc in bucket_for.items()},
                    "results": res,
                },
                indent=2,
            )
            + "\n"
        )
        print("WROTE", OUT, flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
