# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Guard the *previous* stage's headline number against this stage's changes.

This stage touched ``model.ttnn_decode_forward`` (the ``moe_kc``/``logits_out``
plumbing) and made the decode logits a caller-owned persistent buffer, both of
which are on the ``max_batch_size=1`` full-model path even though the compact
MoE bucketing is not. ``doc/optimized_full_model/perf.json`` records batch-1
traced token-out decode at 23.013 ms/token; this re-measures exactly that,
without writing to any earlier stage's artifacts.

    python .../probe_scripts/full_model_batch1_regression.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import ttnn

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator  # noqa: E402

MODEL_DIR = REPO / "models" / "autoports" / "zai_org_glm_4_7_flash"
OUT = MODEL_DIR / "doc" / "optimized_vllm" / "full_model_batch1_regression.json"
ITERS = 64


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    try:
        gen = build_generator(MODEL_DIR, dev, progress=lambda m: None)  # defaults: max_batch_size=1
        assert gen.max_batch_size == 1
        assert gen._decode_kc_buckets == (), "batch 1 must keep the indexed MoE path and a single decode trace"
        gen.reset()
        prompt = gen.tokenizer.encode("Tenstorrent builds AI accelerators. " * 40, add_special_tokens=True)[:128]
        gen.prefill_and_sample(prompt, user_id=0, recapture=True)

        # model trace only
        ttnn.synchronize_device(dev)
        model_only = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            gen.replay_decode_trace()
            ttnn.synchronize_device(dev)
            model_only.append((time.perf_counter() - t0) * 1e3)

        # token-out (model trace + split sampling + the one word readback)
        ttnn.synchronize_device(dev)
        token_out = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            gen.decode_step_traced()
            gen.read_decode_tokens(1)
            token_out.append((time.perf_counter() - t0) * 1e3)

        payload = {
            "statistic_note": (
                "Both mean and median are reported because the baseline this guards is a MEAN: "
                "tests/test_full_model_perf.py's bench() divides total wall time by iterations. Comparing this "
                "probe's median against that mean is not like-for-like and would hide up to a few percent of "
                "regression; use the mean-to-mean row."
            ),
            "purpose": (
                "Batch-1 full-model traced decode after this stage's model.py/generator.py changes, "
                "against doc/optimized_full_model/perf.json's recorded 21.760 (model only) / 23.013 "
                "(token out) ms per token. Writes only into doc/optimized_vllm/."
            ),
            "batch": gen.max_batch_size,
            "iters": ITERS,
            "traced_model_only_ms_per_token_mean": round(statistics.fmean(model_only), 3),
            "token_out_ms_per_token_mean": round(statistics.fmean(token_out), 3),
            "token_out_tokens_per_s_per_user_mean": round(1000.0 / statistics.fmean(token_out), 3),
            "traced_model_only_ms_per_token": round(statistics.median(model_only), 3),
            "token_out_ms_per_token": round(statistics.median(token_out), 3),
            "token_out_tokens_per_s_per_user": round(1000.0 / statistics.median(token_out), 3),
            "median_vs_mean_note": "the *_mean fields are the like-for-like comparison; see statistic_note",
            "recorded_optimized_full_model": {
                "traced_model_only_no_sampling": 21.76,
                "token_out_incl_readback": 23.013,
                "source": "doc/optimized_full_model/perf.json",
                "statistic": "mean (tests/test_full_model_perf.py bench(): total wall time / iterations)",
            },
            "counters": dict(gen.counters),
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
