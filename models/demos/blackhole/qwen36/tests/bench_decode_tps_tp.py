# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode throughput (TPS) benchmark for the batched-decode work (Steps 1/2/4).

Two-point timing isolates steady-state decode: time generate_tp_batched at N1 and N2
new tokens; per-step time = (t(N2) - t(N1)) / (N2 - N1) cancels the (identical) prefill,
warmup, and one-time trace capture. Reports per-sequence and aggregate tokens/sec for
eager vs traced decode at batch B = QWEN_BENCH_B.

Run (per batch):
    QWEN_BENCH_B=1 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/bench_decode_tps_tp.py -v -s
    QWEN_BENCH_B=8 ... (same)
"""
import os
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

N1, N2 = 8, 72  # two-point token counts (Δ = 64 steady-state decode steps)
REPS = 2


@parametrize_mesh_tp()
def test_bench_decode_tps(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    B = int(os.environ.get("QWEN_BENCH_B", "8"))
    eager_only = os.environ.get("EAGER_ONLY") == "1"
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=2048)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
    prompts = [prompt] * B

    def timed(n, use_trace):
        best = float("inf")
        for _ in range(REPS):
            t0 = time.time()
            model.generate_tp_batched(prompts, max_new_tokens=n, use_trace=use_trace)
            ttnn.synchronize_device(mesh_device)
            best = min(best, time.time() - t0)
        return best

    results = {}
    _configs = [("eager", False)] if eager_only else [("eager", False), ("traced", True)]
    for label, use_trace in _configs:
        t_n1 = timed(N1, use_trace)
        t_n2 = timed(N2, use_trace)
        per_step = (t_n2 - t_n1) / (N2 - N1)
        per_seq_tps = 1.0 / per_step
        agg_tps = B / per_step
        results[label] = (per_step * 1e3, per_seq_tps, agg_tps)
        logger.info(
            f"[B={B} {label}] {per_step*1e3:.2f} ms/step | per-seq {per_seq_tps:.1f} tok/s | "
            f"aggregate {agg_tps:.1f} tok/s"
        )

    logger.info(f"DECODE_TPS_SUMMARY B={B}: " + "  ".join(
        f"{k}={v[2]:.1f}tok/s(agg,{v[0]:.1f}ms/step)" for k, v in results.items()))
    # Sanity: traced should be no slower than eager.
    if "traced" in results:
        assert results["traced"][2] >= results["eager"][2] * 0.9, "traced unexpectedly slower than eager"
