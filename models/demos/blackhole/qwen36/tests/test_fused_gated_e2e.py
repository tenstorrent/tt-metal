# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E correctness + TPS for the fused GDN decode (raw kernel read + ttnn gated-RMSNorm/silu(z)).

Validates the QWEN_GDN_FUSED_DECODE=1 path decodes coherent text (lane 0 → 'Paris' for
"The capital of France is") AND measures B=8 traced TPS (two-point). NOTE: this test issues 3
generate() calls, so its TPS is ~15 tok/s below the authoritative bench_decode_tps_tp two-point
(the resident correctness-trace inflates ms/step); use it for CORRECTNESS, bench for the TPS number.

Run: QWEN_GDN_FUSED_DECODE=1 QWEN_BENCH_B=8 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
       pytest models/demos/blackhole/qwen36/tests/test_fused_gated_e2e.py -v -s
"""
import os
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

N1, N2 = 8, 72


@parametrize_mesh_tp()
def test_fused_gated_e2e(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    os.environ["QWEN_GDN_FUSED_DECODE"] = "1"
    mesh_device.enable_program_cache()
    B = int(os.environ.get("QWEN_BENCH_B", "8"))
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=2048)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
    prompts = [prompt] * B

    # ---- Correctness: decode 16 tokens (traced) and print lane-0 text ----
    outs = model.generate_tp_batched(prompts, max_new_tokens=16, use_trace=True)
    text0 = tok.decode(outs[0])
    logger.info(f"FUSED_GATED lane0 text: {text0!r}")
    logger.info(f"FUSED_GATED lane1 text: {tok.decode(outs[1])!r}")

    # ---- TPS: two-point traced ----
    def timed(n):
        best = float("inf")
        for _ in range(2):
            t0 = time.time()
            model.generate_tp_batched(prompts, max_new_tokens=n, use_trace=True)
            ttnn.synchronize_device(mesh_device)
            best = min(best, time.time() - t0)
        return best

    per_step = (timed(N2) - timed(N1)) / (N2 - N1)
    agg = B / per_step
    logger.info(f"FUSED_GATED_TPS B={B}: traced={agg:.1f}tok/s(agg,{per_step*1e3:.1f}ms/step)  lane0={text0!r}")
    assert "Paris" in text0, f"lane0 should mention Paris, got {text0!r}"
