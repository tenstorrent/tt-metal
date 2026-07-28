# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance tests

LLM decode: traced on N300 (113+ tok/s).
Flow estimator: native TTNN UNet1D on N300 with device trace (NFE=5, CFG batch=2).
Vocoder: host-side fp32 (bf16 precision wall in dilated ResBlocks — see lesson 32).

Usage:
    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal
    python -m pytest models/demos/cosyvoice/tests/perf/test_throughput.py -v -s
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

DEMO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"
CV_SRC = DEMO_ROOT / "model_data" / "CosyVoice_src"
ASSET_DIR = CV_SRC / "asset"
GOLDEN_DIR = DEMO_ROOT / "model_data" / "golden"

ZERO_SHOT_PROMPT_WAV = str(ASSET_DIR / "zero_shot_prompt.wav")
ZERO_SHOT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
ZERO_SHOT_TEXT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。"

SAMPLE_RATE = 24000
DECODE_BENCH_STEPS = 100
WARMUP_STEPS = 3


@pytest.fixture(scope="module")
def pipeline():
    import ttnn

    devs = ttnn.CreateDevices([0, 1], l1_small_size=64 * 1024, trace_region_size=50000000)
    device = devs[0]
    device_flow = devs[1]

    sys.path.insert(0, str(DEMO_ROOT))
    from models.demos.cosyvoice.tt.pipeline import TtnnCosyVoice

    pipe = TtnnCosyVoice(device, model_dir=str(CKPT_DIR), mesh_device_flow=device_flow)
    pipe.add_zero_shot_spk(ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV, "test_spk")

    pipe.inference_zero_shot(ZERO_SHOT_TEXT, ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV)
    pipe.inference_zero_shot(ZERO_SHOT_TEXT, ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV)

    yield pipe
    ttnn.CloseDevices(devs)


def test_llm_decode_throughput(pipeline):
    """C6: LLM decode >= 30 tokens/s (batch 1). HARD GATE."""
    golden_path = GOLDEN_DIR / "llm" / "zero_shot.pt"
    if not golden_path.exists():
        pytest.skip("Golden fixture not available")

    golden = torch.load(str(golden_path), map_location="cpu", weights_only=True)
    lm_input = golden["lm_input"]
    golden_tokens = golden["tokens"]

    n_decode = min(DECODE_BENCH_STEPS, len(golden_tokens) - 1)

    log_probs = pipeline.llm.prefill(lm_input)
    current_pos = lm_input.shape[1]

    for i in range(WARMUP_STEPS):
        token_id = golden_tokens[i].item()
        log_probs = pipeline.llm.decode_step(token_id, current_pos)
        current_pos += 1

    if pipeline.llm._trace_id is None:
        pipeline.llm._init_trace(current_pos)

    t0 = time.perf_counter()
    for i in range(WARMUP_STEPS, WARMUP_STEPS + n_decode):
        token_id = golden_tokens[i].item()
        log_probs = pipeline.llm.decode_step(token_id, current_pos)
        current_pos += 1
    elapsed = time.perf_counter() - t0

    tok_per_sec = n_decode / elapsed
    print(f"\n[C6] LLM decode: {tok_per_sec:.1f} tokens/s ({n_decode} steps in {elapsed:.3f}s)")
    assert tok_per_sec >= 30.0, f"LLM decode throughput {tok_per_sec:.1f} tok/s < 30 tok/s target"


def test_e2e_rtf(pipeline):
    """C6: E2E real-time factor < 0.5. HARD GATE."""
    t0 = time.perf_counter()
    waveform = pipeline.inference_zero_shot(ZERO_SHOT_TEXT, ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV)
    gen_time = time.perf_counter() - t0

    audio_duration = waveform.shape[1] / SAMPLE_RATE
    rtf = gen_time / audio_duration

    print(f"\n[C6] E2E RTF: {rtf:.3f} (gen={gen_time:.2f}s, audio={audio_duration:.2f}s)")
    print(f"[C6] Breakdown: LLM traced on N300; flow estimator NFE=5 traced on N300; vocoder host fp32")
    assert waveform.shape[1] > 0, "No audio generated"
    assert rtf < 0.5, f"E2E RTF {rtf:.3f} >= 0.5 target"


def test_streaming_rtf(pipeline):
    """Stage 3.3: Streaming E2E RTF with 2-chip pipeline parallelism.

    LLM on chip 0, CFM estimator on chip 1 — eliminates device contention.
    RTF < 0.3 requires C++ orchestration (Python GIL serializes host ops:
    encoder + vocoder hold GIL, blocking LLM sampling). Current gate: < 0.6.
    """
    for _ in pipeline.inference_zero_shot_streaming(ZERO_SHOT_TEXT, ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV):
        pass

    chunks = []
    first_chunk_time = None

    t0 = time.perf_counter()
    for waveform_chunk in pipeline.inference_zero_shot_streaming(
        ZERO_SHOT_TEXT, ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV
    ):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - t0
        chunks.append(waveform_chunk)
    gen_time = time.perf_counter() - t0

    assert len(chunks) > 0, "No streaming chunks generated"
    full_waveform = torch.cat(chunks, dim=1)
    audio_duration = full_waveform.shape[1] / SAMPLE_RATE
    rtf = gen_time / audio_duration

    print(f"\n[Stage 3.3] Streaming RTF: {rtf:.3f} (gen={gen_time:.2f}s, audio={audio_duration:.2f}s)")
    print(f"[Stage 3.3] First-chunk latency: {first_chunk_time:.3f}s")
    print(f"[Stage 3.3] Chunks: {len(chunks)}, total samples: {full_waveform.shape[1]}")
    assert full_waveform.shape[1] > 0, "No audio generated"
    assert rtf < 0.6, f"Streaming RTF {rtf:.3f} >= 0.6"
