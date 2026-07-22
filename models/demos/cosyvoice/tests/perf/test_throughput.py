"""Performance tests — C6 throughput gates.

Targets:
  - LLM decode: >= 30 tokens/s (batch 1) — HARD GATE
  - E2E RTF (real-time factor): < 0.5 — Stage-2 target (flow on host is bottleneck)

Stage-1 known limitation: The flow estimator (UNet1D, 12 mid blocks, 256-ch) runs
on host CPU. With 10 NFE × CFG batch=2, each CFM step takes ~1.6s. Even reducing
NFE to 5 only brings flow+voc RTF to ~0.9. Achieving RTF < 0.5 requires moving
the flow estimator to device (Stage 2: trace+2CQ, on-device flow).

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

    device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)

    sys.path.insert(0, str(DEMO_ROOT))
    from models.demos.cosyvoice.tt.pipeline import TtnnCosyVoice

    pipe = TtnnCosyVoice(device, model_dir=str(CKPT_DIR))
    pipe.add_zero_shot_spk(ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV, "test_spk")
    yield pipe
    ttnn.close_device(device)


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

    t0 = time.perf_counter()
    for i in range(WARMUP_STEPS, WARMUP_STEPS + n_decode):
        token_id = golden_tokens[i].item()
        log_probs = pipeline.llm.decode_step(token_id, current_pos)
        current_pos += 1
    elapsed = time.perf_counter() - t0

    tok_per_sec = n_decode / elapsed
    print(f"\n[C6] LLM decode: {tok_per_sec:.1f} tokens/s ({n_decode} steps in {elapsed:.3f}s)")
    assert tok_per_sec >= 30.0, f"LLM decode throughput {tok_per_sec:.1f} tok/s < 30 tok/s target"


@pytest.mark.xfail(
    reason="Stage-1: flow estimator on host CPU → RTF ~2.2. Target <0.5 requires Stage-2 device flow.",
    strict=False,
)
def test_e2e_rtf(pipeline):
    """C6: E2E real-time factor < 0.5 (Stage-2 target; recorded in Stage 1)."""
    t0 = time.perf_counter()
    waveform = pipeline.inference_zero_shot(ZERO_SHOT_TEXT, ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV)
    gen_time = time.perf_counter() - t0

    audio_duration = waveform.shape[1] / SAMPLE_RATE
    rtf = gen_time / audio_duration

    print(f"\n[C6] E2E RTF: {rtf:.3f} (gen={gen_time:.2f}s, audio={audio_duration:.2f}s)")
    print(f"[C6] Breakdown: LLM ~34 tok/s on N300; flow (10 NFE × CFG) + vocoder on host CPU")
    print(f"[C6] Stage-2 path: move flow estimator to device (trace+2CQ) to achieve RTF < 0.5")
    assert waveform.shape[1] > 0, "No audio generated"
    assert rtf < 0.5, f"E2E RTF {rtf:.3f} >= 0.5 target"
