"""E2E tests — run pipeline for each mode, assert no errors + output shape sanity.

Usage:
    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal
    python -m pytest models/demos/cosyvoice/tests/e2e/test_modes.py -v --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

DEMO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"
CV_SRC = DEMO_ROOT / "model_data" / "CosyVoice_src"
ASSET_DIR = CV_SRC / "asset"
GOLDEN_DIR = DEMO_ROOT / "model_data" / "golden"

ZERO_SHOT_PROMPT_WAV = str(ASSET_DIR / "zero_shot_prompt.wav")
CROSS_LINGUAL_PROMPT_WAV = str(ASSET_DIR / "cross_lingual_prompt.wav")
ZERO_SHOT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
INSTRUCT_TEXT = "用四川话说这句话<|endofprompt|>"
SFT_SPK_ID = "test_spk"

ZERO_SHOT_TEXT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。"
CROSS_LINGUAL_TEXT = "在他讲述那个荒诞故事的过程中，他突然停下来，因为他自己也被逗笑了。"
SFT_TEXT = "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"

MIN_WAVEFORM_SAMPLES = 2400  # 0.1s at 24kHz


@pytest.fixture(scope="module")
def pipeline():
    import ttnn

    device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024, trace_region_size=5000000)

    sys.path.insert(0, str(DEMO_ROOT))
    from models.demos.cosyvoice.tt.pipeline import TtnnCosyVoice

    pipe = TtnnCosyVoice(device, model_dir=str(CKPT_DIR))
    pipe.add_zero_shot_spk(ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV, SFT_SPK_ID)
    yield pipe
    ttnn.close_device(device)


def test_e2e_zero_shot(pipeline):
    waveform = pipeline.inference_zero_shot(ZERO_SHOT_TEXT, ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV)
    assert waveform.ndim == 2
    assert waveform.shape[0] == 1
    assert waveform.shape[1] >= MIN_WAVEFORM_SAMPLES, f"Waveform too short: {waveform.shape[1]} samples"


def test_e2e_cross_lingual(pipeline):
    waveform = pipeline.inference_cross_lingual(CROSS_LINGUAL_TEXT, CROSS_LINGUAL_PROMPT_WAV)
    assert waveform.ndim == 2
    assert waveform.shape[0] == 1
    assert waveform.shape[1] >= MIN_WAVEFORM_SAMPLES


def test_e2e_instruct2(pipeline):
    waveform = pipeline.inference_instruct2(ZERO_SHOT_TEXT, INSTRUCT_TEXT, ZERO_SHOT_PROMPT_WAV)
    assert waveform.ndim == 2
    assert waveform.shape[0] == 1
    assert waveform.shape[1] >= MIN_WAVEFORM_SAMPLES


def test_e2e_sft(pipeline):
    waveform = pipeline.inference_sft(SFT_TEXT, SFT_SPK_ID)
    assert waveform.ndim == 2
    assert waveform.shape[0] == 1
    assert waveform.shape[1] >= MIN_WAVEFORM_SAMPLES


TOKEN_ACCURACY_MODES = ["zero_shot", "cross_lingual", "instruct2", "sft"]


@pytest.mark.parametrize("mode", TOKEN_ACCURACY_MODES)
def test_token_accuracy(pipeline, mode):
    """C7: Teacher-forced top-25 agreement >95% vs golden (lesson 14 metric).

    Uses the golden lm_input + golden tokens in teacher-forced mode: at each
    decode step, check if the golden token is within the top-25 of the TTNN
    model's log-probabilities. This is the correct metric for bf16 + RAS
    (exact match is ~4% due to different multinomial draws).
    """
    golden_path = GOLDEN_DIR / "llm" / f"{mode}.pt"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not available for {mode}")

    golden = torch.load(str(golden_path), map_location="cpu", weights_only=True)
    lm_input = golden["lm_input"]
    golden_tokens = golden["tokens"]

    n_steps = min(len(golden_tokens), 50)

    log_probs = pipeline.llm.prefill(lm_input)

    top25_hits = 0
    current_pos = lm_input.shape[1]

    for i in range(n_steps):
        gt = golden_tokens[i].item()
        _, top_idxs = log_probs.topk(25)
        if gt in top_idxs.tolist():
            top25_hits += 1

        if i < n_steps - 1:
            log_probs = pipeline.llm.decode_step(gt, current_pos)
            current_pos += 1

    accuracy = top25_hits / n_steps
    print(f"\n[C7] {mode}: top-25 agreement = {accuracy:.3f} ({top25_hits}/{n_steps})")
    assert accuracy > 0.95, f"[{mode}] Top-25 agreement {accuracy:.3f} < 0.95 ({top25_hits}/{n_steps})"
