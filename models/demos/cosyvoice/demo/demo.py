"""CosyVoice2-0.5B E2E demo — generates 20 WAVs (4 modes × 5 languages) on N300.

Usage:
    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal
    python -m pytest models/demos/cosyvoice/demo/demo.py -v --tb=short

Exit gate: 20 WAVs generated in demo/output/ with no errors (C2+C3+C4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import soundfile
import torch

DEMO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CKPT_DIR = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"
CV_SRC = DEMO_ROOT / "model_data" / "CosyVoice_src"
ASSET_DIR = CV_SRC / "asset"

ZERO_SHOT_PROMPT_WAV = str(ASSET_DIR / "zero_shot_prompt.wav")
CROSS_LINGUAL_PROMPT_WAV = str(ASSET_DIR / "cross_lingual_prompt.wav")
ZERO_SHOT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
INSTRUCT_TEXT = "用四川话说这句话<|endofprompt|>"
SFT_SPK_ID = "demo_spk"

LANGS = ["zh", "en", "ja", "yue", "ko"]
MODES = ["sft", "zero_shot", "cross_lingual", "instruct2"]


def _load_texts():
    with open(DATA_DIR / "texts.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline():
    import ttnn

    device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)

    sys.path.insert(0, str(DEMO_ROOT))
    from models.demos.cosyvoice.tt.pipeline import TtnnCosyVoice

    pipe = TtnnCosyVoice(device, model_dir=str(CKPT_DIR))
    pipe.add_zero_shot_spk(ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV, SFT_SPK_ID)
    yield pipe
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def texts():
    return _load_texts()


def _save_wav(waveform: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = waveform.detach().cpu().numpy()
    if audio.ndim == 2:
        audio = audio[0]
    soundfile.write(str(path), audio, 24000)


@pytest.mark.parametrize("lang", LANGS)
def test_sft(pipeline, texts, lang):
    text = texts[lang][0]
    waveform = pipeline.inference_sft(text, SFT_SPK_ID)
    assert waveform.shape[1] > 0, f"sft/{lang}: empty waveform"
    _save_wav(waveform, OUTPUT_DIR / f"sft_{lang}.wav")


@pytest.mark.parametrize("lang", LANGS)
def test_zero_shot(pipeline, texts, lang):
    text = texts[lang][0]
    waveform = pipeline.inference_zero_shot(text, ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV)
    assert waveform.shape[1] > 0, f"zero_shot/{lang}: empty waveform"
    _save_wav(waveform, OUTPUT_DIR / f"zero_shot_{lang}.wav")


@pytest.mark.parametrize("lang", LANGS)
def test_cross_lingual(pipeline, texts, lang):
    text = texts[lang][1]
    waveform = pipeline.inference_cross_lingual(text, CROSS_LINGUAL_PROMPT_WAV)
    assert waveform.shape[1] > 0, f"cross_lingual/{lang}: empty waveform"
    _save_wav(waveform, OUTPUT_DIR / f"cross_lingual_{lang}.wav")


@pytest.mark.parametrize("lang", LANGS)
def test_instruct2(pipeline, texts, lang):
    text = texts[lang][0]
    waveform = pipeline.inference_instruct2(text, INSTRUCT_TEXT, ZERO_SHOT_PROMPT_WAV)
    assert waveform.shape[1] > 0, f"instruct2/{lang}: empty waveform"
    _save_wav(waveform, OUTPUT_DIR / f"instruct2_{lang}.wav")
