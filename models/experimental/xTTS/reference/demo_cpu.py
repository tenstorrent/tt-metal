# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
XTTS-v2 CPU Demo — imports directly from reference/ (self-contained PyTorch modules).

Pipeline (all from local reference/ files):
    reference/xtts_config.py  → XttsConfig
    reference/xtts.py         → Xtts (orchestrates all sub-modules)
      ├── reference/gpt.py            → GPT autoregressive code generator
      │     ├── reference/gpt_inference.py    → GPT2InferenceModel
      │     ├── reference/latent_encoder.py   → ConditioningEncoder
      │     └── reference/perceiver_encoder.py → PerceiverResampler
      ├── reference/hifigan_decoder.py → HifiDecoder + SpeakerEncoder
      ├── reference/tokenizer.py      → VoiceBpeTokenizer
      └── reference/xtts_manager.py   → SpeakerManager, LanguageManager
"""

import os
import time

import scipy.io.wavfile as wavfile
import torch

from models.experimental.xtts.reference.tokenizer import VoiceBpeTokenizer
from models.experimental.xtts.reference.xtts import Xtts
from models.experimental.xtts.reference.xtts_config import XttsConfig

# PyTorch >=2.6 changed torch.load default weights_only=True, which breaks
# XTTS checkpoint loading. Patch back to False (Coqui model is trusted).
_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

MODEL_DIR = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
REF_WAV = "/home/ubuntu/tt-metal/reference.wav"
OUTPUT_WAV = os.path.join(os.path.dirname(__file__), "output_reference.wav")
# ~398 tokens — exercises the GPT near its 402-token limit
TEXT = (
    "voice synthesis has come a long way in recent years. "
    "modern systems can now generate natural sounding speech from text with remarkable accuracy. "
    "the key challenge is capturing the unique characteristics of a speaker's voice, "
    "including their tone, rhythm, pitch, and emotional expression. "
    "deep learning models trained on large datasets have made this possible by learning "
    "complex patterns in speech that were previously difficult to model. "
    "the transformer architecture, originally designed for natural language processing, "
    "has proven particularly effective for audio generation tasks. "
    "by conditioning the model on a short reference audio clip, "
    "we can clone the voice of any speaker and synthesize"
)
LANGUAGE = "en"

# --- Load model from reference/ modules ---
config = XttsConfig()
config.load_json(f"{MODEL_DIR}/config.json")

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, eval=True)
# No model.cuda() — CPU only

# Show token count before running
_tok = VoiceBpeTokenizer(vocab_file=f"{MODEL_DIR}/vocab.json")
token_count = len(_tok.encode(TEXT.strip().lower(), lang=LANGUAGE))
print(f"Token count : {token_count} / 402 max")
print(f"Text        : {TEXT[:80]}...")
print(f"Reference   : {REF_WAV}")
print(f"Language    : {LANGUAGE}\n")

t0 = time.time()

outputs = model.synthesize(
    TEXT,
    config,
    speaker_wav=REF_WAV,
    gpt_cond_len=6,
    language=LANGUAGE,
    enable_text_splitting=True,  # splits text into sentences, each synthesized separately then concatenated
)

elapsed = time.time() - t0
wav = outputs["wav"]
audio_duration = len(wav) / 24000
rtf = elapsed / audio_duration

wavfile.write(OUTPUT_WAV, 24000, wav)

print(f"Output saved : {OUTPUT_WAV}")
print(f"Processing   : {elapsed:.2f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.2f}x")
