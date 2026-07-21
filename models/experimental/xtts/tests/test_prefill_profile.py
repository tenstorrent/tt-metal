# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profiling harness: run ONE GPT prefill pass (prompt -> KV cache) for device-time
measurement. Not a correctness test. Run under tracy:

    python -m tracy -v -r -p -o gpt_prefill \\
        -m "pytest models/experimental/xtts/tests/test_prefill_profile.py -k hello"

then summarise the emitted ops_perf_results CSV with scratch/perf_summary.py.
One pass only (keeps the ~640 ops under the on-device profiler buffer; 2+ passes drop
markers). Vary --prompt via -k to profile different prompt lengths.
"""
import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_conditioning import (
    load_reference_audio,
    wav_to_mel,
    reference_conditioning,
)
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel

TILE = 32
PROMPTS = {
    "hello": "hello world",
    "long": "The quick brown fox jumps over the lazy dog while the sun sets slowly over the hills. "
    "Text to speech synthesis on Tenstorrent hardware is fast, natural, and efficient.",
}


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("prompt", list(PROMPTS))
def test_prefill_profile(device, xtts_state_dict, prompt):
    sd = xtts_state_dict
    wav = load_reference_audio(sample="en_sample.wav")
    mel = wav_to_mel(wav, sd["mel_stats"].cpu())
    with torch.no_grad():
        cond = reference_conditioning(sd)(mel).transpose(1, 2)  # [1, 32, 1024]
    wrapped = wrap_text_ids(preprocess_text(PROMPTS[prompt], lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)

    tt = TtXttsGptModel(sd, device)
    print(f"PREFILLINFO prompt={prompt} len={cond.shape[1] + wrapped.shape[1]}")

    cond_tt = ttnn.from_torch(cond.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    kv = tt.prefill(wrapped, cond_tt)
    ttnn.deallocate(cond_tt)
    for k, v in kv:
        ttnn.deallocate(k)
        ttnn.deallocate(v)
