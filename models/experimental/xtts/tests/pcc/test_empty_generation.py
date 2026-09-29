# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""An empty generation must return empty audio, not reach the vocoder.

``TtTracedDecoder.run`` caps ``cut`` at ``steps_run - 1`` because ``latents_buf[i]`` holds the
latent of code ``i-1`` — n codes need n+1 replayed steps. So two inputs yield zero latents:

* a step-0 STOP (the empty-audio contract, matching eager ``generate()``);
* the smallest budget, where one step runs and no code owns a latent yet.

Either way the traced latents are ``[1, 0, 1024]``. Vocoding those makes the upsampler build a
matmul program config with ``per_core_M = 0`` and divide by it (ZeroDivisionError).

Eager ``generate()`` signals the same outcome with a different sentinel -- ``latents_tt=None`` --
which unpacks as ``None.shape`` in ``TtLatentUpsampler.forward`` (AttributeError). One guard cannot
serve both, so each inference entry point short-circuits on its own sentinel.

All three cases share one model build in a single test on purpose. Building ``TtXtts`` and taking
the first full-pipeline trace is the expensive part on CI's cold kernel cache, and the xtts unit
job runs inside a fixed 25-minute team budget (``.github/time_budget.yaml``); three separate tests
paid it three times. The scenarios are labelled so a failure still names the one that broke.
"""

import pytest
import torch

import ttnn
from models.experimental.xtts.config import (
    GENERATION,
    L1_SMALL_SIZE,
    NUM_LATENTS,
    SESSION_TRACE_REGION,
    STOP_AUDIO_TOKEN,
    STOP_TEXT_TOKEN,
    TILE,
)
from models.experimental.xtts.reference.xtts_conditioning import load_coqui_test_audio
from models.experimental.xtts.reference.xtts_gpt_generate import wrap_text_ids
from models.experimental.xtts.reference.xtts_inference import XttsReference
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_inference import TtXtts
from models.experimental.xtts.tt.xtts_sampler import TtSampler

REF_SECONDS = 6
SAMPLING = dict(
    temperature=GENERATION.temperature,
    top_k=GENERATION.top_k,
    top_p=GENERATION.top_p,
    repetition_penalty=GENERATION.repetition_penalty,
    min_new_tokens=0,
)


def _inputs(device, xtts_state_dict):
    """Build the smallest real inference inputs: model, reference audio, one tile of text."""
    import math

    from scipy.signal import resample_poly

    from models.experimental.xtts.config import DEMO, MEL_SR

    reference = XttsReference(xtts_state_dict)
    wav = load_coqui_test_audio(samples=DEMO.ref_audio.split("+"), max_seconds=REF_SECONDS)
    g = math.gcd(SPK_SR, MEL_SR)
    spk = resample_poly(wav[0].numpy()[: MEL_SR * REF_SECONDS], SPK_SR // g, MEL_SR // g)
    spk_tt = ttnn.from_torch(
        torch.from_numpy(spk.astype("float32")).reshape(1, -1),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.float32,
    )
    ids = wrap_text_ids(preprocess_text("Hello world", lang="en"))
    real_len = ids.shape[1]
    pad_to = -(-real_len // TILE) * TILE
    padded = torch.nn.functional.pad(ids, (0, pad_to - real_len), value=STOP_TEXT_TOKEN)
    tt = TtXtts(device, xtts_state_dict, reference.decoder_full)
    return tt, wav, spk_tt, padded, real_len, pad_to


def _assert_empty_audio(wav_dev, where):
    """Pin the empty-audio contract, not just its emptiness.

    The vocoder returns ``[batch, samples, channels]`` (measured ``(1, 4352, 1)`` bfloat16 TILE), so
    an empty result is zero *samples* with the channel axis intact. Asserting only ``numel() == 0``
    accepts ``(1, 1, 0)`` too, which claims one sample in a tensor with no channels and answers
    ``shape[1] == 1`` to a caller asking how many samples there are.
    """
    assert tuple(wav_dev.shape) == (1, 0, 1), f"{where}: expected (1, 0, 1), got {tuple(wav_dev.shape)}"
    assert wav_dev.dtype == ttnn.bfloat16, f"{where}: expected bfloat16, got {wav_dev.dtype}"
    assert wav_dev.layout == ttnn.TILE_LAYOUT, f"{where}: expected TILE, got {wav_dev.layout}"
    assert ttnn.to_torch(wav_dev).numel() == 0, f"{where}: produced audio samples"


def _max_seq(pad_to, budget):
    """KV geometry for a prompt of pad_to text tokens and a budget-code decode."""
    return -(-(NUM_LATENTS + pad_to + budget + 2) // TILE) * TILE


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": L1_SMALL_SIZE, "trace_region_size": SESSION_TRACE_REGION}], indirect=True
)
def test_empty_generation_returns_empty_audio(device, xtts_state_dict, reset_seeds):
    """Every way of generating nothing returns empty audio through every inference entry point."""
    tt, wav, spk_tt, padded, real_len, pad_to = _inputs(device, xtts_state_dict)

    # --- 1. Smallest budget: one step runs, no code owns a latent yet. -----------------------
    budget = 1
    wav_dev, codes, perf = tt.inference_fully_traced(
        padded, wav, spk_tt, _max_seq(pad_to, budget), max_new_tokens=budget, text_real_len=real_len, **SAMPLING
    )
    assert codes.shape[1] == 0, f"one-step budget produced {codes.shape[1]} codes"
    _assert_empty_audio(wav_dev, "one-step budget, traced")
    assert perf["vocoder_replay_s"] == 0.0, "vocoder ran on an empty generation"

    # --- 2. Step-0 STOP through the traced wrapper. ------------------------------------------
    budget = 8  # a real budget -- STOP is forced, not a consequence of running out of steps
    # The constant is built OUTSIDE the capture, so the traced step only records a
    # device-to-device copy of it into tok_buf.
    stop_const = ttnn.from_torch(
        torch.full((1, 1), STOP_AUDIO_TOKEN, dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.uint32,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TtSampler, "pick_dev", lambda self, logits, gumbel=None, bias=None: stop_const)
        wav_dev, codes, perf = tt.inference_fully_traced(
            padded, wav, spk_tt, _max_seq(pad_to, budget), max_new_tokens=budget, text_real_len=real_len, **SAMPLING
        )
    assert perf["stopped"], "forced STOP was not reported as a stop"
    assert codes.shape[1] == 0, f"step-0 STOP emitted {codes.shape[1]} codes (STOP must not be one)"
    _assert_empty_audio(wav_dev, "step-0 STOP, traced")
    assert perf["vocoder_replay_s"] == 0.0, "vocoder ran on a step-0 STOP"

    # --- 3. Step-0 STOP through the eager wrapper, which signals empty as None. --------------
    # SAMPLING carries temperature 0.65, so generate() builds a TtSampler and picks through pick.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TtSampler, "pick", lambda self, logits: STOP_AUDIO_TOKEN)
        wav_dev, codes = tt.inference(padded, wav, spk_tt, max_new_tokens=8, **SAMPLING)
    assert codes.shape[1] == 0, f"eager step-0 STOP emitted {codes.shape[1]} codes"
    _assert_empty_audio(wav_dev, "step-0 STOP, eager")
