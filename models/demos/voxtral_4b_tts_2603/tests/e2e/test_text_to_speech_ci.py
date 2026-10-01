# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CI-sized accuracy check: the full text-to-speech pipeline on device for a few frames and two voices.

Same chain as the demo (`pipeline.run_text_to_speech`), 32 real prompts per voice, FRAMES audio frames,
each stage scored against the PyTorch reference TEACHER-FORCED onto this run's own codes and hidden
states, through to the vocoded waveform -- the same comparison `test_e2e_text_to_speech.py` makes over a
whole utterance, without its Whisper/UTMOS scoring. Every threshold is a fixed number.

Three cases, one per shape the shared-prefix prefill has to handle:
  * `casual_male`: its tile-rounded prefix fits inside its prompt;
  * `ar_male`: its prefix runs past the prompt (start 70 -> 96 rows, prompt 90), so the prefix ids are
    padded -- the case that crashed for 7 of the 20 presets;
  * `single_prompt`: ONE long text repeated in all 32 rows, which is what the demo does with a single
    `--text`. Every row is identical, so there is no per-row tail at all; a 203-token prompt (224 padded)
    is long enough that the whole-prompt prefill would not fit L1.

First run ~15 min (the CPU reference for 3 x FRAMES frames); later runs reuse the cached reference.
"""
from __future__ import annotations

import pytest
import torch

from models.demos.voxtral_4b_tts_2603.reference import golden
from models.demos.voxtral_4b_tts_2603.tt import common, pipeline

pytestmark = pytest.mark.timeout(3600)

FRAMES = 8
# (case id, voice, text repeated in every row or None for the package's 32 distinct texts)
CASES = (
    ("casual_male", "casual_male", None),
    ("ar_male", "ar_male", None),
    (
        "single_prompt",
        "casual_male",
        "On the first warm evening of the year, the whole neighbourhood gathered in the narrow street outside the "
        "old bakery, sharing bread, lemonade and stories about the winter, while the children chased each other "
        "between the long wooden tables until the lamps came on.",
    ),
)
PCC_TARGET = 0.99
MIN_ACOUSTIC_AGREEMENT = 0.98  # fraction of live acoustic codes equal to the teacher-forced reference
MIN_SEMANTIC_AGREEMENT = 0.99


def _inputs(voice, text, batch=common.DEFAULT_BATCH):
    return pipeline.speech_inputs(texts=None if text is None else [text] * batch, voice=voice, batch=batch)


@pytest.fixture(scope="module")
def pipe(device, hf_model):
    longest = max(int(_inputs(voice, text)[0].shape[-1]) for _, voice, text in CASES)
    return pipeline.build_pipeline(
        device,
        model=hf_model,
        heads=("text_to_speech",),
        kv_capacity=pipeline.tts_kv_capacity(longest, FRAMES),
    )


@pytest.mark.parametrize("case, voice, text", CASES, ids=[c[0] for c in CASES])
def test_text_to_speech_short(pipe, hf_model, case, voice, text):
    common.use_all_cpu_threads()
    input_ids, audio_mask, voice_embedding, _ = _inputs(voice, text, batch=pipe.batch)
    batch = pipe.batch
    x0 = pipe.noise(FRAMES, batch=batch)
    cfg_alpha = torch.full((batch,), pipeline.DEFAULT_CFG_ALPHA)
    staged = pipe.stage_voice(audio_mask, voice_embedding, input_ids=input_ids)
    tt = pipe.run_text_to_speech(
        input_ids=input_ids, x0=x0, cfg_alpha=cfg_alpha, max_frames=FRAMES, collect=True, voice=staged
    )
    frames = tt["frames_decoded"]
    assert frames >= 1 and tuple(tt["codes"].shape) == (batch, 37, frames)
    assert torch.isfinite(tt["waveform"]).all()

    tt_hidden = torch.stack([d["llm_hidden"] for d in tt["diagnostics"]], dim=-1)
    key = common.golden_key(
        arm="ci-aligned",
        ids=input_ids,
        voice=voice_embedding,
        x0=x0,
        cfg=cfg_alpha,
        codes=tt["codes"],
        hidden=tt_hidden,
        chain=golden.CHAIN_VERSION,
    )
    ref = common.cached_golden(
        key,
        lambda: golden.hf_reference_text_to_speech(
            hf_model,
            input_ids,
            x0,
            cfg_alpha,
            FRAMES,
            fed_codes=tt["codes"],
            fed_hidden=tt_hidden,
            audio_mask=audio_mask,
            voice_embedding=voice_embedding,
        ),
    )

    def worst(pairs):
        return min(common.pcc(a, b) for a, b in pairs)

    prefill = worst((tt["prefill_hidden"][i], ref["prefill_hidden"][i]) for i in range(batch))
    decode = worst(
        (tt["diagnostics"][t]["llm_hidden"][i], ref["llm_hiddens"][t][i]) for t in range(frames) for i in range(batch)
    )
    semantic = worst(
        (tt["diagnostics"][t]["semantic_logits"][i], ref["diagnostics"][t]["semantic_logits_raw"][i])
        for t in range(frames)
        for i in range(batch)
    )
    x_final = worst(
        (tt["diagnostics"][t]["x_final"][i].clamp(-1, 1), ref["diagnostics"][t]["x_final"][i])
        for t in range(frames)
        for i in range(batch)
    )
    live = (tt["codes"][:, 0, :] != pipe.stop_token_id).unsqueeze(1).expand(-1, 36, -1)
    wave = min(common.pcc(tt["waveform"][i], ref["waveform"][i]) for i in range(batch))
    peak = float(tt["waveform"].abs().max())
    acoustic_agree = float((tt["codes"][:, 1:, :] == ref["codes"][:, 1:, :])[live].float().mean())
    semantic_agree = float((tt["codes"][:, 0, :] == ref["codes"][:, 0, :]).float().mean())
    print(
        f"\n[{case}] prompt {input_ids.shape[-1]} tokens, {frames} frames: prefill {prefill:.6f} decode {decode:.6f} "
        f"semantic {semantic:.6f} x_final {x_final:.6f} waveform {wave:.6f} (peak {peak:.3f}); acoustic codes {acoustic_agree:.4f}, semantic codes "
        f"{semantic_agree:.4f}"
    )
    assert prefill >= PCC_TARGET, f"[{case}] prefill hidden PCC {prefill:.6f}"
    assert decode >= PCC_TARGET, f"[{case}] decode hidden PCC {decode:.6f}"
    assert semantic >= PCC_TARGET, f"[{case}] semantic logits PCC {semantic:.6f}"
    assert x_final >= PCC_TARGET, f"[{case}] acoustic x_final PCC {x_final:.6f}"
    assert peak <= 1.5, f"[{case}] waveform out of audio range: max |x| = {peak:.3f}"
    assert wave >= PCC_TARGET, f"[{case}] waveform PCC {wave:.6f} against the reference codec on the same codes"
    assert acoustic_agree >= MIN_ACOUSTIC_AGREEMENT, f"[{case}] acoustic code agreement {acoustic_agree:.4f}"
    assert semantic_agree >= MIN_SEMANTIC_AGREEMENT, f"[{case}] semantic code agreement {semantic_agree:.4f}"
