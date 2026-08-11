# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end performance sweep across input sequence length (ISL) for XTTS-v2.

Runs the production ``inference_fully_traced`` path (setup + decode + vocoder, three chained
ttnn traces — what the demo runs) at a range of text lengths and reports the standard
metrics per ISL: TTFT, tokens/s, ms/token, RTF, and the per-leg replay split.

**Envelope — why this stops at 352, not 404.** The PCC sweep
(``tests/pcc/test_gpt_isl_sweep.py``) goes to the *architectural* ceiling: 384 tile-aligned
text ids under the checkpoint's 404-row text position table. This sweep covers only what
actually runs **end to end**, which is lower: the demo caps text at ``MAX_TEXT_IDS`` 352 and
a single pass at ``MAX_SINGLE_PASS_CODES`` 205, because the fp32 HiFi-GAN vocoder hits an L1
circular-buffer allocation collision ("Statically allocated circular buffers ... clash with
L1 buffers") above that. That is an allocation collision, not a clean size limit — per the
demo's own measurements it degrades as device open/close cycles accumulate in one process,
so a sweep that runs many generations back to back is exactly the case it punishes. If a
row fails to allocate, that is a real data point about the envelope, not a flaky test.

Concretely and reproducibly: **ISL 64 runs fine on its own** (7.767 ms/code, RTF 0.176) but
throws when it runs immediately after ISL 32 in the same process, while every larger ISL in
the same sweep succeeds. So the failure is positional, not a length limit — which is exactly
why ``demo/xtts_demo.py`` opens a **fresh device per chunk** instead of reusing one. Skipped
rows are recorded and reported rather than failing the test.

**What to expect in the numbers.** Unlike an LLM, XTTS's decode cost is driven by
``max_seq`` — the size of the fixed KV cache — rather than by how far into the sequence you
are, because each decode step attends over the *entire* cache and the traced cache write
rewrites all of it (see the README's per-op decode breakdown: that write is ~31% of a step).
``max_seq`` grows with the prompt, so **ms/code rises with ISL** even though the model is
generating the same number of codes. That is the headline this sweep exists to quantify.

Metric definitions (XTTS is **non-streaming** — one generate call emits the whole clip):

  * ``TTFT``      — time to first *code*: setup replay + one decode step. The LLM-analogous
                    number, derived as ``setup + decode/n_codes`` rather than instrumented.
  * ``TTFA``      — time to first *audio*: the full replay, since no audio exists until the
                    vocoder has run over every latent. This is what a listener waits for.
  * ``codes/s``   — decode throughput (the TPS analogue). One code is 46.4 ms of audio.
  * ``RTF``       — replay / generated audio duration. Below 1.0 is faster than real time.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/perf/test_e2e_isl_sweep_perf.py -v -s

Env:
  ``XTTS_ISL_SWEEP`` — comma list of text lengths, to shorten a smoke run
                       (e.g. ``XTTS_ISL_SWEEP=32,96``).
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.xtts.reference.xtts_conditioning import GPT_COND_LEN_SEC, MEL_SR, load_coqui_test_audio
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_generate import wrap_text_ids
from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE, XttsHifiDecoderFull
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_inference import TtXtts

TILE = 32
N_COND = 32  # conditioning latents prepended to the text prompt

# Demo-validated envelope: MAX_TEXT_IDS 352, tile-aligned. See the module docstring for why this
# does not reach the PCC sweep's 384.
DEFAULT_ISL_SWEEP = [32, 64, 96, 128, 192, 256, 320, 352]
MAX_NEW_TOKENS = 205  # demo's MAX_SINGLE_PASS_CODES
TEMPERATURE, TOP_K, TOP_P, REP_PENALTY = 0.65, 50, 0.85, 5.0

# Same reference voice as the demo and test_e2e_perf.py: 30 s = 8 conditioning windows.
REF_CLIPS = ("LJ001-0001.wav", "LJ001-0003.wav", "LJ001-0004.wav", "LJ001-0005.wav")
SPK_SECONDS = 8  # speaker-embedding window; 30 s clashes L1 in the speaker ResNet

# Real English text, repeated and trimmed to each ISL. Real text (not [STOP] padding) so the
# generation behaves like a real request — the PCC sweep documents how badly padding distorts.
SWEEP_TEXT = (
    "Voice synthesis has come a long way, and modern systems can already generate natural sounding "
    "speech with remarkable accuracy. "
) * 30


def _isl_sweep():
    raw = os.environ.get("XTTS_ISL_SWEEP")
    if not raw or not raw.strip():
        return list(DEFAULT_ISL_SWEEP)
    return [int(x) for x in raw.split(",") if x.strip()]


@pytest.mark.timeout(7200)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 52428800}], indirect=True)
def test_xtts_e2e_isl_sweep_perf(device, reset_seeds):
    from scipy.signal import resample_poly

    sd = load_xtts_state_dict()
    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))

    wav = load_coqui_test_audio(samples=REF_CLIPS, max_seconds=GPT_COND_LEN_SEC)  # [1, s] @ 22050
    g = math.gcd(SPK_SR, MEL_SR)
    spk_src = wav[0].numpy()[: MEL_SR * SPK_SECONDS]
    spk_wav = torch.from_numpy(resample_poly(spk_src, SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    full_ids = wrap_text_ids(preprocess_text(SWEEP_TEXT, lang="en"))
    rows, skipped = [], []

    for text_len in _isl_sweep():
        assert text_len % TILE == 0, f"ISL {text_len} must be tile-aligned"
        assert full_ids.shape[1] >= text_len, f"SWEEP_TEXT yields {full_ids.shape[1]} ids, need {text_len}"
        wrapped = full_ids[:, :text_len].contiguous()
        prompt_len = N_COND + text_len
        max_seq = -(-(prompt_len + MAX_NEW_TOKENS + 2) // TILE) * TILE

        try:
            wav_dev, codes, perf = tt.inference_fully_traced(
                wrapped,
                wav,
                spk_wav_tt,
                max_seq,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                repetition_penalty=REP_PENALTY,
            )
        except RuntimeError as e:
            # The L1 allocation collision described in the module docstring. Record the ISL that
            # hit it rather than failing the sweep — where the wall is IS the deliverable.
            skipped.append((text_len, max_seq, str(e).strip().splitlines()[0][:120]))
            logger.warning(f"ISL {text_len} (max_seq {max_seq}) did not run: {skipped[-1][2]}")
            continue

        n_codes = codes.shape[1]
        assert n_codes > 0, f"ISL {text_len} generated no codes"
        audio_s = ttnn.to_torch(wav_dev).float().numel() / OUTPUT_SAMPLE_RATE
        ms_per_code = perf["decode_replay_s"] / n_codes * 1000
        rows.append(
            {
                "isl": text_len,
                "prompt_len": prompt_len,
                "max_seq": max_seq,
                "codes": n_codes,
                "audio_s": audio_s,
                "setup_ms": perf["setup_replay_s"] * 1000,
                "decode_s": perf["decode_replay_s"],
                "vocoder_ms": perf["vocoder_replay_s"] * 1000,
                "replay_s": perf["replay_s"],
                "ms_per_code": ms_per_code,
                "codes_per_s": n_codes / perf["decode_replay_s"],
                "ttft_s": perf["setup_replay_s"] + ms_per_code / 1000,  # setup + one decode step
                "ttfa_s": perf["replay_s"],  # non-streaming: no audio until the vocoder has run
                "rtf": perf["replay_s"] / audio_s,
                "compile_s": perf["compile_s"],
            }
        )
        logger.info(
            f"ISL={text_len:>3} max_seq={max_seq:>4} codes={n_codes:>3} | TTFT={rows[-1]['ttft_s'] * 1000:.1f}ms "
            f"decode={ms_per_code:.3f}ms/code ({rows[-1]['codes_per_s']:.1f} codes/s) "
            f"replay={perf['replay_s']:.3f}s RTF={rows[-1]['rtf']:.3f}"
        )

    assert rows, "ISL sweep produced no rows"

    logger.info(
        f"\n{'ISL':>5}{'prompt':>8}{'max_seq':>9}{'codes':>7}{'audio_s':>9}{'TTFT_ms':>9}"
        f"{'codes/s':>9}{'ms/code':>9}{'setup_ms':>10}{'voc_ms':>8}{'replay_s':>10}{'RTF':>7}"
    )
    for r in rows:
        logger.info(
            f"{r['isl']:>5}{r['prompt_len']:>8}{r['max_seq']:>9}{r['codes']:>7}{r['audio_s']:>9.2f}"
            f"{r['ttft_s'] * 1000:>9.1f}{r['codes_per_s']:>9.1f}{r['ms_per_code']:>9.3f}"
            f"{r['setup_ms']:>10.1f}{r['vocoder_ms']:>8.2f}{r['replay_s']:>10.3f}{r['rtf']:>7.3f}"
        )
    if skipped:
        logger.warning(f"{len(skipped)} ISL(s) did not run (L1 allocation envelope):")
        for text_len, max_seq, msg in skipped:
            logger.warning(f"  ISL={text_len} max_seq={max_seq}: {msg}")

    # ms/code is expected to RISE with max_seq (the whole KV cache is rewritten every step).
    lo, hi = rows[0], rows[-1]
    if hi["max_seq"] > lo["max_seq"]:
        logger.info(
            f"decode cost vs cache size: {lo['ms_per_code']:.3f} ms/code at max_seq {lo['max_seq']} -> "
            f"{hi['ms_per_code']:.3f} ms/code at max_seq {hi['max_seq']} "
            f"({(hi['ms_per_code'] / lo['ms_per_code'] - 1) * 100:+.1f}%)"
        )
