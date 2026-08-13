# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end performance cases across input sequence length (ISL) for XTTS-v2.

One pytest case per ISL. Each case opens/closes its own device (same as the demo per
take) — no shared pytest ``device`` fixture — so compiles do not stack L1 leftovers.

Path matches ``demo/xtts_demo.py``: text that fits one pass goes through
``inference_fully_traced``; text over the single-pass code budget is sentence-split
(``CHUNKING``) and replayed off one ``traced_session`` capture.

Single-pass opens with the ~50 MB one-shot trace region (like ``test_e2e_perf``);
chunked opens with ``SESSION_TRACE_REGION`` (all three traces live). Decode budget is
``chunk_max_tokens`` (192) — under the ~205-code vocoder wall (see ``ChunkingConfig``).

Run all ISLs:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/perf/test_e2e_isl_sweep_perf.py -v -s

Run one ISL:
    pytest models/experimental/xtts/tests/perf/test_e2e_isl_sweep_perf.py -v -s -k isl_96

Env:
  ``XTTS_ISL_SWEEP`` — comma list of text lengths to collect as cases
                       (e.g. ``XTTS_ISL_SWEEP=32,96``).
"""

import math
import os
import re

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

from models.experimental.xtts.config import (
    CHUNKING,
    DEMO,
    GENERATION,
    NUM_LATENTS,
    SENTENCE_FINAL_PUNCT_RE,
    SENTENCE_SPLIT_RE,
    SESSION_TRACE_REGION,
    TILE,
)

# Demo-validated envelope: MAX_TEXT_IDS 352, tile-aligned.
# Single-pass vs chunked (SWEEP_TEXT sentence ≈73 wrapped ids, CHUNKING.max_single_pass_codes=205):
#   ISL 32–64 → one pass (``inference_fully_traced``); ISL 96+ → sentence-chunked (``traced_session``).
DEFAULT_ISL_SWEEP = [32, 64, 96, 128, 192, 256, 320, 352]
LANG = DEMO.language

# One-shot path (setup released before decode, decode before vocoder) — same as test_e2e_perf.
ONESHOT_TRACE_REGION = 52_428_800

REF_CLIPS = ("LJ001-0001.wav", "LJ001-0003.wav", "LJ001-0004.wav", "LJ001-0005.wav")
SPK_SECONDS = DEMO.spk_seconds

SWEEP_TEXT = (
    "Voice synthesis has come a long way, and modern systems can already generate natural sounding "
    "speech with remarkable accuracy. "
) * 30


def _isl_sweep():
    raw = os.environ.get("XTTS_ISL_SWEEP")
    if not raw or not raw.strip():
        return list(DEFAULT_ISL_SWEEP)
    return [int(x) for x in raw.split(",") if x.strip()]


def _ids_of(text, lang=LANG):
    from models.experimental.xtts.reference.xtts_gpt_generate import wrap_text_ids
    from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text

    clean = re.sub(SENTENCE_FINAL_PUNCT_RE, "", text.strip())
    return wrap_text_ids(preprocess_text(clean, lang=lang)).shape[1]


def _text_for_isl(target_ids, lang=LANG):
    """Accumulate whole sentences until wrapped length reaches ``target_ids``."""
    sentences = [p.strip() for p in re.split(SENTENCE_SPLIT_RE, SWEEP_TEXT) if p.strip()]
    assert sentences, "SWEEP_TEXT produced no sentences"
    acc, i = [], 0
    while True:
        acc.append(sentences[i % len(sentences)])
        i += 1
        text = " ".join(acc)
        if _ids_of(text, lang) >= target_ids:
            return text


def _prepare_chunks(text, lang=LANG):
    """Demo-identical: sentence-split if needed, wrap, pad all chunks to a common tile length."""
    from models.experimental.xtts.demo.xtts_demo import _split_into_chunks
    from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
    from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text

    chunk_texts = _split_into_chunks(text, lang)
    wrapped_chunks = [
        (clean, wrap_text_ids(preprocess_text(clean, lang=lang)))
        for clean in (re.sub(SENTENCE_FINAL_PUNCT_RE, "", ct.strip()) for ct in chunk_texts)
    ]
    pad_to = -(-max(w.shape[1] for _, w in wrapped_chunks) // TILE) * TILE
    chunks = [(clean, F.pad(w, (0, pad_to - w.shape[1]), value=STOP_TEXT_TOKEN)) for clean, w in wrapped_chunks]
    return chunks, pad_to


def _run_single(tt, wrapped, cond_wav, spk_wav_tt, gen, budget):
    prompt_len = NUM_LATENTS + wrapped.shape[1]
    max_seq = -(-(prompt_len + budget + 2) // TILE) * TILE
    wav_dev, codes, perf = tt.inference_fully_traced(
        wrapped,
        cond_wav,
        spk_wav_tt,
        max_seq,
        max_new_tokens=budget,
        temperature=gen.temperature,
        top_k=gen.top_k,
        top_p=gen.top_p,
        repetition_penalty=gen.repetition_penalty,
        min_new_tokens=gen.min_tokens,
    )
    import ttnn
    from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE

    audio_s = ttnn.to_torch(wav_dev).float().numel() / OUTPUT_SAMPLE_RATE
    return {
        "n_chunks": 1,
        "pad_to": wrapped.shape[1],
        "prompt_len": prompt_len,
        "max_seq": max_seq,
        "codes": codes.shape[1],
        "audio_s": audio_s,
        "setup_s": perf["setup_replay_s"],
        "first_setup_s": perf["setup_replay_s"],
        "decode_s": perf["decode_replay_s"],
        "vocoder_s": perf["vocoder_replay_s"],
        "replay_s": perf["replay_s"],
        "compile_s": perf["compile_s"],
        "ttfa_s": perf["replay_s"],
    }


def _run_chunked(tt, chunks, cond_wav, spk_wav_tt, gen, budget):
    from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE

    text_len = chunks[0][1].shape[1]
    prompt_len = NUM_LATENTS + text_len
    max_seq = -(-(prompt_len + budget + 2) // TILE) * TILE
    session = tt.traced_session(
        cond_wav,
        spk_wav_tt,
        text_len,
        max_seq,
        budget,
        temperature=gen.temperature,
        top_k=gen.top_k,
        top_p=gen.top_p,
        repetition_penalty=gen.repetition_penalty,
        min_new_tokens=gen.min_tokens,
    )
    try:
        setup_s = decode_s = vocoder_s = replay_s = 0.0
        first_setup_s = None
        n_codes = 0
        audio_s = 0.0
        ttfa_s = None
        for j, (_, w) in enumerate(chunks):
            wav_t, codes_j, perf = session.run(w)
            setup_s += perf["setup_replay_s"]
            decode_s += perf["decode_replay_s"]
            vocoder_s += perf["vocoder_replay_s"]
            replay_s += perf["replay_s"]
            n_codes += codes_j.shape[1]
            audio_s += float(wav_t.numel()) / OUTPUT_SAMPLE_RATE
            if ttfa_s is None:
                ttfa_s = perf["replay_s"]
                first_setup_s = perf["setup_replay_s"]
            logger.info(
                f"  chunk {j + 1}/{len(chunks)}: setup {perf['setup_replay_s']:.3f}s | "
                f"decode {perf['decode_replay_s']:.3f}s ({codes_j.shape[1]} codes) | "
                f"vocoder {perf['vocoder_replay_s']:.3f}s"
            )
        return {
            "n_chunks": len(chunks),
            "pad_to": text_len,
            "prompt_len": prompt_len,
            "max_seq": max_seq,
            "codes": n_codes,
            "audio_s": audio_s,
            "setup_s": setup_s,
            "first_setup_s": first_setup_s,
            "decode_s": decode_s,
            "vocoder_s": vocoder_s,
            "replay_s": replay_s,
            "compile_s": session.compile_s,
            "ttfa_s": ttfa_s,
        }
    finally:
        session.close()


def _run_isl_case(text_len):
    """Open device, run one ISL, close device. Meant to run in a fresh process."""
    import random

    import numpy as np
    import ttnn
    from scipy.signal import resample_poly

    from models.experimental.xtts.reference.xtts_conditioning import GPT_COND_LEN_SEC, MEL_SR, load_coqui_test_audio
    from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
    from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN
    from models.experimental.xtts.reference.xtts_hifi_decoder import XttsHifiDecoderFull
    from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
    from models.experimental.xtts.tt.xtts_inference import TtXtts

    # Match conftest ``reset_seeds`` — parent fixture does not apply inside the spawn child.
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)

    assert text_len % TILE == 0, f"ISL {text_len} must be tile-aligned"

    text = _text_for_isl(text_len)
    chunks, pad_to = _prepare_chunks(text)
    chunked = len(chunks) > 1
    # 192 stays under the ~205-code vocoder CB/L1 wall with margin (ChunkingConfig).
    budget = CHUNKING.chunk_max_tokens
    trace_region = SESSION_TRACE_REGION if chunked else ONESHOT_TRACE_REGION
    mode = f"chunked ({len(chunks)} chunks)" if chunked else "single-pass"
    logger.info(
        f"ISL {text_len}: {mode} | single-pass through ISL 64; chunked from ISL 96 "
        f"(pad_to={pad_to}, budget={budget} codes, trace_region={trace_region})"
    )

    device = ttnn.open_device(
        device_id=DEMO.device_id,
        l1_small_size=DEMO.l1_small_size,
        trace_region_size=trace_region,
    )
    try:
        sd = load_xtts_state_dict()
        tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))
        gen = GENERATION

        wav = load_coqui_test_audio(samples=REF_CLIPS, max_seconds=GPT_COND_LEN_SEC)
        g = math.gcd(SPK_SR, MEL_SR)
        spk_src = wav[0].numpy()[: MEL_SR * SPK_SECONDS]
        spk_wav = torch.from_numpy(resample_poly(spk_src, SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)
        spk_wav_tt = ttnn.from_torch(
            spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
        )

        if not chunked:
            # Exact-ISL trim of a longer wrap (sentence ≈73 ids) drops the trailing [STOP].
            # Without it the sampler drones toward the code cap and the vocoder CB-clashes —
            # ISL 64 was the failure mode; the demo always keeps STOP (+ STOP pad) and passes.
            wrapped = chunks[0][1][:, :text_len].contiguous().clone()
            if wrapped.shape[1] < text_len:
                wrapped = F.pad(wrapped, (0, text_len - wrapped.shape[1]), value=STOP_TEXT_TOKEN)
            wrapped[:, -1] = STOP_TEXT_TOKEN
            result = _run_single(tt, wrapped, wav, spk_wav_tt, gen, budget)
        else:
            for i, (clean, w) in enumerate(chunks):
                logger.info(f"  [{i + 1}/{len(chunks)}] {w.shape[1]:3d} tokens  {clean!r}")
            result = _run_chunked(tt, chunks, wav, spk_wav_tt, gen, budget)
    finally:
        ttnn.close_device(device)

    n_codes = result["codes"]
    assert n_codes > 0, f"ISL {text_len} generated no codes"
    ms_per_code = result["decode_s"] / n_codes * 1000
    result.update(
        {
            "isl": text_len,
            "ms_per_code": ms_per_code,
            "codes_per_s": n_codes / result["decode_s"],
            "ttft_s": result["first_setup_s"] + ms_per_code / 1000,
            "rtf": result["replay_s"] / result["audio_s"],
        }
    )
    return result


@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("text_len", _isl_sweep(), ids=lambda n: f"isl_{n}")
def test_xtts_e2e_isl_perf(text_len, reset_seeds):
    """One ISL with its own open_device/close_device (no shared device fixture)."""
    r = _run_isl_case(text_len)
    logger.info(
        f"ISL={r['isl']:>3} chunks={r['n_chunks']} pad_to={r['pad_to']:>3} "
        f"prompt={r['prompt_len']:>4} max_seq={r['max_seq']:>4} codes={r['codes']:>4} "
        f"audio={r['audio_s']:.2f}s | TTFT={r['ttft_s'] * 1000:.1f}ms "
        f"decode={r['ms_per_code']:.3f}ms/code ({r['codes_per_s']:.1f} codes/s) "
        f"setup={r['setup_s'] * 1000:.1f}ms voc={r['vocoder_s'] * 1000:.2f}ms "
        f"replay={r['replay_s']:.3f}s RTF={r['rtf']:.3f} compile={r['compile_s']:.1f}s"
    )
