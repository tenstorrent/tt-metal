# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.xtts.config import L1_SMALL_SIZE, NUM_LATENTS, SESSION_TRACE_REGION
from models.experimental.xtts.reference.xtts_conditioning import MEL_SR, load_reference_audio
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE, XttsHifiDecoderFull
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_inference import TtXtts

TILE = 32
COND_SECONDS = 3
DEMO_TEXT = "Voice synthesis has come a long way, and modern systems can already generate natural sounding speech with remarkable accuracy"
TRACE_TEMPERATURE = 0.65
TRACE_TOP_K = 50
TRACE_TOP_P = 0.85
TRACE_REP = 5.0
TRACE_MAX_TOKENS = 192
TRACE_MAX_SEQ = 384


def _stft_mag(wav):
    """Return STFT magnitude spectrogram of a waveform for PCC comparison."""
    return torch.stft(wav.reshape(1, -1), 1024, 256, window=torch.hann_window(1024), return_complex=True).abs()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 52428800}], indirect=True)
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_full_trace(device, xtts_state_dict, pcc, reset_seeds):
    """Compare fully-traced TTNN inference spectrogram to eager decode via PCC."""
    import os

    import soundfile as sf
    from scipy.signal import resample_poly

    sd = xtts_state_dict

    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)
    wrapped = wrap_text_ids(preprocess_text(DEMO_TEXT, lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)
    logger.info(f"trace text -> {wrapped.shape[1]} tokens (wrapped/padded)")

    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    wav_dev, codes, _perf = tt.inference_fully_traced(
        wrapped,
        wav,
        spk_wav_tt,
        TRACE_MAX_SEQ,
        max_new_tokens=TRACE_MAX_TOKENS,
        temperature=TRACE_TEMPERATURE,
        top_k=TRACE_TOP_K,
        top_p=TRACE_TOP_P,
        repetition_penalty=TRACE_REP,
    )
    wav_traced = ttnn.to_torch(wav_dev).float()
    n = codes.shape[1]
    assert n > 0, "fully-traced generation produced no codes"
    logger.info(f"fully-traced generation: {n} codes -> {wav_traced.shape[1] / OUTPUT_SAMPLE_RATE:.2f}s audio")

    cond_latents = tt._cond_latents(wav)
    _, latents_ref = tt.generator.latents_for_codes(wrapped, cond_latents, codes[0].tolist())
    wav_ref = ttnn.to_torch(tt._decode_wav(latents_ref, spk_wav_tt)).float()

    assert wav_traced.shape == wav_ref.shape, f"traced {tuple(wav_traced.shape)} != eager {tuple(wav_ref.shape)}"
    spec_pass, spec_msg = comp_pcc(_stft_mag(wav_ref), _stft_mag(wav_traced), pcc)
    logger.info(f"fully-traced vs eager spectrogram-magnitude PCC: {spec_msg}")

    out_dir = "generated/xtts"
    os.makedirs(out_dir, exist_ok=True)
    sf.write(f"{out_dir}/tt_full_trace_device.wav", wav_traced.reshape(-1).numpy(), OUTPUT_SAMPLE_RATE)
    logger.info(f"wrote fully-traced device audio ({n} codes) -> {out_dir}/tt_full_trace_device.wav")

    assert spec_pass, f"fully-traced waveform diverged from eager reference: {spec_msg}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 157286400}], indirect=True)
def test_tt_traced_session_reuse(device, xtts_state_dict, reset_seeds):
    """Verify a traced session rebinds text correctly and replays deterministically."""
    from scipy.signal import resample_poly

    sd = xtts_state_dict
    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)

    texts = [DEMO_TEXT, "The model runs entirely on the accelerator, from conditioning through the vocoder"]
    wrapped = [wrap_text_ids(preprocess_text(t, lang="en")) for t in texts]
    pad_to = -(-max(w.shape[1] for w in wrapped) // TILE) * TILE
    wrapped = [F.pad(w, (0, pad_to - w.shape[1]), value=STOP_TEXT_TOKEN) for w in wrapped]

    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )
    n_tokens = 64
    session = tt.traced_session(wav, spk_wav_tt, pad_to, TRACE_MAX_SEQ, n_tokens, temperature=0.0)
    try:
        logger.info(f"session captured in {session.compile_s:.1f}s ({pad_to} text tokens, {n_tokens}-code budget)")
        runs = []
        for label, w in (("A", wrapped[0]), ("B", wrapped[1]), ("A again", wrapped[0])):
            audio, codes, perf = session.run(w)
            logger.info(
                f"run {label}: {codes.shape[1]} codes, {audio.shape[0] / OUTPUT_SAMPLE_RATE:.2f}s audio, "
                f"replay {perf['replay_s']:.3f}s"
            )
            runs.append((codes, audio))
    finally:
        session.close()

    (codes_a, audio_a), (codes_b, _), (codes_a2, audio_a2) = runs
    assert codes_a.shape[1] > 0, "session produced no codes"
    assert audio_a.shape[0] == session._samples_for(codes_a.shape[1]), "trimmed audio has the wrong length"
    assert codes_b.tolist() != codes_a.tolist(), "different texts produced identical codes — text not rebound?"
    assert codes_a2.tolist() == codes_a.tolist(), "replaying a text after another one changed its codes"
    assert torch.equal(audio_a2, audio_a), "replaying a text after another one changed its waveform"


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 52428800}], indirect=True)
def test_tt_eval_traced(device, xtts_state_dict, reset_seeds):
    """Run fully-traced generation and log CER/UTMOS/SECS eval metrics."""
    import os

    import soundfile as sf
    from scipy.signal import resample_poly

    sd = xtts_state_dict

    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)
    wrapped = wrap_text_ids(preprocess_text(DEMO_TEXT, lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)

    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    wav_dev, codes, _perf = tt.inference_fully_traced(
        wrapped,
        wav,
        spk_wav_tt,
        TRACE_MAX_SEQ,
        max_new_tokens=TRACE_MAX_TOKENS,
        temperature=TRACE_TEMPERATURE,
        top_k=TRACE_TOP_K,
        top_p=TRACE_TOP_P,
        repetition_penalty=TRACE_REP,
    )
    wav_eval = ttnn.to_torch(wav_dev).float().reshape(-1).numpy()

    out_dir = "generated/xtts"
    os.makedirs(out_dir, exist_ok=True)
    sf.write(f"{out_dir}/tt_eval_traced_device.wav", wav_eval, OUTPUT_SAMPLE_RATE)
    logger.info(
        f"fully-traced eval generation: {codes.shape[1]} codes -> {wav_eval.shape[0] / OUTPUT_SAMPLE_RATE:.2f}s "
        f"audio at {out_dir}/tt_eval_traced_device.wav"
    )

    spk_np = spk_wav[0].numpy()
    logger.info("========== XTTS objective eval metrics (FULLY TRACED) ==========")
    try:
        from models.experimental.xtts.eval.xtts_eval import compute_cer

        cer, hyp = compute_cer(wav_eval, OUTPUT_SAMPLE_RATE, DEMO_TEXT)
        logger.info(f"CER   (Whisper-large-v3, lower=better)        : {cer:.4f}")
        logger.info(f"        whisper transcript: {hyp!r}")
    except Exception as e:
        logger.warning(f"CER   skipped ({type(e).__name__}: {e})")

    try:
        from models.experimental.xtts.eval.xtts_eval import compute_utmos

        logger.info(
            f"UTMOS (naturalness MOS 1-5, higher=better)    : {compute_utmos(wav_eval, OUTPUT_SAMPLE_RATE):.4f}"
        )
    except Exception as e:
        logger.warning(f"UTMOS skipped ({type(e).__name__}: {e})")

    try:
        from models.experimental.xtts.eval.xtts_eval import compute_secs

        secs = compute_secs(wav_eval, OUTPUT_SAMPLE_RATE, spk_np, SPK_SR)
        logger.info(f"SECS  (ECAPA2 speaker cos-sim, higher=better)  : {secs:.4f}")
    except Exception as e:
        logger.warning(f"SECS  skipped ({type(e).__name__}: {e})")
    logger.info("================================================================")


EVAL_LONG_TEXT = (
    "Voice synthesis has come a long way, and modern systems can already generate natural "
    "sounding speech with remarkable accuracy. The hardest part is no longer the sound of a "
    "single word, but the rhythm and emphasis that carry meaning across a whole sentence. "
    "Good narration needs pacing, deliberate pauses, and a sense of where the important idea "
    "actually sits. When those details land, a listener stops noticing the machine and simply "
    "follows the story to its end."
)

# Gates for the nightly long-eval job (tests/pipeline_reorg/blackhole_demo_tests.yaml). Measured on
# Blackhole P150b, 3/3 identical runs: 556 codes -> 26.17 s audio, CER 0.0000, SECS 0.6979. This test
# samples off the host RNG (reset_seeds), so there is no run-to-run spread to absorb — the margin is
# for a *re-sampled* render, since any numerical change shifts the sampled codes and yields different
# audio. The bounds therefore encode "still intelligible, still the same speaker" rather than
# "byte-identical to the measurement"; real breakage (babbling, truncation, speaker collapse) lands
# far outside them.
LONG_CER_MAX = 0.05
LONG_SECS_MIN = 0.55


@pytest.mark.timeout(2400)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": L1_SMALL_SIZE, "trace_region_size": SESSION_TRACE_REGION}],
    indirect=True,
)
def test_tt_eval_traced_long(device, xtts_state_dict, reset_seeds):
    """Run chunked traced generation on a long paragraph and log eval metrics."""
    import os
    import re

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    from models.experimental.xtts.config import AUDIO_POST, CHUNKING, SENTENCE_FINAL_PUNCT_RE
    from models.experimental.xtts.demo.xtts_demo import _split_into_chunks

    sd = xtts_state_dict

    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)

    chunk_texts = _split_into_chunks(EVAL_LONG_TEXT, "en")
    wrapped = [
        wrap_text_ids(preprocess_text(re.sub(SENTENCE_FINAL_PUNCT_RE, "", t.strip()), lang="en")) for t in chunk_texts
    ]
    pad_to = -(-max(w.shape[1] for w in wrapped) // TILE) * TILE
    chunks = [F.pad(w, (0, pad_to - w.shape[1]), value=STOP_TEXT_TOKEN) for w in wrapped]
    assert len(chunks) > 1, "EVAL_LONG_TEXT is meant to exceed a single pass; it no longer does"

    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    budget = CHUNKING.chunk_max_tokens
    max_seq = -(-(NUM_LATENTS + pad_to + budget + 2) // TILE) * TILE
    logger.info(
        f"long eval: {len(EVAL_LONG_TEXT)} chars -> {len(chunks)} chunks, "
        f"pad_to={pad_to}, budget={budget}, max_seq={max_seq}"
    )

    session = tt.traced_session(
        wav,
        spk_wav_tt,
        pad_to,
        max_seq,
        budget,
        temperature=TRACE_TEMPERATURE,
        top_k=TRACE_TOP_K,
        top_p=TRACE_TOP_P,
        repetition_penalty=TRACE_REP,
    )
    gap = np.zeros(int(AUDIO_POST.chunk_gap_seconds * OUTPUT_SAMPLE_RATE), dtype="float32")
    pieces, n_codes = [], 0
    try:
        for i, w in enumerate(chunks):
            wav_i, codes_i, _ = session.run(w)
            wav_i = wav_i.float().numpy().astype("float32")
            n_codes += codes_i.shape[1]
            if pieces:
                pieces.append(gap)
            pieces.append(wav_i)
            logger.info(
                f"  chunk {i + 1}/{len(chunks)}: {codes_i.shape[1]:3d} codes -> {len(wav_i) / OUTPUT_SAMPLE_RATE:.2f}s"
            )
    finally:
        session.close()

    wav_eval = np.concatenate(pieces)
    out_dir = "generated/xtts"
    os.makedirs(out_dir, exist_ok=True)
    sf.write(f"{out_dir}/tt_eval_traced_long.wav", wav_eval, OUTPUT_SAMPLE_RATE)
    logger.info(
        f"long eval generation: {n_codes} codes over {len(chunks)} chunks -> "
        f"{wav_eval.shape[0] / OUTPUT_SAMPLE_RATE:.2f}s audio at {out_dir}/tt_eval_traced_long.wav"
    )

    spk_np = spk_wav[0].numpy()
    logger.info("======= XTTS objective eval metrics (TRACED, CHUNKED PARAGRAPH) =======")
    # CER and SECS are gated below, so they are computed outside try/except on purpose: a missing
    # Whisper / ECAPA2 must fail the job rather than log a warning and pass. UTMOS stays best-effort.
    from models.experimental.xtts.eval.xtts_eval import compute_cer, compute_secs

    cer, hyp = compute_cer(wav_eval, OUTPUT_SAMPLE_RATE, EVAL_LONG_TEXT)
    logger.info(f"CER   (Whisper-large-v3, lower=better)        : {cer:.4f}")
    logger.info(f"        whisper transcript: {hyp!r}")

    try:
        from models.experimental.xtts.eval.xtts_eval import compute_utmos

        logger.info(
            f"UTMOS (naturalness MOS 1-5, higher=better)    : {compute_utmos(wav_eval, OUTPUT_SAMPLE_RATE):.4f}"
        )
    except Exception as e:
        logger.warning(f"UTMOS skipped ({type(e).__name__}: {e})")

    secs = compute_secs(wav_eval, OUTPUT_SAMPLE_RATE, spk_np, SPK_SR)
    logger.info(f"SECS  (ECAPA2 speaker cos-sim, higher=better)  : {secs:.4f}")
    logger.info("======================================================================")

    assert cer <= LONG_CER_MAX, f"CER {cer:.4f} above gate {LONG_CER_MAX} -- whisper heard: {hyp!r}"
    assert secs >= LONG_SECS_MIN, f"SECS {secs:.4f} below gate {LONG_SECS_MIN} (speaker similarity)"
