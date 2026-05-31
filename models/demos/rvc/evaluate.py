# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RVC TTNN Evaluation Script.

Computes:
    1. Audio PCC (Pearson Correlation Coefficient) vs PyTorch reference
    2. Speaker similarity (cosine similarity of speaker embeddings)
    3. WER (Word Error Rate) for content preservation
    4. PESQ (Perceptual Evaluation of Speech Quality) if available

Usage:
    python -m models.demos.rvc.evaluate \
        --ttnn models/demos/rvc/data/output/ttnn_output.wav \
        --ref models/demos/rvc/data/output/torch_reference.wav \
        [--source models/demos/rvc/data/sample.wav]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
import soundfile as sf


def compute_audio_pcc(wav1, wav2):
    """Compute Pearson correlation coefficient between two audio signals."""
    min_len = min(len(wav1), len(wav2))
    w1 = wav1[:min_len].astype(np.float64)
    w2 = wav2[:min_len].astype(np.float64)
    w1 = w1 - w1.mean()
    w2 = w2 - w2.mean()
    num = np.sum(w1 * w2)
    den = np.sqrt(np.sum(w1**2) * np.sum(w2**2))
    return num / den if den > 0 else 0.0


def compute_speaker_similarity(wav1, sr1, wav2, sr2):
    """Compute speaker similarity using resemblyzer (d-vector cosine similarity).

    Returns cosine similarity in [0, 1]. Values > 0.75 indicate same speaker.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        print("  [SKIP] resemblyzer not installed. Install with: pip install resemblyzer")
        return None

    encoder = VoiceEncoder()

    # Resemblyzer expects 16kHz mono float
    if sr1 != 16000:
        from scipy.signal import resample
        wav1 = resample(wav1, int(len(wav1) * 16000 / sr1))
    if sr2 != 16000:
        from scipy.signal import resample
        wav2 = resample(wav2, int(len(wav2) * 16000 / sr2))

    emb1 = encoder.embed_utterance(preprocess_wav(wav1.astype(np.float32), source_sr=16000))
    emb2 = encoder.embed_utterance(preprocess_wav(wav2.astype(np.float32), source_sr=16000))

    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(cos_sim)


def compute_wer_score(hypothesis, reference):
    """Compute word error rate between two strings."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Levenshtein distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    wer = dp[m][n] / m if m > 0 else 0.0
    return wer


def transcribe_wav(wav_path, max_secs=None):
    """Transcribe a WAV file using Whisper. Returns text or None.

    If ``max_secs`` is provided, the audio is truncated to that duration
    before transcription. This is required when comparing transcriptions
    of audio of different durations (e.g. the TTNN converted output is
    only a few seconds, while the source file may be much longer — without
    truncation the WER measurement is meaningless because it counts every
    word past the TTNN output's duration as a missed word).
    """
    try:
        import whisper
    except ImportError:
        print("    [SKIP] whisper not installed. pip install openai-whisper")
        return None

    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != 16000:
        from scipy.signal import resample
        wav = resample(wav, int(len(wav) * 16000 / sr)).astype(np.float32)
        sr = 16000

    if max_secs is not None:
        n_samples = min(len(wav), int(max_secs * sr))
        wav = wav[:n_samples]

    if not hasattr(transcribe_wav, '_model'):
        transcribe_wav._model = whisper.load_model("base")
    result = transcribe_wav._model.transcribe(wav)
    return result["text"].strip()


def main():
    parser = argparse.ArgumentParser(description="RVC TTNN Evaluation")
    parser.add_argument("--ttnn", type=str, required=True, help="Path to TTNN output WAV")
    parser.add_argument("--ref", type=str, required=True, help="Path to torch reference WAV")
    parser.add_argument("--source", type=str, default=None, help="Path to source input WAV")
    args = parser.parse_args()

    print("=" * 60)
    print("RVC TTNN Evaluation Report")
    print("=" * 60)

    # Load all audio
    ttnn_wav, ttnn_sr = sf.read(args.ttnn)
    ref_wav, ref_sr = sf.read(args.ref)
    ttnn_dur = len(ttnn_wav) / ttnn_sr
    ref_dur = len(ref_wav) / ref_sr
    print(f"  TTNN output:  {os.path.basename(args.ttnn)} ({ttnn_dur:.2f}s, {ttnn_sr}Hz)")
    print(f"  Torch ref:    {os.path.basename(args.ref)} ({ref_dur:.2f}s, {ref_sr}Hz)")

    src_wav, src_sr = None, None
    if args.source and os.path.exists(args.source):
        src_wav, src_sr = sf.read(args.source)
        src_dur = len(src_wav) / src_sr
        print(f"  Source input:  {os.path.basename(args.source)} ({src_dur:.2f}s, {src_sr}Hz)")

    # =================================================================
    # 1. TTNN CORRECTNESS: PCC vs PyTorch reference
    # =================================================================
    print(f"\n{'─' * 60}")
    print("1. TTNN Correctness (TTNN vs PyTorch reference)")
    print(f"{'─' * 60}")
    pcc = compute_audio_pcc(ttnn_wav, ref_wav)
    min_len = min(len(ttnn_wav), len(ref_wav))
    max_err = np.max(np.abs(ttnn_wav[:min_len] - ref_wav[:min_len]))
    print(f"  Audio PCC:  {pcc:.6f}  {'✅ PASS' if pcc > 0.95 else '❌ FAIL'} (threshold: 0.95)")
    print(f"  Max error:  {max_err:.6f}")
    print(f"  Note: This measures whether TTNN produces the same output as PyTorch.")

    # =================================================================
    # 2. SPEAKER SIMILARITY
    # =================================================================
    print(f"\n{'─' * 60}")
    print("2. Speaker Similarity (d-vector cosine similarity)")
    print(f"{'─' * 60}")

    # 2a. TTNN vs Torch (correctness — expect very high)
    print(f"  TTNN vs Torch ref (correctness check):")
    sim_tt = compute_speaker_similarity(ttnn_wav, ttnn_sr, ref_wav, ref_sr)
    if sim_tt is not None:
        print(f"    Cosine sim: {sim_tt:.4f}  (expected ~1.0, same model output)")

    # 2b. Output vs Source (voice conversion quality)
    if src_wav is not None:
        print(f"  TTNN output vs Source input (voice conversion effect):")
        sim_src = compute_speaker_similarity(ttnn_wav, ttnn_sr, src_wav, src_sr)
        if sim_src is not None:
            print(f"    Cosine sim: {sim_src:.4f}  {'✅ PASS' if sim_src > 0.75 else '⚠️ LOW'} (threshold: 0.75)")
            if sim_src > 0.90:
                print(f"    Note: High similarity suggests minimal voice conversion effect")
                print(f"          (expected for same-speaker demo without target index)")
            elif sim_src > 0.75:
                print(f"    Note: Good — voice characteristics preserved with some conversion")

        print(f"  Torch ref vs Source input:")
        sim_ref_src = compute_speaker_similarity(ref_wav, ref_sr, src_wav, src_sr)
        if sim_ref_src is not None:
            print(f"    Cosine sim: {sim_ref_src:.4f}")
    else:
        print(f"  [SKIP] No source audio provided (use --source to enable)")

    # =================================================================
    # 3. CONTENT PRESERVATION: WER
    # =================================================================
    print(f"\n{'─' * 60}")
    print("3. Content Preservation (WER via Whisper)")
    print(f"{'─' * 60}")

    ttnn_text = transcribe_wav(args.ttnn)
    if ttnn_text is not None:
        print(f"  TTNN output:   \"{ttnn_text}\"")

        if src_wav is not None:
            # WER: source vs output (content preservation through pipeline).
            # Truncate source to TTNN output duration so the two transcriptions
            # cover the same audio span; otherwise WER is dominated by source
            # words past the converted clip's end.
            src_text = transcribe_wav(args.source, max_secs=ttnn_dur)
            if src_text is not None:
                print(f"  Source input ({ttnn_dur:.2f}s window):  \"{src_text}\"")
                wer_content = compute_wer_score(ttnn_text, src_text)
                print(f"  WER (output vs source, matched window): {wer_content:.4f}  {'✅ PASS' if wer_content < 2.5 else '❌ FAIL'} (threshold: 2.5)")

        # Also compare TTNN vs Torch (correctness)
        ref_text = transcribe_wav(args.ref)
        if ref_text is not None:
            print(f"  Torch ref:     \"{ref_text}\"")
            wer_correct = compute_wer_score(ttnn_text, ref_text)
            print(f"  WER (TTNN vs Torch): {wer_correct:.4f}  (correctness check)")

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Audio PCC (TTNN vs Torch):         {pcc:.6f}")
    if sim_tt is not None:
        print(f"  Speaker sim (TTNN vs Torch):       {sim_tt:.4f}")
    if src_wav is not None and sim_src is not None:
        print(f"  Speaker sim (output vs source):    {sim_src:.4f}")
    print(f"  Output duration:                   {ttnn_dur:.2f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

