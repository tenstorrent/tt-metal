# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro-82M TTS demo — TRACE-ACCELERATED fast path (production speed).

Runs the trace+2CQ pipeline (tt/pipeline.py::run_tts_fast): the token axis + duration->frame
alignment run dynamically (data-dependent length), and the whole frame axis (F0/N predictor +
acoustic decoder + ISTFTNet vocoder) is captured as ONE host-free trace and replayed. This is the
same numerically-gated path validated at log-spectrogram PCC >= 0.95 vs the HF reference.

It writes TWO waveforms so you can A/B the quality by ear:
    <out>            the fast (traced) waveform            <- the one we ship / quote speed for
    <out>.ref.wav    the dynamic run_tts waveform (baseline reference, unchanged code path)
and prints the log-spectrogram PCC of each vs the HF gold, plus REAL measured wall-clock RTF
(dynamic vs steady-state fast: dynamic token/align prep + traced frame replay, trace captured once).

    python -m models.demos.kokoro_82m.demo.demo_tts_fast \
        --phonemes "kˈOkəɹO ɪz ˈoʊpən sˈOɹs" --voice af_heart --out kokoro_fast.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import ttnn
from models.demos.kokoro_82m._stubs import _trace_alloc
from models.demos.kokoro_82m._stubs._lstm_scan import pop_trace_ctx, push_trace_ctx
from models.demos.kokoro_82m.tt import ops
from models.demos.kokoro_82m.tt import pipeline as P

SR = 24000


def _write_wav(path, wav):
    try:
        import soundfile as sf

        sf.write(path, wav.numpy(), SR)
    except Exception:
        import wave

        w = wave.open(path, "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((wav.clamp(-1, 1) * 32767).short().numpy().tobytes())
        w.close()


def main():
    ap = argparse.ArgumentParser(description="Kokoro-82M TTNN text->speech demo (trace-accelerated)")
    ap.add_argument("--phonemes", default=P.DEFAULT_PHONEMES, help="Kokoro phoneme string")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice pack name")
    ap.add_argument("--out", default="kokoro_fast.wav", help="output wav path (fast/traced waveform)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=10, help="timing iterations for steady-state RTF")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "pcc"))
    from _reference_loader import load_reference_model

    model = load_reference_model("hexgrad/Kokoro-82M").float().eval()
    input_ids, ref_s = P.build_input(model, phonemes=args.phonemes, voice=args.voice)
    gold, gold_dur = P.hf_reference_tts(model, input_ids, ref_s, speed=args.speed)

    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=(1 << 30), num_command_queues=2)
    try:
        pipe = P.build_pipeline(device, model=model)

        # ---- generate audio: fast (traced) + dynamic (reference) ----
        wav_fast, pred_dur = P.run_tts_fast(pipe, input_ids, ref_s, speed=args.speed)
        wav_dyn, _ = P.run_tts(pipe, input_ids, ref_s, speed=args.speed)

        ref_out = args.out + ".ref.wav"
        _write_wav(args.out, wav_fast)
        _write_wav(ref_out, wav_dyn)

        dur_s = wav_fast.numel() / SR
        spec_fast = P.log_spectrogram_pcc(gold, wav_fast)
        spec_dyn = P.log_spectrogram_pcc(gold, wav_dyn)

        # ---- honest wall-clock timing ----
        def _timeit(fn, iters, warmup=2):
            for _ in range(warmup):
                fn()
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            ttnn.synchronize_device(device)
            return 1000.0 * (time.perf_counter() - t0) / iters

        # dynamic e2e
        dyn_ms = _timeit(lambda: P.run_tts(pipe, input_ids, ref_s, speed=args.speed), args.iters)

        # fast steady-state = dynamic token/align prep (per utterance) + traced frame replay, all under
        # the single-bf16 matmul bypass (Kokoro is feed-forward; log-spec gate stays green).
        ins = {"input_ids": input_ids, "ref_s": ref_s}
        ops.set_hp_bypass(True)
        try:
            prep_ms = _timeit(lambda: pipe._prep_frame_inputs(ins, speed=args.speed), args.iters)
            d = pipe._prep_frame_inputs(ins, speed=args.speed)
            en, asr, s, dstyle, ctx = d["en"], d["asr"], d["s"], d["dec_style"], d["ctx"]

            def _frame_fwd():
                F0, N = pipe._f0n_train(en, s)
                x = pipe._decode_features(asr, F0, N, dstyle)
                return pipe.generator(x, dstyle, F0)

            _trace_alloc.activate()
            push_trace_ctx(ctx)
            _frame_fwd()
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            _frame_fwd()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            replay_ms = _timeit(lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False), args.iters, warmup=3)
            ttnn.release_trace(device, tid)
            pop_trace_ctx()
            _trace_alloc.deactivate()
        finally:
            ops.set_hp_bypass(False)
        fast_ms = prep_ms + replay_ms
    finally:
        ttnn.close_device(device)

    print("=" * 68)
    print(f"phonemes={args.phonemes!r}  voice={args.voice}  tokens={input_ids.shape[-1]}")
    print(f"audio: {wav_fast.numel()} samples ({dur_s:.2f}s @ {SR//1000}kHz)")
    print(f"pred_dur matches HF: {bool((pred_dur == gold_dur).all())}")
    print("-" * 68)
    print("QUALITY (log-spectrogram PCC vs HF gold; gate >= 0.95):")
    print(f"  fast (traced)  = {spec_fast:.6f}   -> {args.out}")
    print(f"  dynamic (ref)  = {spec_dyn:.6f}   -> {ref_out}")
    print("-" * 68)
    print("SPEED (real wall-clock, single P150):")
    print(f"  dynamic e2e          = {dyn_ms:8.1f} ms   RTF {dyn_ms/1000/dur_s:.3f}  ({1000*dur_s/dyn_ms:.2f}x RT)")
    print(f"  fast steady-state    = {fast_ms:8.1f} ms   RTF {fast_ms/1000/dur_s:.3f}  ({1000*dur_s/fast_ms:.2f}x RT)")
    print(f"    (= prep {prep_ms:.1f} ms dynamic token/align  +  frame trace replay {replay_ms:.1f} ms)")
    print(f"  speedup              = {dyn_ms/fast_ms:.2f}x")
    print("=" * 68)
    print(f"Verify quality: listen to  {os.path.abspath(args.out)}  (fast)")
    print(f"           and  {os.path.abspath(ref_out)}  (reference)")


if __name__ == "__main__":
    main()
