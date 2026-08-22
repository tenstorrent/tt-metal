# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Diffusion-step (S) sweep: quantify the RTF win and the audio-quality cost of fewer DPM steps.

The on-device DPM-solver (dpmsolver++ order-2) is exact vs the HF scheduler at any matched S, so we can
answer "how few steps hold quality?" with the traced pipeline itself. For each S we:
  * time the traced+2CQ path over a FIXED N diffusion frames (bench_force_diffusion) -> RTF, per-frame;
  * measure quality objectively as the log-STFT L1 distance of the forced-N waveform vs the S=20 anchor
    (same N -> same length -> directly comparable);
  * run the NATURAL path (real token stop) once and write a .wav so the exact bytes can be auditioned.

RTF = wall / audio_seconds (audio_seconds = N*3200/24000). Diffusion is ~37% of the frame and scales
~linearly in S, so halving S should push RTF well under the current 0.907.
"""
import struct
import sys
import time
import wave

import torch

sys.path.insert(0, "/local/ttuser/teja/tt-metal")
import ttnn
from models.demos.vibevoice_1_5b.tt import pipeline as P
from models.demos.vibevoice_1_5b.tt._golden import reference as R

N = 39
SR, CHUNK = 24000, 3200
S_LIST = [20, 15, 12, 10, 8, 6]
ANCHOR_S = 20


def _write_wav(path, wav):
    wav = torch.as_tensor(wav).reshape(-1).float()
    wav = (wav / (wav.abs().max() + 1e-8) * 0.98 * 32767).clamp(-32768, 32767).short()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", int(s)) for s in wav.tolist()))


def _logstft(wav):
    x = torch.as_tensor(wav).reshape(-1).float()
    x = x / (x.abs().max() + 1e-8)
    spec = torch.stft(
        x, n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(1024), return_complex=True
    )
    return torch.log1p(spec.abs())


def _forced(pipe, inputs, tok, device):
    pipe.bench_force_diffusion = True
    pipe.run(inputs, tok, collect=False)  # warm-up (compile / capture)
    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        res = pipe.run(inputs, tok, collect=False)
        ttnn.synchronize_device(device)
        ts.append(time.perf_counter() - t0)
    return min(ts), res


def main():
    model = R.load_reference_model()
    processor = R.build_processor()
    tok = processor.tokenizer
    inputs = dict(R.make_inputs(processor, "Speaker 0: Hello there, this is a test.", R.default_voice_sample()))
    inputs["noises"] = R.make_noises(N + 2, int(model.config.acoustic_vae_dim))
    audio_sec = N * CHUNK / SR

    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=400000000, num_command_queues=2)
    rows = []
    anchor_spec = None
    try:
        for S in S_LIST:
            pipe = P.VibeVoiceTTS(device, model, N=N, S=S, use_trace=True, two_cq=True)
            t, res = _forced(pipe, inputs, tok, device)
            nt = res["diff_count"]
            wav = P._th(res["waveform_tt"]).reshape(-1).clone()
            spec = _logstft(wav)
            if S == ANCHOR_S:
                anchor_spec = spec
            dist = float("nan")
            if anchor_spec is not None and spec.shape == anchor_spec.shape:
                dist = float((spec - anchor_spec).abs().mean())
            rtf = t / (nt * CHUNK / SR)
            rows.append((S, t * 1e3, t / nt * 1e3, rtf, dist))
            print(
                f"[S={S:2d}] {t*1e3:8.1f} ms | {t/nt*1e3:6.1f} ms/frame | RTF {rtf:6.3f} | logSTFT-dist-vs-S20 {dist:.4f}",
                flush=True,
            )

            # natural path -> listenable wav
            pipe.bench_force_diffusion = False
            resn = pipe.run(inputs, tok, collect=False)
            wavn = P._th(resn["waveform_tt"]).reshape(-1).clone()
            path = f"/tmp/vibevoice_S{S}.wav"
            _write_wav(path, wavn)
            print(f"        natural: {resn['diff_count']} frames, {wavn.shape[0]/SR:.2f}s -> {path}", flush=True)
    finally:
        ttnn.close_device(device)

    print(
        "\n===================== S sweep (N=%d forced, %.2fs audio @24kHz) =====================" % (N, audio_sec),
        flush=True,
    )
    print(f"{'S':>3} | {'total ms':>9} | {'ms/frame':>9} | {'RTF':>6} | {'logSTFT dist vs S20':>20}", flush=True)
    for S, tot, pf, rtf, dist in rows:
        print(f"{S:>3} | {tot:>9.1f} | {pf:>9.1f} | {rtf:>6.3f} | {dist:>20.4f}", flush=True)


if __name__ == "__main__":
    main()
