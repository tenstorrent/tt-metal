# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Final customer bench: eager vs the INTEGRATED traced+2CQ pipeline.run() (not a prototype).

Loads the model once, opens one device with a trace region + 2 command queues, and times
`VibeVoiceTTS.run()` in both modes (best-of-3 after warm-up) for N diffusion frames / S ddpm steps.
Reports per-frame latency and RTF = wall / audio_seconds (audio_seconds = N*3200/24000). Also writes
the traced+2CQ waveform to a .wav so we can listen to the exact bytes the customer path emits.
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

N, S = 39, 20
SR, CHUNK = 24000, 3200
WAV_OUT = "/tmp/vibevoice_tt_trace2cq.wav"


def _write_wav(path, wav):
    wav = torch.as_tensor(wav).reshape(-1).float()
    wav = (wav / (wav.abs().max() + 1e-8) * 0.98 * 32767).clamp(-32768, 32767).short()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", int(s)) for s in wav.tolist()))


def _best_of_3(pipe, inputs, tok, device):
    pipe.bench_force_diffusion = True  # measure a fixed N real diffusion frames (comparable)
    pipe.run(inputs, tok, collect=False)  # warm-up (compile / capture)
    ts = []
    for _ in range(3):
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
    try:
        eager = P.VibeVoiceTTS(device, model, N=N, S=S, use_trace=False)
        t_eager, res_e = _best_of_3(eager, inputs, tok, device)
        ne = res_e["diff_count"]

        traced = P.VibeVoiceTTS(device, model, N=N, S=S, use_trace=True, two_cq=True)
        t_traced, res = _best_of_3(traced, inputs, tok, device)
        nt = res["diff_count"]
        wav = P._th(res["waveform_tt"]).reshape(-1).clone()

        print(f"frames actually generated: eager={ne} traced={nt} (forced N={N})", flush=True)
        print(
            "===================== RTF (N=%d frames, S=%d, %.2fs audio @24kHz) ====================="
            % (N, S, audio_sec),
            flush=True,
        )
        print(
            f"eager        : {t_eager*1e3:8.1f} ms | {t_eager/ne*1e3:6.1f} ms/frame | RTF {t_eager/(ne*CHUNK/SR):6.3f}",
            flush=True,
        )
        print(
            f"traced+2CQ   : {t_traced*1e3:8.1f} ms | {t_traced/nt*1e3:6.1f} ms/frame | RTF {t_traced/(nt*CHUNK/SR):6.3f}",
            flush=True,
        )
        print(f"speedup      : {t_eager/t_traced:5.2f}x", flush=True)
    finally:
        ttnn.close_device(device)
    _write_wav(WAV_OUT, wav)
    print(f"wrote {wav.shape[0]} samples ({wav.shape[0]/SR:.2f}s) to {WAV_OUT}", flush=True)


if __name__ == "__main__":
    main()
