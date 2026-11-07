# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host audio backend latency (tt/audio/host_backend.py): ``embed_audio`` per clip and thread count.

Mel input as the processor produces it (computed once per clip, outside the timing: the processor runs in the API
server, not in the model process). Each configuration runs once to warm up, then ``--repeats`` times; min and
median of ``embed_audio`` are reported, and the min of the tokenizer encoder (codes) and of the patch encoder
timed on their own. The load average is printed before and after: other processes on a shared host inflate the
numbers.

    python models/experimental/mimo_v2_6_flash/tests/perf/audio_host_latency.py \\
        --clips <dir> [--threads 8 16] [--clip speech_5s speech_30s ...] [--patch-dtype bf16|fp32]
"""

import argparse
import os
import statistics
import time

import soundfile as sf
import torch

from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import HostAudioBackend


def load_mel(path, cfg):
    import torchaudio.functional as taf  # vLLM resamples with torchaudio's Resample (same defaults)

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)
    if sr != cfg.sampling_rate:
        wav = taf.resample(wav, sr, cfg.sampling_rate)
    return ref.log_mel_spectrogram(wav, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--clips", default=os.environ.get("MIMO_AUDIO_CLIPS", "/data/ttuser-agentx/mm/audio/clips"))
    parser.add_argument(
        "--clip", nargs="+", default=["speech_5s", "speech_30s", "speech_120s", "speech_300s", "speech_600s"]
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--patch-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--dense-attention", action="store_true", help="the reference's masked attention")
    parser.add_argument(
        "--dense-windowed", action="store_true", help="is_causal full layers, but dense masked windowed layers"
    )
    args = parser.parse_args()

    print(f"load average before: {os.getloadavg()}")
    t = time.time()
    backend = HostAudioBackend(
        args.model,
        patch_dtype=torch.bfloat16 if args.patch_dtype == "bf16" else torch.float32,
        fast_attention=not args.dense_attention,
    )
    if args.dense_windowed and backend.encoder is not None:
        backend.encoder.windowed_blocks = False
    print(f"backend load: {time.time() - t:.1f} s")
    cfg = backend.cfg
    mels = {c: load_mel(os.path.join(args.clips, f"{c}.wav"), cfg) for c in args.clip}
    print(
        f"{'clip':>12} {'audio s':>8} {'tokens':>6} {'thr':>3} {'min s':>7} {'median s':>8} {'codes s':>7} {'patch s':>7}"
    )
    for threads in args.threads:
        backend.num_threads = threads
        torch.set_num_threads(threads)
        for clip, mel in mels.items():
            backend.embed_audio(mel)
            totals, enc_times, patch_times = [], [], []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                out = backend.embed_audio(mel)
                t1 = time.perf_counter()
                codes = backend.codes(mel)
                t2 = time.perf_counter()
                with torch.inference_mode():
                    backend.patch(ref.pad_and_group_codes(codes, cfg))
                t3 = time.perf_counter()
                totals.append(t1 - t0)
                enc_times.append(t2 - t1)
                patch_times.append(t3 - t2)
            print(
                f"{clip:>12} {(mel.shape[0] - 1) / 100:8.1f} {out.shape[0]:6d} {threads:3d} {min(totals):7.3f} "
                f"{statistics.median(totals):8.3f} {min(enc_times):7.3f} {min(patch_times):7.3f}",
                flush=True,
            )
    print(f"load average after: {os.getloadavg()}")


if __name__ == "__main__":
    main()
