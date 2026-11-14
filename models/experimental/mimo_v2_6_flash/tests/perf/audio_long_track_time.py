# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio time of one long track (e.g. a video's audio track) per placement, on one 1x8 row:

* front end: the log-mel (both placements), and with MIMO_AUDIO_TOKENIZER=frontend the fp32 host tokenizer encoder
  (HostAudioBackend.codes at ``--threads``, as the API server runs it);
* engine: with MIMO_AUDIO_TOKENIZER=device the device tokenizer encoder (tt/audio/tokenizer.py) and the host RVQ;
  then the patch encoder, device (tt/audio/patch.py) or host (bf16 at ``--threads``).

Also reports the code frames on which the two tokenizers disagree. The wav is mono at 24 kHz (the processor's rate).

    TT_VISIBLE_DEVICES=<the row's 8 chips> python models/experimental/mimo_v2_6_flash/tests/perf/audio_long_track_time.py \\
        --wav track_24k.wav [--threads 8]
"""

import argparse
import os
import time

import soundfile as sf
import torch

import ttnn
from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import HostAudioBackend
from models.experimental.mimo_v2_6_flash.tt.audio.patch import TtAudioPatchEncoder
from models.experimental.mimo_v2_6_flash.tt.audio.tokenizer import TtAudioTokenizerEncoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--wav", required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    wav, sr = sf.read(a.wav, dtype="float32")
    assert sr == 24000 and wav.ndim == 1, (sr, wav.shape)
    cfg = ref.MiMoAudioConfig.from_model_path(a.model)
    t = time.perf_counter()
    mel = ref.log_mel_spectrogram(torch.from_numpy(wav), cfg)
    t_mel = time.perf_counter() - t
    host = HostAudioBackend(a.model, num_threads=a.threads)
    print(f"{wav.shape[0] / sr / 60:.1f} min of audio, {mel.shape[0]} mel frames: mel {t_mel:.2f} s", flush=True)
    t = time.perf_counter()
    codes_host = host.codes(mel)
    t_host_tok = time.perf_counter() - t
    t = time.perf_counter()
    host.embed_audio_codes(codes_host)
    t_host_patch = time.perf_counter() - t
    print(
        f"host: tokenizer {t_host_tok:.1f} s, bf16 patch encoder {t_host_patch:.1f} s ({a.threads} threads)", flush=True
    )
    cache = os.environ.get("MIMO_AUDIO_CACHE")
    sub = lambda w: os.path.join(cache, os.path.basename(os.path.normpath(a.model)), w) if cache else None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=65536, trace_region_size=0)
    try:
        tok = TtAudioTokenizerEncoder.from_checkpoint(mesh, a.model, cache_dir=sub("tokenizer"))
        patch = TtAudioPatchEncoder.from_checkpoint(mesh, a.model, cache_dir=sub("patch"))
        tok.warm()
        patch.warm()
        tok.codes(mel)  # first call of each bucket's shapes
        t = time.perf_counter()
        h = tok.hidden(mel)
        t_dev_tok = time.perf_counter() - t
        t = time.perf_counter()
        codes_dev = tok.quantizer.quantize(h).t().contiguous()
        t_rvq = time.perf_counter() - t
        t = time.perf_counter()
        patch.embed_audio_codes(codes_host)
        t_dev_patch = time.perf_counter() - t
        changed = (codes_dev != codes_host).any(1).float().mean().item()
        print(
            f"device: tokenizer encoder {t_dev_tok:.2f} s + host RVQ {t_rvq:.2f} s, patch encoder {t_dev_patch:.2f} s; "
            f"code frames differing from the host tokenizer {100 * changed:.2f} %",
            flush=True,
        )
        print(
            f"SUMMARY front end + engine audio time: frontend tokenizer {t_mel + t_host_tok:.1f} s + device patch "
            f"{t_dev_patch:.2f} s = {t_mel + t_host_tok + t_dev_patch:.1f} s; device tokenizer: mel {t_mel:.1f} s + "
            f"{t_dev_tok + t_rvq + t_dev_patch:.1f} s on the engine = {t_mel + t_dev_tok + t_rvq + t_dev_patch:.1f} s",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
