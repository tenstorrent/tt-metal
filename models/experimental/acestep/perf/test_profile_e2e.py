# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Tracy device-profiler test for the FULL ACE-Step pipeline.

Each pipeline stage is wrapped in a Tracy signpost so the device-kernel log can be split per stage:
  signpost("caching")            -> build + warmup + program-cache (excluded from perf)
  signpost("encode")             -> text encoder + condition encoder
  signpost("denoise")            -> the ODE denoise loop (DiT x N steps)
  signpost("vae")                -> Oobleck VAE decode
  signpost("performance")        -> whole measured region marker

Run under the Tracy device profiler to get REAL per-op device-kernel durations per stage (not host
wall-clock). Driver: models/experimental/acestep/perf/run_profile_e2e.py.

Skipped unless ACESTEP_PIPELINE_DIR / HF cache has the bundle. This is a profiling harness, not a
correctness test.
"""

import pytest
import torch
import ttnn

from models.experimental.acestep.reference.weight_utils import have_pipeline
from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline


try:
    from tracy import signpost
except ImportError:  # tracy not available -> no-op

    def signpost(_):
        pass


SECONDS = 10.24  # T'=128
INFER_STEPS = 2  # short loop for profiling (per-op durations don't need many steps)


@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step pipeline bundle not downloaded")
def test_profile_e2e(device):
    args = AceStepModelConfig.from_hf()
    pipe = create_tt_pipeline(args, device)

    prompt = "upbeat synthwave, driving bass, nostalgic"
    lyrics = "neon lights over the city tonight"
    seq_len = pipe._latent_len(SECONDS)
    hch = args.audio_acoustic_hidden_dim
    gen = torch.Generator().manual_seed(0)
    noise = torch.randn(1, 1, seq_len, hch, generator=gen)
    context = torch.cat([torch.zeros(1, 1, seq_len, hch), torch.ones(1, 1, seq_len, hch)], dim=-1)

    def _mk(x):
        return ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # ---- caching / warmup (compile + program cache), EXCLUDED from the measured region ----
    # Runs every stage once so all kernels compile + populate the program cache (and conv weight
    # caches) BEFORE the 'start' marker, so the measured region reflects steady-state device time,
    # not one-time compile.
    enc_hs = pipe.encode_prompt(prompt, lyrics)
    lat = pipe.generate(_mk(noise), _mk(context), enc_hs, infer_steps=1)
    pipe.decode(lat)
    ttnn.synchronize_device(device)

    # ---- measured region between start/stop (what post_process_ops_log(has_signposts=True) reads).
    # Per-stage markers ('enc_mark'/'dit_mark'/'vae_mark') are informational rows in the CSV to split
    # stages by row position.
    signpost("start")

    signpost("enc_mark")
    enc_hs = pipe.encode_prompt(prompt, lyrics)
    ttnn.synchronize_device(device)

    signpost("dit_mark")
    lat = pipe.generate(_mk(noise), _mk(context), enc_hs, infer_steps=INFER_STEPS)
    ttnn.synchronize_device(device)

    signpost("vae_mark")
    wav = pipe.decode(lat)
    ttnn.synchronize_device(device)

    signpost("stop")
    assert wav.shape[0] == 1 and wav.shape[1] == 2
