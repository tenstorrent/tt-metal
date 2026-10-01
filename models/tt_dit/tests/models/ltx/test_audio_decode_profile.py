# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Traced LTX audio decode (mel-VAE -> vocoder -> BWE) on a 2x4 submesh: sub-stage split and one arm of an A/B.

The arm is set by LTX_AUDIO_DEVICE_CHAIN (read at module construction). Each run warms eagerly, captures,
then times AUDIO_REPEATS synchronized latent-in to waveform-out decodes, and saves the replayed waveform to
$AUDIO_OUT_DIR/wave_chain<0|1>.pt for a cross-arm bit-identity check. The host-bridged arm (chain=0) also
times one decode split at every host<->device boundary.
Requires LTX_CHECKPOINT. The 6 s clip is 145 frames @ 24 fps (audio latent 1x151x128).
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import ttnn

FRAMES, FPS = 145, 24.0
FIXTURE = Path(__file__).with_name("fixtures") / "girl_audio_latent.npy"


def _split_decode(pipeline, latent):
    """``decode_audio``'s traced host-bridged path, step for step, with a sync at each boundary."""
    mel_d, voc = pipeline.tt_mel_decoder, pipeline.tt_vocoder_with_bwe
    dev = pipeline.audio_mesh_device
    marks = []

    def mark(name):
        ttnn.synchronize_device(dev)
        marks.append((name, time.perf_counter()))

    mark("start")
    n, z = latent.shape[1], mel_d.z_channels
    spatial = latent.reshape(1, n, z, latent.shape[2] // z).permute(0, 2, 1, 3).float()
    x = mel_d._host_to_device(spatial)
    mark("mel_h2d")
    y = mel_d._forward_device(x, traced=True, tracer_trace_key=tuple(spatial.shape))
    mark("mel_trace")
    mel = mel_d._device_to_host(y)
    mark("mel_d2h")
    v = voc.vocoder
    xd = v._host_to_device(mel.float())
    mark("voc_h2d")
    yd = v._forward_device(xd, traced=True, tracer_trace_key=tuple(mel.shape))
    mark("voc_trace")
    wave = v._device_to_host(yd)
    mark("voc_d2h")
    _, _, length_low = wave.shape
    output_length = length_low * voc.output_sampling_rate // voc.input_sampling_rate
    if length_low % voc.hop_length:
        wave = torch.nn.functional.pad(wave, (0, voc.hop_length - length_low % voc.hop_length))
    stft_mel = voc._compute_mel_device(wave).transpose(2, 3).contiguous()
    mark("stft_eager")
    b = voc.bwe_generator
    xb = b._host_to_device(stft_mel)
    mark("bwe_h2d")
    yb = b._forward_device(xb, traced=True, tracer_trace_key=tuple(stft_mel.shape))
    mark("bwe_trace")
    residual = b._device_to_host(yb)
    mark("bwe_d2h")
    skip = voc._resample_device(wave)
    mark("resample_eager")
    out = torch.clamp(residual + skip, -1.0, 1.0)[..., :output_length]
    mark("host_mix")
    return out, [(name, (t - marks[i][1]) * 1000) for i, (name, t) in enumerate(marks[1:])]


@pytest.mark.skipif("LTX_CHECKPOINT" not in os.environ, reason="needs LTX_CHECKPOINT")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 300_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_audio_decode_profile(mesh_device, device_params):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    from models.tt_dit.tests.models.ltx.test_audio_ltx import _build_pipeline

    chain = os.environ.get("LTX_AUDIO_DEVICE_CHAIN", "0") == "1"
    repeats = int(os.environ.get("AUDIO_REPEATS", "5"))
    out_dir = Path(os.environ.get("AUDIO_OUT_DIR", "/tmp"))
    latent = torch.from_numpy(np.load(FIXTURE, allow_pickle=False)).float()
    assert tuple(latent.shape) == (1, 151, 128)

    pipeline, _ = _build_pipeline(
        mesh_device,
        sp_axis=1,
        tp_axis=0,
        checkpoint=os.environ["LTX_CHECKPOINT"],
        num_links=2,
        topology=ttnn.Topology.Linear,
    )
    voc = pipeline.tt_vocoder_with_bwe
    assert voc.device_chain == chain

    def set_trace(enabled):
        pipeline.tt_mel_decoder.use_trace = voc.use_trace = voc.use_trace_bwe = enabled

    try:
        set_trace(False)
        t0 = time.perf_counter()
        eager = pipeline.decode_audio(latent, FRAMES, fps=FPS).waveform.clone()
        print(f"AUDIO chain={int(chain)} eager_first_ms={(time.perf_counter() - t0) * 1000:.1f}")
        set_trace(True)
        pipeline.decode_audio(latent, FRAMES, fps=FPS)  # capture
        times, waves = [], []
        for _ in range(repeats):
            ttnn.synchronize_device(pipeline.audio_mesh_device)
            t0 = time.perf_counter()
            audio = pipeline.decode_audio(latent, FRAMES, fps=FPS)
            ttnn.synchronize_device(pipeline.audio_mesh_device)
            times.append((time.perf_counter() - t0) * 1000)
            waves.append(audio.waveform.clone())
        assert all(torch.equal(w, waves[0]) for w in waves), "replays differ"
        print(
            f"AUDIO chain={int(chain)} replay_ms={' '.join(f'{t:.1f}' for t in times)} min={min(times):.1f} "
            f"shape={tuple(waves[0].shape)} sr={audio.sampling_rate} "
            f"eager_vs_replay_max={(eager - waves[0]).abs().max().item():.3e}"
        )
        torch.save(waves[0], out_dir / f"wave_chain{int(chain)}.pt")
        if not chain:
            for _ in range(2):
                split_wave, split = _split_decode(pipeline, latent)
            target = int(FRAMES / FPS * audio.sampling_rate)
            assert torch.equal(split_wave.squeeze(0).float()[..., :target], waves[0]), "split path drifted"
            print("AUDIO split " + " ".join(f"{k}={v:.1f}" for k, v in split) + f" sum={sum(v for _, v in split):.1f}")
    finally:
        pipeline.release_traces()
