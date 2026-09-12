# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The vocoder's exit criterion: mel -> waveform through the whole HiFT vocoder on device.

The bar is "mel → waveform PCC ≥ 0.99, **zero host ops**". This is that test. It feeds the exact mel and excitation the reference consumed --
captured in `hift.decode` -- and compares the waveform it produced.

Weights come from `scripts/export_weights.py`, not from a live CosyVoice module,
so the tt-metal environment never imports the reference package (see tt/weights.py).
"""
from __future__ import annotations

import os

import pytest

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden, pcc
from models.demos.cosyvoice.tt.hifigan.generator import shape_trace
from models.demos.cosyvoice.tt.weights import default_weights_path

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)

needs_weights = pytest.mark.skipif(
    not os.path.exists(default_weights_path()),
    reason="run scripts/export_weights.py in the CosyVoice venv first",
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "hift.decode.npz")),
    reason="run scripts/gen_golden.py in the CosyVoice venv first",
)


@needs_weights
def test_exported_weights_are_self_consistent():
    """Host-only: the export must contain every module the generator will ask for,
    at the shapes the length graph assumes. Cheap, and it fails with a useful
    message instead of a KeyError forty convolutions into a device run."""
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(default_weights_path())
    assert bag.meta["weight_norm_folded"] is True
    assert bag.count("ups") == bag.meta["num_upsamples"] == 2
    assert bag.count("resblocks") == bag.meta["num_upsamples"] * bag.meta["num_kernels"] == 6
    assert bag.count("source_downs") == bag.count("source_resblocks") == 2
    assert bag.window is not None and len(bag.window) == bag.meta["istft_params"]["n_fft"]

    # Channel plan: 80 -> 512 -> 256 -> 128 -> 18
    assert tuple(bag.sub("conv_pre").tensor("weight").shape) == (512, 80, 7)
    assert tuple(bag.sub("ups.0").tensor("weight").shape) == (512, 256, 16)
    assert tuple(bag.sub("ups.1").tensor("weight").shape) == (256, 128, 16)
    assert tuple(bag.sub("conv_post").tensor("weight").shape) == (18, 128, 7)

    # The source branch must land on exactly the lengths the main path reaches.
    t = shape_trace(282)
    for i, stage in enumerate(t["stages"]):
        k = bag.sub(f"source_downs.{i}").tensor("weight").shape[-1]
        stride = k // 2 if k > 1 else 1
        pad = stride // 2 if k > 1 else 0
        got = (t["stft_frames"] + 2 * pad - k) // stride + 1
        assert got == stage.source_length == stage.padded_length, (i, got, stage)


@needs_weights
@needs_golden
@needs_l1_small
def test_device_hift_decode_matches_golden(device):
    """The whole vocoder on device, against the captured reference waveform.

    282 mel frames -> 72192 samples, i.e. 3.27 s of audio. Everything between is
    TTNN: conv_pre, two transposed-conv upsamples, 8 ResBlocks, the source STFT
    branch, conv_post, and the iSTFT. No host ops in the path.
    """
    import ttnn
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator

    g = load_golden("hift.decode")
    mel_t = as_torch(g["call0.in_x"])  # [1, 80, T_mel]
    s_t = as_torch(g["call0.in_s"])  # [1, 1, T_audio]
    want = as_torch(g["call0.out_speech"])  # [1, T_audio]
    mel_frames = mel_t.shape[-1]

    trace = shape_trace(mel_frames)
    assert s_t.shape[-1] == trace["audio_length"], (s_t.shape, trace["audio_length"])

    model = TtHiFTGenerator.from_export(device)
    mel = ttnn.from_torch(
        mel_t.permute(0, 2, 1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    s = ttnn.from_torch(s_t.permute(0, 2, 1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = model.decode(mel, s, mel_frames)
    got = ttnn.to_torch(out).reshape(1, -1).float()

    p = pcc(got, want)
    print(f"\n  HiFT decode {mel_frames} mel frames -> {got.shape[-1]} samples")
    print(f"  PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= 0.99, p
