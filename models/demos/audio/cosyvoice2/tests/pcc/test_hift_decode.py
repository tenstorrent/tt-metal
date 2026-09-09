# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""HiFT's upsample/resblock/source(NSF-spectrum) stack, generalized to
CosyVoice2's 3-stage [8, 5, 3] topology.

`decode()` takes mel plus an already-computed excitation `s` -- computing `s`
(SineGen2) is a separate, follow-up piece; see the module docstring in
tt/hifigan/generator.py for why that split is the real seam upstream uses too.

No CosyVoice2 checkpoint is available in this environment, so `TorchHiFTDecodeRef`
is randomly initialised (torch.nn.Conv1d/ConvTranspose1d/weight_norm -- an
untouched, independent-of-TTNN implementation) and `TtHiFTDecoder` is built from
its exact weights, so a PCC test between them isolates "the TTNN port is wrong"
from "the weights differ" -- same bar as tests/pcc/test_istft.py.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.audio.cosyvoice2.tt.hifigan.generator import shape_trace

GATE_BF16 = 0.99


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_shape_trace_stage_count_matches_config():
    """The bug this guards against: shape_trace silently defaulting to 2 stages
    against a 3-stage upsample_rates/upsample_kernel_sizes config."""
    trace = shape_trace(8, 512, (8, 5, 3), (16, 11, 7), 16, 4, 80)
    assert len(trace["stages"]) == 3
    assert [st.index for st in trace["stages"]] == [0, 1, 2]


def test_shape_trace_source_downsample_factors():
    """Cumulative downsample factors feeding source_downs[i], hand-derived for
    [8, 5, 3]: stage 0 is 120/8=15 samples-per-mel-frame behind full audio rate,
    stage 1 is 120/40=3, stage 2 (last) is 1 -- i.e. no downsampling needed once
    the activation has been fully upsampled. (120 = prod([8,5,3]); the hop_len=4
    factor cancels out of the ratio.)"""
    trace = shape_trace(8, 512, (8, 5, 3), (16, 11, 7), 16, 4, 80)
    assert [st.source_downsample for st in trace["stages"]] == [15, 3, 1]


def test_shape_trace_reflection_pad_only_last_stage():
    """padded_length == up_length everywhere except the last stage, where
    ReflectionPad1d((1, 0)) adds exactly one sample."""
    trace = shape_trace(8, 512, (8, 5, 3), (16, 11, 7), 16, 4, 80)
    for st in trace["stages"][:-1]:
        assert st.padded_length == st.up_length
    assert trace["stages"][-1].padded_length == trace["stages"][-1].up_length + 1


def test_shape_trace_channel_progression():
    """out_channels halves each stage from base_channels; final conv_post input
    width is base_channels // 2**num_upsamples (64 for CosyVoice2's 3 stages,
    versus 128 for CosyVoice1's 2)."""
    trace = shape_trace(8, 512, (8, 5, 3), (16, 11, 7), 16, 4, 80)
    assert [st.out_channels for st in trace["stages"]] == [256, 128, 64]


def test_shape_trace_two_stage_still_works():
    """CosyVoice1's own [8, 8]/[16, 16] config still traces correctly -- this
    class did not regress the shape it was originally built for."""
    trace = shape_trace(8, 512, (8, 8), (16, 16), 16, 4, 80)
    assert len(trace["stages"]) == 2
    assert [st.source_downsample for st in trace["stages"]] == [8, 1]
    assert [st.out_channels for st in trace["stages"]] == [256, 128]


@pytest.mark.parametrize("mel_frames", [4, 8, 20])
def test_torch_ref_round_trips_waveform_length(mel_frames):
    """The untouched torch reference's actual output length matches shape_trace's
    prediction -- ties the length math to real module execution, not just to
    itself."""
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef

    torch.manual_seed(0)
    ref = TorchHiFTDecodeRef(seed=0)
    trace = shape_trace(mel_frames, 512, (8, 5, 3), (16, 11, 7), 16, 4, 80)
    mel = torch.randn(1, 80, mel_frames)
    s = torch.randn(1, 1, trace["audio_length"])
    with torch.no_grad():
        wav = ref.decode(mel, s)
    assert wav.shape == (1, trace["waveform_length"])


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("mel_frames", [8, 20])
def test_device_decode_matches_real_torch_ref(device, mel_frames):
    """TtHiFTDecoder.decode() vs. TorchHiFTDecodeRef.decode() directly -- zero
    inferential steps, same weights, independent (torch-native) implementation
    on the other side. This is the 3-stage config's own version of
    test_device_istft_matches_real_torch_istft: the strongest single claim for
    this module, not an internal cross-check."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TtHiFTDecoder

    torch.manual_seed(11)
    ref = TorchHiFTDecodeRef(seed=11)
    trace = shape_trace(mel_frames, 512, (8, 5, 3), (16, 11, 7), 16, 4, 80)

    mel_t = torch.randn(1, 80, mel_frames) * 0.5
    s_t = torch.randn(1, 1, trace["audio_length"]) * 0.1  # excitation-scale noise, not full-scale audio

    with torch.no_grad():
        want = ref.decode(mel_t, s_t)

    dec = TtHiFTDecoder(device, ref, dtype=ttnn.bfloat16)
    mel_nlc = ttnn.from_torch(mel_t.permute(0, 2, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_nlc = ttnn.from_torch(s_t.permute(0, 2, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = dec.decode(mel_nlc, s_nlc, mel_frames, batch_size=1)
    got = ttnn.to_torch(out).reshape(1, -1).float()

    assert got.shape == want.shape, (got.shape, want.shape)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device T_mel={mel_frames} (3-stage [8,5,3]) PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_decode_matches_real_torch_ref_two_stage_config(device):
    """Same test, CosyVoice1's [8, 8]/[16, 16] topology -- confirms the
    generalization didn't regress the config this module was validated against
    first."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TtHiFTDecoder

    torch.manual_seed(23)
    mel_frames = 8
    ref = TorchHiFTDecodeRef(
        seed=23,
        upsample_rates=(8, 8),
        upsample_kernel_sizes=(16, 16),
        source_resblock_kernel_sizes=(7, 11),
        source_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5)),
    )
    trace = shape_trace(mel_frames, 512, (8, 8), (16, 16), 16, 4, 80)

    mel_t = torch.randn(1, 80, mel_frames) * 0.5
    s_t = torch.randn(1, 1, trace["audio_length"]) * 0.1

    with torch.no_grad():
        want = ref.decode(mel_t, s_t)

    dec = TtHiFTDecoder(device, ref, dtype=ttnn.bfloat16)
    mel_nlc = ttnn.from_torch(mel_t.permute(0, 2, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_nlc = ttnn.from_torch(s_t.permute(0, 2, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = dec.decode(mel_nlc, s_nlc, mel_frames, batch_size=1)
    got = ttnn.to_torch(out).reshape(1, -1).float()

    assert got.shape == want.shape, (got.shape, want.shape)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device T_mel={mel_frames} (2-stage [8,8]) PCC {pcc}")
    assert passed, pcc
