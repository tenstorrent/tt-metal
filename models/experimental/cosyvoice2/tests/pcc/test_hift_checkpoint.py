# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Real CosyVoice2-0.5B checkpoint weights (`hift.pt`, from
`FunAudioLLM/CosyVoice2-0.5B`) loaded into `TorchHiFTDecodeRef`/`TtHiFTDecoder`
and the excitation linear (`m_source.l_linear`), checked with the SAME
TT-vs-torch-with-shared-weights isolation `test_hift_decode.py`'s existing
device test already uses (a PCC gap here is the device op disagreeing with
real upstream math, not the weights differing) -- just with the random init
that test uses replaced by real trained weights. See `tt/checkpoint.py` for
the download and `TorchHiFTDecodeRef.from_checkpoint`'s docstring for the
exact key mapping (confirmed 1:1 against the real checkpoint directly, zero
renaming needed).

Lowest-risk module in this bring-up's checkpoint-loading order (HiFT vocoder
-> F0 predictor -> flow decoder -> LLM backbone): no autoregression, no
random sampling, and (per `cosyvoice2.yaml`, confirmed) a plain non-causal
stack -- the fewest ways a real-weight numerical surprise could hide. One
found anyway, real and worth recording:

**Real weights need fp32 here; `test_hift_decode.py`'s random-init bf16 gate
(0.99) does not transfer.** Measured directly, not assumed: at bf16, this
test's PCC is ~0.49; at fp32 (same weights, same op graph, only the device
tensor dtype changed), PCC is 0.999+ at both mel lengths. Traced stage by
stage (device vs torch PCC at each boundary) to find why: the resblock/
upsample stack itself agrees well (PCC 0.9988 going into `conv_post`) --
real trained weights (unlike this package's small random init) drive
`conv_post`'s pre-activation values to std~1.7, max|x|~4.5 (verified: this
magnitude is intrinsic to the trained weights, not sensitive to input mel
scale -- checked at mel scales 0.5/0.2/0.1/0.05, `conv_post` max|x| stayed
~4.3-4.5 throughout). `magnitude = exp(x[:, :bins])` then amplifies bf16's
~3-decimal-digit precision at that magnitude into a large relative error
(`exp(4.46)=86.5` vs `exp(4.53)=93.0` from a ~0.07 raw difference), which
`istft`'s magnitude*phase product compounds further. Not a logic bug (fp32
on the exact same graph is correct); a genuine precision requirement real
trained weights expose that random init, by construction, never did. Use
`dtype=ttnn.float32` when loading a real checkpoint into `TtHiFTDecoder`.
`test_hift_decode.py`'s own random-init bf16 tests are correct and
unaffected -- this is specific to the real checkpoint's wider dynamic range.

Does NOT check `f0_predictor`'s real weights (a separate module, own
checkpoint-loading step) or the full `HiFTGenerator.inference` composition
with a real, checkpoint-loaded `f0_predictor` -- both come after
`f0_predictor`'s own real-checkpoint PCC check passes on its own, following
this bring-up's "don't move to the next module until this one's verified"
rule.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE = 0.99

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("mel_frames", [8, 20])
def test_device_hift_decode_matches_torch_reference_real_checkpoint(device, mel_frames):
    """`TtHiFTDecoder.decode()` vs `TorchHiFTDecodeRef.decode()`, both built
    from the SAME real `hift.pt` weights (not random init) -- the real-weight
    counterpart of `test_hift_decode.py`'s
    `test_device_decode_matches_real_torch_ref`, same shape_trace-derived `s`
    length, same synthetic (not real-audio) mel/s inputs. Runs at fp32, not
    bf16 -- see module docstring for why bf16 collapses to PCC ~0.49 here
    even though the op graph is identical and correct."""
    import ttnn
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file
    from models.experimental.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TtHiFTDecoder, shape_trace

    dtype = ttnn.float32
    hift_sd = load_checkpoint_file("hift.pt")
    ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)

    trace = shape_trace(
        mel_frames, ref.conv_pre.out_channels, ref.upsample_rates, (16, 11, 7), ref.n_fft, ref.hop_len, 80
    )

    torch.manual_seed(mel_frames)
    mel_t = torch.randn(1, 80, mel_frames) * 0.5
    s_t = (
        torch.randn(1, 1, trace["audio_length"]) * 0.1
    )  # excitation-scale noise, matches test_hift_decode.py's own convention

    with torch.no_grad():
        want = ref.decode(mel_t, s_t)

    dec = TtHiFTDecoder(device, ref, dtype=dtype)
    mel_nlc = ttnn.from_torch(mel_t.permute(0, 2, 1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    s_nlc = ttnn.from_torch(s_t.permute(0, 2, 1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = dec.decode(mel_nlc, s_nlc, mel_frames, batch_size=1)
    got = ttnn.to_torch(out).reshape(1, -1).float()

    assert got.shape == want.shape, (got.shape, want.shape)
    passed, pcc = comp_pcc(want, got, GATE)
    print(f"\n  real-checkpoint device HiFT decode fp32 (T_mel={mel_frames}) PCC {pcc}")
    assert passed, pcc


def test_source_linear_weight_shape_matches_harmonic_num():
    """`m_source.l_linear` is `Linear(harmonic_num + 1, 1)` (one weight per
    harmonic plus the fundamental) -- checked directly against the real
    checkpoint's own tensor shape, confirming `harmonic_num=8` (this
    package's existing assumption, from `nb_harmonics: 8` in `cosyvoice2.yaml`)
    matches the real trained weights' own shape, not just the config value."""
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file

    hift_sd = load_checkpoint_file("hift.pt")
    assert hift_sd["m_source.l_linear.weight"].shape == (1, 9)  # harmonic_num=8 -> 9 components
    assert hift_sd["m_source.l_linear.bias"].shape == (1,)


# HiFT bucketing (tt/pipeline.py pads the mel to a bucket with silence and trims the audio). HiFT's convs look
# ahead, so the padding reaches back into the valid audio. Measured 2026-09-27 on real speech mels: 0.22-0.38 s
# (docs/VALIDATION.md). This pins the reach: past REACH_S from the end, the pad content changes nothing.
REACH_S, REACH_TOL = 0.5, 1e-3


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_hift_bucket_padding_reach_real_checkpoint(device):
    """Silence padding vs zero padding at the SAME bucket (same kernels), torch F0 injected (D16), same sine noise
    over the valid region: any difference is the pad content leaking backwards. It must vanish (<= REACH_TOL)
    everywhere except the last REACH_S of the valid audio, and must be visible inside it (else the check is blind)."""
    import math

    import ttnn
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
    from models.experimental.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
    from models.experimental.cosyvoice2.tt.hifigan.generator import (
        TorchHiFTDecodeRef,
        TorchHiFTGeneratorInferenceRef,
        TtHiFTDecoder,
        TtHiFTGenerator,
    )

    hift_sd = load_checkpoint_file("hift.pt")
    decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
    f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor."))
    ref = TorchHiFTGeneratorInferenceRef(
        decode_ref, f0_ref, hift_sd["m_source.l_linear.weight"], hift_sd["m_source.l_linear.bias"]
    )
    gen = TtHiFTGenerator(device, ref, TtHiFTDecoder(device, decode_ref, dtype=ttnn.float32), dtype=ttnn.float32)

    frames, bucket = 150, 256
    g = torch.Generator().manual_seed(0)
    mel = (torch.randn(1, frames, 80, generator=g) * 2.0 - 6.0).clamp(min=math.log(1e-5))
    n = frames * 480
    noise = torch.cat([torch.randn(1, n, 9, generator=g), torch.zeros(1, (bucket - frames) * 480, 9)], dim=1)

    def padded_run(value):
        padded = torch.cat([mel, torch.full((1, bucket - frames, 80), value)], dim=1)
        with torch.no_grad():
            f0 = f0_ref(padded.transpose(1, 2)).reshape(1, bucket, 1)
        f0_dev = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        noise_dev = ttnn.from_torch(noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        s, _, _ = gen.source(ttnn.repeat_interleave(f0_dev, gen.upsample_scale, dim=1), sine_noise=noise_dev)
        mel_dev = ttnn.from_torch(padded, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.to_torch(gen.decoder.decode(mel_dev, s, bucket, 1)).reshape(-1).float()[:n]

    leak = (padded_run(math.log(1e-5)) - padded_run(0.0)).abs()
    tail = int(REACH_S * 24000)
    over = torch.nonzero(leak > REACH_TOL)
    reach = (n - over[0].item()) / 24000 if len(over) else 0.0
    print(f"\n  pad content reaches {reach * 1000:.0f} ms back from the end; max |diff| {leak.max().item():.3g}")
    assert leak[: n - tail].max().item() <= REACH_TOL, f"padding leaks {reach:.3f} s back, past {REACH_S} s"
    assert leak[n - tail :].max().item() > REACH_TOL, "no leakage at all near the end: the check cannot see anything"
