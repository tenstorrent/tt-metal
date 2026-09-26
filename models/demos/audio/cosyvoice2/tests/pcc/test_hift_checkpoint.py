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
    from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TtHiFTDecoder, shape_trace

    dtype = ttnn.float32
    hift_sd = load_checkpoint_file("hift.pt")
    ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)

    trace = shape_trace(mel_frames, ref.conv_pre.out_channels, ref.upsample_rates, (16, 11, 7), ref.n_fft, ref.hop_len, 80)

    torch.manual_seed(mel_frames)
    mel_t = torch.randn(1, 80, mel_frames) * 0.5
    s_t = torch.randn(1, 1, trace["audio_length"]) * 0.1  # excitation-scale noise, matches test_hift_decode.py's own convention

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
    from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file

    hift_sd = load_checkpoint_file("hift.pt")
    assert hift_sd["m_source.l_linear.weight"].shape == (1, 9)  # harmonic_num=8 -> 9 components
    assert hift_sd["m_source.l_linear.bias"].shape == (1,)
