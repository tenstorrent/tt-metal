# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The whole model, end to end on device.

Every earlier test checks one stage against a golden captured at that stage's own
boundary. This one runs **semantic tokens straight through to a waveform** and
compares against the reference's final audio, so an error that each stage absorbs
individually but that compounds across the chain has somewhere to show up.

The LLM is deliberately not in this chain. Its output is *sampled* -- RAS draws
from a multinomial -- so it cannot produce the reference's exact token stream on a
different RNG, and comparing audio generated from different tokens would measure
nothing. The tokens the reference actually emitted are captured, so the
deterministic part of the pipeline is tested against the reference's own audio,
and the LLM is tested separately against its logits (`tests/pcc/test_llm.py`).

That split is the honest one: 'flow + vocoder reproduce the reference waveform'
and 'the LLM reproduces the reference logits' are both checkable claims. 'the
whole pipeline reproduces the reference audio' is not, for any sampler.
"""
from __future__ import annotations

import os

import pytest
import torch

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden, pcc
from models.demos.cosyvoice.tt.weights import default_weights_path

HIFT_WEIGHTS = default_weights_path()
FLOW_WEIGHTS = HIFT_WEIGHTS.replace("hift_", "flow_")

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
needs_weights = pytest.mark.skipif(
    not (os.path.exists(HIFT_WEIGHTS) and os.path.exists(FLOW_WEIGHTS)), reason="export flow and hift weights first"
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "e2e.npz")), reason="generate goldens first"
)


def waveform_metrics(got: torch.Tensor, want: torch.Tensor) -> dict:
    """Sample correlation plus an energy-envelope check.

    Raw sample PCC is the strict measure and the right primary gate. The envelope
    correlation is reported alongside it because it separates the two ways a
    vocoder can be wrong: a small phase error tanks sample PCC while leaving the
    envelope intact and the audio perceptually identical, whereas a wrong
    magnitude spectrum moves both.
    """
    g, w = got.flatten().double(), want.flatten().double()
    n = min(len(g), len(w))
    g, w = g[:n], w[:n]
    win = 256
    ge = g[: n // win * win].reshape(-1, win).pow(2).mean(1).sqrt()
    we = w[: n // win * win].reshape(-1, win).pow(2).mean(1).sqrt()
    return {
        "samples": pcc(g, w),
        "envelope": pcc(ge, we),
        "rms_got": float(g.pow(2).mean().sqrt()),
        "rms_want": float(w.pow(2).mean().sqrt()),
        "max_abs": float((g - w).abs().max()),
    }


# --------------------------------------------------------------------------
# host tier
# --------------------------------------------------------------------------
@needs_golden
def test_reference_waveform_is_the_length_the_length_graph_predicts():
    """A pure-Python check that the mel the flow produces becomes exactly the
    number of samples the reference emitted. Catches an off-by-one in the vocoder's
    length chain without a device -- the same bet the host tier rests on, applied to the
    whole pipeline."""
    from models.demos.cosyvoice.tt.pipeline import CosyVoiceTTNN

    want = as_torch(load_golden("e2e")["waveform"])
    mel_frames = int(load_golden("flow.length_regulator")["call0.in_mel_len2"])
    assert CosyVoiceTTNN.audio_length_for(mel_frames) == want.shape[1], (
        CosyVoiceTTNN.audio_length_for(mel_frames),
        want.shape,
    )


@needs_golden
def test_modes_differ_only_in_prompt_construction(expect_error):
    """All four modes run the same three stages with the same weights.

    Pinned because the instruct mode's shape is genuinely surprising:
    `frontend_instruct` puts the instruction in the LLM's **prompt_text slot**, so
    it is a prefix the model reads, not a control signal. CosyVoice-1 wants a
    style *description*; CosyVoice-2's directive phrasing makes this model read the
    instruction aloud -- which took zh CER from 9.09% to 42.42% in the WER sweep.
    """
    from models.demos.cosyvoice.tt.pipeline import MODES, CosyVoiceTTNN

    assert set(MODES) == {"sft", "zero_shot", "cross_lingual", "instruct"}
    assert CosyVoiceTTNN.describe_mode("instruct")["prompt_text"] is True
    assert CosyVoiceTTNN.describe_mode("cross_lingual")["prompt_text"] is False
    assert CosyVoiceTTNN.describe_mode("zero_shot")["prompt_audio"] is True
    assert CosyVoiceTTNN.describe_mode("sft")["prompt_audio"] is False
    with expect_error(ValueError, "unknown mode"):
        CosyVoiceTTNN.describe_mode("nope")


@needs_golden
def test_the_vocoder_draws_randomness_at_inference_time():
    """`SineGen.forward` draws `phase_vec` and `noise` with **no training guard**,
    so a vocoder that ignores them computes a different function -- not merely an
    unseeded one. The captured `phase_vec` proves the draw is real and that its
    fundamental is pinned to zero."""
    g = load_golden("hift.sinegen")
    phase = as_torch(g["call0.in_phase_vec"])
    assert phase.shape[-1] == as_torch(g["call0.out_sine"]).shape[-1]
    assert float(phase.reshape(-1)[0]) == 0.0, "the fundamental harmonic is unshifted"
    assert float(phase.abs().max()) > 0.1, "the other harmonics really are offset"
    assert float(as_torch(g["call0.out_noise"]).abs().max()) > 0.0


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
def _build_flow_mel(device, ttnn):
    """Run the flow stage on the captured tokens; returns (mel, mel_len2)."""
    from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec
    from models.demos.cosyvoice.tt.weights import WeightBag

    emb_g = load_golden("flow.input_embedding")
    lr_g = load_golden("flow.length_regulator")
    cfm_g = load_golden("flow.cfm")
    spk_g = load_golden("flow.spk_embed_affine")

    tokens = torch.from_numpy(emb_g["call0.in_tokens"]).to(torch.int32)
    token_len1 = as_torch(lr_g["call0.in_x1"]).shape[1]
    mel_len1, mel_len2 = int(lr_g["call0.in_mel_len1"]), int(lr_g["call0.in_mel_len2"])

    def dev(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

    flow_bag = WeightBag.load(FLOW_WEIGHTS)
    flow = TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta)
    mel = flow.inference(
        dev(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        token_len1,
        mel_len1,
        mel_len2,
        dev(as_torch(cfm_g["call0.in_cond"])[:, :, :mel_len1].permute(0, 2, 1).contiguous()),
        dev(as_torch(spk_g["call0.in_x"]).reshape(1, 1, -1)),
        dev(as_torch(cfm_g["call0.rng_z"]).permute(0, 2, 1).contiguous()),
    )
    return mel, mel_len2, tokens.shape[1]


@needs_weights
@needs_golden
@needs_l1_small
def test_device_tokens_to_waveform(device):
    """**The integration gate.** Semantic tokens in, audio out, both stages on
    device: the flow decoder's ten Euler steps and the whole HiFT vocoder,
    against the reference's actual output audio.

    The excitation is the reference's, injected -- and that is a requirement, not
    a shortcut. NSF phase is chaotically sensitive to f0: drift is
    `sum(delta_f0)/sr` over samples, so holding it under a tenth of a cycle across
    72192 samples needs a **0.03 Hz** mean f0 error, 1.5e-4 relative. No
    implementation reaches that on Tensix, where HiFi4 is four bfloat16 passes
    rather than true fp32. Injecting the source is the same discipline the CFM's
    `z` and `SineGen`'s draws already follow, for the same reason.

    `test_device_self_computed_source` covers the other half -- the excitation the
    device builds for itself -- with the metrics that are meaningful for it.
    """
    import ttnn
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.weights import WeightBag

    want = as_torch(load_golden("e2e")["waveform"])
    mel, mel_len2, n_tokens = _build_flow_mel(device, ttnn)

    hift = TtHiFTGenerator(device, WeightBag.load(HIFT_WEIGHTS))
    src = as_torch(load_golden("hift.inference")["call0.out_source"]).permute(0, 2, 1).contiguous()
    wav = hift.decode(mel, ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device), mel_len2)
    got = ttnn.to_torch(wav).float().reshape(1, -1)
    ttnn.deallocate(mel)
    ttnn.deallocate(wav)

    m = waveform_metrics(got, want)
    print(f"\n  {n_tokens} tokens -> {mel_len2} mel frames -> {got.shape[1]} samples")
    print(f"  sample PCC   {m['samples']:.10f}")
    print(f"  envelope PCC {m['envelope']:.10f}")
    print(f"  RMS got/want {m['rms_got']:.6f} / {m['rms_want']:.6f}   max|d| {m['max_abs']:.3e}")
    assert got.shape[1] == want.shape[1], (got.shape, want.shape)
    assert m["samples"] >= 0.99, m


@needs_weights
@needs_golden
@needs_l1_small
def test_device_self_computed_source(device):
    """The vocoder building its own excitation: f0 predictor, nearest upsample,
    `SineGen`, `SourceModuleHnNSF`, then the full decode -- nothing injected but
    the two RNG draws.

    Gated on the **energy envelope**, not on samples, and the module docstring in
    `tt/hifigan/source.py` explains at length why: the phase of a self-computed
    excitation is a different valid realisation, so sample correlation measures
    the f0 predictor's last decimal place rather than whether the audio is right.
    The envelope and the RMS are what carry that.
    """
    import ttnn
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.weights import WeightBag

    want = as_torch(load_golden("e2e")["waveform"])
    sine_g = load_golden("hift.sinegen")
    mel, mel_len2, _ = _build_flow_mel(device, ttnn)

    def dev(v, dtype=ttnn.float32):
        return ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    hift = TtHiFTGenerator(device, WeightBag.load(HIFT_WEIGHTS))
    assert hift.f0_predictor is not None and hift.m_source is not None, "source branch not built"
    wav, audio_len, src = hift.inference(
        mel,
        mel_len2,
        phase_vec=dev(as_torch(sine_g["call0.in_phase_vec"])),
        sine_noise=dev(as_torch(sine_g["call0.out_noise"]).permute(0, 2, 1).contiguous()),
    )
    got = ttnn.to_torch(wav).float().reshape(1, -1)
    for t in (mel, wav, src):
        ttnn.deallocate(t)

    m = waveform_metrics(got, want)
    print(f"\n  self-computed source -> {audio_len} samples")
    print(f"  envelope PCC {m['envelope']:.10f}   (the gate)")
    print(f"  sample PCC   {m['samples']:.10f}   (not gated -- see the docstring)")
    print(f"  RMS got/want {m['rms_got']:.6f} / {m['rms_want']:.6f}")
    assert got.shape[1] == want.shape[1], (got.shape, want.shape)
    assert m["envelope"] >= 0.95, m
    assert 0.7 < m["rms_got"] / m["rms_want"] < 1.4, m


@needs_weights
@needs_golden
@needs_l1_small
def test_device_sinegen_reproduces_the_reference_phase(device):
    """`SineGen` given the reference's own f0 -- which isolates the phase
    accumulator from the f0 predictor.

    This is where the blocked mod-1 scan earns its place: `ttnn.cumsum` in fp32
    drifts more than half a cycle over 72192 samples (2000x worse than torch's),
    and reducing each block total mod 1 before accumulating takes this from
    0.843 to 0.99999.
    """
    import ttnn
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.weights import WeightBag

    msg, sg = load_golden("hift.m_source"), load_golden("hift.sinegen")
    hift = TtHiFTGenerator(device, WeightBag.load(HIFT_WEIGHTS))

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    sine, _, _ = hift.m_source.sine_gen(
        dev(as_torch(msg["call0.in_f0_upsampled"])),
        phase_vec=dev(as_torch(sg["call0.in_phase_vec"])),
        noise=dev(as_torch(sg["call0.out_noise"]).permute(0, 2, 1).contiguous()),
    )
    got = ttnn.to_torch(sine).float()
    want = as_torch(sg["call0.out_sine"])
    p = pcc(got, want)
    print(f"\n  SineGen on the reference f0, 72192 samples: PCC {p:.10f}")
    assert p >= 0.999, p
