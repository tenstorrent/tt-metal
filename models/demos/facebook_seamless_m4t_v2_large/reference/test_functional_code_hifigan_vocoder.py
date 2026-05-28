# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``code_hifigan_vocoder_forward``.

Builds an HuggingFace ``SeamlessM4Tv2CodeHifiGan`` at the SeamlessM4T-v2-Large
config defaults, extracts its weights, and verifies the standalone reference
matches bit-for-bit. Saves a golden tensor for downstream TTNN PCC
verification.

The vocoder ships with its own small set of config knobs (unit_embed_dim=1280,
spkr_embed_dim=256, lang_embed_dim=256, plus the conv channel ladder), so
unlike the encoder/decoder modules we do NOT need to shrink anything to keep
the HF instantiation cheap.

Kept in a separate file from ``test_functional.py`` so parallel workers do not
race when writing the shared test module.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import code_hifigan_vocoder_forward

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient of two (flattened) tensors."""
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return (a @ b / denom).item()


def _extract_residual_block_state_dict(module) -> dict:
    convs1 = [{"weight": c.weight.detach().clone(), "bias": c.bias.detach().clone()} for c in module.convs1]
    convs2 = [{"weight": c.weight.detach().clone(), "bias": c.bias.detach().clone()} for c in module.convs2]
    return {"convs1": convs1, "convs2": convs2}


def _extract_hifigan_state_dict(module) -> dict:
    conv_pre = {"weight": module.conv_pre.weight.detach().clone(), "bias": module.conv_pre.bias.detach().clone()}
    upsampler = [{"weight": l.weight.detach().clone(), "bias": l.bias.detach().clone()} for l in module.upsampler]
    resblocks = [_extract_residual_block_state_dict(rb) for rb in module.resblocks]
    conv_post = {"weight": module.conv_post.weight.detach().clone(), "bias": module.conv_post.bias.detach().clone()}
    return {
        "conv_pre": conv_pre,
        "upsampler": upsampler,
        "resblocks": resblocks,
        "conv_post": conv_post,
    }


def _extract_variance_predictor_state_dict(module) -> dict:
    return {
        "conv1": {"weight": module.conv1.weight.detach().clone(), "bias": module.conv1.bias.detach().clone()},
        "ln1": {"weight": module.ln1.weight.detach().clone(), "bias": module.ln1.bias.detach().clone()},
        "conv2": {"weight": module.conv2.weight.detach().clone(), "bias": module.conv2.bias.detach().clone()},
        "ln2": {"weight": module.ln2.weight.detach().clone(), "bias": module.ln2.bias.detach().clone()},
        "proj": {"weight": module.proj.weight.detach().clone(), "bias": module.proj.bias.detach().clone()},
    }


def _extract_code_hifigan_state_dict(module) -> dict:
    """Pull weights out of an HF ``SeamlessM4Tv2CodeHifiGan``."""
    return {
        "unit_embedding": {"weight": module.unit_embedding.weight.detach().clone()},
        "speaker_embedding": {"weight": module.speaker_embedding.weight.detach().clone()},
        "language_embedding": {"weight": module.language_embedding.weight.detach().clone()},
        "dur_predictor": _extract_variance_predictor_state_dict(module.dur_predictor),
        "hifi_gan": _extract_hifigan_state_dict(module.hifi_gan),
    }


def test_code_hifigan_vocoder_matches_hf() -> float:
    from transformers.models.seamless_m4t_v2.configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2CodeHifiGan

    torch.manual_seed(0)

    # Full default config — vocoder block is independent of the main encoder/decoder
    # depth knobs, so this is cheap enough to instantiate end-to-end.
    config = SeamlessM4Tv2Config()
    upsample_rates = tuple(config.upsample_rates)
    upsample_kernel_sizes = tuple(config.upsample_kernel_sizes)
    resblock_kernel_sizes = tuple(config.resblock_kernel_sizes)
    resblock_dilation_sizes = tuple(tuple(d) for d in config.resblock_dilation_sizes)
    leaky_relu_slope = config.leaky_relu_slope
    pad_token_id = config.t2u_pad_token_id
    kernel_size = config.variance_predictor_kernel_size

    hf = SeamlessM4Tv2CodeHifiGan(config)
    hf.eval()

    state_dict = _extract_code_hifigan_state_dict(hf)

    # Sanity-check key shapes against documented expectations.
    assert state_dict["unit_embedding"]["weight"].shape == (
        config.unit_hifi_gan_vocab_size,
        config.unit_embed_dim,
    )
    assert state_dict["speaker_embedding"]["weight"].shape == (
        config.vocoder_num_spkrs,
        config.spkr_embed_dim,
    )
    assert state_dict["language_embedding"]["weight"].shape == (
        config.vocoder_num_langs,
        config.lang_embed_dim,
    )

    # Tiny input per spec: unit_ids [1, 8], speaker 0, lang 0. We include the
    # pad token (1) on purpose so we exercise the _get_dur_output_lengths
    # masking path.
    input_ids = torch.tensor([[1, 8]], dtype=torch.long)
    speaker_id = torch.tensor([[0]], dtype=torch.long)
    lang_id = torch.tensor([[0]], dtype=torch.long)

    with torch.no_grad():
        hf_waveform, hf_lengths = hf(input_ids=input_ids, speaker_id=speaker_id, lang_id=lang_id)

    ref = code_hifigan_vocoder_forward(
        input_ids,
        speaker_id,
        lang_id,
        state_dict,
        pad_token_id=pad_token_id,
        variance_predictor_kernel_size=kernel_size,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        leaky_relu_slope=leaky_relu_slope,
    )
    ref_waveform = ref["waveform"]
    ref_lengths = ref["lengths"]

    assert (
        ref_waveform.shape == hf_waveform.shape
    ), f"waveform shape mismatch ref {tuple(ref_waveform.shape)} vs hf {tuple(hf_waveform.shape)}"

    pcc = _pcc(ref_waveform, hf_waveform)
    max_abs = (ref_waveform - hf_waveform).abs().max().item()
    print(f"[code_hifigan_vocoder] out_shape={tuple(ref_waveform.shape)} " f"pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")
    assert pcc > 0.99, f"code_hifigan_vocoder PCC {pcc} <= 0.99"
    # fp32 reference + fp32 HF -> identical op sequence -> near-exact match.
    assert torch.allclose(
        ref_waveform, hf_waveform, atol=1e-5, rtol=1e-4
    ), f"code_hifigan_vocoder diverged: max_abs={max_abs}"

    # Lengths must match HF exactly (integer op).
    assert torch.equal(ref_lengths.long(), hf_lengths.long()), f"length mismatch ref={ref_lengths} hf={hf_lengths}"

    golden_path = GOLDEN_DIR / "code_hifigan_vocoder.pt"
    torch.save(
        {
            "input_ids": input_ids,
            "speaker_id": speaker_id,
            "lang_id": lang_id,
            "state_dict": state_dict,
            "waveform": ref_waveform,
            "lengths": ref_lengths,
            "dur_out": ref["dur_out"],
            "config": {
                "batch": int(input_ids.shape[0]),
                "seq_in": int(input_ids.shape[1]),
                "time_out": int(ref_waveform.shape[-1]),
                "unit_hifi_gan_vocab_size": config.unit_hifi_gan_vocab_size,
                "unit_embed_dim": config.unit_embed_dim,
                "lang_embed_dim": config.lang_embed_dim,
                "spkr_embed_dim": config.spkr_embed_dim,
                "vocoder_num_langs": config.vocoder_num_langs,
                "vocoder_num_spkrs": config.vocoder_num_spkrs,
                "t2u_pad_token_id": pad_token_id,
                "variance_predictor_kernel_size": kernel_size,
                "upsample_rates": upsample_rates,
                "upsample_kernel_sizes": upsample_kernel_sizes,
                "upsample_initial_channel": config.upsample_initial_channel,
                "resblock_kernel_sizes": resblock_kernel_sizes,
                "resblock_dilation_sizes": resblock_dilation_sizes,
                "leaky_relu_slope": leaky_relu_slope,
                "dtype": "float32",
                "block": "code_hifigan_vocoder",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2CodeHifiGan",
            },
        },
        golden_path,
    )
    print(f"[code_hifigan_vocoder] saved golden to {golden_path}")
    return pcc


if __name__ == "__main__":
    pcc = test_code_hifigan_vocoder_matches_hf()
    print(f"\nFINAL PCC code_hifigan_vocoder: {pcc:.6f}")
