# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``t2u_decoder_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2TextToUnitDecoder`` module at v2-Large hidden_size=1024 and
num_heads=16 (head_dim=64), but with the layer stack REDUCED TO 2 layers for
fast bring-up.

The HF parent ``SeamlessM4Tv2TextToUnitForConditionalGeneration`` strips the
``t2u_`` prefix from every config field before constructing the sub-model
(see modeling_seamless_m4t_v2.py line ~2200), so to instantiate the decoder
directly we must mirror that: copy ``t2u_*`` -> ``*`` (in particular
``t2u_decoder_layers`` -> ``decoder_layers``,
``t2u_variance_predictor_{embed,hidden}_dim`` ->
``variance_predictor_{embed,hidden}_dim``,
``t2u_variance_pred_dropout`` -> ``variance_pred_dropout``,
``t2u_pad_token_id`` -> ``pad_token_id``, etc.) BEFORE calling the
constructor.

A small synthetic case is used: ``char_seq_len = 8``, ``encoder_seq_len = 4``
(so each encoder step expands to 2 chars), then duration prediction
upsamples again to unit resolution. PCC > 0.99 is required.

The golden tensor is saved to ``reference/golden/t2u_decoder.pt`` for the
downstream TTNN PCC check.
"""

import copy
from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import (
    build_sinusoidal_positional_embedding_weights,
    t2u_decoder_forward,
)

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


def _ln_sd(ln) -> dict:
    return {"weight": ln.weight.detach().clone(), "bias": ln.bias.detach().clone()}


def _linear_sd(lin) -> dict:
    sd = {"weight": lin.weight.detach().clone()}
    if lin.bias is not None:
        sd["bias"] = lin.bias.detach().clone()
    return sd


def _attn_sd(attn) -> dict:
    return {
        "q_proj": _linear_sd(attn.q_proj),
        "k_proj": _linear_sd(attn.k_proj),
        "v_proj": _linear_sd(attn.v_proj),
        "out_proj": _linear_sd(attn.out_proj),
    }


def _conv1d_sd(conv) -> dict:
    sd = {"weight": conv.weight.detach().clone()}
    if conv.bias is not None:
        sd["bias"] = conv.bias.detach().clone()
    return sd


def _extract_variance_predictor_sd(module) -> dict:
    return {
        "conv1": _conv1d_sd(module.conv1),
        "ln1": _ln_sd(module.ln1),
        "conv2": _conv1d_sd(module.conv2),
        "ln2": _ln_sd(module.ln2),
        "proj": _linear_sd(module.proj),
    }


def _extract_t2u_decoder_layer_sd(layer) -> dict:
    return {
        "self_attn": _attn_sd(layer.self_attn),
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "conv1": _conv1d_sd(layer.conv1),
        "conv2": _conv1d_sd(layer.conv2),
        "conv_layer_norm": _ln_sd(layer.conv_layer_norm),
    }


def _extract_t2u_decoder_sd(decoder) -> dict:
    return {
        "embed_char": {"weight": decoder.embed_char.weight.detach().clone()},
        "pos_emb_alpha_char": decoder.pos_emb_alpha_char.detach().clone(),
        "pos_emb_alpha": decoder.pos_emb_alpha.detach().clone(),
        "duration_predictor": _extract_variance_predictor_sd(decoder.duration_predictor),
        "layers": [_extract_t2u_decoder_layer_sd(layer) for layer in decoder.layers],
        "layer_norm": _ln_sd(decoder.layer_norm),
    }


def _build_t2u_config(num_layers: int):
    """Build a SeamlessM4Tv2Config with the t2u_-> "" prefix strip applied.

    Mirrors what ``SeamlessM4Tv2TextToUnitForConditionalGeneration.__init__``
    does to its config before constructing the sub-model, then forces the
    decoder layer count down to ``num_layers`` for the small bring-up test.
    """
    from transformers import SeamlessM4Tv2Config

    cfg = SeamlessM4Tv2Config()
    cfg = copy.deepcopy(cfg)
    # Mirror HF's t2u_ -> "" prefix strip (see SeamlessM4Tv2TextToUnitForConditionalGeneration.__init__).
    for param, val in cfg.to_dict().items():
        if param.startswith("t2u_"):
            setattr(cfg, param[4:], val)
    # Reduce layer count to keep the bring-up test cheap.
    cfg.decoder_layers = num_layers
    cfg.t2u_decoder_layers = num_layers
    # Sanity-check the attributes the HF T2U decoder constructor needs are present.
    assert hasattr(cfg, "variance_predictor_embed_dim"), "config missing variance_predictor_embed_dim"
    assert hasattr(cfg, "variance_predictor_hidden_dim"), "config missing variance_predictor_hidden_dim"
    assert hasattr(cfg, "variance_pred_dropout"), "config missing variance_pred_dropout"
    return cfg


def test_t2u_decoder_matches_hf() -> float:
    """Compare reference t2u_decoder forward against HuggingFace (2 layers, small input)."""
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2TextToUnitDecoder

    torch.manual_seed(0)

    num_layers = 2
    batch = 1
    encoder_seq_len = 4
    char_seq_len = 8  # = encoder_seq_len * 2, so each text-token expands to 2 chars
    hidden = 1024
    num_heads = 16
    head_dim = 64
    conv_kernel_size = 7
    variance_predictor_kernel_size = 3

    config = _build_t2u_config(num_layers=num_layers)

    assert config.hidden_size == hidden
    assert config.decoder_attention_heads == num_heads
    assert config.activation_function == "relu"
    assert config.variance_predictor_kernel_size == variance_predictor_kernel_size

    eps = config.layer_norm_eps  # 1e-5
    embed_scale = hidden**0.5  # scale_embedding=True -> sqrt(hidden_size)
    padding_idx = config.pad_token_id  # After t2u_-strip this becomes 1.
    assert padding_idx == 1, f"unexpected padding_idx {padding_idx} (expected 1 after t2u_ strip)"

    # Build HF decoder and put it in eval mode (kill every dropout that could fire).
    hf = SeamlessM4Tv2TextToUnitDecoder(config)
    hf.eval()
    hf.dropout = 0.0  # outer dropout (no-op at eval anyway)
    hf.duration_predictor.dropout_module.p = 0.0
    for layer in hf.layers:
        layer.conv_dropout.p = 0.0
        layer.self_attn.dropout = 0.0
    assert len(hf.layers) == num_layers, f"unexpected layer count {len(hf.layers)}"

    # Inputs.
    char_input_ids = torch.randint(
        low=2,
        high=config.char_vocab_size,
        size=(batch, char_seq_len),
        dtype=torch.long,
    )
    # char_count_per_id: each encoder-step expands to 2 chars; sum across encoder = char_seq_len.
    char_count_per_id = torch.full(
        (batch, encoder_seq_len),
        char_seq_len // encoder_seq_len,
        dtype=torch.long,
    )
    encoder_hidden_states = torch.randn(batch, encoder_seq_len, hidden, dtype=torch.float32)

    # Run HF forward (non-return_dict tuple form).
    with torch.no_grad():
        hf_out = hf(
            char_input_ids=char_input_ids,
            char_count_per_id=char_count_per_id,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    hf_last_hidden = hf_out.last_hidden_state
    hf_padding_mask = hf_out.padding_mask

    # Extract weights for the standalone reference.
    state_dict = _extract_t2u_decoder_sd(hf)

    # Build sinusoidal positional weight tables. The decoder uses the same
    # `padding_idx` for both char-level and unit-level positional embeds.
    # Char-side: max char_seq_len.
    char_positional_weights = build_sinusoidal_positional_embedding_weights(
        num_embeddings=padding_idx + 1 + char_seq_len + 1,
        embedding_dim=hidden,
        padding_idx=padding_idx,
    )
    # Unit-side: we need padding_idx + 1 + upsampled_unit_len rows. Take a
    # generous upper bound by inspecting HF's dur_out (we re-derive it
    # inside the reference too, but here we just want the table size).
    with torch.no_grad():
        # Re-run HF duration prediction to learn the upsampled length cap.
        # This is purely for sizing the positional table -- the reference
        # will reproduce the same durations end-to-end.
        char_padding_mask_for_size = (
            torch.arange(char_seq_len).expand(batch, -1) < char_count_per_id.sum(1, keepdim=True)
        ).float()
        # Replicate the duration math without leaking it into the reference call.
        char_hidden_for_size = torch.repeat_interleave(encoder_hidden_states, char_count_per_id.view(-1), dim=1)
        log_dur_pred = hf.duration_predictor(
            hf.embed_char(char_input_ids) * embed_scale
            + hf.pos_emb_alpha_char * hf.embed_char_positions(inputs_embeds=char_hidden_for_size)
            + char_hidden_for_size,
            padding_mask=char_padding_mask_for_size,
        )
        dur_out_for_size = torch.clamp(torch.round(torch.expm1(log_dur_pred)).long(), min=1)
        dur_out_for_size = dur_out_for_size.masked_fill(~char_padding_mask_for_size.bool(), 0)
        max_unit_len = int(dur_out_for_size.sum(1).max().item())
    positional_weights = build_sinusoidal_positional_embedding_weights(
        num_embeddings=padding_idx + 1 + max_unit_len + 1,
        embedding_dim=hidden,
        padding_idx=padding_idx,
    )

    # Run the standalone reference.
    ref_out = t2u_decoder_forward(
        char_input_ids=char_input_ids,
        char_count_per_id=char_count_per_id,
        encoder_hidden_states=encoder_hidden_states,
        state_dict=state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        embed_scale=embed_scale,
        char_positional_weights=char_positional_weights,
        positional_weights=positional_weights,
        padding_idx=padding_idx,
        eps=eps,
        activation_function="relu",
        variance_predictor_kernel_size=variance_predictor_kernel_size,
        conv_kernel_size=conv_kernel_size,
    )
    ref_last_hidden = ref_out["last_hidden_state"]
    ref_padding_mask = ref_out["padding_mask"]
    ref_dur_out = ref_out["dur_out"]

    # Shape / padding-mask parity sanity checks before PCC.
    assert (
        ref_last_hidden.shape == hf_last_hidden.shape
    ), f"shape mismatch: ref={tuple(ref_last_hidden.shape)} hf={tuple(hf_last_hidden.shape)}"
    assert torch.equal(
        ref_padding_mask.to(hf_padding_mask.dtype), hf_padding_mask
    ), "padding_mask diverged between ref and hf"

    # PCC + tight allclose on hidden states.
    pcc = _pcc(ref_last_hidden, hf_last_hidden)
    max_abs = (ref_last_hidden - hf_last_hidden).abs().max().item()
    print(f"[t2u_decoder/last_hidden_state] pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")
    assert pcc > 0.99, f"t2u_decoder PCC {pcc} <= 0.99"
    assert torch.allclose(
        ref_last_hidden, hf_last_hidden, atol=1e-4, rtol=1e-4
    ), f"t2u_decoder last_hidden_state diverged: max_abs={max_abs}"

    # Save golden tensor for downstream TTNN PCC checks.
    golden_path = GOLDEN_DIR / "t2u_decoder.pt"
    torch.save(
        {
            "char_input_ids": char_input_ids,
            "char_count_per_id": char_count_per_id,
            "encoder_hidden_states": encoder_hidden_states,
            "state_dict": state_dict,
            "char_positional_weights": char_positional_weights,
            "positional_weights": positional_weights,
            "output_last_hidden_state": ref_last_hidden,
            "output_padding_mask": ref_padding_mask,
            "output_dur_out": ref_dur_out,
            "config": {
                "batch": batch,
                "encoder_seq_len": encoder_seq_len,
                "char_seq_len": char_seq_len,
                "hidden": hidden,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "num_layers": num_layers,
                "conv_kernel_size": conv_kernel_size,
                "variance_predictor_kernel_size": variance_predictor_kernel_size,
                "activation_function": "relu",
                "eps": eps,
                "embed_scale": embed_scale,
                "padding_idx": padding_idx,
                "dtype": "float32",
                "block": "t2u_decoder",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2TextToUnitDecoder",
                "note": "reduced to 2 layers for fast bring-up",
            },
        },
        golden_path,
    )
    print(f"[t2u_decoder] saved golden to {golden_path}")
    return pcc


if __name__ == "__main__":
    pcc = test_t2u_decoder_matches_hf()
    print(f"\nFINAL PCC t2u_decoder: {pcc:.6f}")
