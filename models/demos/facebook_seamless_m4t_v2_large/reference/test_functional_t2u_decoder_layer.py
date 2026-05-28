# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``t2u_decoder_layer_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2TextToUnitDecoderLayer`` module using the v2-Large config at
shape ``[1, 64, 1024]`` (batch, seq_len, hidden). Saves a golden tensor for
downstream TTNN PCC verification.

Three paths are verified:
    1. Unmasked: no attention_mask, no padding_mask (the default HF
       NON-causal bidirectional configuration).
    2. With a triangular causal self-attention mask (no padding mask).
    3. With a padding mask zeroing the last 8 positions before each Conv1d
       AND a padding-aware additive log-mask in self-attention.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import t2u_decoder_layer_forward

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
    return {
        "weight": ln.weight.detach().clone(),
        "bias": ln.bias.detach().clone(),
    }


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


def _extract_t2u_decoder_layer_state_dict(layer) -> dict:
    """Pull all sub-block weights out of an HF SeamlessM4Tv2TextToUnitDecoderLayer."""
    return {
        "self_attn": _attn_sd(layer.self_attn),
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "conv1": _conv1d_sd(layer.conv1),
        "conv2": _conv1d_sd(layer.conv2),
        "conv_layer_norm": _ln_sd(layer.conv_layer_norm),
    }


def test_t2u_decoder_layer_matches_hf() -> float:
    """Compare reference t2u_decoder_layer forward against HuggingFace.

    Uses SeamlessM4T-v2-Large config:
        - hidden_size = 1024
        - decoder_attention_heads = 16 (head_dim = 64)
        - activation_function = 'relu'
        - layer_norm_eps = 1e-5
        - Conv1d kernel_size = 7, padding='same'
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2TextToUnitDecoderLayer

    torch.manual_seed(0)

    batch, seq_len, hidden = 1, 64, 1024
    num_heads, head_dim = 16, 64
    conv_kernel_size = 7

    config = SeamlessM4Tv2Config()
    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert config.decoder_attention_heads == num_heads, f"unexpected num_heads {config.decoder_attention_heads}"
    assert config.activation_function == "relu", f"unexpected activation_function {config.activation_function}"

    eps = config.layer_norm_eps  # 1e-5

    hf = SeamlessM4Tv2TextToUnitDecoderLayer(
        config,
        decoder_attention_heads=config.decoder_attention_heads,
        decoder_ffn_dim=config.decoder_ffn_dim,
    )
    hf.eval()
    # Be explicit: zero out the only dropout that could fire inside the layer.
    hf.conv_dropout.p = 0.0
    # Inner self-attention dropout (BART-style attn dropout; no-op at eval anyway).
    hf.self_attn.dropout = 0.0

    # Sanity-check kernel size & "same" padding setup matches expectations.
    assert hf.conv1.kernel_size == (conv_kernel_size,), f"unexpected conv1 kernel {hf.conv1.kernel_size}"
    assert hf.conv2.kernel_size == (conv_kernel_size,), f"unexpected conv2 kernel {hf.conv2.kernel_size}"
    assert hf.conv1.padding == "same", f"unexpected conv1 padding {hf.conv1.padding}"
    assert hf.conv2.padding == "same", f"unexpected conv2 padding {hf.conv2.padding}"

    state_dict = _extract_t2u_decoder_layer_state_dict(hf)

    hidden_states = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    # ---------------------------------------------------------------
    # Path 1: unmasked (default NON-causal bidirectional T2U setup).
    # ---------------------------------------------------------------
    with torch.no_grad():
        hf_out_tuple = hf(
            hidden_states=hidden_states,
            attention_mask=None,
            padding_mask=None,
            output_attentions=False,
        )
    hf_unmasked = hf_out_tuple[0]
    ref_unmasked = t2u_decoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        padding_mask=None,
        eps=eps,
        conv_kernel_size=conv_kernel_size,
        activation_function="relu",
    )
    pcc_unmasked = _pcc(ref_unmasked, hf_unmasked)
    max_abs_unmasked = (ref_unmasked - hf_unmasked).abs().max().item()
    print(f"[t2u_decoder_layer/unmasked]      pcc={pcc_unmasked:.6f} " f"max_abs_diff={max_abs_unmasked:.3e}")
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(
        ref_unmasked, hf_unmasked, atol=1e-5, rtol=1e-4
    ), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # ---------------------------------------------------------------
    # Path 2: triangular causal self-attention mask, no padding mask.
    # ---------------------------------------------------------------
    neg_inf = torch.finfo(torch.float32).min
    causal_mask = torch.full((seq_len, seq_len), neg_inf, dtype=torch.float32)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    self_attention_mask = causal_mask.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len).contiguous()

    with torch.no_grad():
        hf_out_tuple = hf(
            hidden_states=hidden_states,
            attention_mask=self_attention_mask,
            padding_mask=None,
            output_attentions=False,
        )
    hf_causal = hf_out_tuple[0]
    ref_causal = t2u_decoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=self_attention_mask,
        padding_mask=None,
        eps=eps,
        conv_kernel_size=conv_kernel_size,
        activation_function="relu",
    )
    pcc_causal = _pcc(ref_causal, hf_causal)
    max_abs_causal = (ref_causal - hf_causal).abs().max().item()
    print(f"[t2u_decoder_layer/causal]        pcc={pcc_causal:.6f} " f"max_abs_diff={max_abs_causal:.3e}")
    assert pcc_causal > 0.99, f"causal PCC {pcc_causal} <= 0.99"
    assert torch.allclose(ref_causal, hf_causal, atol=1e-5, rtol=1e-4), f"causal diverged: max_abs={max_abs_causal}"

    # ---------------------------------------------------------------
    # Path 3: padding mask + padding-aware attention mask.
    # Mark the last 8 positions as padded.
    # ---------------------------------------------------------------
    padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    padding_mask[:, -8:] = False
    # Build an additive log-mask consistent with the padding_mask (mask K-side
    # padded positions across all queries).
    pad_attention_mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=torch.float32)
    pad_attention_mask[:, :, :, -8:] = neg_inf

    with torch.no_grad():
        hf_out_tuple = hf(
            hidden_states=hidden_states,
            attention_mask=pad_attention_mask,
            padding_mask=padding_mask,
            output_attentions=False,
        )
    hf_padded = hf_out_tuple[0]
    ref_padded = t2u_decoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=pad_attention_mask,
        padding_mask=padding_mask,
        eps=eps,
        conv_kernel_size=conv_kernel_size,
        activation_function="relu",
    )
    pcc_padded = _pcc(ref_padded, hf_padded)
    max_abs_padded = (ref_padded - hf_padded).abs().max().item()
    print(f"[t2u_decoder_layer/padded]        pcc={pcc_padded:.6f} " f"max_abs_diff={max_abs_padded:.3e}")
    assert pcc_padded > 0.99, f"padded PCC {pcc_padded} <= 0.99"
    assert torch.allclose(ref_padded, hf_padded, atol=1e-5, rtol=1e-4), f"padded diverged: max_abs={max_abs_padded}"

    # Save golden tensor for downstream TTNN PCC checks.
    golden_path = GOLDEN_DIR / "t2u_decoder_layer.pt"
    torch.save(
        {
            "input": hidden_states,
            "self_attention_mask": self_attention_mask,
            "pad_attention_mask": pad_attention_mask,
            "padding_mask": padding_mask,
            "state_dict": state_dict,
            "output_unmasked": ref_unmasked,
            "output_causal": ref_causal,
            "output_padded": ref_padded,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "hidden": hidden,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "conv_kernel_size": conv_kernel_size,
                "activation_function": "relu",
                "eps": eps,
                "dtype": "float32",
                "block": "t2u_decoder_layer",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2TextToUnitDecoderLayer",
            },
        },
        golden_path,
    )
    print(f"[t2u_decoder_layer] saved golden to {golden_path}")
    return min(pcc_unmasked, pcc_causal, pcc_padded)


if __name__ == "__main__":
    pcc = test_t2u_decoder_layer_matches_hf()
    print(f"\nFINAL PCC t2u_decoder_layer: {pcc:.6f}")
