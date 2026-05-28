# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate the golden tensor for `conformer_convolution_module` and verify PCC vs. HF.

Builds an HF ``SeamlessM4Tv2ConformerConvolutionModule`` with the SeamlessM4T-v2-Large
default config (``hidden_size=1024``, ``conv_depthwise_kernel_size=31``,
``speech_encoder_hidden_act='swish'``, ``speech_encoder_dropout=0.0``), then compares
its forward pass against the standalone reference implementation in
``functional.conformer_convolution_module_forward``.

Saves the golden output to
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/conformer_convolution_module.pt``.
"""

import os
import sys

import torch
from transformers.models.seamless_m4t_v2.configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ConformerConvolutionModule

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import (  # noqa: E402
    conformer_convolution_module_forward,
)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return float((a @ b) / denom)


def _extract_state_dict(hf_module: SeamlessM4Tv2ConformerConvolutionModule) -> dict:
    """Pull the convolution-module sub-tensors into the dict the reference expects."""
    sd = {
        "layer_norm": {
            "weight": hf_module.layer_norm.weight.detach().clone(),
            "bias": hf_module.layer_norm.bias.detach().clone(),
        },
        "pointwise_conv1": {
            "weight": hf_module.pointwise_conv1.weight.detach().clone(),
        },
        "depthwise_conv": {
            "weight": hf_module.depthwise_conv.weight.detach().clone(),
        },
        "depthwise_layer_norm": {
            "weight": hf_module.depthwise_layer_norm.weight.detach().clone(),
            "bias": hf_module.depthwise_layer_norm.bias.detach().clone(),
        },
        "pointwise_conv2": {
            "weight": hf_module.pointwise_conv2.weight.detach().clone(),
        },
    }
    return sd


def main() -> int:
    torch.manual_seed(0)

    # SeamlessM4T-v2-Large config defaults relevant to this block.
    config = SeamlessM4Tv2Config()
    hidden_size = config.hidden_size  # 1024
    kernel_size = config.conv_depthwise_kernel_size  # 31
    eps = config.layer_norm_eps  # 1e-5

    assert hidden_size == 1024, f"unexpected hidden_size={hidden_size}"
    assert kernel_size == 31, f"unexpected kernel_size={kernel_size}"

    # Build the HF module in fp32 and freeze BN-like state via eval()
    # (dropout becomes a no-op; layer_norm stays the same in train/eval).
    hf_module = SeamlessM4Tv2ConformerConvolutionModule(config).to(torch.float32).eval()

    # Re-init each tensor so we exercise non-trivial random weights, not zeros.
    # (nn.LayerNorm initializes weight=1, bias=0 by default; randomize for stronger PCC test.)
    with torch.no_grad():
        hf_module.layer_norm.weight.normal_(mean=1.0, std=0.05)
        hf_module.layer_norm.bias.normal_(mean=0.0, std=0.05)
        hf_module.depthwise_layer_norm.weight.normal_(mean=1.0, std=0.05)
        hf_module.depthwise_layer_norm.bias.normal_(mean=0.0, std=0.05)
        hf_module.pointwise_conv1.weight.normal_(mean=0.0, std=0.02)
        hf_module.depthwise_conv.weight.normal_(mean=0.0, std=0.02)
        hf_module.pointwise_conv2.weight.normal_(mean=0.0, std=0.02)

    state_dict = _extract_state_dict(hf_module)

    # Spec'd input shape.
    batch, seq_len = 1, 128
    hidden_states = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32)

    # --- No-mask case ---
    with torch.no_grad():
        hf_out = hf_module(hidden_states.clone(), attention_mask=None)
    ref_out = conformer_convolution_module_forward(
        hidden_states.clone(),
        state_dict,
        kernel_size=kernel_size,
        eps=eps,
        attention_mask=None,
    )
    pcc_no_mask = pcc(ref_out, hf_out)
    max_abs_no_mask = (ref_out - hf_out).abs().max().item()
    print(
        f"[conformer_conv_module/no_mask] shapes: ref={tuple(ref_out.shape)} "
        f"hf={tuple(hf_out.shape)}  pcc={pcc_no_mask:.6f}  max_abs={max_abs_no_mask:.3e}"
    )

    # --- Masked case (zero out last 16 positions) ---
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    attention_mask[:, -16:] = 0
    with torch.no_grad():
        hf_out_masked = hf_module(hidden_states.clone(), attention_mask=attention_mask)
    ref_out_masked = conformer_convolution_module_forward(
        hidden_states.clone(),
        state_dict,
        kernel_size=kernel_size,
        eps=eps,
        attention_mask=attention_mask,
    )
    pcc_masked = pcc(ref_out_masked, hf_out_masked)
    max_abs_masked = (ref_out_masked - hf_out_masked).abs().max().item()
    print(f"[conformer_conv_module/masked]  pcc={pcc_masked:.6f}  max_abs={max_abs_masked:.3e}")

    # --- Verify causal boundary explicitly: first-position output must depend only on
    # the input at position 0 (because left padding fills the prior k-1 timesteps).
    # We perturb position seq_len//2 in the input and check that early outputs (before
    # the kernel can reach the perturbation) are unchanged. With kernel_size=31 and
    # perturbation at t=64, outputs at t in [0, 64-1] must be bit-identical.
    perturbed_in = hidden_states.clone()
    perturbed_in[:, seq_len // 2, :] += 5.0  # strong perturbation
    with torch.no_grad():
        hf_out_perturbed = hf_module(perturbed_in, attention_mask=None)
    # Pointwise_conv2 (k=1) means the only spatial mixing is the depthwise conv.
    # A change at t=64 can affect outputs at t in [64, 64+(k-1)] = [64, 94] due to
    # causal padding (kernel reaches BACK kernel_size-1 timesteps). So outputs
    # at t < 64 must be unchanged.
    causal_safe_diff = (hf_out_perturbed[:, : seq_len // 2, :] - hf_out[:, : seq_len // 2, :]).abs().max().item()
    print(f"[conformer_conv_module/causal]  max_abs_diff_pre_perturb (must be 0): " f"{causal_safe_diff:.3e}")
    assert causal_safe_diff < 1e-5, f"Causality violated: pre-perturbation outputs differ by {causal_safe_diff}"

    # Save golden artifacts (no-mask + masked) for downstream TTNN checks.
    golden_dir = os.path.join(os.path.dirname(__file__), "golden")
    os.makedirs(golden_dir, exist_ok=True)
    out_path = os.path.join(golden_dir, "conformer_convolution_module.pt")
    torch.save(
        {
            "input": hidden_states,
            "attention_mask": attention_mask,
            "state_dict": state_dict,
            "output": ref_out,
            "output_masked": ref_out_masked,
            "pcc_no_mask": pcc_no_mask,
            "pcc_masked": pcc_masked,
            "max_abs_no_mask": max_abs_no_mask,
            "max_abs_masked": max_abs_masked,
            "causal_safe_max_abs_diff": causal_safe_diff,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "hidden": hidden_size,
                "kernel_size": kernel_size,
                "eps": eps,
                "activation": "swish",
                "dtype": "float32",
                "block": "conformer_convolution_module",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2ConformerConvolutionModule",
            },
        },
        out_path,
    )
    print(f"saved golden -> {out_path}")

    pcc_min = min(pcc_no_mask, pcc_masked)
    if pcc_min < 0.99:
        print(f"FAIL: min PCC {pcc_min:.6f} < 0.99")
        return 1
    print(f"OK: min PCC {pcc_min:.6f} >= 0.99")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
