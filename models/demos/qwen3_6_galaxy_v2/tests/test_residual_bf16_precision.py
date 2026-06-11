# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit test (pure torch, no device) for the residual-add precision hypothesis.

Op-level probe (test_qwen36_perlayer_pcc.py QWEN36_OP_PROBE) found, in prefill:
  attn_norm(x)            = 1.0000   (norm of the raw residual is clean)
  ff_norm(x + attn_out)   = 0.9824   (norm of residual + attention delta drops)
The norm op itself is HiFi4/fp32-acc, so the suspect is the **bf16 residual add**:
h1 = x + attn_out done in bf16 loses the small delta's low bits (cancellation),
then the (high-precision) RMSNorm faithfully normalizes the already-degraded h1.

This reproduces that in torch:
  - norm(x) bf16 vs fp32        -> expect ~1.0 (matches attn_norm)
  - norm(x + d) bf16 vs fp32    -> expect a drop (matches ff_norm) if hypothesis holds
  - norm(x + d) with fp32 add   -> expect recovery (the candidate fix)
across realistic residual:delta magnitude ratios.

Run: python -m pytest models/demos/qwen3_6_galaxy_v2/tests/test_residual_bf16_precision.py -s
"""
import torch


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rmsnorm(x, w, eps=1e-6):
    # zero-centered Qwen3NextRMSNorm: (1 + w) * x / sqrt(mean(x^2) + eps). Compute in fp32.
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * (1.0 + w.float())


def test_residual_bf16_precision():
    torch.manual_seed(0)
    S, H = 128, 5120
    w = torch.randn(H) * 0.02  # small norm weight perturbation, like trained gamma

    print("\n  ratio(|x|/|d|) | PCC norm(x) | PCC norm(x+d) bf16-add | PCC norm(x+d) fp32-add")
    print("  " + "-" * 80)
    for delta_scale in (1.0, 0.3, 0.1, 0.03, 0.01):
        x_fp32 = torch.randn(S, H)  # residual stream, unit-ish scale
        d_fp32 = torch.randn(S, H) * delta_scale  # attention/MLP delta, smaller

        # bf16 storage of inputs (the model carries residual + delta in bf16)
        x_bf16 = x_fp32.bfloat16()
        d_bf16 = d_fp32.bfloat16()

        # (1) norm of the raw residual: bf16 vs fp32  -> should match attn_norm ~1.0
        pcc_x = _pcc(_rmsnorm(x_bf16, w), _rmsnorm(x_fp32, w))

        # (2) norm(x + d) with the ADD done in bf16 (current model: ttnn.add bf16, no upcast)
        h_bf16add = (x_bf16 + d_bf16).bfloat16()  # bf16 + bf16 -> bf16
        pcc_bf16 = _pcc(_rmsnorm(h_bf16add, w), _rmsnorm(x_fp32 + d_fp32, w))

        # (3) candidate fix: ADD in fp32 (inputs still bf16-quantized, but no add cancellation)
        h_fp32add = x_bf16.float() + d_bf16.float()
        pcc_fp32 = _pcc(_rmsnorm(h_fp32add, w), _rmsnorm(x_fp32 + d_fp32, w))

        ratio = x_fp32.abs().mean().item() / max(d_fp32.abs().mean().item(), 1e-9)
        print(f"  {ratio:13.1f} | {pcc_x:11.5f} | {pcc_bf16:22.5f} | {pcc_fp32:.5f}")


if __name__ == "__main__":
    test_residual_bf16_precision()
