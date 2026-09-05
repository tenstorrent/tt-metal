# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Which `k_chunk_size` values does `sdpa_decode` accept and then get wrong?

The fused-decode-attention work found that `k_chunk_size = 32` passes validation on this
model's key widths and
returns a wrong answer -- PCC 0.016 at width 384 against 0.99998 at 128. That is enough
to fix the model and not enough to report upstream: "32 is bad, 128 is good" is an
anecdote. A useful report needs the boundary.

The op validates one thing about this parameter
(`sdpa_decode_device_operation.cpp:137`):

    mask_shape[3] % k_chunk_size == 0

The values that actually work are encoded nowhere in the op -- they live in a **test
helper**, `sdpa_test_utils.py:get_chunk_size`, which returns 128 for any sequence
between 129 and 1024 and then caps to the largest power of two dividing it. A caller
reading the API sees a free tuning knob.

So: sweep every divisor of two real key widths, score each against a torch golden, and
report which of them the op takes without complaint.

    python3 models/demos/cosyvoice/scripts/probe_sdpa_chunk_sweep.py
"""
from __future__ import annotations

import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

H, DK = 16, 64
CASES = (384, 448, 256, 512)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    cc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    scale = DK**-0.5

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    for w in CASES:
        torch.manual_seed(0)
        q_t = torch.randn(1, H, 1, DK) * 0.1
        k_t = torch.randn(1, H, w, DK) * 0.1
        v_t = torch.randn(1, H, w, DK) * 0.1
        bias_t = torch.randn(1, H, 1, w) * 0.1
        gold = (torch.softmax((q_t @ k_t.transpose(-1, -2) + bias_t) * scale, dim=-1) @ v_t).reshape(H, DK)

        k, v = dev(k_t), dev(v_t)
        q4 = dev(q_t.permute(0, 2, 1, 3).contiguous())
        bias = dev(bias_t.permute(0, 2, 1, 3).contiguous())

        # Every multiple of the 32-row tile that divides the key width -- i.e. every
        # value the op's own validation admits.
        divisors = [c for c in range(32, w + 1, 32) if w % c == 0]
        print(f"\n  key width {w}: {len(divisors)} chunk sizes pass validation")
        print(f"    {'k_chunk_size':>13}{'chunks':>8}{'PCC vs golden':>16}   verdict")
        for kc in divisors:
            prog = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=32,
                k_chunk_size=kc,
                exp_approx_mode=False,
            )
            try:
                out = ttnn.transformer.scaled_dot_product_attention_decode(
                    q4,
                    k,
                    v,
                    is_causal=False,
                    attn_mask=bias,
                    scale=scale,
                    program_config=prog,
                    compute_kernel_config=cc,
                )
            except Exception as exc:  # noqa: BLE001 -- a raise is the *good* outcome
                print(f"    {kc:>13}{w // kc:>8}{'':>16}   RAISED: {str(exc)[:60]}")
                continue
            p = pcc(ttnn.to_torch(out).float().reshape(H, DK), gold)
            verdict = "ok" if p > 0.99 else "**WRONG, silently**"
            print(f"    {kc:>13}{w // kc:>8}{p:>16.10f}   {verdict}")
            ttnn.deallocate(out)

        for t in (k, v, q4, bias):
            ttnn.deallocate(t)

    print("\n  Anything marked WRONG passed `mask_shape[3] % k_chunk_size == 0` and returned")
    print("  a bad answer with no exception, no warning and no NaN.")
    ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
