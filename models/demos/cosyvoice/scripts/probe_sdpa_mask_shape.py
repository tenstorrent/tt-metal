# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Which property of the real mask breaks `sdpa_decode`?

`probe_sdpa_decode.py` measured the fused decode path at PCC 0.99998 and 3.3x.
Wired into the model it gives 0.88. The arithmetic is the same, so the difference
is in the *mask*, and the probe's mask differs from the model's in two ways at once:

  probe   last 32 of 384 suppressed, value -1e4
  model   first 174 of 384 suppressed, value -1e9   (`right_aligned_bias`, `NEG_INF`)

Either could matter, for reasons that are not the same:

**The value.** The kernel computes `exp((QK - row_max) * scale)`
(`sdpa_flash_decode.cpp:435`). At -1e9 and scale 0.125 the argument is -1.25e8;
at -1e4 it is -1250. Both are zero in exact arithmetic, but only one of them is
inside the range an SFPU exponential is built for.

**The position.** A right-aligned cache suppresses a *prefix*, so with
`k_chunk_size = 128` the first chunk is **entirely masked** -- its row max is the
mask value itself, its local softmax is uniform over 128 dead slots, and only the
cross-chunk rescale suppresses it. The probe's trailing mask never produced a fully
dead chunk, so it never exercised that path.

One variable at a time, four cells, against a torch golden.

    python3 models/demos/cosyvoice/scripts/probe_sdpa_mask_shape.py
"""
from __future__ import annotations

import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

H, DK, W = 16, 64, 384
VALID = 210  # what a real step looks like: 174 suppressed slots, all of them leading


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

    torch.manual_seed(0)
    qu_t = torch.randn(1, H, 1, DK) * 0.1
    k_t = torch.randn(1, H, W, DK) * 0.1
    v_t = torch.randn(1, H, W, DK) * 0.1
    bd_t = torch.randn(1, H, 1, W) * 0.1
    qu, k, v = dev(qu_t), dev(k_t), dev(v_t)
    bd_p = dev(bd_t.permute(0, 2, 1, 3).contiguous())
    q4 = dev(qu_t.permute(0, 2, 1, 3).contiguous())

    print(f"\n  {'suppressed':>22}{'value':>10}{'k_chunk':>9}{'PCC vs golden':>16}")
    print("  " + "-" * 58)

    for where in ("trailing 32 (probe)", "leading 174 (model)"):
        for neg in (-1e4, -1e9):
            for kc in (128, 32):
                mask_t = torch.zeros(1, 1, 1, W)
                if where.startswith("trailing"):
                    mask_t[..., W - 32 :] = neg
                else:
                    mask_t[..., : W - VALID] = neg

                gold = (torch.softmax((qu_t @ k_t.transpose(-1, -2) + bd_t) * scale + mask_t, dim=-1) @ v_t).reshape(
                    H, DK
                )

                mask_h = dev(mask_t.expand(1, 1, H, W).contiguous())
                bias = ttnn.add(bd_p, mask_h)
                prog = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                    q_chunk_size=32,
                    k_chunk_size=kc,
                    exp_approx_mode=False,
                )
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
                p = pcc(ttnn.to_torch(out).float().reshape(H, DK), gold)
                print(f"  {where:>22}{neg:>10.0e}{kc:>9}{p:>16.10f}")
                for tns in (out, bias, mask_h):
                    ttnn.deallocate(tns)

    print("\n  Value alone matters   -> clamp the mask fed to the fused path.")
    print("  Position alone        -> a fully-masked k-chunk is the bug; avoid or report it.")
    ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
