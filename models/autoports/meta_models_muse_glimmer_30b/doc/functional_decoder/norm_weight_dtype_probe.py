# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does keeping the folded ``(1 + w)`` norm weight in FP32 improve accuracy?

``MuseGlimmerTextCenteredRMSNorm`` computes ``rms_norm(x.float()) * (1 +
w).float()`` in FP32.  ``FunctionalDecoder.from_state_dict`` folds the ``1 +``
at setup and stores the result as BF16, which rounds each channel by up to
2^-9 relative.  This probe measures whether an FP32 device weight recovers that
error, using a synthetic ``w`` with the same magnitude as the real checkpoint's
centered-norm weights.

Result (Blackhole, this branch): it does not.  ``ttnn.rms_norm`` emits a BF16
output for a BF16 input either way, and the output rounding dominates the
weight rounding by ~3 orders of magnitude, so the FP32 weight buys ~1e-7 of
PCC.  The BF16 fold is kept.

Run with::

    python models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder/norm_weight_dtype_probe.py
"""

from __future__ import annotations

import torch

import ttnn

HIDDEN = 6656
SEQ = 256
EPS = 1e-5


def main() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        torch.manual_seed(0)
        x = torch.randn(1, 1, SEQ, HIDDEN)
        # magnitude matches the real layer-0 input_layernorm weight (mean 0.34, std 0.34)
        folded = 1.0 + (torch.randn(HIDDEN) * 0.34 + 0.34)
        reference = (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)) * folded

        x_dev = ttnn.from_torch(
            x.to(torch.bfloat16),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for label, ttnn_dtype, torch_dtype in (
            ("bfloat16 (shipped)", ttnn.bfloat16, torch.bfloat16),
            ("float32", ttnn.float32, torch.float32),
        ):
            weight = ttnn.from_torch(
                folded.to(torch_dtype).reshape(1, 1, 1, HIDDEN),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = ttnn.rms_norm(x_dev, weight=weight, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            actual = ttnn.to_torch(out).float()
            pcc = float(torch.corrcoef(torch.stack([actual.flatten(), reference.flatten()]))[0, 1])
            print(f"weight dtype {label:20s} -> PCC {pcc:.8f}, output dtype {out.dtype}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
