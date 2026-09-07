# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 RMSNorm.

Plain RMSNorm, matching HF ``Ministral3RMSNorm`` exactly:
``out = weight * (x * rsqrt(mean(x^2) + eps))``. There is no Gemma ``(1 + w)`` fold and no
``use_gemma_norm`` key in the config; the flag survives from the donor only so the absence is
explicit and testable.

**Why this is composed rather than one ``ttnn.rms_norm`` call.** The fixed reference
(``gpt_oss_d_p/tt/rms_norm.py``) calls ``ttnn.rms_norm`` on the full-width replicated residual, which
works at gpt-oss's ``hidden_size`` of 2880 and does NOT work at Mistral's 12288: the interleaved
layernorm kernel keeps whole normalized rows in circular buffers, and at 12288 those grow to
2565120 B against a 1572864 B L1 budget —

    TT_THROW: Statically allocated dataflow buffers on core range [0-0 - 0-0] grow to 2565120 B
              which is beyond max L1 size of 1572864 B

— which is a hard throw, not a slow path. The sibling models dodge it by norming an ``emb/tp``
shard (M3's distributed norm at 6144/4 = 1536 wide), but that presumes an ``emb/tp``-sharded
residual stream, and reshaping the whole model's residual for it is exactly the
sharding/memory optimisation bring-up is not meant to take on.

So the norm is written out of the ttnn ops that DO exist, per the recipe's "compose, do not fall
back" rule — an RMSNorm is a square, a mean, an rsqrt and two multiplies:

    sq     = x * x                    (accumulated in fp32)
    ms     = mean(sq, dim=-1)         (reduction is tile-streaming: width-independent L1 cost)
    scale  = rsqrt(ms + eps)
    out    = (x * scale) * weight

``ttnn.mean`` reduces along W tile-by-tile, so its L1 footprint does not grow with the row, and the
composition runs at any width. Measured against the fp32 torch reference at hidden 12288: PCC
0.99999 at 32 / 128 / 512 tokens — better than the single-op kernel's usual bf16 result, because the
sum of squares is accumulated in fp32 while the donor's kernel does it in bf16.

The mesh interface is unchanged from the donor: the gain is REPLICATED (every TP column normalizes
the same full-width vector, so the output is already replicated) and cached under the same
``weight`` key, so the tilized weight cache layout is the donor's.
"""

from torch import nn

import ttnn
from models.demos.mistral_3_5_d_p.tt.config import MeshConfig
from models.demos.mistral_3_5_d_p.utils.general_utils import get_cache_file_name


class RMSNorm(nn.Module):
    """Plain RMSNorm over the last dim, composed from ttnn primitives (see the module docstring)."""

    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None):
        super().__init__()
        # Mistral is a plain RMSNorm (out = x_normed * weight). The Gemma-style
        # (out = x_normed * (1 + weight)) fold is kept behind use_gemma_norm but defaults off, and
        # tests/unit/test_norm_vs_ref.py asserts it stays off.
        self.use_gemma_norm = bool(getattr(hf_config, "use_gemma_norm", False))
        if state_dict:
            weight = state_dict["weight"]
            if self.use_gemma_norm:
                weight = weight.float() + 1.0
            # [1, 1, 1, width]: the gain multiplies the normalized activation directly, so it is held
            # in the activation's own layout rather than the donor's [1, 1, -1, TILE_SIZE] packing
            # (which the single-op kernel required).
            torch_weight = weight.reshape(1, 1, 1, -1)
        else:
            torch_weight = None

        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device

    def forward(self, x):
        """x: [.., tokens, width] (replicated across TP) -> the same shape, normalized and scaled."""
        # Square in fp32 so the width-12288 sum of squares does not lose the small channels; the
        # reduction is the only place the extra precision matters, and `sq` is freed immediately.
        sq = ttnn.multiply(x, x, dtype=ttnn.float32)
        mean_sq = ttnn.mean(sq, dim=-1, keepdim=True)
        sq.deallocate(True)
        scale = ttnn.rsqrt(ttnn.add(mean_sq, self.eps))
        mean_sq.deallocate(True)
        normed = ttnn.multiply(x, scale, dtype=ttnn.bfloat16)
        scale.deallocate(True)
        out = ttnn.multiply(normed, self.tt_weight, dtype=ttnn.bfloat16)
        normed.deallocate(True)
        return out
