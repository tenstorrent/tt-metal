# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Gemma-4 vision rotary embedding (cos/sin generation).

Mirrors HF ``Gemma4VisionRotaryEmbedding``: 2D (x/y) RoPE where each spatial axis applies
1D RoPE over ``head_dim // 2`` channels, and the per-axis ``cos``/``sin`` are concatenated
into a ``head_dim``-wide table.

The reference computes, per axis ``i``::

    freqs_i = position_ids[..., i] (outer) inv_freq          # [.., spatial_dim//2]
    emb_i   = cat(freqs_i, freqs_i)                           # [.., spatial_dim]
    cos = cat(cos(emb_0), cos(emb_1));  sin = cat(sin(emb_0), sin(emb_1))   # [.., head_dim]

This is algebraically a single matmul: ``emb = position_ids @ freq_proj`` where
``freq_proj`` is the ``[2, head_dim]`` block-diagonal matrix whose row 0 holds the per-axis
``inv_freq`` in the first ``spatial_dim`` columns and row 1 holds it in the last ``spatial_dim``
columns. Then ``cos = cos(emb)``, ``sin = sin(emb)`` reproduce the concatenated tables. Folding
the per-axis duplication and concatenation into ``freq_proj`` (host-built) avoids the
non-tile-aligned on-device concats that the literal formulation would require.

Output layout: this produces the **Meta interleaved** convention (pairwise-duplicated:
``[c0, c0, c1, c1, ...]`` per spatial block), which is what the on-device attention's
``rotary_embedding_llama`` / ``apply_multidimensional_rope`` consumes. This is obtained for free
by duplicating ``inv_freq`` with ``repeat_interleave`` (rather than the reference's half-stacked
``cat(inv_freq, inv_freq)``), and equals ``convert_rope_style_hf_to_meta_md`` applied to the HF
cos/sin.

Note: no typecasts are inserted; cos/sin are produced in the working dtype. The reference forces
float32 for the trig (RoPE frequencies need precision), so if PCC is low the fix is to compute
``emb``/``cos``/``sin`` in float32.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class VisionRotaryEmbedding(LightweightModule):
    def __init__(self, mesh_device, args, dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.dtype = dtype

        vision_config = args.hf_config.vision_config
        head_dim = getattr(vision_config, "head_dim", None) or (
            vision_config.hidden_size // vision_config.num_attention_heads
        )
        base = vision_config.rope_parameters["rope_theta"]
        spatial_dim = head_dim // 2
        self.head_dim = head_dim
        self.attention_scaling = 1.0  # "default" rope_type

        # inv_freq over spatial_dim // 2 channels, identical for each spatial axis (matches reference).
        exponents = torch.arange(0, spatial_dim, 2, dtype=torch.int64).to(torch.float32) / spatial_dim
        inv_freq = 1.0 / (base**exponents)  # [spatial_dim // 2]
        # repeat_interleave (not cat) -> pairwise-duplicated freqs == Meta interleaved cos/sin layout.
        inv_freq_dup = torch.repeat_interleave(inv_freq, 2)  # [spatial_dim]

        # Block-diagonal [2, head_dim]: row 0 -> first spatial_dim cols (x axis), row 1 -> last (y axis).
        freq_proj = torch.zeros(2, head_dim, dtype=torch.float32)
        freq_proj[0, :spatial_dim] = inv_freq_dup
        freq_proj[1, spatial_dim:] = inv_freq_dup

        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.freq_proj = ttnn.as_tensor(
            freq_proj,
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, position_ids):
        """Generate the rotary cos/sin tables for the given patch positions.

        Args:
            position_ids: ttnn.Tensor ``[batch, num_patches, 2]`` patch (x, y) positions in the
                working dtype (e.g. bfloat16), TILE layout.

        Returns:
            cos, sin: ttnn.Tensor ``[1, batch, num_patches, head_dim]`` rotary tables.
        """
        pos = ttnn.unsqueeze_to_4D(position_ids)  # [1, batch, num_patches, 2]
        emb = ttnn.matmul(
            pos,
            self.freq_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
        )  # [1, batch, num_patches, head_dim]

        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)
        ttnn.deallocate(emb)

        if self.attention_scaling != 1.0:
            cos = ttnn.multiply(cos, self.attention_scaling)
            sin = ttnn.multiply(sin, self.attention_scaling)

        return cos, sin
