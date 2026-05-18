# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# TT ``PixtralVisionModel``: patch Conv (Unfold+linear), ``ln_pre``, RoPE, ``TtPixtralTransformer``.

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.tt.tt_pixtral_patch_conv import TtPixtralPatchConv
from models.experimental.devstarl2_small.tt.tt_pixtral_rotary_emb import TtPixtralRotaryEmbedding
from models.experimental.devstarl2_small.tt.tt_pixtral_transformer import TtPixtralTransformer
from models.experimental.devstarl2_small.tt.tt_pixtralnorm import TtPixtralRMSNorm


class TtPixtralVisionModel(LightweightModule):
    """Mirrors HF ``PixtralVisionModel`` compute path (Devstral / Pixtral checkpoints). ``patch_conv`` uses ``TtPixtralPatchConv`` (Unfold + ``ttnn.linear``, equivalent to HF ``nn.Conv2d`` patch embed). Pixel values must be torch ``[N,C,H,W]`` bf16 for the unfold step before device linear."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        configuration,
        weight_cache_path,
        dtype,
        vision_config,
        n_layers: int | None = None,
        vision_prefix: str = "vision_tower.",
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.patch_size = int(configuration.vision_patch_size)
        self.hidden_size = int(configuration.vision_dim)
        self.vision_prefix = vision_prefix

        pc_pref = f"{vision_prefix}patch_conv."
        self.patch_conv = TtPixtralPatchConv(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=pc_pref,
            dtype=dtype,
            in_channels=int(configuration.vision_in_channels),
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.ln_pre = TtPixtralRMSNorm(
            mesh_device,
            state_dict,
            eps=1e-5,
            weight_key=f"{vision_prefix}ln_pre.weight",
            dtype=dtype,
        )

        self.patch_positional_embedding = TtPixtralRotaryEmbedding(mesh_device, vision_config, datatype=dtype)

        self.transformer = TtPixtralTransformer(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            configuration=configuration,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            n_layers=n_layers,
            vision_prefix=f"{vision_prefix}transformer.layers.",
        )

    def forward(self, pixel_values, image_sizes: list[tuple[int, int]], position_ids_tt: ttnn.Tensor):
        """Args: pixel_values: torch ``[N,C,H,W]`` bf16 (patch conv unfold path). image_sizes: ``[(H,W), ...]`` per image (batch aligned with ``N``). position_ids_tt: device ids ``[1, seq]`` uint32, matching HF RoPE indexing."""
        patch_embeds = self.patch_conv(pixel_values)
        patch_embeds = ttnn.transpose(patch_embeds, 1, 2)
        bsz = patch_embeds.shape[0]
        h0, w0 = image_sizes[0]
        gh, gw = h0 // self.patch_size, w0 // self.patch_size
        patch_embeds = ttnn.reshape(patch_embeds, [bsz, self.hidden_size, gh, gw])

        patch_embeds_list = [
            ttnn.slice(
                patch_embeds,
                [0, 0, 0, 0],
                [bsz, self.hidden_size, sz[0] // self.patch_size, sz[1] // self.patch_size],
            )
            for sz in image_sizes
        ]

        reshaped = []
        for p in patch_embeds_list:
            p = ttnn.reshape(p, (1, self.hidden_size, -1))
            p = ttnn.transpose(p, 1, 2)
            reshaped.append(p)
        if len(reshaped) == 1:
            patch_embeds = reshaped[0]
        else:
            patch_embeds = ttnn.concat(reshaped, dim=0)
        patch_embeds = ttnn.reshape(
            patch_embeds,
            (1, 1, patch_embeds.shape[-2], patch_embeds.shape[-1]),
        )
        patch_embeds = self.ln_pre(patch_embeds)

        cos, sin = self.patch_positional_embedding(patch_embeds, position_ids_tt)
        return self.transformer(patch_embeds, attention_mask=None, position_embeddings=(cos, sin))


__all__ = ["TtPixtralVisionModel"]
