# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""TTNN SigLIP2-NaFlex vision tower + LFM2-VL multi-modal projector.

Encoder blocks reuse Gemma3's SigLIP transformer stack (identical structure after key
conversion). Embeddings are LFM-specific SigLIP2-NaFlex (Linear patch embed + bicubic
position-embedding resize). The projector applies pixel-unshuffle then a 2-layer MLP.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.multimodal.gemma3.tt.gemma_image_transformer import TtGemmaImageTransformer
from models.demos.multimodal.lfm25_vl.tt.multi_modal_projector import TtLfm2VlMultiModalProjector
from models.demos.multimodal.lfm25_vl.tt.siglip2_vision_embedding import TtSiglip2VisionEmbeddings
from models.tt_transformers.tt.multimodal.llama_layernorm import TtLayerNorm


class TtLfm25VlVisionModel(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        tt_ccl,
        dtype,
        configuration,
        weight_cache_path=None,
        use_host_vision: bool = False,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.configuration = configuration
        self.use_host_vision = use_host_vision
        self.dtype = dtype

        prefix = "model.vision_tower.vision_model."

        self.embeddings = TtSiglip2VisionEmbeddings(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{prefix}embeddings.",
            dtype=dtype,
            patch_size=configuration.vision_patch_size,
            num_channels=configuration.vision_in_channels,
            hidden_dim=configuration.vision_dim,
            num_patches=configuration.vision_num_patches,
            bias=True,
        )

        # Gemma image blocks read ``configuration.norm_eps``; temporarily point it at the
        # vision LayerNorm eps (1e-6) so SigLIP2 encoder norms match the HF tower.
        _text_norm_eps = configuration.norm_eps
        configuration.norm_eps = configuration.norm_eps_vision
        try:
            self.encoder = TtGemmaImageTransformer(
                mesh_device=mesh_device,
                state_dict=state_dict,
                tt_ccl=tt_ccl,
                state_dict_prefix=f"{prefix}encoder.",
                weight_cache_path=configuration.weight_cache_path(dtype)
                if weight_cache_path is None
                else weight_cache_path,
                dtype=dtype,
                configuration=configuration,
                layers=configuration.vision_n_layers,
                block_key="layers",
            )
        finally:
            configuration.norm_eps = _text_norm_eps

        self.ln_post = TtLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{prefix}ln_post.",
            weight_cache_path=configuration.weight_cache_path(dtype)
            if weight_cache_path is None
            else weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps_vision,
        )

        self.projector = TtLfm2VlMultiModalProjector(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix="model.multi_modal_projector",
            vision_dim=configuration.vision_dim,
            projector_hidden_size=configuration.projector_hidden_size,
            text_dim=configuration.dim,
            downsample_factor=configuration.downsample_factor,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
            bias=configuration.projector_bias,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Args:
            pixel_values: ``[B, C, H, W]`` or SigLIP2 patchified ``[B, N, C*P*P]``
            spatial_shapes: ``[B, 2]`` (required for patchified inputs)
            pixel_attention_mask: ``[B, N]`` optional; used to unpad before projection
        Returns:
            Flattened projected vision tokens ``[total_image_tokens, text_dim]`` as ttnn tensor
            with leading batch dim 1 for ConcatMeshToTensor readout (``[1, T, text_dim]``).
        """
        if self.use_host_vision:
            return self._forward_host(pixel_values, spatial_shapes, pixel_attention_mask)

        if pixel_values.ndim == 4 and spatial_shapes is None:
            _, _, height, width = pixel_values.shape
            grid_h = height // self.configuration.vision_patch_size
            grid_w = width // self.configuration.vision_patch_size
            spatial_shapes = torch.tensor([[grid_h, grid_w]] * pixel_values.shape[0], dtype=torch.long)

        embeddings = self.embeddings(pixel_values, spatial_shapes=spatial_shapes)
        bsz, seq_len, _ = embeddings.shape

        attention_mask = torch.zeros(bsz, 1, seq_len, seq_len)
        if pixel_attention_mask is not None:
            # Mask padding patch positions (False/0 = pad).
            valid = pixel_attention_mask.to(torch.bool)
            # [B, 1, 1, N] * [B, 1, N, 1] -> block invalid tokens
            row = valid[:, None, None, :].to(torch.float32)
            col = valid[:, None, :, None].to(torch.float32)
            attention_mask = attention_mask.masked_fill((row * col) == 0, -1e4)

        tt_mask = ttnn.from_torch(
            attention_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        encoded = self.encoder(embeddings, mask=tt_mask)
        encoded = self.ln_post(encoded)

        # Project each image independently (variable H/W after unpad), then concat tokens.
        host = ttnn.to_torch(encoded, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        # Encoder output may be 4D ([1, B, N, H]) and the composer stacks one identical replica
        # per device on dim 0; flatten leading dims to [num_replicas * B, N, H] and keep the
        # first replica's B images so per-image indexing below sees [N, H] features.
        host = host.reshape(-1, host.shape[-2], host.shape[-1])[:bsz]
        projected_parts = []
        for img_idx in range(bsz):
            feature = host[img_idx]
            if pixel_attention_mask is not None:
                length = int(pixel_attention_mask[img_idx].sum().item())
                feature = feature[:length]
            h, w = int(spatial_shapes[img_idx, 0]), int(spatial_shapes[img_idx, 1])
            feature = feature[: h * w].reshape(1, h, w, -1)
            from models.demos.multimodal.lfm25_vl.reference.functional import pixel_unshuffle

            unshuffled = pixel_unshuffle(feature, factor=self.configuration.downsample_factor)
            unshuffled = unshuffled.reshape(1, -1, unshuffled.shape[-1])
            projected_parts.append(self.projector.forward_sequence(unshuffled))

        # Collect host projections and return as a single ttnn tensor [1, T, dim]
        host_proj = []
        for part in projected_parts:
            host_proj.append(ttnn.to_torch(part, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0])
        combined = torch.cat(host_proj, dim=0).unsqueeze(0)  # [1, T, dim]
        return ttnn.from_torch(
            combined,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _forward_host(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None,
        pixel_attention_mask: torch.Tensor | None,
    ) -> ttnn.Tensor:
        hf_model = getattr(self.configuration, "cached_hf_model", None)
        if hf_model is None:
            raise RuntimeError(
                "use_host_vision=True requires ModelArgs(..., cache_hf=True) so a HF reference "
                "model is available on self.configuration.cached_hf_model."
            )
        with torch.no_grad():
            kwargs = {"pixel_values": pixel_values.float()}
            if spatial_shapes is not None:
                kwargs["spatial_shapes"] = spatial_shapes
            if pixel_attention_mask is not None:
                kwargs["pixel_attention_mask"] = pixel_attention_mask
            image_outputs = hf_model.model.get_image_features(**kwargs)
            features = image_outputs.pooler_output  # list of [T_i, dim]
            projected = torch.cat(features, dim=0).unsqueeze(0)
        return ttnn.from_torch(
            projected,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
