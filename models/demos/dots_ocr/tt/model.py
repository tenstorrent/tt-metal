# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Top-level TT model classes for Dots OCR.

- ``DotsTransformer(TTTransformer)`` — text decoder: for **token ids** ``[B,S]``, the parent
  ``prepare_inputs_prefill`` runs ``embd`` on device and slices **device** ``HfRotarySetup`` cos/sin.
  For **fused host embeddings** ``[B,S,D]``, ``prepare_inputs_prefill`` uploads embeds and optional
  host rot_mats (or uses device RoPE caches when ``rot_mats is None``).
- ``DropInVisionTransformer`` — TT vision path with the same call signature as HF
  ``model.vision_tower`` (``forward(pixel_values, grid_thw)``), built from checkpoint weights
  and :class:`~models.demos.dots_ocr.tt.vision_transformer.VisionTransformerTT`.

Callers build ``DotsModelArgs`` / ``DotsTransformer`` / optional ``DropInVisionTransformer`` and
drive decode via ``models.tt_transformers.tt.generator.Generator``.
"""

from __future__ import annotations

import torch
from loguru import logger

from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
from models.demos.dots_ocr.tt.vision_transformer import VisionTransformerTT
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.model import Transformer as TTTransformer

# ---------------------------------------------------------------------------
# Text decoder
# ---------------------------------------------------------------------------


class DotsTransformer(TTTransformer):
    """
    TT text decoder for Dots OCR.

    Thin subclass of ``tt_transformers`` ``Transformer`` that:
    - Uses the standard ``Attention`` class and RoPE setup for the **text** decoder.
    - Overrides ``prepare_inputs_prefill`` so ``tokens`` can be a [B, S, D] **embedding** tensor
      (text + vision fused on host), or [B, S] token ids (delegates to the parent).
    - Overrides ``_prepare_cos_sin`` to expand and replicate host cos/sin to the mesh.
    """

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=Attention,
            rope_setup_class=None,
        )

    def _prepare_cos_sin(self, rot_mats):
        """Expand host cos/sin to the batch dim and replicate onto the mesh."""
        ttnn = get_ttnn()
        if ttnn is None:
            raise RuntimeError("ttnn is required for _prepare_cos_sin")

        cos_matrix = rot_mats[0]
        sin_matrix = rot_mats[1]
        assert cos_matrix.shape[0] == sin_matrix.shape[0], "cos_matrix and sin_matrix must have the same batch size"

        outputs = []
        for mat in (cos_matrix, sin_matrix):
            outputs.append(
                ttnn.from_torch(
                    mat.expand(cos_matrix.shape[0], -1, -1, -1),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.rope_setup.datatype,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                ),
            )
        return outputs

    def prepare_inputs_prefill(self, tokens, rot_mats=None, start_pos=0, page_table=None, chunk_page_table=None):
        """
        Prepare prefill inputs where ``tokens`` is actually a **fused embedding tensor**.

        ``tokens`` shape: [B=1, S, D] (output of ``merge_vision_tokens`` + host embed).
        Returns: ``(tokens_embd, tt_rot_mats_prefill, tt_page_table, tt_chunk_page_table)``.
        """
        ttnn = get_ttnn()
        if ttnn is None:
            raise RuntimeError("ttnn is required for prepare_inputs_prefill")
        if rot_mats is not None:
            assert isinstance(rot_mats[0], torch.Tensor)
            assert isinstance(rot_mats[1], torch.Tensor)

        # If callers pass token ids [B, S] (text-only), delegate to the parent implementation
        # so embeddings and input layouts match the standard tt_transformers stack.
        if tokens.dim() == 2:
            return super().prepare_inputs_prefill(
                tokens,
                start_pos=start_pos,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
            )

        assert tokens.dim() == 3, "tokens should be embeddings [B, S, D]"
        S = tokens.shape[-2]

        tokens_embd = ttnn.from_torch(
            tokens.unsqueeze(1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape
            ),
        )

        if rot_mats is None:
            # Use the device-side caches generated by `HfRotarySetup` / `RotarySetup`.
            # This avoids subtle host-vs-device RoPE mismatches that can crater PCC.
            cos_matrix = self.rope_setup.cos_matrix_prefill
            sin_matrix = self.rope_setup.sin_matrix_prefill
        else:
            cos_matrix, sin_matrix = self._prepare_cos_sin(rot_mats=rot_mats)

        assert (
            cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {cos_matrix.shape[2]}"

        tt_rot_mats_prefill = [
            cos_matrix[:, :, start_pos : start_pos + S, :],
            sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill, tt_page_table, tt_chunk_page_table


# ---------------------------------------------------------------------------
# Vision tower drop-in wrapper
# ---------------------------------------------------------------------------


class DropInVisionTransformer(torch.nn.Module):
    """
    Drop-in for HF ``model.vision_tower`` (Dots OCR): same
    ``forward(pixel_values, grid_thw) -> torch.Tensor`` as the hub implementation.

    Loads weights from the reference model checkpoint (``standardize_hf_keys_multimodal`` +
    ``convert_hf_to_meta``) and runs :class:`VisionTransformerTT` (TTNN linears, device RoPE,
    TTNN SDPA, TTNN patch merger).
    """

    def __init__(
        self,
        reference_model,
        model_args,
        dtype=None,
        debug: bool = False,
    ):
        super().__init__()
        self.reference_model = reference_model
        self.model_args = model_args
        self.debug = debug

        ttnn = get_ttnn()
        if dtype is None:
            # Default to bf16 for correctness (PCC tests). Callers can still opt into bf8 for perf.
            dtype = ttnn.bfloat16 if ttnn is not None else torch.bfloat16
        self.dtype = dtype

        # Build a state_dict for the TT vision stack from the HF reference model.
        #
        # IMPORTANT: do not run `convert_hf_to_meta` here. That conversion is designed for the
        # *text* stack (decoder) and will remap/permute keys that do not apply to the HF vision
        # tower. For vision PCC we want the raw HF `vision_tower.*` tensors.
        try:
            from models.tt_transformers.tt.load_checkpoints import standardize_hf_keys_multimodal

            hf_sd = reference_model.state_dict()
            hf_sd = standardize_hf_keys_multimodal(hf_sd)
            # Filter to just the HF vision tower weights and normalize `model.` prefixes away.
            state_dict = {}
            for k, v in hf_sd.items():
                k2 = k[len("model.") :] if k.startswith("model.") else k
                if k2.startswith("vision_tower."):
                    state_dict[k2] = v
        except Exception as exc:  # pragma: no cover — bring-up fallback
            logger.warning(
                f"DropInVisionTransformer: falling back to raw HF state_dict (no key mapping). " f"Reason: {exc}"
            )
            # Best-effort filter for raw HF state dict too.
            raw = reference_model.state_dict()
            state_dict = {}
            for k, v in raw.items():
                k2 = k[len("model.") :] if k.startswith("model.") else k
                if k2.startswith("vision_tower."):
                    state_dict[k2] = v
            if not state_dict:
                state_dict = raw

        weight_cache_path = None
        if hasattr(model_args, "weight_cache_path"):
            try:
                weight_cache_path = model_args.weight_cache_path(dtype)
            except Exception:
                weight_cache_path = None

        self.tt_model = VisionTransformerTT(
            mesh_device=model_args.mesh_device,
            model_args=model_args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    @property
    def spatial_merge_size(self):
        return getattr(self.model_args, "spatial_merge_size", 2)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None) -> torch.Tensor:
        """
        Run the TT vision tower and return vision embeddings.

        Args:
            pixel_values: [N_patches, C, H, W] or flattened tensor as produced by the HF processor.
            grid_thw:     [N_images, 3] temporal/height/width grid from the processor.

        Returns:
            [N_image_tokens, out_hidden_size] torch tensor suitable for ``merge_vision_tokens``.
        """
        out = self.tt_model.forward(pixel_values, grid_thw)
        # Normalize shape to [N_image_tokens, D]
        if isinstance(out, torch.Tensor):
            if out.dim() == 4:  # [B, 1, S, D]
                out = out.squeeze(0).squeeze(0)
            elif out.dim() == 3:  # [B, S, D]
                out = out.reshape(-1, out.shape[-1])
        return out
