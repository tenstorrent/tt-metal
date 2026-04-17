# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Top-level TT model classes for Dots OCR.

Mirrors the layout of ``models/demos/qwen25_vl/tt/model.py``:

- ``DotsTransformer(TTTransformer)`` — text decoder subclass that overrides
  ``prepare_inputs_prefill`` to accept host-side embeddings (so we can fuse text + vision
  tokens on host before prefill) and ``_prepare_cos_sin`` to replicate host cos/sin to mesh.
- ``DropInVisionTransformer`` — a ``torch.nn.Module`` wrapper around the TT vision stack
  (``VisionTransformerTT``) with the same signature as the HF reference vision tower
  (``forward(pixel_values, grid_thw) -> torch.Tensor``), suitable to be dropped in place of
  ``reference_model.vision_tower`` in end-to-end demos.

The legacy ``DotsOCRModel`` / ``create_dots_ocr_model`` stubs have been removed; callers
should now build the pieces explicitly (``DotsModelArgs``, ``DotsTransformer``,
``DropInVisionTransformer``) and drive prefill/decode via
``models.tt_transformers.tt.generator.Generator`` the same way qwen25_vl's demo does.
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
    - Uses the standard ``Attention`` class and the default RoPE setup (Dots uses Qwen2-style
      single-position RoPE; no multimodal rope_deltas like qwen25_vl).
    - Overrides ``prepare_inputs_prefill`` so that ``tokens`` is actually a [B, S, D]
      **embedding tensor** (text + vision already fused on host). This is the same contract
      qwen25_vl's ``Transformer.prepare_inputs_prefill`` exposes.
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

    def prepare_inputs_prefill(self, tokens, rot_mats, start_pos=0, page_table=None, chunk_page_table=None):
        """
        Prepare prefill inputs where ``tokens`` is actually a **fused embedding tensor**.

        ``tokens`` shape: [B=1, S, D] (output of ``merge_vision_tokens`` + host embed).
        Returns: ``(tokens_embd, tt_rot_mats_prefill, tt_page_table, tt_chunk_page_table)``.
        """
        ttnn = get_ttnn()
        if ttnn is None:
            raise RuntimeError("ttnn is required for prepare_inputs_prefill")
        assert isinstance(rot_mats[0], torch.Tensor)
        assert isinstance(rot_mats[1], torch.Tensor)
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
    Drop-in replacement for the HF Dots vision tower.

    Mirrors ``qwen25_vl.tt.model.DropInVisionTransformer``:
    - Same interface: ``forward(pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor``.
    - Loads state_dict from the HF reference model (via ``standardize_hf_keys_multimodal`` +
      ``convert_hf_to_meta``) so the TT vision weights match the HF checkpoint.
    - Delegates the heavy compute to :class:`VisionTransformerTT`.

    Usage::

        reference_model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
        vision_model_args = DotsVisionModelArgs(mesh_device=mesh_device, hf_config=reference_model.config)
        visual = DropInVisionTransformer(reference_model, vision_model_args)
        image_embeds = visual(inputs.pixel_values, inputs.image_grid_thw)
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
            dtype = ttnn.bfloat8_b if ttnn is not None else torch.bfloat16
        self.dtype = dtype

        # Build a state_dict for the TT vision stack from the HF reference model.
        # Keep keys as-is so ``VisionTransformerTT`` can look up ``vision_tower.*``; the reference
        # model state_dict already uses those prefixes for Dots.
        try:
            from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys_multimodal

            hf_sd = reference_model.state_dict()
            hf_sd = standardize_hf_keys_multimodal(hf_sd)
            head_dim = getattr(model_args, "vision_head_dim", None) or getattr(model_args, "head_dim", 64)
            state_dict = convert_hf_to_meta(hf_sd, head_dim)
        except Exception as exc:  # pragma: no cover — bring-up fallback
            logger.warning(
                f"DropInVisionTransformer: falling back to raw HF state_dict (no key mapping). " f"Reason: {exc}"
            )
            state_dict = reference_model.state_dict()

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
