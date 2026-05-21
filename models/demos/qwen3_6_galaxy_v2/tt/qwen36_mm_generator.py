# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-MM-GEN: Qwen36MMGenerator — orchestrates the end-to-end multimodal forward.

Composes:
  - Qwen36MMPipeline       (preprocessor + vision_encoder + CPU embed splice)
  - M-RoPE TT cos/sin upload (per-request, eager only — no decode trace)
  - text decoder forward    (existing Qwen36Generator, with M-RoPE plumbed in)

This is a SKELETON / integration plan as of commit b1961e85f9b. The decoder-side
hook (passing pre-built cos/sin into the existing 64-layer text decoder's
attention forward) requires changes to:
  - tt/llama_rope.py — add a `qwen36_mrope_external_cos_sin` mode that bypasses
    the 1D gather and accepts a `[1, 1, S, partial_rotary_dim]` cos/sin pair
  - tt/llama_attention.py qwen3.6 branch — when rope_setup.is_qwen36_mrope_external,
    use the externally-provided cos/sin via `ttnn.experimental.rotary_embedding_llama`
    instead of computing from `get_qwen36_rm_rot_idxs(cur_pos)`
  - tt/llama_model.py — thread cos/sin tensors through the prefill forward
    signature
  - tt/qwen36_generator.py — add a `forward_multimodal_prefill(prompt, images, ...)`
    method that runs the pipeline, builds M-RoPE cos/sin, calls the modified
    prefill with the spliced embeddings + cos/sin

Risk: any change to the existing decoder forward path could break text-only
PCC / decode trace. The integration must be guarded by an explicit
multimodal-mode flag so the text-only fast path stays untouched.

For now, this class documents the API and orchestrates the CPU-side pieces.
The TT decoder forward is a TODO marked clearly.

Usage (when complete):
    gen = Qwen36MMGenerator(mesh_device, ...)
    output_ids = gen.generate(
        "<|vision_start|><|image_pad|><|vision_end|>Describe this",
        images=[pil_image],
        max_new_tokens=64,
    )
    text = gen.tokenizer.decode(output_ids[0])
"""

from __future__ import annotations

import torch
from PIL.Image import Image

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_pipeline import Qwen36MMPipeline
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import build_mrope_tt_tensors
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.tt_dit.parallel.manager import CCLManager


class Qwen36MMGenerator:
    """Multimodal generator skeleton — vision pipeline + (TODO) text decoder forward.

    Currently implemented:
      - `prepare_decoder_inputs`: runs full vision pipeline → fused embeddings
      - `build_mrope_tt_tensors`: builds M-RoPE cos/sin TT tensors per-request

    TODO (requires text-decoder modifications):
      - `prefill_multimodal`: call text decoder with fused embeddings + M-RoPE cos/sin
      - `decode_step`: single-token decode after prefill (uses M-RoPE for new tokens)
      - `generate`: full generation loop (prefill → repeated decode → tokens → text)
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        vision_model_args: Qwen36VisionModelArgs,
        *,
        text_embed_weight: torch.Tensor | None = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.vision_model_args = vision_model_args

        # qwen3.6 rope params for M-RoPE
        tc = vision_model_args.hf_config.text_config
        rp = tc.rope_parameters
        head_dim = tc.head_dim
        self._partial_rotary_dim = int(head_dim * rp["partial_rotary_factor"])
        self._mrope_section = rp["mrope_section"]
        self._rope_theta = rp["rope_theta"]

        self.pipeline = Qwen36MMPipeline(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            model_args=vision_model_args,
            text_embed_weight=text_embed_weight,
            dtype=dtype,
        )

        # Convenience: expose the tokenizer + processor for callers
        self.tokenizer = self.pipeline.preprocessor.processor.tokenizer
        self.processor = self.pipeline.preprocessor.processor

    def prepare_inputs(
        self,
        prompt: str,
        images: list[Image] | None = None,
    ):
        """CPU side: produce (Qwen36MMInputs, fused_embeddings)."""
        return self.pipeline.prepare_decoder_inputs(prompt, images=images)

    def build_rope_tensors(self, position_ids_3d: torch.Tensor):
        """Build M-RoPE cos/sin TT tensors for the given 3D position_ids.

        Shape: each cos/sin is `[1, 1, S, partial_rotary_dim=64]` replicated.
        Same format the existing qwen3.6 attention's `rot_mats` argument expects.
        """
        return build_mrope_tt_tensors(
            position_ids_3d,
            rope_theta=self._rope_theta,
            partial_rotary_dim=self._partial_rotary_dim,
            mrope_section=self._mrope_section,
            mesh_device=self.mesh_device,
        )

    def prefill_multimodal(self, prompt: str, images: list[Image] | None = None):
        """TODO: run vision pipeline + text decoder prefill with M-RoPE cos/sin.

        Implementation outline:
            1. inputs, fused_embeddings = self.prepare_inputs(prompt, images)
            2. cos_tt, sin_tt = self.build_rope_tensors(inputs.position_ids_3d)
            3. Upload fused_embeddings → mesh (replicated)
            4. Call text-decoder prefill with (embeddings_tt, cos_tt, sin_tt,
               attention_mask_tt). Decoder needs:
               - mode='multimodal_prefill' to skip the internal embed lookup
                 and accept pre-built embeddings
               - mode='multimodal_prefill' to use externally-provided
                 cos/sin instead of the 1D gather
            5. Return logits for the last position → sample next token

        Blocked on text-decoder forward modifications (see module docstring).
        """
        raise NotImplementedError("Decoder integration pending — see module docstring for the plan.")

    def generate(
        self,
        prompt: str,
        images: list[Image] | None = None,
        max_new_tokens: int = 64,
    ) -> list[int]:
        """TODO: full generation loop = prefill_multimodal + decode_step × max_new_tokens."""
        raise NotImplementedError("Decoder integration pending — see module docstring.")
