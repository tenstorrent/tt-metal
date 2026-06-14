# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""V4 multimodal generator: vision pipeline + text decoder forward with M-RoPE.

Wires together:
  - Qwen36MMPipeline (preprocessor + vision_encoder + CPU embed splice)
  - M-RoPE TT cos/sin upload (per-request, eager — no decode trace per VLM rule)
  - Text decoder forward (TtTransformer with mode="prefill" or "decode")

Insight: the existing 64-layer text decoder already accepts external cos/sin
and pre-embedded hidden state via `model.forward(x, rot_mats=..., mode="prefill")`
— exact precedent in `tests/test_64layer_full_pcc.py`. No decoder code
changes are needed; we just bypass `ttnn_prefill_forward`'s embed lookup
and pass our fused embeddings + M-RoPE cos/sin straight in.

Decode after prefill: text tokens after the last vision token advance t/h/w
position axes by 1 each. Since the M-RoPE section interleaves t/h/w freqs,
all-axes-equal positions degenerate to standard 1D RoPE — meaning the
existing decode path (1D RoPE gather on cur_pos) is mathematically correct
for text-only decode after multimodal prefill, provided cur_pos starts at
``max(position_ids_3d) + 1``.
"""

from __future__ import annotations

import os

import torch
from PIL.Image import Image

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_pipeline import Qwen36MMPipeline
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import build_mrope_tt_tensors
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.tt_dit.parallel.manager import CCLManager


def _send_col_sharded_hidden(t: torch.Tensor, mesh_device, cluster_shape) -> ttnn.Tensor:
    """Upload `[1, T, H]` torch hidden state as col-sharded `[1, 1, T, H/cols]` per chip.

    Mirrors `tests/test_64layer_full_pcc.py::_send_col_sharded_hidden`, the known-good
    upload format the qwen3.6 text decoder expects for prefill input.

    V4: optionally upload as fp32 (V4_FP32_INPUT=1) — matmul becomes fp32_act × bf16_weight
    with fp32 accumulator, preserving more precision through the first few layers.
    Output dtype is still bf16 per the existing decoder code.
    """
    if t.dim() == 2:
        t = t.unsqueeze(0)
    assert t.dim() == 3, f"expected [B, T, H] got {t.shape}"
    B, T, H = t.shape
    use_fp32 = os.environ.get("V4_FP32_INPUT", "0") == "1"
    upload_dtype = ttnn.float32 if use_fp32 else ttnn.bfloat16
    torch_dtype = torch.float32 if use_fp32 else torch.bfloat16
    return ttnn.from_torch(
        t.reshape(1, 1, T, H).to(torch_dtype),
        device=mesh_device,
        dtype=upload_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=cluster_shape),
    )


def _gather_logits_to_torch(tt_logits, mesh_device, cluster_shape) -> torch.Tensor:
    """Pull col-sharded logits (vocab-dim sharded across cols) back to CPU as `[1, T, V_pad]`."""
    out = ttnn.to_torch(
        tt_logits,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=cluster_shape),
    )
    out = out[0:1]
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out


class Qwen36MMGenerator:
    """End-to-end multimodal generator: vision encoder + text decoder.

    Usage:
        gen = Qwen36MMGenerator(mesh_device, ccl_manager, vision_args, text_model)
        logits = gen.prefill_multimodal("<|image_pad|>What's in this image?", [pil_image])
        tokens = gen.generate("<|image_pad|>Describe", [pil_image], max_new_tokens=16)

    The text_model is a `TtTransformer` (qwen3.6 64-layer) already constructed and
    placed on the mesh. Decode after prefill uses 1D RoPE on cur_pos since
    text-tokens-after-vision have all-axes-equal positions (mathematically
    equivalent to standard 1D RoPE for the M-RoPE section interleave).
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        vision_model_args: Qwen36VisionModelArgs,
        text_model=None,  # TtTransformer (qwen3.6 64-layer) — required for prefill/generate
        *,
        text_embed_weight: torch.Tensor | None = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.vision_model_args = vision_model_args
        self.text_model = text_model

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

        self.tokenizer = self.pipeline.preprocessor.processor.tokenizer
        self.processor = self.pipeline.preprocessor.processor

    def prepare_inputs(
        self,
        prompt: str,
        images: list[Image] | None = None,
        videos: list | None = None,
        video_metadata: list | None = None,
    ):
        """CPU side: produce (Qwen36MMInputs, fused_embeddings) for images and/or videos."""
        return self.pipeline.prepare_decoder_inputs(prompt, images=images, videos=videos, video_metadata=video_metadata)

    def build_rope_tensors(self, position_ids_3d: torch.Tensor):
        """Build M-RoPE cos/sin TT tensors for the given 3D position_ids.

        Shape: each cos/sin is `[1, 1, S, partial_rotary_dim=64]` replicated.
        """
        return build_mrope_tt_tensors(
            position_ids_3d,
            rope_theta=self._rope_theta,
            partial_rotary_dim=self._partial_rotary_dim,
            mrope_section=self._mrope_section,
            mesh_device=self.mesh_device,
        )

    def prefill_multimodal(
        self,
        prompt: str,
        images: list[Image] | None = None,
        *,
        return_all_logits: bool = False,
        pre_computed_inputs=None,
        pre_computed_fused_embeddings: torch.Tensor | None = None,
        force_degenerate_positions: bool = False,
    ):
        """Run vision pipeline + text decoder prefill with M-RoPE cos/sin.

        If `pre_computed_inputs` and `pre_computed_fused_embeddings` are
        provided, skip the vision pipeline (useful for tests that want to
        compare against a CPU reference using the exact same pipeline outputs
        — avoids bf16 non-determinism between two vision encoder runs).

        Returns:
            (logits, inputs) where
              - logits: torch tensor `[T, vocab_size]` (or `[1, vocab_size]` if
                `return_all_logits=False`)
              - inputs: `Qwen36MMInputs` — needed by caller to continue with decode
                (position_ids_3d for cur_pos, etc.)
        """
        from models.demos.qwen3_6_galaxy_v2.tt.generator import get_padded_prefill_len

        assert self.text_model is not None, "text_model must be provided to prefill_multimodal"

        # 1. Vision pipeline → fused embeddings on CPU
        if pre_computed_inputs is not None and pre_computed_fused_embeddings is not None:
            inputs, fused_embeddings = pre_computed_inputs, pre_computed_fused_embeddings
        else:
            inputs, fused_embeddings = self.prepare_inputs(prompt, images=images)
        # fused_embeddings: [B=1, S, H=5120]
        S_unpadded = fused_embeddings.shape[1]
        S = get_padded_prefill_len(S_unpadded)
        if S > S_unpadded:
            # Pad fused_embeddings with zeros and position_ids with continuing positions
            pad_len = S - S_unpadded
            fused_embeddings = torch.cat(
                [
                    fused_embeddings,
                    torch.zeros(
                        *fused_embeddings.shape[:-2], pad_len, fused_embeddings.shape[-1], dtype=fused_embeddings.dtype
                    ),
                ],
                dim=-2,
            )
            # position_ids_3d: [3, B, S_unpadded] → pad to [3, B, S]
            last_pos = inputs.position_ids_3d[:, :, -1:].max().item()
            pad_positions = torch.arange(last_pos + 1, last_pos + 1 + pad_len, dtype=inputs.position_ids_3d.dtype)
            pad_positions_3d = pad_positions.view(1, 1, pad_len).expand(3, inputs.position_ids_3d.shape[1], pad_len)
            position_ids_3d_padded = torch.cat([inputs.position_ids_3d, pad_positions_3d], dim=-1)
        else:
            position_ids_3d_padded = inputs.position_ids_3d

        # Optional control: override 3D positions with degenerate (axes-equal) `arange(S)`.
        # Useful diagnostic — isolates "non-degenerate M-RoPE" from "vision-features-as-input".
        if force_degenerate_positions:
            S_total = position_ids_3d_padded.shape[-1]
            positions_1d = torch.arange(S_total, dtype=position_ids_3d_padded.dtype)
            position_ids_3d_padded = positions_1d.view(1, 1, S_total).expand(3, 1, S_total).contiguous()

        # 2. Build M-RoPE cos/sin on mesh
        cos_tt, sin_tt = self.build_rope_tensors(position_ids_3d_padded)

        # 3. Upload fused embeddings col-sharded
        args = self.text_model.args
        x_tt = _send_col_sharded_hidden(fused_embeddings, self.mesh_device, args.cluster_shape)

        # 4. chunk_start_idx = 0 device tensor
        chunk_start_idx_tt = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # 5. Text decoder forward — bypass ttnn_prefill_forward's embed lookup.
        # The model.forward path accepts pre-embedded x + external rot_mats
        # (test_64layer_full_pcc.py is the precedent).
        get_last_token = -1 if return_all_logits else (S_unpadded - 1)
        tt_logits = self.text_model.forward(
            x_tt,
            current_pos=None,
            rot_mats=(cos_tt, sin_tt),
            user_id=0,
            mode="prefill",
            page_table=None,
            chunk_page_table=None,
            chunk_start_idx=chunk_start_idx_tt,
            start_pos=0,
            get_last_token=get_last_token,
            kv_cache=None,
            batch_size=1,
        )

        # 6. Cleanup transient device tensors
        ttnn.deallocate(x_tt)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)
        ttnn.deallocate(chunk_start_idx_tt)

        # 7. Gather logits to CPU
        if isinstance(tt_logits, (list, tuple)):
            tt_logits = tt_logits[0]
        logits_torch = _gather_logits_to_torch(tt_logits, self.mesh_device, args.cluster_shape)
        # Slice to real vocab
        V = args.vocab_size
        logits_torch = logits_torch[..., :V]
        return logits_torch, inputs

    def generate(
        self,
        prompt: str,
        images: list[Image] | None = None,
        max_new_tokens: int = 16,
    ) -> list[int]:
        """Prefill once, then sample max_new_tokens with greedy decode.

        Returns list of generated token ids (excludes the prompt).
        Decode loop is not yet wired (requires KV cache + decode position
        tracking) — for now this is prefill-only with a single-token sample.
        """
        logits, inputs = self.prefill_multimodal(prompt, images=images, return_all_logits=False)
        # logits: [1, 1, V] or [1, V]
        while logits.dim() > 2:
            logits = logits.squeeze(0)
        # logits: [1, V] or [V]
        if logits.dim() == 2:
            logits = logits[-1]
        next_token = int(logits.argmax().item())
        # TODO: decode loop with KV cache + cur_pos tracking
        return [next_token]
