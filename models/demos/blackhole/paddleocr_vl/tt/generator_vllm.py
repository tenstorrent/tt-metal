# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM entry point for PaddleOCR-VL on Blackhole.

Registers ``PaddleOCRVLForConditionalGeneration`` on vLLM's OpenAI-compatible
server, reusing vLLM's own multimodal processor (tokenization, placeholder
expansion, ``smart_resize``) and qwen3_vl's model-agnostic ``Generator`` --
whose ``last_token_idx % 32`` selection out of the prefill block matters here
too; see ``tt/model.py``. No deepstack; splice and padding happen on the host.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
from loguru import logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.paddleocr_vl import (
    PaddleOCRVLDummyInputsBuilder,
    PaddleOCRVLMultiModalProcessor,
    PaddleOCRVLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.demos.blackhole.paddleocr_vl.tt.common import multimodal_rope_from_hf, splice_image_embeddings
from models.demos.blackhole.paddleocr_vl.tt.model import Transformer
from models.demos.blackhole.paddleocr_vl.tt.vision.model import DropInVisionTransformer
from models.demos.blackhole.paddleocr_vl.tt.vision.vision_model_config import VisionModelArgs
from models.demos.blackhole.paddleocr_vl.tt.weight_mapping import map_vision_state_dict
from models.demos.qwen3_vl.tt.generator import Generator as VLGenerator
from models.demos.qwen3_vl.tt.generator_vllm import _prefill_single_user_with_sliced_page_table, allocate_vllm_kv_cache
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.model_config import ModelArgs

MAX_SEQ_LEN_NATIVE = 131072


class TT_PaddleOCRVLProcessingInfo(PaddleOCRVLProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Upstream advertises unlimited images. The tower runs one image per
        # forward and the decoder is batch-1, so cap it rather than let a request
        # in that would be silently slow.
        return {"image": 1}


@MULTIMODAL_REGISTRY.register_processor(
    PaddleOCRVLMultiModalProcessor,
    info=TT_PaddleOCRVLProcessingInfo,
    dummy_inputs=PaddleOCRVLDummyInputsBuilder,
)
class PaddleOCRVLForConditionalGeneration(VLGenerator, SupportsMultiModal):
    model_capabilities = {
        # Prefix caching would need the vision splice to be cache-aware; not claimed.
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        # Measured, not assumed: produced garbage at batch 1 (CER 77-332%,
        # finish_reason "length"), the same corruption class as qwen3_vl's
        # #48037. Not needed either -- allow_force_argmax alone (tt/model.py)
        # already clears the decode-rate gate.
        "supports_sample_on_device": False,
    }

    def __init__(self, *args, **kwargs):
        self.reference_model = kwargs.pop("reference_model", None)
        self.visual_model = kwargs.pop("visual_model", None)
        assert self.reference_model is not None, "reference_model is required (host embeddings and M-RoPE)"
        assert self.visual_model is not None, "visual_model is required"
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        assert tt_data_parallel == 1, "PaddleOCR-VL runs on a single submesh"
        # tt_data_parallel alone doesn't bound mesh_device's width: some multi-chip
        # widths (e.g. TP=2) pass every downstream divisibility check silently instead
        # of failing, so the device count is checked explicitly here.
        assert mesh_device.get_num_devices() == 1, (
            f"PaddleOCR-VL supports a single Blackhole die only; got a "
            f"{mesh_device.get_num_devices()}-device mesh {tuple(mesh_device.shape)}"
        )
        if max_seq_len > MAX_SEQ_LEN_NATIVE:
            logger.warning(f"max_seq_len {max_seq_len} exceeds native {MAX_SEQ_LEN_NATIVE}; clamping")
            max_seq_len = MAX_SEQ_LEN_NATIVE

        dtype = ttnn.bfloat8_b

        # Accuracy over performance: bfp8 measured the same as bf16 on the vision
        # tower (PCC 0.979 vs 0.978), and OCR pays for precision in characters.
        text_args = ModelArgs(
            mesh_device,
            instruct=True,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
        )
        vision_args = VisionModelArgs(
            mesh_device,
            instruct=True,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # One load, split once: the text weights are already in meta naming and
        # pass through, the vision weights are renamed onto the qwen36 tower's
        # layout, and three tensors go to the host seam.
        device_sd, host_sd = map_vision_state_dict(
            vision_args.load_state_dict(), vision_head_dim=vision_args.head_dim, strict=True
        )

        model = Transformer(
            args=text_args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=device_sd,
            weight_cache_path=text_args.weight_cache_path(dtype),
            use_paged_kv_cache=True,
        )

        visual_model = DropInVisionTransformer(
            model_args=vision_args,
            device_state_dict=device_sd,
            host_state_dict=host_sd,
            dtype=dtype,
        )

        # Host-only reference: the embedding table, get_rope_index and the text
        # rotary module. Its decoder weights are never used for inference.
        from transformers import AutoModelForImageTextToText

        reference_model = AutoModelForImageTextToText.from_pretrained(text_args.CKPT_DIR, dtype=torch.bfloat16)
        reference_model.eval()

        return cls(
            model,
            text_args,
            mesh_device,
            tokenizer=text_args.tokenizer,
            reference_model=reference_model,
            visual_model=visual_model,
        )

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(
            *args, **kwargs, model=self.model, model_args=self.model_args, tt_cache_path=self.cache_path
        )

    def _image_inputs_for(self, kwargs, user_id, batch_size):
        """Pull one user's pixel_values/grid out of the per-request lists."""
        pv_all = kwargs.get("pixel_values") or []
        grid_all = kwargs.get("image_grid_thw") or []
        if user_id >= len(pv_all) or pv_all[user_id] is None:
            return None, None

        pv, grid = pv_all[user_id], grid_all[user_id]
        if isinstance(pv, list):
            if not pv:
                return None, None
            pv = torch.cat(pv, dim=0)
            grid = torch.stack([g.to(torch.int32) for g in grid], dim=0)
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        return pv, grid

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, enable_trace, **kwargs):
        """Vision, splice, M-RoPE and prefill, one user at a time.

        Returns ``(logits, rope_deltas)``. The second element is required because
        this checkpoint declares ``mrope_section``, so vLLM's ``uses_mrope`` is
        True and the plugin threads the deltas into every decode step.
        """
        if enable_trace:
            # A trace captured here would be invalidated by the first image at a
            # new vision bucket, which compiles. Buckets are warmed explicitly
            # instead; see tt/vision/model.py.
            logger.warning("prefill tracing is not enabled for PaddleOCR-VL; running eager")

        batch_size = tokens.shape[0]
        vocab = self.model_args.vocab_size
        pad_token_id = self.tokenizer.pad_token_id or 0
        image_token_id = self.reference_model.config.image_token_id
        embed_tokens = self.reference_model.model.language_model.embed_tokens

        output_logits = torch.zeros(batch_size, 1, vocab)
        all_rope_deltas = []

        for user_id in range(batch_size):
            prompt_len = int(prompt_lens[user_id])
            input_ids = tokens[user_id][:prompt_len].to(torch.int64)

            pixel_values, grid = self._image_inputs_for(kwargs, user_id, batch_size)

            with torch.no_grad():
                text_embeds = embed_tokens(input_ids.unsqueeze(0))[0]

            if pixel_values is not None:
                image_embeds = self.visual_model(pixel_values.to(torch.bfloat16), grid)
                merged = splice_image_embeddings(input_ids, text_embeds, image_embeds, image_token_id)
            else:
                merged = text_embeds

            padded_len = get_padded_prefill_len(prompt_len)
            if padded_len > prompt_len:
                merged = torch.cat(
                    [merged, torch.zeros(padded_len - prompt_len, merged.shape[-1], dtype=merged.dtype)], dim=0
                )

            cos, sin, rope_deltas = multimodal_rope_from_hf(
                input_ids,
                grid,
                self.reference_model,
                self.model_args,
                pad_token_id=pad_token_id,
                min_positions=max(padded_len, self.model_args.max_seq_len),
            )
            all_rope_deltas.append(rope_deltas)

            embeds_tt = ttnn.from_torch(
                merged.unsqueeze(0).to(torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            # Expose only the blocks allocated for this sequence; the persistent
            # vLLM page table is padded to max_model_len and padded zero entries
            # would redirect K/V writes into physical block 0.
            logits = _prefill_single_user_with_sliced_page_table(
                self,
                embeds_tt,
                page_table,
                user_id,
                prompt_len,
                (cos, sin),
                kv_cache,
                None,  # no deepstack embeddings for this model
            )
            output_logits[user_id] = torch.as_tensor(logits).reshape(1, -1)[:, :vocab]
            ttnn.deallocate(embeds_tt)

        rope_deltas = torch.stack([rd.reshape(-1)[:1] for rd in all_rope_deltas], dim=0)
        return output_logits, rope_deltas

    def warmup_model_prefill(self, *args, enable_trace: bool = False, **kwargs):
        """Compile the vision buckets in the plugin's phase-one warmup.

        Prefill tracing stays off for this model, so the phase-two call (this
        method with ``enable_trace=True``) has nothing to do; see
        ``DropInVisionTransformer.warmup_buckets`` for why phase one matters.
        """
        if enable_trace:
            return
        warmed = self.visual_model.warmup_buckets()
        logger.info(f"vision tower: {warmed} bucket(s) compiled before trace capture")

    def decode_forward(self, *args, **kwargs):
        rope_deltas_list = kwargs.pop("rope_deltas_all_users", None)
        if rope_deltas_list is not None:
            super().update_rope_deltas(rope_deltas_list)
        return super().decode_forward(*args, **kwargs)
