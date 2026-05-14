# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-Metal Qwen3-VL model runner for benchmark evaluation."""

import os
import math
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import logging as transformers_logging

from models.demos.qwen3_vl.tt.model import DropInVisionTransformer, Transformer
from models.demos.qwen3_vl.tt.model_config import VisionModelArgs
from models.demos.qwen3_vl.tt.generator import Generator
from models.demos.qwen3_vl.tt.common import (
    PagedAttentionConfig,
    get_pad_embedding,
    merge_vision_tokens_single_user_ttnn,
    multimodal_rope_single_user_from_hf,
    preprocess_inputs_prefill_single_user_ttnn,
    sample_host,
)
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs


class Qwen3VL2BRunner:
    """Wraps TT-Metal Qwen3-VL-2B inference for benchmark evaluation.

    Usage:
        runner = Qwen3VL2BRunner()
        runner.setup()
        output = runner.generate(messages)
        runner.teardown()
    """

    def __init__(
        self,
        hf_model: str = "Qwen/Qwen3-VL-2B-Instruct",
        max_seq_len: int = 8192,
        max_new_tokens: int = 256,
        use_tt_vision: bool = True,
        page_block_size: int = 32,
        page_max_num_blocks: int = 1024,
        dtype=None,
    ):
        self.hf_model = hf_model
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.use_tt_vision = use_tt_vision
        self.page_block_size = page_block_size
        self.page_max_num_blocks = page_max_num_blocks
        self.dtype = dtype or ttnn.bfloat8_b

        self.mesh_device = None
        self.model = None
        self.generator = None
        self.reference_model = None
        self.processor = None
        self.model_args = None

    def setup(self):
        """Initialize model, load weights, prepare for inference."""
        logger.info(f"Setting up Qwen3VL2BRunner (hf_model={self.hf_model})")
        transformers_logging.set_verbosity_error()

        self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

        # TT text model
        self.model_args = ModelArgs(
            self.mesh_device,
            instruct=True,
            max_batch_size=1,
            max_seq_len=self.max_seq_len,
        )
        state_dict = self.model_args.load_state_dict()
        paged_cfg = PagedAttentionConfig(
            block_size=self.page_block_size,
            max_num_blocks=self.page_max_num_blocks,
        )
        self.model = Transformer(
            args=self.model_args,
            mesh_device=self.mesh_device,
            dtype=self.dtype,
            state_dict=state_dict,
            weight_cache_path=self.model_args.weight_cache_path(self.dtype),
            paged_attention_config=paged_cfg,
        )
        self.tt_kv_cache = [l.attention.layer_past for l in self.model.layers]

        # HF reference model (used for embeddings and vision preprocessing)
        config = Qwen3VLForConditionalGeneration.config_class.from_pretrained(self.hf_model)
        self.reference_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.hf_model, config=config, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.hf_model)

        # Vision model
        if self.use_tt_vision:
            vision_args = VisionModelArgs(
                self.mesh_device,
                max_batch_size=1,
                max_seq_len=self.max_seq_len,
                optimizations=DecodersPrecision.accuracy(config.vision_config.depth, self.hf_model),
            )
            vision_args.hf_config.vision_config.depth = config.vision_config.depth
            self.visual_model = DropInVisionTransformer(
                self.reference_model.visual, vision_args, debug=False
            )
        else:
            _runner = self

            class _CPUVision:
                def forward_single_user(self, pixel_values, grid_thw):
                    if grid_thw.dim() == 1:
                        grid_thw = grid_thw.unsqueeze(0)
                    with torch.no_grad():
                        img_pt, ds_pt = _runner.reference_model.visual(pixel_values, grid_thw)
                    to_tt = lambda t: ttnn.from_torch(
                        t.float().to(torch.bfloat16),
                        device=_runner.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                    )
                    return to_tt(img_pt), [to_tt(d) for d in ds_pt]

            self.visual_model = _CPUVision()

        # Generator
        self.model_args.use_qk_fused = False
        self.generator = Generator(
            self.model,
            self.model_args,
            self.mesh_device,
            processor=self.processor,
            tokenizer=self.model_args.tokenizer,
        )

        # Pad embedding (used for prefill padding)
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        self.pad_token_id = pad_token_id
        self.pad_embedding_tt = get_pad_embedding(self.reference_model, pad_token_id, self.model_args)
        logger.info("Model setup complete.")

    def teardown(self):
        """Release device resources."""
        if self.mesh_device is not None:
            ttnn.close_mesh_device(self.mesh_device)
            self.mesh_device = None

    def _make_fresh_page_table(self):
        """Create a fresh sequential page table for this inference.

        With paged attention, KV blocks are addressed via the page table.
        A fresh page table means old data in unreferenced blocks won't be accessed.
        """
        from models.demos.qwen3_vl.tt.common import num_blocks_in_seq
        paged_cfg = PagedAttentionConfig(self.page_block_size, self.page_max_num_blocks)
        n_blocks = num_blocks_in_seq(self.max_seq_len, paged_cfg.block_size)
        return torch.arange(n_blocks, dtype=torch.int32).unsqueeze(0)  # [1, n_blocks]

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Run inference on a single conversation and return generated text.

        Args:
            messages: HuggingFace-style conversation list, e.g.
                [{"role": "user", "content": [{"type": "image", "image": url}, {"type": "text", "text": "Q?"}]}]
            max_new_tokens: Override default max_new_tokens.
            temperature: Sampling temperature (0 = argmax).

        Returns:
            Generated text string (without system/user/assistant tokens).
        """
        from qwen_vl_utils import process_vision_info

        max_tokens = max_new_tokens or self.max_new_tokens

        # --- Tokenize ---
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        has_images = image_inputs is not None and len(image_inputs) > 0

        # --- Vision model ---
        if has_images and inputs.get("pixel_values") is not None:
            pixel_values = inputs.pixel_values
            grid_thw = inputs.image_grid_thw[0]
            image_embeds, deepstack = self.visual_model.forward_single_user(
                pixel_values, grid_thw=grid_thw
            )
        else:
            image_embeds = ttnn.from_torch(
                torch.tensor([], dtype=torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
            )
            deepstack = None
            grid_thw = None

        # --- Text embeddings ---
        text_embeds_pt = self.reference_model.model.language_model.embed_tokens(input_ids.unsqueeze(0))
        text_embeds_tt = ttnn.from_torch(
            text_embeds_pt.squeeze(0),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.model_args.mesh_device,
                dims=(None, 1),
                mesh_shape=self.model_args.cluster_shape,
            ),
        )

        # --- Merge vision tokens ---
        input_embeds, deepstack = merge_vision_tokens_single_user_ttnn(
            input_ids,
            text_embeds_tt,
            image_embeds,
            self.reference_model.config,
            deepstack,
            self.model_args,
        )
        ttnn.deallocate(text_embeds_tt)
        ttnn.deallocate(image_embeds)

        # --- Preprocess for prefill ---
        input_prefill, deepstack_proc, decoding_pos, prefill_len = (
            preprocess_inputs_prefill_single_user_ttnn(
                input_embeds,
                self.model_args,
                attention_mask,
                pad_embedding=self.pad_embedding_tt,
            )
        )
        ttnn.deallocate(input_embeds)

        # --- RoPE ---
        cos, sin, rope_delta = multimodal_rope_single_user_from_hf(
            input_ids,
            grid_thw.unsqueeze(0) if grid_thw is not None else None,
            self.reference_model,
            self.model_args,
            pad_token_id=self.pad_token_id,
        )
        self.generator.update_rope_deltas([rope_delta.squeeze(0).item()])

        # --- Clear KV cache to prevent stale data from previous inference ---
        for layer_cache in self.tt_kv_cache:
            for kv in layer_cache:
                ttnn.mul(kv, 0, output_tensor=kv)

        # --- Prefill --- (fresh page table for each inference)
        page_table = self._make_fresh_page_table()
        page_table_user = self.generator._ttt_generator._get_prefill_user_page_table(
            page_table, self.tt_kv_cache, decoding_pos
        )
        logits = self.generator.prefill_forward_single_user_text(
            ttnn.unsqueeze(input_prefill, 0),
            page_table=page_table_user,
            user_id=0,
            last_token_idx=decoding_pos - 1,
            rot_mats=(cos, sin),
            kv_cache=self.tt_kv_cache,
            deepstack_visual_embeds=deepstack_proc,
        )
        ttnn.deallocate(input_prefill)
        if deepstack_proc:
            for d in deepstack_proc:
                ttnn.deallocate(d)

        output_logits = torch.zeros(1, 1, self.model_args.vocab_size)
        output_logits[0] = logits
        current_pos = torch.tensor([decoding_pos])
        out_tok = torch.argmax(output_logits, dim=-1)
        all_tokens = [int(out_tok[0].item())]

        stop_tokens = set(self.model_args.tokenizer.stop_tokens or [self.model_args.tokenizer.eos_token_id])

        # --- Decode loop ---
        for _ in range(max_tokens - 1):
            if all_tokens[-1] in stop_tokens:
                break
            logits, _ = self.generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=False,
                page_table=page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=None,
            )
            _, out_tok = sample_host(logits, None, temperature=temperature, on_host=True)
            tok = int(out_tok[0].item())
            all_tokens.append(tok)
            current_pos += 1
            if tok in stop_tokens:
                break

        # Decode without special tokens
        text_out = self.model_args.tokenizer.decode(all_tokens, skip_special_tokens=True)
        return text_out.strip()


