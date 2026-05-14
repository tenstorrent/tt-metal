# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-Metal Phi-3.5-vision model runner for benchmark evaluation."""

import os

import torch
import ttnn
from loguru import logger
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers import logging as transformers_logging

from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    get_block_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs


def _prefill_single_user(model, model_args, embeddings, user_id, decoding_pos, page_table, kv_cache):
    """Run prefill for a single user with pre-computed embeddings."""
    seq_len = embeddings.shape[0]
    padded_len = get_padded_prefill_len(seq_len)

    if padded_len > seq_len:
        padding = torch.zeros(padded_len - seq_len, embeddings.shape[-1], dtype=embeddings.dtype)
        embeddings = torch.cat([embeddings, padding], dim=0)

    embeddings_tt = ttnn.from_torch(
        embeddings.unsqueeze(0).unsqueeze(0),
        device=model_args.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            model_args.mesh_device, dims=(None, 3), mesh_shape=model_args.cluster_shape
        ),
    )

    cos_slice = model.rope_setup.cos_matrix_prefill[:, :, :padded_len, :]
    sin_slice = model.rope_setup.sin_matrix_prefill[:, :, :padded_len, :]
    rot_mats = [cos_slice, sin_slice]

    if page_table is not None:
        block_size = get_block_size(kv_cache)
        n_blocks = num_blocks_in_seq(padded_len, block_size)
        page_table_user = page_table[:, :n_blocks]
        page_table_tt = ttnn.from_torch(
            page_table_user,
            device=model_args.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(model_args.mesh_device),
        )
    else:
        page_table_tt = None

    last_token_idx = decoding_pos - 1
    tt_logits = model.ttnn_prefill_forward(
        embeddings_tt,
        rot_mats_global=rot_mats,
        user_id=user_id,
        page_table=page_table_tt,
        get_last_token=(last_token_idx // 32) * 32,
        kv_cache=kv_cache,
    )

    logits = model.process_output_prefill(tt_logits.cpu(), last_token_idx=(last_token_idx % 32))

    ttnn.deallocate(tt_logits)
    ttnn.deallocate(embeddings_tt)
    if page_table_tt is not None:
        ttnn.deallocate(page_table_tt)

    return logits


class Phi3VRunner:
    """Wraps TT-Metal Phi-3.5-vision inference for benchmark evaluation.

    Usage:
        runner = Phi3VRunner()
        runner.setup()
        output = runner.generate(messages)
        runner.teardown()
    """

    def __init__(
        self,
        hf_model: str = "microsoft/Phi-3.5-vision-instruct",
        max_seq_len: int = 4096,
        max_new_tokens: int = 256,
        page_block_size: int = 32,
        page_max_num_blocks: int = 1024,
        dtype=None,
    ):
        self.hf_model = hf_model
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
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
        logger.info(f"Setting up Phi3VRunner (hf_model={self.hf_model})")
        transformers_logging.set_verbosity_error()

        self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

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

        # HF reference model (CPU, for vision encoder + text embedding)
        hf_config = AutoConfig.from_pretrained(self.hf_model, trust_remote_code=True, local_files_only=True)
        if hasattr(hf_config, "_attn_implementation"):
            hf_config._attn_implementation = "eager"
        if hasattr(hf_config, "_attn_implementation_autoset"):
            hf_config._attn_implementation_autoset = False
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model,
            config=hf_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            attn_implementation="eager",
            local_files_only=True,
        )
        self.reference_model.eval()
        self.processor = AutoProcessor.from_pretrained(self.hf_model, trust_remote_code=True, local_files_only=True)

        self.generator = Generator(
            [self.model], [self.model_args], self.mesh_device, tokenizer=self.model_args.tokenizer
        )

        self.paged_cfg = paged_cfg
        logger.info("Model setup complete.")

    def teardown(self):
        """Release device resources."""
        if self.mesh_device is not None:
            ttnn.close_mesh_device(self.mesh_device)
            self.mesh_device = None

    def _make_fresh_page_table(self):
        n_blocks = num_blocks_in_seq(self.max_seq_len, self.paged_cfg.block_size)
        return torch.arange(n_blocks, dtype=torch.int32).unsqueeze(0)

    def _convert_messages(self, messages):
        """Convert HF-style messages (Qwen format) to Phi-3.5-vision format.

        Input format (from benchmarks):
            [{"role": "user", "content": [
                {"type": "image", "image": PIL.Image},
                {"type": "text", "text": "question?"}
            ]}]

        Output: (text_for_chat_template, list_of_PIL_images)
        """
        images = []
        parts = []

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
                continue

            img_count = 0
            text_parts = []
            for item in content:
                if item["type"] == "image":
                    img_count += 1
                    img = item["image"]
                    if isinstance(img, str):
                        if img.startswith("http"):
                            import requests
                            from io import BytesIO

                            resp = requests.get(img, timeout=30, headers={"User-Agent": "Phi3V-Eval/1.0"})
                            img = Image.open(BytesIO(resp.content)).convert("RGB")
                        else:
                            img = Image.open(img).convert("RGB")
                    elif not isinstance(img, Image.Image):
                        img = img.convert("RGB")
                    else:
                        img = img.convert("RGB")
                    max_pixels = 512
                    if max(img.size) > max_pixels:
                        ratio = max_pixels / max(img.size)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                    images.append(img)
                    text_parts.append(f"<|image_{len(images)}|>")
                elif item["type"] == "text":
                    text_parts.append(item["text"])

            parts.append("\n".join(text_parts))

        final_text = "\n".join(parts)
        chat_messages = [{"role": "user", "content": final_text}]
        return chat_messages, images

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Run inference and return generated text.

        Args:
            messages: HF-style conversation list (Qwen format with type: image/text).
            max_new_tokens: Override default.
            temperature: 0 = argmax.
        """
        max_tokens = max_new_tokens or self.max_new_tokens

        chat_messages, images = self._convert_messages(messages)

        text = self.processor.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        if images:
            inputs = self.processor(text=text, images=images, return_tensors="pt")
        else:
            inputs = self.processor(text=text, return_tensors="pt")

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values", None)
        image_sizes = inputs.get("image_sizes", None)

        with torch.no_grad():
            if pixel_values is not None and image_sizes is not None:
                merged_embeds = self.reference_model.model.vision_embed_tokens(
                    input_ids, pixel_values=pixel_values, image_sizes=image_sizes
                )
            else:
                merged_embeds = self.reference_model.model.embed_tokens(input_ids)

        merged_embeds = merged_embeds.squeeze(0).float()
        actual_seq_len = merged_embeds.shape[0]

        if actual_seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {actual_seq_len} exceeds max_seq_len {self.max_seq_len}, skipping sample"
            )

        page_table = self._make_fresh_page_table()

        logits = _prefill_single_user(
            self.model,
            self.model_args,
            merged_embeds,
            user_id=0,
            decoding_pos=actual_seq_len,
            page_table=page_table,
            kv_cache=self.tt_kv_cache,
        )

        output_logits = torch.zeros(1, 1, self.model_args.vocab_size)
        output_logits[0] = logits
        current_pos = torch.tensor([actual_seq_len])
        out_tok = torch.argmax(output_logits, dim=-1)
        all_tokens = [int(out_tok[0].item())]

        tokenizer = self.model_args.tokenizer
        stop_tokens = set(tokenizer.stop_tokens or [])

        for _ in range(max_tokens - 1):
            if all_tokens[-1] in stop_tokens:
                break
            decode_logits, _ = self.generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=False,
                page_table=page_table,
                kv_cache=[self.tt_kv_cache],
            )
            out_tok = torch.argmax(decode_logits, dim=-1).unsqueeze(1)
            tok = int(out_tok[0].item())
            all_tokens.append(tok)
            current_pos += 1
            if tok in stop_tokens:
                break

        text_out = tokenizer.decode(all_tokens)
        # Strip special tokens
        for special in ["<|end|>", "<|endoftext|>", "<|assistant|>"]:
            text_out = text_out.replace(special, "")
        return text_out.strip()
