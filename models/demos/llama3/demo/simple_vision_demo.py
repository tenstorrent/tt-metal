# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional
from loguru import logger

from PIL import Image as PIL_Image
from termcolor import cprint

import llama_models.llama3.reference_impl.generation as llama_reference_generation
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.api.chat_format import ChatFormat, ModelInput

from llama_models.llama3.api.datatypes import ImageMedia, UserMessage

from pkg_resources import resource_filename

IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))

import torch
import pytest
import os
import ttnn
import time


class LlamaVision:
    def __init__(self, model, model_args, mesh_device, vllm=False):
        """
        Creating a LlamaVision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        LlamaVision is general to text and chat.

        For bringup, make this class general to any backend implementation, as long as it takes torch tensors and returns torch tensors.

        """
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.vllm = vllm

    def prefill_forward_single_user(
        self,
        vision_images,
        vision_mask,
        tokens,
        xattn_caches,
        user_id,
        total_len,
        prefill_len,
    ):
        """
        Performs vision encode step then text prefill.
        Returns (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits)
        """
        B = tokens.shape[0]
        xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = self.model.compute_vision_tokens_masks(
            batch_images=[vision_images],
            batch_masks=[vision_mask],
            total_len=total_len,
            xattn_caches=xattn_caches,
            user_id=user_id,
        )

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            rot_mats,
            transformation_mats,
        ) = self.model.prepare_inputs_prefill(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, prefill_len=prefill_len
        )

        tt_logits = self.model.ttnn_prefill_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            xattn_caches,
            tt_position_id,
            rot_mats,
            transformation_mats,
            user_id,
        )

        logits = self.model.process_output_prefill(tt_logits, B, prefill_len)

        return xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits

    def decode_forward(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,
    ):
        """
        Performs text decode step.
        Returns logits
        """

        # forward_decode should be traced callable
        # decorator does compilation, capture, execute
        # B = 1 # TODO: Only supports batch=1 right now! Might make tokens input a tensor.
        # S = 1
        B, S = tokens.shape

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            _,
            tt_position_id,
            rot_mats,
            transformation_mats,
        ) = self.model.prepare_inputs_decode(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, position_id=position_id
        )

        tt_logits = self.model.ttnn_decode_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            xattn_caches,
            tt_position_id,
            rot_mats,
            transformation_mats,
        )

        logits = self.model.process_output_decode(tt_logits, B, S)
        return logits

    def capture_trace(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,
    ):
        """
        Captures a trace for the decode_forward method.
        """
        pass


def get_sampler(temperature, top_p, tokenizer):
    def sample(logits):
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = llama_reference_generation.sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        token = next_token[0].item()
        text = tokenizer.decode(next_token.tolist())
        return token, text

    return sample


def create_multimodal_model(mesh_device, max_batch_size, max_seq_len, dtype=ttnn.bfloat16):
    from models.demos.llama3.tt.multimodal.llama_vision_model import CrossAttentionTransformer
    from models.demos.llama3.tt.model_config import TtModelArgs

    tt_model_args = TtModelArgs(mesh_device, max_batch_size=max_batch_size)
    # limit length or we'll run out of space
    tt_model_args.max_seq_len = max_seq_len
    tt_model_args.kv_seq_len = max_seq_len
    tt_model_args.sliding_window = max_seq_len
    checkpoint = torch.load(tt_model_args.consolidated_weights_path, map_location="cpu", weights_only=True)
    model = CrossAttentionTransformer(
        mesh_device,
        checkpoint,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=tt_model_args,
    )
    return tt_model_args, model


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "warmup_iters",
    (0, 1),
    ids=["cold", "warm"],
)
@pytest.mark.parametrize(
    "test_case",
    [
        "normal",
    ],
)
def test_llama_multimodal_demo_text(
    mesh_device,
    warmup_iters,
    test_case,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = 200,
    model_parallel_size: Optional[int] = None,
):
    """
    Simple multimodal demo with limited dependence on reference code.
    """
    ckpt_dir = os.environ["LLAMA_DIR"]
    tokenizer_path = str(Path(ckpt_dir) / "tokenizer.model")

    mesh_device.enable_program_cache()
    mesh_device.enable_async(True)
    model_args, model = create_multimodal_model(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
    model = LlamaVision(model, model_args, mesh_device)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    formatter = ChatFormat(tokenizer)

    xattn_caches = model.model.setup_cache(model_args.max_batch_size)

    with open(IMG_PATH / "dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    with open(IMG_PATH / "pasta.jpeg", "rb") as f:
        img2 = PIL_Image.open(f).convert("RGB")

    with open(IMG_PATH / "ocr_image.jpeg", "rb") as f:
        ocr_image = PIL_Image.open(f).convert("RGB")

    with open(IMG_PATH / "clutter.jpeg", "rb") as f:
        clutter = PIL_Image.open(f).convert("RGB")

    dialogs = [
        # image understanding
        [UserMessage(content=[ImageMedia(image=img), "Write a haiku for this image."])],
        [UserMessage(content=[ImageMedia(image=img2), "What is for dinner?"])],
        [UserMessage(content=[ImageMedia(image=ocr_image), "What is the full text of this image? Do OCR"])],
        [UserMessage(content=[ImageMedia(image=clutter), "What objects are in this image?"])],
    ]

    sampler = get_sampler(temperature, top_p, tokenizer)

    for iter_num in range(warmup_iters + 1):
        for dialog in dialogs:
            for msg in dialog:
                print(f"{msg.role.capitalize()}: {msg.content}\n")

            if iter_num <= warmup_iters:
                logger.info(f"Warmup iteration {iter_num}")

            model_input = formatter.encode_dialog_prompt(dialog, tool_prompt_format=False)

            # Do initial prefill
            vision_images = model_input.vision.images
            vision_mask = model_input.vision.mask
            prompt_tokens = model_input.tokens
            prefill_len = len(prompt_tokens)
            total_len = prefill_len + max_gen_len  # Prepares mask for full length of output
            # Create tokens tensor
            pad_id = tokenizer.pad_id
            bsz = 1
            tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
            tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long)
            prefill_start = time.perf_counter()
            prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long).reshape(1, -1)  # B, S
            (
                xattn_caches,
                cross_attention_masks,
                full_text_row_masked_out_mask,
                logits,
            ) = model.prefill_forward_single_user(
                vision_images,
                vision_mask,
                prompt_tokens_tensor,
                xattn_caches,
                user_id=0,
                total_len=total_len,
                prefill_len=prefill_len,
            )
            prefill_end = time.perf_counter()

            next_token, text = sampler(logits)
            # logger.info(f"Prefill output: {next_token}:{text}")
            tokens[0, prefill_len] = next_token

            decode_times = []

            for gen_idx in range(max_gen_len - 1):
                decode_start = time.perf_counter()
                position_id = prefill_len + gen_idx
                next_token_tensor = torch.tensor([next_token], dtype=torch.long).reshape(1, 1)  # B, S
                logits = model.decode_forward(
                    position_id,
                    next_token_tensor,
                    cross_attention_masks,
                    full_text_row_masked_out_mask,
                    xattn_caches,
                )
                next_token, text = sampler(logits)
                # Update next token
                tokens[0, position_id + 1] = next_token
                # logger.info(f"Decode output {position_id}: {next_token}:{text}")
                decode_end = time.perf_counter()
                decode_times.append(decode_end - decode_start)

                if text in ["<|eot_id|>", "<|eom_id|>"]:
                    break

            # Log full text output
            vision_tokens = [tokenizer.special_tokens["<|image|>"], 128256]
            # Remove <|image|> tokens since they break the tokenizer
            tokens_out = [
                t if t not in vision_tokens else tokenizer.pad_id for t in tokens[0].tolist()[: position_id + 2]
            ]
            text = tokenizer.decode(tokens_out)
            logger.info(f"Full text: {text}")

            prefill_time_ms = (prefill_end - prefill_start) * 1000
            logger.info(f"Prefill time: {prefill_time_ms:.2f} ms")
            decode_time_ms = sum(decode_times) / (gen_idx + 1) * 1000
            logger.info(f"Decode time: {decode_time_ms:.2f} ms")
