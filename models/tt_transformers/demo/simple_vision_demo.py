# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional
from loguru import logger

from PIL import Image as PIL_Image
from termcolor import cprint

import llama_models.llama3.reference_impl.generation as llama_reference_generation
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import ImageMedia, UserMessage

from pkg_resources import resource_filename

IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))

import torch
import pytest
import os
import ttnn
import time

from models.tt_transformers.tt.generator import Generator


def get_batch_sampler(temperature, top_p, tokenizer):
    def sample(logits):
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = llama_reference_generation.sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_tokens = next_token.reshape(-1)
        texts = [tokenizer.decode([next_tokens[i].item()]) for i in range(len(next_tokens))]
        return next_tokens, texts

    return sample


def create_multimodal_model(mesh_device, max_batch_size, max_seq_len, dtype=ttnn.bfloat16, use_paged_kv_cache=False):
    from models.tt_transformers.tt.multimodal.llama_vision_model import CrossAttentionTransformer
    from models.tt_transformers.tt.model_config import ModelArgs

    tt_model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size)
    # limit length or we'll run out of space
    tt_model_args.max_seq_len = max_seq_len
    checkpoint = torch.load(tt_model_args.consolidated_weights_path, map_location="cpu", weights_only=True)
    model = CrossAttentionTransformer(
        mesh_device,
        checkpoint,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=tt_model_args,
        use_paged_kv_cache=use_paged_kv_cache,
    )
    return tt_model_args, model


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "test_type,max_seq_len",
    (("normal", 512),),
    ids=["normal"],
)
@pytest.mark.parametrize(
    "warmup_iters, enable_trace, max_batch_size, include_text_only_prompts",
    [
        (0, False, 1, False),  # batch1-notrace
        (0, True, 1, False),  # batch1-trace
        (0, True, 32, False),  # batch32-trace
        (0, True, 4, True),  # batch4-trace-with-text-prompts
    ],
    ids=["batch1-notrace", "batch1-trace", "batch32-trace", "batch4-trace-with-text-prompts"],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 14951424, "num_command_queues": 2}], indirect=True)
def test_multimodal_demo_text(
    mesh_device,
    warmup_iters,
    enable_trace,
    max_batch_size,
    include_text_only_prompts,
    test_type,
    max_seq_len,
    temperature: float = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = 500,
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
    generator = Generator(model, model_args, mesh_device)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    formatter = ChatFormat(tokenizer)

    xattn_caches = generator.model.setup_cache(model_args.max_batch_size)

    with open(IMG_PATH / "ocr_image.jpeg", "rb") as f:
        ocr_image = PIL_Image.open(f).convert("RGB")

    with open(IMG_PATH / "clutter.jpeg", "rb") as f:
        clutter = PIL_Image.open(f).convert("RGB")

    if not include_text_only_prompts:
        with open(IMG_PATH / "dog.jpg", "rb") as f:
            img = PIL_Image.open(f).convert("RGB")

        with open(IMG_PATH / "pasta.jpeg", "rb") as f:
            img2 = PIL_Image.open(f).convert("RGB")

        dialogs = [
            # image understanding
            [UserMessage(content=[ImageMedia(image=img), "Write a haiku for this image."])],
            [UserMessage(content=[ImageMedia(image=img2), "What is for dinner?"])],
            [UserMessage(content=[ImageMedia(image=ocr_image), "What is the full text of this image? Do OCR"])],
            [UserMessage(content=[ImageMedia(image=clutter), "What objects are in this image?"])],
        ]
    else:
        dialogs = [
            # image understanding + text-only prompts
            [UserMessage(content=["Write a haiku."])],
            [UserMessage(content=["What is for dinner?"])],
            [UserMessage(content=[ImageMedia(image=ocr_image), "What is the full text of this image? Do OCR"])],
            [UserMessage(content=[ImageMedia(image=clutter), "What objects are in this image?"])],
        ]
    if len(dialogs) < max_batch_size:
        dialogs *= max_batch_size // len(dialogs)

    assert len(dialogs) % max_batch_size == 0
    num_batches = len(dialogs) // max_batch_size

    sampler = get_batch_sampler(temperature, top_p, tokenizer)

    for iter_num in range(warmup_iters + 1):
        logger.info(f"Iteration {iter_num}")
        for batch_idx in range(num_batches):
            batch_dialogs = dialogs[batch_idx * max_batch_size : (batch_idx + 1) * max_batch_size]
            for dialog in batch_dialogs:
                for msg in dialog:
                    print(f"{msg.role.capitalize()}: {msg.content}\n")
            batch_model_input = [
                formatter.encode_dialog_prompt(dialog, tool_prompt_format=False) for dialog in batch_dialogs
            ]

            # Do initial prefill
            vision_images = [
                model_input.vision.images if model_input.vision else None for model_input in batch_model_input
            ]
            vision_mask = [model_input.vision.mask if model_input.vision else None for model_input in batch_model_input]
            prompt_tokens = [model_input.tokens for model_input in batch_model_input]
            # Get max length of prompts in batch
            prefill_lens = torch.tensor([len(tokens) for tokens in prompt_tokens], dtype=torch.long)
            total_lens = prefill_lens + max_gen_len

            # Create padded tokens tensor for batch
            pad_id = tokenizer.pad_id
            bsz = len(prompt_tokens)
            tokens = torch.full((bsz, max(total_lens)), pad_id, dtype=torch.long)

            # Fill in actual tokens for each sequence in batch
            for i, seq in enumerate(prompt_tokens):
                tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

            prefill_start = time.perf_counter()
            batch_logits, batch_xattn_masks, batch_text_masks = generator.prefill_forward(
                vision_images,
                vision_mask,
                tokens,
                xattn_caches,
                total_lens,
                prefill_lens,
            )

            prefill_end = time.perf_counter()
            next_tokens, next_texts = sampler(batch_logits)
            for i, (next_token, next_text) in enumerate(zip(next_tokens, next_texts)):
                tokens[i, prefill_lens[i]] = next_token
            print(f"Next tokens: {next_tokens}")
            print(f"Next texts: {next_texts}")
            decode_times = []

            for gen_idx in range(max_gen_len - 1):
                decode_start = time.perf_counter()
                position_id = prefill_lens + gen_idx
                next_token_tensor = next_tokens.reshape(max_batch_size, 1)

                logits = generator.decode_forward(
                    position_id,
                    next_token_tensor,
                    batch_xattn_masks,
                    batch_text_masks,
                    xattn_caches,
                    enable_trace=enable_trace,
                )

                next_tokens, next_texts = sampler(logits)
                # Update next token
                tokens[torch.arange(max_batch_size), position_id + 1] = next_tokens
                decode_end = time.perf_counter()
                decode_times.append(decode_end - decode_start)

                # Disable checking for eot until I have more robust code for batch > 1
                # if text in ["<|eot_id|>", "<|eom_id|>"]:
                #     break
            # Log full text output for each user in batch
            vision_tokens = [tokenizer.special_tokens["<|image|>"], 128256]

            for user_id in range(max_batch_size):
                # Remove <|image|> tokens since they break the tokenizer
                tokens_out = [
                    t if t not in vision_tokens else tokenizer.pad_id
                    for t in tokens[user_id].tolist()[: position_id[user_id] + 2]
                ]
                text = tokenizer.decode(tokens_out)
                logger.info(f"User {user_id} full text: {text}")

            prefill_time_ms = (prefill_end - prefill_start) * 1000
            logger.info(f"Prefill time: {prefill_time_ms:.2f} ms")
            decode_time_ms = sum(decode_times) / (gen_idx + 1) * 1000
            logger.info(f"Average decode time per token: {decode_time_ms:.2f} ms")

            # ttnn.release_trace(generator.mesh_device, trace_id)
