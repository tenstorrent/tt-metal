# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.gemma3.tt.gemma_e2e_model import TtGemmaModel
from models.demos.gemma3.tt.model_config import ModelArgs
from models.tt_transformers.tt.generator import Generator


@skip_for_grayskull("Requires wormhole_b0 to run")
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
    "device_params",
    [{"fabric_config": True, "trace_region_size": 21448704, "num_command_queues": 2, "l1_small_size": 24576}],
    indirect=True,
)
@pytest.mark.parametrize("max_gen_len", [50])
@pytest.mark.parametrize("max_batch_size", [1])
def test_gemma_dummy(
    mesh_device,
    max_batch_size,
    max_gen_len,
):
    dtype = ttnn.bfloat8_b
    tt_model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size, dummy_weights=True)
    tt_model_args.n_layers = 1
    tt_model_args.vision_n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    tt_model = TtGemmaModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat8_b),
        dtype=dtype,
        args=tt_model_args,
    )

    generator = Generator([tt_model], [tt_model_args], mesh_device)

    num_prompts = 3
    num_images = [8, 8, 1, 1]
    num_tokens = [2589, 2589, 776, 773]
    image_size = tt_model_args.vision_chunk_size
    in_channels = tt_model_args.vision_in_channels

    for i in range(num_prompts):
        vision_images = [torch.rand((num_images[i], in_channels, image_size, image_size)) * 2 - 1]
        vision_mask = [None]
        tokens = torch.randint(0, tt_model_args.vocab_size, (1, num_tokens[i]), dtype=torch.long)
        xattn_caches = [None]
        total_lens = torch.tensor([num_tokens[i]])
        prefill_lens = torch.tensor([num_tokens[i] - max_gen_len])

        logger.info(f"Processing prompt {i}, num_images: {num_images[i]}, num_tokens: {num_tokens[i]}")
        logger.info("Starting prefill...")
        (
            batch_logits,
            prefill_batch_xattn_masks,
            prefill_batch_text_masks,
            decode_batch_xattn_masks,
            decode_batch_text_masks,
        ) = generator.prefill_forward(
            vision_images,
            vision_mask,
            tokens,
            xattn_caches,
            total_lens,
            prefill_lens,
        )
        logger.info("Finished prefill")

        next_token = torch.argmax(batch_logits[:, -1], dim=-1)
        next_tokens = next_token.reshape(-1)

        for gen_idx in range(max_gen_len - 1):
            position_id = prefill_lens + gen_idx
            next_token_tensor = next_tokens.reshape(max_batch_size, 1)

            logger.info(f"Processing generation {gen_idx}, position_id: {position_id}")
            logits = generator.decode_forward(
                position_id,
                next_token_tensor,
                prefill_batch_xattn_masks,
                prefill_batch_text_masks,
                decode_batch_xattn_masks,
                decode_batch_text_masks,
                xattn_caches,
                enable_trace=True,
            )
            logger.info("Finished decode")

            next_token = torch.argmax(logits[:, -1], dim=-1)
            next_tokens = next_token.reshape(-1)

            # Update next token
            tokens[torch.arange(max_batch_size), position_id + 1] = next_tokens

    logger.info(f"tt_output_torch shape: {tokens.shape}")
    logger.info(f"Finished test")
