# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from datetime import datetime

import pytest
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

import ttnn
from models.common.sampling import SamplingParams
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    get_block_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, parse_decoder_json


def create_tt_page_table(paged_attention_config, tt_model_args):
    if paged_attention_config is None:
        return None
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    return reverse_permutation.reshape(
        tt_model_args.max_batch_size, paged_attention_config.max_num_blocks // tt_model_args.max_batch_size
    )


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    dtype=ttnn.bfloat8_b,
    use_paged_kv_cache=False,
):
    tt_model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    state_dict = tt_model_args.load_state_dict()

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        if use_paged_kv_cache
        else None
    )

    model = Transformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if use_paged_kv_cache else None

    return tt_model_args, model, paged_attention_config, tt_kv_cache


def load_and_process_images(image_sources):
    """Load images from URLs or file paths."""
    images = []
    for src in image_sources:
        if src.startswith("http://") or src.startswith("https://"):
            import requests
            from io import BytesIO

            response = requests.get(src, timeout=30)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(src).convert("RGB")
        images.append(img)
    return images


def get_prefill_page_table(page_table, kv_cache, prefill_len):
    block_size = get_block_size(kv_cache)
    n_blocks = num_blocks_in_seq(prefill_len, block_size)
    return page_table[:, :n_blocks]


def prefill_single_user(
    model,
    model_args,
    embeddings,
    user_id,
    decoding_pos,
    page_table,
    kv_cache,
):
    """
    Run prefill for a single user with pre-computed embeddings.
    Bypasses prepare_inputs_prefill since we already have embeddings
    instead of token IDs.
    """
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


@pytest.mark.parametrize(
    "input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stop_at_eos",
    [
        (
            "models/demos/phi3v/demo/sample_prompts/demo.json",
            True,
            1,
            4096,
            1,
            200,
            True,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            True,
        ),
    ],
    ids=[
        "batch-1",
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
    ],
    ids=[
        "performance",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 28467200, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_demo(
    input_prompts,
    instruct,
    repeat_batches,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    optimizations,
    stop_at_eos,
    mesh_device,
    is_ci_env,
    reset_seeds,
    request,
):
    logger.info(f"mesh_device: {mesh_device}")
    enable_trace = True

    input_prompts = request.config.getoption("--input_prompts") or input_prompts
    if request.config.getoption("--instruct") in [0, 1]:
        instruct = request.config.getoption("--instruct")
    repeat_batches = request.config.getoption("--repeat_batches") or repeat_batches
    max_seq_len = request.config.getoption("--max_seq_len") or max_seq_len
    batch_size = request.config.getoption("--batch_size") or batch_size
    max_generated_tokens = request.config.getoption("--max_generated_tokens") or max_generated_tokens
    paged_attention = request.config.getoption("--paged_attention") or paged_attention
    page_params = request.config.getoption("--page_params") or page_params
    sampling_params = request.config.getoption("--sampling_params") or sampling_params
    if request.config.getoption("--stop_at_eos") in [0, 1]:
        stop_at_eos = request.config.getoption("--stop_at_eos")
    json_config_file = request.config.getoption("--decoder_config_file")

    if json_config_file:
        optimizations = parse_decoder_json(json_config_file)
    else:
        optimizations = request.config.getoption("--optimizations") or optimizations

    if paged_attention:
        page_cache_max_seq_len = page_params["page_block_size"] * page_params["page_max_num_blocks"] / batch_size
        assert max_seq_len <= page_cache_max_seq_len, (
            f"max_seq_len ({max_seq_len}) needs to be <= page_cache_max_seq_len ({page_cache_max_seq_len})"
        )

    # Load prompts
    logger.info("Reading inputs...")
    with open(input_prompts, "r") as f:
        user_prompts = json.load(f)
    if len(user_prompts) < batch_size:
        user_prompts = user_prompts * batch_size
    user_prompts = user_prompts[:batch_size]
    logger.info(f"Loaded {batch_size} input prompts")

    # Create TT model (text decoder)
    model_args, model, paged_attention_config, tt_kv_cache = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=paged_attention,
    )

    tokenizer = model_args.tokenizer

    # Create base Generator for decode loop
    generator = Generator([model], [model_args], mesh_device, tokenizer=tokenizer)

    # Load HF reference model on CPU (for vision encoder + text embedding)
    ref_model_name = model_args.CKPT_DIR
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(ref_model_name, trust_remote_code=True, local_files_only=True)
    if hasattr(hf_config, "_attn_implementation"):
        hf_config._attn_implementation = "eager"
    if hasattr(hf_config, "_attn_implementation_autoset"):
        hf_config._attn_implementation_autoset = False
    reference_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        config=hf_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
        local_files_only=True,
    )
    reference_model.eval()
    processor = AutoProcessor.from_pretrained(ref_model_name, trust_remote_code=True, local_files_only=True)

    for batch_idx in range(repeat_batches):
        logger.info(f"Processing batch {batch_idx}")

        page_table = create_tt_page_table(paged_attention_config, model_args)

        output_logits = torch.zeros(batch_size, 1, model_args.vocab_size)
        all_decoding_pos = []

        for user_id in range(batch_size):
            logger.info(f"Processing user {user_id + 1}/{batch_size}")
            prompt = user_prompts[(user_id + batch_idx) % len(user_prompts)]

            # Load images if present
            images = None
            if "images" in prompt:
                images = load_and_process_images(prompt["images"])

            # Build chat messages for processor
            messages = [{"role": prompt["role"], "content": prompt["content"]}]

            # Apply chat template
            text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Process with HF processor
            if images:
                inputs = processor(text=text, images=images, return_tensors="pt")
            else:
                inputs = processor(text=text, return_tensors="pt")

            input_ids = inputs["input_ids"]
            pixel_values = inputs.get("pixel_values", None)
            image_sizes = inputs.get("image_sizes", None)

            # Compute merged vision + text embeddings using HF model
            with torch.no_grad():
                if pixel_values is not None and image_sizes is not None:
                    merged_embeds = reference_model.model.vision_embed_tokens(
                        input_ids, pixel_values=pixel_values, image_sizes=image_sizes
                    )
                else:
                    merged_embeds = reference_model.model.embed_tokens(input_ids)

            # merged_embeds shape: [1, seq_len, hidden_dim]
            merged_embeds = merged_embeds.squeeze(0).float()
            actual_seq_len = merged_embeds.shape[0]

            # Prefill
            logits = prefill_single_user(
                model,
                model_args,
                merged_embeds,
                user_id=user_id,
                decoding_pos=actual_seq_len,
                page_table=page_table,
                kv_cache=tt_kv_cache,
            )

            output_logits[user_id] = logits
            all_decoding_pos.append(actual_seq_len)

        prefilled_token = torch.argmax(output_logits, dim=-1)
        logger.info(f"Prefill finished for all {batch_size} users")

        # Decode loop
        current_pos = torch.tensor(all_decoding_pos)
        iteration = 0
        users_decoding = True
        user_done = [False] * batch_size
        all_outputs = [[] for _ in range(batch_size)]
        for user in range(batch_size):
            all_outputs[user].append(int(prefilled_token[user].item()))

        out_tok = prefilled_token

        logger.info("Starting decode loop...")
        while users_decoding:
            logits, log_probs = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=enable_trace,
                page_table=page_table,
                kv_cache=[tt_kv_cache],
            )

            out_tok = torch.argmax(logits, dim=-1).unsqueeze(1)
            current_pos += 1

            for user in range(batch_size):
                user_tok = out_tok[user].item()
                if user_tok not in tokenizer.stop_tokens and not user_done[user]:
                    all_outputs[user].append(user_tok)
                else:
                    if stop_at_eos:
                        user_done[user] = True
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False

            if not is_ci_env:
                for user in range(batch_size):
                    text = "".join(tokenizer.decode(all_outputs[user]))
                    if len(text) > 100:
                        text = "..." + text[-97:]
                    text = text.replace("\n", " ")
                    logger.info("[User {}] {}".format(user, text))

            iteration += 1
            if iteration >= max_generated_tokens:
                users_decoding = False

            if not users_decoding:
                logger.info("Finished decoding, printing the final outputs...\n")
                for i, prompt in enumerate(user_prompts[:batch_size]):
                    text = tokenizer.decode(all_outputs[i])
                    logger.info(f"\n==USER {i} - OUTPUT\n{text.strip()}\n")

        # Reset KV caches between batches
        if batch_idx < repeat_batches - 1:
            logger.info("KV cache reset to prevent interference between batches")
            for layer in model.layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)
