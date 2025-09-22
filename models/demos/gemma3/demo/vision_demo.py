# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from pathlib import Path
from typing import Optional

import llama_models.llama3.reference_impl.generation as llama_reference_generation
import requests
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import ImageMedia, UserMessage
from llama_models.llama3.api.tokenizer import Tokenizer
from loguru import logger
from PIL import Image as PIL_Image
from pkg_resources import resource_filename

from models.tt_transformers.tt.generator import create_submeshes

IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))

import os
import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import hf_multimodal_encode
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import CheckpointType, DecodersPrecision


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


def create_random_image(width, height):
    """Create a random RGB image of specified dimensions."""
    # Generate random RGB values
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return PIL_Image.fromarray(random_array, "RGB")


def create_multimodal_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat16,
    use_paged_kv_cache=False,
    checkpoint=None,
    optimizations=None,
    num_layers=None,
):
    from models.demos.gemma3.tt.gemma_e2e_model import TtGemmaModel
    from models.demos.gemma3.tt.model_config import ModelArgs
    from models.tt_transformers.tt.multimodal.llama_vision_model import CrossAttentionTransformer

    tt_model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size, optimizations=optimizations)
    assert tt_model_args.is_vision(), "This model is multimodal"

    # limit length or we'll run out of space
    tt_model_args.max_seq_len = max_seq_len
    if num_layers is not None:
        tt_model_args.n_layers = num_layers
        tt_model_args.vision_n_layers = num_layers
    if tt_model_args.is_90b:
        assert tt_model_args.device_name == "T3K", "90B model only supported on T3K right now"
        # for 90B model on T3K, use bfp8 and performance optimizations or the model won't fit in memory
        dtype = ttnn.bfloat8_b
        logger.info(f"Setting dtype to bfloat8_b for 90B model on T3K to fit model in memory")

    if checkpoint is None:
        checkpoint = tt_model_args.load_state_dict()

    if tt_model_args.is_gemma:
        model = TtGemmaModel(
            mesh_device=mesh_device,
            state_dict=checkpoint,
            weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat8_b),
            dtype=ttnn.bfloat8_b,
            args=tt_model_args,
            use_paged_kv_cache=use_paged_kv_cache,
        )
    else:
        model = CrossAttentionTransformer(
            mesh_device,
            state_dict=checkpoint,
            weight_cache_path=tt_model_args.weight_cache_path(dtype),
            dtype=dtype,
            configuration=tt_model_args,
            use_paged_kv_cache=use_paged_kv_cache,
        )
    return tt_model_args, model, checkpoint


def prepare_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat16,
    use_paged_kv_cache=False,
    optimizations=None,
    num_layers=None,
):
    submesh_devices = create_submeshes(mesh_device, data_parallel)
    state_dict = None

    model_args = []
    model = []

    for submesh in submesh_devices:
        model_args_i, model_i, state_dict = create_multimodal_model(
            mesh_device=submesh,
            max_batch_size=max_batch_size // data_parallel,
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_paged_kv_cache=use_paged_kv_cache,
            checkpoint=state_dict,
            optimizations=optimizations,
            num_layers=num_layers,
        )
        model_args.append(model_args_i)
        model.append(model_i)

    return model_args, model


@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "test_type,max_seq_len",
    (("normal", 8 * 1024),),
    ids=["normal"],
)
@pytest.mark.parametrize(
    "warmup_iters, enable_trace, max_batch_size, include_text_only_prompts, multi_image, max_gen_len, num_layers",
    [
        (0, False, 1, False, False, 500, None),  # batch1-notrace
        (0, True, 1, False, False, 500, None),  # batch1-trace
        (0, True, 32, False, False, 500, None),  # batch32-trace
        (0, True, 4, True, False, 500, None),  # batch4-trace-with-text-prompts
        (0, False, 1, True, True, 500, None),  # batch1-multi-image-notrace
        (0, True, 1, True, True, 500, None),  # batch1-multi-image-trace
        (0, True, 1, False, False, 5, 1),  # tracy
    ],
    ids=[
        "batch1-notrace",
        "batch1-trace",
        "batch32-trace",
        "batch4-trace-with-text-prompts",
        "batch1-multi-image-notrace",
        "batch1-multi-image-trace",
        "tracy",
    ],
)
@pytest.mark.parametrize(
    "data_parallel",
    [
        1,
        # 4,
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 21448704, "num_command_queues": 2, "l1_small_size": 24576}],
    indirect=True,
)
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
    ],
    ids=["performance", "accuracy"],
)
def test_multimodal_demo_text(
    mesh_device,
    warmup_iters,
    enable_trace,
    max_batch_size,
    include_text_only_prompts,
    multi_image,
    data_parallel,
    test_type,
    max_seq_len,
    is_ci_env,
    optimizations,
    max_gen_len,
    num_layers,
    temperature: float = 0,
    top_p: float = 0.9,
    model_parallel_size: Optional[int] = None,
):
    """
    Simple multimodal demo with limited dependence on reference code.
    """
    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    max_batch_size *= data_parallel  # input batch_size is interpreted as size per DP group

    model_args, model = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        optimizations=optimizations,
        num_layers=num_layers,
    )

    HF_MODEL = model_args[0].checkpoint_type == CheckpointType.HuggingFace

    if not HF_MODEL:
        ckpt_dir = os.environ["LLAMA_DIR"]
        tokenizer_path = str(Path(ckpt_dir) / "tokenizer.model")

        tokenizer = Tokenizer(model_path=tokenizer_path)
        formatter = ChatFormat(tokenizer)
    else:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_args[0].CKPT_DIR, use_fast=True, do_convert_rgb=True)

    generator = Generator(model, model_args, mesh_device)

    xattn_caches = [
        model.setup_cache(model_args[i].max_batch_size) if not HF_MODEL else None
        for i, model in enumerate(generator.model)
    ]

    # Create random images for trace capture with specific dimensions

    trace_img_1120x560 = create_random_image(1120, 560)

    with open(IMG_PATH / "ocr_image.jpeg", "rb") as f:
        ocr_image = PIL_Image.open(f).convert("RGB")

    with open(IMG_PATH / "clutter.jpeg", "rb") as f:
        clutter = PIL_Image.open(f).convert("RGB")

    with open(IMG_PATH / "dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    if multi_image:
        handwriting_dataset_base_url = (
            "https://huggingface.co/datasets/tavishm/100-handwritten-medical-records/resolve/main/"
        )
        handwriting_dataset_images_names = [
            "1wMr9ofP.jpg",
            "4J7Jyojz.jpg",
            "68eycMkU.jpg",
            "8RQQmApQ.jpg",
            "A9Dx6iCN.jpg",
            "ANQONi6m.jpg",
            "BMfPWBSX.jpg",
            "By829MQ1.jpg",
        ]
        handwriting_dataset_images = [
            PIL_Image.open(BytesIO(requests.get(f"{handwriting_dataset_base_url}{image_name}").content))
            for image_name in handwriting_dataset_images_names
        ]
        num_handwritten_images = len(handwriting_dataset_images)

        # Trace capture dialogs with random images
        multi_image_dialogs = [
            [
                UserMessage(
                    content=[ImageMedia(image=handwriting_dataset_images[i]) for i in range(num_handwritten_images)]
                    + ["Read the handwriting on all these images."]
                )
            ],
        ]

    # Trace capture dialogs with random images
    trace_dialogs = [
        [UserMessage(content=[ImageMedia(image=ocr_image), "What is the full text of this image? Do OCR"])],
    ]

    if multi_image:
        trace_dialogs = multi_image_dialogs

    if len(trace_dialogs) < max_batch_size:
        trace_dialogs *= max_batch_size // len(trace_dialogs)

    num_trace_batches = len(trace_dialogs) // max_batch_size

    if not include_text_only_prompts:
        with open(IMG_PATH / "dog.jpg", "rb") as f:
            img = PIL_Image.open(f).convert("RGB")
        logger.info(f"Dog image dimensions: {img.size} (width x height)")

        with open(IMG_PATH / "pasta.jpeg", "rb") as f:
            img2 = PIL_Image.open(f).convert("RGB")
        logger.info(f"Pasta image dimensions: {img2.size} (width x height)")

        # Regular testing dialogs with original images
        dialogs = [
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

    if multi_image:
        dialogs = multi_image_dialogs + dialogs

    if len(dialogs) < max_batch_size:
        dialogs *= max_batch_size // len(dialogs)

    assert len(dialogs) % max_batch_size == 0
    total_users = len(dialogs)
    num_batches = total_users // max_batch_size

    sampler = get_batch_sampler(temperature, top_p, model_args[0].tokenizer)
    _num_prefill_tokens = 0
    _num_decode_tokens = 0

    prompt_encoder = hf_multimodal_encode if HF_MODEL else formatter.encode_dialog_prompt

    for iter_num in range(warmup_iters + 1):
        logger.info(f"Iteration {iter_num}")
        current_dialogs = trace_dialogs + dialogs
        for batch_idx in range(num_batches):
            batch_dialogs = current_dialogs[batch_idx * max_batch_size : (batch_idx + 1) * max_batch_size]
            for dialog in batch_dialogs:
                for msg in dialog:
                    print(f"{msg.role.capitalize()}: {msg.content}\n")

            logger.info(f"Starting processor for batch {batch_idx}")
            batch_model_input = [
                prompt_encoder(dialog, processor) if HF_MODEL else prompt_encoder(dialog, tool_prompt_format=False)
                for dialog in batch_dialogs
            ]

            if HF_MODEL:
                # Use the processor's tokenizer instead of model_args tokenizer to ensure consistency
                tokenizer = processor.tokenizer

            # Do initial prefill
            vision_images = [
                model_input.vision.images if model_input.vision else None for model_input in batch_model_input
            ]
            vision_mask = [model_input.vision.mask if model_input.vision else None for model_input in batch_model_input]
            prompt_tokens = [model_input.tokens for model_input in batch_model_input]
            # Get max length of prompts in batch
            prefill_lens = torch.tensor([len(tokens) for tokens in prompt_tokens], dtype=torch.long)
            _num_prefill_tokens += prefill_lens.sum().item()
            total_lens = prefill_lens + max_gen_len

            # Create padded tokens tensor for batch
            stop_tokens = model_args[0].tokenizer.stop_tokens
            pad_id = tokenizer.pad_token_id if HF_MODEL else tokenizer.pad_id
            bsz = len(prompt_tokens)
            tokens = torch.full((bsz, max(total_lens)), pad_id, dtype=torch.long)

            # Fill in actual tokens for each sequence in batch
            for i, seq in enumerate(prompt_tokens):
                tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

            prefill_start = time.perf_counter()
            if batch_idx < num_trace_batches:  # Get compile time for first batch
                with profiler("compile_prefill", iteration=batch_idx):
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

            # Get cached prefill time
            with profiler("inference_prefill", iteration=batch_idx):
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

            prefill_end = time.perf_counter()
            next_tokens, next_texts = sampler(batch_logits)
            for i, (next_token, next_text) in enumerate(zip(next_tokens, next_texts)):
                tokens[i, prefill_lens[i]] = next_token
            print(f"Next tokens: {next_tokens}")
            print(f"Next texts: {next_texts}")
            decode_times = []

            with profiler(f"inference_decode", iteration=batch_idx):
                for gen_idx in range(max_gen_len - 1):
                    if batch_idx == 0 and gen_idx == 0:  # First decode accounts for compile time
                        profiler.start(f"compile_decode", iteration=batch_idx)

                    decode_start = time.perf_counter()
                    position_id = prefill_lens + gen_idx
                    next_token_tensor = next_tokens.reshape(max_batch_size, 1)

                    logits = generator.decode_forward(
                        position_id,
                        next_token_tensor,
                        prefill_batch_xattn_masks,
                        prefill_batch_text_masks,
                        decode_batch_xattn_masks,
                        decode_batch_text_masks,
                        xattn_caches,
                        enable_trace=enable_trace,
                    )

                    next_tokens, next_texts = sampler(logits)
                    # Update next token
                    tokens[torch.arange(max_batch_size), position_id + 1] = next_tokens
                    decode_end = time.perf_counter()
                    decode_times.append(decode_end - decode_start)
                    if batch_idx == 0 and gen_idx == 0:
                        profiler.end(f"compile_decode", iteration=batch_idx)

                    # Disable checking for eot until I have more robust code for batch > 1
                    if HF_MODEL:
                        if any([t in stop_tokens for t in next_tokens]):
                            break
                    else:
                        # Disable checking for eot until I have more robust code for batch > 1
                        pass
                        # if text in ["<|eot_id|>", "<|eom_id|>"]:
                        #     break
                _num_decode_tokens += (
                    gen_idx * max_batch_size
                )  # gen_idx is (num_tokens - 1) to avoid counting compile iter

            # Log full text output for each user in batch
            if HF_MODEL:
                # For HF models, get vision tokens from the processor if they exist
                vision_tokens = []
            else:
                vision_tokens = [tokenizer.special_tokens["<|image|>"], 128256]

            for user_id in range(max_batch_size):
                # Remove <|image|> tokens since they break the tokenizer
                tokens_out = [
                    t if t not in vision_tokens else pad_id
                    for t in tokens[user_id].tolist()[: position_id[user_id] + 2]
                ]
                text = tokenizer.decode(tokens_out)
                logger.info(f"User {user_id} full text: {text}")

            prefill_time_ms = (prefill_end - prefill_start) * 1000
            logger.info(f"Prefill time: {prefill_time_ms:.2f} ms")
            decode_time_ms = sum(decode_times) / (gen_idx + 1) * 1000
            logger.info(f"Average decode time per token: {decode_time_ms:.2f} ms")

            # ttnn.release_trace(generator.mesh_device, trace_id)

    # End profiling
    profiler.end("run")

    # Calculate measurements
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")
    total_inference_prefill_time = profiler.get_duration_sum("inference_prefill")
    total_inference_decode_time = profiler.get_duration_sum("inference_decode", start_iteration=0) - compile_decode_time
    avg_ttft = total_inference_prefill_time / num_batches  # One first token per batch
    avg_prefill_t_s = _num_prefill_tokens / total_inference_prefill_time
    avg_decode_t_s = _num_decode_tokens / total_inference_decode_time
    avg_decode_t_s_u = _num_decode_tokens / total_inference_decode_time / max_batch_size

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_ttft,
        "prefill_t/s": avg_prefill_t_s,
        "decode_t/s/u": avg_decode_t_s_u,
        "decode_t/s": avg_decode_t_s,
    }

    # Print performance metrics
    logger.info("")
    logger.info(f"Performance metrics for batch 0")
    logger.info(f"Prefill compile time: {round(measurements['compile_prefill'], 4)}s")
    logger.info(f"Decode compile time: {round(measurements['compile_decode'], 4)}s")
    logger.info(f"Prefill inference time per user: {round(avg_ttft, 4)}s")
    logger.info(
        f"Total Decode inference time ({max_gen_len} iterations): {round(measurements['inference_decode'], 4)}s"
    )
    logger.info("")
    logger.info(f"Time to first token: {round(measurements['prefill_time_to_token']* 1000, 2)}ms")
    logger.info(f"Prefill t/s: {round(measurements['prefill_t/s'], 2)} tok/s")
    logger.info(
        f"Average speed: {round(1/avg_decode_t_s_u * 1000, 2)}ms @ {round(avg_decode_t_s_u, 2)} tok/s/user ({round(avg_decode_t_s, 2)} tok/s throughput)"
    )
    logger.info("")

    logger.info(f"is_ci_env: {is_ci_env}")
    if is_ci_env and max_batch_size == 1 and enable_trace:  # Only profiling these parametrizations
        tt_device_name = model_args[0].device_name
        base_model_name = model_args[0].base_model_name
        target_prefill_tok_s = {
            "N300_Llama-3.2-11B": 23,
            "T3K_Llama-3.2-11B": 20,
            "T3K_Llama-3.2-90B": 3,
            "N150_gemma-3-4b": 285,
            "N300_gemma-3-4b": 390,
            "T3K_gemma-3-27b": 265,
        }[f"{tt_device_name}_{base_model_name}"]

        target_decode_tok_s_u = {
            "N300_Llama-3.2-11B": 21.5,
            "T3K_Llama-3.2-11B": 35,
            "T3K_Llama-3.2-90B": 6,
            "N150_gemma-3-4b": 24,
            "N300_gemma-3-4b": 28,
            "T3K_gemma-3-27b": 13,
        }[f"{tt_device_name}_{base_model_name}"]

        target_decode_tok_s = target_decode_tok_s_u * max_batch_size
        targets = {
            "prefill_t/s": target_prefill_tok_s,
            "decode_t/s": target_decode_tok_s,
            "decode_t/s/u": target_decode_tok_s_u,
        }

        # Save benchmark data for CI
        N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}
        benchmark_data = create_benchmark_data(profiler, measurements, N_warmup_iter, targets)
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=f"{base_model_name}-Vision",
            ml_model_type="vlm",
            num_layers=model_args[0].n_layers,
            batch_size=max_batch_size,
            input_sequence_length=max(prefill_lens).item(),
            output_sequence_length=max_gen_len,
        )

        skip_perf_verification = [
            "gemma-3-4b",  # Gemma-3 functional only - perf tests are not reliable yet
            "gemma-3-27b",  # Gemma-3 functional only - perf tests are not reliable yet
        ]
        if base_model_name not in skip_perf_verification:
            verify_perf(measurements, targets, high_tol_percentage=1.15)
