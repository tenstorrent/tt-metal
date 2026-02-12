# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import datetime

import pytest
import torch
from loguru import logger
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

import ttnn
from models.demos.qwen25_vl.tt.common import (
    PagedAttentionConfig,
    merge_vision_tokens,
    multimodal_rope_from_hf,
    preprocess_inputs_prefill,
    sample_host,
)
from models.demos.qwen25_vl.tt.generator import Generator
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer, Transformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import SamplingParams
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, parse_decoder_json


def create_tt_page_table(paged_attention_config, tt_model_args):
    if paged_attention_config is None:
        return None

    # Implied shuffling of blocks
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    # Page table which maps virtual blocks to physical
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


# List of supported Parameters for demo.py
#
# input_prompts (string): input json file with prompts to process. See models/tt_transformers/demo/*.json for list of input files
# instruct (bool): Whether to use instruct weights or general weights
# repeat_batches (int): Number of consecutive batches of users to run (default: 1)
# max_seq_len (int): Maximum context length supported by the model (Llama3.1 and Llama3.2 models have a maximum context length of 128k, i.e., 128 * 1024)
# batch_size (int): Number of users in a batch (Supports 1/2/4/8/16/32 batches)
# max_generated_tokens (int): Maximum number of tokens to generate for each user (Note that the users will stop generation before this limit if they reach a EoS token)
# paged_attention (bool): Whether to use paged attention or default attention (vLLM requires paged attention)
# page_params (dict): Page parameters for paged attention (block_size, max_num_blocks) For smaller context lengths use block_size=32 and max_num_blocks=1024, for larger context use block_size=64 and max_num_blocks=2048
# sampling_params (dict): Sampling parameters for decoding (temperature, top_p). If temperature is set to 0, argmax (greedy decode) is used.
# stop_at_eos (bool): Whether to stop decoding when the model generates an EoS token
#
# optimization (ModelOptimizations): Optimization level to use for the model (performance or accuracy)
# MESH_DEVICE (str): Fake device to use for testing (N150, N300, T3K, TG). Usage: `export MESH_DEVICE=N150`, will enable running a single-chip demo on a multi-chip system.
@pytest.mark.parametrize(
    "input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stop_at_eos, ci_only",
    [
        (  # Batch-1 run (Latency) - single user, small prompt
            "models/demos/qwen25_vl/demo/sample_prompts/demo.json",  # single qwen demo prompt
            True,  # instruct mode
            1,  # repeat_batches to simulate multiple users (batch_size=1) with the same prompt
            4096,  # max_seq_len, allow for image tokens
            1,  # batch_size -- samples to load from the prompt JSON
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
        (  # Batch-32 run (Throughput) - 32 users, small prompts
            "models/demos/qwen25_vl/demo/sample_prompts/multi_prompts_32.json",
            True,  # instruct mode
            1,  # repeat_batches to simulate multiple users with the same prompt
            4096,  # max_seq_len, allow for image tokens
            32,  # batch_size -- samples to load from the prompt JSON
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 4096},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
        (  # Batch-1 run with full model for more stable BERTScore checks (CI only)
            "models/demos/qwen25_vl/demo/sample_prompts/test_bert_score.json",
            True,  # instruct mode
            2,  # repeat_batches to simulate multiple users with the same prompt
            4096,  # max_seq_len, allow for image tokens
            32,  # batch_size -- samples to load from the prompt JSON
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 4096},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            True,  # ci_only
        ),
        (  # Batch-1 run with text only prompts hence skipping vision model (CI only)
            "models/demos/qwen25_vl/demo/sample_prompts/text_only.json",
            True,  # instruct mode
            1,  # repeat_batches to simulate multiple users with the same prompt
            4096,  # max_seq_len, allow for image tokens
            1,  # batch_size -- samples to load from the prompt JSON
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 4096},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
        ),
        (  # Batch-4 run with 300 dpi scanned document (Latency) - 16k long context, real-world test
            "models/demos/qwen25_vl/demo/sample_prompts/demo_300dpi.json",  # single qwen demo prompt
            True,  # instruct mode
            1,  # repeat_batches to simulate multiple users (batch_size=1) with the same prompt
            16384,  # max_seq_len, allow for image tokens
            4,  # batch_size -- samples to load from the prompt JSON
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
        (  # Batch-2 run with 300 dpi scanned document (Latency) - 32k long context, real-world test
            "models/demos/qwen25_vl/demo/sample_prompts/demo_300dpi.json",  # single qwen demo prompt
            True,  # instruct mode
            1,  # repeat_batches to simulate multiple users (batch_size=1) with the same prompt
            32768,  # max_seq_len, allow for image tokens
            2,  # batch_size -- samples to load from the prompt JSON
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
        (  # Batch-1 run with 300 dpi scanned document (Latency) - 64k long context, real-world test
            "models/demos/qwen25_vl/demo/sample_prompts/demo_300dpi.json",  # single qwen demo prompt
            True,  # instruct mode
            1,  # repeat_batches to simulate multiple users (batch_size=1) with the same prompt
            65536,  # max_seq_len, allow for image tokens
            1,  # batch_size -- samples to load from the prompt JSON
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
        (  # Batch-1 run with 300 dpi scanned document (Latency) - 128k long context, real-world test
            "models/demos/qwen25_vl/demo/sample_prompts/demo_300dpi.json",  # single qwen demo prompt
            True,  # instruct mode
            1,  # repeat_batches to simulate multiple users (batch_size=1) with the same prompt
            131072,  # max_seq_len, allow for image tokens
            1,  # batch_size -- samples to load from the prompt JSON
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
    ],
    ids=[
        "batch-1",  # latency
        "batch-32",  # 32 users (special because it fills tile size)
        "ci-only-bert-score",  # ci_only batch-bert-score for testing coverage in CI pipelines
        "ci-only-text-only",  # ci_only batch-text-only for testing coverage in CI pipelines
        "long-context-16k",  # real-world test for 300DPI scanned document with 16k long context
        "long-context-32k",  # real-world test for 300DPI scanned document with 32k long context
        "long-context-64k",  # real-world test for 300DPI scanned document with 64k long context
        "long-context-128k",  # real-world test for 300DPI scanned document with 128k long context
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
    ci_only,
    reset_seeds,
    request,
):
    """
    Simple demo with limited dependence on reference code.
    """
    test_id = request.node.callspec.id
    if is_ci_env and (("accuracy" in test_id) or not ci_only):
        pytest.skip("CI only runs the CI-only tests")
    if not is_ci_env and ci_only:
        pytest.skip("CI only runs the CI-only tests")

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("MESH_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    if mesh_device.get_num_devices() == 1 and "Qwen2.5-VL-7B" in os.environ.get("HF_MODEL", ""):
        pytest.skip("Qwen2.5-VL-7B does not support running on N150")

    logger.info(f"mesh_device: {mesh_device}")
    use_tt_vision = True
    enable_trace = True  # Use tracing for better perf
    print_to_file = False  # Enable this flag to print the output of all users to a file

    # Override parameters from command line if they are provided
    input_prompts = request.config.getoption("--input_prompts") or input_prompts
    if request.config.getoption("--instruct") in [
        0,
        1,
    ]:  # If the flag is provided, use it. Take an int instead of bool due to parser limitations
        instruct = request.config.getoption("--instruct")
    repeat_batches = request.config.getoption("--repeat_batches") or repeat_batches
    max_seq_len = request.config.getoption("--max_seq_len") or max_seq_len
    batch_size = request.config.getoption("--batch_size") or batch_size
    max_generated_tokens = request.config.getoption("--max_generated_tokens") or max_generated_tokens
    paged_attention = request.config.getoption("--paged_attention") or paged_attention
    page_params = request.config.getoption("--page_params") or page_params
    sampling_params = request.config.getoption("--sampling_params") or sampling_params
    if request.config.getoption("--stop_at_eos") in [
        0,
        1,
    ]:  # If the flag is provided, use it. Take an int instead of bool due to parser limitations
        stop_at_eos = request.config.getoption("--stop_at_eos")
    json_config_file = request.config.getoption("--decoder_config_file")

    if json_config_file:
        optimizations = parse_decoder_json(json_config_file)
    else:
        optimizations = request.config.getoption("--optimizations") or optimizations

    if paged_attention:
        page_cache_max_seq_len = page_params["page_block_size"] * page_params["page_max_num_blocks"] / batch_size
        assert (
            max_seq_len <= page_cache_max_seq_len
        ), f"max_seq_len ({max_seq_len}) needs to be <= than page_cache_max_seq_len ({page_cache_max_seq_len})"

    if not stop_at_eos:
        logger.info(f"The decode generation will only stop at the max_generated_tokens limit == {max_generated_tokens}")

    if print_to_file:
        # Creat batch output file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = "models/demos/qwen25_vl/demo/output"
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o755)
        output_filename = f"{output_directory}/llama_text_demo_output_{timestamp}.txt"

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    input_prompts = load_inputs(input_prompts, batch_size)
    profiler.end("loading_inputs")
    assert (
        len(input_prompts) >= batch_size
    ), f"Loaded {len(input_prompts)} input prompts, expected at least {batch_size}"
    if len(input_prompts) > batch_size:
        input_prompts = input_prompts[:batch_size]
    logger.info(f"Loaded {batch_size} input prompts")

    # To simulate a deployment environment, the demo supports repeating batched prompts.
    # This loop will rotate the prompts between the users for each batch, to simulate users sending different requests
    # If batch_size=1, the same prompt is repeated for each batch
    repeat_batch_prompts = []
    for i in range(repeat_batches):
        repeat_batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

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

    processor = model_args.processor
    tokenizer = model_args.tokenizer

    # NOTE: For qwen 2.5 vl, we do not use QK fused ops
    model_args.use_qk_fused = False
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    # Load vision model and processor
    # reduce the number of layers to 1 for fast ci runs (also useful for debugging)
    from transformers import logging as transformers_logging

    # Set logging level to ERROR to suppress warnings about unexpected keys
    ref_model_name = model_args.CKPT_DIR  # allows for local model loading as well
    transformers_logging.set_verbosity_error()
    config = Qwen2_5_VLForConditionalGeneration.config_class.from_pretrained(ref_model_name)
    reference_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ref_model_name, config=config, torch_dtype="auto", device_map="auto"
    )
    if use_tt_vision:
        # Create the TorchVisionTransformer wrapper using the original vision model as reference
        vision_model_args = VisionModelArgs(
            mesh_device,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            optimizations=DecodersPrecision.accuracy(config.vision_config.depth, ref_model_name),
        )
        vision_model_args.hf_config.vision_config.depth = config.vision_config.depth
        visual_model = DropInVisionTransformer(reference_model.visual, vision_model_args, debug=False)  # show PCC
    else:
        visual_model = reference_model.visual
    processor = AutoProcessor.from_pretrained(ref_model_name)
    num_tokens_generated_decode = []
    num_image_tokens = []

    text_outputs = []
    text_outputs_all_users_all_batches = []
    logger.info("Starting inference...")
    for batch_idx, input_prompts in enumerate(repeat_batch_prompts):
        logger.info(f"Processing batch {batch_idx}")

        # Create new page table for each batch
        page_table = create_tt_page_table(paged_attention_config, model_args)

        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        text = processor.apply_chat_template(input_prompts, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(input_prompts)
        inputs = processor(
            text=text,  # [INFO] Qwen2VLProcessor handles the case where text is a string or a list of strings
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if image_inputs:
            merge_length = processor.image_processor.merge_size**2
            num_image_tokens.append([inputs.image_grid_thw[i].prod().item() // merge_length for i in range(batch_size)])
            logger.info(f"num_image_tokens: {num_image_tokens[-1]}")
        else:
            # text-only
            num_image_tokens.append([0] * batch_size)

        # Vision prefill
        logger.info(f"Vision model prefill batch {batch_idx}")
        profiler.start(f"vision_model_prefill", iteration=batch_idx)
        image_embeds = (
            visual_model(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
            if "pixel_values" in inputs
            else torch.tensor([], dtype=torch.bfloat16)
        )
        profiler.end(f"vision_model_prefill", iteration=batch_idx)

        # Prepare text + vision inputs for decoder model
        logger.info(f"Prepare text + vision inputs for decoder model batch {batch_idx}")
        # FIXME: on-host embeddings - run as part of vision model prefill when merge_vision_tokens is ported to ttnn
        text_embeds = reference_model.model.language_model.embed_tokens(inputs.input_ids)
        input_embeds = merge_vision_tokens(inputs.input_ids, text_embeds, image_embeds, reference_model.config)
        pad_token_id = tokenizer.pad_token_id
        assert (
            model_args.max_seq_len >= max(len(x) for x in input_embeds) + max_generated_tokens
        ), f"max_seq_len ({model_args.max_seq_len}) must be >= than max prompt length ({max(len(x) for x in input_embeds)}) + max generated tokens ({max_generated_tokens})"
        (
            input_prefill_pt,
            decoding_pos,  # Position where decoding should start for each user
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_embeds,
            model_args,
            inputs.attention_mask,
            pad_embedding=reference_model.model.language_model.embed_tokens(torch.tensor(pad_token_id)),
        )
        # Get user-specific rotary position embeddings
        cos, sin, rope_deltas = multimodal_rope_from_hf(
            inputs, input_embeds, reference_model, model_args, pad_token_id=pad_token_id
        )
        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

        logger.info("Starting prefill warmup...")
        profiler.start(f"compile_prefill", iteration=batch_idx)
        # [INFO] prefill_forward_text is read-only of the cos/sin matrices
        logits = generator.prefill_forward_text(
            input_prefill_pt[0].unsqueeze(0),  # Just warmup prefill for 1 user
            rot_mats=(cos, sin),
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )
        profiler.end(f"compile_prefill", iteration=batch_idx)
        logger.info("Finished prefill warmup")

        logger.info(f"Starting prefill...")
        profiler.start(f"inference_prefill", iteration=batch_idx)
        logits = generator.prefill_forward_text(
            input_prefill_pt,
            rot_mats=(cos, sin),
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )
        # [INFO] update the cos/sin matrices in the rope_setup to get ready for decode
        generator.update_rope_deltas([rope_delta.item() for rope_delta in rope_deltas])
        # torch.save(logits, f"ttnn_logits.pt")
        prefilled_token = torch.argmax(logits, dim=-1)
        profiler.end(f"inference_prefill", iteration=batch_idx)
        logger.info(f"Prefill finished")

        # Initial positions continuing from prefill, no need to offset by rope_deltas
        current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])

        # Start decoding
        iteration = 0
        argmax_on_device = model._supports_on_device_sampling
        if argmax_on_device:
            logger.info(f"Using on-device sampling with temperature=0.0, top_k=-1, top_p=1.0")
            device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
        else:
            logger.info(
                f"Using host sampling with temperature={sampling_params['temperature']}, top_p={sampling_params['top_p']}"
            )
            device_sampling_params = None

        users_decoding = True
        user_done = [False] * batch_size  # Keeps track when a user reaches EoD token
        # Keep track of generated outputs to print out every iteration
        all_outputs = [
            [] for _ in range(batch_size)
        ]  # we don't know how much of the prompt was prefilled because some will be image tokens
        for user in range(batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        out_tok = prefilled_token

        logger.info(f"Starting decode loop...")

        # Log total inference (accounting for compile_decode as well)
        profiler.start(f"inference_decode", iteration=batch_idx)
        while users_decoding:
            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=batch_idx)

            # Run decode forward
            logits, _ = generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=enable_trace,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                sampling_params=device_sampling_params,
            )

            # Get the next token
            if argmax_on_device:
                out_tok = logits.unsqueeze(1)
            else:
                # TODO Fix use case with temperature > 0
                _, out_tok = sample_host(
                    logits,
                    None,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    on_host=True,
                )

            if iteration == 0:  # First iteration will account the compile time
                profiler.end(f"compile_decode", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=batch_idx)
            else:
                profiler.end(f"inference_decode_time_{iteration}", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=batch_idx)

            # Always print perf after every iteration
            tokens_per_second_per_user = 1 / decode_iteration_time
            logger.info(
                f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            current_pos += 1

            # Save output token to print out later
            for user in range(batch_size):
                user_tok = out_tok[user].item()
                if (
                    user_tok not in tokenizer.stop_tokens and user_done[user] == False
                ):  # Read until an eos token (e.g. <|eot_id|>); create_tokenizer adds stop_tokens to HF tokenizers
                    all_outputs[user].append(user_tok)
                else:
                    if (
                        stop_at_eos
                    ):  # For performance gathering in CI, we want to sometimes force decoding for a fixed number of iterations
                        user_done[user] = True
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False
                    else:
                        all_outputs[user].append(user_tok)

            # Print out generated outputs for each user at the end of every iteration
            if not is_ci_env:
                for user in range(batch_size):
                    text = "".join(tokenizer.decode(all_outputs[user]))
                    if len(text) > 100:
                        text = "..." + text[-97:]
                    text = text.replace("\n", " ")
                    logger.info("[User {}] {}".format(user, text))

            iteration += 1

            # Upper limit of generated tokens for each user
            if iteration >= max_generated_tokens:
                users_decoding = False

            # Final print
            if not users_decoding:
                profiler.start(f"log_saving_file", iteration=batch_idx)
                logger.info("Finished decoding, printing the final outputs...\n")
                for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                    text = tokenizer.decode(output)
                    prompt_including_assistant_tags = tokenizer.decode(
                        model_args.encode_prompt(prompt, instruct=instruct)
                    )
                    text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
                    if print_to_file:
                        with open(output_filename, "a") as f:
                            f.write(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n"
                            )
                    else:
                        # Strip leading newlines from output when sent to terminal
                        short_prompt = (
                            (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:])
                            if len(prompt) > 200
                            else prompt
                        )
                        logger.info(
                            f"\n==REPEAT BATCH {batch_idx}\n==USER {i} - PROMPT\n{short_prompt} \n==USER {i} - OUTPUT\n{text_after_prompt.strip()}\n"
                        )
                    text_outputs_all_users_all_batches.append(text_after_prompt)

                profiler.end(f"log_saving_file", iteration=batch_idx)

        num_tokens_generated_decode.append(iteration)  # Save the number of tokens generated for each repeat batch

        profiler.end(f"inference_decode", iteration=batch_idx)

        # when doing repeating batches, set kv-caches to zero, to avoid context leaking
        logger.info("KV cache reset after warmup to prevent interference between users")
        if batch_idx != 0:
            for layer in model.layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        text_outputs.append(text_after_prompt)

    # Finish profiling at the end of inference for all repeated batches
    profiler.end("run")

    # Quick sanity check that the model doesn't produce special tokens=garbage output
    is_special_tokens_produced = [False] * len(all_outputs)
    for i, output in enumerate(all_outputs):
        # output = output[len(encoded_prompts[i]):]
        is_eos = [token in tokenizer.stop_tokens for token in output]
        if any(is_eos):
            output = output[: is_eos.index(True)]
        is_special_tokens_produced[i] = any(token in tokenizer.all_special_ids for token in output)
    if any(is_special_tokens_produced):
        logger.warning(f"{sum(is_special_tokens_produced)}/{len(all_outputs)} users produced special tokens")
        if is_ci_env:
            raise RuntimeError("Model produced special tokens")

    if is_ci_env and "bert-score" in test_id:
        expected_output = load_expected_text(model_args.base_model_name)
        from bert_score import score as bert_score

        candidates = text_outputs_all_users_all_batches
        references = [expected_output] * len(candidates)
        P0, R0, F10 = bert_score(
            candidates,
            references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            rescale_with_baseline=False,
            batch_size=64,
        )
        for i, (p, r, f) in enumerate(zip(P0, R0, F10)):
            logger.debug(
                f"BERTScore (rescaled) P/R/F1 for sample {i % batch_size} of batch {i // batch_size}: "
                f"{p.item():.3f}/{r.item():.3f}/{f.item():.3f}"
            )
        logger.info(f"Mean BERTScore F1 (raw): {F10.mean().item():.3f}")
        assert F10.mean().item() > 0.75, f"BERTScore F1 (raw) is lower than expected."

    # Prepare profile benchmark metrics for the last repeat batch only -- batch_idx'th batch
    compile_prefill_time = profiler.get_duration("compile_prefill", iteration=batch_idx)
    compile_decode_time = profiler.get_duration("compile_decode", iteration=batch_idx)

    total_inference_prefill_time = profiler.get_duration("inference_prefill", iteration=batch_idx)
    total_inference_decode_time = 0
    for i in range(1, iteration):  # i == 0 is the compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}", iteration=batch_idx)

    # Average prefill time for each user
    avg_time_to_first_token = total_inference_prefill_time / batch_size
    # Average decode time per batch iteration
    avg_decode_iteration_time = total_inference_decode_time / (iteration - 1)

    prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * batch_size
    decode_tok_s_user = (num_tokens_generated_decode[0] - 1) / total_inference_decode_time  # Remove the compile time
    decode_tok_s = (
        (num_tokens_generated_decode[0] - 1) / total_inference_decode_time * batch_size
    )  # Remove the compile time

    vision_model_time = profiler.get_duration("vision_model_prefill", iteration=batch_idx)
    vision_model_time_per_user = vision_model_time / batch_size
    vision_model_t_s = sum(num_image_tokens[0]) / vision_model_time
    vision_model_t_u_s = vision_model_t_s / batch_size

    measurements = {
        # Required measurements
        "vision_model_prefill": vision_model_time,
        "vision_model_prefill time per user": vision_model_time_per_user,
        "vision_model_prefill time per user per token": vision_model_t_u_s,
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,  # tokens/s
        "decode_t/s/u": decode_tok_s_user,  # tokens/s/u
        "decode_t/s": decode_tok_s,  # tokens/s
        # Optional measurements
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # Decode performance for some specific tokens
    tok_1_perf = profiler.get_duration(
        f"inference_decode_time_{1}", iteration=batch_idx
    )  # inference_decode_time_0 includes compile time
    tok_128_perf = profiler.get_duration(f"inference_decode_time_{127}", iteration=batch_idx) if 127 < iteration else 0
    tok_1024_perf = (
        profiler.get_duration(f"inference_decode_time_{1023}", iteration=batch_idx) if 1023 < iteration else 0
    )
    tok_4096_perf = (
        profiler.get_duration(f"inference_decode_time_{4095}", iteration=batch_idx) if 4095 < iteration else 0
    )

    if not stop_at_eos:
        logger.info(f"Please note that 'stop_at_eos' is disabled. Output repetition is expected.")

    logger.info("")
    logger.info(f"=== Performance metrics ===")
    logger.info(
        f"1st token decode time: {tok_1_perf*1000:.2f}ms [{round(1/tok_1_perf, 2)} t/s/u, {round((1/tok_1_perf)*batch_size, 2)} t/s]"
    )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf*1000:.2f}ms [{round(1/tok_128_perf, 2)} t/s/u, {round((1/tok_128_perf)*batch_size, 2)} t/s]"
        )
    if tok_1024_perf > 0:
        logger.info(
            f"1024th token decode time: {tok_1024_perf*1000:.2f}ms [{round(1/tok_1024_perf, 2)} t/s/u, {round((1/tok_1024_perf)*batch_size, 2)} t/s]"
        )
    if tok_4096_perf > 0:
        logger.info(
            f"4096th token decode time: {tok_4096_perf*1000:.2f}ms [{round(1/tok_4096_perf, 2)} t/s/u, {round((1/tok_4096_perf)*batch_size, 2)} t/s]"
        )

    # Print some of the perf metrics
    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Vision model prefill time: {round(vision_model_time, 2)}s")
    logger.info(
        f"Vision model prefill speed: {round(vision_model_t_u_s, 2)} tok/s/user ({round(vision_model_t_s, 2)} tok/s)"
    )
    logger.info("")
    logger.info(f"Text model average Time to First Token (TTFT): {round(avg_time_to_first_token*1000, 2)}ms")
    logger.info(
        f"Text model average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ {round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )

    # Benchmark targets
    supported_models = []
    supported_devices = []

    tt_device_name = model_args.device_name

    if model_args.base_model_name in supported_models:
        assert tt_device_name in supported_devices, f"Device {tt_device_name} not supported"

        # Set the target times to first token for every combination of device and model
        target_prefill_tok_s = {}[f"{tt_device_name}_{model_args.base_model_name}"]

        # Set the target decode timesfor every combination of device and model
        target_decode_tok_s_u = {}[f"{tt_device_name}_{model_args.base_model_name}"]

        target_decode_tok_s = target_decode_tok_s_u * batch_size
        targets = {
            "prefill_t/s": target_prefill_tok_s,
            "decode_t/s": target_decode_tok_s,
            "decode_t/s/u": target_decode_tok_s_u,
        }
    else:
        logger.warning(f"Model {model_args.base_model_name} not does not have performance targets set")
        targets = {}

    # Save benchmark data for CI dashboard
    if is_ci_env:
        # Instead of running warmup iterations, the demo profiles the initial compile iteration
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        # Save the decode performance of every iteration for plotting in superset
        for i in range(1, iteration):
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                f"time_to_token_{i}",
                profiler.get_duration(f"inference_decode_time_{i}", iteration=batch_idx) * 1000,
                step_warm_up_num_iterations=None,
                target=None,
            )

        # Also save the avg decode performance for the 128 iterations (excluding the compile time)
        inference_decode_time_first_128 = sum(
            profiler.get_duration(f"inference_decode_time_{i}", iteration=batch_idx) for i in range(1, 128)
        )
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "avg_decode_time_first_128",
            inference_decode_time_first_128 * 1000 / 127,
            step_warm_up_num_iterations=None,
            target=None,
        )

        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=model_args.base_model_name,
            ml_model_type="llm",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            config_params={"data_parallel": 1, "tensor_parallel": mesh_device.get_num_devices()},
            input_sequence_length=max(prefill_lens),
            output_sequence_length=num_tokens_generated_decode[0],
        )


def load_inputs(input_file, batch_size):
    with open(input_file, "r") as f:
        user_input = json.load(f)
    if len(user_input) < batch_size:
        logger.warning(
            f"Number of users in the file {input_file} is less than the provided batch={batch_size}. Repeating the prompts to match the batch size."
        )
        user_input = user_input * batch_size
    return user_input


def load_expected_text(model_name):
    if "Qwen2.5-VL-72B" in model_name:
        input_file = "models/demos/qwen25_vl/demo/sample_prompts/expected_text_72B.txt"
    elif "Qwen2.5-VL-32B" in model_name:
        input_file = "models/demos/qwen25_vl/demo/sample_prompts/expected_text_32B.txt"
    elif "Qwen2.5-VL-7B" in model_name:
        input_file = "models/demos/qwen25_vl/demo/sample_prompts/expected_text_7B.txt"
    elif "Qwen2.5-VL-3B" in model_name:
        input_file = "models/demos/qwen25_vl/demo/sample_prompts/expected_text_3B.txt"
    else:
        raise ValueError(f"Model {model_name} not supported")

    with open(input_file, "r") as f:
        expected_text = f.read()

    return expected_text
