# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image as PIL_Image
from transformers import AutoProcessor

from models.common.llama_models import create_vision_mask, extract_images_from_messages, sample_top_p
from models.tt_transformers.tt.generator import create_submeshes

IMG_PATH = Path("models/tt_transformers/demo/sample_prompts/llama_models").resolve()

import os
import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import Generator


def get_batch_sampler(temperature, top_p, tokenizer):
    def sample(logits):
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
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


def _is_trace(filename):
    return "trace" in filename


# load input prompts from json, return as a (list of inputs, number of trace batches)
def load_inputs(user_input, batch):
    if isinstance(user_input, (list, tuple)):
        # multiple sources, e.g. ("data_trace.json", "data.json")
        user_inputs = []
        num_trace_batches = 0
        for input_ in user_input:
            cur_inputs, n_trace = load_inputs(input_, batch)
            if n_trace > 0:
                assert num_trace_batches * batch == len(user_inputs), "trace inputs must go first"
                num_trace_batches += n_trace
            user_inputs.extend(cur_inputs)

        return user_inputs, num_trace_batches

    is_trace = False
    if isinstance(user_input, str):
        is_trace = _is_trace(user_input)
        with open(user_input, "r") as f:
            user_input = json.load(f)

    for dialog in user_input:
        for message in dialog:
            for content in message["content"]:
                if content["type"] == "image":
                    if "random" in content:
                        # [width, height] is stored
                        img = create_random_image(*content["random"])
                        del content["random"]
                    elif "llama_models" in content:
                        # image_name from llama_models resources is stored
                        with open(IMG_PATH / content["llama_models"], "rb") as f:
                            img = PIL_Image.open(f).convert("RGB")
                    elif "image" in content:
                        if isinstance(content["image"], str):
                            with open(content["image"], "rb") as f:
                                img = PIL_Image.open(f).convert("RGB")
                        elif isinstance(content["image"], PIL_Image.Image):
                            img = content["image"]
                    else:
                        raise ValueError(f"Unknown image type for {content}")

                    content["image"] = img

    if len(user_input) < batch:
        user_input *= batch // len(user_input)
    assert len(user_input) % batch == 0

    return user_input, is_trace * len(user_input) // batch


def load_expected_text(input_prompts, model_name, batch):
    if "Llama-3.2-11B" in model_name:
        model_suffix = "llama32_11B"
    elif "Llama-3.2-90B" in model_name:
        model_suffix = "llama32_90B"
    else:
        raise ValueError(f"Model {model_name} not supported")

    expected_output = []
    for path in input_prompts:
        if _is_trace(path):
            continue

        directory, filename = os.path.split(path)
        filename = os.path.splitext(filename)[0]
        filename = f"expected_{filename}_{model_suffix}.json"
        path = os.path.join(directory, filename)
        with open(path, "r") as f:
            output = json.load(f)

        if len(output) < batch:
            output *= batch // len(output)
        assert len(output) % batch == 0

        expected_output.extend(output)
    return expected_output


def create_multimodal_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat16,
    use_paged_kv_cache=False,
    checkpoint=None,
):
    from models.tt_transformers.tt.model_config import ModelArgs
    from models.tt_transformers.tt.multimodal.llama_vision_model import CrossAttentionTransformer

    tt_model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size)
    assert tt_model_args.is_llama_vision(), "This model is multimodal"

    # limit length or we'll run out of space
    tt_model_args.max_seq_len = max_seq_len
    if tt_model_args.is_90b:
        assert tt_model_args.device_name == "T3K", "90B model only supported on T3K right now"
        # for 90B model on T3K, use bfp8 and performance optimizations or the model won't fit in memory
        dtype = ttnn.bfloat8_b
        logger.info("Setting dtype to bfloat8_b for 90B model on T3K to fit model in memory")

    if checkpoint is None:
        checkpoint = tt_model_args.load_state_dict()
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
    data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat16,
    use_paged_kv_cache=False,
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
    (("normal", 512),),
    ids=["normal"],
)
@pytest.mark.parametrize(
    "warmup_iters, enable_trace, max_batch_size, input_prompts",
    [
        (
            0,  # warmup_iters
            False,  # enable_trace
            1,  # max_batch_size
            (
                "models/tt_transformers/demo/sample_prompts/vision_input_data_trace.json",
                "models/tt_transformers/demo/sample_prompts/vision_input_data.json",
            ),  # input_prompts
        ),  # batch1-notrace
        (
            0,  # warmup_iters
            True,  # enable_trace
            1,  # max_batch_size
            (
                "models/tt_transformers/demo/sample_prompts/vision_input_data_trace.json",
                "models/tt_transformers/demo/sample_prompts/vision_input_data.json",
            ),  # input_prompts
        ),  # batch1-trace
        (
            0,  # warmup_iters
            True,  # enable_trace
            16,  # max_batch_size
            (
                "models/tt_transformers/demo/sample_prompts/vision_input_data_trace.json",
                "models/tt_transformers/demo/sample_prompts/vision_input_data.json",
            ),  # input_prompts
        ),  # batch16-trace
        (
            0,  # warmup_iters
            True,  # enable_trace
            32,  # max_batch_size
            (
                "models/tt_transformers/demo/sample_prompts/vision_input_data_trace.json",
                "models/tt_transformers/demo/sample_prompts/vision_input_data.json",
            ),  # input_prompts
        ),  # batch32-trace
        (
            0,  # warmup_iters
            True,  # enable_trace
            4,  # max_batch_size
            (
                "models/tt_transformers/demo/sample_prompts/vision_input_data_trace.json",
                "models/tt_transformers/demo/sample_prompts/vision_input_data_w_text_only.json",
            ),  # input_prompts
        ),  # batch4-trace-with-text-prompts
    ],
    ids=["batch1-notrace", "batch1-trace", "batch16-trace", "batch32-trace", "batch4-trace-with-text-prompts"],
)
@pytest.mark.parametrize(
    "data_parallel",
    [
        1,
        # 4,
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": True, "trace_region_size": 17000000, "num_command_queues": 2}], indirect=True
)
def test_multimodal_demo_text(
    mesh_device,
    warmup_iters,
    enable_trace,
    max_batch_size,
    input_prompts,
    data_parallel,
    test_type,
    max_seq_len,
    is_ci_env,
    request,
    temperature: float = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = 500,
    model_parallel_size: Optional[int] = None,
):
    """
    Simple multimodal demo with limited dependence on reference code.
    """
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    if num_devices == 2:
        if max_batch_size == 1:
            pytest.skip(
                "Batch size=1 on N300 mesh experiences ND hangs: https://github.com/tenstorrent/tt-metal/issues/28247"
            )
        if max_batch_size not in (4, 16):
            pytest.skip(f"Batch size={max_batch_size} is not tested for N300 mesh")
    if num_devices == 8 and max_batch_size not in (1, 4, 32):
        pytest.skip(f"Batch size={max_batch_size} is not tested for T3K mesh")

    logger.info("Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    ckpt_dir = os.environ["HF_MODEL"]

    max_batch_size *= data_parallel  # input batch_size is interpreted as size per DP group

    model_args, model = prepare_generator_args(
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    processor = AutoProcessor.from_pretrained(ckpt_dir, local_files_only=is_ci_env)
    tokenizer = processor.tokenizer
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    xattn_caches = [model.setup_cache(model_args[i].max_batch_size) for i, model in enumerate(generator.model)]

    # Override parameters from command line if they are provided
    input_prompts = request.config.getoption("--input_prompts") or input_prompts
    dialogs, num_trace_batches = load_inputs(input_prompts, max_batch_size)

    assert len(dialogs) % max_batch_size == 0
    total_users = len(dialogs)
    num_batches = total_users // max_batch_size

    sampler = get_batch_sampler(temperature, top_p, tokenizer)
    _num_prefill_tokens = 0
    _num_decode_tokens = 0

    non_trace_generated_texts = []

    for iter_num in range(warmup_iters + 1):
        logger.info(f"Iteration {iter_num}")
        current_dialogs = dialogs
        for batch_idx in range(num_batches):
            batch_dialogs = current_dialogs[batch_idx * max_batch_size : (batch_idx + 1) * max_batch_size]
            for dialog in batch_dialogs:
                for msg in dialog:
                    content = " ".join(
                        str(value) for content in msg["content"] for key, value in content.items() if key != "type"
                    )
                    logger.info(f"{msg['role'].capitalize()}: {content}\n")
            batch_inputs = [
                processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
                for messages in batch_dialogs
            ]

            # Do initial prefill
            # TBD: rewrite generator since images are processed twice (in processor and generator)
            vision_images = [extract_images_from_messages(messages) or None for messages in batch_dialogs]
            vision_mask = [
                create_vision_mask(model_input["input_ids"][0], processor.image_token_id) or None
                for model_input in batch_inputs
            ]
            prompt_tokens = [inputs["input_ids"][0] for inputs in batch_inputs]
            # Get max length of prompts in batch
            prefill_lens = torch.tensor([len(tokens) for tokens in prompt_tokens], dtype=torch.long)
            _num_prefill_tokens += prefill_lens.sum().item()
            total_lens = prefill_lens + max_gen_len

            # Create padded tokens tensor for batch
            pad_id = tokenizer.pad_token_id
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
            logger.info(f"Next tokens: {next_tokens}")
            logger.info(f"Next texts: {next_texts}")
            decode_times = []

            with profiler("inference_decode", iteration=batch_idx):
                for gen_idx in range(max_gen_len - 1):
                    if batch_idx == 0 and gen_idx == 0:  # First decode accounts for compile time
                        profiler.start("compile_decode", iteration=batch_idx)

                    decode_start = time.perf_counter()
                    position_id = prefill_lens + gen_idx
                    next_token_tensor = next_tokens.reshape(max_batch_size, 1)

                    logits = generator.decode_forward_llama_vision(
                        position_id,
                        next_token_tensor,
                        prefill_batch_xattn_masks,
                        prefill_batch_text_masks,
                        decode_batch_xattn_masks,
                        decode_batch_text_masks,
                        xattn_caches,
                        enable_trace=enable_trace,
                    )

                    if isinstance(logits, tuple):
                        logits = logits[0]

                    next_tokens, next_texts = sampler(logits)
                    # Update next token
                    tokens[torch.arange(max_batch_size), position_id + 1] = next_tokens
                    decode_end = time.perf_counter()
                    decode_times.append(decode_end - decode_start)
                    if batch_idx == 0 and gen_idx == 0:
                        profiler.end("compile_decode", iteration=batch_idx)

                    # Disable checking for eot until I have more robust code for batch > 1
                    # if text in ["<|eot_id|>", "<|eom_id|>"]:
                    #     break
                _num_decode_tokens += (
                    gen_idx * max_batch_size
                )  # gen_idx is (num_tokens - 1) to avoid counting compile iter

            # Log full text output for each user in batch
            for user_id in range(max_batch_size):
                tokens_out = [t for t in tokens[user_id].tolist()[: position_id[user_id] + 2]]
                text = tokenizer.decode(tokens_out)
                logger.info(f"User {user_id} full text: {text}")
                if batch_idx >= num_trace_batches:
                    generated_text = tokenizer.decode(tokens_out[prefill_lens[user_id] :])
                    if tokenizer.eos_token in generated_text:
                        generated_text = generated_text[: generated_text.index(tokenizer.eos_token)]
                    non_trace_generated_texts.append(generated_text)

            prefill_time_ms = (prefill_end - prefill_start) * 1000
            logger.info(f"Prefill time: {prefill_time_ms:.2f} ms")
            decode_time_ms = sum(decode_times) / (gen_idx + 1) * 1000
            logger.info(f"Average decode time per token: {decode_time_ms:.2f} ms")

            # ttnn.release_trace(generator.mesh_device, trace_id)

    # End profiling
    profiler.end("run")

    if is_ci_env and mesh_device.get_num_devices() <= 2:
        # TODO: fix issue that models on T3K "don't see images" https://github.com/tenstorrent/tt-metal/issues/32284
        expected_output = load_expected_text(input_prompts, model_args[0].base_model_name, max_batch_size)
        from bert_score import score as bert_score

        candidates = non_trace_generated_texts
        references = expected_output
        assert len(candidates) == len(references)
        P0, R0, F10 = bert_score(
            candidates,
            references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            rescale_with_baseline=False,
            batch_size=64,
        )
        for i, (p, r, f) in enumerate(zip(P0, R0, F10)):
            logger.info(f"BERTScore (rescaled) P/R/F1 for sample {i}: {p.item():.3f}/{r.item():.3f}/{f.item():.3f}")
        # TODO: create separate targets for different samples, investigate different outputs for different batch_size (4 vs 16)
        assert F10.min().item() > 0.55, f"min BERTScore F1 ({F10.min().item()}) is lower than expected (0.55)."
        assert F10.mean().item() > 0.70, f"mean BERTScore F1 ({F10.mean().item()}) is lower than expected (0.70)."

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
    logger.info("Performance metrics for batch 0")
    logger.info(f"Prefill compile time: {round(measurements['compile_prefill'], 4)}s")
    logger.info(f"Decode compile time: {round(measurements['compile_decode'], 4)}s")
    logger.info(f"Prefill inference time per user: {round(avg_ttft, 4)}s")
    logger.info(
        f"Total Decode inference time ({max_gen_len} iterations): {round(measurements['inference_decode'], 4)}s"
    )
    logger.info("")
    logger.info(f"Time to first token: {round(measurements['prefill_time_to_token'] * 1000, 2)}ms")
    logger.info(f"Prefill t/s: {round(measurements['prefill_t/s'], 2)} tok/s")
    logger.info(
        f"Average speed: {round(1 / avg_decode_t_s_u * 1000, 2)}ms @ {round(avg_decode_t_s_u, 2)} tok/s/user ({round(avg_decode_t_s, 2)} tok/s throughput)"
    )
    logger.info("")

    logger.info(f"is_ci_env: {is_ci_env}")
    if is_ci_env and enable_trace:
        tt_device_name = model_args[0].device_name
        base_model_name = model_args[0].base_model_name

        run_config = (tt_device_name, base_model_name, max_batch_size)
        targets_prefill_tok_s = {
            ("N300", "Llama-3.2-11B", 16): 18.3,
            ("T3K", "Llama-3.2-90B", 1): 14.2,
        }
        targets_decode_tok_s_u = {
            ("N300", "Llama-3.2-11B", 16): (17, None),  # None to default to tolerance percentage (1.15)
            # second value to override default tolerance percentage (1.15); observing variance across different CI machines
            # For T3K Llama-3.2-90B, the decode_t/s/u target used to be set to 3 with a wide tolerance (4.3, i.e. 330% increase) due to high variance observed across CI machines.
            # Empirical data from CI runs (see https://github.com/tenstorrent/tt-metal/pull/31605) shows that decode performance can vary significantly, sometimes falling well below the nominal target.
            # The slow CI machine seems to be out of circulation for now, so we can use a high target to avoid spurious test failures.
            ("T3K", "Llama-3.2-90B", 1): (12, None),
        }

        perf_targets = {}
        if run_config in targets_prefill_tok_s:
            assert (
                run_config in targets_decode_tok_s_u
            ), f"Prefill targets exist, but decode targets are missing for {run_config}"

            perf_targets = {
                "prefill_t/s": targets_prefill_tok_s[run_config],
                "decode_t/s": targets_decode_tok_s_u[run_config][0] * max_batch_size,
                "decode_t/s/u": targets_decode_tok_s_u[run_config][0],
            }

            perf_tolerance = targets_decode_tok_s_u[run_config][1] or 1.15  # default to 15% tolerance

        # Save benchmark data for CI
        N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}
        benchmark_data = create_benchmark_data(profiler, measurements, N_warmup_iter, perf_targets)
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=f"{base_model_name}-Vision",
            ml_model_type="vlm",
            num_layers=model_args[0].n_layers,
            batch_size=max_batch_size,
            config_params={"data_parallel": data_parallel, "tensor_parallel": num_devices // data_parallel},
            input_sequence_length=max(prefill_lens).item(),
            output_sequence_length=max_gen_len,
        )

        if perf_targets:
            verify_perf(measurements, perf_targets, high_tol_percentage=perf_tolerance)
