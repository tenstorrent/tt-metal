# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from functools import partial

import torch
import torch.nn.functional as F
import ttnn
from loguru import logger
from models.demos.falcon7b_common.tt.falcon_causallm import TtFalconCausalLM
from models.demos.falcon7b_common.tt.model_config import get_model_config
from models.demos.falcon7b_common.tests.test_utils import (
    initialize_kv_cache,
    load_hf_model,
    synchronize_devices,
    get_num_devices,
)
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf, check_tokens_match
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    nearest_32,
    tt_tensors_to_torch_tensors,
)
from models.perf.benchmarking_utils import BenchmarkProfiler
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.generation.utils import top_k_top_p_filtering

END_OF_TEXT = 11
SPACE = 204


# load from jason, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["question"])
    return in_prompt


def post_process(logits, index):
    next_token_logits = logits[:, index, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    ids = next_tokens[:, None]
    return ids


def preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len, perf_mode=False):
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        input_prompts, padding="max_length", max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )
    prefill_ids = tokenized_inputs["input_ids"]

    tokenized_inputs_nopad = tokenizer(
        input_prompts, padding=False, max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )

    num_users = len(tokenized_inputs_nopad["input_ids"])
    num_input_tokens = len(tokenized_inputs_nopad["input_ids"][0])
    for input_prompt in tokenized_inputs_nopad["input_ids"]:
        assert len(input_prompt) == num_input_tokens

    if not perf_mode:
        prefill_ids = prefill_ids[:, : nearest_32(num_input_tokens)]  # only pad up to nearest 32, not max seq len
    else:
        num_input_tokens = max_seq_len - 1

    logger.info(f"# of users: {num_users}")
    logger.info(f"# of input tokens per user: {num_input_tokens}")

    return prefill_ids, num_users, num_input_tokens


def print_output_prompts(generated_ids, tokenizer, batch_size, num_users_to_display=None):
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for user_id, output_prompt in enumerate(output_prompts[:num_users_to_display]):
        if user_id % batch_size == 0:
            logger.info(f"\n\n=============== Results for device {(user_id // batch_size) + 1} ===============\n")
        logger.info(f"Output for user {user_id}:\n{output_prompt}")


def update_model_config(model, model_config_str, prefill_seq_len=0, decode_batch_size=32):
    model.model_config.update(get_model_config(model_config_str, prefill_seq_len))


def top_pk_logits(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    next_token_logscores = top_k_top_p_filtering(logits, top_k=k, top_p=p)
    probs = F.softmax(next_token_logscores / temperature, dim=-1)
    token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    if return_probs:
        return token, probs
    else:
        return token


def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    probs = torch.nan_to_num(probs)  # convert nan to num to prevent error in multinomial
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


def run_falcon_demo_kv(
    user_input,
    batch_size,
    max_seq_len,
    model_config_strs_prefill_decode,
    model_location_generator,
    get_tt_cache_path,
    device_mesh,  # can be ttnn.Device or ttnn.DeviceMesh
    model_version="tiiuae/falcon-7b-instruct",
    num_layers=32,
    perf_mode=False,  # Option to measure perf using max seq length (with invalid outputs)
    greedy_sampling=False,  # Option to use greedy decoding instead of top-k/p
    expected_perf_metrics=None,  # Expected perf (t/s) for prefill and decode in perf mode
    expected_greedy_output_path=None,  # Path for expected outputs for greedy decoding
    save_generated_text_path=None,  # If provided, save generated text to this path (e.g. set to expected_greedy_output_path to update expected output)
    csv_perf_targets={},  # Optional perf targets for CSV output
    is_ci_env=False,  # Whether is running in CI environment
):
    profiler = BenchmarkProfiler()
    profiler.start("run")

    assert not (expected_perf_metrics and expected_greedy_output_path), "Cannot verify both perf and output!"
    assert not (perf_mode and save_generated_text_path), "Cannot save generated text in perf mode!"
    if expected_greedy_output_path is not None:
        assert (
            not perf_mode and greedy_sampling
        ), "Output verification only supported for greedy sampling in default mode!"
    elif expected_perf_metrics is not None:
        assert perf_mode, "Performance verification is only supported for perf mode!"

    # Set up warmup iterations and targets dicts for saving benchmark data
    if perf_mode:
        N_warmup_iter = {"inference_prefill": 5, "inference_decode": 10}  # Number of warmup iterations for perf mode
    else:
        N_warmup_iter = {}

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    num_devices = get_num_devices(device_mesh)
    global_batch = batch_size * num_devices

    torch.manual_seed(0)

    if perf_mode:
        logger.info("Running in performance measurement mode (invalid outputs)!")

    profiler.start(f"loading_inputs")
    if num_devices > 1:
        assert len(user_input) == global_batch, "Number of users must be equal to batch size * number of devices!"
    if len(user_input) == 1:
        input_prompts = user_input
    else:
        input_prompts = load_inputs(user_input, global_batch)

    profiler.end(f"loading_inputs")

    logger.info("Tokenizing inputs...")
    profiler.start(f"tokenizing_inputs")

    tokenizer = AutoTokenizer.from_pretrained(model_version)
    prefill_ids, num_users, num_input_tokens = preprocess_and_validate_inputs(
        input_prompts, tokenizer, max_seq_len, perf_mode
    )
    profiler.end(f"tokenizing_inputs")

    model_config = get_model_config(model_config_strs_prefill_decode[0], nearest_32(num_input_tokens), batch_size)
    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    # State dict is needed for embeddings
    logger.info("Loading huggingface weights...")
    profiler.start(f"loading_weights")
    hugging_face_reference_model, state_dict = load_hf_model(model_location_generator, model_version)
    configuration = hugging_face_reference_model.config
    logger.info("Loading weights finished!")
    profiler.end(f"loading_weights")

    synchronize_devices(device_mesh)

    logger.info("Moving weights (single layer) to device...")
    base_url = ""

    tt_FalconCausalLM_singlelayer = TtFalconCausalLM(
        device_mesh,
        state_dict,
        base_url,
        1,
        configuration,
        max_seq_len,
        model_config,
        tt_cache_path,
        nearest_32(num_input_tokens),
    )  # single layer only used for compile
    logger.info("Moved weights (single layer) to device!")

    synchronize_devices(device_mesh)

    logger.info("Initializing KV cache...")
    profiler.start(f"initializing_KV_cache")
    kv_cache_singlelayer = initialize_kv_cache(
        configuration,
        1,
        batch_size,
        max_seq_len,
        device_mesh,
    )  # only used for compile
    kv_cache = initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device_mesh)
    profiler.end(f"initializing_KV_cache")

    ### First prefill run with compile ###
    logger.info("Running 1st run prefill stage with compile...")
    use_cache = True
    profiler.start("compile_prefill")
    N = num_users // num_devices if not perf_mode else 1
    for user_id in tqdm(range(N)):
        (
            tt_prefill_input_ids,
            tt_prefill_attention_mask,
        ) = tt_FalconCausalLM_singlelayer.model_preprocessing(
            "prefill", prefill_ids[user_id::batch_size], 0, num_input_tokens=nearest_32(num_input_tokens)
        )
        assert tt_prefill_attention_mask is not None

        tt_logits, kv_cache_singlelayer = tt_FalconCausalLM_singlelayer(
            input_ids=tt_prefill_input_ids,
            llm_mode="prefill",
            attention_mask=tt_prefill_attention_mask,
            user_id=user_id,
            layer_past=kv_cache_singlelayer,
            layer_past_len=0,
            use_cache=use_cache,
        )
        synchronize_devices(device_mesh)

        tt_prefill_input_ids.deallocate()
        if tt_prefill_attention_mask is not None:
            if isinstance(tt_prefill_attention_mask, ttnn.Tensor):
                tt_prefill_attention_mask.deallocate()
            elif isinstance(tt_prefill_attention_mask, list):
                for tt_attention_mask_element in tt_prefill_attention_mask:
                    tt_attention_mask_element.deallocate()
            else:
                raise ValueError("Invalid type for tt_attention_mask")
        tt_logits.deallocate()

    profiler.end("compile_prefill")

    synchronize_devices(device_mesh)
    logger.info("Finished 1st run prefill stage with compile!")

    ### First run decode stage with compile ###
    logger.info("Running 1st run decode stage with compile...")

    # Update model config
    update_model_config(tt_FalconCausalLM_singlelayer, model_config_strs_prefill_decode[1], batch_size)

    decode_ids = torch.randint(low=0, high=configuration.vocab_size - 1, size=(global_batch, 1), dtype=torch.int64)

    profiler.start("compile_decode")
    for kv_cache_len in tqdm(range(num_input_tokens, max_seq_len, 32)):
        (
            tt_decode_input_ids,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM_singlelayer.model_preprocessing(
            "decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1
        )
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache_singlelayer = tt_FalconCausalLM_singlelayer(
            input_ids=tt_decode_input_ids,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache_singlelayer,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        synchronize_devices(device_mesh)

        tt_decode_input_ids.deallocate()
        if tt_decode_attention_mask is not None:
            tt_decode_attention_mask.deallocate()
        tt_logits.deallocate()

    profiler.end("compile_decode")

    logger.info("Finished 1st run decode stage with compile!")
    synchronize_devices(device_mesh)

    del tt_logits
    del tt_prefill_input_ids
    del tt_prefill_attention_mask
    del decode_ids
    del tt_decode_input_ids
    del tt_FalconCausalLM_singlelayer
    del kv_cache_singlelayer

    logger.info("Moving weights (all layers) to device; might take some time...")
    profiler.start(f"moving_to_device")
    tt_FalconCausalLM = TtFalconCausalLM(
        device_mesh,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_seq_len,
        get_model_config(model_config_strs_prefill_decode[0], nearest_32(num_input_tokens), batch_size),
        tt_cache_path,
        nearest_32(num_input_tokens),
    )
    logger.info("Moved weights (all layers) to device!")
    profiler.end(f"moving_to_device")

    ### Second prefill run without compile ###
    enable_persistent_kernel_cache()

    post_processor = partial(post_process)
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    logger.info("Running inference prefill stage...")
    profiler.start("inference_prefill_decode")
    profiler.start("inference_prefill")
    time_prefill_inference = 0
    if not perf_mode:
        N_prefill = num_users // num_devices
        N_warmup_prefill = 0
    else:
        N_prefill = 15
        N_warmup_prefill = N_warmup_iter["inference_prefill"]
    for i in tqdm(range(N_prefill)):
        user_id = i if not perf_mode else 0
        time_prefill_inference_start = time.time()
        (
            tt_prefill_input_ids,
            tt_prefill_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing(
            "prefill", prefill_ids[user_id::batch_size], 0, num_input_tokens=nearest_32(num_input_tokens)
        )
        assert tt_prefill_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_ids=tt_prefill_input_ids,
            llm_mode="prefill",
            attention_mask=tt_prefill_attention_mask,
            user_id=user_id,
            layer_past=kv_cache,
            layer_past_len=0,
            use_cache=use_cache,
        )
        synchronize_devices(device_mesh)

        if tt_prefill_attention_mask is not None:
            if isinstance(tt_prefill_attention_mask, ttnn.Tensor):
                tt_prefill_attention_mask.deallocate()
            elif isinstance(tt_prefill_attention_mask, list):
                for tt_attention_mask_element in tt_prefill_attention_mask:
                    tt_attention_mask_element.deallocate()
            else:
                raise ValueError("Invalid type for tt_attention_mask")

        logits = tt_tensors_to_torch_tensors(tt_logits, device_mesh, concat_dim=0).squeeze(1)

        tt_prefill_input_ids.deallocate()
        tt_logits.deallocate()

        user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
        output_ids[user_id::batch_size] = user_output_ids

        if i >= N_warmup_prefill:
            time_prefill_inference += time.time() - time_prefill_inference_start

    profiler.end("inference_prefill")
    logger.info("Finished inference prefill stage!")
    num_users_generated_prefill = num_users if not perf_mode else (N_prefill - N_warmup_prefill) * num_devices
    prefill_time_to_token_per_user = time_prefill_inference / (N_prefill - N_warmup_prefill)

    if not perf_mode:
        generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    ### Inference run decode ###
    logger.info("Running inference decode stage...")

    # Update model config
    update_model_config(tt_FalconCausalLM, model_config_strs_prefill_decode[1], batch_size)

    decode_ids = torch.zeros(global_batch, 1, dtype=torch.int64)
    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]

    profiler.start("inference_decode")
    time_decode_inference = 0
    if not perf_mode:
        N_decode = max_seq_len - num_input_tokens
        N_warmup_decode = 0
    else:
        N_decode = 30
        N_warmup_decode = N_warmup_iter["inference_decode"]
    print_per_generated_token = (
        expected_greedy_output_path is None and num_devices == 1 and not is_ci_env
    )  # print per generated token if not verifying outputs and single device
    for output_token_index in (
        range(N_decode) if print_per_generated_token else tqdm(range(N_decode), desc="Generating tokens")
    ):
        time_decode_inference_start = time.time()
        (
            tt_decode_input_ids,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing("decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1)
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_ids=tt_decode_input_ids,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        synchronize_devices(device_mesh)

        logits = tt_tensors_to_torch_tensors(tt_logits, device_mesh, concat_dim=2).squeeze(1)

        tt_decode_input_ids.deallocate()
        if tt_decode_attention_mask is not None:
            tt_decode_attention_mask.deallocate()
        tt_logits.deallocate()

        if greedy_sampling:
            decode_ids = post_processor(logits=logits, index=...).reshape(global_batch, 1)
        else:
            decode_ids = top_pk_logits_efficient(logits.reshape(global_batch, -1)).reshape(global_batch, 1)

        if output_token_index >= N_warmup_decode:
            time_decode_inference += time.time() - time_decode_inference_start

        if not perf_mode:
            for user_id, user_decode_id in enumerate(decode_ids[:num_users]):
                if user_decode_id == END_OF_TEXT:
                    prompt_is_done[user_id] = True
                if prompt_is_done[user_id]:
                    decode_ids[user_id] = SPACE

            if all(prompt_is_done):
                break

            generated_ids = torch.concat((generated_ids, decode_ids[:num_users]), dim=1)
            kv_cache_len += 1

            if print_per_generated_token:
                os.system("clear")
                print_output_prompts(generated_ids, tokenizer, batch_size)

    profiler.end("inference_decode")
    profiler.end("inference_prefill_decode")
    logger.info("Finished inference decode stage!")
    num_tokens_generated_decode = global_batch * (output_token_index - N_warmup_decode + 1)
    decode_time_to_token_per_user = time_decode_inference / (output_token_index - N_warmup_decode + 1)
    logger.info(f"Total number of tokens generated in decode: {num_tokens_generated_decode}")

    if not perf_mode:
        print_output_prompts(generated_ids, tokenizer, batch_size)
        generated_text = tokenizer.batch_decode(generated_ids.tolist())
        if save_generated_text_path is not None:
            with open(save_generated_text_path, "w") as f:
                json.dump(generated_text, f)
    else:
        generated_text = None

    time_prefill_compile = profiler.get_duration("compile_prefill")
    time_decode_compile = profiler.get_duration("compile_decode")
    measurements = {
        "preprocessing": profiler.get_duration("tokenizing_inputs"),
        "loading_weights": profiler.get_duration("loading_weights"),
        "moving_to_device": profiler.get_duration("moving_to_device"),
        "initializing_KV_cache": profiler.get_duration("initializing_KV_cache"),
        "compile_prefill": time_prefill_compile,
        "compile_decode": time_decode_compile,
        "compile_total": time_prefill_compile + time_decode_compile,
        "inference_prefill": time_prefill_inference,
        "inference_decode": time_decode_inference,
        "inference_total": time_prefill_inference + time_decode_inference,
        "prefill_time_to_token": prefill_time_to_token_per_user,  # time to first output token (1 user)
        "inference_user_throughput_prefill": num_users_generated_prefill / time_prefill_inference,  # users/s
        "prefill_t/s": num_users_generated_prefill / time_prefill_inference * prefill_ids.shape[1],  # tokens/s
        "decode_t/s": num_tokens_generated_decode / time_decode_inference,  # tokens/s
        "decode_t/s/u": num_tokens_generated_decode / time_decode_inference / global_batch,  # tokens/s/user
        "prefill_decode_t/s/u": 1.0 / (prefill_time_to_token_per_user + decode_time_to_token_per_user),  # tokens/s/user
    }

    # Add token verification measurement (1 for pass or 0 for fail)
    if expected_greedy_output_path is not None:
        token_check_does_pass, expected_output = check_tokens_match(generated_text, expected_greedy_output_path)
        measurements["token_verification"] = float(token_check_does_pass)

    logger.info(f"pre processing: {round(measurements['preprocessing'], 5)} s")
    logger.info(f"loading weights (+downloading if not on machine): {round(measurements['loading_weights'], 5)} s")
    logger.info(
        f"conversion to TT (if downloaded) and moving weights to device: {round(measurements['moving_to_device'], 5)} s"
    )
    logger.info(f"initializing KV cache: {round(measurements['initializing_KV_cache'], 5)} s")
    logger.info(f"prefill compile time (single layer): {round(measurements['compile_prefill'],5)} s")
    logger.info(f"decode compile time (single layer): {round(measurements['compile_decode'], 5)} s")
    logger.info(f"total compile time (single layer): {round(measurements['compile_total'], 5)} s")
    logger.info(f"prefill inference time: {round(measurements['inference_prefill'], 5)} s")
    logger.info(f"decode inference time: {round(measurements['inference_decode'], 5)} s")
    logger.info(f"total inference time: {round(measurements['inference_total'], 5)} s")
    logger.info(f"time to first token (prefill, 1st user): {round(measurements['prefill_time_to_token'], 5)} s")
    logger.info(f"inference throughput prefill: {round(measurements['inference_user_throughput_prefill'], 5)} users/s")
    logger.info(
        f"inference throughput prefill | seq_len={prefill_ids.shape[1]} : {round(measurements['prefill_t/s'], 5)} tok/s"
    )
    logger.info(f"inference throughput decode: {round(measurements['decode_t/s'], 5)} tok/s")
    logger.info(f"inference throughput decode (per user): {round(measurements['decode_t/s/u'], 5)} tok/s/user")
    logger.info(
        f"inference throughput prefill+decode (per user): {round(measurements['prefill_decode_t/s/u'], 5)} tok/s/user"
    )

    profiler.end("run")
    logger.info(f"Total demo duration: {(profiler.get_duration('run')):.2f} s")

    # Save benchmark data
    benchmark_data = create_benchmark_data(profiler, measurements, N_warmup_iter, csv_perf_targets)
    benchmark_data.prep_csvs(
        profiler,
        run_type=f"demo_perf_{num_devices}chip" if perf_mode else f"demo_generate_{num_devices}chip",
        ml_model_name=model_version,
        ml_model_type="llm",
        num_layers=num_layers,
        batch_size=batch_size,
        config_params=configuration.to_dict(),
        precision=f"prefill[{model_config_strs_prefill_decode[0]}]_decode[{model_config_strs_prefill_decode[1]}]",
        input_sequence_length=num_input_tokens,
        output_sequence_length=1 if perf_mode else output_token_index + 1,
    )

    # Verify output or perf if expected values are provided
    assert expected_perf_metrics is None or expected_greedy_output_path is None
    if expected_perf_metrics is not None:
        verify_perf(measurements, expected_perf_metrics)
    elif expected_greedy_output_path is not None:
        if token_check_does_pass:
            logger.info("Output Check Passed!")
        else:
            assert (
                token_check_does_pass
            ), f"Generated text does not match expected output! \n\n Generated text:\n {generated_text} \n\n Expected output:\n {expected_output}"

    return generated_text, measurements
