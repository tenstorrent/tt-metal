# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
from time import time
from datetime import datetime
from loguru import logger
import os
import ttnn
import pytest
from models.demos.wormhole.llama31_8b.tt.llama_common import (
    prepare_inputs_ttnn,
    sample,
    get_single_rot_mat,
    cache_attention,
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
    encode_prompt_llama_instruct,
    HostEmbedding,
)
from models.demos.wormhole.llama31_8b.tt.llama_model import TtTransformer
from models.demos.wormhole.llama31_8b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["prompt"])
    return in_prompt


def preprocess_inputs_prefill(input_prompts, tokenizer, model_args, dtype, embd, instruct, device):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    assert (
        max_prompt_len <= model_args.max_seq_len
    ), f"Max prompt length {max_prompt_len} exceeds model max seq len {model_args.max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len, f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

    if min_prompt_len < 128:
        prefill_seq_len = 0  # For short prompts do decode-as-prefill instead
    else:
        prefill_seq_len = (
            1024 if min_prompt_len > 1024 else (512 if min_prompt_len > 512 else 128)
        )  # TODO Only supports prefill lengths of 128, 512, 1024
        # Initial prefill tensor full of pad tokens
        input_tokens_prefill = torch.full((len(input_prompts), prefill_seq_len), tokenizer.pad_id, dtype=torch.int32)

    # Initial decode tensor full of pad tokens
    input_tokens_decode = torch.full(
        (len(input_prompts), max_prompt_len - prefill_seq_len), tokenizer.pad_id, dtype=torch.long
    )

    for i, encoded in enumerate(encoded_prompts):
        if prefill_seq_len > 0:
            input_tokens_prefill[i] = torch.tensor(encoded[:prefill_seq_len]).to(input_tokens_prefill)
        input_tokens_decode[i, : len(encoded[prefill_seq_len:])] = torch.tensor(encoded[prefill_seq_len:]).to(
            input_tokens_decode
        )

    input_mask = (input_tokens_decode != tokenizer.pad_id).to(torch.bool)

    num_users = len(encoded_prompts)
    logger.info(f"# of users: {num_users}")

    # Select the first token from the prompts for initial decoding
    pt_tokenized_inputs_decode = torch.tensor(input_tokens_decode)
    pt_tokenized_inputs_prefill = torch.tensor(input_tokens_prefill)
    emb_inputs_decode = embd(pt_tokenized_inputs_decode[:, 0]).view(model_args.max_batch_size, 1, -1)
    if prefill_seq_len > 0:
        emb_prefill_inputs = [
            embd(pt_tokenized_inputs_prefill[b, :]).view(1, prefill_seq_len, -1)
            for b in range(model_args.max_batch_size)
        ]
    else:
        emb_prefill_inputs = None

    return (
        emb_inputs_decode,
        pt_tokenized_inputs_decode,
        emb_prefill_inputs,
        input_mask,
        None,
        prefill_seq_len,
        encoded_prompts,
    )


def run_llama_demo(user_input, batch_size, device, instruct_mode, is_ci_env, num_batches, print_to_file, is_n300):
    # Creat batch output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/wormhole/llama31_8b/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/demo_user_output_{timestamp}.txt"

    # Set Llama flags for CI
    if is_ci_env and instruct_mode:  # Update paths for instruct mode, otherwise use default paths for general weights
        os.environ["LLAMA_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/"
        os.environ["LLAMA_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/"
        os.environ["LLAMA_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/"
    # This module requires the env paths above for CI runs
    from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs

    embed_on_device = False
    dtype = ttnn.bfloat8_b

    # We disregard any warmup iteration for profiling, in favour of just measuring compile time on the first iteration
    N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if len(user_input) == 1:
        input_prompts = user_input * batch_size
    else:
        input_prompts = load_inputs(user_input, batch_size)
    profiler.end("loading_inputs")

    # Generate the batched prompts (rotate the inputs between the users, for each batch)
    # If batch_size == 1, the same prompt is repeated for each batch
    batch_prompts = []
    for i in range(num_batches):
        batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(device, instruct=instruct_mode)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    model_args.n_layers = 32

    logger.info("Loading weights...")
    profiler.start("weight_loading")
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }
    profiler.end("weight_loading")
    logger.info("Loading weights finished!")

    # TODO Should we keep initial embedding on host?
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    profiler.start("preprocess_prefill_inputs")
    (
        _,
        _,
        _,
        _,
        rot_emb_matrix_list,
        prefill_seq_len,
        _,
    ) = preprocess_inputs_prefill(input_prompts, tokenizer, model_args, dtype, embd, instruct_mode, device)
    profiler.end("preprocess_prefill_inputs")

    generation_start_pos = prefill_seq_len
    max_generated_tokens = 120
    users_decoding = True

    # pre-compute the rotational embedding matrix and send to device
    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        device,
        start_pos=0,
    )
    logger.info(f"caching attention for {prefill_seq_len} prefill tokens + {max_generated_tokens} generated tokens")
    profiler.start("cache_attention")
    cache_attention(device, state_dict, model_args, current_rot_mat, dtype, prefill_seq_len + max_generated_tokens)
    profiler.end("cache_attention")

    # Load TTNN Llama3.1 model
    logger.info("Loading weights to device...")
    profiler.start("loading_weights_to_device")
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layers=list(range(model_args.n_layers)),
        rot_mat=rot_emb_matrix_list,
        start_pos=generation_start_pos,
    )
    tt_embd = TtLlamaEmbedding(
        device=device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    profiler.end("loading_weights_to_device")
    logger.info("Finished loading weights to device. Starting inference...")

    num_tokens_generated_decode = []
    for batch_idx, input_prompts in enumerate(batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        # Preprocess initial prompt inputs
        (
            pt_encoded_input,
            tt_decode_input,
            pt_prefill_input,
            input_mask,
            rot_emb_matrix_list,
            prefill_seq_len,
            encoded_prompts,
        ) = preprocess_inputs_prefill(input_prompts, tokenizer, model_args, dtype, embd, instruct_mode, device)
        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)
        generation_start_pos = prefill_seq_len

        # set kv cache to zeros if not first batch, to avoid context leaking
        if batch_idx != 0:
            for layer in tt_model.layers:
                k_cache, v_cache = layer.attention.layer_past_list[0]
                k_cache = k_cache * 0
                v_cache = v_cache * 0
                # Deallocation is necessary to avoid memory leaks and running out of L1 in later batches
                layer.attention.layer_past_list[0][0].deallocate(True)
                layer.attention.layer_past_list[0][1].deallocate(True)
                layer.attention.layer_past_list[0] = [k_cache, v_cache]

        if prefill_seq_len > 0:
            logger.info(f"Starting prefill [{prefill_seq_len} tokens]...")
            profiler.start(f"prepare_rot_mat_for_prefill", iteration=batch_idx)
            rot_mats_prefill = get_prefill_rot_mat(
                model_args.head_dim, model_args.max_seq_len, device, seq_len=prefill_seq_len
            )
            head_dim = model_args.dim // model_args.n_heads
            transformation_mat_torch = get_rot_transformation_mat(head_dim)
            transformation_mats = ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            profiler.end(f"prepare_rot_mat_for_prefill", iteration=batch_idx)

            # First user is used for compile time
            num_users_generated_prefill = batch_size - 1 if batch_size > 1 else 1  # First user is used for compile time

            profiler.start(f"inference_prefill", iteration=batch_idx)
            for batch_id in range(batch_size):
                if batch_id == 0:  # First user prefill also accounts for compile time
                    profiler.start(f"compile_prefill", iteration=batch_idx)
                prefill_input, attn_mask, _ = prepare_inputs_ttnn_prefill(
                    pt_prefill_input[batch_id],
                    device,
                )
                tt_out = tt_model(
                    prefill_input,
                    0,  # Current position
                    attn_mask,
                    rot_mats_prefill,
                    transformation_mats,
                    user_id=batch_id,
                    mode="prefill",
                )
                if batch_id == 0:  # First user prefill also accounts for compile time
                    profiler.end(f"compile_prefill", iteration=batch_idx)

            # Do another prefill run if batch_size == 1, to correctly measure inference prefill time
            if batch_size == 1:
                for batch_id in range(batch_size):
                    prefill_input, attn_mask, _ = prepare_inputs_ttnn_prefill(
                        pt_prefill_input[batch_id],
                        device,
                    )
                    tt_out = tt_model(
                        prefill_input,
                        0,  # Current position
                        attn_mask,
                        rot_mats_prefill,
                        transformation_mats,
                        user_id=batch_id,
                        mode="prefill",
                    )
            # Device synchrozization ensures profiler is accurate in end-to-end timing
            ttnn.synchronize_device(device)
            profiler.end(f"inference_prefill", iteration=batch_idx)
            logger.info(f"Prefill finished [{prefill_seq_len} tokens]!")

        logger.info("Starting decode...")

        profiler.start(f"get_single_rot_mat_decode_{batch_idx}")
        current_rot_mat, rot_matrix = get_single_rot_mat(
            model_args.head_dim,
            device,
            start_pos=prefill_seq_len,
        )
        profiler.end(f"get_single_rot_mat_decode_{batch_idx}")

        # Keep track of generated outputs to print out every iteration
        all_outputs = [encoded_prompts[b][:prefill_seq_len] for b in range(batch_size)]
        user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

        iteration = 0
        users_decoding = True  # reset to handle next batch

        profiler.start(f"inference_decode", iteration=batch_idx)

        # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
        while users_decoding:
            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)

            iteration_time_start = time()
            curr_pos = generation_start_pos + iteration

            # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
            # TODO Move the attn mask to device
            profiler.start(f"prepare_input_decode", iteration=batch_idx)
            decode_input, current_pos = prepare_inputs_ttnn(
                pt_encoded_input,
                curr_pos,
                model_args.dim,
                model_args.sliding_window,
                tt_model.device,
            )
            profiler.end(f"prepare_input_decode", iteration=batch_idx)

            profiler.start(f"decode_and_argmax", iteration=batch_idx)
            # Run ttnn llama3.1 model
            tt_out = tt_model(decode_input, current_pos, rot_mat=current_rot_mat)
            tt_out = ttnn.untilize(
                tt_out, use_multicore=False
            )  # multi-core OOMs (https://github.com/tenstorrent/tt-metal/issues/9022)
            tt_output_torch = (
                ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)[:batch_size, :, :]
            )  # [batch, seq, hidden_dim]
            # Update rotation matrix for next iteration
            current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)
            # If temperature is 0, does greedy decoding (top-1)
            tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8)

            # TODO argmax on device
            # tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
            # tt_out = ttnn.permute(tt_out, (2, 1, 0, 3))
            # tt_out = ttnn.reshape(tt_out, (tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]))  # Squeeze(1)
            # tt_out_argmax = ttnn.argmax(tt_out, dim=-1)
            # Typecast from bf16 to uint32 for embedding
            # tt_out_tok = ttnn.clone(tt_out_argmax, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.uint32)
            # tt_out_tok = ttnn.experimental.tensor.typecast(tt_out_tok, dtype=ttnn.uint32)

            if iteration < input_mask.shape[1]:  # If prefill
                # If token is pad token, start generating new token, otherwise, push the next prompt token to the model
                tt_out_tok = torch.where(
                    input_mask[:, iteration], tt_decode_input[:, iteration], tt_out_tok[:, 0]
                ).unsqueeze(1)

            profiler.end(f"decode_and_argmax", iteration=batch_idx)

            # Save output token to print out later
            for user in range(batch_size):
                user_tok = tt_out_tok[user].tolist()
                if (
                    user_tok[0] != 28803 and user_done[user] == False
                ):  # Stop saving the ouput after hitting the EOS token
                    all_outputs[user].append(user_tok[0])
                else:
                    user_done[user] = True
                    if (
                        iteration < input_mask.shape[1]
                    ):  # Still in prefill, so ignore EOS token and save the generated token
                        # all_outputs[user].append(user_tok[0])
                        pass
                    else:
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False

            profiler.start(f"decode_embedding", iteration=batch_idx)
            if embed_on_device:
                tt_out_tok = ttnn.from_torch(tt_out_tok, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                pt_encoded_input = tt_embd(tt_out_tok)
            else:
                pt_encoded_input = embd(tt_out_tok)
            profiler.end(f"decode_embedding", iteration=batch_idx)

            # Print out generated outputs for each user at the end of every iteration
            iteration_time = time() - iteration_time_start
            tokens_per_second_per_user = 1 / iteration_time

            profiler.start(f"log_printing_iter_{iteration}", iteration=batch_idx)
            # Print out generated outputs for each user at the end of every iteration
            if not is_ci_env:
                if len(user_input) == 1:
                    logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
                else:
                    for user in range(batch_size):
                        text = "".join(tokenizer.decode(all_outputs[user]))
                        if len(text) > 100:
                            text = "..." + text[-97:]
                        text = text.replace("\n", " ")
                        logger.info("[User {}] {}".format(user, text))
            # Always print perf at every iteration
            logger.info(
                f"Iteration {iteration}: {1000*iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )
            profiler.end(f"log_printing_iter_{iteration}", iteration=batch_idx)

            if iteration == 0:  # First iteration also accounts for compile time
                profiler.end(f"compile_decode", iteration=batch_idx)

            iteration += 1

            # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
            if iteration >= max_generated_tokens:
                users_decoding = False

            if not users_decoding:
                profiler.start(f"log_saving_file", iteration=batch_idx)
                with open(output_filename, "a") as f:
                    for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                        text = tokenizer.decode(output)
                        if instruct_mode:
                            split_text = text.split("[/INST]", 1)
                        else:
                            split_text = text.split(prompt, 1)
                        if len(split_text) > 1:
                            text_after_prompt = split_text[1]
                        else:
                            text_after_prompt = text  # If prompt is not found, use the whole text
                        if print_to_file:
                            f.write(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n"
                            )
                        else:
                            logger.info(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n"
                            )
                profiler.end(f"log_saving_file", iteration=batch_idx)

        num_tokens_generated_decode.append(
            iteration - 1
        )  # Save the number of tokens generated for each batch (excluding the first token which is used for compile time)

        profiler.end(f"inference_decode", iteration=batch_idx)

        # When running in CI, check the output against the expected output to avoid accuracy regressions
        # TODO Extend the expected output validation to further batches
        if is_ci_env and batch_idx == 0:  # Only check output of batch 0
            expected_output = "models/demos/wormhole/llama31_8b/demo/expected_outputs_prefill_128.json"
            with open(expected_output, "r") as f:
                expected_out = json.load(f)
            # assert (
            #     len(expected_out) >= batch_size * 2
            # ), f"expected_outputs.json should have {batch_size * 2} outputs: {batch_size} for general weights and {batch_size} for instruct weights!"

            for i in range(batch_size):
                user_output = "".join(tokenizer.decode(all_outputs[i]))
                if instruct_mode:  # The instruct outputs are at the end of the expected outputs file
                    user_expect = expected_out[i + batch_size]["output_instruct"]
                else:
                    user_expect = expected_out[i]["output_general"]

                assert user_output == user_expect, f"Output for user {i} does not match expected output!"
            logger.info("[CI-Only] Output token validation passed!")

    # Finish profiling at the end of all batches
    profiler.end("run")

    # Benchmark metrics for batch 0
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")
    inference_prefill_time = profiler.get_duration("inference_prefill")
    inference_decode_time = profiler.get_duration("inference_decode")
    log_printing_time = sum(profiler.get_duration(f"log_printing_iter_{i}") for i in range(max_generated_tokens))
    log_saving_file_time = profiler.get_duration(f"log_saving_file")

    # Correct the inference decode time to remove the time spent on compile (1st iteration) and log_printing (at the end of every iteration)
    inference_decode_time = inference_decode_time - compile_decode_time - log_printing_time - log_saving_file_time
    # Correct the inference prefill time to remove the time spent on compile (1st iteration)
    inference_prefill_time = inference_prefill_time - compile_prefill_time

    # FIXME: Currently our prefill pass does not generate the first token, so we correct the time_to_first to include 1 prefill step + 1 decode step
    prefill_time_to_first = (inference_prefill_time / num_users_generated_prefill) + (
        inference_decode_time / num_tokens_generated_decode[0]
    )

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": inference_prefill_time,
        "inference_decode": inference_decode_time,
        "prefill_time_to_token": prefill_time_to_first,
        "prefill_t/s": num_users_generated_prefill / inference_prefill_time * prefill_seq_len,  # tokens/s
        "decode_t/s/u": num_tokens_generated_decode[0] / inference_decode_time,  # tokens/s
        "decode_t/s": num_tokens_generated_decode[0] / inference_decode_time * batch_size,  # tokens/s/user
        # Optional measurements
        "loading_inputs": profiler.get_duration("loading_inputs"),
        "weight_loading": profiler.get_duration("weight_loading"),
        "preprocess_prefill_inputs": profiler.get_duration("preprocess_prefill_inputs"),
        "loading_weights_to_device": profiler.get_duration("loading_weights_to_device"),
        "cache_attention": profiler.get_duration("cache_attention"),
        "prepare_rot_mat_for_prefill": profiler.get_duration("prepare_rot_mat_for_prefill"),
        "prepare_input_decode": profiler.get_duration("prepare_input_decode"),
        "decode_and_argmax": profiler.get_duration("decode_and_argmax"),
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # Print some of the perf metrics as well
    logger.info("---")
    logger.info(f"Performance metrics for batch 0")
    logger.info(f"Prefill compile time: {round(measurements['compile_prefill'], 4)}s")
    logger.info(f"Decode compile time: {round(measurements['compile_decode'], 4)}s")
    logger.info(f"Prefill inference time per user: {round(inference_prefill_time/num_users_generated_prefill, 4)}s")
    logger.info(
        f"Total Decode inference time ({max_generated_tokens-1} iterations): {round(measurements['inference_decode'], 4)}s"
    )
    logger.info(
        f"Average Decode inference time per user: {round(inference_decode_time / num_tokens_generated_decode[0], 4)}s"
    )
    logger.info("---")
    logger.info(f"Time to first token: {round(measurements['prefill_time_to_token']* 1000, 4)}ms")
    logger.info(f"Average tokens/sec/user: {round(measurements['decode_t/s/u'], 2)}")

    target_prefill_ts = 5000  # TODO update target
    target_decode_ts = 1056
    decode_tsu = 33
    targets = {"prefill_t/s": target_prefill_ts, "decode_t/s": target_decode_ts, "decode_t/s/u": decode_tsu}

    # TODO move token verification here?
    # if expected_greedy_output_path is not None:
    #     token_check_does_pass, expected_output = check_tokens_match(generated_text, expected_greedy_output_path)
    #     measurements["token_verification"] = float(token_check_does_pass)

    # Save benchmark data for CI dashboard
    if is_ci_env and is_n300:
        benchmark_data = create_benchmark_data(profiler, measurements, N_warmup_iter, targets)
        benchmark_data.prep_csvs(
            profiler,
            run_type=f"demo_with_prefill",
            ml_model_name="Llama3.1-8B",
            ml_model_type="llm",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            input_sequence_length=prefill_seq_len,
            output_sequence_length=1,
            # config_params=,
            # precision=,
        )


@pytest.mark.parametrize(
    "input_prompts, instruct_weights, num_batches",
    [
        ("models/demos/wormhole/llama31_8b/demo/input_data_prefill_128.json", False, 1),
        ("models/demos/wormhole/llama31_8b/demo/input_data_prefill_128.json", False, 3),
        ("models/demos/wormhole/llama31_8b/demo/input_data_questions_prefill_128.json", True, 1),
        ("models/demos/wormhole/llama31_8b/demo/input_data_questions_prefill_128.json", True, 3),
    ],
    ids=[
        "general_weights-1_batch",
        "general_weights-3_batch",
        "instruct_weights-1_batch",
        "instruct_weights-3_batch",
    ],
)
def test_llama_demo(
    device, use_program_cache, input_prompts, instruct_weights, is_ci_env, is_single_card_n300, num_batches
):
    if is_ci_env and instruct_weights == False:
        pytest.skip("CI demo test only runs instruct weights to reduce CI pipeline load (both are supported)")

    return run_llama_demo(
        user_input=input_prompts,
        batch_size=1,
        device=device,
        instruct_mode=instruct_weights,
        is_ci_env=is_ci_env,
        num_batches=num_batches,
        print_to_file=False,
        is_n300=is_single_card_n300,
    )
