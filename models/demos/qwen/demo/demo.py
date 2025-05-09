# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from time import time

import pytest
import requests
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen.tt.qwen_common import HostEmbedding, get_rot_transformation_mat, get_single_rot_mat
from models.demos.qwen.tt.qwen_embedding import TtQwenEmbedding
from models.demos.qwen.tt.qwen_model import TtTransformer
from models.perf.benchmarking_utils import BenchmarkProfiler


def load_and_cache_context(context_url, cache_dir):
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()

    if cache_file.exists():
        with open(cache_file, "r") as f:
            context_text = f.read()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        try:
            response = requests.get(context_url)
            if response.status_code == 200:
                context_text = response.text
                with open(cache_file, "w") as f:
                    f.write(context_text)
                logger.info(f"Downloaded and cached context: {context_url}")
            else:
                logger.warning(f"Failed to fetch context from URL: {context_url}. Status code: {response.status_code}")
                context_text = ""
        except Exception as e:
            logger.error(f"Error fetching context from URL: {context_url}. Error: {str(e)}")
            context_text = ""

    return context_text


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    cache_dir = Path("models/demos/qwen/demo/context_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i in range(batch):
        prompt = user_input[i]["prompt"]
        if "context" in user_input[i]:
            context_text = load_and_cache_context(user_input[i]["context"], cache_dir)
            prompt = context_text + "\n\n" + prompt
        in_prompt.append(prompt)
    return in_prompt


def preprocess_inputs_prefill(
    input_prompts,
    tokenizer,
    model_args,
    instruct,
    max_generated_tokens,
    max_prefill_len=128 * 1024,
):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    # The maximum KV-cache len supported is 32k. To avoid going out of memory, clip the max prefill length by the maximum number of tokens that will be generated
    if max_prefill_len == 128 * 1024:
        max_prefill_len = 128 * 1024 - max_generated_tokens

    if instruct:
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            for prompt in input_prompts
        ]
        input_prompts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    # Print the length of encoded prompts
    logger.info("Encoded prompt lengths:" + ", ".join(str(len(prompt)) for prompt in encoded_prompts))

    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    # The large input demo we provide contains more tokens than the maximum (32k tokens)
    # To avoid running out of memory, clip to max_prefill_len
    if min_prompt_len > max_prefill_len:
        logger.info(f"Clipping prompts to {max_prefill_len}")
        if instruct:  # When clipping, make sure to add the ` 】 token at the end (4 tokens)
            encoded_prompts = [encod[: max_prefill_len - 4] for encod in encoded_prompts]
            dec_prompts = [tokenizer.decode(encod) + " 】" for encod in encoded_prompts]
            encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in dec_prompts]
        else:
            encoded_prompts = [encod[:max_prefill_len] for encod in encoded_prompts]

        # Update prompt lengths
        prompt_lens = [len(x) for x in encoded_prompts]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)

    assert (
        max_prompt_len <= model_args.max_seq_len
    ), f"Max prompt length {max_prompt_len} exceeds model max seq len {model_args.max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len, f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

    logger.info(f"# of users: {len(encoded_prompts)}")
    input_tokens_prefill = []
    decoding_pos = []
    prefill_lens = []

    # Always prefill the nearest power of 2 for each user. This means that the majority of cases we will prefill more tokens than needed.
    # To avoid issues, we keep track of the decoding position to decode correctly the user's prompt
    for i, encoded in enumerate(encoded_prompts):
        # Prefill size is nearest power of 2
        prefill_seq_len = max(2 ** math.ceil(math.log(len(encoded), 2)), 128)

        # Initial prefill tensors full of pad tokens
        input_tokens_prefill_i = torch.full((1, prefill_seq_len), 0, dtype=torch.int32)
        input_tokens_prefill_i[0, : len(encoded[:])] = torch.tensor(encoded[:]).to(input_tokens_prefill_i)
        input_tokens_prefill.append(input_tokens_prefill_i)

        # Keep the correct decoding position of each user
        decoding_pos.append(len(encoded))
        prefill_lens.append(prefill_seq_len)

    return (
        input_tokens_prefill,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    )


def run_qwen_demo(
    user_input, batch_size, single_layer, mesh_device, instruct_mode, is_ci_env, num_batches, print_to_file
):
    # Creat batch output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/qwen/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/demo_user_output_{timestamp}.txt"

    # This module requires the env paths above for CI runs
    from models.demos.qwen.tt.model_config import TtModelArgs

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
    model_args = TtModelArgs(mesh_device, instruct=instruct_mode)
    # tokenizer = Tokenizer(model_args.tokenizer_path)

    if instruct_mode:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")  # Tokenizer(model_args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    if single_layer:
        model_args.n_layers = 1

    logger.info("Loading weights...")
    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    # Load TTNN Qwen3.1 model
    logger.info("Loading weights to device...")
    profiler.start("loading_weights_to_device")
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )
    tt_embd = TtQwenEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    embd = HostEmbedding(model_args)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}embed_tokens.weight"]})
    profiler.end("loading_weights_to_device")
    logger.info("Finished loading weights to device.")

    max_generated_tokens = 100  # Maximum number of tokens to generate per user
    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    for batch_idx, input_prompts in enumerate(batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        # Preprocess initial prompt inputs
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts,
            tokenizer,
            model_args,
            instruct_mode,
            max_generated_tokens,
        )
        # Prefill embeddings are on host since we need to mask out the tokens after the prefill length after embeddings are computed
        pt_prefill_input = [embd(input_tokens_prefill_pt[b]).view(1, prefill_lens[b], -1) for b in range(batch_size)]
        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

        # set kv cache to zeros if not first batch, to avoid context leaking when doing multiple batches
        if batch_idx != 0:
            for layer in tt_model.layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        logger.info(f"Starting prefill...")

        profiler.start(f"prepare_rot_mat_for_prefill", iteration=batch_idx)
        transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
        transformation_mats = ttnn.from_torch(
            transformation_mat_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        profiler.end(f"prepare_rot_mat_for_prefill", iteration=batch_idx)

        # Do not count the first user for prefill time and instead log it as compile time
        num_users_generated_prefill = batch_size - 1 if batch_size > 1 else 1

        pt_out = []

        profiler.start(f"inference_prefill", iteration=batch_idx)
        for batch_id in range(batch_size):
            prefill_seq_len = prefill_lens[batch_id]

            # TODO: uncomment this when prefill is fixed
            # # rot_mats_prefill = get_prefill_rot_mat(
            # #     model_args.head_dim, model_args.max_seq_len, mesh_device, seq_len=prefill_seq_len
            # # )

            # #FIXME: Remove this workaroun when device RoPE is fixed
            # freqs = precompute_freqs(model_args.head_dim, prefill_seq_len)
            # cos, sin = freqs[0], freqs[1]
            # freqs_cis = torch.complex(cos, sin)
            # rot_mats_prefill = freqs_cis[:prefill_seq_len, :]

            # if decoding_pos[batch_id] < prefill_seq_len:
            #     pt_prefill_input[batch_id][
            #         :, decoding_pos[batch_id] :, :
            #     ] = 0  # Zero out the tokens after the prefill length

            # prefill_input = model_args.prepare_inputs_ttnn_prefill(
            #     pt_prefill_input[batch_id],
            # )
            # if batch_id == 0:  # First user prefill accounts for compile time
            #     profiler.start(f"compile_prefill", iteration=batch_idx)

            # tt_out = tt_model(
            #     prefill_input,
            #     None,  # Current position
            #     rot_mats_prefill,
            #     transformation_mats,
            #     user_id=batch_id,
            #     mode="prefill",
            #     get_last_token=((decoding_pos[batch_id] - 1) // 32) * 32,
            # )

            # if (
            #     batch_id == 0
            # ):  # First user prefill accounts for compile time (which will be removed from the full prefill inference time)
            #     profiler.end(f"compile_prefill", iteration=batch_idx)

            # # [PROFILER-ONLY] In runs where there is only one user, run the prefill twice to measure compile and inference prefill times
            # if batch_size == 1:
            #     ttnn.deallocate(tt_out)
            #     tt_out = tt_model(
            #         prefill_input,
            #         None,  # Current position
            #         rot_mats_prefill,
            #         transformation_mats,
            #         user_id=batch_id,
            #         mode="prefill",
            #         get_last_token=((decoding_pos[batch_id] - 1) // 32) * 32,
            #     )

            # pt_out.append(
            #     ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[
            #         0, 0, (decoding_pos[batch_id] - 1) % 32, :
            #     ]
            # )
            # ttnn.deallocate(tt_out)

            for i in range(decoding_pos[0]):
                start_pos = i
                pt_decode_input = embd(input_tokens_prefill_pt[batch_id][:, i]).view(model_args.max_batch_size, 1, -1)
                current_rot_mat, rot_matrix = get_single_rot_mat(
                    model_args.head_dim,
                    mesh_device,
                    model_args.num_devices,
                    start_pos=start_pos,
                )
                pt_decode_input = model_args.prepare_inputs_ttnn_decode(
                    pt_decode_input,
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                current_pos_tensor = ttnn.from_torch(
                    torch.tensor([start_pos]),
                    device=mesh_device,
                    dtype=ttnn.int32,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                tt_out = tt_model(pt_decode_input, current_pos_tensor, rot_mat=current_rot_mat)

            pt_out.append(ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[0, 0, 0, :])
            ttnn.deallocate(tt_out)
        # Synchronize devices to ensure the profile captures the correct timing of all devices
        ttnn.synchronize_device(mesh_device)
        profiler.end(f"inference_prefill", iteration=batch_idx)
        logger.info(f"Prefill finished")

        # Preparing first decode token
        profiler.start(f"prepare_first_decode_token_{batch_idx}")
        pt_out_batched = torch.stack(pt_out, dim=-2)
        pt_out_batched = torch.argmax(pt_out_batched, dim=-1)
        tt_out_tok = ttnn.from_torch(
            torch.nn.functional.pad(pt_out_batched.unsqueeze(0).unsqueeze(0).unsqueeze(0), (0, 31), "constant", 0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint32,
        )
        profiler.end(f"prepare_first_decode_token_{batch_idx}")

        # Keep track of generated outputs to print out every iteration
        all_outputs = [encoded_prompts[b][:prefill_seq_len] for b in range(batch_size)]
        for user in range(batch_size):
            user_tok = int(pt_out_batched[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

        logger.info("Starting decode...")

        profiler.start(f"get_single_rot_mat_decode_{batch_idx}")
        current_rot_mat, rot_matrix = get_single_rot_mat(
            model_args.head_dim,
            mesh_device,
            model_args.num_devices,
            start_pos=decoding_pos[0] - 2,
        )
        profiler.end(f"get_single_rot_mat_decode_{batch_idx}")

        # Create events
        profiler.start(f"compile_trace_{batch_idx}")

        current_pos = ttnn.from_torch(
            torch.tensor(decoding_pos, dtype=torch.int32),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.int32,
        )

        # Compile
        logger.info(f"Compiling model trace...")
        decode_input = ttnn.unsqueeze_to_4D(tt_embd(tt_out_tok))
        tt_out = tt_model(decode_input, current_pos, rot_mat=current_rot_mat)
        if tt_model.args.num_devices > 1:
            tt_out_gathered = ttnn.all_gather(tt_out, dim=3, num_links=1, topology=ttnn.Topology.Linear)
            ttnn.deallocate(tt_out)
        else:
            tt_out_gathered = tt_out
        tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True)
        ttnn.deallocate(tt_out_gathered)
        tt_out_tok = ttnn.argmax(tt_out_rm, dim=3, keepdim=True, use_multicore=True, output_tensor=tt_out_tok)
        ttnn.deallocate(tt_out_rm)
        new_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)
        current_rot_mat = ttnn.copy(new_rot_mat, current_rot_mat)
        ttnn.plus_one(current_pos)
        profiler.end(f"compile_trace_{batch_idx}")

        # Capture Trace
        logger.info(f"Capturing model trace...")
        profiler.start(f"capture_trace_{batch_idx}")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

        decode_input = ttnn.unsqueeze_to_4D(tt_embd(tt_out_tok))
        tt_out = tt_model(decode_input, current_pos, rot_mat=current_rot_mat)
        if tt_model.args.num_devices > 1:
            tt_out_gathered = ttnn.all_gather(tt_out, dim=3, num_links=1, topology=ttnn.Topology.Linear)
            ttnn.deallocate(tt_out)
        else:
            tt_out_gathered = tt_out
        tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True)
        ttnn.deallocate(tt_out_gathered)
        tt_out_tok = ttnn.argmax(tt_out_rm, dim=3, keepdim=True, use_multicore=True, output_tensor=tt_out_tok)
        ttnn.deallocate(tt_out_rm)
        new_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)
        current_rot_mat = ttnn.copy(new_rot_mat, current_rot_mat)
        ttnn.plus_one(current_pos)

        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        # Reset the decoding position for the proper run of the model
        current_pos_reset = ttnn.from_torch(
            torch.tensor(decoding_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if tt_model.args.num_devices > 1 else None,
        )
        tt_out_tok_reset = ttnn.from_torch(
            torch.nn.functional.pad(pt_out_batched.unsqueeze(0).unsqueeze(0).unsqueeze(0), (0, 31), "constant", 0),
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if tt_model.args.num_devices > 1 else None,
        )

        ttnn.copy_host_to_device_tensor(current_pos_reset, current_pos)
        ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)

        profiler.end(f"capture_trace_{batch_idx}")

        # Start decoding
        iteration = 0
        users_decoding = True  # reset to handle next batch
        total_decoding_time = 0  # Track total decoding time
        total_tokens_generated = 0  # Track total tokens generated

        logger.info(f"Starting decode loop...")
        profiler.start(f"inference_decode", iteration=batch_idx)

        write_event = ttnn.record_event(mesh_device, 1)
        while users_decoding:
            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)
            iteration_time_start = time()

            # Execute trace
            ttnn.wait_for_event(0, write_event)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            op_event = ttnn.record_event(mesh_device, 0)

            # Write to host
            ttnn.wait_for_event(1, op_event)
            tt_output_torch = ttnn.to_torch(
                tt_out_tok.cpu(blocking=True, cq_id=1), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)
            )[0, 0, 0, :batch_size]
            write_event = ttnn.record_event(mesh_device, 1)

            # Save output token to print out later
            for user in range(batch_size):
                user_tok = tt_output_torch[user].tolist()
                if (
                    user_tok != tokenizer.eos_token_id and user_done[user] == False
                ):  # Stop saving the ouput after hitting the EOS token
                    all_outputs[user].append(user_tok)
                else:
                    user_done[user] = True
                    logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                    if all(user_done):
                        users_decoding = False

            # Print out generated outputs for each user at the end of every iteration
            iteration_time = time() - iteration_time_start

            # Ignore the first iteration for average speed calculation
            if iteration > 0:
                total_decoding_time += iteration_time
                total_tokens_generated += 1

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

            # Reset rotation matrix every 100 iterations
            profiler.start(f"reset_rot_mat_{iteration-1}", iteration=batch_idx)
            if iteration % 1 == 0:
                current_rot_mat_reset, rot_matrix_reset = get_single_rot_mat(
                    model_args.head_dim,
                    mesh_device,
                    model_args.num_devices,
                    start_pos=decoding_pos[0] + iteration,
                    on_host=True,
                )
                ttnn.copy_host_to_device_tensor(current_rot_mat_reset, current_rot_mat)
            profiler.end(f"reset_rot_mat_{iteration-1}", iteration=batch_idx)

            # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
            if iteration >= max_generated_tokens:
                users_decoding = False

            if not users_decoding:
                profiler.start(f"log_saving_file", iteration=batch_idx)
                for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                    text = tokenizer.decode(output)
                    if instruct_mode:
                        split_text = text.split("<|start_header_id|>assistant<|end_header_id|>", 1)
                    else:
                        split_text = text.split(prompt, 1)
                    if len(split_text) > 1:
                        text_after_prompt = split_text[1]
                    else:
                        text_after_prompt = text  # If prompt is not found, use the whole text
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
                            f"\nbatch: {batch_idx} user: {i}\nprompt: {short_prompt} \noutput:\n{text_after_prompt.strip()}\n"
                        )
                profiler.end(f"log_saving_file", iteration=batch_idx)

        num_tokens_generated_decode.append(
            total_tokens_generated
        )  # Save the number of tokens generated for each batch (excluding the first token)

        # Release trace
        ttnn.release_trace(mesh_device, trace_id)

        profiler.end(f"inference_decode", iteration=batch_idx)

    # Finish profiling at the end of all batches inference
    profiler.end("run")

    # Prepare profile benchmark metrics for batch 0
    compile_prefill_time = profiler.get_duration("compile_decode")  # FIXME: fix this when prefill works
    compile_decode_time = profiler.get_duration("compile_decode")
    inference_prefill_time = profiler.get_duration("inference_prefill")
    inference_decode_time = profiler.get_duration("inference_decode")
    log_printing_time = sum(profiler.get_duration(f"log_printing_iter_{i}") for i in range(total_tokens_generated))
    log_saving_file_time = profiler.get_duration(f"log_saving_file")

    # Correct the inference decode time to remove the time spent on compile (1st iteration) and log_printing (at the end of every iteration)
    inference_decode_time = inference_decode_time - compile_decode_time - log_printing_time - log_saving_file_time
    # Correct the inference prefill time to remove the time spent on compile (1st iteration)
    inference_prefill_time = inference_prefill_time - compile_prefill_time
    # Average prefill time for each user
    prefill_time_to_first = inference_prefill_time / num_users_generated_prefill

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": inference_prefill_time,
        "inference_decode": inference_decode_time,
        "prefill_time_to_token": prefill_time_to_first,
        "prefill_t/s": num_users_generated_prefill / inference_prefill_time * prefill_seq_len,  # tokens/s
        "decode_t/s/u": num_tokens_generated_decode[0] / inference_decode_time,  # tokens/s/u
        "decode_t/s": num_tokens_generated_decode[0] / inference_decode_time * batch_size,  # tokens/s
        # Optional measurements
        "loading_inputs": profiler.get_duration("loading_inputs"),
        "weight_loading": profiler.get_duration("weight_loading"),
        "prepare_first_decode_token": profiler.get_duration("prepare_first_decode_token_0"),
        "get_single_rot_mat_decode": profiler.get_duration("get_single_rot_mat_decode_0"),  # Only for batch 0
        "preprocess_prefill_inputs": profiler.get_duration("preprocess_prefill_inputs"),
        "loading_weights_to_device": profiler.get_duration("loading_weights_to_device"),
        "prepare_rot_mat_for_prefill": profiler.get_duration("prepare_rot_mat_for_prefill"),
        "compile_trace": profiler.get_duration("compile_trace_0"),  # Only for batch 0
        "capture_trace": profiler.get_duration("capture_trace_0"),  # Only for batch 0
        "reset_rot_mat": sum(profiler.get_duration(f"reset_rot_mat_{i}") for i in range(total_tokens_generated)),
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # Print some of the perf metrics
    logger.info("")
    logger.info(f"Performance metrics for batch 0")
    logger.info(f"Prefill compile time: {round(measurements['compile_prefill'], 4)}s")
    logger.info(f"Decode compile time: {round(measurements['compile_decode'], 4)}s")
    logger.info(f"Prefill inference time per user: {round(inference_prefill_time/num_users_generated_prefill, 4)}s")
    logger.info(
        f"Total Decode inference time ({max_generated_tokens-1} iterations): {round(measurements['inference_decode'], 4)}s"
    )
    logger.info("")
    logger.info(f"Time to first token: {round(measurements['prefill_time_to_token']* 1000, 2)}ms")
    logger.info(
        f"Average speed: {round(inference_decode_time / num_tokens_generated_decode[0] * 1000, 2)}ms @ {round(measurements['decode_t/s/u'], 2)} tok/s/user ({round(measurements['decode_t/s'], 2)} tok/s throughput)"
    )


@pytest.mark.parametrize(
    "input_prompts, instruct_weights, num_batches, single_layer",
    [
        ("models/demos/qwen/demo/input_data_prefill_128.json", False, 1, False),
        ("models/demos/qwen/demo/input_data_prefill_128.json", False, 2, False),
        ("models/demos/qwen/demo/input_data_questions_prefill_128.json", True, 1, False),
        ("models/demos/qwen/demo/input_data_questions_prefill_128.json", True, 2, False),
        ("models/demos/qwen/demo/input_data_questions_prefill_128.json", True, 1, True),
    ],
    ids=[
        "general_weights-1_batch",
        "general_weights-2_batch",
        "instruct_weights-1_batch",
        "instruct_weights-2_batch",
        "single_layer",
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 14951424, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_qwen_demo(
    mesh_device, use_program_cache, input_prompts, instruct_weights, is_ci_env, num_batches, single_layer, reset_seeds
):
    if is_ci_env and (instruct_weights == False or "long" in input_prompts or single_layer == True):
        pytest.skip("CI demo test only runs instruct weights to reduce CI pipeline load (both are supported)")

    return run_qwen_demo(
        user_input=input_prompts,
        batch_size=1,
        single_layer=single_layer,
        mesh_device=mesh_device,
        instruct_mode=instruct_weights,
        is_ci_env=is_ci_env,
        num_batches=num_batches,
        print_to_file=False,
    )
