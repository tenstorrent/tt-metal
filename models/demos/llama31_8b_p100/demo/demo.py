# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import models.ttt_compat.demo.simple_text_demo as tt_demo
import models.ttt_compat.tt.common as tt_common
import models.ttt_compat.tt.model as tt_model
from models.demos.llama31_8b_p100.tt.model import Transformer


def create_tt_model(*args, **kwargs):
    saved = tt_model.Transformer
    tt_model.Transformer = Transformer
    try:
        return tt_common.create_tt_model(*args, **kwargs)
    finally:
        tt_model.Transformer = saved


tt_demo.create_tt_model = create_tt_model

from models.demos.llama31_8b_p100.tt.readback import PipelinedDecode  # noqa: E402
from models.ttt_compat.demo.simple_text_demo import *  # noqa: E402,F401,F403

_tt_test_demo_text = tt_demo.test_demo_text


def test_demo_text(
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
    is_ci_v2_env,
    ci_only,
    data_parallel,
    reset_seeds,
    request,
    token_accuracy,
    stress_test,
    enable_trace,
    model_location_generator,
    num_layers,
    mode,
    use_prefetcher,
):
    hf_dir = os.getenv("HF_MODEL", "")
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    test_id = request.node.callspec.id
    if is_ci_env:
        if not ci_only:
            pytest.skip("CI only runs the CI-only tests")
        if "accuracy" in test_id and "ci-token-matching" not in test_id:
            pytest.skip("CI only runs the tests with performance optimizations except for ci-token-matching case")

    if os.environ.get("MESH_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    print_to_file = False

    input_prompts = request.config.getoption("--input_prompts") or input_prompts
    if request.config.getoption("--instruct") in [
        0,
        1,
    ]:
        instruct = request.config.getoption("--instruct")
    repeat_batches = request.config.getoption("--repeat_batches") or repeat_batches
    max_seq_len = request.config.getoption("--max_seq_len") or max_seq_len
    batch_size = request.config.getoption("--batch_size") or batch_size
    max_generated_tokens = request.config.getoption("--max_generated_tokens") or max_generated_tokens
    data_parallel = request.config.getoption("--data_parallel") or data_parallel
    cli_paged_attention = request.config.getoption("--paged_attention")
    if cli_paged_attention is not None:
        paged_attention = cli_paged_attention
    cli_page_params = request.config.getoption("--page_params")
    if cli_page_params:
        page_params = {**page_params, **cli_page_params}
    cli_sampling_params = request.config.getoption("--sampling_params")
    if cli_sampling_params:
        sampling_params = {**sampling_params, **cli_sampling_params}
    json_config_file = request.config.getoption("--decoder_config_file")
    cli_token_accuracy = request.config.getoption("--token_accuracy")
    if cli_token_accuracy is not None:
        token_accuracy = cli_token_accuracy
    cli_stress_test = request.config.getoption("--stress_test")
    if cli_stress_test is not None:
        stress_test = cli_stress_test
    arg_enable_trace = request.config.getoption("--enable_trace")
    if arg_enable_trace is not None:
        enable_trace = arg_enable_trace
    num_layers = request.config.getoption("--num_layers") or num_layers
    mode = request.config.getoption("--mode") or mode
    skip_perf_report = request.config.getoption("--skip_perf_report")
    use_prefetcher = request.config.getoption("--use_prefetcher") or use_prefetcher
    if use_prefetcher and not is_blackhole():
        logger.warning("--use_prefetcher requested but DRAM prefetcher is only supported on Blackhole; disabling.")
        use_prefetcher = False
    use_prefetcher = (
        use_prefetcher and is_prefetcher_supported(hf_dir, num_devices) and "Llama" in hf_dir and "8B" in hf_dir
    )
    global_batch_size = batch_size * data_parallel
    use_hf_rope = request.config.getoption("--use_hf_rope")
    is_device_perf_test = "device-perf" in test_id

    if stress_test and token_accuracy:
        pytest.skip("Stress test cannot be run with token accuracy mode")

    if json_config_file:
        optimizations = parse_decoder_json(json_config_file)
    else:
        optimizations = request.config.getoption("--optimizations") or optimizations

    if request.config.getoption("--stop_at_eos") in [
        0,
        1,
    ]:
        stop_at_eos = request.config.getoption("--stop_at_eos")

    if "phi-3-mini-128k-instruct" in hf_dir.lower():
        max_context_supported = 32 * 1024 * num_devices
        if (max_context_supported < max_seq_len) or (
            max_context_supported < page_params["page_block_size"] * page_params["page_max_num_blocks_per_dp"]
        ):
            pytest.skip(
                f"Max sequence length: {max_seq_len} for batch: {batch_size} not supported for model: {hf_dir} on device: {mesh_device}"
            )

    if data_parallel > num_devices or num_devices % data_parallel != 0:
        pytest.skip(f"Invalid number of DP groups: {data_parallel}, for {num_devices} devices")

    if is_ci_env:
        hf_model = os.getenv("HF_MODEL", "")
        is_33_70b = "3.3-70B" in hf_model
        is_32_1b = "3.2-1B" in hf_model
        is_31_8b = "3.1-8B" in hf_model

        tg_enabled = (data_parallel == 4 and is_33_70b) or (data_parallel in [4, 16, 32] and is_31_8b)

        if num_devices == 32 and not tg_enabled and "seqlen-sweep" not in test_id:
            pytest.skip("CI only runs Llama3 70b DP = 4, TP = 8 or Llama3 8b DP = 4/16/32, TP = 8/2/1 on TG")
        if num_devices == 8 and data_parallel > 1 and not (is_32_1b or is_31_8b) and is_wormhole_b0():
            pytest.skip("CI only runs hybrid Llama3 1b and 8b on T3K")

    if is_ci_v2_env:
        hf_model = os.getenv("HF_MODEL", "")
        model_location = model_location_generator(hf_model, download_if_ci_v2=True, ci_v2_timeout_in_s=900)
        os.environ["HF_MODEL"] = str(model_location)

    if not stop_at_eos:
        logger.info(f"The decode generation will only stop at the max_generated_tokens limit == {max_generated_tokens}")

    if print_to_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = "models/ttt_compat/demo/output"
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o755)
        output_filename = f"{output_directory}/llama_text_demo_output_{timestamp}.txt"

    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    is_seqlen_sweep = "seqlen-sweep" in test_id

    max_context_len = None if is_seqlen_sweep else hf_max_context_len()
    if max_context_len is not None and max_context_len < max_seq_len:
        pytest.skip(
            f"Max seq len {max_seq_len} not supported by model {os.getenv('HF_MODEL')}. The model's max context len is {max_context_len}"
        )

    if is_seqlen_sweep:
        block_size = page_params["page_block_size"]
        blocks_needed = -(-(max_seq_len + max_generated_tokens) // block_size)
        if blocks_needed < page_params["page_max_num_blocks_per_dp"]:
            block_size = 32
            blocks_needed = -(-(max_seq_len + max_generated_tokens) // block_size)
            page_params = {
                **page_params,
                "page_block_size": block_size,
                "page_max_num_blocks_per_dp": blocks_needed,
            }
            logger.info(
                f"Seqlen sweep: capped page params for small SKU — block_size={block_size}, "
                f"page_max_num_blocks_per_dp={blocks_needed} (max_seq_len={max_seq_len})"
            )

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    sweep_prompt_files = None
    if is_seqlen_sweep:
        sweep_prompt_files = input_prompts
        input_prompts, all_prompts = load_inputs(sweep_prompt_files[0], global_batch_size, instruct)
    elif len(input_prompts) == 1:
        input_prompts = input_prompts * global_batch_size
        all_prompts = input_prompts
    else:
        input_prompts, all_prompts = load_inputs(input_prompts, global_batch_size, instruct)
    profiler.end("loading_inputs")

    profiler.start("load_weights")
    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        local_data_parallel,
        local_submesh_indices,
    ) = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        instruct=instruct,
        global_batch_size=global_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        num_layers=num_layers,
        use_prefetcher=use_prefetcher,
        use_hf_rope=use_hf_rope,
    )
    profiler.end("load_weights")
    logger.info(f"Model build (load_weights) took {profiler.get_duration('load_weights'):.2f}s")

    global_batch_size = batch_size * local_data_parallel
    input_prompts = select_local_data_parallel_items(input_prompts, batch_size, data_parallel, local_submesh_indices)
    sampling_params = slice_sampling_params_for_local_submeshes(
        sampling_params, batch_size, data_parallel, local_submesh_indices
    )

    if ("ci-eval-1" in test_id or "ci-eval-32" in test_id) and model_args[0].device_name == "P100":
        pytest.skip("ci-eval-1 and ci-eval-32 tests are not supported on P100 devices")

    if token_accuracy:
        token_acc = TokenAccuracy(model_name=model_args[0].model_name)

    for m_args in model_args:
        if m_args.max_context_len < max_seq_len:
            if is_seqlen_sweep:
                max_seq_len = m_args.max_context_len
                logger.info(f"Seqlen sweep: capping max_seq_len to model's max_context_len={max_seq_len}")
            else:
                pytest.skip(
                    f"Max seq len {max_seq_len} not supported by model {m_args.model_name}. The model's max context len is {m_args.max_context_len}"
                )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    if token_accuracy:
        input_prompts[0] = token_acc.prepare_ref_tokens(tokenizer)

    repeat_batch_prompts = []
    if is_seqlen_sweep:
        filtered_files = []
        for f in sweep_prompt_files:
            label = Path(f).stem.split("_")[-1]
            if label.endswith("k") and label[:-1].isdigit():
                min_seqlen = int(label[:-1]) * 1024
                if min_seqlen <= max_seq_len:
                    filtered_files.append(f)
        if not filtered_files:
            pytest.skip(f"No sweep prompt files fit within model's max context length ({max_seq_len})")
        repeat_batches = len(filtered_files)
        logger.info(
            f"Seqlen sweep: running {repeat_batches} steps with files: {[Path(f).name for f in filtered_files]}"
        )
        for f in filtered_files:
            batch_prompts, _ = load_inputs(f, global_batch_size, instruct)
            repeat_batch_prompts.append(
                select_local_data_parallel_items(batch_prompts, batch_size, data_parallel, local_submesh_indices)
            )
    else:
        for i in range(repeat_batches):
            if token_accuracy:
                repeat_batch_prompts.append(input_prompts)
            else:
                global_prompts_for_batch = [all_prompts[(j + i) % len(all_prompts)] for j in range(len(all_prompts))][
                    : batch_size * data_parallel
                ]
                repeat_batch_prompts.append(
                    select_local_data_parallel_items(
                        global_prompts_for_batch, batch_size, data_parallel, local_submesh_indices
                    )
                )

    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    for batch_idx, input_prompts in enumerate(repeat_batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts, tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
        )

        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        assert (
            max_generated_tokens + max_encoded_prompt_len <= max_seq_len
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

        if paged_attention:
            paged_cache_max_seq_len = (
                page_params["page_block_size"] * page_params["page_max_num_blocks_per_dp"] / batch_size
            )
            assert (
                max_generated_tokens + max_encoded_prompt_len <= paged_cache_max_seq_len
            ), f"max_generated_tokens ({max_generated_tokens}) needs to be <= than paged_cache_max_seq_len ({paged_cache_max_seq_len})"
        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

        if repeat_batches > 1:
            for i in range(len(model)):
                for layer in model[i].layers:
                    k_cache, v_cache = layer.attention.layer_past
                    k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                    v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)
        device_sampling_params = (
            SamplingParams(
                temperature=sampling_params["temperature"],
                top_k=sampling_params["top_k"],
                top_p=sampling_params["top_p"],
                seed=sampling_params["seed"] if "seed" in sampling_params else None,
                frequency_penalty=sampling_params["frequency_penalty"]
                if "frequency_penalty" in sampling_params
                else 0.0,
                presence_penalty=sampling_params["presence_penalty"] if "presence_penalty" in sampling_params else 0.0,
                repetition_penalty=sampling_params["repetition_penalty"]
                if "repetition_penalty" in sampling_params
                else 1.0,
                enable_log_probs=sampling_params["enable_log_probs"]
                if "enable_log_probs" in sampling_params
                else False,
            )
            if model[0]._supports_on_device_sampling
            else None
        )

        if device_sampling_params is None and isinstance(sampling_params["temperature"], List):
            sampling_params["temperature"] = sampling_params["temperature"][0]
            sampling_params["top_p"] = sampling_params["top_p"][0]
            if "enable_log_probs" in sampling_params:
                sampling_params["enable_log_probs"] = sampling_params["enable_log_probs"][0]

        prefill_sampling_params = device_sampling_params if device_sampling_params is not None else None

        if mode == "prefill" or mode == "full":
            logger.info("Starting prefill warmup...")
            profiler.start(f"compile_prefill", iteration=batch_idx)
            logits = generator.prefill_forward_text(
                input_tokens_prefill_pt,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=decoding_pos,
                sampling_params=prefill_sampling_params,
                warmup_prefill=not is_device_perf_test,
                enable_trace=enable_trace,
            )
            profiler.end(f"compile_prefill", iteration=batch_idx)
            logger.info("Finished prefill warmup")

            logger.info(f"Starting prefill...")
            profiler.start(f"inference_prefill", iteration=batch_idx)
            prefill_out = generator.prefill_forward_text(
                input_tokens_prefill_pt,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=decoding_pos,
                sampling_params=prefill_sampling_params,
                warmup_prefill=not is_device_perf_test,
                enable_trace=enable_trace,
            )
            if prefill_sampling_params is not None and isinstance(prefill_out, tuple):
                prefilled_token, prefill_log_probs = prefill_out
            else:
                logits = prefill_out
                prefilled_token = torch.argmax(logits, dim=-1)
            profiler.end(f"inference_prefill", iteration=batch_idx)
            logger.info(f"Prefill finished")
        else:
            profiler.start(f"compile_prefill", iteration=batch_idx)
            profiler.end(f"compile_prefill", iteration=batch_idx)
            profiler.start(f"inference_prefill", iteration=batch_idx)
            profiler.end(f"inference_prefill", iteration=batch_idx)
            logger.info(f"Skipping prefill forward pass when decode mode is enabled")

            prefilled_token = torch.zeros(global_batch_size, 1, 1)

        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(global_batch_size)]
        for user in range(global_batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * global_batch_size

        current_pos = torch.tensor([decoding_pos[b] if mode == "full" else 0 for b in range(global_batch_size)])

        iteration = 0
        users_decoding = True

        out_tok = prefilled_token

        logger.info(f"Starting decode loop...")

        profiler.start(f"inference_decode", iteration=batch_idx)

        if mode == "prefill":
            profiler.start(f"compile_decode", iteration=batch_idx)
            profiler.end(f"compile_decode", iteration=batch_idx)
            profiler.start(f"inference_decode_time_{1}", iteration=batch_idx)
            profiler.end(f"inference_decode_time_{1}", iteration=batch_idx)
            logger.info(f"Skipping decode forward pass when prefill mode is enabled")

        pipe = (
            PipelinedDecode(generator, is_tokens=True)
            if device_sampling_params is not None and (token_accuracy or enable_trace) and not stress_test
            else None
        )

        def consume(out_tok):
            nonlocal users_decoding
            for user in range(global_batch_size):
                user_tok = out_tok[user].item()
                if user_tok not in tokenizer.stop_tokens and user_done[user] == False:
                    all_outputs[user].append(user_tok)
                else:
                    if stop_at_eos:
                        user_done[user] = True
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False
                    else:
                        all_outputs[user].append(user_tok)

            for user in range(global_batch_size):
                text = "".join(tokenizer.decode(all_outputs[user]))
                if len(text) > 100:
                    text = "..." + text[-97:]
                text = text.replace("\n", " ")
                logger.debug("[User {}] {}".format(user, text))

        last_from_pipe = False
        while users_decoding and mode != "prefill":
            if iteration == 0:
                profiler.start(f"compile_decode", iteration=batch_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=batch_idx)
            if token_accuracy:
                if pipe is None or iteration == 0:
                    out_tok[0] = token_acc.collect_predicted_tokens(out_tok[0].item())
                else:
                    out_tok[0] = token_acc.reference_tokens[min(token_acc.gt_pos + 1, token_acc.maxindex)].unsqueeze(-1)

            reload_decode_inputs = iteration == 0 or not enable_trace or device_sampling_params is None
            decode_kwargs = dict(
                enable_trace=enable_trace,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                sampling_params=device_sampling_params,
                prompt_tokens=input_tokens_prefill_pt,
                output_tokens=out_tok,
                reload_inputs=reload_decode_inputs,
                reload_page_table=False,
                reload_sampling_params=device_sampling_params is not None,
                reset_sampling_state=device_sampling_params is not None and iteration == 0,
            )
            if pipe is None:
                logits, log_probs = generator.decode_forward(out_tok, current_pos, **decode_kwargs)
            else:
                previous = pipe.step(out_tok.clone(), current_pos, **decode_kwargs)
                logits = None if previous is None else previous[0]

            if logits is not None:
                if device_sampling_params is not None:
                    out_tok = logits.unsqueeze(1)

                else:
                    _, out_tok = sample_host(
                        logits,
                        temperature=sampling_params["temperature"],
                        top_p=sampling_params["top_p"],
                        on_host=True,
                    )
                if pipe is not None and token_accuracy:
                    token_acc.collect_predicted_tokens(out_tok[0].item())

            if iteration == 0:
                profiler.end(f"compile_decode", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=batch_idx)
            else:
                profiler.end(f"inference_decode_time_{iteration}", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=batch_idx)

            tokens_per_second_per_user = 1 / decode_iteration_time
            logger.debug(
                f"Iteration {iteration}: {1000 * decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({global_batch_size * tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            if not stress_test:
                current_pos += 1
            if logits is not None:
                consume(out_tok)

            iteration += 1

            if iteration >= max_generated_tokens:
                if users_decoding and pipe is not None:
                    last_from_pipe = True
                users_decoding = False

        if pipe is not None:
            last = pipe.flush()
            if last_from_pipe and last is not None:
                out_tok = last[0].unsqueeze(1)
                consume(out_tok)
            users_decoding = False

        if not users_decoding:
            profiler.start(f"log_saving_file", iteration=batch_idx)
            logger.info("Finished decoding, printing the final outputs...\n")
            for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                text = tokenizer.decode(output)
                prompt_including_assistant_tags = tokenizer.decode(
                    model_args[0].encode_prompt(prompt, instruct=instruct)
                )
                text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
                if print_to_file:
                    with open(output_filename, "a") as f:
                        f.write(f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n")
                else:
                    short_prompt = (
                        (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:])
                        if len(prompt) > 200
                        else prompt
                    )
                    logger.info(
                        f"\n==REPEAT BATCH {batch_idx}\n==USER {i} - PROMPT\n{short_prompt} \n==USER {i} - OUTPUT\n{text_after_prompt.strip()}\n"
                    )
            profiler.end(f"log_saving_file", iteration=batch_idx)

        num_tokens_generated_decode.append(iteration)

        if "ci-eval-1" in test_id:
            if not hasattr(test_demo_text, "batch_outputs"):
                test_demo_text.batch_outputs = []
            if batch_idx == 0:
                test_demo_text.batch_outputs = []

            if all_outputs and len(all_outputs) > 0:
                final_output_text = tokenizer.decode(all_outputs[0])
                test_demo_text.batch_outputs.append(final_output_text)
                logger.info(f"Stored output for batch {batch_idx}: {final_output_text[:100]}...")

        if "ci-eval-32" in test_id:
            if not hasattr(test_demo_text, "batch32_outputs"):
                test_demo_text.batch32_outputs = []
            if batch_idx == 0:
                test_demo_text.batch32_outputs = []

            batch_outputs = []
            if all_outputs and len(all_outputs) > 0:
                for user_idx in range(len(all_outputs)):
                    final_output_text = tokenizer.decode(all_outputs[user_idx])
                    batch_outputs.append(final_output_text)
                test_demo_text.batch32_outputs.append(batch_outputs)
                logger.info(f"Stored outputs for batch {batch_idx}: {len(batch_outputs)} users")

        if token_accuracy:
            acc = token_acc.compute_accuracy()
            logger.info(f"=== Top1 and Top5 Token Accuracy ===")
            logger.info(f" Top1 Accuracy: {acc[0] * 100:.2f}%, Top5 Accuracy: {acc[1] * 100:.2f}%")

        profiler.end(f"inference_decode", iteration=batch_idx)

    profiler.end("run")

    is_special_tokens_produced = [False] * len(all_outputs)
    for i, output in enumerate(all_outputs):
        output = output[len(encoded_prompts[i]) :]
        is_eos = [token in tokenizer.stop_tokens for token in output]
        if any(is_eos):
            output = output[: is_eos.index(True)]
        is_special_tokens_produced[i] = any(token in tokenizer.all_special_ids for token in output)
    if any(is_special_tokens_produced):
        logger.warning(f"{sum(is_special_tokens_produced)}/{len(all_outputs)} users produced special tokens")
        if is_ci_env:
            assert False, "model produced special tokens"

    compile_prefill_time = profiler.get_duration("compile_prefill") if mode != "decode" else 0
    compile_decode_time = profiler.get_duration("compile_decode") if mode != "prefill" else 0

    total_inference_prefill_time = profiler.get_duration("inference_prefill") if mode != "decode" else 0
    total_inference_decode_time = 0
    for i in range(1, num_tokens_generated_decode[0]):
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    avg_time_to_first_token = total_inference_prefill_time / global_batch_size

    avg_decode_iteration_time = (
        total_inference_decode_time / (num_tokens_generated_decode[0] - 1) if iteration > 1 else 0
    )

    prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * global_batch_size if mode != "decode" else 0
    decode_tok_s_user = (
        (num_tokens_generated_decode[0] - 1) / total_inference_decode_time if mode != "prefill" and iteration > 1 else 0
    )
    decode_tok_s = (
        ((num_tokens_generated_decode[0] - 1) / total_inference_decode_time * global_batch_size)
        if mode != "prefill" and iteration > 1
        else 0
    )

    measurements = {
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,
        "decode_t/s/u": decode_tok_s_user,
        "decode_t/s": decode_tok_s,
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    tok_1_perf = profiler.get_duration(f"inference_decode_time_{1}") if 1 < num_tokens_generated_decode[0] else 0
    tok_128_perf = profiler.get_duration(f"inference_decode_time_{127}") if 127 < num_tokens_generated_decode[0] else 0
    tok_1024_perf = (
        profiler.get_duration(f"inference_decode_time_{1023}") if 1023 < num_tokens_generated_decode[0] else 0
    )
    tok_4096_perf = (
        profiler.get_duration(f"inference_decode_time_{4095}") if 4095 < num_tokens_generated_decode[0] else 0
    )

    if not stop_at_eos:
        logger.info(f"Please note that 'stop_at_eos' is disabled. Output repetition is expected.")

    logger.info("")
    logger.info(f"=== Performance metrics ===")
    if tok_1_perf > 0:
        logger.info(
            f"1st token decode time: {tok_1_perf * 1000:.2f}ms [{round(1 / tok_1_perf, 2)} t/s/u, {round((1 / tok_1_perf) * global_batch_size, 2)} t/s]"
        )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf * 1000:.2f}ms [{round(1 / tok_128_perf, 2)} t/s/u, {round((1 / tok_128_perf) * global_batch_size, 2)} t/s]"
        )
    if tok_1024_perf > 0:
        logger.info(
            f"1024th token decode time: {tok_1024_perf * 1000:.2f}ms [{round(1 / tok_1024_perf, 2)} t/s/u, {round((1 / tok_1024_perf) * global_batch_size, 2)} t/s]"
        )
    if tok_4096_perf > 0:
        logger.info(
            f"4096th token decode time: {tok_4096_perf * 1000:.2f}ms [{round(1 / tok_4096_perf, 2)} t/s/u, {round((1 / tok_4096_perf) * global_batch_size, 2)} t/s]"
        )

    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token * 1000, 2)}ms")
    logger.info(
        f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ {round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )

    tt_device_name = determine_device_name(mesh_device)
    model_name = model_args[0].base_model_name
    resolved_perf_targets = resolve_perf_targets(
        model_name=model_name,
        sku=tt_device_name,
        batch_size=global_batch_size,
        seq_len=max_seq_len,
    )
    if not resolved_perf_targets:
        logger.info(
            f"Model {model_name} does not have centralized performance targets set for "
            f"device {tt_device_name}, batch_size={global_batch_size}, seq_len={max_seq_len}"
        )
        targets = {}
    else:
        targets = {
            "prefill_t/s": resolved_perf_targets.get("prefill_t/s"),
            "decode_t/s": resolved_perf_targets.get("decode_t/s"),
            "decode_t/s/u": resolved_perf_targets.get("decode_t/s/u"),
        }

    if is_ci_env and not skip_perf_report:
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        benchmark_data.add_measurement(
            profiler,
            0,
            "load_weights",
            "time(s)",
            profiler.get_duration("load_weights"),
            step_warm_up_num_iterations=None,
            target=None,
        )

        if not token_accuracy:
            for i in range(1, num_tokens_generated_decode[0]):
                benchmark_data.add_measurement(
                    profiler,
                    0,
                    "inference_decode",
                    f"time_to_token_{i}",
                    profiler.get_duration(f"inference_decode_time_{i}") * 1000,
                    step_warm_up_num_iterations=None,
                    target=None,
                )

            for token_pos in [1, 128, 1024, 2048, 4096, 8192]:
                step_key = token_pos
                if step_key < num_tokens_generated_decode[0]:
                    benchmark_data.add_measurement(
                        profiler,
                        0,
                        "inference_decode",
                        f"decode_latency_ms_token_{token_pos}",
                        profiler.get_duration(f"inference_decode_time_{step_key}") * 1000,
                        step_warm_up_num_iterations=None,
                        target=None,
                    )

            num_iterations_for_avg = min(128, num_tokens_generated_decode[0])
            inference_decode_time_first_128 = sum(
                profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, num_iterations_for_avg)
            )
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                "avg_decode_time_first_128",
                inference_decode_time_first_128 * 1000 / max(1, num_iterations_for_avg - 1),
                step_warm_up_num_iterations=None,
                target=None,
            )

        if token_accuracy:
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                "top1_token_accuracy",
                acc[0] * 100,
                step_warm_up_num_iterations=None,
                target=None,
            )
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                "top5_token_accuracy",
                acc[1] * 100,
                step_warm_up_num_iterations=None,
                target=None,
            )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo_accuracy" if token_accuracy else "demo_perf",
            ml_model_name=model_name,
            ml_model_type="llm",
            device_name=tt_device_name,
            num_layers=model_args[0].n_layers,
            batch_size=global_batch_size,
            config_params={"data_parallel": data_parallel, "tensor_parallel": num_devices // data_parallel},
            input_sequence_length=max(prefill_lens),
            output_sequence_length=num_tokens_generated_decode[0],
        )

        if "performance" in test_id and "ci-32" in test_id:
            logger.info(
                f"Checking measurements against CI performance targets for batch size 32 of {model_name} on {tt_device_name}"
            )
            verify_perf(
                measurements,
                expected_measurements={
                    "prefill_time_to_token": True,
                    "decode_t/s": True,
                    "decode_t/s/u": True,
                },
                model_name=model_name,
                sku=tt_device_name,
                batch_size=global_batch_size,
                seq_len=max_seq_len,
            )
    if token_accuracy:
        total_top1_acc = math.ceil(acc[0] * 100)
        total_top5_acc = math.ceil(acc[1] * 100)

        if not json_config_file:
            min_top1_acc, min_top5_acc = get_accuracy_thresholds(model_args[0], seq_len=max(prefill_lens))
            verify_accuracy(
                measurements={
                    "top1_token_accuracy": total_top1_acc,
                    "top5_token_accuracy": total_top5_acc,
                },
                expected_accuracy_metrics={"top1": min_top1_acc, "top5": min_top5_acc},
            )
            logger.info("Checks of top-1 and top-5 accuracy against centralized model targets passed")

    if (
        "ci-eval-1" in test_id
        and hasattr(test_demo_text, "batch_outputs")
        and len(test_demo_text.batch_outputs) == repeat_batches
    ):
        logger.info("=== Repeat Batch Output Comparison ===")

        comparisons = [(i, i + 1) for i in range(0, repeat_batches, 2)]
        comparison_names = [f"Batch{i // 2 + 1}<->Batch{i // 2 + 1}" for i in range(0, repeat_batches, 2)]

        all_matches = True
        for i, (batch1_idx, batch2_idx) in enumerate(comparisons):
            output1 = test_demo_text.batch_outputs[batch1_idx]
            output2 = test_demo_text.batch_outputs[batch2_idx]

            if output1 == output2:
                logger.info(
                    f"{comparison_names[i]} comparison PASSED: Batches {batch1_idx + 1} and {batch2_idx + 1} outputs match"
                )
            else:
                logger.warning(
                    f"{comparison_names[i]} comparison FAILED: Batches {batch1_idx + 1} and {batch2_idx + 1} outputs differ"
                )
                logger.info(f"  Batch {batch1_idx + 1} output: {output1[:100]}...")
                logger.info(f"  Batch {batch2_idx + 1} output: {output2[:100]}...")
                all_matches = False

        assert all_matches, "Repeat batch outputs should be identical"

    if (
        "ci-eval-32" in test_id
        and hasattr(test_demo_text, "batch32_outputs")
        and len(test_demo_text.batch32_outputs) > 1
    ):
        logger.info("=== Batch32 Shifting Test Output Comparison ===")

        num_batches = len(test_demo_text.batch32_outputs)
        num_users = len(test_demo_text.batch32_outputs[0])

        logger.info(f"Comparing {num_batches} batches with {num_users} users each")

        consistency_checks = []

        for batch_idx in range(num_batches - 1):
            current_batch = test_demo_text.batch32_outputs[batch_idx]
            next_batch = test_demo_text.batch32_outputs[batch_idx + 1]

            logger.info(f"Comparing batch {batch_idx} vs batch {batch_idx + 1}")

            for user_offset in range(num_users):
                current_user_idx = (user_offset + 1) % num_users
                next_user_idx = user_offset

                current_output = current_batch[current_user_idx]
                next_output = next_batch[next_user_idx]

                expected_prompt_idx = (user_offset + batch_idx + 1) % num_users

                if current_output == next_output:
                    consistency_checks.append(True)
                    logger.info(
                        f"User {current_user_idx} (batch {batch_idx}) matches User {next_user_idx} (batch {batch_idx + 1}) - both used prompt[{expected_prompt_idx}]"
                    )
                else:
                    consistency_checks.append(False)
                    logger.warning(
                        f"User {current_user_idx} (batch {batch_idx}) differs from User {next_user_idx} (batch {batch_idx + 1}) - both should use prompt[{expected_prompt_idx}]"
                    )
                    logger.info(f"  Batch {batch_idx} User {current_user_idx}: {current_output[:50]}...")
                    logger.info(f"  Batch {batch_idx + 1} User {next_user_idx}: {next_output[:50]}...")

        all_consistent = all(consistency_checks)
        failed_checks = sum(1 for check in consistency_checks if not check)
        total_checks = len(consistency_checks)

        assert (
            all_consistent
        ), f"Batch32 repeat batch outputs should be identical - {failed_checks} out of {total_checks} consistency checks failed"


test_demo_text.pytestmark = list(getattr(_tt_test_demo_text, "pytestmark", []))
