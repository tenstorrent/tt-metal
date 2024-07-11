# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import time
import json

from models.demos.mamba.demo.demo import (
    get_tokenizer,
    get_cpu_reference_model,
    get_tt_metal_model,
    display_tokens,
)

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    skip_for_grayskull,
    skip_for_wormhole_b0,
)
from tt_metal.tools.profiler.process_model_log import get_samples_per_s


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch, iterations, expected_compile_time, expected_inference_time",
    ((32, 10, 12.5, 0.40),),  # Issue 7816 Compile time
)
def test_mamba_e2e_perf(
    device,
    batch,
    iterations,
    expected_compile_time,
    expected_inference_time,
    use_program_cache,
    reset_seeds,
    get_tt_cache_path,
):
    model_version = "state-spaces/mamba-2.8b-slimpj"
    display_decoded_seq = False

    tokenizer = get_tokenizer()

    # Clear global profiler state before starting measurements
    profiler.clear()

    # Load prompts
    with open("models/demos/mamba/demo/prompts.json", "r") as f:
        prompts = json.load(f)

    profiler.start("pytorch_ref_model_setup")
    reference_model = get_cpu_reference_model(model_version, batch)
    profiler.end("pytorch_ref_model_setup")

    profiler.start("tt_model_setup")
    tt_model = get_tt_metal_model(model_version, device, cache_dir=get_tt_cache_path(model_version), batch_size=batch)
    profiler.end("tt_model_setup")

    sequences: torch.Tensor = tokenizer(prompts, return_tensors="pt", padding=True).input_ids

    # Required due to non-deterministic hang on CI (#8606)
    disable_persistent_kernel_cache()

    # prefill
    prefill_iterations = sequences.shape[1] - 1
    for idx in range(prefill_iterations):
        if idx == 0:
            profiler.start("ref_model_run_for_inference_0")
        ref_logits = reference_model(sequences[:, idx].unsqueeze(1))
        if idx == 0:
            profiler.end("ref_model_run_for_inference_0")

        if idx == 0:
            profiler.start("model_run_for_inference_0")
        tt_logits = tt_model(sequences[:, idx].unsqueeze(1))
        if idx == 0:
            profiler.end("model_run_for_inference_0")

    # Decode Starts
    start = time.time()
    token_counts = 0
    total_iterations = iterations + prefill_iterations
    inference_profile_iteration = total_iterations - 2  # Profile the second last iteration

    for idx in range(prefill_iterations, total_iterations):
        if idx == inference_profile_iteration:
            profiler.start(f"model_run_for_inference_{idx}")
        tt_logits = tt_model(sequences[:, idx].unsqueeze(1))
        if idx == inference_profile_iteration:
            profiler.end(f"model_run_for_inference_{idx}")

        probs = torch.nn.functional.softmax(tt_logits.squeeze(1), dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        sequences = torch.cat([sequences, next_token.unsqueeze(-1)], dim=1)

        token_counts += sequences.shape[0]

        if display_decoded_seq:
            decoded = tokenizer.batch_decode(sequences, skip_special_tokens=False)
            display_tokens(decoded)
            throughput = token_counts / (time.time() - start)
            print(f"Current total throughput: {throughput:.2f} tok/s")
            print(f"Current throughput per user: {(throughput/32):.2f} tok/s/u")

    # profiler.print()
    comment = ""
    ref_model_run_for_inference = profiler.get("ref_model_run_for_inference_0")
    first_iter_time = profiler.get("model_run_for_inference_0")
    second_iter_time = profiler.get(f"model_run_for_inference_{inference_profile_iteration}")

    prep_perf_report(
        model_name=f"{model_version}",
        batch_size=batch,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        inference_time_cpu=ref_model_run_for_inference,
        comments=comment,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch, warmup, expected_device_fw_duration_ms",
    ((32, True, 1.6),),
)
def test_mamba_perf_device(batch, warmup, expected_device_fw_duration_ms, reset_seeds):
    subdir = "ttnn_mamba"
    margin = 0.03
    if warmup:
        inference_iterations = 2
    else:
        inference_iterations = 1
    command = f"pytest models/demos/mamba/tests/test_full_model.py::test_device_perf[{inference_iterations}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    # convert expected perf (ms) to samples/s
    expected_device_fw_duration_ns = expected_device_fw_duration_ms * 1e6  # convert ms to ns
    expected_total_device_fw_samples = get_samples_per_s(expected_device_fw_duration_ns * inference_iterations, batch)

    inference_time_key = "AVG DEVICE FW SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_total_device_fw_samples}

    post_processed_results = run_device_perf(command, subdir, 1, cols, batch)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    comment = ""
    prep_device_perf_report(
        model_name=f"mamba-2.8b_batch_{batch}",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=comment,
    )
