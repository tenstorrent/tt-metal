# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import re
import os
import ttnn
from models.demos.qwen.tt.qwen_common import (
    HostEmbedding,
    get_single_rot_mat,
)
from models.demos.qwen.tt.qwen_model import TtTransformer
from models.demos.qwen.tt.qwen_embedding import TtQwenEmbedding
from models.demos.qwen.tt.model_config import TtModelArgs
from models.demos.qwen.reference.tokenizer import Tokenizer

from models.perf.perf_utils import prep_perf_report
from models.utility_functions import profiler, skip_for_grayskull

if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
    from tracy import signpost


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "kv_cache_len, expected_compile_time",
    (
        (32, 20),
        (128, 20),
        (1024, 20),
    ),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 1),
    ],
    indirect=True,
)
def test_qwen_model_perf(mesh_device, kv_cache_len, expected_compile_time, use_program_cache, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(mesh_device)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    expected_inference_time = 0.08

    # model_args.n_layers = 1
    # Clear global profiler state before starting measurements
    profiler.clear()

    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()

    profiler.end("weight_loading")

    prompts = ["This is a test"] * model_args.max_batch_size
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    # Embedding on host
    embd = HostEmbedding(model_args)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}embed_tokens.weight"]})

    generation_start_pos = kv_cache_len
    generation_length = 1

    profiler.start("TtQwen_model_setup")

    # Load TTNN model
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )
    # Load TTNN embedding module
    tt_embd = TtQwenEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    profiler.end("TtQwen_model_setup")

    # Call the function
    profiler.start(f"end_to_end_inference_with_compile")
    run_inference(tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"end_to_end_inference_with_compile")
    profiler.print()
    compile_and_iter_time = profiler.get("model_run_for_inference_0")

    ttnn.DumpDeviceProfiler(mesh_device)

    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("Model perf run")

    profiler.start(f"end_to_end_inference")
    run_inference(tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"end_to_end_inference")
    profiler.print()
    iter_time = profiler.get("end_to_end_inference")

    comment = f"kv_cache_len={kv_cache_len}_num_layers={model_args.n_layers}"

    # Extract the version, number of weights and device name from the cache folder
    qwen_version = "2.0"
    qwen_weight = re.search(r"(\d+)B", model_args.DEFAULT_CACHE_PATH).group(1)
    qwen_device = model_args.device_name

    prep_perf_report(
        model_name=f"Qwen{qwen_version}_{qwen_weight}B_{qwen_device}_{comment}",
        batch_size=model_args.max_batch_size,
        inference_and_compile_time=compile_and_iter_time,
        inference_time=iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )


def run_inference(tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length):
    seqlen = 1  # Generating one token per user at a time
    batch = tt_model.args.max_batch_size
    mesh_device = tt_model.mesh_device

    # pre-compute the rotational embedding matrix and send to device
    current_rot_mat, rot_matrix = get_single_rot_mat(
        tt_model.args.head_dim,
        tt_model.mesh_device,
        tt_model.args.num_devices,
        start_pos=0,
    )

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]

    # Initialize tt_out_tok with the first token
    tt_out_tok = ttnn.from_torch(
        torch.nn.functional.pad(
            encoded_prompts_tensor[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0), (0, 31), "constant", 0
        ),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
    )

    # Generate first input on host
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    # Send first input to device
    tt_decode_input = pt_decode_input
    decode_input = tt_model.args.prepare_inputs_ttnn_decode(
        tt_decode_input,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    current_pos = ttnn.from_torch(
        torch.tensor([generation_start_pos] * batch),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.int32,
    )

    for i in range(generation_length):
        # Run TT model
        profiler.start(f"model_run_for_inference_{i}")

        decode_input = ttnn.unsqueeze_to_4D(tt_embd(tt_out_tok))
        tt_out = tt_model(decode_input, current_pos, rot_mat=current_rot_mat)
        tt_out_rm = ttnn.untilize(tt_out, use_multicore=True)
        ttnn.deallocate(tt_out)
        tt_out_tok = ttnn.argmax(tt_out_rm, dim=3, keepdim=True, use_multicore=True, output_tensor=tt_out_tok)
        ttnn.deallocate(tt_out_rm)

        # Update the rotation matrix for the next iteration
        new_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)
        current_rot_mat = ttnn.copy(new_rot_mat, current_rot_mat)
        ttnn.plus_one(current_pos)

        profiler.end(f"model_run_for_inference_{i}")

    # Synchronize devices to ensure all profiling data is captured accurately
    ttnn.synchronize_device(mesh_device)
