# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import re
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_common import (
    sample,
    HostEmbedding,
)
from models.demos.llama3.tt.llama_model import TtTransformer
from models.demos.llama3.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3.tt.model_config import TtModelArgs, LlamaOptimizations
from models.demos.llama3.tt.llama_rope import TtLlamaRotarySetup
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.demos.llama3.tt.llama_common import PagedAttentionConfig
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import profiler, skip_for_grayskull

if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
    from tracy import signpost


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "seq_len, expected_compile_time",
    (
        (32, 30),
        (128, 30),
        (1024, 30),
    ),
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False
    ),
    ids=(
        "paged_attention",
        # "default_attention"
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_llama_model_perf(
    batch_size,
    seq_len,
    expected_compile_time,
    paged_attention,
    page_params,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device, optimizations=LlamaOptimizations.performance, max_batch_size=batch_size, max_seq_len=seq_len)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    if "3.2-1B" in model_args.DEFAULT_CACHE_PATH:
        expected_inference_time = 0.045
    elif "3.2-3B" in model_args.DEFAULT_CACHE_PATH:
        expected_inference_time = 0.065
    elif "3.1-8B" in model_args.DEFAULT_CACHE_PATH:
        expected_inference_time = 0.08
    elif "3.2-11B" in model_args.DEFAULT_CACHE_PATH:
        expected_inference_time = 0.085
    elif "3.1-70B" in model_args.DEFAULT_CACHE_PATH:
        expected_inference_time = 0.15
    else:
        assert False, f"Llama model not found. Supported Llama models: [3.2-1B, 3.2-3B, 3.1-8B, 3.2-11B, 3.1-70B]"

    # model_args.n_layers = 1
    # Clear global profiler state before starting measurements
    profiler.clear()

    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()

    profiler.end("weight_loading")

    prompts = ["This is a test"] * model_args.max_batch_size
    encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]

    # Embedding on host
    embd = HostEmbedding(model_args)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    generation_start_pos = seq_len
    generation_length = 1

    # Setup RoPE transformation matrices
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
    )
    transformation_mats_decode = rope_setup.get_trans_mats()
    transformation_mats = {"decode": transformation_mats_decode}

    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    profiler.start("TtLlama_model_setup")

    # Load TTNN model
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
    )
    # Load TTNN embedding module
    tt_embd = TtLlamaEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    profiler.end("TtLlama_model_setup")

    # Call the function
    profiler.start(f"end_to_end_inference_with_compile")
    run_inference(
        tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length, rope_setup, page_table_tt
    )
    profiler.end(f"end_to_end_inference_with_compile")
    profiler.print()
    compile_and_iter_time = profiler.get("model_run_for_inference_0")

    ttnn.DumpDeviceProfiler(mesh_device.get_devices()[0])

    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("Model perf run")

    profiler.start(f"end_to_end_inference")
    run_inference(
        tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length, rope_setup, page_table_tt
    )
    profiler.end(f"end_to_end_inference")
    profiler.print()
    iter_time = profiler.get("end_to_end_inference")

    comment = f"kv_cache_len={seq_len}_num_layers={model_args.n_layers}"

    # Extract the version, number of weights and device name from the cache folder
    if "3.1" in model_args.DEFAULT_CACHE_PATH:
        llama_version = "3.1"
    else:
        llama_version = "3.2"
    llama_weight = re.search(r"(\d+)B", model_args.DEFAULT_CACHE_PATH).group(1)
    llama_device = model_args.device_name

    prep_perf_report(
        model_name=f"Llama{llama_version}_{llama_weight}B_{llama_device}_{comment}",
        batch_size=model_args.max_batch_size,
        inference_and_compile_time=compile_and_iter_time,
        inference_time=iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )


def run_inference(
    tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length, rope_setup, page_table
):
    seqlen = 1  # Generating one token per user at a time
    batch = tt_model.args.max_batch_size
    mesh_device = tt_model.mesh_device

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

    # Send first input to device
    current_pos = torch.tensor([generation_start_pos] * batch)
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Get cos/sin matrices for the current position of each user
    rot_mats = rope_setup.get_rot_mats(current_pos)

    for i in range(generation_length):
        # Run TT model
        profiler.start(f"model_run_for_inference_{i}")

        decode_input = ttnn.unsqueeze_to_4D(tt_embd(tt_out_tok))
        decode_input = ttnn.to_memory_config(decode_input, tt_model.args.model_config["DECODE_RESIDUAL_MEMCFG"])
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table,
        )
        tt_out_rm = ttnn.untilize(tt_out, use_multicore=True)
        ttnn.deallocate(tt_out)
        tt_out_tok = ttnn.argmax(
            tt_out_rm,
            dim=3,
            use_multicore=True if tt_model.args.max_batch_size == 1 else False,
            output_tensor=tt_out_tok,
        )
        ttnn.deallocate(tt_out_rm)

        # Update the rotation matrix for the next iteration
        ttnn.plus_one(current_pos_tensor)

        # Update rot_mats for next iteration
        current_pos += 1
        rot_mats = rope_setup.get_rot_mats(current_pos)

        profiler.end(f"model_run_for_inference_{i}")

    # Synchronize devices to ensure all profiling data is captured accurately
    for i in range(tt_model.args.num_devices):
        ttnn.synchronize_device(mesh_device.get_devices()[i])
