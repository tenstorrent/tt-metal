# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.wormhole.llama31_8b.tt.llama_common import (
    prepare_inputs_ttnn,
    sample,
    HostEmbedding,
    get_single_rot_mat,
)
from models.demos.wormhole.llama31_8b.tt.llama_model import TtTransformer
from models.demos.wormhole.llama31_8b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import profiler, skip_for_grayskull

if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
    from tracy import signpost


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "kv_cache_len, expected_compile_time, expected_inference_time",
    (
        (32, 6, 0.09),
        (128, 6, 0.09),
        (1024, 11, 0.09),
    ),
)
def test_llama_model_perf(
    device, kv_cache_len, expected_compile_time, expected_inference_time, use_program_cache, reset_seeds
):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(device)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    model_args.n_layers = 32
    # Clear global profiler state before starting measurements
    profiler.clear()

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

    prompts = ["This is a test"] * model_args.max_batch_size
    encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]

    # Embedding on host
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
    # TODO Add argmax + embedding on device, same as the demo.py code

    generation_start_pos = kv_cache_len
    generation_length = 1

    profiler.start("TtLlama_model_setup")

    # Load TTNN model
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layers=list(range(model_args.n_layers)),
        rot_mat=None,
        start_pos=generation_start_pos,
    )
    # Load TTNN embedding module
    tt_embd = TtLlamaEmbedding(
        device=device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    profiler.end("TtLlama_model_setup")

    # Call the function
    profiler.start(f"end_to_end_inference_with_compile")
    run_inference(tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"end_to_end_inference_with_compile")
    profiler.print()
    compile_and_iter_time = profiler.get("model_run_for_inference_0")

    ttnn.DumpDeviceProfiler(device)
    ttnn.synchronize_device(device)

    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("Model perf run")

    profiler.start(f"end_to_end_inference")
    run_inference(tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"end_to_end_inference")
    profiler.print()
    iter_time = profiler.get("end_to_end_inference")

    comment = f"kv_cache_len={kv_cache_len}_num_layers={model_args.n_layers}"

    prep_perf_report(
        model_name=f"Llama_31_8B_{comment}",
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

    # pre-compute the rotational embedding matrix and send to device
    current_rot_mat, rot_matrix = get_single_rot_mat(
        tt_model.args.head_dim,
        tt_model.device,
        start_pos=0,
    )

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]

    for i in range(generation_length):
        current_pos = generation_start_pos + i
        pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
        tt_decode_input = pt_decode_input
        decode_input, pos = prepare_inputs_ttnn(
            tt_decode_input,
            current_pos,
            tt_model.args.dim,
            tt_model.args.sliding_window,
            tt_model.device,
        )

        # Run TT model
        profiler.start(f"model_run_for_inference_{i}")
        tt_out = tt_model(decode_input, pos, rot_mat=current_rot_mat)

        # Convert ttnn tensor to torch tensor
        profiler.start(f"result_wait_for_inference_{i}")
        tt_out = ttnn.untilize(
            tt_out, use_multicore=False
        )  # multi-core OOMs (https://github.com/tenstorrent/tt-metal/issues/9022)
        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        profiler.end(f"model_run_for_inference_{i}")
        profiler.end(f"result_wait_for_inference_{i}")

        # Greedy decode the generated token and pass it back in, this is just a perf test
        tt_out_tok = sample(tt_output_torch, temperature=0, top_p=1)

        # Update the rotation matrix for the next iteration
        current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)
