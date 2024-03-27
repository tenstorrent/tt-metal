# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.mistral7b.tt.mistral_common import (
    precompute_freqs,
    prepare_inputs_ttnn,
    freqs_to_rotation_matrix,
    sample,
)
from models.demos.mistral7b.tt.mistral_model import TtTransformer
from models.demos.mistral7b.tt.mistral_embedding import TtMistralEmbedding
from models.demos.mistral7b.tt.model_config import TtModelArgs
from models.demos.mistral7b.reference.model import Transformer
from models.demos.mistral7b.reference.tokenizer import Tokenizer

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import profiler, enable_persistent_kernel_cache, skip_for_grayskull


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch, iterations, expected_compile_time, expected_inference_time",
    ((32, 12, 155, 0.16),),
)
def test_mistral_model_perf(
    device, batch, iterations, expected_compile_time, expected_inference_time, use_program_cache
):
    dtype = ttnn.bfloat8_b

    run_ref_pt = True

    model_args = TtModelArgs(device)
    model_args.max_batch_size = batch
    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Clear global profiler state before starting measurements
    profiler.clear()

    profiler.start("weight_loading")
    state_dict = torch.load(model_args.consolidated_weights_path)
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }
    profiler.end("weight_loading")

    prompts = ["This is a test"] * 32

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    if run_ref_pt:
        profiler.start("Mistral_pytorch_ref_model_setup")
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(state_dict)

        cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
        freqs_cis = torch.complex(cos, sin)
        profiler.end("Mistral_pytorch_ref_model_setup")

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
    # TODO Add argmax + embedding on device, same as the demo.py code

    generation_start_pos = 0
    generation_length = iterations
    seqlen = 1  # Generating one token per user at a time
    batch = 32

    profiler.start("TtMistral_model_setup")

    # pre-compute the rotational embedding matrix and send to device
    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    rot_emb_matrix = freqs_to_rotation_matrix(cos, sin)

    rot_emb_matrix_list = []
    for i in range(rot_emb_matrix.shape[0]):
        rot_emb_matrix_list.append(
            ttnn.from_torch(
                rot_emb_matrix[i, :, :].unsqueeze(0).unsqueeze(0), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT
            )
        )  # ttnn.bfloat16

    # Load TTNN model
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
    # Load TTNN embedding module
    tt_embd = TtMistralEmbedding(
        device=device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    profiler.end("TtMistral_model_setup")

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    tt_decode_input = pt_decode_input

    profiler.disable()  # Disable profiler for first 10 iterations
    for i in range(generation_length):
        current_pos = generation_start_pos + i

        if i == 0 or i == 10:  # Skip the first few iterations to warm up
            profiler.enable()
            enable_persistent_kernel_cache()
            profiler.start(f"input_processing_{i}")

        decode_input, pos = prepare_inputs_ttnn(
            tt_decode_input,
            current_pos,
            model_args.dim,
            model_args.sliding_window,
            tt_model.device,
        )
        if i == 0 or i == 10:  # Skip the first few iterations to warm up
            profiler.end(f"input_processing_{i}")
            profiler.start(f"model_run_for_inference_{i}")

        # Run TT model
        tt_out = tt_model(decode_input, pos)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        if i == 0 or i == 10:  # Skip the first few iterations to warm up
            profiler.end(f"model_run_for_inference_{i}")

        if run_ref_pt:  # Run reference model
            if i == 0:  # Skip the first few iterations to warm up
                profiler.start(f"ref_model_run_for_inference_{i}")

            freqs_cis_i = freqs_cis[current_pos, :].unsqueeze(0)
            positions = torch.tensor([current_pos])
            ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)

            if i == 0:  # Skip the first few iterations to warm up
                profiler.end(f"ref_model_run_for_inference_{i}")

        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            # tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            tt_out_tok = ttnn.from_torch(
                encoded_prompts_tensor[:, i].unsqueeze(-1),
                device=device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            tt_decode_input = tt_embd(tt_out_tok)  # Embedding on device
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Greedy decode (temperature = 0) the generated token and save it to print out later
            tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8)
            tt_out_tok = ttnn.from_torch(tt_out_tok, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            tt_decode_input = tt_embd(tt_out_tok)  # Embedding on device
            if run_ref_pt:
                pt_out_tok = sample(ref_output, temperature=0, top_p=0.8)
                pt_decode_input = embd(pt_out_tok)

    profiler.print()
    comment = f"num_layers={model_args.n_layers}"
    weight_loading = profiler.get("weight_loading")
    input_processing = profiler.get("input_processing")
    ref_model_run_for_inference = profiler.get("ref_model_run_for_inference_0")
    first_iter_time = profiler.get("model_run_for_inference_0")
    second_iter_time = profiler.get("model_run_for_inference_10")

    prep_perf_report(
        model_name=f"Mistral7B",
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
    "batch, iterations, expected_perf",
    ((32, 17, 0.16),),
)
def test_mistral_perf_device(batch, iterations, expected_perf):
    subdir = "ttnn_mistral7b"
    margin = 0.03
    command = f"pytest models/demos/mistral7b/tests/test_mistral_model.py::test_mistral_model_inference[{iterations}-generative]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, iterations, cols, batch)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"mistral-7B_{batch}batch",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
