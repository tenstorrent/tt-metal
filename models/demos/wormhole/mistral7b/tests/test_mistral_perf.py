# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    precompute_freqs,
    prepare_inputs_ttnn,
    freqs_to_rotation_matrix,
    sample,
)
from models.demos.wormhole.mistral7b.tt.mistral_model import TtTransformer
from models.demos.wormhole.mistral7b.tt.mistral_embedding import TtMistralEmbedding
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.demos.wormhole.mistral7b.reference.model import Transformer
from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import profiler, skip_for_grayskull

if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
    from tracy import signpost


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "kv_cache_len, expected_compile_time, expected_inference_time",
    (
        (32, 15, 0.105),
        (1024, 15, 0.225),
    ),
)
def test_mistral_model_perf(
    device, kv_cache_len, expected_compile_time, expected_inference_time, use_program_cache, reset_seeds
):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(device)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    model_args.n_layers = 1
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

    prompts = ["This is a test"] * model_args.max_batch_size
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
    # TODO Add argmax + embedding on device, same as the demo.py code

    generation_start_pos = kv_cache_len
    generation_length = 1

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

    # Call the function
    profiler.start(f"end_to_end_inference_with_compile")
    run_inference(tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"end_to_end_inference_with_compile")
    profiler.print()
    compile_and_iter_time = profiler.get("model_run_for_inference_0")

    ttnn.DumpDeviceProfiler(device)

    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("Model perf run")

    profiler.clear()
    profiler.start(f"end_to_end_inference")
    run_inference(tt_model, tt_embd, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"end_to_end_inference")
    profiler.print()
    iter_time = profiler.get("model_run_for_inference_0")

    comment = f"kv_cache_len={kv_cache_len}_num_layers={model_args.n_layers}"
    iter_time = profiler.get("model_run_for_inference_0")

    prep_perf_report(
        model_name=f"Mistral7B_{comment}",
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
        tt_out = tt_model(decode_input, pos)

        # Convert ttnn tensor to torch tensor
        profiler.start(f"result_wait_for_inference_{i}")
        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        profiler.end(f"model_run_for_inference_{i}")
        profiler.end(f"result_wait_for_inference_{i}")

        # Greedy decode the generated token and pass it back in, this is just a perf test
        tt_out_tok = sample(tt_output_torch, temperature=0, top_p=1)
