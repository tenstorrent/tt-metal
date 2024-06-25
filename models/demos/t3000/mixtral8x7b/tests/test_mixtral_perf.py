# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from ttnn import ConcatMeshToTensor
import tt_lib

if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
    from tracy import signpost

from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn,
    prepare_rotation_mat_ttnn,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import profiler, enable_persistent_kernel_cache


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@pytest.mark.model_perf_t3000
@pytest.mark.parametrize(
    "generation_start_pos, expected_compile_time, expected_inference_time",
    (
        (32, 150, 0.058),  # FIXME: Perf regression (issue #9479)
        (128, 150, 0.058),  # FIXME: Perf regression (issue #9479)
        (1024, 150, 0.058),  # FIXME: Perf regression (issue #9479)
        (2048, 150, 0.058),  # FIXME: Perf regression (issue #9479)
    ),
)
def test_mixtral_model_perf(
    t3k_device_mesh,
    generation_start_pos,
    expected_compile_time,
    expected_inference_time,
    use_program_cache,
    reset_seeds,
):
    dtype = ttnn.bfloat8_b

    # Can use dummy_weights=True correctness is not tested, but it is much slower
    model_args = TtModelArgs(t3k_device_mesh.get_device(0), dummy_weights=False)
    model_args.n_layers = 1

    # Clear global profiler state before starting measurements
    profiler.clear()

    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    prompts = ["Once"] * 32
    if model_args.dummy_weights:
        encoded_prompts = [[1, 5713]] * len(prompts)  # manual encoding of the "Once" prompt
    else:
        tokenizer = Tokenizer(model_args.tokenizer_path)
        encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    generation_length = 1

    profiler.start("Mixtral_model_setup")

    # Load TTNN model
    tt_model = TtTransformer(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    rot_mat = prepare_rotation_mat_ttnn(
        tt_model.args.head_dim,
        tt_model.args.max_seq_len,
        tt_model.device_mesh,
    )
    profiler.end("TtMistral_model_setup")

    # Call the function
    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("Model warmup")
    profiler.start(f"end_to_end_inference_with_compile")
    run_inference(tt_model, embd, encoded_prompts, generation_start_pos, generation_length, rot_mat)
    profiler.end(f"end_to_end_inference_with_compile")
    profiler.print(units="ms")
    compile_and_iter_time = profiler.get("model_run_for_inference_0")

    for device_id in t3k_device_mesh.get_device_ids():
        tt_lib.device.DumpDeviceProfiler(t3k_device_mesh.get_device(device_id))

    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("Model perf run")
    profiler.clear()
    profiler.start(f"end_to_end_inference")
    run_inference(tt_model, embd, encoded_prompts, generation_start_pos, generation_length, rot_mat)
    profiler.end(f"end_to_end_inference")
    profiler.print(units="ms")
    iter_time = profiler.get("model_run_for_inference_0")

    comment = f"kv_cache_len={generation_start_pos}_num_layers={model_args.n_layers}"

    prep_perf_report(
        model_name=f"Mixtral8x7B_{comment}",
        batch_size=model_args.max_batch_size,
        inference_and_compile_time=compile_and_iter_time,
        inference_time=iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )


def run_inference(tt_model, embd, encoded_prompts, generation_start_pos, generation_length, rot_mat):
    seqlen = 1  # Generating one token per user at a time
    batch = tt_model.args.max_batch_size

    profiler.start(f"torch_embed_initial")
    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    profiler.end(f"torch_embed_initial")

    for i in range(generation_length):
        start_pos = generation_start_pos + i
        current_pos = start_pos % tt_model.args.sliding_window

        profiler.start(f"prepare_inputs_for_inference_{i}")
        decode_input, attn_mask = prepare_inputs_ttnn(
            pt_decode_input,
            tt_model.args.dim,
            start_pos,
            tt_model.args.sliding_window,
            tt_model.device_mesh,
        )
        profiler.end(f"prepare_inputs_for_inference_{i}")

        # Run TT model
        profiler.start(f"model_run_for_inference_{i}")
        profiler.start(f"python_dispatch_for_inference_{i}")
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask, rot_mat)
        profiler.end(f"python_dispatch_for_inference_{i}")

        # Convert ttnn tensor to torch tensor
        profiler.start(f"result_wait_for_inference_{i}")
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(tt_model.device_mesh, dim=0))[0]
            .squeeze(1)
            .view(batch, seqlen, -1)
            .detach()
            .float()
        )

        profiler.end(f"model_run_for_inference_{i}")
        profiler.end(f"result_wait_for_inference_{i}")

        profiler.start(f"torch_argmax_and_embed_{i}")
        # Greedy decode the generated token and pass it back in, this is just a perf test
        tt_token_batch = tt_output_torch.squeeze().argmax(axis=-1)
        tt_decode_input = embd(tt_token_batch).view(batch, seqlen, -1)
        profiler.end(f"torch_argmax_and_embed_{i}")

        profiler.start(f"deallocate_tt_tensors_{i}")
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del decode_input, attn_mask, tt_decode_input
        tt_out.deallocate(force=True)
        profiler.end(f"deallocate_tt_tensors_{i}")
