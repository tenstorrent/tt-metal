# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest

# Set Grok flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["GROK_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"

import ttnn
from ttnn import ConcatMeshToTensor

if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
    from tracy import signpost

from models.experimental.grok.tt.grok_common import (
    prepare_inputs_ttnn,
    prepare_rotation_mat_ttnn,
)
from models.experimental.grok.tt.grok_model import TtTransformer
from models.experimental.grok.reference.tokenizer import Tokenizer
from models.experimental.grok.tt.model_config import TtModelArgs
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import profiler, enable_persistent_kernel_cache
from transformers import AutoTokenizer


@pytest.mark.model_perf_t3000
@pytest.mark.parametrize(
    "generation_start_pos, expected_compile_time, expected_inference_time",
    (
        (32, 150, 0.025),
        (128, 150, 0.025),
        (1024, 150, 0.025),
        (2048, 150, 0.025),
    ),
)
def test_grok_model_perf(
    t3k_mesh_device,
    generation_start_pos,
    expected_compile_time,
    expected_inference_time,
    use_program_cache,
    reset_seeds,
):
    dtype = ttnn.bfloat8_b
    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(True)

    # Can use dummy_weights=True correctness is not tested, but it is much slower
    model_args = TtModelArgs(t3k_mesh_device.get_device(0), dummy_weights=False)
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
        tokenizer = AutoTokenizer.from_pretrained("hpcai-tech/grok-1", trust_remote_code=True)
        encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    # Embedding on host
    embd = torch.nn.Embedding(model_args.vocab_size, model_args.hidden_size)
    embd.load_state_dict({"weight": state_dict["model.embed_tokens.weight"]})

    generation_length = 1

    profiler.start("model_setup")

    # Load TTNN model
    tt_model = TtTransformer(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    rot_mat = prepare_rotation_mat_ttnn(
        tt_model.args.head_dim,
        tt_model.args.max_seq_len,
        tt_model.mesh_device,
    )
    profiler.end("model_setup")

    # Call the function
    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("Model warmup")
    profiler.start(f"end_to_end_inference_with_compile")
    run_inference(tt_model, embd, encoded_prompts, generation_start_pos, generation_length, rot_mat)
    profiler.end(f"end_to_end_inference_with_compile")
    profiler.print(units="ms")
    compile_and_iter_time = profiler.get("model_run_for_inference_0")

    for device_id in t3k_mesh_device.get_device_ids():
        ttnn.DumpDeviceProfiler(t3k_mesh_device.get_device(device_id))

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
        model_name=f"Grok8x7B_{comment}",
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
        current_pos = generation_start_pos + i

        profiler.start(f"prepare_inputs_for_inference_{i}")
        decode_input, attn_mask = prepare_inputs_ttnn(
            pt_decode_input,
            tt_model.args.dim,
            current_pos,
            tt_model.mesh_device,
        )
        profiler.end(f"prepare_inputs_for_inference_{i}")

        # Run TT model
        profiler.start(f"model_run_for_inference_{i}")
        profiler.start(f"python_dispatch_for_inference_{i}")
        tt_multidevice_out = tt_model(decode_input, current_pos, attn_mask, rot_mat)
        profiler.end(f"python_dispatch_for_inference_{i}")

        # Convert ttnn tensor to torch tensor
        profiler.start(f"result_wait_for_inference_{i}")
        tt_output_torch = (
            ttnn.to_torch(tt_multidevice_out, mesh_composer=ConcatMeshToTensor(tt_model.mesh_device, dim=-1))
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
        tt_multidevice_out.deallocate(force=True)
        profiler.end(f"deallocate_tt_tensors_{i}")
