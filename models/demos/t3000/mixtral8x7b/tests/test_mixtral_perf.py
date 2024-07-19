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
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh
import tt_lib

if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
    from tracy import signpost

from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    preprocess_inputs_prefill,
    prepare_inputs_ttnn_prefill,
    prepare_inputs_ttnn,
    get_rot_transformation_mat,
    get_prefill_rot_mat,
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
        (32, 150, 0.085),
        (128, 150, 0.085),
        (1024, 150, 0.085),
        (2048, 150, 0.085),
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
    model_args.n_layers = 32

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

    profiler.end("TtMixtral_model_setup")

    # Call the function
    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("Model warmup")
    profiler.start(f"end_to_end_inference_with_compile")
    run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"end_to_end_inference_with_compile")
    profiler.print(units="ms")
    compile_and_iter_time = profiler.get("model_run_for_inference_0")

    for device_id in t3k_device_mesh.get_device_ids():
        tt_lib.device.DumpDeviceProfiler(t3k_device_mesh.get_device(device_id))

    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("Model perf run")
    profiler.clear()
    profiler.start(f"end_to_end_inference")
    run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"end_to_end_inference")
    profiler.print(units="ms")
    iter_time = profiler.get("model_run_for_inference_0")

    comment = f"kv_cache_len={generation_start_pos}_num_layers={model_args.n_layers}"

    prep_perf_report(
        model_name=f"Mixtral8x7B_decode-mode_{comment}",
        batch_size=model_args.max_batch_size,
        inference_and_compile_time=compile_and_iter_time,
        inference_time=iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )


@pytest.mark.model_perf_t3000
@pytest.mark.parametrize(
    "prefill_seqlen, expected_compile_time, expected_inference_time",
    (
        (128, 80, 4.5),
        (1024, 80, 15),
        (1024 * 2, 80, 33),
        # (1024*4, 80, 60),
        # (1024*8, 150, 80),
        # (1024*16, 150, 100),
        # (1024*32, 150, 130),
    ),
    ids=[
        "prefill_128",
        "prefill_1k",
        "prefill_2k",
        # "prefill_4k",  # FIXME out of memory (decode)
        # "prefill_8k",  # FIXME out of memory (decode)
        # "prefill_16k",  # FIXME out of memory (decode)
        # "prefill_32k",  # FIXME out of memory (decode)
    ],
)
def test_mixtral_model_with_prefill_perf(
    t3k_device_mesh,
    prefill_seqlen,
    expected_compile_time,
    expected_inference_time,
    use_program_cache,
    reset_seeds,
):
    dtype = ttnn.bfloat8_b
    batch_size = 32

    # Can use dummy_weights=True correctness is not tested, but it is much slower
    model_args = TtModelArgs(t3k_device_mesh.get_device(0), dummy_weights=False)
    model_args.n_layers = 32

    # Clear global profiler state before starting measurements
    profiler.clear()

    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    # Prompt with size with a bit more than 128 tokens. increase the prompt based on the prefill seqlen to accomodate every seqlen.
    prompts = [
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way – in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only."
        * (prefill_seqlen // 128)
    ] * batch_size
    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    profiler.start("preprocessing_inputs")

    (
        input_tokens_prefill_tt,
        input_tokens_decode_tt,
        max_prompt_len,
        input_mask,
        input_tokens_prefill_pt,
        input_tokens_decode_pt,
        input_mask_pt,
        prefill_seq_len,
        encoded_prompts,
    ) = preprocess_inputs_prefill(prompts, tokenizer, model_args, dtype, False, t3k_device_mesh)

    profiler.end("preprocessing_inputs")

    assert (
        prefill_seq_len == prefill_seqlen
    ), f"The used prompt #tokens {len(encoded_prompts[0])} do not comply with the prefill seqlen being used. Expected: {prefill_seqlen} != from prompt:{prefill_seq_len}"

    pt_decode_input = embd(input_tokens_decode_pt[:, 0]).view(batch_size, 1, -1)
    pt_prefill_input = [embd(input_tokens_prefill_pt[b, :]).view(1, prefill_seq_len, -1) for b in range(batch_size)]

    profiler.start("Mixtral_model_setup")

    # Load TTNN model
    tt_model = TtTransformer(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
        start_pos=prefill_seq_len,
    )

    profiler.end("TtMixtral_model_setup")

    # Prefill (run warmup for single user before running perf for all users)
    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("prefill warmup")
    profiler.clear()
    profiler.start(f"e2e_prefill_warmup")
    run_inference_prefill(tt_model, model_args, prefill_seq_len, t3k_device_mesh, pt_prefill_input, 1)
    profiler.end(f"e2e_prefill_warmup")
    profiler.print(units="ms")
    prefill_warmup_time = profiler.get("prepare_inputs_and_prefill_1_users")

    # Profiler dump, ready for real run
    for device_id in t3k_device_mesh.get_device_ids():
        tt_lib.device.DumpDeviceProfiler(t3k_device_mesh.get_device(device_id))

    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("prefill perf run")
    profiler.clear()
    profiler.start(f"e2e_prefill_{batch_size}_users")
    run_inference_prefill(tt_model, model_args, prefill_seq_len, t3k_device_mesh, pt_prefill_input, batch_size)
    profiler.end(f"e2e_prefill_{batch_size}_users")
    profiler.print(units="ms")
    prefill_time = profiler.get("prepare_inputs_and_prefill_32_users")

    # Decode (Run 1 warmup iteration before running 1 perf iteration)
    generation_start_pos = prefill_seq_len
    generation_length = 1

    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("decode warmup")
    profiler.clear()
    profiler.start(f"e2e_decode_warmup")
    run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"e2e_decode_warmup")
    profiler.print(units="ms")
    decode_warmup_time = profiler.get("model_run_for_inference_0")

    # Profiler dump, ready for real run
    for device_id in t3k_device_mesh.get_device_ids():
        tt_lib.device.DumpDeviceProfiler(t3k_device_mesh.get_device(device_id))

    if not os.getenv("CI") == "true":  # Enable tracy signpost support in local runs only
        signpost("decode perf run")
    profiler.clear()
    profiler.start(f"e2e_decode_{batch_size}_users")
    run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"e2e_decode_{batch_size}_users")

    profiler.end("e2e_decode_1_iteration")
    profiler.print(units="ms")
    decode_time = profiler.get("model_run_for_inference_0")

    comment = f"seqlen_len={prefill_seqlen}_num_layers={model_args.n_layers}"

    prep_perf_report(
        model_name=f"Mixtral8x7B_prefill_{comment}",
        batch_size=model_args.max_batch_size,
        inference_and_compile_time=prefill_warmup_time + decode_warmup_time,
        inference_time=prefill_time + decode_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )


def run_inference_prefill(tt_model, model_args, prefill_seq_len, device_mesh, pt_prefill_input, batch_size):
    # Get rotary matrix
    profiler.start("prefill_prepare_rot_matrices")
    rot_mats_prefill = get_prefill_rot_mat(
        model_args.head_dim, model_args.max_seq_len, device_mesh, seq_len=prefill_seq_len
    )

    head_dim = model_args.dim // model_args.n_heads
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    profiler.end("prefill_prepare_rot_matrices")

    profiler.start(f"prepare_inputs_and_prefill_{batch_size}_users")
    # Prefill all users, one by one
    for batch_id in range(batch_size):
        prefill_input, attn_mask, _ = prepare_inputs_ttnn_prefill(
            pt_prefill_input[batch_id],
            device_mesh,
        )
        # profiler.start(f"e2e_prefill_user_{batch_id}")
        tt_out = tt_model(
            prefill_input,
            0,  # Start position
            0,  # Current position
            attn_mask,
            rot_mats_prefill,
            transformation_mats,
            user_id=batch_id,
            mode="prefill",
        )
        # profiler.end(f"e2e_prefill_user_{batch_id}")

    profiler.end(f"prepare_inputs_and_prefill_{batch_size}_users")


def run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length):
    seqlen = 1  # Generating one token per user at a time
    batch = tt_model.args.max_batch_size

    profiler.start(f"torch_embed_initial")
    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    profiler.end(f"torch_embed_initial")

    for i in range(generation_length):
        pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
        start_pos = generation_start_pos + i
        current_pos = start_pos % tt_model.args.sliding_window

        profiler.start(f"prepare_inputs_for_inference_{i}")
        decode_input, attn_mask = prepare_inputs_ttnn(
            pt_decode_input,
            tt_model.args.dim,
            start_pos,
            tt_model.args,
            tt_model.device_mesh,
        )
        profiler.end(f"prepare_inputs_for_inference_{i}")

        # Run TT model
        profiler.start(f"model_run_for_inference_{i}")
        profiler.start(f"python_dispatch_for_inference_{i}")
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask)
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
        # tt_decode_input = embd(tt_token_batch).view(batch, seqlen, -1)
        profiler.end(f"torch_argmax_and_embed_{i}")

        profiler.start(f"deallocate_tt_tensors_{i}")

        tt_out.deallocate(force=True)
        profiler.end(f"deallocate_tt_tensors_{i}")
