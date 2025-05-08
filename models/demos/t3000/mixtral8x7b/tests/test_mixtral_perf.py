# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import ttnn
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh

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
from models.utility_functions import profiler


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@pytest.mark.model_perf_t3000
@pytest.mark.timeout(400)
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
    t3k_mesh_device,
    generation_start_pos,
    expected_compile_time,
    expected_inference_time,
    use_program_cache,
    reset_seeds,
    is_ci_env,
):
    if not is_ci_env:  # Enable tracy signpost support in local runs only
        from tracy import signpost

    dtype = ttnn.bfloat8_b

    batch_size = 32
    # Although in decode-only mode change the max seqlen to 16k to avoid KV cache running out of memory size
    max_seqlen = 16384

    # Can use dummy_weights=True correctness is not tested, but it is much slower
    model_args = TtModelArgs(t3k_mesh_device, dummy_weights=False, max_batch_size=batch_size, max_seq_len=max_seqlen)
    model_args.n_layers = 32

    # Clear global profiler state before starting measurements
    profiler.clear()

    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    prompts = ["Once"] * batch_size
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
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        start_pos_ids=[generation_start_pos] * model_args.max_batch_size,
        dtype=dtype,
        rotary_on_host=False,
    )
    profiler.end("TtMixtral_model_setup")

    # Call the function
    if not is_ci_env:  # Enable tracy signpost support in local runs only
        signpost("Model warmup")
    profiler.start(f"e2e_decode_compile")
    run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"e2e_decode_compile")
    profiler.print(units="ms")
    compile_and_iter_time = profiler.get("e2e_decode_compile")

    ttnn.DumpDeviceProfiler(t3k_mesh_device)

    if not is_ci_env:  # Enable tracy signpost support in local runs only
        signpost("Model perf run")
    profiler.clear()
    profiler.start(f"e2e_decode_inference")
    run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"e2e_decode_inference")
    profiler.print(units="ms")
    iter_time = profiler.get("e2e_decode_inference")

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
@pytest.mark.timeout(400)
@pytest.mark.parametrize(
    "prefill_seqlen, expected_compile_time, expected_inference_time",
    (
        (128, 80, 0.23),
        (1024, 80, 1.55),  # FIXME #12318
        (1024 * 2, 80, 5.6),  # FIXME #12318
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
    t3k_mesh_device,
    prefill_seqlen,
    expected_compile_time,
    expected_inference_time,
    use_program_cache,
    reset_seeds,
    is_ci_env,
):
    if not is_ci_env:  # Enable tracy signpost support in local runs only
        from tracy import signpost

    dtype = ttnn.bfloat8_b

    if prefill_seqlen >= 16 * 1024:
        seq_len = 32 * 1024  # Cap the sequence length to a max of 32k
        batch_size = 8
    elif prefill_seqlen >= 8 * 1024:
        seq_len = 16 * 1024
        batch_size = 16
    else:
        seq_len = (
            prefill_seqlen * 2
        )  # The prompts being used here have sligtly more tokens than the prefill seq. this accounts for that.
        batch_size = 32

    # Can use dummy_weights=True correctness is not tested, but it is much slower
    model_args = TtModelArgs(t3k_mesh_device, dummy_weights=False, max_batch_size=batch_size, max_seq_len=seq_len)
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
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        prompts,
        tokenizer,
        model_args,
        dtype,
        False,
        t3k_mesh_device,
        2,
    )

    profiler.end("preprocessing_inputs")

    pt_prefill_input = [embd(input_tokens_prefill_pt[b]).view(1, prefill_lens[b], -1) for b in range(batch_size)]

    profiler.start("Mixtral_model_setup")
    # Load TTNN model
    tt_model = TtTransformer(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
        start_pos_ids=decoding_pos,
    )
    profiler.end("TtMixtral_model_setup")

    # Prefill (run warmup for single user before running perf for all users)
    if not is_ci_env:  # Enable tracy signpost support in local runs only
        signpost("prefill warmup")
    profiler.clear()
    profiler.start(f"e2e_prefill_warmup")
    run_inference_prefill(tt_model, model_args, prefill_seqlen, t3k_mesh_device, pt_prefill_input, 1)
    profiler.end(f"e2e_prefill_warmup")
    profiler.print(units="ms")
    prefill_warmup_time = profiler.get("e2e_prefill_warmup")

    # Profiler dump, ready for real run
    ttnn.DumpDeviceProfiler(t3k_mesh_device)

    if not is_ci_env:  # Enable tracy signpost support in local runs only
        signpost("prefill perf run")
    profiler.clear()
    profiler.start(f"e2e_prefill_1_user")
    # Prefill a single user, as this will be the real-world usage
    prefill_out = run_inference_prefill(tt_model, model_args, prefill_seqlen, t3k_mesh_device, pt_prefill_input, 1)
    profiler.end(f"e2e_prefill_1_user")
    profiler.print(units="ms")
    prefill_time = profiler.get("e2e_prefill_1_user")

    # profile dump
    ttnn.DumpDeviceProfiler(t3k_mesh_device)

    # Decode (Run 1 warmup iteration before running 1 perf iteration)
    generation_start_pos = prefill_seqlen
    generation_length = 1

    if not is_ci_env:  # Enable tracy signpost support in local runs only
        signpost("decode warmup")
    profiler.clear()
    profiler.start(f"e2e_decode_warmup")
    run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"e2e_decode_warmup")
    profiler.print(units="ms")
    decode_warmup_time = profiler.get("e2e_decode_warmup")

    # Profiler dump, ready for real run
    ttnn.DumpDeviceProfiler(t3k_mesh_device)

    if not is_ci_env:  # Enable tracy signpost support in local runs only
        signpost("decode perf run")
    profiler.clear()
    profiler.start(f"e2e_decode_inference_{batch_size}_users")
    run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length)
    profiler.end(f"e2e_decode_inference_{batch_size}_users")
    profiler.print(units="ms")
    decode_time = profiler.get("e2e_decode_inference_32_users")

    comment = f"time_to_1st_token_seqlen={prefill_seqlen}_num_layers={model_args.n_layers}"

    prefill_time_to_first = prefill_time

    prep_perf_report(
        model_name=f"Mixtral8x7B_prefill_{comment}",
        batch_size=model_args.max_batch_size,
        inference_and_compile_time=prefill_warmup_time,
        inference_time=prefill_time_to_first,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )


def run_inference_prefill(tt_model, model_args, prefill_seqlen, mesh_device, pt_prefill_input, batch_size):
    # Get rotary matrix
    profiler.start("prefill_prepare_rot_matrices")
    rot_mats_prefill = get_prefill_rot_mat(
        model_args.head_dim, model_args.max_seq_len, mesh_device, seq_len=prefill_seqlen
    )

    head_dim = model_args.dim // model_args.n_heads
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    profiler.end("prefill_prepare_rot_matrices")

    # Prefill all users, one by one
    for batch_id in range(batch_size):
        profiler.start(f"e2e_prefill_prepare_inputs_{batch_id}")
        prefill_input, attn_mask, _ = prepare_inputs_ttnn_prefill(
            pt_prefill_input[batch_id],
            mesh_device,
        )
        profiler.end(f"e2e_prefill_prepare_inputs_{batch_id}")

        profiler.start(f"e2e_prefill_inference_{batch_id}")
        tt_out = tt_model(
            prefill_input,
            [0] * model_args.max_batch_size,
            attn_mask,
            rot_mats_prefill,
            transformation_mats,
            user_id=batch_id,
            mode="prefill",
        )
        profiler.end(f"e2e_prefill_inference_{batch_id}")

        # Device sync to get proper e2e timing
        profiler.start(f"e2e_prefill_inference_sync_{batch_id}")
        ttnn.synchronize_device(mesh_device)
        profiler.end(f"e2e_prefill_inference_sync_{batch_id}")


def run_inference_decode(tt_model, embd, encoded_prompts, generation_start_pos, generation_length):
    seqlen = 1  # Generating one token per user at a time
    batch = tt_model.args.max_batch_size

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    for i in range(generation_length):
        profiler.start(f"Decode_token_embedding_{i}")
        pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
        profiler.end(f"Decode_token_embedding_{i}")

        start_pos = generation_start_pos + i

        profiler.start(f"Decode_prepare_inputs_{i}")
        decode_input = prepare_inputs_ttnn(
            pt_decode_input,
            tt_model.args.dim,
            tt_model.mesh_device,
        )
        profiler.end(f"Decode_prepare_inputs_{i}")

        # Run TT model
        profiler.start(f"inference_decode_{i}")
        profiler.start(f"python_dispatch_for_inference_{i}")
        tt_out = tt_model(decode_input, [start_pos] * batch)
        profiler.end(f"python_dispatch_for_inference_{i}")

        # Convert ttnn tensor to torch tensor
        profiler.start(f"python_wait_for_inference_out_{i}")
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(tt_model.mesh_device, dim=0))[0]
            .squeeze(1)
            .view(32, seqlen, -1)
            .detach()
            .float()
        )[:batch, ...]

        profiler.end(f"inference_decode_{i}")
        profiler.end(f"python_wait_for_inference_out_{i}")

        profiler.start(f"torch_argmax_{i}")
        # Greedy decode the generated token and pass it back in, this is just a perf test
        tt_token_batch = tt_output_torch.squeeze().argmax(axis=-1)
        # tt_decode_input = embd(tt_token_batch).view(batch, seqlen, -1)
        profiler.end(f"torch_argmax_{i}")

        profiler.start(f"deallocate_tt_tensors_{i}")
        tt_out.deallocate(force=True)
        profiler.end(f"deallocate_tt_tensors_{i}")
