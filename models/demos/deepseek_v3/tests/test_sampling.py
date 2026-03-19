#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_fabric_config, make_deepseek_sampling_args


def _make_lm_head_sharded_logits(torch_input, mesh_device):
    return ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )


def _extract_all_tokens(tt_out_tok, mesh_device, batch_size_per_row):
    composed = ttnn.to_torch(
        tt_out_tok,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=tuple(mesh_device.shape)),
    )
    if composed.ndim == 4:
        if tt_out_tok.shape[-2] == batch_size_per_row:
            tokens = composed[:, :, :, 0]
        elif tt_out_tok.shape[-1] == batch_size_per_row:
            tokens = composed[:, :, 0, :batch_size_per_row]
        else:
            tokens = composed
        tokens = tokens.reshape(-1)
    else:
        tokens = composed.reshape(-1)
    batch_size = batch_size_per_row * int(mesh_device.shape[0])
    return tokens[:batch_size].to(torch.int64)


def _sample_device_tokens(mesh_device, ccl, args, torch_input, user_params):
    batch_size = USERS_PER_ROW * int(mesh_device.shape[0])
    tt_input = _make_lm_head_sharded_logits(torch_input, mesh_device)
    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=ccl, enable_internal_trace=False)
    params = format_sampling_params(user_params, max_batch_size=batch_size)
    sampling.reset_sampling_params(params)
    sampling.reset_prompt_tokens(torch.zeros((USERS_PER_ROW, 1), dtype=torch.int64))
    sampling.reset_output_state(torch.zeros((USERS_PER_ROW, 1), dtype=torch.int64))
    sampling.seed_manager.reset_seed(params.seed, list(range(batch_size)))
    sampling.seed_manager.get_new_values()
    tt_tokens, _ = sampling.sample(tt_input, enable_trace=False)
    device_tokens = _extract_all_tokens(tt_tokens, mesh_device, USERS_PER_ROW)
    ttnn.deallocate(tt_tokens)
    ttnn.deallocate(tt_input)
    return device_tokens


@torch.no_grad()
@pytest.mark.parametrize(
    "sampling_params",
    [
        {"temperature": 0.0, "top_k": 32, "top_p": 0.00, "seed": 42},
        {"temperature": 0.0, "top_k": 32, "top_p": 0.95, "seed": 42},
        {"temperature": 1.0, "top_k": 1, "top_p": 0.00, "seed": 42},  # top-k=1 (always argmax)
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": get_fabric_config()}], indirect=True)
def test_deepseek_device_sampling_argmax_path(mesh_device, ccl, hf_config, device_params, sampling_params):
    vocab_size = int(hf_config.vocab_size)
    args = make_deepseek_sampling_args(mesh_device, vocab_size=vocab_size)
    batch_size = USERS_PER_ROW * int(mesh_device.shape[0])
    seed = int(sampling_params.get("seed", 0))
    torch.manual_seed(seed)
    torch_input = torch.randn(1, 1, batch_size, args.padded_vocab_size) * 0.01
    forced_tokens = torch.tensor([(u * 1237 + 31) % vocab_size for u in range(batch_size)], dtype=torch.int64)
    batch_indices = torch.arange(batch_size, dtype=torch.int64)
    torch_input[0, 0, batch_indices, forced_tokens] = 50.0
    if args.padded_vocab_size > vocab_size:
        torch_input[:, :, :, vocab_size:] = -float("inf")

    user_params = SamplingParams(
        temperature=[sampling_params["temperature"]] * batch_size,
        top_k=[sampling_params["top_k"]] * batch_size,
        top_p=[sampling_params["top_p"]] * batch_size,
        seed=[seed] * batch_size,
    )
    device_tokens = _sample_device_tokens(mesh_device, ccl, args, torch_input, user_params)

    assert device_tokens.numel() == batch_size, (
        f"Expected {batch_size} sampled tokens, got {device_tokens.numel()}. "
        "This usually indicates incorrect mesh token reconstruction."
    )
    assert torch.equal(
        device_tokens, forced_tokens
    ), "Device sampling generator produced tokens mismatch for LM-head sharded DeepSeek logits."


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"fabric_config": get_fabric_config()}], indirect=True)
@pytest.mark.parametrize("use_tracing", [False, True], ids=["no_trace", "trace_mode"])
def test_deepseek_device_sampling_stochastic_behavior(mesh_device, ccl, hf_config, device_params, use_tracing):
    vocab_size = int(hf_config.vocab_size)
    args = make_deepseek_sampling_args(mesh_device, vocab_size=vocab_size)
    batch_size = USERS_PER_ROW * int(mesh_device.shape[0])

    torch_input = torch.full((1, 1, batch_size, args.padded_vocab_size), -1e9, dtype=torch.float32)
    candidate_tokens = torch.tensor([3, 7, 11, 19], dtype=torch.int64)
    candidate_logits = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    torch_input[0, 0, :, candidate_tokens] = candidate_logits
    if args.padded_vocab_size > vocab_size:
        torch_input[:, :, :, vocab_size:] = -float("inf")

    num_samples = 100
    per_user_seeds = [1000 + u for u in range(batch_size)]
    user_params = SamplingParams(
        temperature=[1.0] * batch_size,
        top_k=[4] * batch_size,
        top_p=[0.95] * batch_size,
        seed=per_user_seeds,
    )

    tt_input = _make_lm_head_sharded_logits(torch_input, mesh_device)
    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=ccl, enable_internal_trace=use_tracing)
    params = format_sampling_params(user_params, max_batch_size=batch_size)
    sampling.reset_sampling_params(params)
    sampling.reset_prompt_tokens(torch.zeros((USERS_PER_ROW, 1), dtype=torch.int64))
    sampling.reset_output_state(torch.zeros((USERS_PER_ROW, 1), dtype=torch.int64))
    sampling.seed_manager.reset_seed(params.seed, list(range(batch_size)))

    sampled_tokens = []
    try:
        for _ in range(num_samples):
            sampling.seed_manager.get_new_values()
            tt_tokens, tt_log_probs = sampling.sample(tt_input, enable_trace=use_tracing)
            device_tokens = _extract_all_tokens(tt_tokens, mesh_device, USERS_PER_ROW)
            sampled_tokens.append(int(device_tokens[0].item()))
            # In trace mode, sampling reuses captured output tensors across iterations.
            # Deallocating those per-step breaks subsequent trace replays.
            if not use_tracing:
                ttnn.deallocate(tt_tokens)
                if tt_log_probs is not None:
                    ttnn.deallocate(tt_log_probs)
    finally:
        if use_tracing:
            # Release cached trace metadata/tensors before exiting the test.
            sampling.reset_trace()
        ttnn.deallocate(tt_input)

    candidate_set = set(candidate_tokens.tolist())
    sampled_set = set(sampled_tokens)
    assert sampled_set.issubset(
        candidate_set
    ), f"Sampled tokens outside candidate set. got={sorted(sampled_set)}, expected subset of {sorted(candidate_set)}"
    assert len(sampled_set) >= 2, (
        f"Only {len(sampled_set)} unique token(s) in {num_samples} samples; sampling may be stuck. "
        f"sampled_set={sorted(sampled_set)}"
    )
