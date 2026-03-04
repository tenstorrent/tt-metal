#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.tt_sampling import TTSampling
from models.demos.deepseek_v3.utils.config_dataclass import DeepseekSamplingArgs
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW

MAX_TOP_K = 32


def _make_deepseek_sampling_args(mesh_device, vocab_size: int):
    cluster_shape = tuple(mesh_device.shape)
    sampling_dp = int(cluster_shape[0])  # one sampling group per row
    num_tp = int(mesh_device.shape[1])
    per_device_vocab = int(math.ceil(vocab_size / num_tp))
    padded_per_device_vocab = int(math.ceil(per_device_vocab / ttnn.TILE_SIZE) * ttnn.TILE_SIZE)
    padded_vocab_size = padded_per_device_vocab * num_tp
    return DeepseekSamplingArgs(
        vocab_size=vocab_size,
        padded_vocab_size=padded_vocab_size,
        max_top_k=MAX_TOP_K,
        max_batch_size=USERS_PER_ROW,
        sampling_dp=sampling_dp,
        cluster_shape=cluster_shape,
    )


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


@torch.no_grad()
@pytest.mark.parametrize(
    "sampling_params",
    [
        # {"temperature": 0.0, "top_k": 32, "top_p": 0.00, "seed": 42},  # Greedy
        # {"temperature": 0.0, "top_k": 32, "top_p": 0.95, "seed": 42},  # Greedy (top_p ignored)
        {"temperature": 1.0, "top_k": 1, "top_p": 0.00, "seed": 42},  # top-k=1 (always argmax)
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_deepseek_greedy_sampling(sampling_params, mesh_device, ccl, hf_config, device_params):
    vocab_size = int(hf_config.vocab_size)
    args = _make_deepseek_sampling_args(mesh_device, vocab_size=vocab_size)
    batch_size = USERS_PER_ROW * int(mesh_device.shape[0])
    seed = int(sampling_params.get("seed", 0))

    # Create full LM-head-style logits [1, 1, batch_total, padded_vocab].
    # We force one clear winner per user to avoid precision/tie ambiguity.
    torch.manual_seed(seed)
    torch_input = torch.randn(1, 1, batch_size, args.padded_vocab_size) * 0.01
    forced_tokens = torch.tensor([(u * 9973 + 17) % vocab_size for u in range(batch_size)], dtype=torch.int64)
    batch_indices = torch.arange(batch_size, dtype=torch.int64)
    torch_input[0, 0, batch_indices, forced_tokens] = 50.0
    if args.padded_vocab_size > vocab_size:
        torch_input[:, :, :, vocab_size:] = -float("inf")

    # Reference is the explicitly forced per-user winner token.
    ref_tokens = forced_tokens

    tt_input = _make_lm_head_sharded_logits(torch_input, mesh_device)
    # Normalize greedy params to TT sampling runtime convention.
    temperature = sampling_params["temperature"]
    top_k = sampling_params["top_k"]
    top_p = sampling_params["top_p"]
    if temperature == 0.0:
        temperature = 1.0
        top_k = 1
        top_p = 0.0

    k = torch.full((batch_size,), int(top_k), dtype=torch.int32)
    p = torch.full((batch_size,), float(top_p), dtype=torch.float32)
    temp = torch.full((batch_size,), float(temperature), dtype=torch.float32)

    tt_sampling = TTSampling(args=args, mesh_device=mesh_device, tt_ccl=ccl, k=k, p=p, temp=temp)
    tt_out_tok, _ = tt_sampling(tt_input)
    logger.info(f"tt_out_tok.shape after sample: {tt_out_tok.shape}")
    device_tokens = _extract_all_tokens(tt_out_tok, mesh_device, USERS_PER_ROW)
    logger.info(f"device_tokens.shape after extract_all_tokens: {device_tokens.shape}")

    assert (device_tokens >= 0).all(), "Sampled token IDs must be non-negative"
    assert (device_tokens < vocab_size).all(), f"Sampled token IDs must be < {vocab_size}"
    assert torch.equal(device_tokens, ref_tokens), (
        "Argmax mismatch between DeepSeek device sampling and torch reference. " f"params={sampling_params}"
    )
    logger.info(f"DeepSeek greedy sampling passed for params={sampling_params}")
