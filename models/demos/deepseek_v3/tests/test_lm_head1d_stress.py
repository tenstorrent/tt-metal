# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_HIFI4


def _read_env_int(name: str, default: int, *, min_value: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _read_mode() -> str:
    mode = os.getenv("DEEPSEEK_LM_HEAD_BURN_MODE", "decode").strip().lower()
    if mode not in ("decode", "prefill"):
        raise ValueError(f"DEEPSEEK_LM_HEAD_BURN_MODE must be 'decode' or 'prefill', got {mode}")
    return mode


@pytest.mark.timeout(28800)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.requires_device(["TG", "DUAL", "QUAD"])
def test_lm_head_linear_hifi4_burn(mesh_device: ttnn.Device, set_deterministic_env) -> None:
    """
    Op-level LM-head burn test:
    - Runs only one op (`ttnn.linear`) in a tight loop.
    - Uses LM-head style sharding across a 2D mesh.
    - Forces HiFi4 compute fidelity, independent of model-level settings.
    - Does not invoke CCL/all-gather.

    Environment knobs:
    - DEEPSEEK_LM_HEAD_BURN_MODE: decode|prefill (default: decode)
    - DEEPSEEK_LM_HEAD_BURN_TOKENS_PER_ROW (default: 32 for decode, 1024 for prefill)
    - DEEPSEEK_LM_HEAD_BURN_HIDDEN_SIZE (default: 7168)
    - DEEPSEEK_LM_HEAD_BURN_VOCAB_SIZE (default: 129280)
    - DEEPSEEK_LM_HEAD_BURN_ITERS (default: 25000)
    - DEEPSEEK_LM_HEAD_BURN_SYNC_EVERY (default: 1)
    - DEEPSEEK_LM_HEAD_BURN_LOG_EVERY (default: 100)
    """
    del set_deterministic_env

    mode = _read_mode()
    hidden_size = _read_env_int("DEEPSEEK_LM_HEAD_BURN_HIDDEN_SIZE", 7168)
    vocab_size = _read_env_int("DEEPSEEK_LM_HEAD_BURN_VOCAB_SIZE", 129280)
    default_tokens_per_row = 32 if mode == "decode" else 1024
    tokens_per_row = _read_env_int("DEEPSEEK_LM_HEAD_BURN_TOKENS_PER_ROW", default_tokens_per_row)
    num_iters = _read_env_int("DEEPSEEK_LM_HEAD_BURN_ITERS", 25000)
    sync_every = _read_env_int("DEEPSEEK_LM_HEAD_BURN_SYNC_EVERY", 1)
    log_every = _read_env_int("DEEPSEEK_LM_HEAD_BURN_LOG_EVERY", 100)

    num_rows, num_cols = mesh_device.shape
    if vocab_size % num_cols != 0:
        raise ValueError(f"DEEPSEEK_LM_HEAD_BURN_VOCAB_SIZE ({vocab_size}) must be divisible by mesh cols ({num_cols})")

    batch_size = tokens_per_row * num_rows
    io_memory_config = ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

    logger.info(
        "LM-head HiFi4 burn config: mode={} mesh={}x{} batch={} hidden={} vocab={} iters={} sync_every={}",
        mode,
        num_rows,
        num_cols,
        batch_size,
        hidden_size,
        vocab_size,
        num_iters,
        sync_every,
    )

    torch.manual_seed(2026)
    torch_input = torch.empty((1, 1, batch_size, hidden_size), dtype=torch.bfloat16).normal_(mean=0.0, std=0.5)
    torch_weight = torch.empty((hidden_size, vocab_size), dtype=torch.bfloat16).normal_(mean=0.0, std=0.5)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=io_memory_config,
    )
    tt_weight = ttnn.from_torch(
        torch_weight,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Compile + warmup once before the long burn loop.
    tt_output = ttnn.linear(
        tt_input,
        tt_weight,
        memory_config=io_memory_config,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI4,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(tt_output)
    del tt_output

    for iteration in range(1, num_iters + 1):
        tt_output = ttnn.linear(
            tt_input,
            tt_weight,
            memory_config=io_memory_config,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI4,
        )
        ttnn.deallocate(tt_output)
        del tt_output

        if iteration % sync_every == 0:
            ttnn.synchronize_device(mesh_device)

        if iteration % log_every == 0:
            logger.info("LM-head HiFi4 burn progress: {}/{}", iteration, num_iters)

    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_weight)
    del tt_input
    del tt_weight
    del torch_input
    del torch_weight
    gc.collect()
