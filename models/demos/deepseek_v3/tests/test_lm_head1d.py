# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import gc
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.lm_head1d import LMHead1D
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)

TEST_CHECK_ITERS = 100
CI_ACTIVE = os.getenv("CI") == "true"
_CI_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE,
    reason="CI runs traced coverage only.",
)


class DeepseekV3LMHead(nn.Module):
    """
    Language model head for Deepseek V3.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 10485760},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, batch_size_per_row",
    [
        ("decode", 32),
        ("prefill", 1024),
    ],
)
@pytest.mark.parametrize(
    "trace_mode",
    [
        pytest.param(False, marks=_CI_SKIP_MARK, id="eager"),
        pytest.param(True, id="tracing"),
    ],
)
@pytest.mark.parametrize(
    "use_real_weights",
    [
        pytest.param(True, id="real_weights"),
        pytest.param(False, marks=_CI_SKIP_MARK, id="random_weights"),
    ],
)
@pytest.mark.requires_device(["TG", "DUAL", "QUAD"])
def test_forward_pass(
    mode: str,
    batch_size_per_row: int,
    trace_mode: bool,
    use_real_weights: bool,
    hf_config: Any,
    mesh_device: ttnn.Device,
    ccl: CCL,
    cache_path: Path,
    force_recalculate_weight_config: bool,
    set_deterministic_env: Any,
):
    if trace_mode and mode != "decode":
        pytest.skip("Tracing is only supported for decode mode.")

    reference_model = DeepseekV3LMHead(hf_config).eval()
    if use_real_weights:
        state_dict = sub_state_dict(reference_model.state_dict(), "lm_head.")
    else:
        random_state_dict = {}
        for name, tensor in reference_model.state_dict().items():
            if torch.is_floating_point(tensor):
                random_state_dict[name] = torch.randn_like(tensor) * 0.02
            else:
                random_state_dict[name] = torch.zeros_like(tensor)
        reference_model.load_state_dict(random_state_dict)
        state_dict = sub_state_dict(random_state_dict, "lm_head.")
    batch_size = batch_size_per_row * mesh_device.shape[0]
    torch_input = torch.randn(1, 1, batch_size, hf_config.hidden_size)
    reference_output = reference_model(torch_input)

    weight_cache_root = cache_path if use_real_weights else cache_path / "random_weights"
    weight_config = get_test_weight_config(
        LMHead1D,
        hf_config,
        (state_dict,),
        weight_cache_root,
        mesh_device,
        force_recalculate_weight_config or not use_real_weights,
    )
    model_config = get_model_config(LMHead1D, mode, mesh_device)
    model_state = LMHead1D.create_state(mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    def make_tt_input() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        tt_base = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=mesh_device.shape),
            dtype=ttnn.bfloat16,
            memory_config=run_config["input_memory_config"],
            layout=ttnn.TILE_LAYOUT,
        )
        tt_base = ttnn.to_memory_config(tt_base, run_config["input_memory_config"])
        # LMHead1D deallocates input internally; keep a base view alive to preserve the buffer.
        tt_view = ttnn.view(tt_base, tt_base.shape)
        return tt_base, tt_view

    def to_torch_output(tt_output: ttnn.Tensor) -> torch.Tensor:
        expected_output_memory_config = run_config["output_memory_config"]
        actual_output_memory_config = tt_output.memory_config()
        assert (
            actual_output_memory_config == expected_output_memory_config
        ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

        tt_output_shape = tt_output.shape
        logger.info(f"tt_output shape: {tt_output_shape}")

        # Get seq_len from dimension 2
        seq_len = tt_output_shape[2]

        if seq_len > 16384:  # Only apply chunking if seq_len > 16k tokens
            logger.info("running ttnn.to_torch with chunking")
            # Introduce chunking because to_torch isn't compatible for large tensors >4GB
            actual_batch_dim = tt_output_shape[2]
            chunk_size = 16384
            chunks = []

            for start_idx in range(0, actual_batch_dim, chunk_size):
                end_idx = min(start_idx + chunk_size, actual_batch_dim)
                logger.info(f"Processing chunk {start_idx}:{end_idx} of {actual_batch_dim}")

                # Slice the tensor along the batch dimension
                tt_chunk = ttnn.slice(
                    tt_output,
                    [0, 0, start_idx, 0],
                    [tt_output_shape[0], tt_output_shape[1], end_idx, tt_output_shape[3]],
                )

                # Convert chunk to torch
                chunk_torch = ttnn.to_torch(
                    tt_chunk,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
                )

                # Deallocate the TTNN chunk tensor immediately after conversion
                ttnn.deallocate(tt_chunk)
                del tt_chunk

                chunks.append(chunk_torch)

                # Periodic garbage collection to help free memory
                if (start_idx // chunk_size) % 2 == 1:
                    gc.collect()

            # Deallocate the original output tensor as all chunks have been processed
            ttnn.deallocate(tt_output)
            del tt_output
            gc.collect()

            # Concatenate all chunks along the batch dimension
            tt_output_torch = torch.cat(chunks, dim=-2)

            # Clear the chunks list to free memory
            del chunks
            gc.collect()

            logger.info("finished ttnn.to_torch with chunking")
        else:
            # For seq_len <= 16k, use regular to_torch without chunking
            logger.info("running ttnn.to_torch without chunking (seq_len <= 16k)")
            tt_output_torch = ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
            )
            # Deallocate the TTNN output tensor after conversion
            ttnn.deallocate(tt_output)
            del tt_output
            gc.collect()

        logger.info(f"tt_output_torch shape: {tt_output_torch.shape}")
        return tt_output_torch

    if trace_mode:
        # Iteration 0: eager compile run (not traced)
        warmup_base, warmup_input = make_tt_input()
        tt_output = run_module_forward(LMHead1D, mode, warmup_input, run_config)
        ttnn.synchronize_device(mesh_device)
        tt_output_torch = to_torch_output(tt_output)
        assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
        ttnn.deallocate(warmup_base)

        # Reset CCL semaphore counters before trace capture
        ccl.reset_sem_counters()

        trace_input_base, trace_input = make_tt_input()
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = run_module_forward(LMHead1D, mode, trace_input, run_config)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for _ in range(TEST_CHECK_ITERS - 1):
            ttnn.execute_trace(mesh_device, trace_id, blocking=True)
        ttnn.synchronize_device(mesh_device)

        tt_output_torch = to_torch_output(trace_output)
        assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(trace_input_base)
    else:
        for iter_idx in range(TEST_CHECK_ITERS):
            tt_base, tt_input = make_tt_input()
            tt_output = run_module_forward(LMHead1D, mode, tt_input, run_config)
            ttnn.synchronize_device(mesh_device)
            if iter_idx in (0, TEST_CHECK_ITERS - 1):
                tt_output_torch = to_torch_output(tt_output)
                assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
            else:
                ttnn.deallocate(tt_output)
            ttnn.deallocate(tt_base)

    gc.collect()
