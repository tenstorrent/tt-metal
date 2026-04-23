# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
16-stage single-pod Blitz pipeline hardware smoke test.

Launched under tt-run on a Blackhole pod (4 hosts x 4 slices = 16 MPI ranks,
8 BH devices per rank). Each rank builds its own stage of the single-pod pipeline
(Embedding -> 3x Dense -> 10x MoE -> LMHead -> Passthrough) with
SyntheticWeightProvider, runs the 4-phase setup_and_run, and rank 0 drives a
short prefill + decode loop. Non-zero ranks rendezvous at Pipeline.terminate().

Uses the local conftest.py mesh_device override (bh_2d_mesh_device_context-backed)
so the FABRIC_2D_TORUS_Y + fabric_router_config + worker_l1_size device_params
reach set_fabric correctly.
"""

from __future__ import annotations

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config

VOCAB_SIZE = 129280
SINGLE_POD_NUM_PROCS = 16


@pytest.mark.parametrize(
    "prompt_len, max_new_tokens",
    [
        (1, 4),
        (8, 4),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@pytest.mark.timeout(1800)
def test_single_pod_pipeline_setup_and_decode(
    mesh_device: ttnn.MeshDevice,
    prompt_len: int,
    max_new_tokens: int,
) -> None:
    if not is_slow_dispatch():
        pytest.skip("ModelPipeline requires slow dispatch mode (TT_METAL_SLOW_DISPATCH_MODE=1)")

    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != SINGLE_POD_NUM_PROCS:
        pytest.skip(f"Single-pod pipeline requires {SINGLE_POD_NUM_PROCS} distributed processes, got {num_procs}")

    my_mesh_id = mesh_device.get_system_mesh_id()
    logger.info(
        "Building 16-stage single-pod ModelPipeline (rank {}/{}), prompt_len={}, max_new_tokens={}",
        my_mesh_id,
        num_procs,
        prompt_len,
        max_new_tokens,
    )

    model_pipeline = ModelPipeline(mesh_device=mesh_device, weights_mode="synthetic")
    try:
        pipeline_config = model_pipeline.pipeline._pipeline_config
        assert len(pipeline_config) == SINGLE_POD_NUM_PROCS + 1, (
            f"Expected {SINGLE_POD_NUM_PROCS + 1} pipeline config entries "
            f"(16 stages + loopback), got {len(pipeline_config)}"
        )
        assert model_pipeline.pipeline.my_mesh_id == my_mesh_id

        if my_mesh_id == 0:
            assert model_pipeline.model is not None
            prompt_token_ids = list(range(prompt_len))
            logger.info("Rank 0 running inference on {} prompt tokens", prompt_len)
            generated_tokens = model_pipeline.run_inference(
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=max_new_tokens,
                return_generated_tokens=True,
            )
            assert generated_tokens is not None
            assert (
                len(generated_tokens) == max_new_tokens
            ), f"Expected {max_new_tokens} generated tokens, got {len(generated_tokens)}"
            for i, tok in enumerate(generated_tokens):
                assert isinstance(tok, int), f"Token {i} is {type(tok).__name__}, expected int"
                assert 0 <= tok < VOCAB_SIZE, f"Token {i}={tok} outside [0, {VOCAB_SIZE})"
            logger.info("Rank 0 decode complete; generated {} tokens", len(generated_tokens))
        else:
            assert model_pipeline.model is None
    finally:
        model_pipeline.terminate()
