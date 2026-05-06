# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
16-stage single-pod Blitz pipeline test using FakeMoeDecoderStage in place
of the 10 MoE decoder stages (stages 4–13).

The original test (test_single_pod_pipeline.py) currently fails inside
MoEDecoderStage.setup with:
  TT_FATAL: Tensor must be sharded to automatically create a CBDescriptor
when fed synthetic weights. This is a known issue in the MoE op's synthetic
path (see FAKE_MOE_PLAN.md §"Goal").

This test swaps those stages with a FakeMoeDecoderStage (a thin
PassthroughStage(ACTIVATION) factory — see _fake_moe_helpers.py) so the
pipeline framework — sockets, FIFO sizing, kernel scaffolding,
fabric_router_config, slow dispatch, mesh-graph descriptor — runs
end-to-end without the broken MoeOp.

The MoE-shaped CCL traffic itself is validated separately by
test_fake_moe_traffic.py::test_fake_moe_chain_4x2_single_pod (Phase 2
Tier 3); the goal here is end-to-end pipeline reachability, not extra CCL
validation.

Launch (16 ranks under tt-run, single-pod blitz_decode rank-binding):
  See /tmp/run_fake_moe_tier3.sh for the working invocation. Substitute
  the test target with:
    test_single_pod_pipeline_fake_moe.py::test_single_pod_pipeline_fake_moe
"""

from __future__ import annotations

from typing import Callable

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.exabox.single_pod._fake_moe_helpers import (
    make_fake_lm_head_stage_factory,
    make_fake_moe_decoder_stage_factory,
    make_synthetic_embedding_weights,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.exabox.single_pod._vendored.pipeline import (
    PipelineConfiguration,
    create_fabric_router_config,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.exabox.single_pod._vendored.stage import (
    EmbeddingStage,
    PassthroughPayload,
    PassthroughStage,
    StageKind,
)

VOCAB_SIZE = 129280
SINGLE_POD_NUM_PROCS = 16


def _build_fake_moe_single_pod_config() -> PipelineConfiguration:
    """16-stage single-pod with stages 4–13 replaced by FakeMoeDecoderStage.

    Stages 1–3 (Dense decoder) are also swapped to PassthroughStage(ACTIVATION)
    because DenseDecoderStage's setup builds DecoderBlock program contexts that
    share the synthetic-weight failure path with MoEDecoderStage. Keeping
    Embedding (stage 0), LMHead (stage 14), Token-passthrough (stage 15) as-is
    so the host-side I/O contract (token in → token out) still flows.
    """
    fake_moe = make_fake_moe_decoder_stage_factory()
    fake_lm_head = make_fake_lm_head_stage_factory()
    activation_passthrough: Callable[[ttnn.MeshDevice], StageKind] = lambda d: PassthroughStage(
        PassthroughPayload.ACTIVATION
    )

    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {
        0: lambda d: EmbeddingStage(make_synthetic_embedding_weights(d)),
        1: activation_passthrough,
        2: activation_passthrough,
        3: activation_passthrough,
        # Stages 4..13 are the real MoE decoder slots; we substitute fake-MoE.
        **{i: fake_moe for i in range(4, 14)},
        # LMHead replaced with a no-compute stub (same socket sizing); the
        # real LMHead's compute kernel deadlocks teardown when no token is
        # driven through (see _fake_moe_helpers.make_fake_lm_head_stage_factory).
        14: fake_lm_head,
        15: lambda d: PassthroughStage(PassthroughPayload.TOKEN),
    }
    return PipelineConfiguration(stage_factories)


# Only one parametrization for now: the second case (prompt_len=8) hangs during
# fixture teardown / re-initialization on the QUAD_BH cluster (verified
# 2026-04-30). After the first case PASSES on all 16 ranks and the mesh device
# is closed, the second case's bh_2d_mesh_device_context re-open never returns.
# The first case alone is sufficient to demonstrate that the FakeMoeDecoderStage
# substitution lets the 16-stage pipeline framework run end-to-end without the
# broken MoeOp synthetic-weight path.
@pytest.mark.parametrize(
    "prompt_len, max_new_tokens",
    [
        (1, 4),
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
            "worker_l1_size": 1431568,
        }
    ],
    indirect=True,
)
@pytest.mark.timeout(1800)
def test_single_pod_pipeline_fake_moe(
    mesh_device: ttnn.MeshDevice,
    prompt_len: int,
    max_new_tokens: int,
) -> None:
    if not is_slow_dispatch():
        pytest.skip("Single-pod pipeline requires slow dispatch (TT_METAL_SLOW_DISPATCH_MODE=1)")

    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != SINGLE_POD_NUM_PROCS:
        pytest.skip(f"Single-pod pipeline requires {SINGLE_POD_NUM_PROCS} processes, got {num_procs}")

    my_mesh_id = mesh_device.get_system_mesh_id()
    logger.info(
        "Building fake-MoE single-pod pipeline (rank {}/{}, prompt_len={}, max_new_tokens={})",
        my_mesh_id,
        num_procs,
        prompt_len,
        max_new_tokens,
    )

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    config = _build_fake_moe_single_pod_config()
    assert (
        config.num_stages == SINGLE_POD_NUM_PROCS
    ), f"Expected {SINGLE_POD_NUM_PROCS} pipeline stages, got {config.num_stages}"

    pipeline = config.build_pipeline(mesh_device)
    assert pipeline.my_mesh_id == my_mesh_id

    try:
        pipeline.setup_and_run()
        # Reaching this point on every rank is the primary acceptance signal.
        # (DeepSeekV3 host-side decoding loop omitted — this test stops at
        # the framework-reachability assertion. A follow-up can wire in
        # ModelPipeline-style prefill+decode if desired.)
        logger.info("[rank={}] fake-MoE pipeline setup_and_run completed", my_mesh_id)
    finally:
        pipeline.terminate()
