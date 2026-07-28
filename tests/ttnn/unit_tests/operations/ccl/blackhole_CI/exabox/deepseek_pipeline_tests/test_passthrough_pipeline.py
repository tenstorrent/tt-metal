# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Local fork of test_multi_host_pipeline.py::test_passthrough_pipeline_block.

Why we don't reuse the shared
``create_passthrough_pipeline_configuration`` helper from
``models/demos/deepseek_v3_b1/demo/pipeline.py``: that helper has two
hard-coded payloads that don't agree with each other on current main:

  * Passthrough chain (stages 1..N-1): defaults to
    ``PassthroughPayload.ACTIVATION``.
  * EmbeddingStage upstream / loopback edge: hard-coded
    ``loopback_payload=PassthroughPayload.ACTIVATION``.

Meanwhile PR #43389 hard-coded ``EmbeddingStage``'s **downstream**
socket size to ``ACTIVATION_W_TOKEN_META_FIFO_SIZE`` (~10× larger than
``ACTIVATION_FIFO_SIZE``). So:

  * stage 0 downstream → stage 1 upstream:
      ACTIVATION_W_TOKEN_META vs ACTIVATION → MISMATCH.
  * stage N-1 downstream → stage 0 upstream (fabric loopback):
      ACTIVATION_W_TOKEN_META vs ACTIVATION → MISMATCH.

Either failure trips the per-rank socket-size handshake check at
``tt_metal/distributed/mesh_socket_utils.cpp:153``::

  TT_FATAL: Mismatch in socket FIFO size during handshake.

We sidestep the helper and build the stage factories directly so every
D2D socket pair along the 16-stage ring (including the wraparound) uses
``PassthroughPayload.ACTIVATION_W_TOKEN_META`` end-to-end.

Cluster requirements: 16 ranks across 4 hosts (4 ranks/host) on a BH
Galaxy single pod, each rank seeing a (4, 2) submesh. Bootstrap is
handled by ``scripts/bootstrap_pipeline_dir.sh``.
"""

from __future__ import annotations

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.pipeline import (
    PipelineConfiguration,
    create_fabric_router_config,
)
from models.demos.deepseek_v3_b1.demo.stage import (
    EmbeddingStage,
    PassthroughPayload,
    PassthroughStage,
)
from models.demos.deepseek_v3_b1.demo.weight_provider import SyntheticWeightProvider
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import StageMetadata


def _build_pipeline_config(weight_provider, num_procs):
    """16-stage passthrough chain with consistent ACTIVATION_W_TOKEN_META payloads.

    Stage 0:        EmbeddingStage(loopback_payload=ACTIVATION_W_TOKEN_META)
    Stages 1..N-1:  PassthroughStage(ACTIVATION_W_TOKEN_META)

    Mirrors what create_passthrough_pipeline_configuration does, except every
    socket boundary (forward + loopback wraparound) uses the same payload.
    """
    payload = PassthroughPayload.ACTIVATION_W_TOKEN_META

    def stage_0(device):
        return EmbeddingStage(
            weight_provider.load_embedding(device),
            loopback_payload=payload,
        )

    factories = {0: stage_0}
    for i in range(1, num_procs):
        factories[i] = lambda _d, p=payload: PassthroughStage(p)
    return PipelineConfiguration(factories)


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize("deepseek_pipeline_mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_passthrough_pipeline_block(deepseek_pipeline_mesh_device):
    if not is_slow_dispatch():
        pytest.skip("Pipeline framework requires TT_METAL_SLOW_DISPATCH_MODE=1")

    ttnn.enable_asynchronous_slow_dispatch(deepseek_pipeline_mesh_device)

    num_procs = int(ttnn.distributed_context_get_size())

    config = _build_pipeline_config(SyntheticWeightProvider(), num_procs)

    # Pre-generate pipeline_config + stages_metadata at the test level so each
    # stage's PipelineBlock.__init__ skips its own internal regen of
    # generate_blitz_decode_pipeline(). Without this, stages whose LoopbackConfig
    # is no_loopback regenerate with initialize_loopback=False, producing a
    # placeholder exit_node_coord==entry_node_coord for the last stage that
    # trips the unique-fabric-node check at
    # tt_metal/distributed/experimental/blitz_decode_pipeline.cpp:262.
    # Canonical pattern from test_lm_head_sampling.py::test_persistent_mode_spec_decode.
    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(True)
    stages_metadata = {i: StageMetadata(rank=i, mesh_id=i) for i in range(num_procs)}

    pipeline = config.build_pipeline(
        deepseek_pipeline_mesh_device,
        stages_metadata=stages_metadata,
        pipeline_config=pipeline_config,
    )
    my_id = pipeline.my_mesh_id
    logger.info(f"[rank={my_id}] pipeline built, calling setup_and_run()")

    try:
        pipeline.setup_and_run()
        # Reaching this on every rank is the primary acceptance signal for the
        # framework smoke test. The upstream test additionally pushes
        # vocab_size tokens through and PCC-checks each one — that's a
        # correctness gate, not a framework gate, and we drop it here.
        logger.info(f"[rank={my_id}] setup_and_run() completed")
        pipeline.barrier()
    finally:
        pipeline.terminate()
