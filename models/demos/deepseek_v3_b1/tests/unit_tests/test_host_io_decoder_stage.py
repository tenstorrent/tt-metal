# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for HostIoDecoderStage.

Runs a single ``HostIoDecoderStage`` (combined H2D + multi-upstream D2H pipeline
block) on a Blackhole 2D submesh, pushes tokens through the H2D socket, and reads
the assembled D2H pages back to host. Verifies that the decoder MoE final output
emerging from the multi-upstream D2H sender matches the standalone DecoderBlock
golden reference.

Subsequent steps (token I/O loop, golden comparison, teardown) are added on top
of the scaffolding in this file. Step 0 (this revision) only sets up fixtures /
skip guards and confirms the test plumbing wires up the same way as
``test_decoder_block.test_decoder``.
"""

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.decoder_stage import HostIoDecoderStage
from models.demos.deepseek_v3_b1.demo.stage import ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES, StageContext
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import StageMetadata
from models.demos.deepseek_v3_b1.model import InputField, TokenType, parse_output_page, to_spec_input
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D
from models.demos.deepseek_v3_b1.model_dimensions import RoutedExpert
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import ROUTED_EXPERT_LAYER_IDX
from models.demos.deepseek_v3_b1.weights.prepare import DeepSeekV3EmbeddingLayerWeights, prepare_moe_layer_weights

# Embedding vocabulary used by this test. Kept small (well below D.VOCAB_SIZE = 129280)
# so we don't spend time uploading a >1.8 GB embedding table on every test run.
TEST_VOCAB_SIZE = 32


@dataclass
class _SinglePipelineStage:
    """Duck-typed stand-in for ``ttnn.experimental.BlitzDecodePipelineStage``.

    ``generate_blitz_decode_pipeline`` does not handle ``num_meshes == 1`` (it OOBs on
    ``hops[0]`` for the no-loopback path) and the C++ ``BlitzDecodePipelineStage`` binding
    is read-only, so we construct the 1-stage descriptor in Python. ``HostIoDecoderStage``
    and ``PipelineBlock._init_combined_h2d_d2h_stage`` only read ``entry_node_coord`` and
    ``exit_node_coord`` from this object.
    """

    entry_node_coord: ttnn.MeshCoordinate
    exit_node_coord: ttnn.MeshCoordinate


@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
@pytest.mark.parametrize("position_id", [0])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1431568,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "enable_routing, use_hardcoded_expert_index, num_routed_experts",
    [
        (True, False, 8),
    ],
    ids=["routing_8e"],
)
@pytest.mark.parametrize("decoder_layer_idx", [ROUTED_EXPERT_LAYER_IDX])
@pytest.mark.parametrize("num_slots", [8])
@pytest.mark.requires_grid_size((13, 10))
def test_host_io_decoder_stage(
    bh_2d_mesh_device,
    device_params,
    mesh_rows,
    mesh_cols,
    max_seq_len,
    position_id,
    enable_routing,
    use_hardcoded_expert_index,
    num_routed_experts,
    decoder_layer_idx,
    num_slots,
    get_reference_model_state_dict,
):
    """End-to-end test of HostIoDecoderStage: H2D → decoder → multi-upstream D2H."""
    torch.manual_seed(0)
    num_devices = mesh_rows * mesh_cols
    logger.info(f"Number of devices: {num_devices}")

    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than available")
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode (H2D/D2H sockets require slow dispatch)")

    logger.info("Enabling asynchronous slow dispatch on the parent mesh device")

    logger.info(f"Creating {mesh_rows}x{mesh_cols} submesh ({num_devices} devices)")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    ttnn.enable_asynchronous_slow_dispatch(submesh)

    logger.info(f"Loading reference model state dict for layer {decoder_layer_idx} (random weights)...")
    state_dict = get_reference_model_state_dict(
        layer_idx=decoder_layer_idx,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=num_routed_experts,
        random_weights=True,
    )
    logger.info(f"Uploading MoE layer weights to submesh (num_routed_experts={num_routed_experts})...")
    layer_weights = prepare_moe_layer_weights(
        submesh,
        state_dict,
        ROUTED_EXPERT_LAYER_IDX,
        num_routed_experts=num_routed_experts,
        move_to_device=True,
    )
    logger.info("MoE layer weights uploaded")

    # Bypass prepare_embedding_weights — it asserts the full D.VOCAB_SIZE x D.HIDDEN_SIZE
    # shape, which would force a multi-GB upload. We only need a small token-id range here,
    # so we build the same DRAM-interleaved row-major bf16 tensor that HostInterface expects
    # directly and wrap it in DeepSeekV3EmbeddingLayerWeights.
    logger.info(f"Building random {TEST_VOCAB_SIZE}x{D.HIDDEN_SIZE} bf16 embedding...")
    torch_embedding = torch.randn(TEST_VOCAB_SIZE, D.HIDDEN_SIZE, dtype=torch.bfloat16)
    logger.info("Uploading embedding to DRAM (replicated across submesh)...")
    ttnn_embedding = ttnn.from_torch(
        torch_embedding,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    embedding_weights = DeepSeekV3EmbeddingLayerWeights(embedding=ttnn_embedding)
    logger.info("Embedding uploaded")

    # Single-stage pipeline (num_procs == 1, no loopback). Coords match
    # test_decoder_block.test_decoder: broadcast source = (1, 0), reduce-to-one root = (1, 1).
    logger.info("Building single-stage pipeline_config (entry=(1,0), exit=(1,1)) and StageContext")
    pipeline_config = [
        _SinglePipelineStage(
            entry_node_coord=ttnn.MeshCoordinate(1, 0),
            exit_node_coord=ttnn.MeshCoordinate(1, 1),
        )
    ]
    stages_metadata = {0: StageMetadata(rank=0, mesh_id=0)}
    ctx = StageContext(
        mesh_device=submesh,
        pipeline_config=pipeline_config,
        my_stage_idx=0,
        stages_metadata=stages_metadata,
    )

    logger.info("Instantiating HostIoDecoderStage")
    stage = HostIoDecoderStage(
        weights=layer_weights,
        embedding_weights=embedding_weights,
        layer_idx=decoder_layer_idx,
        # Setup-time metadata only — position_id seeds the initial KV-cache state; the
        # runtime per-token slot_id / position_id come from each H2D page below.
        metadata=DeepseekMetadata(position_id=position_id),
        max_seq_len=max_seq_len,
        num_slots=num_slots,
        # persistent_mode=True dispatches the decoder kernel once and advances per-token
        # via the next_iter / termination semaphores managed by PersistentLoop. Teardown
        # in step 11 will need a push_dummy_token + drain_dummy_output to wake the loop
        # one last iteration so it observes the termination semaphore.
        persistent_mode=True,
        # device_params["fabric_config"] is FABRIC_2D_TORUS_X for this test, matching the
        # is_torus=True default the production MoE/Dense decoder stages use.
        is_torus=True,
        is_moe=True,
        num_routed_experts=num_routed_experts,
        use_hardcoded_expert_index=use_hardcoded_expert_index,
        enable_routing=enable_routing,
    )

    logger.info("Building PipelineBlock (combined H2D + multi-upstream D2H branch)")
    pipeline_block = stage.create_pipeline_block(ctx)
    logger.info("Running stage.setup() — allocating decoder tensors and program context")
    stage.setup(ctx, pipeline_block)
    logger.info("Stage setup complete")

    logger.info("Dispatching H2D + multi-upstream D2H persistent kernels via pipeline_block.run()")
    pipeline_block.run()
    logger.info("Dispatching persistent decoder compute via stage.launch_compute()")
    stage.launch_compute(ctx, pipeline_block)
    logger.info("All persistent kernels dispatched")

    # Sweep the test vocab for every slot. Outer loop = slot_id; inner loop = position_id
    # restarting at 0 for each user. Each iteration uses token_id = position_id, mirroring
    # decode-step inputs from demo/model_pipeline.py (TokenType.BASE, prefill_token_id=-1,
    # temperature=0.6, top_k=1, probability_mass_threshold=1.0).
    metadata_bytes = DeepseekMetadata.aligned_size_bytes()  # 256
    page_size_datums = metadata_bytes // 4  # 64 uint32 words on the H2D page
    metadata_bf16_count = metadata_bytes // dtype_size(ttnn.bfloat16)  # 128 bf16 elems on D2H tail
    out_words = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES // dtype_size(ttnn.bfloat16)
    logger.info(
        f"Starting sweep: {num_slots} slots x {TEST_VOCAB_SIZE} tokens "
        f"(page_size_datums={page_size_datums}, out_words={out_words})"
    )
    for slot_id in range(num_slots):
        for pos_id in range(TEST_VOCAB_SIZE):
            token_id = pos_id
            input_tensor = to_spec_input(
                token_id=token_id,
                prefill_token_id=-1,
                user_id=slot_id,
                position_id=pos_id,
                page_size_datums=page_size_datums,
                token_type=TokenType.BASE,
                temperature=0.6,
                top_k=1,
                probability_mass_threshold=1.0,
            )
            pipeline_block.write_token(input_tensor)

            torch_output = torch.zeros(1, out_words, dtype=torch.bfloat16)
            output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            pipeline_block.read_output(output_tensor)

            # The D2H page is `activation || DeepseekMetadata`. Extract the trailing
            # `metadata_bytes` bytes and reinterpret them as the 64-uint32 metadata page,
            # then parse via parse_output_page (which expects a ttnn.uint32 tensor).
            torch_full = ttnn.to_torch(output_tensor).flatten()
            metadata_words = (
                torch_full[-metadata_bf16_count:].contiguous().view(torch.int32).reshape(1, page_size_datums)
            )
            metadata_tensor = ttnn.from_torch(metadata_words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            parsed = parse_output_page(metadata_tensor)

            # parse_output_page exposes slot_id (InputField.USER_ID = idx 6) and the
            # output-side fields (token_0_*); the input fields position_id (idx 8) and
            # token_id (idx 7) need to be read off the same uint32 buffer directly.
            metadata_flat = metadata_words.flatten()
            actual_position_id = int(metadata_flat[InputField.POSITION_ID].item())
            actual_token_id = int(metadata_flat[InputField.TOKEN_ID].item())

            assert parsed.slot_id == slot_id, (
                f"slot={slot_id} pos={pos_id}: D2H metadata slot_id mismatch "
                f"(got {parsed.slot_id}, expected {slot_id})"
            )
            assert actual_position_id == pos_id, (
                f"slot={slot_id} pos={pos_id}: D2H metadata position_id mismatch "
                f"(got {actual_position_id}, expected {pos_id})"
            )
            assert actual_token_id == token_id, (
                f"slot={slot_id} pos={pos_id}: D2H metadata token_id mismatch "
                f"(got {actual_token_id}, expected {token_id})"
            )
            assert parsed.token_0_type == TokenType.BASE, (
                f"slot={slot_id} pos={pos_id}: D2H metadata token_0_type mismatch "
                f"(got {parsed.token_0_type}, expected TokenType.BASE = {TokenType.BASE})"
            )
            logger.info(f"slot={slot_id} pos={pos_id} token_id={token_id} write/read/parse OK")

    logger.info(f"HostIoDecoderStage sweep complete: {num_slots} slots x {TEST_VOCAB_SIZE} tokens")

    # Teardown: matches Pipeline.terminate() in demo/pipeline.py minus the multi-process
    # distributed barriers (we're single-process here).
    logger.info("Tearing down: signalling persistent decoder termination")
    stage.terminate(ctx, pipeline_block)
    logger.info("Pushing dummy token to wake decoder one last iteration")
    pipeline_block.push_dummy_token()
    logger.info("Draining dummy D2H output")
    pipeline_block.drain_dummy_output()
    logger.info("Terminating H2D + multi-upstream D2H kernels")
    pipeline_block.terminate()
    ttnn.synchronize_device(submesh)
    logger.info("HostIoDecoderStage teardown complete")

    # TODO(steps 10-11): golden comparison, teardown.
