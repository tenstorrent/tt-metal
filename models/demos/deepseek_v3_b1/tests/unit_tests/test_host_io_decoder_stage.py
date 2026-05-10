# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for HostIoDecoderStage with real DeepSeek V3 weights.

Runs a single ``HostIoDecoderStage`` (combined H2D + multi-upstream D2H pipeline
block) on a Blackhole 2D submesh, pushes tokens through the H2D socket, and reads
the assembled D2H pages back to host. The decoder is configured with the same
shape the production ``MoEDecoderStage`` uses (``num_routed_experts=256``,
``max_seq_len=128*1024``, ``num_slots=64``, ``persistent_mode=True``,
``is_torus=True``) and consumes real weights via ``CacheWeightProvider``.

The sweep covers a small subset of slots and positions (8 users x 32 token IDs)
and asserts that the input metadata fields (slot_id, position_id, token_id,
token_0_type) round-trip through the decoder to the D2H page tail.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.decoder_stage import HostIoDecoderStage
from models.demos.deepseek_v3_b1.demo.stage import ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES, StageContext
from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import StageMetadata
from models.demos.deepseek_v3_b1.model import InputField, TokenType, parse_output_page, to_spec_input
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import ROUTED_EXPERT_LAYER_IDX
from models.demos.deepseek_v3_b1.weights.prepare import NUM_ROUTED_EXPERTS

# Sweep dimensions. Outer loop walks NUM_USERS_TO_SWEEP slots; inner loop walks
# NUM_POSITIONS_PER_USER positions, with token_id == position_id. These are decoupled
# from the production-shaped num_slots parametrize so the sweep stays fast even when
# the KV cache is allocated for the full production slot count.
NUM_USERS_TO_SWEEP = 8
NUM_POSITIONS_PER_USER = 32

# Default model + cache paths. Either env var overrides the corresponding default.
_DEFAULT_HF_MODEL_PATH = "/mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized"
_DEFAULT_CACHE_PATH = str(Path.home() / ".cache")


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
# Production-shape decoder: max_seq_len=128*1024, num_slots=64 (matches MoEDecoderStage defaults).
@pytest.mark.parametrize("max_seq_len", [128 * 1024])
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
@pytest.mark.parametrize("decoder_layer_idx", [ROUTED_EXPERT_LAYER_IDX])
@pytest.mark.parametrize("num_slots", [64])
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(15000)
def test_host_io_decoder_stage(
    bh_2d_mesh_device,
    device_params,
    mesh_rows,
    mesh_cols,
    max_seq_len,
    position_id,
    decoder_layer_idx,
    num_slots,
    tmp_path,
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

    # Real weights via CacheWeightProvider. Either env var overrides the default;
    # if the resolved HF model path doesn't exist we skip rather than fail loudly.
    hf_model_path = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", _DEFAULT_HF_MODEL_PATH))
    cache_path = Path(os.getenv("DEEPSEEK_V3_CACHE_PATH", _DEFAULT_CACHE_PATH))
    if not hf_model_path.exists():
        pytest.skip(f"HF model path does not exist: {hf_model_path}")
    logger.info(f"Using HF model path: {hf_model_path}")
    logger.info(f"Using cache path: {cache_path}")
    provider = CacheWeightProvider(cache_path=cache_path, model_path=hf_model_path)

    logger.info(f"Loading real MoE layer {decoder_layer_idx} weights via CacheWeightProvider...")
    layer_weights = provider.load_moe_layer(layer_id=decoder_layer_idx, device=submesh)
    logger.info("MoE layer weights ready")

    logger.info("Loading real embedding via CacheWeightProvider (full VOCAB_SIZE x HIDDEN_SIZE)...")
    embedding_weights = provider.load_embedding(device=submesh)
    logger.info("Embedding ready")

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

    # Production-shape decoder, matching MoEDecoderStage's defaults:
    #   num_routed_experts = NUM_ROUTED_EXPERTS (256), enable_routing = True,
    #   use_hardcoded_expert_index = False, persistent_mode = True, is_torus = True.
    # max_seq_len and num_slots come from the parametrize (also production defaults).
    logger.info("Instantiating HostIoDecoderStage with production-shape config")
    stage = HostIoDecoderStage(
        weights=layer_weights,
        embedding_weights=embedding_weights,
        layer_idx=decoder_layer_idx,
        # Setup-time metadata only — position_id seeds the initial KV-cache state; the
        # runtime per-token slot_id / position_id come from each H2D page below.
        metadata=DeepseekMetadata(position_id=position_id),
        max_seq_len=max_seq_len,
        num_slots=num_slots,
        persistent_mode=True,
        is_torus=True,
        is_moe=True,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        use_hardcoded_expert_index=False,
        enable_routing=True,
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

    # Sweep a small subset of the (slot, position) space. Outer loop = slot_id; inner
    # loop = position_id restarting at 0 for each user. Each iteration uses
    # token_id = position_id, mirroring decode-step inputs from demo/model_pipeline.py
    # (TokenType.BASE, prefill_token_id=-1, temperature=0.6, top_k=1,
    # probability_mass_threshold=1.0). Sweep bounds are decoupled from the production
    # num_slots so the iteration count stays small.
    metadata_bytes = DeepseekMetadata.aligned_size_bytes()  # 256
    page_size_datums = metadata_bytes // 4  # 64 uint32 words on the H2D page
    metadata_bf16_count = metadata_bytes // dtype_size(ttnn.bfloat16)  # 128 bf16 elems on D2H tail
    out_words = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES // dtype_size(ttnn.bfloat16)
    assert (
        NUM_USERS_TO_SWEEP <= num_slots
    ), f"NUM_USERS_TO_SWEEP ({NUM_USERS_TO_SWEEP}) must be <= num_slots ({num_slots})"
    logger.info(
        f"Starting sweep: {NUM_USERS_TO_SWEEP} users x {NUM_POSITIONS_PER_USER} positions "
        f"(page_size_datums={page_size_datums}, out_words={out_words})"
    )
    for slot_id in range(NUM_USERS_TO_SWEEP):
        for pos_id in range(NUM_POSITIONS_PER_USER):
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

    logger.info(
        f"HostIoDecoderStage sweep complete: " f"{NUM_USERS_TO_SWEEP} users x {NUM_POSITIONS_PER_USER} positions"
    )

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

    # Dump on-device KV cache to a torch binary for inspection. Must run under a
    # fast-dispatch context — slow-dispatch (used by H2D/D2H sockets above) doesn't
    # support arbitrary on-device reads. Use DEEPSEEK_V3_KV_CACHE_DUMP_DIR if set,
    # else dump under pytest's tmp_path so the file survives between test runs.
    kv_cache_dump_dir = Path(os.getenv("DEEPSEEK_V3_KV_CACHE_DUMP_DIR", str(tmp_path)))
    logger.info(f"Dumping KV cache to {kv_cache_dump_dir}")
    with ttnn.device.setup_fast_dispatch(submesh):
        stage.dump_kv_cache(out_dir=kv_cache_dump_dir, stage_idx=0)
    logger.info("KV cache dump complete")
