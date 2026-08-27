# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for post-combine reduce module in isolation.

This test verifies that the TTNN reduce module produces the same output as the
PyTorch reference implementation when reducing sparse combine outputs.

Uses synthetic sparse inputs to isolate the reduce operation from dispatch/combine.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.reduce import TorchReduceModule
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_y_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    create_sparse_combine_output,
    extract_mesh_config,
    get_tp_mesh_composer,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule, _weights_have_output_channel_dim
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from tests.ttnn.utils_for_testing import comp_pcc

REDUCE_MESH_PARAMS = [
    pytest.param(
        (4, 1),
        torus_y_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
        id="torus-y-4x1",
    ),
    pytest.param(
        (4, 2),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
        id="fabric2d-mesh-4x2",
    ),
    # Blackhole accepts 32-device meshes only, so neither shape above runs on the galaxy and every row
    # there skips -- rc=0, which reads as coverage. This row is the only one that executes there, and
    # so the only one covering mistral_small_4 (or any model) on Blackhole.
    pytest.param(
        (8, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-mesh-8x4",
    ),
]


@pytest.mark.parametrize(
    "weights_shape, combine_output_shape, expected",
    [
        # A top-k=1 score lacks the output channel and must be unsqueezed.
        ((4, 64, 1), (1, 4, 64, 1, 1024), False),
        # The same score with an explicit output channel omits only a leading
        # mesh dimension and must not receive a second trailing singleton.
        ((4, 64, 1, 1), (1, 4, 64, 1, 1024), True),
        ((1, 4, 64, 1, 1), (1, 4, 64, 1, 1024), True),
        # A top-k>1 score has no ambiguity: without the output channel it is
        # not broadcastable to the complete combine output.
        ((4, 64, 8), (1, 4, 64, 8, 1024), False),
    ],
)
def test_weights_output_channel_shape_detection(weights_shape, combine_output_shape, expected):
    actual = _weights_have_output_channel_dim(weights_shape, combine_output_shape)
    assert actual is expected, (
        f"Expected channel-dimension inference to be {expected} for weights={weights_shape} "
        f"and combine_output={combine_output_shape}, but got {actual}"
    )


def run_reduce(
    mesh_device,
    device_params,
    seq_len,
    emb_dim,
    topk,
    use_weights,
):
    """Run the TTNN reduce module in isolation against the torch reference. Shared body for the
    per-model test entrypoints below — they differ only on the (emb_dim, topk) shape axis."""
    # The ND-sharded combine output is chunked by topk across the mesh, so a topk that does not divide
    # the device count cannot be laid out:
    #   TT_FATAL: ND sharding requires the number of chunks 24 to match the mesh dimension size 32
    # Observed on the 8x4 galaxy row: topk 8 and 4 lay out, topk 6 (dsv4_pro / dsv4_flash) does not.
    # Skipped rather than xfailed because it is a property of the shape, not a defect to fix here.
    num_devices = mesh_device.get_num_devices()
    if num_devices % topk:
        pytest.skip(f"topk={topk} does not divide {num_devices} devices; ND sharding cannot chunk it")

    torch.manual_seed(42)

    signpost(f"reduce-{mesh_device.shape}-seq{seq_len}-{'weighted' if use_weights else 'unweighted'}")

    num_links = 1

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.debug(f"Testing with {mesh_device.shape=}, {num_devices=} {dispatch_group_size=} {num_dispatch_groups=}")

    ttnn.visualize_mesh_device(mesh_device)

    # Create synthetic sparse combine output
    torch_combine_output = create_sparse_combine_output(
        num_chips=dispatch_group_size,
        seq_len=seq_len,
        topk=topk,
        emb_dim=emb_dim,
        sparsity=0.75,
    )
    logger.debug(f"Created sparse combine output: {torch_combine_output.shape}")

    num_routed_experts = 64

    # Create random gate weights for weighted reduce (if enabled)
    torch_gate_weights = None
    if use_weights:
        _, torch_gate_weights, _ = initialize_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len,
            emb_dim=emb_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=topk,
            max_dispatched_tokens_per_expert=1000,
            validate=False,
            skip_x_initialization=True,
        )
        logger.debug(f"Created gate weights: {torch_gate_weights.shape}")

    # Create indices and dispatch table for the reduce kernel.
    # Use a dispatch table where ALL experts are valid (no -1 entries) so
    # the kernel processes every expert slot — matching the torch reference.
    # Non-local expert skipping correctness is tested separately in
    # test_deepseek_moe_post_combine_reduce.py::test_skip_nonlocal_experts.
    torch_indices = torch.randint(0, num_routed_experts, (dispatch_group_size, seq_len, topk), dtype=torch.int32)
    expert_dispatch_table = torch.zeros((num_dispatch_groups, num_routed_experts), dtype=torch.int32)

    # Compute reference output using torch
    torch_reduce = TorchReduceModule(
        topk_dim=2,  # topk is dim 2 in [chips, seq, topk, hidden]
    )
    torch_shards = torch_reduce(torch_combine_output, weights=torch_gate_weights)
    logger.debug(f"Torch reference output: {len(torch_shards)} shards of shape {torch_shards[0].shape}")

    # Convert to TTNN tensor distributed across mesh
    # For reduce_scatter, we need each chip to have its portion of the input
    # Shape transformation: [num_reduce_chips, seq, topk, hidden] -> per-chip [seq, topk, hidden]

    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, 2),  # Shard batch within dispatch group; shard topk across dispatch groups
    )

    tt_combine_output = ttnn.from_torch(
        torch_combine_output,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    logger.debug(f"{tt_combine_output.shape=}")

    # Convert gate weights to TTNN tensor with same sharding as combine_output (if enabled)
    tt_gate_weights = None
    if use_weights:
        tt_gate_weights = ttnn.from_torch(
            torch_gate_weights,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )
        logger.debug(f"{tt_gate_weights.shape=}")

    # Convert indices and dispatch table to TTNN tensors
    indices_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),
    )
    tt_indices = ttnn.from_torch(
        torch_indices,
        mesh_mapper=indices_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint16,
    )
    tt_expert_dispatch_table = ttnn.from_torch(
        expert_dispatch_table,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(None, 0),
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # Run TTNN reduce
    # NOTE: TTNN adds a batch dim, so [seq, topk, hidden] becomes [1, seq, topk, hidden]
    # topk is at dim=2 in the 4D tensor
    tt_reduce = TtReduceModule(
        mesh_device=mesh_device,
        topk_dim=2,  # topk is dim 2 in [1, seq, topk, hidden]
        cluster_axis=1,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"])[1],
    )

    tt_output = tt_reduce(
        tt_combine_output,
        weights=tt_gate_weights,
        indices=tt_indices,
        expert_dispatch_table=tt_expert_dispatch_table,
    )
    logger.debug(f"{tt_output.shape=}")

    composer = get_tp_mesh_composer(mesh_device)
    tt_host = ttnn.to_torch(tt_output, mesh_composer=composer, dtype=torch.bfloat16)
    logger.debug(f"{tt_host.shape=}")
    threshold = 0.999
    _, pcc = comp_pcc(torch_shards.float(), tt_host.float())

    logger.debug(f"TTNN reduce operation matches torch reference! (PCC={pcc:.6f})")
    assert pcc > threshold, f"PCC {pcc:.6f} below threshold {threshold}"


# Model-independent sanity shape — small seq/emb that exercises the reduce kernel without
# tying to any model's dimensions. Kept in a single test so it is not duplicated per model.
@pytest.mark.parametrize("use_weights", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize(
    "seq_len, emb_dim, topk",
    [(32, 2048, 8)],
    ids=["generic"],
)
@pytest.mark.parametrize("mesh_device, device_params", REDUCE_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_ttnn_reduce(mesh_device, device_params, seq_len, emb_dim, topk, use_weights):
    run_reduce(mesh_device, device_params, seq_len, emb_dim, topk, use_weights)


@pytest.mark.parametrize("use_weights", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize(
    "mesh_device, device_params", REDUCE_MESH_PARAMS[:1], indirect=["mesh_device", "device_params"]
)
def test_ttnn_reduce_single_expert(mesh_device, device_params, use_weights):
    """Top-k=1 remains on the original single-axis mesh, migrated to TorusY."""
    run_reduce(mesh_device, device_params, seq_len=32, emb_dim=1024, topk=1, use_weights=use_weights)


# Per-model reduce shapes as (id_prefix, config, extended_model). Each model uses seq_len 640 and
# topk = NUM_EXPERTS_PER_TOKEN at its own emb_dim. DeepSeek V3 is the baseline and runs by default;
# every other model is gated behind @pytest.mark.extended_model.
REDUCE_MODELS = [
    ("dsv3", DeepSeekV3Config, False),
    ("glm_51", GLM51Config, True),
    ("kimi_k26", KimiK26Config, True),
    ("minimax_m27", MiniMaxM27Config, True),
    ("dsv4_pro", DeepSeekV4ProConfig, True),
    ("dsv4_flash", DeepSeekV4FlashConfig, True),
    ("gptoss_120b", GptOss120BConfig, True),
    # Mistral-Small-4-119B: emb_dim 4096, topk 4. Top-4 is the smallest topk here that the mesh-4x2
    # mapper can still shard across the two dispatch groups (2 each) — top-1 needs the linear-4
    # mesh, which is why it has its own test above.
    ("mistral4", MistralSmall4Config, True),
]


def reduce_shape_params():
    """Build the per-model (seq_len, emb_dim, topk) parametrization. Non-baseline models carry the
    extended_model marker on their params so they stay gated exactly as the separate tests were."""
    params = []
    for name, config, extended in REDUCE_MODELS:
        marks = (pytest.mark.extended_model,) if extended else ()
        params.append(pytest.param(640, config.EMB_SIZE, config.NUM_EXPERTS_PER_TOKEN, marks=marks, id=name))
    return params


@pytest.mark.parametrize("use_weights", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("seq_len, emb_dim, topk", reduce_shape_params())
@pytest.mark.parametrize("mesh_device, device_params", REDUCE_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_ttnn_reduce_models(mesh_device, device_params, seq_len, emb_dim, topk, use_weights):
    run_reduce(mesh_device, device_params, seq_len, emb_dim, topk, use_weights)
