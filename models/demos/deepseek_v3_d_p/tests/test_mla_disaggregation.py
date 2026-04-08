# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tests.test_mla import run_mla_inference
from models.demos.deepseek_v3_d_p.tt.mla.utils import reverse_reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    create_kv_chunk_address_table,
    init_kvpe_cache,
)
from tests.ttnn.utils_for_testing import assert_equal


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(32, 4), (8, 4)],
    ids=["32x4", "8x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        },
    ],
    ids=["line", "ring"],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize("seq_len", [128 * 1024, 100 * 1024], ids=["seq128k", "seq100k"])
@pytest.mark.timeout(0)  # Disable timeout — first run computes and caches CPU reference for large seq lengths
def test_mla_disaggregation(
    use_pretrained,
    request,
    mesh_device,
    seq_len,
    is_ci_env,
    is_ci_v2_env,
    device_params,
):
    """
    Test comparing reference and TT MLA modules with same weights.

    Args:
        use_pretrained: Whether to use pretrained weights
        request: Pytest request object for conditional fixture loading
        mesh_device: Mesh device fixture
        seq_len: Sequence length
    """
    weight_type = "Pretrained" if use_pretrained else "Random"
    logger.info("=" * 80)
    logger.info(f"Test: Reference vs TT Comparison ({weight_type} Weights)")
    logger.info("=" * 80)

    # Conditionally load fixtures - only load what we need!
    if use_pretrained:
        config, weights = request.getfixturevalue("pretrained_weights")
    else:
        config, weights = request.getfixturevalue("random_weights")

    fabric_config = device_params.get("fabric_config", ttnn.FabricConfig.FABRIC_1D)
    topology = ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear

    sp_axis = 0
    tp_axis = 1

    mesh_shape = list(mesh_device.shape)

    # temp hack
    config.max_seq_len = seq_len

    # Create reference MLA
    if use_pretrained:
        logger.info("Creating reference MLA with pretrained weights...")
        mla_ref = create_mla_reference(
            config=config,
            state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
            layer_idx=0,
            module_path="model.layers.0.self_attn",
        )
    else:
        logger.info("Creating reference MLA with random weights...")
        mla_ref = create_mla_reference(
            config=config,
            state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
            layer_idx=0,
            module_path="model.layers.0.self_attn",
        )

    # Verify reference MLA exists
    assert mla_ref is not None, "Reference MLA should exist"

    # Test forward pass comparison
    logger.info("=" * 80)
    logger.info(f"Testing forward pass comparison (seq_len={seq_len})")
    logger.info("=" * 80)

    # Initialize KVPE cache
    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank  # 576

    # Initialise kvpe cache with 2 layers, but run only 1 layer.
    # Other layer should be zeros
    num_kvpe_cache_layers = 2
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_kvpe_cache_layers,
    )

    # Create and populate KV chunk address table using utility function
    CHUNK_SIZE_BYTES = 19584  # [1, 1, 32, 576] bfp8
    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = num_kvpe_cache_layers
    config.max_sequence_length = seq_len
    config.num_slots = 1
    config.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    config.chunk_size_bytes = CHUNK_SIZE_BYTES

    lookup_table = create_kv_chunk_address_table(
        config=config,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tt_kvpe_cache=tt_kvpe_cache,
        chunk_size_bytes=CHUNK_SIZE_BYTES,
    )

    # Run MLA inference using utility function
    # Fill first layer with actual kv cache
    # Layer 1 remains zeros
    _, _, chunk_order, _ = run_mla_inference(
        config=config,
        weights=weights,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=True,
        topology=topology,
        tt_kvpe_cache=tt_kvpe_cache,
    )

    tt_kvpe_cache_torch = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)

    # remember layer0 results
    tt_kvpe_cache_torch_layer0 = tt_kvpe_cache_torch[:1, :1, :, :]

    # assert layer1 is zeros
    tt_kvpe_cache_torch_layer1 = tt_kvpe_cache_torch[1:2, :1, :, :]
    torch_layer_zeros = torch.zeros(1, 1, seq_len, kvpe_cache_head_dim)
    logger.info(f"Checking that layer 1 is all zeros")
    assert_equal(tt_kvpe_cache_torch_layer1, torch_layer_zeros)
    logger.info(f"Check complete")

    # reorder into position continuous torch cache
    tt_kvpe_cache_torch_layer0 = reverse_reorder_tensor_chunks(tt_kvpe_cache_torch_layer0, chunk_order, seq_dim=2)

    # migrate layer0 of tt_kvpe_cache into layer1 of tt_kvpe_cache

    # perform migration

    # now assert equality of tt_kvpe_cache_torch_layer0 with tt_kvpe_cache_torch_layer1
    # tt_kvpe_cache_torch = ttnn.to_torch(
    #     tt_kvpe_cache,
    #     mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    # ).to(torch.bfloat16)
    # tt_kvpe_cache_torch_layer1 = tt_kvpe_cache_torch[1:2, :1, :, :]
    # tt_kvpe_cache_torch_layer1 = reverse_reorder_tensor_chunks(tt_kvpe_cache_torch_layer1, chunk_order, seq_dim=2)
    # assert_equal(tt_kvpe_cache_torch_layer0, tt_kvpe_cache_torch_layer1)
    # we can also make sure layer 0 didn't get polluted
    # assert_equal(tt_kvpe_cache_torch_layer0, tt_kvpe_cache_torch_layer0_new)
    # ...

    logger.info("Starting synchronize call")
    ttnn.synchronize_device(mesh_device)
    logger.info("Synchronize call ended")

    logger.debug("  Distributed synchronization started")
    ttnn.distributed_context_barrier()
    logger.debug("✓ Distributed synchronization completed")
