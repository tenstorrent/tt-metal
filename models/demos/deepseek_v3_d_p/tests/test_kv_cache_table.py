# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""


import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tests.test_mla import run_mla_inference
from models.demos.deepseek_v3_d_p.tt.mla.utils import reverse_reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    BH_NUM_DRAM_BANKS,
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    PREFILL_CHUNK_OUTPUT_TOKENS,
    create_kv_chunk_address_table_ds,
    create_kv_chunk_address_table_kimi,
    init_kvpe_cache,
)
from tests.ttnn.utils_for_testing import assert_equal


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    ids=["8x4"],
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
@pytest.mark.parametrize("seq_len", [25 * 1024], ids=["seq25k"])
@pytest.mark.timeout(0)  # Disable timeout — first run computes and caches CPU reference for large seq lengths
def test_kv_cache_table(
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
        config, sd = request.getfixturevalue("pretrained_transformer_weights")
        weights = sd["layers"][0]["mla_weights"]
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

    num_kvpe_cache_layers = 1
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
    lookup_table_config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    lookup_table_config.num_layers = num_kvpe_cache_layers
    lookup_table_config.max_sequence_length = seq_len
    lookup_table_config.num_slots = 1
    lookup_table_config.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    lookup_table_config.chunk_size_bytes = CHUNK_SIZE_BYTES

    lookup_table = create_kv_chunk_address_table_ds(
        config=lookup_table_config,
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

    # reorder into position continuous torch cache
    tt_kvpe_cache_torch_layer0 = reverse_reorder_tensor_chunks(tt_kvpe_cache_torch_layer0, chunk_order, seq_dim=2)

    # Walk every chunk in layer 0, read it back via the address table, and
    # compare against the corresponding 32-token slice of the gathered cache.
    chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, kvpe_cache_head_dim]
    for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        raw_bytes = lookup_table.read_device_chunk(layer=0, position=position, slot=0)
        chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw_bytes, chunk_shape)
        chunk_torch = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
        expected_chunk = tt_kvpe_cache_torch_layer0[:, :, position : position + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, :]
        assert_equal(chunk_torch, expected_chunk)


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    ids=["8x4"],
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
@pytest.mark.parametrize("seq_len", [5 * 1024], ids=["seq5k"])
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_kv_cache_table(
    use_pretrained,
    request,
    mesh_device,
    seq_len,
    variant,
    device_params,
):
    """
    Readback test for the Kimi (non-balanced / sequential) KV chunk address table.

    Runs Kimi MLA (variant kimi_k2_6) to fill a sequentially laid-out KVPE cache,
    builds the table with create_kv_chunk_address_table_kimi, then reads every chunk
    back through the table and checks it against the gathered cache. The sequential
    gather is already position-continuous, so no chunk reorder is needed.
    """

    # Conditionally load fixtures - only load what we need!
    if use_pretrained:
        config, sd = request.getfixturevalue("pretrained_transformer_weights")
        weights = sd["layers"][0]["mla_weights"]
    else:
        config, weights = request.getfixturevalue("random_weights")

    assert config.num_attention_heads == 64, f"Not Kimi config: {config.num_attention_heads} heads"

    logger.info(f"model={variant.name} num_heads={config.num_attention_heads} hidden={config.hidden_size}")

    fabric_config = device_params.get("fabric_config", ttnn.FabricConfig.FABRIC_1D)
    topology = ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    config.max_seq_len = seq_len

    # Test forward pass comparison
    logger.info("=" * 80)
    logger.info(f"Testing forward pass comparison (seq_len={seq_len})")
    logger.info("=" * 80)

    # Initialize KVPE cache
    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank  # 576

    num_kvpe_cache_layers = 1
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
    lookup_table_config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    lookup_table_config.num_layers = num_kvpe_cache_layers
    lookup_table_config.max_sequence_length = seq_len
    lookup_table_config.num_slots = 1
    lookup_table_config.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    lookup_table_config.chunk_size_bytes = CHUNK_SIZE_BYTES

    lookup_table = create_kv_chunk_address_table_kimi(
        config=lookup_table_config,
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
    run_mla_inference(
        config=config,
        weights=weights,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=topology,
        tt_kvpe_cache=tt_kvpe_cache,
    )

    tt_kvpe_cache_torch = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)

    # remember layer0 results
    tt_kvpe_cache_torch_layer0 = tt_kvpe_cache_torch[:1, :1, :, :]

    # Walk every chunk in layer 0, read it back via the address table, and compare
    # against the corresponding 32-token slice of the gathered cache.
    chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, kvpe_cache_head_dim]
    for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        raw_bytes = lookup_table.read_device_chunk(layer=0, position=position, slot=0)
        chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw_bytes, chunk_shape)
        chunk_torch = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
        expected_chunk = tt_kvpe_cache_torch_layer0[:, :, position : position + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, :]
        assert_equal(chunk_torch, expected_chunk)


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    ids=["8x4"],
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
@pytest.mark.parametrize("seq_len", [5 * 1024, 10 * 1024, 25 * 1024], ids=["seq5k", "seq10k", "seq25k"])
@pytest.mark.parametrize("num_users", [1, 2], ids=["1user", "2users"])
@pytest.mark.parametrize("num_layers", [1, 2], ids=["1layer", "2layers"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_kv_cache_mock(
    mesh_device,
    seq_len,
    num_users,
    num_layers,
    device_params,
):
    sp_axis = 0
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[sp_axis]
    kvpe_cache_head_dim = 576  # qk_rope_head_dim(64) + kv_lora_rank(512); same for Kimi and DeepSeek
    num_kvpe_cache_layers = num_users * num_layers

    chunk_tokens = PREFILL_CHUNK_OUTPUT_TOKENS
    tokens_per_chunk_per_device = chunk_tokens // sp_factor  # 640
    num_seq_chunks = seq_len // chunk_tokens

    torch.manual_seed(42)
    reference = torch.randn(num_kvpe_cache_layers, 1, seq_len, kvpe_cache_head_dim).to(torch.bfloat16)

    # For 10k, device 0 holds tokens [0-639, 5120-5759], device 1 holds [640-1279, 5760-6399], … device 7 holds [4480-5119, 9600-10239].
    device_buffers = []
    for d in range(sp_factor):
        lo = d * tokens_per_chunk_per_device
        hi = lo + tokens_per_chunk_per_device
        slices = [reference[:, :, c * chunk_tokens + lo : c * chunk_tokens + hi, :] for c in range(num_seq_chunks)]
        device_buffers.append(torch.cat(slices, dim=2))
    host_stacked = torch.cat(device_buffers, dim=2)  # [num_users*num_layers, 1, seq_len, head_dim], shardable across SP

    # 32-token shards round-robined over the dram banks.
    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    kv_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, kvpe_cache_head_dim],
            grid=ttnn.CoreRangeSet(core_ranges),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )

    shard_dims = [None, None]
    shard_dims[sp_axis] = -2
    tt_kvpe_cache = ttnn.from_torch(
        host_stacked,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    CHUNK_SIZE_BYTES = 19584  # [1, 1, 32, 576] bfp8
    lookup_table_config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    lookup_table_config.num_layers = num_layers
    lookup_table_config.max_sequence_length = seq_len
    lookup_table_config.num_slots = num_users
    lookup_table_config.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    lookup_table_config.chunk_size_bytes = CHUNK_SIZE_BYTES
    lookup_table = create_kv_chunk_address_table_kimi(
        config=lookup_table_config,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tt_kvpe_cache=tt_kvpe_cache,
        chunk_size_bytes=CHUNK_SIZE_BYTES,
        num_users=num_users,
    )

    reference_bf8 = ttnn.to_torch(ttnn.from_torch(reference, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).to(
        torch.bfloat16
    )

    chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, kvpe_cache_head_dim]
    for slot in range(num_users):
        for layer in range(num_layers):
            batch_idx = slot * num_layers + layer  # cache batch index
            for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                pos_end = position + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
                raw_bytes = lookup_table.read_device_chunk(layer=layer, position=position, slot=slot)
                chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw_bytes, chunk_shape)
                chunk_torch = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
                expected_chunk = reference_bf8[batch_idx : batch_idx + 1, :, position:pos_end, :]
                assert_equal(chunk_torch, expected_chunk)
