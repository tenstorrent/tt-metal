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
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import build_weights
from models.demos.deepseek_v3_d_p.tests.test_mla import run_mla_inference
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, reverse_reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    BH_NUM_DRAM_BANKS,
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    PREFILL_CHUNK_OUTPUT_TOKENS,
    create_kv_chunk_address_table_ds,
    create_kv_chunk_address_table_kimi,
    init_kvpe_cache,
    populate_kv_chunk_address_table_kimi,
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
    ],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [5 * 1024], ids=["seq5k"])
@pytest.mark.parametrize("variant", ["glm_5_1"], indirect=True, ids=["glm"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA (indexer / sparse SDPA) is Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_kv_cache_table(
    mesh_device,
    seq_len,
    variant,
    config_only,
    device_params,
):
    """
    Readback test for the GLM (glm_5_1) INDEXER key cache. Stands up a single sparse GLM MLA layer with
    random weights, runs one forward at seq_len=5k to fill the indexer's block-cyclic key cache
    (tt_index_cache: [1, 1, S/sp, index_head_dim] bfp8, 1 layer / 1 user — caller-allocated and passed
    into forward(index_kv_cache=...), the same ownership as the MLA KVPE cache), then builds a KV
    chunk address table over THAT cache with create_kv_chunk_address_table_kimi and reads every 32-token
    chunk back, comparing to the gathered cache. The index cache row is index_head_dim(128) wide, so a
    32-token DRAM-bank chunk is [1, 1, 32, 128] bfp8 = 4 tiles. For a single full-seq chunk the
    block-cyclic layout coincides with the sequential (Kimi) layout, so no chunk reorder is needed.
    """
    config = config_only
    fabric_config = device_params.get("fabric_config", ttnn.FabricConfig.FABRIC_1D)
    topology = ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    config.max_seq_len = seq_len
    logger.info(
        f"model={variant.name} num_heads={config.num_attention_heads} hidden={config.hidden_size} topology={topology}"
    )

    # Random GLM weights, incl. the indexer weights the sparse path needs (the random_weights fixture is
    # dense-only, so build_weights — random by default — is used to also populate indexer.*).
    weights, _ = build_weights(variant, config, seed=42)

    # Single-chunk block-cyclic forward: chunk == full sequence. Even with one chunk the input is
    # arranged through the block-cyclic positions (so an SP-contiguous shard lands the block-cyclic rows
    # on each chip), matching update_padded_kv_cache's writer layout — not relying on the aligned
    # single-chunk case collapsing to natural order.
    chunk_size_global = seq_len
    sp = mesh_shape[sp_axis]
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        slot_num=1,
        layer_num=1,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(cache_seq_len_global=seq_len, chunk_size_global=chunk_size_global)

    # KVPE cache: uncompressed bf16 + ROW_MAJOR, the format sparse_sdpa reads natively (see MLA.forward).
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Indexer key cache: caller-owned (like the KVPE cache), NOT self-allocated by the indexer. Block-cyclic
    # bfp8 TILE, index_head_dim(128) wide, 1 layer / 1 user; passed into forward(index_kv_cache=...) so the
    # indexer writes its roped keys into it. write_k typecasts to this cache's dtype before the in-place write.
    tt_index_cache = init_kvpe_cache(
        kvpe_cache_head_dim=mla_tt._indexer.index_args.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )

    torch.manual_seed(42)
    hidden = torch.randn(seq_len, config.hidden_size, dtype=torch.bfloat16)
    # Gather rows into block-cyclic (device-major) order: shard row r holds natural position p[r], so an
    # SP-contiguous split puts each chip's block-cyclic rows on it (identity for a single aligned chunk,
    # but keeps the layout correct/general).
    p = blockcyclic_positions(sp, chunk_size_global, seq_len)
    chunk_in = hidden[p]
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    tt_hidden = ttnn.from_torch(
        chunk_in.reshape(1, 1, seq_len, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    tt_out = mla_tt.forward(
        tt_hidden, rope_tensors, tt_kvpe_cache, actual_start=0, cache_user_id=0, index_kv_cache=tt_index_cache
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[glm] forward complete: out shape {tuple(tt_out.shape)}")

    # Build ONE multi-config KV chunk address table (commit 7a5be3a5e76) holding BOTH caches instead of
    # two separate tables, using the same config ordering as the serving path (build_and_serialize_kv_chunk_table):
    # config 0 = the MLA KVPE cache (bf16 ROW_MAJOR, kvpe_head_dim wide); config 1 = the block-cyclic
    # index-key cache (bfp8 TILE, index_head_dim wide). The two configs share the
    # device-group / fabric-host side table but each carries its own grid + chunk_size_bytes and is
    # addressed by config_id on every accessor (set / read_device_chunk). Both caches are
    # [num_users*num_layers, 1, S/sp, head_dim], ND-sharded 32-tokens-per-bank (round-robin over the DRAM
    # banks) — the layout populate_kv_chunk_address_table_kimi addresses.
    index_kbuf = tt_index_cache
    index_head_dim = mla_tt._indexer.index_args.index_head_dim  # 128
    kvpe_head_dim = config.kv_lora_rank + config.qk_rope_head_dim  # 576

    # index config: [1, 1, 32, 128] bfp8 = (32/32)*(128/32) = 4 tiles; a 32x32 bfp8 tile is 1024 data + 64
    # exponent bytes. kvpe config: bf16 ROW_MAJOR, [1, 1, 32, 576] = 32*576*2 bytes contiguous.
    INDEX_CHUNK_SIZE_BYTES = 4 * 1088  # 4352
    KVPE_CHUNK_SIZE_BYTES = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK * kvpe_head_dim * 2  # 36864
    KVPE_CONFIG_ID, INDEX_CONFIG_ID = 0, 1

    def _table_config(chunk_size_bytes):
        c = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
        c.num_layers = 1
        c.max_sequence_length = seq_len
        c.num_slots = 1
        c.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
        c.chunk_size_bytes = chunk_size_bytes
        return c

    index_config = _table_config(INDEX_CHUNK_SIZE_BYTES)
    kvpe_config = _table_config(KVPE_CHUNK_SIZE_BYTES)

    # A list of configs -> config i is named "i" (id i). config 0 = kvpe, config 1 = index (serving order).
    lookup_table = ttnn.experimental.disaggregation.KvChunkAddressTable([kvpe_config, index_config])
    assert lookup_table.num_configs() == 2, f"expected 2 configs, got {lookup_table.num_configs()}"

    populate_kv_chunk_address_table_kimi(
        lookup_table=lookup_table,
        config=index_config,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tt_kvpe_cache=index_kbuf,
        chunk_size_bytes=INDEX_CHUNK_SIZE_BYTES,
        num_users=1,
        config_id=INDEX_CONFIG_ID,
    )
    populate_kv_chunk_address_table_kimi(
        lookup_table=lookup_table,
        config=kvpe_config,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tt_kvpe_cache=tt_kvpe_cache,
        chunk_size_bytes=KVPE_CHUNK_SIZE_BYTES,
        num_users=1,
        config_id=KVPE_CONFIG_ID,
    )

    # --- readback the index cache (config 1, bfp8 TILE) ---
    # Gather the index cache to a single [1, 1, seq_len, index_head_dim] torch tensor (SP-concat on seq,
    # TP is replicated so take the first column group), then compare every 32-token chunk to the readback.
    index_kbuf_torch = ttnn.to_torch(
        index_kbuf,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)[:1, :1, :, :]

    chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, index_head_dim]
    for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        pos_end = position + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
        raw_bytes = lookup_table.read_device_chunk(layer=0, position=position, slot=0, config_id=INDEX_CONFIG_ID)
        chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw_bytes, chunk_shape)
        chunk_torch = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
        expected_chunk = index_kbuf_torch[:, :, position:pos_end, :]
        assert_equal(chunk_torch, expected_chunk)
    logger.info(f"[glm] index-cache (config {INDEX_CONFIG_ID}) address-table readback verified over {seq_len} tokens")

    # --- readback the MLA KVPE cache (config 0, bf16 ROW_MAJOR) ---
    # The KVPE cache is UNCOMPRESSED bf16 ROW_MAJOR (not bfp8 TILE), so a 32-token DRAM-bank chunk is
    # [1, 1, 32, kvpe_head_dim] bf16 = 32*kvpe_head_dim*2 bytes contiguous, decoded via tensor_from_bf16_bytes
    # (the RM/bf16 analogue of tensor_from_bfp8_bytes).
    tt_kvpe_cache_torch = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)[:1, :1, :, :]

    kvpe_chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, kvpe_head_dim]
    for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        pos_end = position + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
        raw_bytes = lookup_table.read_device_chunk(layer=0, position=position, slot=0, config_id=KVPE_CONFIG_ID)
        chunk_tt = ttnn.experimental.disaggregation.tensor_from_bf16_bytes(raw_bytes, kvpe_chunk_shape)
        chunk_torch = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
        expected_chunk = tt_kvpe_cache_torch[:, :, position:pos_end, :]
        assert_equal(chunk_torch, expected_chunk)
    logger.info(
        f"[glm] kvpe-cache (config {KVPE_CONFIG_ID}, bf16 RM) address-table readback verified over {seq_len} tokens"
    )
