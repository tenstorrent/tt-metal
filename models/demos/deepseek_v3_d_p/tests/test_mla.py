# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_mla_inference(
    config,
    weights,
    mesh_device,
    seq_len,
    mesh_shape,
    sp_axis,
    tp_axis,
    is_balanced,
    topology,
    tt_kvpe_cache,
):
    """
    Utility function to run MLA inference without host comparison.

    Args:
        config: Model configuration
        weights: Model weights dictionary
        mesh_device: Mesh device for TT
        seq_len: Sequence length
        mesh_shape: Shape of mesh device
        sp_axis: Sequence parallel axis
        tp_axis: Tensor parallel axis
        is_balanced: Whether to use balanced chunk ordering
        topology: Topology (Linear or Ring)
        tt_kvpe_cache: Initialized KVPE cache on device

    Returns:
        Tuple of (tt_output, hidden_states, chunk_order, shard_dims)
    """
    # Create TT MLA
    logger.info("Creating TT MLA...")

    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
        topology=topology,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)
    rope_tensors = rope_setup.get_rope_tensors(seq_len)

    # Verify TT MLA exists
    assert mla_tt is not None, "TT MLA should exist"

    # Create test inputs
    batch_size = 1
    hidden_size = config.hidden_size

    logger.info(f"Creating test inputs: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

    # Create random input tensor (generate in float32, then convert to bfloat16)
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16)

    # Reorder hidden_states for balanced ring attention
    sp_factor = mesh_shape[sp_axis]
    chunk_order = create_balanced_chunk_order(sp_factor) if is_balanced else None
    tt_input = hidden_states.unsqueeze(0)  # [1, batch, seq, hidden]
    if is_balanced:
        tt_input = reorder_tensor_chunks(tt_input, chunk_order, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    shard_dims[sp_axis] = -2
    tt_hidden_states = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    tt_output = mla_tt.forward(
        hidden_states=tt_hidden_states,
        rope_tensors=rope_tensors,
        kvpe_cache=tt_kvpe_cache,
    )

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    return tt_output, hidden_states, chunk_order, shard_dims


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(32, 4), (8, 4), (2, 4)],
    ids=["32x4", "8x4", "2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else 1344544,
        },
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else 1344544,
        },
    ],
    ids=["line", "ring"],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize("scale_down_sl", [False, True], ids=["max_sl", "scaled_sl"])
@pytest.mark.parametrize("seq_len", [128 * 1024, 100 * 1024], ids=["seq128k", "seq100k"])
@pytest.mark.parametrize("skip_host_comparison", [False, True], ids=["check_pcc", "skip_check"])
@pytest.mark.parametrize("is_balanced", [False, True], ids=["sequential", "balanced"])
@pytest.mark.timeout(0)  # Disable timeout — first run computes and caches CPU reference for large seq lengths
def test_mla(
    use_pretrained,
    request,
    mesh_device,
    seq_len,
    skip_host_comparison,
    scale_down_sl,
    is_balanced,
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

    production_mesh = [32, 4]
    sp_axis = 0
    tp_axis = 1

    mesh_shape = list(mesh_device.shape)

    if scale_down_sl:
        seq_len = (seq_len // production_mesh[sp_axis]) * mesh_shape[sp_axis]

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
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    # Run MLA inference using utility function
    tt_output, hidden_states, chunk_order, shard_dims = run_mla_inference(
        config=config,
        weights=weights,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
        topology=topology,
        tt_kvpe_cache=tt_kvpe_cache,
    )

    batch_size = 1

    # Host comparison: Run reference forward pass if needed
    if skip_host_comparison == False:
        # Check for cached reference results to avoid expensive host attention computation
        cache_dir = Path(os.environ.get("DEEPSEEK_V3_MLA_REF_CACHE", "/tmp/deepseek_v3_mla_ref_cache"))
        cache_key = f"{weight_type.lower()}_seq{seq_len}"
        cache_path = cache_dir / f"{cache_key}.pt"

        if cache_path.exists():
            logger.info(f"Loading cached reference results from {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            ref_output = cached["ref_output"]
            ref_kvpe = cached["ref_kvpe"]
            logger.info(f"✓ Loaded cached reference results")
            logger.info(f"  Output shape: {ref_output.shape}")
        else:
            assert not (
                (is_ci_env or is_ci_v2_env) and not scale_down_sl
            ), "We should not execute CPU computation in the CI for max sl, output cache is missing"
            # Create position IDs
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

            # Run reference forward pass with cache to capture KVPE
            # Uses F.scaled_dot_product_attention with is_causal=True (no explicit mask needed)
            logger.info("Running reference CPU forward pass...")
            mla_ref = mla_ref.eval().to(torch.bfloat16)
            ref_cache = DynamicCache()
            with torch.no_grad():
                ref_output, _, ref_cache = mla_ref(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    past_key_value=ref_cache,
                    use_cache=True,
                )

            ref_kvpe = ref_cache.key_cache[0]  # layer 0

            if not (is_ci_env or is_ci_v2_env):
                # Save to cache for future runs
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"ref_output": ref_output, "ref_kvpe": ref_kvpe}, cache_path)
                logger.info(f"✓ Saved reference results to {cache_path}")

            logger.info(f"✓ Reference forward pass complete")
            logger.info(f"  Input shape:  {hidden_states.shape}")
            logger.info(f"  Output shape: {ref_output.shape}")
            logger.info(f"  Output dtype: {ref_output.dtype}")
            logger.info(f"  Output mean:  {ref_output.mean().item():.4f}")
            logger.info(f"  Output std:   {ref_output.std().item():.4f}")

        # Compare TT output with reference output
        tt_output_cpu = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)

        if is_balanced:
            tt_output_cpu = reverse_reorder_tensor_chunks(tt_output_cpu, chunk_order, seq_dim=2)

        _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output_cpu, 0.98)
        logger.info(f"Output PCC is {pcc_message}")

        # Validate KVPE cache contents
        # Reference KVPE: [batch, 1, seq_len, kv_lora_rank + qk_rope_head_dim]
        # ref_kvpe is already available (loaded from cache or computed above)

        # Read back KVPE cache from device
        # Cache is replicated across TP, so concat TP replicas on dim 1 (unused) and discard extras
        tt_kvpe_cache_torch = ttnn.to_torch(
            tt_kvpe_cache,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)
        tt_kvpe_cache_torch = tt_kvpe_cache_torch[:1, :1, :, :]

        logger.info("Starting synchronize call")
        ttnn.synchronize_device(mesh_device)
        logger.info("Synchronize call ended")

        logger.debug("  Distributed synchronization started")
        ttnn.distributed_context_barrier()
        logger.debug("✓ Distributed synchronization completed")

        if is_balanced:
            tt_kvpe_cache_torch = reverse_reorder_tensor_chunks(tt_kvpe_cache_torch, chunk_order, seq_dim=2)

        # Check PCC separately for KV (latent) and PE (rope) parts
        kv_lora_rank = config.kv_lora_rank
        _, kv_pcc_message = assert_with_pcc(
            ref_kvpe[:, :, :, :kv_lora_rank], tt_kvpe_cache_torch[:, :, :, :kv_lora_rank], 0.99
        )
        logger.info(f"KVPE cache KV part PCC is {kv_pcc_message}")
        _, pe_pcc_message = assert_with_pcc(
            ref_kvpe[:, :, :, kv_lora_rank:], tt_kvpe_cache_torch[:, :, :, kv_lora_rank:], 0.99
        )
        logger.info(f"KVPE cache PE part PCC is {pe_pcc_message}")
    else:
        logger.info("Starting synchronize call")
        ttnn.synchronize_device(mesh_device)
        logger.info("Synchronize call ended")

        logger.debug("  Distributed synchronization started")
        ttnn.distributed_context_barrier()
        logger.debug("✓ Distributed synchronization completed")

    logger.success(f"✓ Reference and TT comparison with {weight_type} weights successful")


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4), (2, 4)],
    ids=["8x4", "2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else 1344544,
        },
    ],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [8192], ids=["seq8k"])
@pytest.mark.parametrize("num_chunks", [1, 2, 4, 8], ids=lambda n: f"N{n}")
def test_mla_chunked_prefill(
    request,
    mesh_device,
    seq_len,
    num_chunks,
    device_params,
):
    """
    Chunked-prefill correctness test for MLA (is_balanced=False).

    Loops MLA forward over N equal chunks, writing each chunk's K/V into the
    cache at the chunk's local offset and falling back to host SDPA + wkv_b2
    over the cumulative cache. Asserts:
      - per-chunk output PCC against the matching token slice of the reference
      - merged cache PCC against the full reference KVPE (KV and PE parts)
      - concatenated output PCC against the reference output

    Uses random weights only — the chunked vs single-shot equivalence is
    independent of weight distribution.
    """
    config, weights = request.getfixturevalue("random_weights")

    sp_axis = 0
    tp_axis = 1
    is_balanced = False
    topology = ttnn.Topology.Linear

    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]

    chunk_size = seq_len // num_chunks
    if seq_len % num_chunks != 0 or chunk_size % (ttnn.TILE_SIZE * sp) != 0:
        pytest.skip(
            f"chunked-prefill alignment: seq_len={seq_len}, num_chunks={num_chunks}, sp={sp} "
            f"requires chunk_size % (TILE_SIZE*sp) == 0 (got chunk_size={chunk_size})"
        )

    config.max_seq_len = seq_len

    logger.info(
        f"Chunked prefill: seq_len={seq_len} num_chunks={num_chunks} chunk_size={chunk_size} "
        f"sp={sp} tp={mesh_shape[tp_axis]}"
    )

    # Reference (always run — small seq_len keeps this cheap and avoids cache I/O).
    mla_ref = create_mla_reference(
        config=config,
        state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
        layer_idx=0,
        module_path="model.layers.0.self_attn",
    )
    mla_ref = mla_ref.eval().to(torch.bfloat16)

    torch.manual_seed(42)
    hidden_states = torch.randn(1, seq_len, config.hidden_size).to(torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    ref_cache = DynamicCache()
    with torch.no_grad():
        ref_output, _, ref_cache = mla_ref(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=ref_cache,
            use_cache=True,
        )
    ref_kvpe = ref_cache.key_cache[0]

    # TT MLA + rope + cache.
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
        topology=topology,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    hidden_shard_dims = [None, None]
    hidden_shard_dims[tp_axis] = -1
    hidden_shard_dims[sp_axis] = -2

    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = -1
    out_concat_dims[sp_axis] = -2

    per_chunk_outputs = []
    for c in range(num_chunks):
        chunk_start = c * chunk_size
        chunk_end = chunk_start + chunk_size

        chunk_h = hidden_states[:, chunk_start:chunk_end, :].unsqueeze(0)  # [1, 1, chunk_size, hidden]
        tt_chunk_h = ttnn.from_torch(
            chunk_h,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=hidden_shard_dims
            ),
        )

        rope_tensors = rope_setup.get_rope_tensors(chunk_size, start_pos=chunk_start)

        tt_chunk_out = mla_tt.forward(
            hidden_states=tt_chunk_h,
            rope_tensors=rope_tensors,
            kvpe_cache=tt_kvpe_cache,
            chunk_start_global=chunk_start,
        )

        out_host = ttnn.to_torch(
            tt_chunk_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)
        # out_host: [1, 1, chunk_size, hidden_size]
        per_chunk_outputs.append(out_host)

        _, msg = assert_with_pcc(
            ref_output[:, chunk_start:chunk_end, :].unsqueeze(0),
            out_host,
            0.98,
        )
        logger.info(f"  chunk {c}: per-chunk output PCC {msg}")

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    # Concatenated output PCC.
    tt_output_full = torch.cat(per_chunk_outputs, dim=2)  # [1, 1, seq_len, hidden_size]
    _, output_pcc = assert_with_pcc(ref_output.unsqueeze(0), tt_output_full, 0.98)
    logger.info(f"Full output PCC is {output_pcc}")

    # Final cache: gather, merge chunked-interleaved layout to global order, compare.
    cache_stacked = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
    cache_stacked = cache_stacked[:1, :1, :, :]  # [1, 1, sp*seq_len_local, kvpe_dim]

    chunk_local = chunk_size // sp
    cache_merged = cache_stacked.reshape(1, 1, sp, num_chunks, chunk_local, kvpe_dim)
    cache_merged = cache_merged.permute(0, 1, 3, 2, 4, 5).reshape(1, 1, seq_len, kvpe_dim)

    kv_lora_rank = config.kv_lora_rank
    _, kv_pcc = assert_with_pcc(ref_kvpe[:, :, :, :kv_lora_rank], cache_merged[:, :, :, :kv_lora_rank], 0.99)
    logger.info(f"KVPE cache KV part PCC is {kv_pcc}")
    _, pe_pcc = assert_with_pcc(ref_kvpe[:, :, :, kv_lora_rank:], cache_merged[:, :, :, kv_lora_rank:], 0.99)
    logger.info(f"KVPE cache PE part PCC is {pe_pcc}")

    logger.success(f"✓ Chunked prefill (N={num_chunks}) test passed")


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    ids=["2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else 1344544,
        },
    ],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [8192], ids=["seq8k"])
@pytest.mark.parametrize("num_chunks", [2, 4], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("num_users", [2], ids=lambda u: f"U{u}")
def test_mla_chunked_prefill_multi_user(
    request,
    mesh_device,
    seq_len,
    num_chunks,
    num_users,
    device_params,
):
    """
    Multi-user chunked-prefill correctness for MLA.

    Allocates one shared KVPE cache sized for N users (user-major layout:
    slot = user_id * num_layers + layer_idx, num_layers=1 here). For each
    chunk, interleaves a prefill call per user into its own slot via
    cache_user_id + num_cache_layers, then asserts:
      - per-user, per-chunk output PCC against that user's own reference
      - each user's cache slot PCC against that user's reference KVPE

    Cross-user contamination would show up as a PCC drop for one of the users.
    """
    config, weights = request.getfixturevalue("random_weights")

    sp_axis = 0
    tp_axis = 1
    is_balanced = False
    topology = ttnn.Topology.Linear
    num_cache_layers = 1  # one layer in this test; cache batch dim = num_users * 1

    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]

    chunk_size = seq_len // num_chunks
    if seq_len % num_chunks != 0 or chunk_size % (ttnn.TILE_SIZE * sp) != 0:
        pytest.skip(
            f"chunked-prefill alignment: seq_len={seq_len}, num_chunks={num_chunks}, sp={sp} "
            f"requires chunk_size % (TILE_SIZE*sp) == 0 (got chunk_size={chunk_size})"
        )

    config.max_seq_len = seq_len

    logger.info(
        f"Multi-user chunked prefill: num_users={num_users} seq_len={seq_len} "
        f"num_chunks={num_chunks} chunk_size={chunk_size} sp={sp} tp={mesh_shape[tp_axis]}"
    )

    # Reference: one MLA module (weights are shared), run independently per user.
    mla_ref = create_mla_reference(
        config=config,
        state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
        layer_idx=0,
        module_path="model.layers.0.self_attn",
    )
    mla_ref = mla_ref.eval().to(torch.bfloat16)

    torch.manual_seed(42)
    hidden_states_per_user = [torch.randn(1, seq_len, config.hidden_size).to(torch.bfloat16) for _ in range(num_users)]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    ref_outputs = []
    ref_kvpes = []
    for u in range(num_users):
        ref_cache = DynamicCache()
        with torch.no_grad():
            ref_output, _, ref_cache = mla_ref(
                hidden_states=hidden_states_per_user[u],
                position_ids=position_ids,
                past_key_value=ref_cache,
                use_cache=True,
            )
        ref_outputs.append(ref_output)
        ref_kvpes.append(ref_cache.key_cache[0])

    # TT MLA + rope + shared multi-user cache.
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
        topology=topology,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_cache_layers,
        num_users=num_users,
    )
    assert (
        tt_kvpe_cache.shape[0] == num_users * num_cache_layers
    ), f"expected cache batch dim {num_users * num_cache_layers}, got {tt_kvpe_cache.shape[0]}"

    hidden_shard_dims = [None, None]
    hidden_shard_dims[tp_axis] = -1
    hidden_shard_dims[sp_axis] = -2

    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = -1
    out_concat_dims[sp_axis] = -2

    # per_user_chunk_outputs[u] is the list of chunk outputs for user u.
    per_user_chunk_outputs = [[] for _ in range(num_users)]

    for c in range(num_chunks):
        chunk_start = c * chunk_size
        chunk_end = chunk_start + chunk_size
        rope_tensors = rope_setup.get_rope_tensors(chunk_size, start_pos=chunk_start)

        # Interleave users at the same chunk position. Each user reads/writes only
        # its own slot via cache_user_id.
        for u in range(num_users):
            chunk_h = hidden_states_per_user[u][:, chunk_start:chunk_end, :].unsqueeze(0)
            tt_chunk_h = ttnn.from_torch(
                chunk_h,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=hidden_shard_dims
                ),
            )

            tt_chunk_out = mla_tt.forward(
                hidden_states=tt_chunk_h,
                rope_tensors=rope_tensors,
                kvpe_cache=tt_kvpe_cache,
                chunk_start_global=chunk_start,
                cache_user_id=u,
                num_cache_layers=num_cache_layers,
            )

            out_host = ttnn.to_torch(
                tt_chunk_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape
                ),
            ).to(torch.bfloat16)
            per_user_chunk_outputs[u].append(out_host)

            _, msg = assert_with_pcc(
                ref_outputs[u][:, chunk_start:chunk_end, :].unsqueeze(0),
                out_host,
                0.98,
            )
            logger.info(f"  user {u} chunk {c}: per-chunk output PCC {msg}")

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    # Concatenated per-user output PCC.
    for u in range(num_users):
        full = torch.cat(per_user_chunk_outputs[u], dim=2)
        _, pcc = assert_with_pcc(ref_outputs[u].unsqueeze(0), full, 0.98)
        logger.info(f"User {u} full output PCC: {pcc}")

    # Gather the full cache once and slice per user.
    cache_stacked = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
    # cache_stacked: [num_users * num_layers, tp, sp * seq_len_local, kvpe_dim]; pick one TP replica.
    cache_stacked = cache_stacked[:, :1, :, :]

    chunk_local = chunk_size // sp
    kv_lora_rank = config.kv_lora_rank
    for u in range(num_users):
        slot = cache_stacked[u : u + 1, :, :, :]  # [1, 1, sp*seq_len_local, kvpe_dim]
        # Merge chunked-interleaved SP layout back to global token order.
        merged = slot.reshape(1, 1, sp, num_chunks, chunk_local, kvpe_dim)
        merged = merged.permute(0, 1, 3, 2, 4, 5).reshape(1, 1, seq_len, kvpe_dim)
        _, kv_pcc = assert_with_pcc(ref_kvpes[u][:, :, :, :kv_lora_rank], merged[:, :, :, :kv_lora_rank], 0.99)
        logger.info(f"User {u} cache KV part PCC: {kv_pcc}")
        _, pe_pcc = assert_with_pcc(ref_kvpes[u][:, :, :, kv_lora_rank:], merged[:, :, :, kv_lora_rank:], 0.99)
        logger.info(f"User {u} cache PE part PCC: {pe_pcc}")

    logger.success(f"✓ Multi-user chunked prefill (U={num_users}, N={num_chunks}) test passed")


@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    ids=["2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else 1344544,
        },
    ],
    ids=["line"],
    indirect=True,
)
def test_kvpe_cache_multi_user_slot_indexing(mesh_device, device_params):
    """
    Unit test for the user-major slot indexing of init_kvpe_cache +
    ttnn.kv_cache.fill_cache_for_user_.

    Writes a constant V = (layer_idx + 1) * (user_id + 1) into every slot
    (batch_index = user_id * num_layers + layer_idx, the user-major layout),
    plus a +0.5 offset on user 2's slots so each slot has a unique value (no
    aliasing between (1,2) and (2,1) of the bare product). Then reads the
    cache back to host and checks each slot contains its V.
    Confirms:
      - the flat slot index plumbs through fill_cache_for_user_ → cache buffer
      - no slot bleeds into a neighbor (each (user, layer) lands on its own row)

    Values (1-indexed in description, 0-indexed in code):
      user 1, layer 1 → 1.0   user 2, layer 1 → 2.5
      user 1, layer 2 → 2.0   user 2, layer 2 → 4.5
      user 1, layer 3 → 3.0   user 2, layer 3 → 6.5
    """
    sp_axis = 0
    tp_axis = 1
    num_users = 2
    num_layers = 3
    seq_len = 512
    kvpe_dim = 64 + 512  # qk_rope_head_dim + kv_lora_rank — both multiples of TILE_SIZE

    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    seq_len_local = seq_len // sp
    assert seq_len_local % ttnn.TILE_SIZE == 0, f"seq_len_local {seq_len_local} must be tile-aligned"

    tt_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
    )
    assert (
        tt_cache.shape[0] == num_users * num_layers
    ), f"cache batch dim {tt_cache.shape[0]} != num_users*num_layers ({num_users * num_layers})"

    # Write a constant per slot. User 2 (user_id == 1) gets a +0.5 offset so
    # the otherwise-commutative product doesn't alias (u=0,l=1) with (u=1,l=0).
    expected = {}
    for user_id in range(num_users):
        for layer_idx in range(num_layers):
            slot_idx = user_id * num_layers + layer_idx
            V = (layer_idx + 1) * (user_id + 1) + (0.5 if user_id == 1 else 0.0)
            expected[slot_idx] = (user_id, layer_idx, V)

            input_torch = torch.full((1, 1, seq_len_local, kvpe_dim), float(V), dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(
                input_torch,
                device=mesh_device,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            ttnn.kv_cache.fill_cache_for_user_(tt_cache, tt_input, slot_idx)
            logger.info(f"  wrote slot {slot_idx} (user={user_id}, layer={layer_idx}) = {V}")

    ttnn.synchronize_device(mesh_device)

    # Read back. ConcatMesh2dToTensor concats SP along seq (dim=2) and TP along head (dim=1);
    # cache is TP-replicated so all TP shards carry the same data — pick TP shard 0.
    cache_host = ttnn.to_torch(
        tt_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)
    cache_host = cache_host[:, :1, :, :]
    assert cache_host.shape == (
        num_users * num_layers,
        1,
        seq_len,
        kvpe_dim,
    ), f"unexpected gathered cache shape {cache_host.shape}"

    failures = []
    for slot_idx, (user_id, layer_idx, V) in expected.items():
        slot = cache_host[slot_idx]
        slot_min = slot.min().item()
        slot_max = slot.max().item()
        slot_mean = slot.mean().item()
        ok = abs(slot_min - V) < 1e-3 and abs(slot_max - V) < 1e-3
        marker = "✓" if ok else "✗"
        logger.info(
            f"  {marker} slot {slot_idx} (user={user_id}, layer={layer_idx}): "
            f"expected {V}, got min={slot_min} max={slot_max} mean={slot_mean:.4f}"
        )
        if not ok:
            failures.append(
                f"slot {slot_idx} (user={user_id}, layer={layer_idx}): expected {V}, "
                f"got min={slot_min} max={slot_max}"
            )

    assert not failures, "Slot indexing mismatch:\n  " + "\n  ".join(failures)
    logger.success(f"✓ Multi-user KVPE cache slot indexing verified (U={num_users}, L={num_layers})")
