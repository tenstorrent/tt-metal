#!/usr/bin/env python3
"""
Minimal test to reproduce OLMo 16K eager mode hang.

Usage:
    # Test 8K (should pass)
    pytest models/demos/olmo_galaxy/tests/test_olmo_16k_eager_hang.py -v -s -k "8192"

    # Test 16K (expected to hang without DEBUG_PREFILL_LAYERS)
    pytest models/demos/olmo_galaxy/tests/test_olmo_16k_eager_hang.py -v -s -k "16384"

    # Test 16K with DEBUG_PREFILL_LAYERS (expected to work)
    DEBUG_PREFILL_LAYERS=1 pytest ... -k "16384"
"""

import os
from time import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.olmo_galaxy.tt.llama_common import PagedAttentionConfig, gather_cos_sin, precompute_freqs_yarn
from models.demos.olmo_galaxy.tt.llama_model import TtTransformer
from models.demos.olmo_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.tt_transformers.tt.common import copy_host_to_device


def get_model_path():
    """Get OLMo model path from environment or default."""
    return os.environ.get(
        "OLMO_MODEL_PATH",
        "/home/cust-team/models/models--allenai--OLMo-3.1-32B-Think/snapshots/832c3f543499af8fe68b88359501de9cb7840544",
    )


@pytest.fixture(scope="module")
def mesh_device():
    """Initialize mesh device for TG (32 devices)."""
    logger.info("Opening mesh device (4x8)...")
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(4, 8),
        dispatch_core_config=ttnn.DispatchCoreConfig(
            ttnn.DispatchCoreType.WORKER,
            ttnn.DispatchCoreAxis.COL,
        ),
    )
    yield mesh
    logger.info("Closing mesh device...")
    ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def model_and_cache(mesh_device):
    """Initialize OLMo model in prefill mode."""
    model_path = get_model_path()

    # Model args for OLMo
    logger.info(f"Creating model args for OLMo from {model_path}...")
    model_args = TtOlmoModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        max_seq_len=128 * 1024,  # 128K max
    )
    model_args.n_layers = 64
    model_args.load_weights_path = model_path

    # Paged attention config
    paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=4096)

    # Initialize model in prefill mode
    logger.info("Initializing TtTransformer in prefill mode...")
    tt_model = TtTransformer(
        mesh_device=mesh_device,
        args=model_args,
        mode="prefill",
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=True,
    )

    # Create KV cache
    logger.info("Setting up KV cache...")
    kv_cache = tt_model.setup_cache(max_batch_size=1)

    logger.info(f"Model initialized. support_seqlens = {tt_model.tt_ccl.support_seqlens}")
    logger.info(f"MAX_TRACE_SEQLEN = {max(tt_model.tt_ccl.support_seqlens)}")

    yield tt_model, kv_cache, model_args

    # Cleanup
    logger.info("Tearing down model...")


def run_eager_prefill(tt_model, kv_cache, mesh_device, model_args, seq_len: int):
    """
    Run a single eager prefill and measure time.

    Args:
        seq_len: Padded sequence length (must be power of 2)

    Returns:
        (success: bool, elapsed_seconds: float, error: str or None)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing eager prefill at ISL={seq_len}")
    logger.info(f"  seq_len in support_seqlens: {seq_len in tt_model.tt_ccl.support_seqlens}")
    logger.info(f"  Expected mode: {'TRACED' if seq_len in tt_model.tt_ccl.support_seqlens else 'EAGER'}")
    logger.info(f"  DEBUG_PREFILL_LAYERS={os.environ.get('DEBUG_PREFILL_LAYERS', 'not set')}")
    logger.info(f"{'='*60}")

    # Reset CCL indices
    tt_model.tt_ccl.reset_gather_and_buffer_idx()

    # Create dummy input tokens
    tokens = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)

    # Create page table
    block_size = 64
    num_blocks = (seq_len + block_size - 1) // block_size
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    # Compute RoPE
    logger.info("  Computing RoPE...")
    ttnn_cos, ttnn_sin, _ = precompute_freqs_yarn(
        seq_len,
        model_args.head_dim,
        model_args.yarn_factor,
        model_args.yarn_original_context_len,
        model_args.yarn_beta_fast,
        model_args.yarn_beta_slow,
        model_args.yarn_attention_factor,
    )
    position_ids = torch.arange(seq_len)
    cos_gathered, sin_gathered = gather_cos_sin(position_ids, ttnn_cos, ttnn_sin)
    rot_mats_prefill = [
        ttnn.from_torch(
            cos_gathered.unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        ttnn.from_torch(
            sin_gathered.unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
    ]

    # Prepare inputs
    logger.info("  Preparing inputs...")
    host_inputs = tt_model.prepare_prefill_inputs_host(tokens, user_id=0, page_table=page_table)
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
    transformed_inputs = tt_model.transform_prefill_inputs_device(*device_inputs)

    # Run prefill (eager mode - no trace)
    logger.info("  Running prefill forward (eager mode)...")
    start_time = time()

    try:
        tt_out = tt_model.ttnn_prefill_forward(
            *transformed_inputs,
            kv_cache=kv_cache,
            batch_size=1,
        )

        # Synchronize to ensure completion
        logger.info("  Synchronizing device...")
        ttnn.synchronize_device(mesh_device)

        elapsed = time() - start_time
        logger.info(f"  SUCCESS: Prefill completed in {elapsed:.2f}s")

        # Cleanup RoPE tensors
        for rm in rot_mats_prefill:
            rm.deallocate()

        return True, elapsed, None

    except Exception as e:
        elapsed = time() - start_time
        logger.error(f"  FAILED: {e}")
        return False, elapsed, str(e)


@pytest.mark.parametrize(
    "seq_len",
    [
        pytest.param(4096, id="4096-traced"),
        pytest.param(8192, id="8192-eager"),
        pytest.param(16384, id="16384-eager"),
    ],
)
def test_olmo_eager_prefill(model_and_cache, mesh_device, seq_len):
    """Test OLMo prefill at various ISLs to identify where eager mode hangs."""
    tt_model, kv_cache, model_args = model_and_cache

    success, elapsed, error = run_eager_prefill(tt_model, kv_cache, mesh_device, model_args, seq_len)

    assert success, f"Prefill at ISL={seq_len} failed: {error}"

    # Log performance
    tokens_per_second = seq_len / elapsed if elapsed > 0 else 0
    logger.info(f"\nPerformance: {tokens_per_second:.0f} prefill tok/s, TTFT={elapsed*1000:.0f}ms")
