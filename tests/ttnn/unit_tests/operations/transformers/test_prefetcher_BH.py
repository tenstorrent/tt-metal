# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Prefetcher with ring matmul on Blackhole.
These tests use the Prefetcher class from models/tt_transformers/tt/prefetcher.py
and the prefetcher matmul/memory configs from models/tt_transformers/tt/model_config.py.

The test runs all 5 matmuls: QKV, WO, FF1, FF3, FF2 for X number of layers.
Weights are tensor-parallelized across devices:
- QKV: TP on qkv_size dimension (N-sharded)
- WO: TP on n_heads*head_dim dimension (K-sharded)
- FF1, FF3: TP on hidden_dim dimension (N-sharded)
- FF2: TP on hidden_dim dimension (K-sharded)
"""

import math
import os
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_blackhole
from models.tt_transformers.tt.prefetcher import (
    Prefetcher,
    VERIFIED_MODEL_CONFIGS,
    is_prefetcher_supported,
)
from models.tt_transformers.tt.common import Mode
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def round_up(n, multiple):
    """Round up n to the nearest multiple."""
    return ((n + multiple - 1) // multiple) * multiple


def create_weight_tensors(
    mesh_device,
    model_dims,
    num_layers,
    prefetcher,
):
    """
    Create weight tensors for all 5 matmuls with tensor parallelism:
    - QKV: TP on qkv_size dimension (N-sharded)
    - WO: TP on n_heads*head_dim dimension (K-sharded)
    - FF1, FF3: TP on hidden_dim dimension (N-sharded)
    - FF2: TP on hidden_dim dimension (K-sharded)

    Dimensions are padded to be divisible by ring_size (total receiver cores).

    Returns the weight tensors and their PyTorch equivalents for verification.
    """

    num_devices = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)
    ring_size = prefetcher.ring_size
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    n_heads = model_dims["n_heads"]
    n_kv_heads = model_dims["n_kv_heads"]
    head_dim = dim // n_heads
    qkv_size = head_dim * (n_heads + 2 * n_kv_heads)
    dram_cores = len(prefetcher.dram_banks())

    # Verify dimensions are divisible by num_devices for TP
    assert n_heads % num_devices == 0, f"n_heads {n_heads} must be divisible by num_devices {num_devices}"
    assert n_kv_heads % num_devices == 0, f"n_kv_heads {n_kv_heads} must be divisible by num_devices {num_devices}"
    assert qkv_size % num_devices == 0, f"qkv_size {qkv_size} must be divisible by num_devices {num_devices}"
    assert dim % num_devices == 0, f"dim {dim} must be divisible by num_devices {num_devices}"
    assert hidden_dim % num_devices == 0, f"hidden_dim {hidden_dim} must be divisible by num_devices {num_devices}"

    # Helper to pad N dimension to be divisible by ring_size
    # Only N (width) needs explicit padding - K (height) is handled internally by the prefetcher
    def pad_n_to_ring_size(n_size):
        """Pad N dimension to be divisible by ring_size * TILE_SIZE."""
        per_core = round_up(math.ceil(n_size / ring_size), ttnn.TILE_SIZE)
        return per_core * ring_size

    # Create DRAM sharded memory configs for weights
    # K uses original dimension (prefetcher handles internal rounding)
    # N must be padded to be divisible by ring_size for equal distribution across matmul cores
    def create_dram_sharded_mem_config(k, n_padded):
        """Create DRAM sharded memory config with original K and pre-padded N dimension."""
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
        shard_spec = ttnn.ShardSpec(dram_grid, (k, n_padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    # Weight configs: (name, K, N, dtype, shard_dims, shard_type)
    # shard_dims: (dim_for_mesh_dim_0, dim_for_mesh_dim_1)
    # dims=(2, 3) -> shard tensor dim 3 across mesh dim 1 (N-sharded)
    # dims=(3, 2) -> shard tensor dim 2 across mesh dim 1 (K-sharded)
    # shard_type: 'N' for N-sharded (column), 'K' for K-sharded (row)
    weight_configs = [
        # QKV: [dim, qkv_size] -> TP on qkv_size (N-sharded)
        ("qkv", dim, qkv_size, ttnn.bfloat8_b, (2, 3), "N"),
        # WO: [n_heads*head_dim, dim] -> TP on n_heads*head_dim (K-sharded)
        ("wo", n_heads * head_dim, dim, ttnn.bfloat8_b, (3, 2), "K"),
        # FF1: [dim, hidden_dim] -> TP on hidden_dim (N-sharded)
        ("ff1", dim, hidden_dim, ttnn.bfloat8_b, (2, 3), "N"),
        # FF3: [dim, hidden_dim] -> TP on hidden_dim (N-sharded)
        ("ff3", dim, hidden_dim, ttnn.bfloat8_b, (2, 3), "N"),
        # FF2: [hidden_dim, dim] -> TP on hidden_dim (K-sharded)
        ("ff2", hidden_dim, dim, ttnn.bfloat8_b, (3, 2), "K"),
    ]

    pt_weights = {}  # PyTorch weights for verification (full, unsharded)
    tt_weights = {}  # TT weights (sharded across devices)
    weight_metadata = {}  # Store shard info for verification

    for layer_idx in range(num_layers):
        for name, k, n, dtype, shard_dims, shard_type in weight_configs:
            key = f"layer_{layer_idx}_{name}"
            # N-sharded: [K, N/num_devices], K-sharded: [K/num_devices, N]
            is_n_shard = shard_type == "N"
            k_per_device = k if is_n_shard else k // num_devices
            n_padded = pad_n_to_ring_size(n // num_devices if is_n_shard else n)
            mem_config = create_dram_sharded_mem_config(k_per_device, n_padded)
            pt_weight_2d = torch.randn(k, n_padded * num_devices if is_n_shard else n_padded)
            pt_weights[key] = pt_weight_2d
            # Convert to 4D for TT tensor: [1, 1, K, N]
            pt_weight_4d = pt_weight_2d.unsqueeze(0).unsqueeze(0)
            if name not in weight_metadata:
                weight_metadata[name] = {
                    "shard_type": shard_type,
                    "k": k,
                    "n": n,
                    "k_per_device": k_per_device,
                    "n_per_device": n_padded,
                }
            logger.info(f"Weight {name}: original ({k}, {n}) -> per device ({k_per_device}, {n_padded})")
            tt_weight = ttnn.as_tensor(
                pt_weight_4d,
                device=mesh_device,
                dtype=dtype,
                memory_config=mem_config,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device,
                    dims=shard_dims,
                    mesh_shape=mesh_shape,
                ),
            )
            tt_weights[key] = tt_weight

            # Insert tensor into prefetcher
            prefetcher.insert_tensor(tt_weight)

    return pt_weights, tt_weights, weight_metadata


def create_matmul_program_configs(model_dims, prefetcher, mesh_device, weight_metadata):
    """
    Create matmul program configs for the prefetcher ring matmuls.
    Uses original K dimension and padded N dimension.
    """
    ring_size = prefetcher.ring_size
    num_receiver_cores = prefetcher.num_receiver_cores
    num_dram_banks = len(prefetcher.dram_banks())
    assert (
        ring_size % num_dram_banks == 0
    ), f"ring_size {ring_size} must be divisible by num_dram_banks {num_dram_banks}"

    def create_ring_config(M, K, N_padded, num_cores, num_global_cb_receivers, num_dram_banks, untilize_out=False):
        """Create ring matmul config
        K: original weight K dimension
        N_padded: padded output N dimension
        """
        # in0_block_w uses original K
        in0_block_w = K // num_cores // ttnn.TILE_SIZE
        while in0_block_w > 0 and (K / ttnn.TILE_SIZE) % in0_block_w != 0:
            in0_block_w -= 1
        if in0_block_w == 0:
            in0_block_w = 1
        out_block_h = M // ttnn.TILE_SIZE
        out_block_w = N_padded // num_cores // ttnn.TILE_SIZE
        out_subblock_h = 1
        out_subblock_w = 8
        while out_block_w % out_subblock_w != 0:
            out_subblock_w -= 1
        # Calculate grid size from num_cores
        grid = ttnn.CoreGrid(y=num_cores // num_dram_banks, x=num_dram_banks)
        hop_grid = []
        hop_core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in hop_grid
            }
        )
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=hop_core_range_set,
            num_global_cb_receivers=num_global_cb_receivers,
            untilize_out=untilize_out,
        )

    M = 32  # Batch size for decode

    # Create configs using original K (for block width) and padded N (for output)
    configs = {}
    for name in ["qkv", "wo", "ff1", "ff3", "ff2"]:
        meta = weight_metadata[name]
        k_original = meta["k_per_device"]  # Original K per device
        n_padded = meta["n_per_device"]  # Already padded N per device
        untilize = name == "qkv"
        configs[name] = create_ring_config(
            M, k_original, n_padded, ring_size, num_receiver_cores, num_dram_banks, untilize_out=untilize
        )
        logger.info(f"Config {name}: K={k_original}, N_pad={n_padded}, ring_size={ring_size}")

    return configs


def create_input_tensors(mesh_device, model_dims, prefetcher, weight_metadata):
    """
    Create input tensors for the matmuls.
    For K-sharded matmuls (WO, FF2), inputs need to be sharded across devices.
    For N-sharded matmuls (QKV, FF1, FF3), inputs are replicated.
    """
    ring_size = prefetcher.ring_size
    num_devices = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)

    # Input memory configs - sharded on receiver cores within each device
    receiver_core_range_set = prefetcher.to_core_range_set(
        prefetcher.receiver_cores(sender_active=True, receiver_active=True)
    )

    def create_input_mem_config(k_per_shard):
        """Create input memory config with K_per_shard (padded per core)."""
        return ttnn.create_sharded_memory_config(
            shape=(32, k_per_shard),
            core_grid=receiver_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    qkv_k = weight_metadata["qkv"]["k_per_device"]  # dim (original)
    wo_k = weight_metadata["wo"]["k_per_device"]  # n_heads*head_dim/num_devices (original)
    ff1_k = weight_metadata["ff1"]["k_per_device"]  # dim (original)
    ff2_k = weight_metadata["ff2"]["k_per_device"]  # hidden_dim/num_devices (original)

    # Calculate K_per_shard for memory config (padded to tile size per core)
    def calc_k_per_shard(k):
        return round_up(math.ceil(k / ring_size), ttnn.TILE_SIZE)

    qkv_k_per_shard = calc_k_per_shard(qkv_k)
    wo_k_per_shard = calc_k_per_shard(wo_k)
    ff1_k_per_shard = calc_k_per_shard(ff1_k)
    ff2_k_per_shard = calc_k_per_shard(ff2_k)

    logger.info(
        f"Input K dimensions: QKV {qkv_k} (shard={qkv_k_per_shard}), WO {wo_k} (shard={wo_k_per_shard}), FF1 {ff1_k} (shard={ff1_k_per_shard}), FF2 {ff2_k} (shard={ff2_k_per_shard})"
    )

    pt_inputs = {
        "attn_input": torch.randn(1, 1, 32, qkv_k),  # Used for QKV (replicated)
        "mlp_input": torch.randn(1, 1, 32, ff1_k),  # Used for FF1, FF3 (replicated)
        # WO input: K-sharded across devices, each device gets [1, 1, 32, wo_k]
        "wo_input": torch.randn(1, 1, 32, wo_k * num_devices),
        # FF2 input: K-sharded across devices, each device gets [1, 1, 32, ff2_k]
        "ff2_input": torch.randn(1, 1, 32, ff2_k * num_devices),
    }

    tt_inputs = {}

    # Replicated inputs (for N-sharded matmuls)
    # attn_input for QKV
    attn_mem_config = create_input_mem_config(qkv_k_per_shard)
    tt_inputs["attn_input"] = ttnn.from_torch(
        pt_inputs["attn_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=attn_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # mlp_input for FF1, FF3
    mlp_mem_config = create_input_mem_config(ff1_k_per_shard)
    tt_inputs["mlp_input"] = ttnn.from_torch(
        pt_inputs["mlp_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mlp_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # K-sharded inputs (for K-sharded matmuls: WO, FF2)
    # WO input: shard on dim 3 (n_heads*head_dim dimension)
    wo_mem_config = create_input_mem_config(wo_k_per_shard)
    tt_inputs["wo_input"] = ttnn.from_torch(
        pt_inputs["wo_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=wo_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3), mesh_shape=mesh_shape),
    )

    # FF2 input: shard on dim 3 (hidden_dim dimension)
    ff2_mem_config = create_input_mem_config(ff2_k_per_shard)
    tt_inputs["ff2_input"] = ttnn.from_torch(
        pt_inputs["ff2_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ff2_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3), mesh_shape=mesh_shape),
    )

    return pt_inputs, tt_inputs


def create_output_mem_configs(prefetcher, weight_metadata):
    """
    Create output memory configs for the matmuls.
    Uses padded N dimensions from weight_metadata.
    """
    ring_size = prefetcher.ring_size

    receiver_core_range_set = prefetcher.to_core_range_set(
        prefetcher.receiver_cores(sender_active=True, receiver_active=True)
    )

    def create_output_mem_config(n_padded):
        """Create output memory config with pre-padded N dimension."""
        return ttnn.create_sharded_memory_config(
            shape=(32, n_padded // ring_size),
            core_grid=receiver_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # Output sizes per device (use padded N from weight_metadata)
    # QKV: N-sharded -> output [32, n_per_device] per device
    # WO: K-sharded -> output [32, n_per_device] per device (partial, needs all-reduce)
    # FF1/FF3: N-sharded -> output [32, n_per_device] per device
    # FF2: K-sharded -> output [32, n_per_device] per device (partial, needs all-reduce)
    return {
        "qkv": create_output_mem_config(weight_metadata["qkv"]["n_per_device"]),
        "wo": create_output_mem_config(weight_metadata["wo"]["n_per_device"]),
        "ff1": create_output_mem_config(weight_metadata["ff1"]["n_per_device"]),
        "ff3": create_output_mem_config(weight_metadata["ff3"]["n_per_device"]),
        "ff2": create_output_mem_config(weight_metadata["ff2"]["n_per_device"]),
    }


def run_prefetcher_all_matmuls(
    mesh_device,
    num_layers,
    model_dims,
    enable_trace=True,
    receiver_mapping_override=None,
    num_receiver_cores=None,
):
    """
    Run the prefetcher with all 5 matmuls (QKV, WO, FF1, FF3, FF2) for num_layers.
    Weights are tensor-parallelized across devices.

    Args:
        receiver_mapping_override: Optional dict mapping sender cores to receiver cores.
            If provided, keys become sender cores and values become their receivers.
    """
    logger.info(f"Running prefetcher test with {num_layers} layers on Blackhole with tensor parallelism")

    num_devices = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)
    logger.info(f"Mesh shape: {mesh_shape}, num_devices: {num_devices}")

    # Number of tensors per layer: QKV, WO, FF1, FF3, FF2 = 5
    num_tensors_per_layer = 5

    # Initialize prefetcher
    prefetcher = Prefetcher(
        mesh_device=mesh_device,
        num_tensors=num_tensors_per_layer,
        num_layers=num_layers,
        num_receiver_cores=num_receiver_cores,
    )

    # Initialize prefetcher for decode mode
    prefetcher.init(mode=Mode.DECODE)

    # Get worker sub device id
    worker_sub_device_id = prefetcher.worker_sub_device_id

    # Create weight tensors (this also inserts them into prefetcher)
    pt_weights, tt_weights, weight_metadata = create_weight_tensors(mesh_device, model_dims, num_layers, prefetcher)

    # Create program configs using padded dimensions from weight_metadata
    program_configs = create_matmul_program_configs(model_dims, prefetcher, mesh_device, weight_metadata)

    # Create input and output memory configs using padded dimensions
    pt_inputs, tt_inputs = create_input_tensors(mesh_device, model_dims, prefetcher, weight_metadata)
    output_mem_configs = create_output_mem_configs(prefetcher, weight_metadata)

    # Compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    def run_op():
        # Start prefetcher op
        prefetcher.run()

        """Run all matmuls for all layers."""
        outputs_all = []

        for layer_idx in range(num_layers):
            layer_outputs = {}

            # Run QKV matmul (N-sharded weight)
            # Input: replicated, Weight: N-sharded, Output: N-sharded
            qkv_out = ttnn.linear(
                tt_inputs["attn_input"],
                tt_weights[f"layer_{layer_idx}_qkv"],
                program_config=program_configs["qkv"],
                memory_config=output_mem_configs["qkv"],
                compute_kernel_config=compute_kernel_config,
                dtype=ttnn.bfloat16,
                global_cb=prefetcher.global_cb,
                sub_device_id=worker_sub_device_id,
            )
            layer_outputs["qkv"] = ttnn.to_memory_config(qkv_out, ttnn.DRAM_MEMORY_CONFIG)

            # Run WO matmul (K-sharded weight)
            # Input: K-sharded (wo_input), Weight: K-sharded, Output: partial (needs all-reduce)
            wo_out = ttnn.linear(
                tt_inputs["wo_input"],
                tt_weights[f"layer_{layer_idx}_wo"],
                program_config=program_configs["wo"],
                memory_config=output_mem_configs["wo"],
                compute_kernel_config=compute_kernel_config,
                dtype=ttnn.bfloat16,
                global_cb=prefetcher.global_cb,
                sub_device_id=worker_sub_device_id,
            )
            layer_outputs["wo"] = ttnn.to_memory_config(wo_out, ttnn.DRAM_MEMORY_CONFIG)

            # Run FF1 matmul (N-sharded weight)
            # Input: replicated, Weight: N-sharded, Output: N-sharded
            ff1_out = ttnn.linear(
                tt_inputs["mlp_input"],
                tt_weights[f"layer_{layer_idx}_ff1"],
                program_config=program_configs["ff1"],
                memory_config=output_mem_configs["ff1"],
                compute_kernel_config=compute_kernel_config,
                dtype=ttnn.bfloat16,
                global_cb=prefetcher.global_cb,
                sub_device_id=worker_sub_device_id,
            )
            layer_outputs["ff1"] = ttnn.to_memory_config(ff1_out, ttnn.DRAM_MEMORY_CONFIG)

            # Run FF3 matmul (N-sharded weight)
            # Input: replicated, Weight: N-sharded, Output: N-sharded
            ff3_out = ttnn.linear(
                tt_inputs["mlp_input"],
                tt_weights[f"layer_{layer_idx}_ff3"],
                program_config=program_configs["ff3"],
                memory_config=output_mem_configs["ff3"],
                compute_kernel_config=compute_kernel_config,
                dtype=ttnn.bfloat16,
                global_cb=prefetcher.global_cb,
                sub_device_id=worker_sub_device_id,
            )
            layer_outputs["ff3"] = ttnn.to_memory_config(ff3_out, ttnn.DRAM_MEMORY_CONFIG)

            # Run FF2 matmul (K-sharded weight)
            # Input: K-sharded (ff2_input), Weight: K-sharded, Output: partial (needs all-reduce)
            ff2_out = ttnn.linear(
                tt_inputs["ff2_input"],
                tt_weights[f"layer_{layer_idx}_ff2"],
                program_config=program_configs["ff2"],
                memory_config=output_mem_configs["ff2"],
                compute_kernel_config=compute_kernel_config,
                dtype=ttnn.bfloat16,
                global_cb=prefetcher.global_cb,
                sub_device_id=worker_sub_device_id,
            )
            layer_outputs["ff2"] = ttnn.to_memory_config(ff2_out, ttnn.DRAM_MEMORY_CONFIG)

            outputs_all.append(layer_outputs)

        mesh_device.reset_sub_device_stall_group()
        return outputs_all

    # Compile
    logger.info("Compiling model...")
    outputs = run_op()

    if enable_trace:
        # Capture trace
        logger.info("Capturing trace...")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outputs = run_op()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        # Execute trace
        logger.info("Executing trace...")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    # Verify results using per-device comparison (same as attention test debug)
    logger.info("Verifying results...")
    all_passing = True

    for layer_idx in range(num_layers):
        for matmul_name in ["qkv", "wo", "ff1", "ff3", "ff2"]:
            shard_type = weight_metadata[matmul_name]["shard_type"]

            # Get the appropriate TT input tensor
            if matmul_name == "qkv":
                tt_input = tt_inputs["attn_input"]
            elif matmul_name == "wo":
                tt_input = tt_inputs["wo_input"]
            elif matmul_name in ["ff1", "ff3"]:
                tt_input = tt_inputs["mlp_input"]
            else:  # ff2
                tt_input = tt_inputs["ff2_input"]

            # Set PCC threshold (all weights use bfloat8_b)
            pcc_threshold = 0.99

            # Per-device verification
            # This verifies each device's matmul independently
            for device_idx in range(num_devices):
                # Get tensors from specific device
                in0_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_input)[device_idx])
                in1_torch = ttnn.to_torch(
                    ttnn.get_device_tensors(tt_weights[f"layer_{layer_idx}_{matmul_name}"])[device_idx]
                )
                out_ttnn = ttnn.to_torch(ttnn.get_device_tensors(outputs[layer_idx][matmul_name])[device_idx])

                # Expected output: input @ weight (both from same device)
                out_torch = in0_torch.float() @ in1_torch.float()

                passing, output_str = comp_pcc(out_torch, out_ttnn, pcc_threshold)
                logger.info(f"Layer {layer_idx} {matmul_name} ({shard_type}-sharded) device {device_idx}: {output_str}")
                all_passing = passing and all_passing

    # Cleanup
    prefetcher.stop()

    # Deallocate all tensors
    for tt_input in tt_inputs.values():
        ttnn.deallocate(tt_input)
    for tt_weight in tt_weights.values():
        ttnn.deallocate(tt_weight)
    for layer_outputs in outputs:
        for out in layer_outputs.values():
            ttnn.deallocate(out)

    # Clean up sub device manager
    mesh_device.clear_loaded_sub_device_manager()

    assert all_passing, "PCC check failed for one or more matmuls"


# =============================================================================
# Test Cases
# =============================================================================
@pytest.mark.skipif(not is_blackhole(), reason="This test only runs on Blackhole")
@pytest.mark.parametrize("enable_trace", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_receiver_cores",
    [1, 2, 3, 8, 10],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "Llama-3.2-1B",
        "Llama-3.2-3B",
        "Llama-3.1-8B",
        "Llama-3.3-70B",
        "Qwen3-32B",
        "Qwen3-VL-7B",
        "Qwen3-VL-14B",
        "Qwen3-VL-72B",
        "Gemma3-4B",
        "Gemma3-27B",
    ],
)
def test_prefetcher_BH(
    mesh_device,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
    num_receiver_cores,
    model_name,
    enable_trace,
):
    """
    Test prefetcher with tensor-parallelized weights on Blackhole.
    Automatically detects the mesh shape and uses corresponding model dimensions.
    Supported mesh shapes: 1x1 (P150), 1x2 (P300), 1x4 (P150x4 or P300x2), 1x8 (P150x8)
    Parameters:
        use_custom_mapping: If False, uses default prefetcher (column 0/7 senders, 2 receivers each).
                           If True, uses custom 64 or 80-core mapping (8 senders, 8 or 10 receivers each).
    Tensor parallelism:
    - QKV: TP on qkv_size dimension (N-sharded)
    - WO: TP on n_heads*head_dim dimension (K-sharded)
    - FF1, FF3: TP on hidden_dim dimension (N-sharded)
    - FF2: TP on hidden_dim dimension (K-sharded)
    """
    mesh_shape = tuple(mesh_device.shape)
    if not is_prefetcher_supported(model_name, mesh_device.get_num_devices(), num_receiver_cores * 8):
        pytest.skip(
            f"Model {model_name} does not fit in global CB with {mesh_device.get_num_devices()} devices and num_receiver_cores={num_receiver_cores}"
        )
    logger.info(
        f"Testing DRAM Prefetcher + Ring Matmul for model {model_name} with dimensions: {VERIFIED_MODEL_CONFIGS[model_name]}"
    )
    os.environ["HF_MODEL"] = model_name
    run_prefetcher_all_matmuls(
        mesh_device=mesh_device,
        num_layers=1,
        model_dims=VERIFIED_MODEL_CONFIGS[model_name],
        enable_trace=enable_trace,
        num_receiver_cores=num_receiver_cores,
    )
