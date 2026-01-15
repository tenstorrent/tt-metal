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
from models.tt_transformers.tt.prefetcher import Prefetcher
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def round_up(n, multiple):
    """Round up n to the nearest multiple."""
    return ((n + multiple - 1) // multiple) * multiple


@pytest.fixture
def model_dims():
    """
    Model dimensions for testing.
    These are based on Llama-3.1-8B dimensions.
    """
    # Llama-3.1-8B dimensions
    return {
        "dim": 4096,  # Model dimension
        "hidden_dim": 14336,  # MLP hidden dimension
        "n_heads": 32,  # Number of attention heads
        "n_kv_heads": 8,  # Number of KV heads
        "head_dim": 128,  # Head dimension (dim // n_heads)
    }


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

    Returns the weight tensors and their PyTorch equivalents for verification.
    """
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    n_heads = model_dims["n_heads"]
    n_kv_heads = model_dims["n_kv_heads"]
    head_dim = model_dims["head_dim"]

    num_devices = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)

    # Calculate QKV size: Q heads + K heads + V heads
    qkv_size = head_dim * (n_heads + 2 * n_kv_heads)

    # Verify dimensions are divisible by num_devices for TP
    assert n_heads % num_devices == 0, f"n_heads {n_heads} must be divisible by num_devices {num_devices}"
    assert n_kv_heads % num_devices == 0, f"n_kv_heads {n_kv_heads} must be divisible by num_devices {num_devices}"
    assert qkv_size % num_devices == 0, f"qkv_size {qkv_size} must be divisible by num_devices {num_devices}"
    assert dim % num_devices == 0, f"dim {dim} must be divisible by num_devices {num_devices}"
    assert hidden_dim % num_devices == 0, f"hidden_dim {hidden_dim} must be divisible by num_devices {num_devices}"

    # Number of receiver cores
    dram_cores = mesh_device.dram_grid_size().x

    # Create DRAM sharded memory configs for weights
    def create_dram_sharded_mem_config(k, n):
        padded_n = math.ceil(n / (32 * dram_cores)) * (32 * dram_cores)
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
        shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    # Weight configs: (name, K, N, dtype, shard_dims, shard_type)
    # shard_dims: (dim_for_mesh_dim_0, dim_for_mesh_dim_1)
    # For 1x2 mesh: mesh dim 0 has size 1 (no sharding), mesh dim 1 has size 2 (actual sharding)
    # dims=(2, 3) -> shard tensor dim 3 across mesh dim 1 (N-sharded)
    # dims=(3, 2) -> shard tensor dim 2 across mesh dim 1 (K-sharded)
    # shard_type: 'N' for N-sharded (column), 'K' for K-sharded (row)
    weight_configs = [
        # QKV: [dim, qkv_size] -> TP on qkv_size (N-sharded)
        ("qkv", dim, qkv_size, ttnn.bfloat8_b, (2, 3), "N"),
        # WO: [n_heads*head_dim, dim] -> TP on n_heads*head_dim (K-sharded)
        ("wo", n_heads * head_dim, dim, ttnn.bfloat8_b, (3, 2), "K"),
        # FF1: [dim, hidden_dim] -> TP on hidden_dim (N-sharded)
        ("ff1", dim, hidden_dim, ttnn.bfloat4_b, (2, 3), "N"),
        # FF3: [dim, hidden_dim] -> TP on hidden_dim (N-sharded)
        ("ff3", dim, hidden_dim, ttnn.bfloat4_b, (2, 3), "N"),
        # FF2: [hidden_dim, dim] -> TP on hidden_dim (K-sharded)
        ("ff2", hidden_dim, dim, ttnn.bfloat8_b, (3, 2), "K"),
    ]

    pt_weights = {}  # PyTorch weights for verification (full, unsharded)
    tt_weights = {}  # TT weights (sharded across devices)
    weight_metadata = {}  # Store shard info for verification

    for layer_idx in range(num_layers):
        for name, k, n, dtype, shard_dims, shard_type in weight_configs:
            key = f"layer_{layer_idx}_{name}"

            # Create full weight tensor (2D for PT computation)
            pt_weight_2d = torch.randn(k, n)
            pt_weights[key] = pt_weight_2d

            # Convert to 4D for TT tensor: [1, 1, K, N]
            pt_weight_4d = pt_weight_2d.unsqueeze(0).unsqueeze(0)

            # Determine sharded dimension size for memory config
            if shard_type == "N":
                # N-sharded: each device has [K, N/num_devices]
                sharded_n = n // num_devices
                mem_config = create_dram_sharded_mem_config(k, sharded_n)
            else:  # K-sharded
                # K-sharded: each device has [K/num_devices, N]
                sharded_k = k // num_devices
                mem_config = create_dram_sharded_mem_config(sharded_k, n)

            weight_metadata[key] = {"shard_type": shard_type, "k": k, "n": n}

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


def create_matmul_program_configs(model_dims, prefetcher, mesh_device):
    """
    Create matmul program configs for the prefetcher ring matmuls.
    These are based on the configs in model_config.py.
    Account for tensor parallelism in weight dimensions.
    """
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    n_heads = model_dims["n_heads"]
    n_kv_heads = model_dims["n_kv_heads"]
    head_dim = model_dims["head_dim"]

    num_devices = mesh_device.get_num_devices()

    qkv_size = head_dim * (n_heads + 2 * n_kv_heads)
    ring_size = prefetcher.ring_size
    num_receiver_cores = prefetcher.num_receiver_cores

    def create_ring_config(M, K, N, num_cores, num_global_cb_receivers, untilize_out=False):
        """Create ring matmul config similar to matmul_1d_ring_config in model_config.py"""
        in0_block_w = K // num_cores // ttnn.TILE_SIZE
        out_block_h = M // ttnn.TILE_SIZE
        out_block_w = N // num_cores // ttnn.TILE_SIZE

        out_subblock_h = 1
        out_subblock_w = 8
        while out_block_w % out_subblock_w != 0:
            out_subblock_w -= 1

        # Calculate grid size from num_cores
        if num_cores % 8 == 0:
            grid = ttnn.CoreGrid(y=num_cores // 8, x=8)
        elif num_cores == 12:
            grid = ttnn.CoreGrid(y=2, x=6)
        elif num_cores == 20:
            grid = ttnn.CoreGrid(y=4, x=5)
        else:
            grid = ttnn.CoreGrid(y=1, x=num_cores)

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

    # TP dimensions per device:
    # QKV: K=dim, N=qkv_size/num_devices (N-sharded)
    # WO: K=n_heads*head_dim/num_devices, N=dim (K-sharded)
    # FF1/FF3: K=dim, N=hidden_dim/num_devices (N-sharded)
    # FF2: K=hidden_dim/num_devices, N=dim (K-sharded)

    configs = {
        "qkv": create_ring_config(M, dim, qkv_size // num_devices, ring_size, num_receiver_cores, untilize_out=True),
        "wo": create_ring_config(M, (n_heads * head_dim) // num_devices, dim, ring_size, num_receiver_cores),
        "ff1": create_ring_config(M, dim, hidden_dim // num_devices, ring_size, num_receiver_cores),
        "ff3": create_ring_config(M, dim, hidden_dim // num_devices, ring_size, num_receiver_cores),
        "ff2": create_ring_config(M, hidden_dim // num_devices, dim, ring_size, num_receiver_cores),
    }

    return configs


def create_input_tensors(mesh_device, model_dims, prefetcher):
    """
    Create input tensors for the matmuls.
    For K-sharded matmuls (WO, FF2), inputs need to be sharded across devices.
    For N-sharded matmuls (QKV, FF1, FF3), inputs are replicated.
    """
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    n_heads = model_dims["n_heads"]
    head_dim = model_dims["head_dim"]
    ring_size = prefetcher.ring_size
    num_devices = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)

    # Input memory configs - sharded on receiver cores within each device
    receiver_core_range_set = prefetcher.to_core_range_set(
        prefetcher.receiver_cores(sender_active=True, receiver_active=True)
    )

    def create_input_mem_config(dim_size):
        return ttnn.create_sharded_memory_config(
            shape=(32, dim_size // ring_size),
            core_grid=receiver_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # Create input tensors
    # For attn_input and mlp_input: replicated across devices (used by N-sharded QKV, FF1, FF3)
    # For wo_input: K-sharded across devices (n_heads*head_dim is sharded)
    # For ff2_input: K-sharded across devices (hidden_dim is sharded)
    pt_inputs = {
        "attn_input": torch.randn(1, 1, 32, dim),
        "mlp_input": torch.randn(1, 1, 32, dim),
        # WO input: shape [1, 1, 32, n_heads*head_dim] -> each device gets [1, 1, 32, n_heads*head_dim/num_devices]
        "wo_input": torch.randn(1, 1, 32, n_heads * head_dim),
        # FF2 input: shape [1, 1, 32, hidden_dim] -> each device gets [1, 1, 32, hidden_dim/num_devices]
        "ff2_input": torch.randn(1, 1, 32, hidden_dim),
    }

    tt_inputs = {}

    # Replicated inputs (for N-sharded matmuls)
    for name in ["attn_input", "mlp_input"]:
        pt_tensor = pt_inputs[name]
        input_dim = pt_tensor.shape[-1]
        mem_config = create_input_mem_config(input_dim)
        tt_tensor = ttnn.from_torch(
            pt_tensor,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_inputs[name] = tt_tensor

    # K-sharded inputs (for K-sharded matmuls: WO, FF2)
    # WO input: shard on dim 3 (n_heads*head_dim dimension)
    wo_input_sharded_dim = (n_heads * head_dim) // num_devices
    wo_mem_config = create_input_mem_config(wo_input_sharded_dim)
    tt_inputs["wo_input"] = ttnn.from_torch(
        pt_inputs["wo_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=wo_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3), mesh_shape=mesh_shape),
    )

    # FF2 input: shard on dim 3 (hidden_dim dimension)
    ff2_input_sharded_dim = hidden_dim // num_devices
    ff2_mem_config = create_input_mem_config(ff2_input_sharded_dim)
    tt_inputs["ff2_input"] = ttnn.from_torch(
        pt_inputs["ff2_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ff2_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3), mesh_shape=mesh_shape),
    )

    return pt_inputs, tt_inputs


def create_output_mem_configs(model_dims, prefetcher, mesh_device):
    """
    Create output memory configs for the matmuls.
    Account for tensor parallelism in output dimensions.
    """
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    n_heads = model_dims["n_heads"]
    n_kv_heads = model_dims["n_kv_heads"]
    head_dim = model_dims["head_dim"]

    num_devices = mesh_device.get_num_devices()

    qkv_size = head_dim * (n_heads + 2 * n_kv_heads)
    ring_size = prefetcher.ring_size

    receiver_core_range_set = prefetcher.to_core_range_set(
        prefetcher.receiver_cores(sender_active=True, receiver_active=True)
    )

    def create_output_mem_config(n_size):
        return ttnn.create_sharded_memory_config(
            shape=(32, n_size // ring_size),
            core_grid=receiver_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # Output sizes per device:
    # QKV: N-sharded -> output [32, qkv_size/num_devices] per device
    # WO: K-sharded -> output [32, dim] per device (partial, needs all-reduce)
    # FF1/FF3: N-sharded -> output [32, hidden_dim/num_devices] per device
    # FF2: K-sharded -> output [32, dim] per device (partial, needs all-reduce)
    return {
        "qkv": create_output_mem_config(qkv_size // num_devices),
        "wo": create_output_mem_config(dim),
        "ff1": create_output_mem_config(hidden_dim // num_devices),
        "ff3": create_output_mem_config(hidden_dim // num_devices),
        "ff2": create_output_mem_config(dim),
    }


def run_prefetcher_all_matmuls(
    mesh_device,
    num_layers,
    model_dims,
    enable_trace=True,
):
    """
    Run the prefetcher with all 5 matmuls (QKV, WO, FF1, FF3, FF2) for num_layers.
    Weights are tensor-parallelized across devices.
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
    )

    # Initialize prefetcher for decode mode
    prefetcher.init(mode="decode")

    # Get worker sub device id
    worker_sub_device_id = prefetcher.worker_sub_device_id

    # Create weight tensors (this also inserts them into prefetcher)
    pt_weights, tt_weights, weight_metadata = create_weight_tensors(mesh_device, model_dims, num_layers, prefetcher)

    # Create program configs
    program_configs = create_matmul_program_configs(model_dims, prefetcher, mesh_device)

    # Create input and output memory configs
    pt_inputs, tt_inputs = create_input_tensors(mesh_device, model_dims, prefetcher)
    output_mem_configs = create_output_mem_configs(model_dims, prefetcher, mesh_device)

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
            shard_type = weight_metadata[f"layer_{layer_idx}_{matmul_name}"]["shard_type"]

            # Get the appropriate TT input tensor
            if matmul_name == "qkv":
                tt_input = tt_inputs["attn_input"]
            elif matmul_name == "wo":
                tt_input = tt_inputs["wo_input"]
            elif matmul_name in ["ff1", "ff3"]:
                tt_input = tt_inputs["mlp_input"]
            else:  # ff2
                tt_input = tt_inputs["ff2_input"]

            # Set PCC threshold based on dtype
            if matmul_name in ["ff1", "ff3"]:  # bfloat4_b
                pcc_threshold = 0.98
            else:  # bfloat8_b
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

    # Clean up sub device manager
    mesh_device.clear_loaded_sub_device_manager()

    assert all_passing, "PCC check failed for one or more matmuls"


# =============================================================================
# Test Cases
# =============================================================================

# Mapping from mesh shape to model dimensions
# Each mesh shape has corresponding dimensions that work with the prefetcher
MESH_SHAPE_TO_MODEL_DIMS = {
    (1, 1): {"dim": 2048, "hidden_dim": 3584, "n_heads": 32, "n_kv_heads": 8},
    (1, 2): {"dim": 4096, "hidden_dim": 7168, "n_heads": 32, "n_kv_heads": 8},
    (1, 4): {"dim": 4096, "hidden_dim": 14336, "n_heads": 32, "n_kv_heads": 8},
    (1, 8): {"dim": 4096, "hidden_dim": 14336, "n_heads": 32, "n_kv_heads": 8},
}


@pytest.mark.skipif(not is_blackhole(), reason="This test only runs on Blackhole")
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
def test_prefetcher_ring_matmul_BH(
    mesh_device,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
):
    """
    Test prefetcher with tensor-parallelized weights on Blackhole.

    Automatically detects the mesh shape and uses corresponding model dimensions.
    Supported mesh shapes: 1x1 (P150), 1x2 (P300), 1x4 (P150x4 or P300x2), 1x8 (P150x8)

    Tensor parallelism:
    - QKV: TP on qkv_size dimension (N-sharded)
    - WO: TP on n_heads*head_dim dimension (K-sharded)
    - FF1, FF3: TP on hidden_dim dimension (N-sharded)
    - FF2: TP on hidden_dim dimension (K-sharded)
    """
    # Get mesh shape from device
    mesh_shape = tuple(mesh_device.shape)

    # Skip if mesh shape is not supported
    if mesh_shape not in MESH_SHAPE_TO_MODEL_DIMS:
        pytest.skip(
            f"Mesh shape {mesh_shape} is not supported. " f"Supported shapes: {list(MESH_SHAPE_TO_MODEL_DIMS.keys())}"
        )

    # Get model dimensions for this mesh shape
    model_dims = MESH_SHAPE_TO_MODEL_DIMS[mesh_shape].copy()
    model_dims["head_dim"] = model_dims["dim"] // model_dims["n_heads"]

    logger.info(f"Running prefetcher test with mesh shape {mesh_shape}")
    logger.info(f"Model dimensions: {model_dims}")

    run_prefetcher_all_matmuls(
        mesh_device=mesh_device,
        num_layers=1,
        model_dims=model_dims,
        enable_trace=False,
    )
