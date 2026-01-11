# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Prefetcher with ring matmul on Blackhole.
These tests use the Prefetcher class from models/tt_transformers/tt/prefetcher.py
and the prefetcher matmul/memory configs from models/tt_transformers/tt/model_config.py.

The test runs all 5 matmuls: QKV, WO, FF1, FF3, FF2 for X number of layers.
"""

import math
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_grayskull, is_wormhole_b0, is_blackhole
from models.tt_transformers.tt.prefetcher import Prefetcher
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def round_up(n, multiple):
    """Round up n to the nearest multiple."""
    return ((n + multiple - 1) // multiple) * multiple


@pytest.fixture
def model_dims():
    """
    Model dimensions for testing.
    These are based on Llama-3.1-70B-like dimensions scaled for testing.
    """
    return {
        "dim": 2048,  # Model dimension (scaled down for testing)
        "hidden_dim": 5632,  # MLP hidden dimension (scaled down)
        "n_heads": 16,  # Number of attention heads
        "n_kv_heads": 8,  # Number of KV heads
        "head_dim": 128,  # Head dimension
    }


def create_weight_tensors(
    mesh_device,
    model_dims,
    num_layers,
    prefetcher,
):
    """
    Create weight tensors for all 5 matmuls: QKV, WO, FF1, FF3, FF2.
    Returns the weight tensors and their PyTorch equivalents for verification.
    """
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    n_heads = model_dims["n_heads"]
    n_kv_heads = model_dims["n_kv_heads"]
    head_dim = model_dims["head_dim"]

    # Calculate QKV size: Q heads + K heads + V heads
    qkv_size = head_dim * (n_heads + 2 * n_kv_heads)

    # Weight shapes:
    # QKV: [dim, qkv_size] - combined Q, K, V projection
    # WO: [dim, dim] - output projection
    # FF1: [dim, hidden_dim] - gate projection
    # FF3: [dim, hidden_dim] - up projection
    # FF2: [hidden_dim, dim] - down projection

    # Number of receiver cores
    ring_size = prefetcher.ring_size
    dram_cores = mesh_device.dram_grid_size().x

    # Create DRAM sharded memory configs for weights
    def create_dram_sharded_mem_config(k, n):
        padded_n = math.ceil(n / (32 * dram_cores)) * (32 * dram_cores)
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
        shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    weight_configs = [
        ("qkv", dim, qkv_size, ttnn.bfloat8_b),
        ("wo", dim, dim, ttnn.bfloat8_b),
        ("ff1", dim, hidden_dim, ttnn.bfloat8_b),
        ("ff3", dim, hidden_dim, ttnn.bfloat8_b),
        ("ff2", hidden_dim, dim, ttnn.bfloat8_b),
    ]

    pt_weights = {}  # PyTorch weights for verification
    tt_weights = {}  # TT weights

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    for layer_idx in range(num_layers):
        for name, k, n, dtype in weight_configs:
            key = f"layer_{layer_idx}_{name}"
            pt_weight = torch.randn(k, n)
            pt_weights[key] = pt_weight

            mem_config = create_dram_sharded_mem_config(k, n)
            tt_weight = ttnn.as_tensor(
                pt_weight,
                device=mesh_device,
                dtype=dtype,
                memory_config=mem_config,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
            )
            tt_weights[key] = tt_weight

            # Insert tensor into prefetcher
            prefetcher.insert_tensor(tt_weight)

    return pt_weights, tt_weights


def create_matmul_program_configs(model_dims, prefetcher, mesh_device):
    """
    Create matmul program configs for the prefetcher ring matmuls.
    These are based on the configs in model_config.py.
    """
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    n_heads = model_dims["n_heads"]
    n_kv_heads = model_dims["n_kv_heads"]
    head_dim = model_dims["head_dim"]

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
            hop_cores=ttnn.CoreRangeSet({}),
            num_global_cb_receivers=num_global_cb_receivers,
            untilize_out=untilize_out,
        )

    M = 32  # Batch size for decode

    configs = {
        "qkv": create_ring_config(M, dim, qkv_size, ring_size, num_receiver_cores, untilize_out=True),
        "wo": create_ring_config(M, dim, dim, ring_size, num_receiver_cores),
        "ff1": create_ring_config(M, dim, hidden_dim, ring_size, num_receiver_cores),
        "ff3": create_ring_config(M, dim, hidden_dim, ring_size, num_receiver_cores),
        "ff2": create_ring_config(M, hidden_dim, dim, ring_size, num_receiver_cores),
    }

    return configs


def create_input_tensors(mesh_device, model_dims, prefetcher):
    """Create input tensors for the matmuls."""
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    ring_size = prefetcher.ring_size

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # Input memory configs - sharded on receiver cores
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
    pt_inputs = {
        "attn_input": torch.randn(1, 1, 32, dim),
        "mlp_input": torch.randn(1, 1, 32, dim),
        "ff2_input": torch.randn(1, 1, 32, hidden_dim),
    }

    tt_inputs = {}
    for name, pt_tensor in pt_inputs.items():
        input_dim = pt_tensor.shape[-1]
        mem_config = create_input_mem_config(input_dim)
        tt_tensor = ttnn.from_torch(
            pt_tensor,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )
        tt_inputs[name] = tt_tensor

    return pt_inputs, tt_inputs


def create_output_mem_configs(model_dims, prefetcher):
    """Create output memory configs for the matmuls."""
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    n_heads = model_dims["n_heads"]
    n_kv_heads = model_dims["n_kv_heads"]
    head_dim = model_dims["head_dim"]

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

    return {
        "qkv": create_output_mem_config(qkv_size),
        "wo": create_output_mem_config(dim),
        "ff1": create_output_mem_config(hidden_dim),
        "ff3": create_output_mem_config(hidden_dim),
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
    """
    logger.info(f"Running prefetcher test with {num_layers} layers on Blackhole")

    # Number of tensors per layer: QKV, WO, FF1, FF3, FF2 = 5
    num_tensors_per_layer = 5

    # Initialize prefetcher
    prefetcher = Prefetcher(
        mesh_device=mesh_device,
        num_tensors=num_tensors_per_layer,
        num_layers=num_layers,
    )

    # Create weight tensors (this also inserts them into prefetcher)
    pt_weights, tt_weights = create_weight_tensors(mesh_device, model_dims, num_layers, prefetcher)

    # Initialize prefetcher for decode mode
    prefetcher.init(mode="decode")

    # Get worker sub device id
    worker_sub_device_id = prefetcher.worker_sub_device_id

    # Create program configs
    program_configs = create_matmul_program_configs(model_dims, prefetcher, mesh_device)

    # Create input and output memory configs
    pt_inputs, tt_inputs = create_input_tensors(mesh_device, model_dims, prefetcher)
    output_mem_configs = create_output_mem_configs(model_dims, prefetcher)

    # Compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape))

    def run_op():
        """Run all matmuls for all layers."""
        # Start prefetcher
        prefetcher.run()

        outputs_all = []

        for layer_idx in range(num_layers):
            layer_outputs = {}

            # Run QKV matmul
            qkv_out = ttnn.matmul(
                tt_inputs["attn_input"],
                tt_weights[f"layer_{layer_idx}_qkv"],
                program_config=program_configs["qkv"],
                memory_config=output_mem_configs["qkv"],
                compute_kernel_config=compute_kernel_config,
                global_cb=prefetcher.global_cb,
                sub_device_id=worker_sub_device_id,
            )
            layer_outputs["qkv"] = ttnn.to_memory_config(qkv_out, ttnn.DRAM_MEMORY_CONFIG)

            # Run WO matmul
            wo_out = ttnn.matmul(
                tt_inputs["attn_input"],
                tt_weights[f"layer_{layer_idx}_wo"],
                program_config=program_configs["wo"],
                memory_config=output_mem_configs["wo"],
                compute_kernel_config=compute_kernel_config,
                global_cb=prefetcher.global_cb,
                sub_device_id=worker_sub_device_id,
            )
            layer_outputs["wo"] = ttnn.to_memory_config(wo_out, ttnn.DRAM_MEMORY_CONFIG)

            # Run FF1 matmul
            ff1_out = ttnn.matmul(
                tt_inputs["mlp_input"],
                tt_weights[f"layer_{layer_idx}_ff1"],
                program_config=program_configs["ff1"],
                memory_config=output_mem_configs["ff1"],
                compute_kernel_config=compute_kernel_config,
                global_cb=prefetcher.global_cb,
                sub_device_id=worker_sub_device_id,
            )
            layer_outputs["ff1"] = ttnn.to_memory_config(ff1_out, ttnn.DRAM_MEMORY_CONFIG)

            # Run FF3 matmul
            ff3_out = ttnn.matmul(
                tt_inputs["mlp_input"],
                tt_weights[f"layer_{layer_idx}_ff3"],
                program_config=program_configs["ff3"],
                memory_config=output_mem_configs["ff3"],
                compute_kernel_config=compute_kernel_config,
                global_cb=prefetcher.global_cb,
                sub_device_id=worker_sub_device_id,
            )
            layer_outputs["ff3"] = ttnn.to_memory_config(ff3_out, ttnn.DRAM_MEMORY_CONFIG)

            # Run FF2 matmul
            ff2_out = ttnn.matmul(
                tt_inputs["ff2_input"],
                tt_weights[f"layer_{layer_idx}_ff2"],
                program_config=program_configs["ff2"],
                memory_config=output_mem_configs["ff2"],
                compute_kernel_config=compute_kernel_config,
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

    # Verify results
    logger.info("Verifying results...")
    all_passing = True

    for layer_idx in range(num_layers):
        for matmul_name in ["qkv", "wo", "ff1", "ff3", "ff2"]:
            tt_out = ttnn.to_torch(outputs[layer_idx][matmul_name], mesh_composer=mesh_composer)[:1, :1, ...]

            # Get input and weight for this matmul
            if matmul_name in ["qkv", "wo"]:
                pt_input = pt_inputs["attn_input"]
            elif matmul_name in ["ff1", "ff3"]:
                pt_input = pt_inputs["mlp_input"]
            else:  # ff2
                pt_input = pt_inputs["ff2_input"]

            pt_weight = pt_weights[f"layer_{layer_idx}_{matmul_name}"]
            pt_out = pt_input @ pt_weight

            # Set PCC threshold based on dtype
            if matmul_name in ["ff1", "ff3"]:  # bfloat4_b
                pcc_threshold = 0.98
            else:  # bfloat8_b
                pcc_threshold = 0.99

            passing, output_str = comp_pcc(pt_out, tt_out, pcc_threshold)
            logger.info(f"Layer {layer_idx} {matmul_name}: {output_str}")
            all_passing = passing and all_passing

    # Cleanup
    prefetcher.stop()

    # Clean up sub device manager
    mesh_device.clear_loaded_sub_device_manager()

    assert all_passing, "PCC check failed for one or more matmuls"


# =============================================================================
# Test Cases
# =============================================================================


@pytest.mark.skipif(not is_blackhole(), reason="Only runs on Blackhole")
@pytest.mark.parametrize(
    "num_layers",
    [
        pytest.param(1, id="1_layer"),
        pytest.param(2, id="2_layers"),
        pytest.param(4, id="4_layers"),
        pytest.param(8, id="8_layers"),
        pytest.param(16, id="16_layers"),
        pytest.param(32, id="32_layers"),
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_prefetcher_ring_matmul_BH(
    mesh_device,
    num_layers,
    model_dims,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
):
    """
    Test prefetcher with ring matmul on Blackhole for varying number of layers.
    Runs all 5 matmuls: QKV, WO, FF1, FF3, FF2.
    """
    run_prefetcher_all_matmuls(
        mesh_device=mesh_device,
        num_layers=num_layers,
        model_dims=model_dims,
        enable_trace=True,
    )


@pytest.mark.skipif(not is_blackhole(), reason="Only runs on Blackhole")
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_prefetcher_ring_matmul_BH_no_trace(
    mesh_device,
    model_dims,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
):
    """
    Test prefetcher without trace capture for debugging.
    """
    run_prefetcher_all_matmuls(
        mesh_device=mesh_device,
        num_layers=2,
        model_dims=model_dims,
        enable_trace=False,
    )


@pytest.mark.skipif(not is_blackhole(), reason="Only runs on Blackhole")
@pytest.mark.parametrize(
    "dim, hidden_dim, n_heads, n_kv_heads",
    [
        pytest.param(2048, 5632, 16, 8, id="small_model"),
        pytest.param(4096, 11264, 32, 8, id="medium_model"),
        pytest.param(8192, 22528, 64, 8, id="large_model"),
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_prefetcher_ring_matmul_BH_model_sizes(
    mesh_device,
    dim,
    hidden_dim,
    n_heads,
    n_kv_heads,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
):
    """
    Test prefetcher with different model sizes on Blackhole.
    """
    custom_model_dims = {
        "dim": dim,
        "hidden_dim": hidden_dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": dim // n_heads,
    }

    run_prefetcher_all_matmuls(
        mesh_device=mesh_device,
        num_layers=2,
        model_dims=custom_model_dims,
        enable_trace=True,
    )


@pytest.mark.skipif(not is_blackhole(), reason="Only runs on Blackhole")
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_prefetcher_ring_matmul_BH_4_devices(
    mesh_device,
    model_dims,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
):
    """
    Test prefetcher with 4 devices (1x4 mesh) on Blackhole.
    """
    run_prefetcher_all_matmuls(
        mesh_device=mesh_device,
        num_layers=4,
        model_dims=model_dims,
        enable_trace=True,
    )


@pytest.mark.skipif(not is_blackhole(), reason="Only runs on Blackhole")
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_prefetcher_ring_matmul_BH_8_devices(
    mesh_device,
    model_dims,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
):
    """
    Test prefetcher with 8 devices (1x8 mesh) on Blackhole.
    """
    run_prefetcher_all_matmuls(
        mesh_device=mesh_device,
        num_layers=4,
        model_dims=model_dims,
        enable_trace=True,
    )
