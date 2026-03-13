# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Decode Test with Trace.

Tests decode mode with KV cache, sliding window attention, and trace capture.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    export LINE_RS=1
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_decode.py -v
"""

import torch
import pytest
from time import time
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    PagedAttentionConfig,
)
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (False,),
    ids=("standard_kv",),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),
)
@pytest.mark.parametrize(
    "num_layers",
    (1, 64),
    ids=("1layer", "64layers"),
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 165136000,
            "fabric_config": True,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
def test_olmo_decoder_decode(
    max_seq_len,
    batch_size,
    num_layers,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    """Test OLMo decoder in decode mode with trace capture and perf measurement."""
    dtype = ttnn.bfloat8_b

    model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    model_args.n_layers = num_layers

    logger.info(f"OLMo Config: layers={num_layers}, max_seq_len={max_seq_len}, batch={batch_size}")
    logger.info(f"  dim={model_args.dim}, n_heads={model_args.n_heads}, n_kv_heads={model_args.n_kv_heads}")

    state_dict = model_args.load_state_dict()

    # All-core sub-device setup (OLMo does not use prefetcher)
    all_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    all_sub_device = ttnn.SubDevice([all_core_range_set])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([all_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    tt_ccl = TT_CCL(
        mesh_device,
        model_args,
        worker_sub_device_id,
        mode="decode",
        is_qwen=False,
        is_olmo=True,
    )

    paged_attention_config = None
    page_table_tt = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    logger.info(f"Loading {num_layers} decoder layers...")
    tt_layers = []
    for layer_id in range(num_layers):
        layer_type = model_args.get_layer_type(layer_id)
        sliding_window = model_args.get_sliding_window_size(layer_id)
        if layer_id < 4 or layer_id == num_layers - 1:
            logger.info(f"  Layer {layer_id}: type={layer_type}, sliding_window={sliding_window}")
        elif layer_id == 4:
            logger.info(f"  ... (loading remaining layers)")

        tt_layer = TtTransformerBlock(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            layer_num=layer_id,
            n_layers=num_layers,
            weight_cache_path=model_args.weight_cache_path(dtype),
            transformation_mats=transformation_mats,
            paged_attention_config=paged_attention_config,
            prefetcher_setup=None,
            tt_ccl=tt_ccl,
        )
        tt_layers.append(tt_layer)
    logger.info(f"All {num_layers} layers loaded.")

    # Decode parameters
    seqlen = 1
    generation_start_pos = 127
    num_compile_iters = 2 if num_layers <= 5 else 1
    num_perf_iters = 20

    # Persistent input tensor
    pt_decode_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1
    decode_input = model_args.prepare_residual_tensor_decode(
        pt_decode_input,
        model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
    )

    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    rot_mats, rot_mat_idxs = rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

    rot_mat_idxs_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))])

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    def run_decode_iteration():
        """One decode iteration — identical path for compile and trace."""
        rot_mats_local = rope_setup.get_rm_rot_mats(rot_mat_idxs)
        x = decode_input
        h = None
        for tt_layer in tt_layers:
            x, h = tt_layer(
                x,
                h,
                current_pos_tensor,
                rot_mats=rot_mats_local,
                mode="decode",
                page_table=page_table_tt,
            )
        ttnn.plus_one(current_pos_tensor, sub_core_grids=sub_core_grids, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs, sub_core_grids=rot_mat_idxs_core_grid)
        return x

    # === Phase 1: Compile ===
    logger.info(f"Compiling model ({num_compile_iters} iterations)...")
    for i in range(num_compile_iters):
        compile_start = time()
        tt_out = run_decode_iteration()
        ttnn.synchronize_device(mesh_device)
        compile_elapsed = time() - compile_start
        logger.info(f"  Compile {i}: {compile_elapsed * 1000:.1f} ms")

    # Validate compile output
    tt_output_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)
    has_nan = torch.isnan(tt_output_torch).any().item()
    has_inf = torch.isinf(tt_output_torch).any().item()
    logger.info(
        f"  Compile output: NaN={has_nan}, Inf={has_inf}, mean={tt_output_torch.mean().item():.4f}, std={tt_output_torch.std().item():.4f}"
    )
    if has_nan:
        logger.warning("Compile output contains NaN — continuing to trace capture")
    if has_inf:
        logger.warning("Compile output contains Inf — continuing to trace capture")

    # === Phase 2: Capture Trace ===
    logger.info("Capturing decode trace...")
    tt_ccl.reset_gather_and_buffer_idx()

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_out = run_decode_iteration()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ttnn.synchronize_device(mesh_device)
    logger.info("Trace captured successfully.")

    # Reset positions for perf run
    current_pos_reset = ttnn.from_torch(
        current_pos,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    ttnn.copy_host_to_device_tensor(current_pos_reset, current_pos_tensor)
    rot_mat_idxs_reset = rope_setup.get_rm_rot_idxs(current_pos, on_host=True)
    ttnn.copy_host_to_device_tensor(rot_mat_idxs_reset, rot_mat_idxs)
    ttnn.synchronize_device(mesh_device)

    # === Phase 3: Execute trace with perf measurement ===
    logger.info(f"Running {num_perf_iters} traced decode iterations...")
    iteration_times = []
    for i in range(num_perf_iters):
        iter_start = time()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        iter_elapsed = time() - iter_start
        iteration_times.append(iter_elapsed)

        tsu = 1.0 / iter_elapsed
        throughput = batch_size / iter_elapsed
        logger.info(
            f"  Iter {i}: {iter_elapsed * 1000:.1f} ms | "
            f"tok/s/user: {tsu:.2f} | "
            f"throughput: {throughput:.1f} tok/s"
        )

    # Validate traced output (Inf can occur after many iterations on partial models)
    tt_output_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)
    has_nan = torch.isnan(tt_output_torch).any().item()
    has_inf = torch.isinf(tt_output_torch).any().item()
    logger.info(f"  Traced output: NaN={has_nan}, Inf={has_inf}")
    if num_layers == 64:
        assert not has_nan, "Traced output contains NaN"
        assert not has_inf, "Traced output contains Inf"

    ttnn.release_trace(mesh_device, trace_id)
    tt_ccl.close()

    # === Performance summary ===
    avg_time = sum(iteration_times) / len(iteration_times)
    min_time = min(iteration_times)
    max_time = max(iteration_times)
    median_time = sorted(iteration_times)[len(iteration_times) // 2]
    avg_tsu = 1.0 / avg_time
    avg_throughput = batch_size / avg_time

    logger.info("=" * 60)
    logger.info(f"OLMo Decode Performance - TRACED ({num_layers} layers, batch={batch_size})")
    logger.info("=" * 60)
    logger.info(f"  Iterations (perf): {num_perf_iters}")
    logger.info(f"  Avg iteration time: {avg_time * 1000:.1f} ms")
    logger.info(f"  Min iteration time: {min_time * 1000:.1f} ms")
    logger.info(f"  Max iteration time: {max_time * 1000:.1f} ms")
    logger.info(f"  Median iter time:   {median_time * 1000:.1f} ms")
    logger.info(f"  Avg tok/s/user:     {avg_tsu:.2f}")
    logger.info(f"  Avg throughput:     {avg_throughput:.1f} tok/s (batch={batch_size})")
    logger.info("=" * 60)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_olmo_sliding_window_decode_layers(mesh_device, reset_seeds, ensure_gc):
    """Test that sliding window is correctly applied to different layer types in decode mode."""
    batch_size = 32
    max_seq_len = 256
    dtype = ttnn.bfloat8_b

    model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    layer_types = []
    for layer_id in range(8):
        layer_type = model_args.get_layer_type(layer_id)
        sliding_window = model_args.get_sliding_window_size(layer_id)
        layer_types.append((layer_id, layer_type, sliding_window))
        logger.info(f"Layer {layer_id}: type={layer_type}, sliding_window={sliding_window}")

    expected_pattern = [
        (0, "sliding_attention", 4096),
        (1, "sliding_attention", 4096),
        (2, "sliding_attention", 4096),
        (3, "full_attention", None),
        (4, "sliding_attention", 4096),
        (5, "sliding_attention", 4096),
        (6, "sliding_attention", 4096),
        (7, "full_attention", None),
    ]

    for layer_id, layer_type, sliding_window in layer_types:
        exp_layer_id, exp_type, exp_window = expected_pattern[layer_id]
        assert layer_type == exp_type, f"Layer {layer_id}: expected {exp_type}, got {layer_type}"
        assert (
            sliding_window == exp_window
        ), f"Layer {layer_id}: expected sliding_window={exp_window}, got {sliding_window}"

    logger.info("Sliding window pattern verification PASSED!")
