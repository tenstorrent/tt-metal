# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Decode Performance / Tracy Profiling Test.

Runs eager (no trace) decode iterations with tracy signposts bracketing only
the measured decode steps — warmup compile iters are run but NOT signposted
so they are excluded from the tracy capture window.

Run with tracy profiler (profile decode ops only):
    python -m tracy -p -v -r -- python -m pytest \
        models/demos/olmo_galaxy/tests/test_olmo_decode_perf.py \
        -k "1layer" -s -v

Run without tracy (just timing):
    export HF_MODEL=~/models/OLMo-3.1-32B-Think
    pytest models/demos/olmo_galaxy/tests/test_olmo_decode_perf.py \
        -k "1layer" -s -v

The test skips trace capture entirely so all ops appear individually in tracy
(not collapsed into a single Execute Trace op).
"""

from time import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.olmo_galaxy.tt.llama_ccl import TT_CCL
from models.demos.olmo_galaxy.tt.llama_common import PagedAttentionConfig
from models.demos.olmo_galaxy.tt.llama_decoder import TtTransformerBlock
from models.demos.olmo_galaxy.tt.llama_rope import TtLlamaRotarySetup
from models.demos.olmo_galaxy.tt.olmo_model_config import TtOlmoModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 2048}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "num_layers",
    (1, 64),
    ids=("1layer", "64layers"),
)
@pytest.mark.parametrize(
    "generation_start_pos",
    (127,),
    ids=("pos127",),
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            # No trace_region_size — we are not capturing a trace
            "fabric_config": True,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
def test_olmo_decode_perf(
    num_layers,
    batch_size,
    generation_start_pos,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    """
    Eager-mode (no trace) decode perf test for OLMo 1-layer and 64-layer.

    Tracy captures every kernel individually since there is no trace replay.
    Warmup iterations run before tracy signposts so they are excluded from
    the profiling window.
    """
    dtype = ttnn.bfloat8_b
    num_warmup_iters = 2
    num_perf_iters = 10

    model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=4096,
    )
    model_args.n_layers = num_layers

    logger.info(f"OLMo decode perf: layers={num_layers}, batch={batch_size}, pos={generation_start_pos}")

    state_dict = model_args.load_state_dict()

    # Sub-device setup (OLMo does not use prefetcher)
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

    logger.info(f"Loading {num_layers} layer(s)...")
    tt_layers = []
    for layer_id in range(num_layers):
        tt_layer = TtTransformerBlock(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            layer_num=layer_id,
            n_layers=num_layers,
            weight_cache_path=model_args.weight_cache_path(dtype),
            transformation_mats=rope_setup.get_both_trans_mats(),
            paged_attention_config=paged_attention_config,
            prefetcher_setup=None,
            tt_ccl=tt_ccl,
        )
        tt_layers.append(tt_layer)
    logger.info("Layers loaded.")

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    rot_mat_idxs_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))])

    pt_decode_input = (torch.rand(batch_size, 1, model_args.dim) * 2) - 1
    decode_input = model_args.prepare_residual_tensor_decode(
        pt_decode_input,
        model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
    )

    current_pos = torch.tensor([generation_start_pos] * batch_size)
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
    rot_mat_idxs = rope_setup.get_rm_rot_idxs(current_pos, on_host=False)

    def run_decode_iter():
        rot_mats = rope_setup.get_rm_rot_mats(rot_mat_idxs)
        x = decode_input
        h = None
        for tt_layer in tt_layers:
            x, h = tt_layer(
                x,
                h,
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
            )
        ttnn.plus_one(current_pos_tensor, sub_core_grids=sub_core_grids, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs, sub_core_grids=rot_mat_idxs_core_grid)
        return x

    # === Warmup (excluded from tracy window) ===
    logger.info(f"Warmup ({num_warmup_iters} iters, not profiled)...")
    for i in range(num_warmup_iters):
        t0 = time()
        tt_out = run_decode_iter()
        ttnn.synchronize_device(mesh_device)
        logger.info(f"  Warmup {i}: {(time()-t0)*1000:.1f} ms")

    # Reset positions back to start before profiled run
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

    # === Profiled decode iterations (tracy captures these) ===
    # ttnn.start_tracy_zone / stop_tracy_zone bracket the region visible in tracy UI.
    # ttnn.tracy_frame marks each iteration as a separate frame in the timeline.
    # ttnn.tracy_message adds text annotations visible in the tracy event log.
    logger.info(f"Profiled decode run ({num_perf_iters} iters)...")

    elapsed_ms = []
    for i in range(num_perf_iters):
        ttnn.tracy_message(f"olmo_decode_iter_{i}")
        ttnn.start_tracy_zone("test_olmo_decode_perf.py", "run_decode_iter", i)
        t0 = time()
        tt_out = run_decode_iter()
        ttnn.synchronize_device(mesh_device)
        elapsed = (time() - t0) * 1000
        ttnn.stop_tracy_zone()
        ttnn.tracy_frame()
        elapsed_ms.append(elapsed)
        logger.info(f"  Iter {i}: {elapsed:.1f} ms")

    avg_ms = sum(elapsed_ms[1:]) / len(elapsed_ms[1:])  # skip first (may include leftover compile)
    logger.info(f"\n{'='*50}")
    logger.info(f"OLMo decode perf ({num_layers} layer(s), batch={batch_size}):")
    logger.info(f"  Avg (iters 1+): {avg_ms:.2f} ms/step")
    logger.info(f"  Min: {min(elapsed_ms[1:]):.2f} ms")
    logger.info(f"  Max: {max(elapsed_ms[1:]):.2f} ms")
    if num_layers == 64:
        tps = batch_size / (avg_ms / 1000)
        logger.info(f"  Throughput: {tps:.1f} tok/s ({batch_size} users)")
    logger.info(f"{'='*50}")
