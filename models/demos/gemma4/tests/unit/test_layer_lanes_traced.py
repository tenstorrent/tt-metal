# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One decoder layer of the lane engine, traced: the smallest full-mesh unit for a decode-step knob.

The lane engine's traced 128-user step is device-bound (window-2/3 profiles 2026-10-24: 1698 device ops, 30-31 ms
of kernel time, 32.6 ms traced wall), so a knob's worth is the traced per-layer delta. This test builds ONE sliding
decoder layer (random weights, the reduced expert count of the test factory, MoE experts and the decode MoE kernel
on) in the engine's orientation (TP over the rows, lanes over the columns), feeds it the decode frame (32 rows per
lane, one row tile per chip), warms the eager path, captures the layer's decode step as a trace and replays it
``LANE_LAYER_REPLAYS`` times; it logs ``LANE_LAYER ms/replay`` plus the CCL knobs it ran under and checks the traced
output against the eager one. One arm per pytest process: the knobs are read from the environment
(GEMMA4_CCL_NUM_LINKS, GEMMA4_CCL_ALLREDUCE_AG, GEMMA4_CCL_CHUNKS_PER_SYNC, GEMMA4_CCL_NUM_WORKERS,
GEMMA4_CCL_NUM_BUFFERS, the fusion flags).

    GEMMA4_MOE_DECODE_KERNEL=1 GEMMA4_CCL_ALLREDUCE_AG=1 pytest \\
        models/demos/gemma4/tests/unit/test_layer_lanes_traced.py -k 8x4 -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.lane_frame import check_lane_decode_frame
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.tt_transformers.tt.common import PagedAttentionConfig

from ...tests.test_factory import TestFactory, lanes_fabric_config, parametrize_mesh_with_fabric, skip_if_not_moe
from .test_layer import (
    _create_gemma4_model_args,
    _create_hf_reference_layer,
    _create_hf_text_config,
    _hf_state_to_tt_state,
    _reduced_experts,
)

ROWS_PER_LANE = 32
CACHE_LEN = 96
BLOCK_SIZE = 64
BLOCKS_PER_USER = (CACHE_LEN + 64 + BLOCK_SIZE - 1) // BLOCK_SIZE  # the decode position plus headroom
REPLAYS = int(os.environ.get("LANE_LAYER_REPLAYS", "20"))
KNOBS = (
    "GEMMA4_CCL_NUM_LINKS",
    "GEMMA4_CCL_ALLREDUCE_AG",
    "GEMMA4_CCL_CHUNKS_PER_SYNC",
    "GEMMA4_CCL_NUM_WORKERS",
    "GEMMA4_CCL_NUM_BUFFERS",
    "GEMMA4_MOE_COMPUTE_CUSTOM_OP",
    "GEMMA4_NORM_FUSED",
)


def _engine_mesh_config(mesh_device):
    tp, ep = mesh_device.shape[0], mesh_device.shape[1]
    cfg = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=tp, ep=ep),
        prefill=ModeConfig(tp=tp, ep=ep),
        tp_axis=0,
        weight_fracture=True,
    )
    cfg.lane_sharded = True
    return cfg


@skip_if_not_moe
@pytest.mark.skipif(not is_blackhole(), reason="the decode MoE kernel is Blackhole-only")
@parametrize_mesh_with_fabric(
    mesh_shapes=[(2, 4), (8, 4)],
    device_params_extra={
        "fabric_config": lanes_fabric_config(),
        "l1_small_size": 65536,
        "trace_region_size": 64_000_000,
    },
)
@pytest.mark.parametrize("layer_idx", [0], ids=["sliding_layer0"])
def test_layer_lanes_traced_step(mesh_device, layer_idx, reset_seeds, monkeypatch):
    # CI runs the whole dir on 4-chip, 1-chip and Wormhole boxes: skip, never error, off the lane engine's shapes
    shape = tuple(int(x) for x in mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1)
    if shape not in ((2, 4), (8, 4)):
        pytest.skip(f"the lane engine's shapes are (2,4) and (8,4); got {shape}")
    try:
        fabric = str(ttnn.get_fabric_config())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"fabric config unavailable: {exc}")
    if "FABRIC_1D" not in fabric:
        pytest.skip(f"lanes fabric (FABRIC_1D / FABRIC_1D_RING) not active: {fabric}")
    monkeypatch.setenv("GEMMA4_MOE_DECODE_KERNEL", os.environ.get("GEMMA4_MOE_DECODE_KERNEL", "1"))
    monkeypatch.setenv("GEMMA4_GALAXY_FRACTURE", "1")
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    if tuple(mesh_device.shape) == (2, 4):
        monkeypatch.setenv("GEMMA4_MOE_DECODE_FRAME", os.environ.get("GEMMA4_MOE_DECODE_FRAME", "128"))
    mesh_config = _engine_mesh_config(mesh_device)
    lanes = mesh_config.lanes
    users = lanes * ROWS_PER_LANE

    hf_text_config = _create_hf_text_config(num_experts=_reduced_experts(mesh_device), top_k=2)
    hf_layer = _create_hf_reference_layer(hf_text_config, layer_idx)
    tt_state = _hf_state_to_tt_state(hf_layer.state_dict(), layer_idx)
    model_args = _create_gemma4_model_args(hf_text_config)
    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)
    ccl_manager = CCLManager(mesh_device)  # the engine's default links
    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=CACHE_LEN + 64,
        max_local_batch_size=ROWS_PER_LANE,
    )
    # The engine's paged cache: per column block 0 is the lane scratch, then one block run per row
    # (test_decode_roofline_lanes.py's pool); the attention's heads come from mesh_config (effective block 64).
    paged_cfg = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=1 + ROWS_PER_LANE * BLOCKS_PER_USER)
    kv_cache = init_kv_cache(
        mesh_device,
        attn_cfg,
        max_batch_size=ROWS_PER_LANE,
        max_seq_len=CACHE_LEN + 64,
        paged_attention_config=paged_cfg,
        cache_dtype=ttnn.bfloat16,
        mesh_config=mesh_config,
    )
    tt_layer.self_attn.kv_cache = kv_cache

    # The decode frame exactly as prepare_decode_inputs_host hands it to the layer (lane-major batch of
    # lanes x 32 rows, each tensor lane-sharded over the columns): hidden rows on dim 2, the RoPE positions
    # [1, batch] uint32 on dim 1, the cache positions [batch] int32 on dim 0 (the paged cache update and the
    # decode SDPA read their batch from dim 0: a [1, rows] tensor here is ONE update index for 32 rows, window 5),
    # the page table [batch, blocks] int32 on dim 0. Checked on the host before anything reaches the device.
    x_torch = torch.randn(1, 1, users, model_args.hidden_size, dtype=torch.bfloat16)
    rope_pos = torch.full((1, users), CACHE_LEN, dtype=torch.int64)
    cache_pos = torch.full((users,), CACHE_LEN, dtype=torch.int32)
    page_table = torch.zeros(users, BLOCKS_PER_USER, dtype=torch.int32)
    for row in range(users):
        slot = row % ROWS_PER_LANE
        page_table[row] = torch.arange(1 + slot * BLOCKS_PER_USER, 1 + (slot + 1) * BLOCKS_PER_USER, dtype=torch.int32)
    columns = check_lane_decode_frame(
        lanes,
        ROWS_PER_LANE,
        hidden=tuple(x_torch.shape),
        rope_positions=tuple(rope_pos.shape),
        cache_positions=tuple(cache_pos.shape),
        page_table=tuple(page_table.shape),
    )
    logger.info(f"LANE_LAYER frame per column: {columns}")
    lane_map = lambda d: mesh_config.lane_shard_mapper(mesh_device, d)  # noqa: E731
    x_tt = ttnn.from_torch(
        x_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=lane_map(2)
    )
    rope_pos_tt = ttnn.from_torch(
        rope_pos, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=lane_map(1)
    )
    cache_pos_tt = ttnn.from_torch(
        cache_pos, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=lane_map(0)
    )
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=lane_map(0)
    )
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, CACHE_LEN + 64, layer_idx)

    def step():
        return tt_layer(
            x_tt,
            rope_mats=(cos_tt, sin_tt),
            position_idx=rope_pos_tt,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            is_decode=True,
            token_index=None,  # per-user RoPE from the positions (the engine passes None under its 2-D rope caches)
            position_idx_cache=cache_pos_tt,
        )

    # warm (compiles every program), then one graph-captured eager step: the program count and the collectives of
    # this arm (the inert-knob rule: an arm that leaves the collectives unchanged says so), and the eager reference
    out = step()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = step()
    ttnn.synchronize_device(mesh_device)
    captured = ttnn.graph.end_graph_capture()
    eager = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ttnn.deallocate(out)
    dev_ops = [
        n["params"].get("name", "?")
        for n in captured
        if n["node_type"] == "function_start" and str(n["params"].get("name", "")).endswith("DeviceOperation")
    ]
    ccl_ops = [n for n in dev_ops if any(k in n for k in ("AllGather", "ReduceScatter", "AllReduce", "AllToAll"))]
    ccl_counts = {}
    for n in ccl_ops:
        ccl_counts[n] = ccl_counts.get(n, 0) + 1
    knobs = " ".join(f"{k}={os.environ.get(k, '-')}" for k in KNOBS)
    ccl_line = " ".join(f"{k.replace('DeviceOperation', '')}={v}" for k, v in sorted(ccl_counts.items()))
    logger.info(f"LANE_LAYER programs/step={len(dev_ops)} ccl_ops={len(ccl_ops)} {ccl_line} | {knobs}")

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_tr = step()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)  # one untimed replay
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - t0) / REPLAYS * 1e3
    traced = ttnn.to_torch(ttnn.get_device_tensors(out_tr)[0]).float()
    logger.info(
        f"LANE_LAYER layer={layer_idx} mesh={mesh_device.shape[0]}x{mesh_device.shape[1]} users={users}: "
        f"{ms:.3f} ms/replay over {REPLAYS} replays (x30 layers = {30 * ms:.1f} ms) | {knobs}"
    )
    ttnn.release_trace(mesh_device, trace_id)
    assert torch.isfinite(traced).all(), "traced layer output is not finite"
    pcc = torch.corrcoef(torch.stack([eager.flatten(), traced.flatten()]))[0, 1].item()
    logger.info(f"LANE_LAYER parity traced vs eager pcc={pcc:.6f}")
    assert pcc > 0.999, f"traced step diverges from the eager one: pcc {pcc}"
