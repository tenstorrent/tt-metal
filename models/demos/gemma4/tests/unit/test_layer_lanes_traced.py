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
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer

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
    kv_cache = init_kv_cache(
        mesh_device,
        attn_cfg,
        max_batch_size=ROWS_PER_LANE,
        max_seq_len=CACHE_LEN + 64,
        cache_dtype=ttnn.bfloat16,
        mesh_config=mesh_config,
    )
    tt_layer.self_attn.kv_cache = kv_cache

    # the decode frame: 32 rows per lane (lane-sharded over the columns), one row tile per chip
    x_torch = torch.randn(1, 1, users, model_args.hidden_size, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_config.lane_shard_mapper(mesh_device, 2),
    )
    position_idx_tt = ttnn.from_torch(
        torch.full((1, ROWS_PER_LANE), CACHE_LEN, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, CACHE_LEN + 64, layer_idx)

    def step():
        return tt_layer(
            x_tt,
            rope_mats=(cos_tt, sin_tt),
            position_idx=position_idx_tt,
            page_table=None,
            kv_cache=kv_cache,
            is_decode=True,
            token_index=CACHE_LEN,
        )

    # warm (compiles every program), then the eager reference output
    out = step()
    ttnn.synchronize_device(mesh_device)
    eager = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ttnn.deallocate(out)

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
    knobs = " ".join(f"{k}={os.environ.get(k, '-')}" for k in KNOBS)
    logger.info(
        f"LANE_LAYER layer={layer_idx} mesh={mesh_device.shape[0]}x{mesh_device.shape[1]} users={users}: "
        f"{ms:.3f} ms/replay over {REPLAYS} replays (x30 layers = {30 * ms:.1f} ms) | {knobs}"
    )
    ttnn.release_trace(mesh_device, trace_id)
    assert torch.isfinite(traced).all(), "traced layer output is not finite"
    pcc = torch.corrcoef(torch.stack([eager.flatten(), traced.flatten()]))[0, 1].item()
    logger.info(f"LANE_LAYER parity traced vs eager pcc={pcc:.6f}")
    assert pcc > 0.999, f"traced step diverges from the eager one: pcc {pcc}"
