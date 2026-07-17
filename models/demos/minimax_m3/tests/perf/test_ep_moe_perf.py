# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Device-perf for the MiniMax-M3 EP MoE op (TtMiniMaxMoE), via the standard tt-metal
device-profiler harness (mirrors deepseek_v3_d_p/tests/perf/test_moe_perf.py).

Two pieces:
- INNER `test_ep_moe_fwd`: builds TtMiniMaxMoE (random wts) and runs ONE forward at a
  big seq, wrapped in tracy `start`/`stop` signposts. Run *under* the profiler.
- OUTER `test_ep_moe_device_perf`: spawns the inner under Tracy (`run_device_perf`),
  sums DEVICE KERNEL DURATION [ns] between the signposts, logs ns + tokens/s.

Inner config via env: PERF_ROWS/PERF_COLS (mesh), PERF_SEQ (seq_len_per_chip).
Set TT_MESH_GRAPH_DESC_PATH to the matching MGD (e.g. stock single_bh_galaxy [8,4]).

Run the measurement:
  source python_env/bin/activate
  export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  pytest models/demos/minimax_m3/tests/perf/test_ep_moe_perf.py::test_ep_moe_device_perf -s
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn

_THIS = "models/demos/minimax_m3/tests/perf/test_ep_moe_perf.py"
# Real M3 MoE dims: 128 experts / top-4, hidden 6144, moe_intermediate 3072.
EMB, HID, E, K = 6144, 3072, 128, 4


class _M3FabricCfg:
    """Minimal fabric config for open_mesh_device (it reads only FABRIC_PAYLOAD_SIZE)."""

    FABRIC_PAYLOAD_SIZE = EMB  # M3 hidden_size (max fabric packet payload)


def _open_mesh_with_trace(shape, model_cfg, trace_region_size):
    """open_mesh_device (runner_utils) + a trace region so we can capture/replay."""
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config

    sp = shape[0]
    fabric_config = ttnn.FabricConfig.FABRIC_1D if sp <= 8 else ttnn.FabricConfig.FABRIC_2D
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        create_fabric_router_config(max_payload_size=model_cfg.FABRIC_PAYLOAD_SIZE),
    )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), trace_region_size=trace_region_size)


def test_ep_moe_fwd():
    """Inner: one TtMiniMaxMoE forward at a big seq, signpost-delimited. Run under Tracy.

    TRACE=1 → capture the post-gate device path (forward with external routing from a
    one-shot gate call) into a ttnn trace and profile the REPLAY — removes host op-launch
    overhead so wall ≈ device floor. Else → eager forward (internal HOST_ALL gate)."""
    from models.demos.common.prefill.runners.runner_utils import open_mesh_device
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
        compute_constants,
        create_gate_weights,
        create_torch_expert_weights,
        extract_mesh_config,
    )
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.minimax_m3.tt.moe.tt_minimax_moe import TtMiniMaxMoE

    rows, cols = int(os.getenv("PERF_ROWS", "8")), int(os.getenv("PERF_COLS", "4"))
    seq = int(os.getenv("PERF_SEQ", "2048"))
    use_trace = os.getenv("TRACE", "0") == "1"
    torch.manual_seed(0)
    mesh = (
        _open_mesh_with_trace((rows, cols), _M3FabricCfg, 200_000_000)
        if use_trace
        else open_mesh_device((rows, cols), _M3FabricCfg)
    )
    try:
        mc = extract_mesh_config(mesh)
        dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(seq, E, K, mesh.get_num_devices(), dgs, 2)
        tt_moe = TtMiniMaxMoE(
            mesh_device=mesh,
            dispatch_group_size=dgs,
            num_dispatch_groups=ndg,
            experts_per_chip=experts_per_chip,
            num_routed_experts=E,
            num_experts_per_tok=K,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_tok,
            max_dispatch_buffer_token_size=max_buf,
            seq_len_per_chip=seq,
            emb_dim=EMB,
            hidden_dim=HID,
            num_links=2,
            topology=ttnn.Topology.Linear,
            routed_expert_weights=create_torch_expert_weights(E, EMB, HID, seed=1234),
            routed_expert_weights_dtype=ttnn.bfloat4_b,
            gate_weights=create_gate_weights(E, EMB, seed=9012),
            gate_fallback_mode=GateComputeMode.HOST_ALL,
        )
        x = ttnn.from_torch(
            torch.randn(dgs, seq, EMB, dtype=torch.bfloat16),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(0, -1)),
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            dtype=ttnn.bfloat16,
        )
        if use_trace:
            # one-shot HOST_ALL gate → routing tensors (correct layout); then trace the
            # pure-device post-gate path forward(x, idx, scores) and profile the replay.
            scores, indices, gl = tt_moe.gate(ttnn.view(x, (x.shape[0] * x.shape[1], x.shape[2])))
            ttnn.deallocate(gl)
            tt_moe(x, topk_indices=indices, topk_weights=scores)  # warmup/compile
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = tt_moe(x, topk_indices=indices, topk_weights=scores)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            signpost(header="start")
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            signpost(header="stop")
            ttnn.deallocate(out)
            logger.info(
                f"[ep-moe-perf] TRACED replay done: mesh=({rows},{cols}) dgs={dgs} ndg={ndg} experts/chip={experts_per_chip} seq/chip={seq}"
            )
        else:
            tt_moe(x)  # warmup (compile kernels) — excluded from profiled region
            ttnn.synchronize_device(mesh)
            signpost(header="start")
            out = tt_moe(x)
            ttnn.synchronize_device(mesh)
            signpost(header="stop")
            ttnn.deallocate(out)
            logger.info(
                f"[ep-moe-perf] eager forward done: mesh=({rows},{cols}) dgs={dgs} ndg={ndg} experts/chip={experts_per_chip} seq/chip={seq}"
            )
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.timeout(0)
def test_ep_moe_device_perf():
    """Outer: profile the inner forward, report device-kernel ns + tokens/s."""
    from models.perf.device_perf_utils import run_device_perf

    rows, cols = int(os.getenv("PERF_ROWS", "8")), int(os.getenv("PERF_COLS", "4"))
    seq = int(os.getenv("PERF_SEQ", "2048"))
    total_tokens = rows * seq  # tokens across the dispatch group (one MoE-layer's worth)

    res = run_device_perf(
        command=f"pytest {_THIS}::test_ep_moe_fwd -s",
        subdir="minimax_ep_moe",
        num_iterations=1,
        cols=["DEVICE KERNEL"],
        batch_size=1,
        has_signposts=True,
    )
    ns = res["AVG DEVICE KERNEL DURATION [ns]"]
    logger.info(
        f"\n[ep-moe-perf] mesh=({rows},{cols}) seq/chip={seq} tokens={total_tokens}\n"
        f"  EP MoE device-kernel time: {ns/1000:.1f} us ({ns:.0f} ns)\n"
        f"  throughput: {total_tokens / (ns/1e9):,.0f} tokens/s (one MoE layer)"
    )
