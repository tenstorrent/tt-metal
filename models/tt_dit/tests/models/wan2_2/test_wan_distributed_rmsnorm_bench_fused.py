# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 DistributedRMSNorm production-shape benchmarks: composite vs fused.

Sweeps the 7 attention call sites from Wan2.2 720p generation across two
topologies that the new fused device op fully supports:

  * **TP=4 LINE** on a natural 1x4 submesh of the BH 2x4 parent mesh (one row,
    no reshape). Each chip gets H/4 = 1280; num_heads_per_device = 10.
  * **TP=8 RING** on a natively-opened 1x8 mesh (the parent BH 2x4 mapped
    directly as 1x8 — chips form a Hamiltonian cycle in mesh-coord order so
    fabric multi-hop mcast routes correctly). Each chip gets H/8 = 640;
    num_heads_per_device = 5.

For each config we run **two methods**:
  * `composite`: ``ttnn.experimental.wan_fused_distributed_rmsnorm(..., use_device_op=False)``
    — the existing C++ composite that chains pre / all_gather_async / post.
  * `fused`:     ``ttnn.experimental.wan_fused_distributed_rmsnorm(..., use_device_op=True)``
    — the new single-program device op.

Both methods are traced (`begin_trace_capture` / `execute_trace`) and timed
over `NUM_ITERS` iterations to amortize dispatch overhead. The summary tests
write a CSV and print a comparison table.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.manager import CCLManager
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.tensor import bf16_tensor, from_torch
from ....utils.test import line_params, ring_params

# Wan2.2 14B model configuration
DIM = 5120
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS  # 128

NORM_EPS = 1e-6
B = 1
NUM_ITERS = 100
TP_AXIS = 1  # axis of the 1xN mesh that holds the TP cluster

CSV_FILENAME_TP4_LINE = "wan_rmsnorm_bench_fused_tp4_line.csv"
CSV_FILENAME_TP8_RING = "wan_rmsnorm_bench_fused_tp8_ring.csv"

# (config_id, seq_len, use_rope) — the 7 Wan2.2 720p attention call sites.
BENCH_CONFIGS_ALL = [
    ("self_sp4_N18944", 18944, True),
    ("self_sp8_N9472", 9472, True),
    ("self_sp32_N2368", 2368, True),
    ("cross_q_sp4_N18944", 18944, False),
    ("cross_q_sp8_N9472", 9472, False),
    ("cross_q_sp32_N2368", 2368, False),
    ("cross_k_prompt_L512", 512, False),
]

# All 7 Wan configs (multi-chunk in MUX writer enables larger N now).
BENCH_CONFIGS = BENCH_CONFIGS_ALL


@dataclass
class _BuiltInputs:
    tt_input: ttnn.Tensor
    tt_rope_cos: ttnn.Tensor | None
    tt_rope_sin: ttnn.Tensor | None
    tt_trans_mat: ttnn.Tensor | None
    weight: ttnn.Tensor


# ---------------------------------------------------------------------------
# Submesh selectors
# ---------------------------------------------------------------------------


def _make_line_submesh_tp4(parent_mesh: ttnn.MeshDevice) -> ttnn.MeshDevice:
    """Carve a 1x4 submesh from the parent (one row, no reshape)."""
    return parent_mesh.create_submesh(ttnn.MeshShape(1, 4))


def _use_parent_mesh_tp8(parent_mesh: ttnn.MeshDevice) -> ttnn.MeshDevice:
    """Use the native 1x8 parent mesh directly — chips already form a cycle."""
    return parent_mesh


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _build_inputs(submesh: ttnn.MeshDevice, seq_len: int, use_rope: bool) -> _BuiltInputs:
    torch.manual_seed(0)
    torch_input = torch.randn((1, B, seq_len, DIM), dtype=torch.bfloat16)
    tt_input = bf16_tensor(torch_input, device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)

    weight_torch = torch.randn(DIM, dtype=torch.bfloat16)
    weight = bf16_tensor(
        weight_torch.reshape(1, DIM),
        device=submesh,
        mesh_axis=TP_AXIS,
        shard_dim=-1,
    )

    if use_rope:
        rope_cos_raw = torch.randn(B, seq_len, 1, HEAD_DIM // 2)
        rope_sin_raw = torch.randn(B, seq_len, 1, HEAD_DIM // 2)
        torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos_raw, rope_sin_raw)
        tt_rope_cos = from_torch(torch_rope_cos.permute(0, 2, 1, 3), device=submesh, dtype=ttnn.float32)
        tt_rope_sin = from_torch(torch_rope_sin.permute(0, 2, 1, 3), device=submesh, dtype=ttnn.float32)
        tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=submesh)
    else:
        tt_rope_cos = None
        tt_rope_sin = None
        tt_trans_mat = None

    return _BuiltInputs(tt_input, tt_rope_cos, tt_rope_sin, tt_trans_mat, weight)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def _run(
    inp: _BuiltInputs,
    submesh: ttnn.MeshDevice,
    ag_sem,
    topology: ttnn.Topology,
    n_local_heads: int,
    *,
    use_device_op: bool,
    persistent_output_buffer: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    # Fused op uses 2 fabric links per direction (multi-link MUX). Composite
    # uses default (1 link) to keep its baseline numbers stable across this
    # optimization sweep.
    num_preferred_links = 2 if use_device_op else None
    return ttnn.experimental.wan_fused_distributed_rmsnorm(
        inp.tt_input,
        TP_AXIS,
        submesh,
        ag_sem,
        topology=topology,
        epsilon=NORM_EPS,
        num_heads_per_device=n_local_heads,
        weight=inp.weight,
        transformation_mat=inp.tt_trans_mat,
        rope_cos=inp.tt_rope_cos,
        rope_sin=inp.tt_rope_sin,
        dtype=None,
        persistent_output_buffer=persistent_output_buffer,
        num_preferred_links=num_preferred_links,
        use_device_op=use_device_op,
    )


def _trace_and_time(submesh: ttnn.MeshDevice, run_op, *, num_iters: int = NUM_ITERS) -> float:
    """Compile, capture trace, run num_iters traced iterations, return avg us/iter."""
    run_op()
    ttnn.synchronize_device(submesh)

    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    run_op()
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    t0 = time.perf_counter()
    for _ in range(num_iters):
        ttnn.execute_trace(submesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(submesh)
    elapsed_us = (time.perf_counter() - t0) * 1e6

    ttnn.release_trace(submesh, trace_id)
    return elapsed_us / num_iters


def _bench_one(
    submesh: ttnn.MeshDevice,
    ag_sem,
    topology: ttnn.Topology,
    n_local_heads: int,
    seq_len: int,
    use_rope: bool,
    *,
    method: str,
) -> float:
    """Run one Wan config with the chosen method, return avg us/iter."""
    inp = _build_inputs(submesh, seq_len, use_rope)
    use_device_op = method == "fused"
    # Device-op MUX path needs a caller-allocated mesh-coherent stats buffer.
    # Returns None for shapes / TP that don't trigger MUX, in which case the
    # device op also doesn't need a buffer.
    persistent_output_buffer = (
        ttnn.experimental.wan_fused_distributed_rmsnorm_create_stats_buffer(
            inp.tt_input,
            TP_AXIS,
            submesh,
            num_heads_per_device=n_local_heads,
        )
        if use_device_op
        else None
    )
    run_op = lambda: _run(  # noqa: E731
        inp,
        submesh,
        ag_sem,
        topology,
        n_local_heads,
        use_device_op=use_device_op,
        persistent_output_buffer=persistent_output_buffer,
    )
    logger.info(f"[{method}] compiling+tracing (seq_len={seq_len}, use_rope={use_rope})")
    per_iter_us = _trace_and_time(submesh, run_op)
    logger.info(f"[{method}] seq_len={seq_len} use_rope={use_rope}: {per_iter_us:.2f} us/iter")
    return per_iter_us


def _format_summary_table(rows: list[dict], title: str) -> None:
    """Print a comparison table to stdout."""
    id_w = max(len("config_id"), max(len(r["config_id"]) for r in rows))
    header = (
        f"{'config_id':<{id_w}}  {'seq_len':>7}  {'rope':>4}  " f"{'composite us':>12}  {'fused us':>9}  {'speedup':>8}"
    )
    sep = "-" * len(header)
    box = "=" * max(len(header), len(title))
    print()
    print(box)
    print(title)
    print(box)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['config_id']:<{id_w}}  "
            f"{r['seq_len']:>7}  {('Y' if r['use_rope'] else 'N'):>4}  "
            f"{r['us_composite']:>12.2f}  {r['us_fused']:>9.2f}  "
            f"{r['speedup']:>7.2f}x"
        )
    print(box)


def _write_csv(rows: list[dict], filename: str) -> None:
    fieldnames = ["config_id", "seq_len", "use_rope", "us_composite", "us_fused", "speedup"]
    csv_path = Path.cwd() / filename
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"Wrote CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# Per-config parametrized perf test for debugging isolated shapes.
@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("seq_len", "use_rope"),
    [pytest.param(seq_len, use_rope, id=cfg_id) for (cfg_id, seq_len, use_rope) in BENCH_CONFIGS],
)
def test_wan_rmsnorm_bench_fused_tp4_line_single(mesh_device: ttnn.MeshDevice, seq_len: int, use_rope: bool) -> None:
    """TP=4 LINE — single Wan config, composite vs fused."""
    submesh = _make_line_submesh_tp4(mesh_device)
    topology = ttnn.Topology.Linear
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=topology)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(TP_AXIS)
    n_local_heads = NUM_HEADS // 4

    logger.info(f"=== TP=4 LINE seq_len={seq_len} use_rope={use_rope} ===")
    us_composite = _bench_one(submesh, ag_sem, topology, n_local_heads, seq_len, use_rope, method="composite")
    us_fused = _bench_one(submesh, ag_sem, topology, n_local_heads, seq_len, use_rope, method="fused")
    speedup = us_composite / us_fused
    logger.info(f"composite={us_composite:.2f} us, fused={us_fused:.2f} us, speedup={speedup:.2f}x")


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((1, 8), {**ring_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_1x8_ring"],
)
@pytest.mark.parametrize(
    ("seq_len", "use_rope"),
    [pytest.param(seq_len, use_rope, id=cfg_id) for (cfg_id, seq_len, use_rope) in BENCH_CONFIGS],
)
def test_wan_rmsnorm_bench_fused_tp8_ring_single(mesh_device: ttnn.MeshDevice, seq_len: int, use_rope: bool) -> None:
    """TP=8 RING — single Wan config, composite vs fused."""
    submesh = _use_parent_mesh_tp8(mesh_device)
    topology = ttnn.Topology.Ring
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=topology)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(TP_AXIS)
    n_local_heads = NUM_HEADS // 8

    logger.info(f"=== TP=8 RING seq_len={seq_len} use_rope={use_rope} ===")
    us_composite = _bench_one(submesh, ag_sem, topology, n_local_heads, seq_len, use_rope, method="composite")
    us_fused = _bench_one(submesh, ag_sem, topology, n_local_heads, seq_len, use_rope, method="fused")
    speedup = us_composite / us_fused
    logger.info(f"composite={us_composite:.2f} us, fused={us_fused:.2f} us, speedup={speedup:.2f}x")


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
def test_wan_rmsnorm_bench_fused_tp4_line(mesh_device: ttnn.MeshDevice) -> None:
    """TP=4 LINE on natural 1x4 submesh: sweep all 7 Wan configs, composite vs fused."""
    submesh = _make_line_submesh_tp4(mesh_device)
    topology = ttnn.Topology.Linear
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=topology)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(TP_AXIS)
    n_local_heads = NUM_HEADS // 4  # 10

    rows: list[dict] = []
    for cfg_id, seq_len, use_rope in BENCH_CONFIGS:
        logger.info(f"=== {cfg_id} (seq_len={seq_len}, use_rope={use_rope}) ===")
        us_composite = _bench_one(submesh, ag_sem, topology, n_local_heads, seq_len, use_rope, method="composite")
        us_fused = _bench_one(submesh, ag_sem, topology, n_local_heads, seq_len, use_rope, method="fused")
        rows.append(
            {
                "config_id": cfg_id,
                "seq_len": seq_len,
                "use_rope": use_rope,
                "us_composite": us_composite,
                "us_fused": us_fused,
                "speedup": us_composite / us_fused,
            }
        )

    _write_csv(rows, CSV_FILENAME_TP4_LINE)
    _format_summary_table(rows, "Wan2.2 DistributedRMSNorm (TP=4 LINE, natural 1x4)")


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((1, 8), {**ring_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_1x8_ring"],
)
def test_wan_rmsnorm_bench_fused_tp8_ring(mesh_device: ttnn.MeshDevice) -> None:
    """TP=8 RING on native 1x8 mesh: sweep all 7 Wan configs, composite vs fused."""
    submesh = _use_parent_mesh_tp8(mesh_device)
    topology = ttnn.Topology.Ring
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=topology)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(TP_AXIS)
    n_local_heads = NUM_HEADS // 8  # 5

    rows: list[dict] = []
    for cfg_id, seq_len, use_rope in BENCH_CONFIGS:
        logger.info(f"=== {cfg_id} (seq_len={seq_len}, use_rope={use_rope}) ===")
        us_composite = _bench_one(submesh, ag_sem, topology, n_local_heads, seq_len, use_rope, method="composite")
        us_fused = _bench_one(submesh, ag_sem, topology, n_local_heads, seq_len, use_rope, method="fused")
        rows.append(
            {
                "config_id": cfg_id,
                "seq_len": seq_len,
                "use_rope": use_rope,
                "us_composite": us_composite,
                "us_fused": us_fused,
                "speedup": us_composite / us_fused,
            }
        )

    _write_csv(rows, CSV_FILENAME_TP8_RING)
    _format_summary_table(rows, "Wan2.2 DistributedRMSNorm (TP=8 RING, native 1x8)")
