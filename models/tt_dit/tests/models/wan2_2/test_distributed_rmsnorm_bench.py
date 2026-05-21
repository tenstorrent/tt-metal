# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Benchmark DistributedRMSNorm as called by Wan2.2 attention.

Emulates a single TP=4 ring slice of the Wan2.2 14B 720p call sites by
opening the BH loudbox (2x4 parent) and carving a 2x2 submesh (which forms a
4-cycle) that is then reshaped to 1x4 so the wraparound links exist.

Sweeps the seven invocation shapes from the production attention block:
  - self-attention spatial Q/K (RoPE fused in) at N in {18944, 9472, 2368}
  - cross-attention spatial Q (no RoPE)        at N in {18944, 9472, 2368}
  - cross-attention prompt   K (no RoPE)       at L = 512

dtype=None for all configs (bfloat8_b output cast is intentionally excluded).

Two implementations are benchmarked:
  - "reference": the current 3-op Python flow via ``DistributedRMSNorm.forward``.
  - "fused":     the new ``ttnn.experimental.wan_fused_distributed_rmsnorm`` op
                 (first-draft composite C++ wrapper).
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

from ....layers.normalization import DistributedRMSNorm
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.tensor import bf16_tensor, from_torch, to_torch
from ....utils.test import ring_params

# Wan2.2 14B model configuration
DIM = 5120
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS  # 128

# TP factor emulated by the 1x4 ring submesh
TP_FACTOR = 4
N_LOCAL_HEADS = NUM_HEADS // TP_FACTOR  # 10

NORM_EPS = 1e-6
B = 1
NUM_ITERS = 100
TP_AXIS = 1  # axis of the reshaped 1x4 submesh that holds the TP cluster

CSV_FILENAME = "distributed_rmsnorm_bench_results.csv"

# (config_id, seq_len, use_rope)
BENCH_CONFIGS = [
    ("self_sp4_N18944", 18944, True),
    ("self_sp8_N9472", 9472, True),
    ("self_sp32_N2368", 2368, True),
    ("cross_q_sp4_N18944", 18944, False),
    ("cross_q_sp8_N9472", 9472, False),
    ("cross_q_sp32_N2368", 2368, False),
    ("cross_k_prompt_L512", 512, False),
]


@dataclass
class _BuiltInputs:
    tt_input: ttnn.Tensor
    tt_rope_cos: ttnn.Tensor | None
    tt_rope_sin: ttnn.Tensor | None
    tt_trans_mat: ttnn.Tensor | None


def _make_ring_submesh(parent_mesh: ttnn.MeshDevice) -> ttnn.MeshDevice:
    """Carve a 2x2 submesh (4-cycle) and present it as 1x4 so wraparound links exist."""
    submesh = parent_mesh.create_submesh(ttnn.MeshShape(2, 2))
    submesh.reshape(ttnn.MeshShape(1, 4))
    return submesh


def _build_module(submesh: ttnn.MeshDevice, ccl_manager: CCLManager) -> DistributedRMSNorm:
    """Build a DistributedRMSNorm module with random TP-sharded weight."""
    tt_model = DistributedRMSNorm(
        embedding_dim=DIM,
        norm_eps=NORM_EPS,
        norm_elementwise_affine=True,
        bias=False,
        mesh_device=submesh,
        mesh_axis=TP_AXIS,
        ccl_manager=ccl_manager,
    )
    tt_model.load_torch_state_dict({"weight": torch.randn(DIM, dtype=torch.bfloat16)})
    return tt_model


def _build_inputs(submesh: ttnn.MeshDevice, seq_len: int, use_rope: bool) -> _BuiltInputs:
    torch.manual_seed(0)
    torch_input = torch.randn((1, B, seq_len, DIM), dtype=torch.bfloat16)
    tt_input = bf16_tensor(torch_input, device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)

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

    return _BuiltInputs(tt_input, tt_rope_cos, tt_rope_sin, tt_trans_mat)


def _run_reference(tt_model: DistributedRMSNorm, inp: _BuiltInputs) -> ttnn.Tensor:
    return tt_model(
        inp.tt_input,
        num_heads_per_device=N_LOCAL_HEADS,
        rope_cos=inp.tt_rope_cos,
        rope_sin=inp.tt_rope_sin,
        trans_mat=inp.tt_trans_mat,
        dtype=None,
    )


def _run_fused(
    tt_model: DistributedRMSNorm,
    ccl_manager: CCLManager,
    submesh: ttnn.MeshDevice,
    inp: _BuiltInputs,
    seq_len: int,
) -> ttnn.Tensor:
    """Call the new fused composite op directly, reusing the module's TP-sharded weight."""
    ag_buffer = ccl_manager.get_ag_ping_pong_buffer(
        shape=(1, 1, seq_len, 32),
        dim=3,
        mesh_axis=TP_AXIS,
        dtype=ttnn.float32,
    )
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(TP_AXIS)
    return ttnn.experimental.wan_fused_distributed_rmsnorm(
        inp.tt_input,
        TP_AXIS,
        submesh,
        ag_sem,
        topology=ttnn.Topology.Ring,
        epsilon=NORM_EPS,
        num_heads_per_device=N_LOCAL_HEADS,
        weight=tt_model.weight.data,
        transformation_mat=inp.tt_trans_mat,
        rope_cos=inp.tt_rope_cos,
        rope_sin=inp.tt_rope_sin,
        dtype=None,
        persistent_output_buffer=ag_buffer,
        num_preferred_links=ccl_manager.num_links,
        compute_kernel_config=tt_model.compute_kernel_config,
    )


def _trace_and_time(submesh: ttnn.MeshDevice, run_op, *, num_iters: int = NUM_ITERS) -> float:
    """Compile, capture trace, run num_iters traced iterations, return avg us/iter."""
    run_op()
    ttnn.synchronize_device(submesh)

    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    run_op()
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

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
    ccl_manager: CCLManager,
    seq_len: int,
    use_rope: bool,
    *,
    method: str = "reference",
) -> float:
    """Run one (seq_len, use_rope) config with the chosen method, return avg us/iter."""
    tt_model = _build_module(submesh, ccl_manager)
    inp = _build_inputs(submesh, seq_len, use_rope)

    if method == "reference":
        run_op = lambda: _run_reference(tt_model, inp)  # noqa: E731
    elif method == "fused":
        run_op = lambda: _run_fused(tt_model, ccl_manager, submesh, inp, seq_len)  # noqa: E731
    else:
        raise ValueError(f"unknown method: {method}")

    logger.info(f"[{method}] compiling+tracing (seq_len={seq_len}, use_rope={use_rope})")
    per_iter_us = _trace_and_time(submesh, run_op)
    logger.info(f"[{method}] seq_len={seq_len} use_rope={use_rope}: {per_iter_us:.2f} us/iter")
    return per_iter_us


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


_DEVICE_PARAMS = [
    ((2, 4), {**ring_params, "trace_region_size": 90112}),
]


@pytest.mark.parametrize(("mesh_device", "device_params"), _DEVICE_PARAMS, indirect=True, ids=["bh_2x4_ring"])
@pytest.mark.parametrize(
    ("seq_len", "use_rope"),
    [pytest.param(seq_len, use_rope, id=cfg_id) for (cfg_id, seq_len, use_rope) in BENCH_CONFIGS],
)
def test_distributed_rmsnorm_bench(mesh_device: ttnn.MeshDevice, seq_len: int, use_rope: bool) -> None:
    submesh = _make_ring_submesh(mesh_device)
    ccl_manager = CCLManager(mesh_device=submesh, num_links=2, topology=ttnn.Topology.Ring)
    _bench_one(submesh, ccl_manager, seq_len, use_rope, method="reference")


@pytest.mark.parametrize(("mesh_device", "device_params"), _DEVICE_PARAMS, indirect=True, ids=["bh_2x4_ring"])
@pytest.mark.parametrize(
    ("seq_len", "use_rope"),
    [pytest.param(seq_len, use_rope, id=cfg_id) for (cfg_id, seq_len, use_rope) in BENCH_CONFIGS],
)
def test_wan_fused_distributed_rmsnorm_bench(mesh_device: ttnn.MeshDevice, seq_len: int, use_rope: bool) -> None:
    submesh = _make_ring_submesh(mesh_device)
    ccl_manager = CCLManager(mesh_device=submesh, num_links=2, topology=ttnn.Topology.Ring)
    _bench_one(submesh, ccl_manager, seq_len, use_rope, method="fused")


@pytest.mark.parametrize(("mesh_device", "device_params"), _DEVICE_PARAMS, indirect=True, ids=["bh_2x4_ring"])
@pytest.mark.parametrize(
    ("seq_len", "use_rope"),
    [pytest.param(seq_len, use_rope, id=cfg_id) for (cfg_id, seq_len, use_rope) in BENCH_CONFIGS],
)
def test_wan_fused_distributed_rmsnorm_correctness(mesh_device: ttnn.MeshDevice, seq_len: int, use_rope: bool) -> None:
    """Run reference (DistributedRMSNorm.forward) and fused op on identical inputs; assert PCC."""
    submesh = _make_ring_submesh(mesh_device)
    ccl_manager = CCLManager(mesh_device=submesh, num_links=2, topology=ttnn.Topology.Ring)

    tt_model = _build_module(submesh, ccl_manager)
    inp = _build_inputs(submesh, seq_len, use_rope)

    ref_out = _run_reference(tt_model, inp)
    fused_out = _run_fused(tt_model, ccl_manager, submesh, inp, seq_len)

    # Post-allgather output has shape [1, num_heads_per_device, N, head_dim] per device,
    # with the num_heads dim (tensor dim 1) TP-fractured along mesh axis 1.
    mesh_axes = [None, TP_AXIS, None, None]
    ref_torch = to_torch(ref_out, mesh_axes=mesh_axes)
    fused_torch = to_torch(fused_out, mesh_axes=mesh_axes)

    logger.info(f"ref_out shape: {ref_torch.shape}, fused_out shape: {fused_torch.shape}")
    assert_quality(ref_torch, fused_torch, pcc=0.9995)


@pytest.mark.parametrize(("mesh_device", "device_params"), _DEVICE_PARAMS, indirect=True, ids=["bh_2x4_ring"])
def test_distributed_rmsnorm_bench_summary(mesh_device: ttnn.MeshDevice) -> None:
    """Sweep all configs for both reference and fused methods, save CSV, print summary table."""
    submesh = _make_ring_submesh(mesh_device)
    ccl_manager = CCLManager(mesh_device=submesh, num_links=2, topology=ttnn.Topology.Ring)

    rows: list[dict] = []
    for cfg_id, seq_len, use_rope in BENCH_CONFIGS:
        per_cfg = {"config_id": cfg_id, "seq_len": seq_len, "use_rope": use_rope}
        for method in ("reference", "fused"):
            logger.info(f"=== {cfg_id} [{method}] ===")
            per_cfg[f"us_per_iter_{method}"] = _bench_one(submesh, ccl_manager, seq_len, use_rope, method=method)
        per_cfg["speedup"] = per_cfg["us_per_iter_reference"] / per_cfg["us_per_iter_fused"]
        rows.append(per_cfg)

    fieldnames = ["config_id", "seq_len", "use_rope", "us_per_iter_reference", "us_per_iter_fused", "speedup"]
    csv_path = Path.cwd() / CSV_FILENAME
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"Wrote CSV: {csv_path}")

    id_w = max(len("config_id"), max(len(r["config_id"]) for r in rows))
    header = (
        f"{'config_id':<{id_w}}  {'seq_len':>7}  {'use_rope':>8}  "
        f"{'ref us/iter':>12}  {'fused us/iter':>14}  {'speedup':>8}"
    )
    sep = "-" * len(header)
    title = "DistributedRMSNorm benchmark (TP=4 ring, BH 2x4)"
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
            f"{r['seq_len']:>7d}  "
            f"{str(r['use_rope']):>8}  "
            f"{r['us_per_iter_reference']:>12.2f}  "
            f"{r['us_per_iter_fused']:>14.2f}  "
            f"{r['speedup']:>7.2f}x"
        )
    print(box)
    print(f"CSV: {csv_path}")
    print()
