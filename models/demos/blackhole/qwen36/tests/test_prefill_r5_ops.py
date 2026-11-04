# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device tests (8-chip TP ring, e.g. one Galaxy row with QWEN36_FABRIC_RING=1) of two r5 prefill flags, op level:

* QWEN36_PREFILL_AGMM_TUNED: every (shape, rows) entry of tp_common._AGMM_TUNED, through all_gather_matmul_prefill /
  all_gather_swiglu_prefill as the model calls them, gives the bit-identical output of today's config; logs both times.
* QWEN36_PREFILL_RS_WORKERS (with QWEN36_PREFILL_RS_LINKS=2): the prefill tt_all_reduce reduce-scatter of a
  [rows, 5120] bf16 partial is bit-identical to today's (1 link, 2 workers); logs both times.

  source <8-chip env>; python -m pytest models/demos/blackhole/qwen36/tests/test_prefill_r5_ops.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce

_ROUTER = tpc.fabric_router_config()
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "trace_region_size": 1 << 26,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        **({"fabric_router_config": _ROUTER} if _ROUTER is not None else {}),
    }
]
REPS = 10
# name -> (swiglu, K, N/tp, compute config, persistent gather buffer) as the model calls the four prefill AGMMs on TP8
SHAPES = {
    "mlp_gateup": (True, 5120, 4352, "mlp", False),
    "gdn_qkvzab": (False, 5120, 2060, "hifi2", False),
    "attn_qkv": (False, 5120, 2048, "hifi2", True),
    "gdn_out": (False, 6144, 640, "hifi2", True),
}


def _key(swiglu, k, n):
    return (swiglu, -(-k // 32) * 32, -(-n // 32) * 32)


CASES = [
    (name, rows)
    for name, (swiglu, k, n, _, _) in SHAPES.items()
    for rows in sorted(tpc._AGMM_TUNED.get(_key(swiglu, k, n), {}))
]


def _compute_cfg(name):
    if name == "mlp":  # Qwen36MLP.compute_kernel_config_agmm at the default HiFi2
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
        )
    return tpc.COMPUTE_HIFI2


def _read(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def _time(mesh, fn, reps=REPS):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return 1e3 * (time.perf_counter() - t0) / reps


@pytest.mark.parametrize("name,rows", CASES, ids=[f"{n}-{r}" for n, r in CASES])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_agmm_tuned_bit_identical(mesh_device, name, rows, monkeypatch):
    mesh = mesh_device
    nd = mesh.get_num_devices()
    if nd != 8:
        pytest.skip("the tuned table is for TP8")
    tt_ccl = TT_CCL(mesh)
    swiglu, k, n, cname, pbuf = SHAPES[name]
    torch.manual_seed(rows + k + n)
    shard = ttnn.ShardTensorToMesh(mesh, dim=-1)
    w = ttnn.from_torch(
        torch.randn(1, 1, k, n * nd) * 0.02,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, rows, k),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    cfg = _compute_cfg(cname)

    def run():
        if swiglu:
            return tpc.all_gather_swiglu_prefill(x, w, tt_ccl, cfg, ttnn.Topology.Ring)
        return tpc.all_gather_matmul_prefill(
            x,
            w,
            tt_ccl,
            cfg,
            ttnn.Topology.Ring,
            persistent_output_buffer=tpc.agmm_gather_buffer(tt_ccl, x) if pbuf else None,
        )

    res = {}
    for flag in (False, True):
        monkeypatch.setattr(tpc, "_PREFILL_AGMM_TUNED", flag)
        if flag:
            assert tpc.agmm_tuned_config(rows, w, k // nd, swiglu, mesh) is not None
        out = run()
        got = _read(mesh, out)
        ttnn.deallocate(out)
        res[flag] = (got, _time(mesh, run))
    (ref, t_off), (got, t_on) = res[False], res[True]
    logger.info(f"[AGMM_TUNED] {name} rows={rows}: today {t_off:.4f} ms -> tuned {t_on:.4f} ms")
    assert torch.equal(_bits(got), _bits(ref)), f"{name} rows={rows}: tuned AGMM output differs from today's"


@pytest.mark.parametrize("rows", [128, 512, 1024, 2048, 8192])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rs_workers_bit_identical(mesh_device, rows, monkeypatch):
    mesh = mesh_device
    nd = mesh.get_num_devices()
    if nd != 8:
        pytest.skip("8-chip ring")
    tt_ccl = TT_CCL(mesh)
    torch.manual_seed(rows)
    x_t = torch.randn(1, 1, rows, 5120 * nd)
    mc = tpc.l1_or_dram(rows)  # the down-proj / wo partial: L1 up to 2048 rows, DRAM above

    def run():
        x = ttnn.from_torch(
            x_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        return x

    res = {}
    for arm, env in (("today", {}), ("tuned", {"QWEN36_PREFILL_RS_LINKS": "2", "QWEN36_PREFILL_RS_WORKERS": "8"})):
        monkeypatch.setattr(tpc, "_PREFILL_RS_LINKS", None)
        monkeypatch.setattr(tpc, "_PREFILL_RS_WORKERS", None)
        for key in ("QWEN36_PREFILL_RS_LINKS", "QWEN36_PREFILL_RS_WORKERS"):
            monkeypatch.delenv(key, raising=False)
        for key, val in env.items():
            monkeypatch.setenv(key, val)

        def call(x):  # tt_all_reduce frees its input, as in the model
            return tt_all_reduce(
                x,
                mesh,
                tt_ccl,
                cluster_axis=0,
                dim=3,
                num_reduce_scatter_links=tpc.prefill_rs_links(rows),
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **tpc.prefill_rs_kwargs(rows),
            )

        out = call(run())
        got = _read(mesh, out)
        ttnn.deallocate(out)
        ts = []
        for _ in range(REPS + 2):  # one staged input at a time (an L1 partial is ~21 MB per chip at 2048 rows)
            xx = run()
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            yy = call(xx)
            ttnn.synchronize_device(mesh)
            ts.append(1e3 * (time.perf_counter() - t0))
            ttnn.deallocate(yy)
        res[arm] = (got, sorted(ts[2:])[len(ts[2:]) // 2])  # median, synced per call
    (ref, t_off), (got, t_on) = res["today"], res["tuned"]
    logger.info(f"[RS_WORKERS] rows={rows}: today (1 link x 2) {t_off:.4f} ms -> 2 links x 8 workers {t_on:.4f} ms")
    assert torch.equal(_bits(got), _bits(ref)), f"rows={rows}: reduce-scatter output differs from today's"
