# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests (8-chip TP ring, one Galaxy row with QWEN36_FABRIC_RING=1) of QWEN36_PREFILL_SHORT_DECODE_MM, op level.

For the four prefill AGMMs as the model calls them (tp_common.all_gather_matmul_prefill / all_gather_swiglu_prefill) at
64 / 128 / 256 rows: today's fused all_gather_minimal_matmul_async vs the short path (all_gather_async + 1D mcast_in0
matmul on the same weight, + the split-SwiGLU kernel for gate/up).

* in-projections (gdn_qkvzab, attn_qkv, gdn_out col-par): expected BIT-IDENTICAL (same weight, same compute config,
  same K order). SHORT_MM_REQUIRE_BITEXACT=0 downgrades a mismatch to a logged PCC / max-abs report.
* mlp_gateup: not bit-identical by design (SiLU on the bf16-rounded gate, mlp_swiglu.py); PCC >= 0.9999 required.
* timing: eager (host dispatch included) and TRACED replay (as the production prefill buckets run), both arms, logged
  as "[SHORT_MM] name rows=R: agmm A ms -> short B ms (traced a -> b)". The gate for keeping the flag is the traced number.

  source <8-chip env>; QWEN36_FABRIC_RING=1 python -m pytest \
      models/demos/blackhole/qwen36/tests/test_prefill_short_decode_mm.py -s -p no:cacheprovider
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import TT_CCL

_ROUTER = tpc.fabric_router_config()
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "trace_region_size": 1 << 27,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        **({"fabric_router_config": _ROUTER} if _ROUTER is not None else {}),
    }
]
REPS = 20
REQUIRE_BITEXACT = os.environ.get("SHORT_MM_REQUIRE_BITEXACT", "1") == "1"
# name -> (swiglu, K, N/tp, compute config, persistent gather buffer, weight dtype) as the model calls them on TP8
SHAPES = {
    "mlp_gateup": (True, 5120, 4352, "mlp", False, ttnn.bfloat4_b),
    "gdn_qkvzab": (False, 5120, 2060, "hifi2", False, ttnn.bfloat8_b),
    "attn_qkv": (False, 5120, 2048, "hifi2", True, ttnn.bfloat8_b),
    "gdn_out": (False, 6144, 640, "hifi2", True, ttnn.bfloat8_b),
}
ROWS = [64, 128, 256]
CASES = [(n, r) for n in SHAPES for r in ROWS]


def _compute_cfg(name):
    if name == "mlp":  # Qwen36MLP.compute_kernel_config_agmm under QWEN36_MLP_FIDELITY=lofi (prod)
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
    return tpc.COMPUTE_HIFI2


def _read(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _time_eager(mesh, fn, reps=REPS):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return 1e3 * (time.perf_counter() - t0) / reps


def _time_traced(mesh, fn, reps=REPS):
    ttnn.deallocate(fn())  # compile outside the trace
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = 1e3 * (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    return dt


@pytest.mark.parametrize("name,rows", CASES, ids=[f"{n}-{r}" for n, r in CASES])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_short_decode_mm(mesh_device, name, rows, monkeypatch):
    mesh = mesh_device
    nd = mesh.get_num_devices()
    if nd != 8:
        pytest.skip("TP8 shapes")
    tt_ccl = TT_CCL(mesh)
    swiglu, k, n, cname, pbuf, wdt = SHAPES[name]
    torch.manual_seed(rows + k + n)
    shard = ttnn.ShardTensorToMesh(mesh, dim=-1)
    w = ttnn.from_torch(
        torch.randn(1, 1, k, n * nd) * 0.02,
        dtype=wdt,
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
    for arm, val in (("agmm", "0"), ("short", "256")):
        monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM", val)
        assert tpc.short_decode_mm_ok(rows, "gateup" if swiglu else "inproj") == (arm == "short")
        out = run()
        got = _read(mesh, out)
        ttnn.deallocate(out)
        res[arm] = (got, _time_eager(mesh, run), _time_traced(mesh, run))
    (ref, e0, t0), (got, e1, t1) = res["agmm"], res["short"]
    same = torch.equal(_bits(got), _bits(ref))
    pcc = _pcc(got, ref)
    mad = (got.float() - ref.float()).abs().max().item()
    logger.info(
        f"[SHORT_MM] {name} rows={rows}: agmm {e0:.4f} ms -> short {e1:.4f} ms (traced {t0:.4f} -> {t1:.4f}) "
        f"bitexact={same} pcc={pcc:.6f} maxabs={mad:.3g}"
    )
    if swiglu:
        assert pcc >= 0.9999, f"{name} rows={rows}: short gate/up PCC {pcc}"
    elif REQUIRE_BITEXACT:
        assert same, f"{name} rows={rows}: short path differs from the AGMM (pcc {pcc}, max abs {mad})"
    else:
        assert pcc >= 0.99999, f"{name} rows={rows}: short path PCC {pcc}"
