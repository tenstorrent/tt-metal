# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LoFi vs HiFi2, crossed with in0 placement and weight sharding, for every drafter matmul.

Closes out the drafter's matmul search with the one axis the other sweeps left alone — **math
fidelity** — and crosses it with the two placement axes so the combinations are measured rather
than assumed to inherit:

    arm                in0            weight              fidelity
    hifi2_dram         DRAM-IL        DRAM-IL             HiFi2      <- shipped
    lofi_dram          DRAM-IL        DRAM-IL             LoFi
    lofi_nofp32        DRAM-IL        DRAM-IL             LoFi, fp32_dest_acc OFF
    hifi2_l1in         L1-IL          DRAM-IL             HiFi2
    lofi_l1in          L1-IL          DRAM-IL             LoFi
    hifi2_shard        L1 width-sh    DRAM width-sharded  HiFi2
    lofi_shard         L1 width-sh    DRAM width-sharded  LoFi

RESULT: nothing to apply. LoFi is free of charge and free of benefit; sharding loses everywhere.

    matmul    hifi2_dram    lofi_dram   lofi_l1in   hifi2_shard   lofi_shard
    fc          654.3 us      652.0       653.3        791.4        791.0
    q_proj      112.1         111.8       111.6        133.5        133.7
    kv_proj      55.8          56.0        55.5         71.4         67.7
    o_proj      112.0         111.0       111.5        136.0        135.9
    gate        260           260        1,334        illegal      illegal
    up          260.4         259.8      1,338         426.6        350.4
    down        447.0         446.3       447.0      1,366.2      1,337.3

* **LoFi: no win, real cost.** Every LoFi-vs-HiFi2 delta on the shipped path is <= 0.9 %, at or
  below the run-to-run floor (the same ``fc`` config measured 652.6 us in the fc sweep and 654.3 us
  here, so +/-0.3 % is the resolution). Per-matmul PCC drops 0.999967 -> 0.999880, and
  ``lofi_nofp32`` to 0.9996. The report's own columns predicted it: **60-69 % of DRAM bandwidth
  against 9-17 % of FLOPs peak** means the math passes are not on the critical path, so halving them
  buys nothing while still truncating the BFP8 mantissa.
* **Sharding: loses at every shape** — +19 % to +198 %, and illegal on ``gate`` (the DRAM-sharded
  matmul rejects a fused activation).
* **The one informative loser: on the SHARDED path LoFi *does* pay** — ``up`` 426.6 -> 350.4 us
  (-18 %), ``kv_proj`` 71.4 -> 67.7 (-5 %). Same fidelity change, opposite outcome, and it confirms
  the mechanism rather than just the numbers: the sharded path is compute/mcast-bound, so fidelity
  sits on its critical path; the shipped path is bandwidth-bound, so it does not. The only way to
  make LoFi matter for these matmuls is to first make them ~64 % slower.

WHAT TO EXPECT, AND WHY IT WAS STILL WORTH RUNNING
--------------------------------------------------
Fidelity buys time only on the math passes: LoFi is one pass where HiFi2 is two. These matmuls
report **60–69 % of DRAM bandwidth against 9–17 % of FLOPs peak**, i.e. they are weight-streaming,
so halving the math should do close to nothing — and it costs accuracy, because LoFi truncates the
BFP8 weight's mantissa. The reason to measure anyway is that the four ``SLOW``-tagged shapes are
NOT DRAM-bound by the report's own classification, so something else is on their critical path and
fidelity is a candidate for it.

PCC is recorded per arm because that is the deciding number if any LoFi arm wins on time: the full
drafter sits at 0.9942 against a 0.99 gate (``tests/test_dflash_drafter_tp.py``), so there is very
little accuracy headroom left to spend — the BFP4 MLP already spent most of it.

Shapes, dtypes and program configs mirror production **as shipped** (BFP4 gate/up, explicit 1D
configs on q/kv only). Companion files: ``..._attn_...``, ``..._mlp_...``, ``..._fc_...`` (program
configs per call site) and ``..._in0_l1_...`` (in0 placement alone, where L1 was +415 % on
gate/up).

Run::

    QWEN_DFLASH_MM_SWEEP=1 MESH_DEVICE=N150 python -m tracy -p --op-support-count 100000 -r -v -m \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_dflash_fidelity_sharding_sweep.py

    python models/demos/blackhole/qwen36/tests/perf/dflash_mm_sweep_report.py \\
      generated/profiler/reports/<dir>/ops_perf_results_<dir>.csv
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt import tp_common as tpc

_SKIP = os.environ.get("QWEN_DFLASH_MM_SWEEP") != "1"
ITERS = 5

try:
    from tracy import signpost as _SP
except ImportError:  # pragma: no cover
    _SP = None

HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
)
LOFI = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
LOFI_NO_FP32 = tpc.COMPUTE_LOFI_NO_FP32_ACC

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG

DIM, HIDDEN = 5120, 17408
#: (label, M, K, N, weight dtype, fused activation, explicit 1D progcfg?)
CASES = [
    ("fc", 16, 25600, DIM, ttnn.bfloat8_b, None, False),
    ("q_proj", 16, DIM, 4096, ttnn.bfloat8_b, None, True),
    ("kv_proj", 32, DIM, 2048, ttnn.bfloat8_b, None, True),
    ("o_proj", 16, 4096, DIM, ttnn.bfloat8_b, None, False),
    ("gate", 16, DIM, HIDDEN, ttnn.bfloat4_b, "silu", False),
    ("up", 16, DIM, HIDDEN, ttnn.bfloat4_b, None, False),
    ("down", 16, HIDDEN, DIM, ttnn.bfloat8_b, None, False),
]


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}.get(
        name, (1, 1)
    )


MESH_SHAPE = _mesh_shape()
_MULTI = MESH_SHAPE != (1, 1)


@pytest.mark.skipif(_SKIP, reason="set QWEN_DFLASH_MM_SWEEP=1 to run the fidelity/sharding sweep")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {})}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dflash_fidelity_sharding_sweep(mesh_device, device_params):
    """LoFi/HiFi2 x {in0 DRAM, in0 L1, both sharded} for every matmul in the drafter."""
    del device_params
    mesh_device.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    mapper = {"mesh_mapper": rep} if rep else {}

    for label, M, K, N, w_dt, act, use_pc in CASES:
        torch.manual_seed(0)
        x_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.05
        w_t = torch.randn(K, N, dtype=torch.bfloat16) * (K**-0.5)
        ref = x_t.float()[0, 0] @ w_t.float()
        if act:
            ref = torch.nn.functional.silu(ref)

        w = ttnn.from_torch(w_t, dtype=w_dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
        x_dram = ttnn.from_torch(
            x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=_DRAM, **mapper
        )
        x_l1 = ttnn.to_memory_config(x_dram, _L1)
        pc = tpc.create_matmul_1d_decode_progcfg(M, K, N, num_cores=64, grid_w=8) if use_pc else None
        logger.info(f"--- {label}: M={M} K={K} N={N} {w_dt} progcfg={'1D 8x8' if use_pc else 'auto'}")

        # (arm, in0, weight, program_config, compute config, out memcfg)
        arms = [
            ("hifi2_dram", x_dram, w, pc, HIFI2, _DRAM),
            ("lofi_dram", x_dram, w, pc, LOFI, _DRAM),
            ("lofi_nofp32", x_dram, w, pc, LOFI_NO_FP32, _DRAM),
            ("hifi2_l1in", x_l1, w, pc, HIFI2, _DRAM),
            ("lofi_l1in", x_l1, w, pc, LOFI, _DRAM),
        ]
        try:
            w_sh = ttnn.from_torch(
                w_t,
                dtype=w_dt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=tpc.create_dram_sharded_mem_config(K, N),
                **mapper,
            )
            x_sh = ttnn.to_memory_config(x_dram, tpc.create_activation_shard_config(K))
            pc_sh = tpc.create_dram_sharded_matmul_program_config(M, K, N)
            arms += [
                ("hifi2_shard", x_sh, w_sh, pc_sh, HIFI2, ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
                ("lofi_shard", x_sh, w_sh, pc_sh, LOFI, ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
            ]
        except Exception as e:
            logger.warning(f"  shard setup UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:130]}")
            w_sh = x_sh = None

        for arm, xin, win, pcfg, ckc, out_cfg in arms:
            kwargs = dict(compute_kernel_config=ckc, memory_config=out_cfg)
            if pcfg is not None:
                kwargs["program_config"] = pcfg
            if act:
                kwargs["activation"] = act
            try:
                o = ttnn.linear(xin, win, **kwargs)
                got = ttnn.to_torch(ttnn.get_device_tensors(o)[0] if _MULTI else o).float()[0, 0]
                pcc = float(comp_pcc(ref, got, 0.0)[1])
                ttnn.deallocate(o)
            except Exception as e:
                logger.warning(f"  {arm:13} UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
                continue
            if _SP is not None:
                _SP(f"{label}__{arm}_start")
            for _ in range(ITERS):
                ttnn.deallocate(ttnn.linear(xin, win, **kwargs))
            ttnn.synchronize_device(mesh_device)
            if _SP is not None:
                _SP(f"{label}__{arm}_stop")
            logger.info(f"  {arm:13} pcc={pcc:.6f}")

        for t in (w, x_dram, x_l1, w_sh, x_sh):
            if t is not None:
                ttnn.deallocate(t)
