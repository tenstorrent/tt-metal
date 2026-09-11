# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does tt-perf-report's "place input 0 in L1" hint pay for ANY of the drafter's matmuls?

This is the one piece of advice in the report's Matmul Optimization section that the three
per-call-site sweeps did NOT test: every candidate in those kept in0 DRAM-interleaved, because they
were sweeping program configs. The report prints the hint on all eight ``SLOW``-bound rows:

    6881  32 x 5120 x 17408  264 us  SLOW  - If possible place input 0 in L1 (currently DRAM)
    6961  32 x 5120 x 4096   114 us  SLOW  - If possible place input 0 in L1 (currently DRAM)
    ...

so it is worth one cross-cutting run rather than three edits. The other two hints in that section
are already settled and both are wrong here: "try a DRAM-sharded program config" lost at every
shape (+11 % to +207 %, and it is illegal on ``gate`` — it rejects a fused activation), and the
HiFi4/fp32-acc variants lost on time while costing PCC.

RESULT: the hint is worthless at five shapes and CATASTROPHIC at the two it is printed on hardest

    matmul     in0 DRAM     in0 L1      delta
    fc          653.6 us    653.0 us     -0.1 %   noise
    q_proj      112.5 us    111.7 us     -0.7 %   noise
    kv_proj      55.7 us     56.4 us     +1.3 %   noise
    o_proj      111.4 us    111.2 us     -0.2 %   noise
    down        448.1 us    447.0 us     -0.2 %   noise
    gate        259.0 us  1,335.0 us   **+415 %**
    up          260.1 us  1,335.4 us   **+413 %**

``gate`` and ``up`` are the BFP4 N=17408 pair, and they are exactly the rows the report tags SLOW
and prints "If possible place input 0 in L1" on. Doing it makes them **5x slower**. PCC is identical
in every arm, so this is purely a scheduling effect: with in0 in L1 the auto config resolves to a
different (far worse) blocking for these two shapes. The mechanism is not confirmed beyond that —
what is confirmed is the direction and the size.

So all three hints in the report's Matmul Optimization section are, for this model, either neutral
or harmful. Note the scope of that claim: this file tests L1 **interleaved** in0, the literal
reading of "currently in DEV_0_DRAM_INTERLEAVED". The other reading — L1 **width-sharded** in0, as
the DRAM-sharded matmul requires — is the ``dramshard`` arm in the three per-call-site sweeps, and
it lost at every shape too. Both readings lose.

WHY IT WAS WORTH MEASURING ANYWAY
---------------------------------
These are weight-streaming ops: in0 is 164 KB–1.1 MB against 50–139 MB of weight, so moving it to
L1 removes ~0.2 % of the bytes — the volume argument says "no effect", and for five of the seven
that is exactly what happened. But in0 is also **multicast to every core**, and sourcing that from
L1 could in principle shorten the mcast regardless of volume. The reasoning genuinely did not
settle it, and the answer it did not predict was a 5x regression.

WHAT IT WOULD COST AT THE CALL SITE
-----------------------------------
Nothing extra, if it wins: each in0 is produced by an op that can be asked to write L1 directly
(``rms_norm`` for q/gate/up, ``ttnn.mul`` for down, the head-concat for o, the ``concat`` for kv).
No conversion op is needed — which is why this arm is fair to compare 1:1 with the DRAM one, unlike
the DRAM-sharded arm that had to pay for an in0 reshard.

Shapes and dtypes mirror production **as shipped**, including BFP4 gate/up.

Run::

    QWEN_DFLASH_MM_SWEEP=1 MESH_DEVICE=N150 python -m tracy -p --op-support-count 100000 -r -v -m \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_dflash_in0_l1_sweep.py

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

CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True)
_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG

DIM, HIDDEN = 5120, 17408
#: (label, M, K, N, weight dtype, fused activation, explicit progcfg?) — production as shipped.
#: ``progcfg`` marks the two call sites that carry an explicit 1D config (see drafter._proj_pc);
#: everything else runs on auto, so the arm must too or it is not measuring the same op.
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


@pytest.mark.skipif(_SKIP, reason="set QWEN_DFLASH_MM_SWEEP=1 to run the in0-placement sweep")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {})}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dflash_in0_l1_sweep(mesh_device, device_params):
    """in0 in DRAM vs in0 in L1, for every matmul in the drafter, at its shipped config."""
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
        in0_mb = M * K * 2 / 1e6
        logger.info(
            f"--- {label}: M={M} K={K} N={N} {w_dt} progcfg={'1D 8x8' if use_pc else 'auto'} "
            f"| in0 {in0_mb:.2f} MB vs weight {K * N * (0.5625 if w_dt == ttnn.bfloat4_b else 1.0625) / 1e6:.1f} MB"
        )

        for arm, xin in (("in0_dram", x_dram), ("in0_l1", x_l1)):
            kwargs = dict(compute_kernel_config=CKC, memory_config=_DRAM)
            if pc is not None:
                kwargs["program_config"] = pc
            if act:
                kwargs["activation"] = act
            try:
                o = ttnn.linear(xin, w, **kwargs)
                got = ttnn.to_torch(ttnn.get_device_tensors(o)[0] if _MULTI else o).float()[0, 0]
                pcc = float(comp_pcc(ref, got, 0.0)[1])
                ttnn.deallocate(o)
            except Exception as e:
                logger.warning(f"  {arm:10} UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:130]}")
                continue
            if _SP is not None:
                _SP(f"{label}__{arm}_start")
            for _ in range(ITERS):
                ttnn.deallocate(ttnn.linear(xin, w, **kwargs))
            ttnn.synchronize_device(mesh_device)
            if _SP is not None:
                _SP(f"{label}__{arm}_stop")
            logger.info(f"  {arm:10} pcc={pcc:.6f}")

        for t in (w, x_dram, x_l1):
            ttnn.deallocate(t)
