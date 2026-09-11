# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Can the drafter's SiLU be fused away? Three ways, measured over the whole SwiGLU block.

``_layer_mlp`` passes ``activation="silu"`` to ``gate_proj``, but on the **auto** program config
ttnn does NOT fuse that into the packer — it emits a separate ``UnaryDeviceOperation``, visible in
the profile as 5 ops / 94 us per step. So the SwiGLU block currently costs four ops per layer:

    gate matmul (auto, BFP4)   264 us
    Unary (the silu)            18 us   <- the op this file is trying to remove
    up matmul (auto, BFP4)     266 us
    BinaryNg (the mul)          26 us
                              -------
                               574 us per layer, 2,870 us per step

THE THREE OPTIONS
-----------------
1. ``packer_fused`` — hand ``gate_proj`` an explicit 1D progcfg with ``fused_activation=SILU``,
   which genuinely fuses (one op, no Unary). Known catch, measured at BFP8 in
   ``test_dflash_mlp_matmul_sweep.py``: the explicit config is ~8 % slower on the matmul than auto,
   which at BFP8 cost 38 us to save 18 us. Re-measured here because gate is **BFP4** now and 8 % of
   264 us is a smaller number than 8 % of 473 us.
2. ``swiglu_fused`` — fuse the gate|up **weights** into one matmul and call ``ttnn.swiglu``, which
   splits the result and does silu-and-multiply in one op, removing BOTH the Unary and the mul.
   Note ``ttnn.swiglu`` applies SiLU to the **second** half, so the fused weight must be ordered
   ``[up | gate]``, not ``[gate | up]``. Catch: the fused matmul measured +33 us against the two
   separate ones at BFP4 (554 vs 521), because N = 34816 blocks worse than N = 17408.
3. ``manual_silu`` — a control, not a candidate: auto matmul with no activation plus an explicit
   ``ttnn.silu``. Same op count as today; it isolates how much of option 1's cost is the progcfg
   rather than the fusion.

Whole-block totals are what decide, not the matmul alone — option 2 changes the mul as well, and
option 1's saving and its cost land on different ops.

Run::

    QWEN_DFLASH_MM_SWEEP=1 MESH_DEVICE=N150 python -m tracy -p --op-support-count 100000 -r -v -m \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_dflash_swiglu_fusion_sweep.py

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

M, DIM, HIDDEN = 16, 5120, 17408
W_DTYPE = ttnn.bfloat4_b  # as shipped for gate/up


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}.get(
        name, (1, 1)
    )


MESH_SHAPE = _mesh_shape()
_MULTI = MESH_SHAPE != (1, 1)


@pytest.mark.skipif(_SKIP, reason="set QWEN_DFLASH_MM_SWEEP=1 to run the SwiGLU fusion sweep")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {})}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dflash_swiglu_fusion_sweep(mesh_device):
    """Four ways to compute silu(x@Wg) * (x@Wu), timed as whole blocks."""
    mesh_device.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    mapper = {"mesh_mapper": rep} if rep else {}

    torch.manual_seed(0)
    x_t = torch.randn(1, 1, M, DIM, dtype=torch.bfloat16) * 0.05
    wg_t = torch.randn(DIM, HIDDEN, dtype=torch.bfloat16) * (DIM**-0.5)
    wu_t = torch.randn(DIM, HIDDEN, dtype=torch.bfloat16) * (DIM**-0.5)
    ref = torch.nn.functional.silu(x_t.float()[0, 0] @ wg_t.float()) * (x_t.float()[0, 0] @ wu_t.float())

    def up(t, dtype=W_DTYPE):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)

    x, wg, wu = up(x_t, ttnn.bfloat16), up(wg_t), up(wu_t)
    # ttnn.swiglu applies SiLU to the SECOND half, so the fused weight is [up | gate].
    w_ug = up(torch.cat([wu_t, wg_t], dim=-1))
    pc_silu = tpc.create_matmul_1d_decode_progcfg(
        M, DIM, HIDDEN, num_cores=64, fused_activation=ttnn.UnaryOpType.SILU, grid_w=8
    )
    pc_plain = tpc.create_matmul_1d_decode_progcfg(M, DIM, HIDDEN, num_cores=64, grid_w=8)

    def current():
        g = ttnn.linear(x, wg, activation="silu", compute_kernel_config=CKC, memory_config=_DRAM)
        u = ttnn.linear(x, wu, compute_kernel_config=CKC, memory_config=_DRAM)
        h = ttnn.mul(g, u, memory_config=_DRAM)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        return h

    def packer_fused():
        g = ttnn.linear(x, wg, compute_kernel_config=CKC, program_config=pc_silu, memory_config=_DRAM)
        u = ttnn.linear(x, wu, compute_kernel_config=CKC, memory_config=_DRAM)
        h = ttnn.mul(g, u, memory_config=_DRAM)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        return h

    def swiglu_fused():
        gu = ttnn.linear(x, w_ug, compute_kernel_config=CKC, memory_config=_DRAM)
        h = ttnn.swiglu(gu, dim=-1, memory_config=_DRAM)
        ttnn.deallocate(gu)
        return h

    def manual_silu():
        g = ttnn.linear(x, wg, compute_kernel_config=CKC, program_config=pc_plain, memory_config=_DRAM)
        s = ttnn.silu(g, memory_config=_DRAM)
        ttnn.deallocate(g)
        u = ttnn.linear(x, wu, compute_kernel_config=CKC, memory_config=_DRAM)
        h = ttnn.mul(s, u, memory_config=_DRAM)
        ttnn.deallocate(s)
        ttnn.deallocate(u)
        return h

    logger.info(f"--- SwiGLU block: M={M} DIM={DIM} HIDDEN={HIDDEN} weights={W_DTYPE}")
    for name, fn in [
        ("current", current),
        ("packer_fused", packer_fused),
        ("swiglu_fused", swiglu_fused),
        ("manual_silu", manual_silu),
    ]:
        try:
            o = fn()
            # ttnn.swiglu returns the TILE-PADDED height (M=32 for a 16-row input, verified
            # directly), so compare only the real rows -- and see the docstring: that padding is
            # what disqualifies the arm at the call site, not its speed.
            got = ttnn.to_torch(ttnn.get_device_tensors(o)[0] if _MULTI else o).float()[0, 0][:M]
            pcc = float(comp_pcc(ref, got, 0.0)[1])
            ttnn.deallocate(o)
        except Exception as e:
            logger.warning(f"  {name:14} UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:130]}")
            continue
        if _SP is not None:
            _SP(f"swiglu__{name}_start")
        for _ in range(ITERS):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        if _SP is not None:
            _SP(f"swiglu__{name}_stop")
        logger.info(f"  {name:14} pcc={pcc:.6f}")

    for t in (x, wg, wu, w_ug):
        ttnn.deallocate(t)
