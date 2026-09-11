# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-config sweep for the three matmuls in the DFlash drafter's ``_layer_mlp``.

Companion to ``test_dflash_attn_matmul_sweep.py`` — same signpost convention, same report script,
same finding to test for: at M = one tile row the **auto** program config can be far off the DRAM
roofline, and there is no way to tell which matmul is which without measuring.

The MLP is **55 % of the drafter's step** (``DFLASH_DRAFTER_OP_MAPPING.md``) and every one of its
matmuls is on auto:

    label    M   K      N       bf8 weight   roofline @288 GB/s   measured (auto)   efficiency
    gate     16  5120   17408     94.7 MB         329 us              486 us           68 %
    up       16  5120   17408     94.7 MB         329 us              480 us           69 %
    down     16  17408  5120      94.7 MB         329 us              458 us           72 %

WHAT IS SWEPT

Same families as the attention sweep: ``auto``, ``1d_nc<N>`` (``create_matmul_1d_decode_progcfg``,
explicit wide-first grid, mcast_in0), ``_l1out``, ``dramshard`` (+ ``dramshard_pre`` for the
reshard-free ceiling), ``prefill_mlp``, and ``COMPUTE_HIFI2_NO_FP32_ACC``.

``gate`` is swept **with its SILU fused in the packer**, which is how the drafter calls it — a
config that cannot take the fused activation is not a candidate for this call site.

One extra, structural rather than a config: ``gate_up_fused``, one column-parallel matmul emitting
``[gate | up]`` side by side. gate and up read the same normed activation, which is the same lever
that paid for ``kv_proj`` in the attention path. It is reported but NOT free to adopt — SILU cannot
be fused in the packer for a fused weight (it would hit the ``up`` half too), so the call site would
have to slice the halves apart and silu one of them, and those ops are not in this matmul's number.
Read it as a ceiling, and subtract 2 slices + 1 unary on a 1.1 MB tensor before believing it.

Run::

    QWEN_DFLASH_MM_SWEEP=1 MESH_DEVICE=N150 python -m tracy -p --op-support-count 100000 -r -v -m \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_dflash_mlp_matmul_sweep.py

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
CKC_NO_FP32 = tpc.COMPUTE_HIFI2_NO_FP32_ACC

DIM, HIDDEN = 5120, 17408
#: (label, M, K, N, fused activation) — M is the LOGICAL row count (the 16-slot block).
CASES = [
    ("gate", 16, DIM, HIDDEN, ttnn.UnaryOpType.SILU),
    ("up", 16, DIM, HIDDEN, None),
    ("down", 16, HIDDEN, DIM, None),
    ("gate_up_fused", 16, DIM, 2 * HIDDEN, None),
]
CORE_COUNTS = [8, 16, 24, 32, 40, 48, 56, 64]
L1_OUT_CORES = [32, 48, 64]
NOFP32_CORES = [32, 64]
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}.get(
        name, (1, 1)
    )


MESH_SHAPE = _mesh_shape()
_MULTI = MESH_SHAPE != (1, 1)


@pytest.mark.skipif(_SKIP, reason="set QWEN_DFLASH_MM_SWEEP=1 to run the drafter MLP matmul sweep")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {})}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dflash_mlp_matmul_sweep(mesh_device, device_params):
    """Sweep program configs for gate / up / down at the drafter's real shapes."""
    del device_params
    mesh_device.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    mapper = {"mesh_mapper": rep} if rep else {}

    for label, M, K, N, act in CASES:
        torch.manual_seed(0)
        x_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.05
        w_t = torch.randn(K, N, dtype=torch.bfloat16) * (K**-0.5)
        ref = x_t.float()[0, 0] @ w_t.float()
        if act is not None:
            ref = torch.nn.functional.silu(ref)
        wbytes = K * N * 1.0625 / 1e6

        x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
        w = ttnn.from_torch(w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
        logger.info(
            f"--- {label}: M={M}(pad 32) K={K} N={N} silu={act is not None} | bf8 weight {wbytes:.1f} MB "
            f"| roofline @288GB/s = {wbytes * 1000 / 288:.0f} us"
        )

        # ttnn.linear takes the fused activation as a string; the progcfg factory takes the enum.
        act_str = "silu" if act is not None else None
        extra_weights: list = []
        cands: list[tuple[str, callable]] = [
            ("auto", lambda: ttnn.linear(x, w, activation=act_str, compute_kernel_config=CKC, memory_config=_DRAM)),
        ]
        seen = set()
        for nc in CORE_COUNTS:
            pc = tpc.create_matmul_1d_decode_progcfg(M, K, N, num_cores=nc, fused_activation=act, grid_w=8)
            grid = (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)
            key = (grid, pc.per_core_N, pc.in0_block_w, pc.out_subblock_h, pc.out_subblock_w)
            if key in seen:
                continue
            seen.add(key)
            tag = f"g{grid[0]}x{grid[1]}_pcN{pc.per_core_N}_sub{pc.out_subblock_h}x{pc.out_subblock_w}"
            cands.append(
                (
                    f"1d_nc{nc}_{tag}",
                    lambda pc=pc: ttnn.linear(x, w, compute_kernel_config=CKC, program_config=pc, memory_config=_DRAM),
                )
            )
            if nc in L1_OUT_CORES:
                cands.append(
                    (
                        f"1d_nc{nc}_l1out",
                        lambda pc=pc: ttnn.linear(
                            x, w, compute_kernel_config=CKC, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG
                        ),
                    )
                )
            if nc in NOFP32_CORES:
                pc8 = tpc.create_matmul_1d_decode_progcfg(
                    M, K, N, num_cores=nc, fused_activation=act, fp32_acc=False, grid_w=8
                )
                cands.append(
                    (
                        f"1d_nc{nc}_nofp32_sub{pc8.out_subblock_h}x{pc8.out_subblock_w}",
                        lambda pc=pc8: ttnn.linear(
                            x, w, compute_kernel_config=CKC_NO_FP32, program_config=pc, memory_config=_DRAM
                        ),
                    )
                )

        try:
            w_ds = ttnn.from_torch(
                w_t,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=tpc.create_dram_sharded_mem_config(K, N),
                **mapper,
            )
            pc_ds = tpc.create_dram_sharded_matmul_program_config(M, K, N)
            act_cfg = tpc.create_activation_shard_config(K)
            x_ds = ttnn.to_memory_config(x, act_cfg)

            def _dramshard():
                xs = ttnn.to_memory_config(x, act_cfg)
                o = ttnn.linear(
                    xs,
                    w_ds,
                    activation=act_str,
                    compute_kernel_config=CKC,
                    program_config=pc_ds,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                ttnn.deallocate(xs)
                return o

            cands.append(("dramshard", _dramshard))
            cands.append(
                (
                    "dramshard_pre",
                    lambda: ttnn.linear(
                        x_ds,
                        w_ds,
                        activation=act_str,
                        compute_kernel_config=CKC,
                        program_config=pc_ds,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    ),
                )
            )
        except Exception as e:
            logger.warning(f"  dramshard setup UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            w_ds = x_ds = None

        # WEIGHT DTYPE, not a program config -- but it is the only axis left once auto wins, and the
        # TARGET model already ships bf4 for its own gate/up (see test_mlp_decode_matmul_sweep.py).
        # These matmuls are pure weight streaming, so half the bytes should be ~half the time. PCC
        # is what decides, and PCC is not the drafter's real metric (acceptance is), so this arm is
        # reported and NOT adopted here.
        for dt_name, dt in [("bf4", ttnn.bfloat4_b)]:
            try:
                w_dt = ttnn.from_torch(
                    w_t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=_DRAM, **mapper
                )
            except Exception as e:
                logger.warning(f"  {dt_name} setup UNSUPPORTED: {str(e).splitlines()[0][:130]}")
                continue
            cands.append(
                (
                    f"auto_{dt_name}",
                    lambda w_dt=w_dt: ttnn.linear(
                        x, w_dt, activation=act_str, compute_kernel_config=CKC, memory_config=_DRAM
                    ),
                )
            )
            extra_weights.append(w_dt)

        cands.append(
            (
                "prefill_mlp",
                lambda: ttnn.linear(
                    x,
                    w,
                    activation=act_str,
                    compute_kernel_config=CKC,
                    program_config=tpc.create_prefill_mlp_matmul_program_config(M, K, N, max_cols=8),
                    memory_config=_DRAM,
                ),
            )
        )

        for name, builder in cands:
            try:
                o = builder()
                got = ttnn.to_torch(ttnn.get_device_tensors(o)[0] if _MULTI else o).float()[0, 0]
                pcc = float(comp_pcc(ref, got, 0.0)[1])
                ttnn.deallocate(o)
            except Exception as e:
                logger.warning(f"  {name:34} UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:130]}")
                continue
            if _SP is not None:
                _SP(f"{label}__{name}_start")
            for _ in range(ITERS):
                ttnn.deallocate(builder())
            ttnn.synchronize_device(mesh_device)
            if _SP is not None:
                _SP(f"{label}__{name}_stop")
            logger.info(f"  {name:34} pcc={pcc:.6f}")

        for t in (x, w, w_ds, x_ds, *extra_weights):
            if t is not None:
                ttnn.deallocate(t)
