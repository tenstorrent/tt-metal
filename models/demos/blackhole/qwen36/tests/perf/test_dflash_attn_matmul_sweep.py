# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-config sweep for the three matmuls in the DFlash drafter's ``_layer_attention``.

WHY
---
The drafter calls all three on the **auto** program config (no ``program_config=``), and the
per-layer profile (``DFLASH_DRAFTER_OP_MAPPING.md``) shows two of them well under the DRAM
roofline. At M = 32 (one tile row — a 16-slot block padded, or 16 ctx + 16 block for k/v) these are
pure weight-streaming ops, so time should be ``weight_bytes / BW`` and nothing else:

    label     M   K     N      bf8 weight   roofline @288 GB/s   measured (auto)   efficiency
    q_proj    16  5120  4096    22.3 MB          77 us               121 us           64 %
    kv_proj   32  5120  2048    11.1 MB          39 us               116 us           33 %
    o_proj    16  4096  5120    22.3 MB          77 us               113 us           68 %

``kv_proj`` is the outlier and the report says why: N = 2048 is 64 tiles, the auto config spreads
them over 64 cores, and ``per_core_N = 1`` gives a 1x1 output subblock. Fixing that is a program
config, not a kernel.

WHAT IS SWEPT
-------------
* ``auto`` — no ``program_config``; the baseline, i.e. what the drafter does today.
* ``1d_nc<N>`` — ``create_matmul_1d_decode_progcfg`` on an explicit ~N-core wide-first grid
  (mcast_in0, interleaved weight). This is the family that already won the MLP decode matmuls.
* ``1d_nc<N>_l1`` — the same with the output in L1 rather than DRAM. Legal at the call site: q's
  output feeds ``reshape``, k/v's feeds ``nlp_create_qkv_heads`` (which only requires INTERLEAVED),
  o's feeds ``ttnn.add`` — and the tensors are 128–320 KB.
* ``dramshard`` — DRAM-WIDTH_SHARDED weight + ``create_dram_sharded_matmul_program_config``, the
  hint tt-perf-report prints for ``o_proj``. The in0 reshard it needs is INSIDE the timed window,
  because the call site would have to pay it; ``dramshard_pre`` is the same candidate with the
  reshard hoisted, i.e. the unreachable ceiling, reported only to show what the reshard costs.
* ``prefill_mlp`` / ``kpass1`` — the repo's 2D multicast prefill factories, for completeness at a
  shape they were not built for.
* ``*_nofp32`` — ``COMPUTE_HIFI2_NO_FP32_ACC`` on the leading 1D grids. ``fp32_dest_acc_en`` caps
  the output subblock area at 4; without it the cap is 8. This one costs PCC, so it is reported
  with its PCC and taken only if the win is large.

* ``shipped_bf4`` — the weight at BFP4 on whatever config that call site ships. **Measured and
  REJECTED for all three**, despite being the biggest win available here: against the shipped
  config it is ``q_proj`` -43 % (109.7 -> 62.8 us), ``o_proj`` -19 % (111.1 -> 90.4),
  ``kv_proj`` -11 % (53.9 -> 48.2), ~370 us/step in total. Every variant fails the drafter's 0.99
  gate (``tests/test_dflash_drafter_tp.py``, full 5-layer drafter):

      all bf8 (shipped)   0.9942
      o_proj  bf4         0.9899   <- misses by 0.0001
      kv_proj bf4         0.9788
      q_proj  bf4         0.9718
      all three bf4       0.9545

  The pattern across every bf4 experiment in this model is about WHERE a projection sits, not how
  big it is: ``gate``/``up`` into a SwiGLU hold 0.9942 and ship; the two residual writers
  (``o_proj`` 0.9899, ``down_proj`` 0.9891) both land at ~0.989 just under the gate; the
  attention-logit projections ``q``/``kv`` are worst at 0.972-0.979, because their error moves the
  SDPA logits where softmax amplifies it rather than averaging it away; and ``fc`` at 0.9796 is bad
  because all five layers read it.

Every candidate is bracketed by tracy signposts ``<case>__<cand>_start/_stop``, so device time is
read per candidate out of the CSV rather than by counting rows. Run::

    QWEN_DFLASH_MM_SWEEP=1 MESH_DEVICE=N150 python -m tracy -p --op-support-count 100000 -r -v -m \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_dflash_attn_matmul_sweep.py

One device is enough and keeps the CSV small: the drafter is **replicated**, so every device runs
this same matmul and there is no collective in the window.

Then join the capture to the signposts::

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

#: The drafter's own compute config (drafter.py __init__) and the no-fp32-acc variant.
CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True)
CKC_NO_FP32 = tpc.COMPUTE_HIFI2_NO_FP32_ACC

#: (label, M, K, N) — M is the LOGICAL row count, so tile padding matches the real call.
CASES = [
    ("q_proj", 16, 5120, 4096),
    ("kv_proj", 32, 5120, 2048),
    ("o_proj", 16, 4096, 5120),
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


@pytest.mark.skipif(_SKIP, reason="set QWEN_DFLASH_MM_SWEEP=1 to run the drafter attention matmul sweep")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {})}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dflash_attn_matmul_sweep(mesh_device, device_params):
    """Sweep program configs for q_proj / kv_proj / o_proj at the drafter's real shapes."""
    del device_params
    mesh_device.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    mapper = {"mesh_mapper": rep} if rep else {}

    for label, M, K, N in CASES:
        torch.manual_seed(0)
        x_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.05
        w_t = torch.randn(K, N, dtype=torch.bfloat16) * (K**-0.5)
        ref = x_t.float()[0, 0] @ w_t.float()
        wbytes = K * N * 1.0625 / 1e6

        x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
        w = ttnn.from_torch(w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
        logger.info(
            f"--- {label}: M={M}(pad 32) K={K} N={N} | bf8 weight {wbytes:.1f} MB "
            f"| roofline @288GB/s = {wbytes * 1000 / 288:.0f} us"
        )

        # (candidate name, callable -> output tensor). Each runs once for PCC, then ITERS timed.
        cands: list[tuple[str, callable]] = [
            ("auto", lambda: ttnn.linear(x, w, compute_kernel_config=CKC, memory_config=_DRAM)),
        ]
        seen = set()
        for nc in CORE_COUNTS:
            pc = tpc.create_matmul_1d_decode_progcfg(M, K, N, num_cores=nc, grid_w=8)
            grid = (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)
            key = (grid, pc.per_core_N, pc.in0_block_w, pc.out_subblock_h, pc.out_subblock_w)
            if key in seen:  # many core counts collapse to the same config
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
                pc8 = tpc.create_matmul_1d_decode_progcfg(M, K, N, num_cores=nc, fp32_acc=False, grid_w=8)
                cands.append(
                    (
                        f"1d_nc{nc}_nofp32_sub{pc8.out_subblock_h}x{pc8.out_subblock_w}",
                        lambda pc=pc8: ttnn.linear(
                            x, w, compute_kernel_config=CKC_NO_FP32, program_config=pc, memory_config=_DRAM
                        ),
                    )
                )

        # DRAM-sharded: different WEIGHT layout, so a separate upload. The in0 reshard is timed.
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
                        compute_kernel_config=CKC,
                        program_config=pc_ds,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    ),
                )
            )
        except Exception as e:
            logger.warning(f"  dramshard setup UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            w_ds = x_ds = None

        for name, builder in [
            (
                "prefill_mlp",
                lambda: ttnn.linear(
                    x,
                    w,
                    compute_kernel_config=CKC,
                    program_config=tpc.create_prefill_mlp_matmul_program_config(M, K, N, max_cols=8),
                    memory_config=_DRAM,
                ),
            ),
            (
                "kpass1",
                lambda: ttnn.linear(
                    x,
                    w,
                    compute_kernel_config=CKC,
                    program_config=tpc.create_prefill_kpass1_matmul_program_config(M, K, N),
                    memory_config=_DRAM,
                ),
            ),
        ]:
            cands.append((name, builder))

        # WEIGHT DTYPE at the SHIPPED program config -- the axis this file originally left out.
        # gate/up already ship bf4 (see weights.MLP_DTYPE); the question here is whether the
        # attention projections can take it too. Reported with PCC; adopting needs the full ladder.
        w_bf4 = ttnn.from_torch(w_t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
        pc_ship = tpc.create_matmul_1d_decode_progcfg(M, K, N, num_cores=64, grid_w=8) if label != "o_proj" else None
        _bf4_kwargs = dict(compute_kernel_config=CKC, memory_config=_DRAM)
        if pc_ship is not None:
            _bf4_kwargs["program_config"] = pc_ship
        cands.append(("shipped_bf4", lambda: ttnn.linear(x, w_bf4, **_bf4_kwargs)))

        for name, builder in cands:
            try:
                o = builder()
                got = ttnn.to_torch(ttnn.get_device_tensors(o)[0] if _MULTI else o).float()[0, 0]
                pcc = float(comp_pcc(ref, got, 0.0)[1])
                ttnn.deallocate(o)
            except Exception as e:  # an illegal blocking at this shape IS a result
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

        for t in (x, w, w_ds, x_ds, w_bf4):
            if t is not None:
                ttnn.deallocate(t)
