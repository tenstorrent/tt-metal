# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-config sweep for the DFlash drafter's ``fc`` tap projection.

Third of the drafter's matmul sweeps (``..._attn_...`` for q/kv/o, ``..._mlp_...`` for gate/up/down),
same signpost convention and same report script. ``fc`` is the odd one out in three ways, which is
why it gets its own file rather than a case in either of those:

* It runs **once per step**, not once per layer — so a win here is worth a fifth of the same win in
  a layer, and it is the only matmul whose cost is not multiplied by ``num_hidden_layers``.
* Its K is enormous and its M is tiny: ``M=16 K=25600 N=5120``. The 25,600 is five target taps
  concatenated and all-gathered, so K is ``num_target_layers × hidden_size``, not a model dim.
* Its weight rows are **permuted at load time** to match the gather's device-major column order
  (``weights.reorder_fc_rows``), so the weight is not interchangeable with the checkpoint's.

    bf8 weight   roofline @288 GB/s   measured (auto)   efficiency
    139.3 MB          484 us              669 us            72 %

WHAT IS SWEPT
-------------
``auto`` (today's config), ``1d_nc<N>`` (``create_matmul_1d_decode_progcfg``, explicit wide-first
grid, mcast_in0), ``_l1out``, ``dramshard`` (+ ``dramshard_pre`` for the reshard-free ceiling),
``prefill_mlp``, ``COMPUTE_HIFI2_NO_FP32_ACC``, and a **bf4 weight** arm on the auto config.

RESULT: **auto wins; there is nothing to apply.** Every explicit program config lost, the best by
+1.6 %, and the two families tt-perf-report's own hints point at lost badly:

    auto                     652.6 us    (shipped, unchanged)
    best explicit (1D 8x8)   663.2 us    +1.6 %
    dramshard                799.9 us   +22.6 %   (+ an in0 reshard op)
    prefill_mlp              912.7 us   +39.8 %
    auto + bf4 weight        522.7 us   -19.9 %   <- REJECTED, see below

That makes three call sites swept with the same candidate families and only one — the attention
q/kv projections — where an explicit config beat auto. See ``_layer_mlp``'s docstring for the other
negative result.

**bf4 on ``fc`` is rejected on accuracy, decisively.** It is the same -20 % that bf4 bought in the
MLP, but ``fc`` sits at the *entrance* to the drafter and every layer reads its output, so the error
propagates five times rather than once. MEASURED on ``tests/test_dflash_drafter_tp.py``:

                        fc + hidden_norm   full 5-layer drafter
    fc bf8 (shipped)         0.9999              0.9942
    fc bf4                   0.9932              0.9796   <- 4 of the suite's 5 tests FAIL

Compare the MLP, where bf4 on gate/up cost only 0.9978 -> 0.9942 and held the gate. Same dtype,
same kind of matmul, opposite verdict — the difference is position in the graph, not the op.

Run::

    QWEN_DFLASH_MM_SWEEP=1 MESH_DEVICE=N150 python -m tracy -p --op-support-count 100000 -r -v -m \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_dflash_fc_matmul_sweep.py

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

#: 5 taps x 5120 hidden, gathered, projected back to hidden. M is the newly accepted context rows.
M, K, N = 16, 25600, 5120
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


@pytest.mark.skipif(_SKIP, reason="set QWEN_DFLASH_MM_SWEEP=1 to run the drafter fc matmul sweep")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {})}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dflash_fc_matmul_sweep(mesh_device, device_params):
    """Sweep program configs for the tap projection at the drafter's real shape."""
    del device_params
    mesh_device.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    mapper = {"mesh_mapper": rep} if rep else {}

    torch.manual_seed(0)
    x_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.05
    w_t = torch.randn(K, N, dtype=torch.bfloat16) * (K**-0.5)
    ref = x_t.float()[0, 0] @ w_t.float()
    wbytes = K * N * 1.0625 / 1e6

    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
    w = ttnn.from_torch(w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
    logger.info(
        f"--- fc: M={M}(pad 32) K={K} N={N} | bf8 weight {wbytes:.1f} MB "
        f"| roofline @288GB/s = {wbytes * 1000 / 288:.0f} us"
    )

    extra: list = []
    cands: list[tuple[str, callable]] = [
        ("auto", lambda: ttnn.linear(x, w, compute_kernel_config=CKC, memory_config=_DRAM)),
    ]
    seen = set()
    for nc in CORE_COUNTS:
        pc = tpc.create_matmul_1d_decode_progcfg(M, K, N, num_cores=nc, grid_w=8)
        grid = (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)
        key = (grid, pc.per_core_N, pc.in0_block_w, pc.out_subblock_h, pc.out_subblock_w)
        if key in seen:
            continue
        seen.add(key)
        tag = f"g{grid[0]}x{grid[1]}_pcN{pc.per_core_N}_bw{pc.in0_block_w}_sub{pc.out_subblock_h}x{pc.out_subblock_w}"
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
        extra += [w_ds, x_ds]
    except Exception as e:
        logger.warning(f"  dramshard setup UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:150]}")

    cands.append(
        (
            "prefill_mlp",
            lambda: ttnn.linear(
                x,
                w,
                compute_kernel_config=CKC,
                program_config=tpc.create_prefill_mlp_matmul_program_config(M, K, N, max_cols=8),
                memory_config=_DRAM,
            ),
        )
    )

    # Weight dtype, reported not adopted -- see the module docstring.
    w_bf4 = ttnn.from_torch(w_t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
    extra.append(w_bf4)
    cands.append(("auto_bf4", lambda: ttnn.linear(x, w_bf4, compute_kernel_config=CKC, memory_config=_DRAM)))

    for name, builder in cands:
        try:
            o = builder()
            got = ttnn.to_torch(ttnn.get_device_tensors(o)[0] if _MULTI else o).float()[0, 0]
            pcc = float(comp_pcc(ref, got, 0.0)[1])
            ttnn.deallocate(o)
        except Exception as e:
            logger.warning(f"  {name:40} UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:130]}")
            continue
        if _SP is not None:
            _SP(f"fc__{name}_start")
        for _ in range(ITERS):
            ttnn.deallocate(builder())
        ttnn.synchronize_device(mesh_device)
        if _SP is not None:
            _SP(f"fc__{name}_stop")
        logger.info(f"  {name:40} pcc={pcc:.6f}")

    for t in (x, w, *extra):
        ttnn.deallocate(t)
