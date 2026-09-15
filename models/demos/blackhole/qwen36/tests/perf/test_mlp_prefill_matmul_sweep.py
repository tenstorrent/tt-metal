# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grid / blocking sweep for the TP MLP PREFILL matmuls, ranked by DEVICE KERNEL DURATION.

WHY THIS EXISTS
---------------
``tp_common._best_prefill_cols`` picks the prefill grid width by maximising the output subblock.
That heuristic was tuned on the 9B, whose per-device ``hidden_dim/tp`` is a tile count with many
divisors. The 27B's is 68 tiles (17408/8/32) which is 2*2*17 -- so the heuristic settles on a
**6-wide grid (48 of 64 cores)** for gate/up, and nobody has measured whether trading the wide
subblock for the missing 16 cores is actually the right call at that shape.

This sweep runs the three real MLP prefill matmuls (gate, up, down) at the shapes the model
actually issues, over a grid of (cols, in0_block_w, out_subblock_h) candidates, and reports device
kernel duration + PCC against a torch reference for each.

READ THE CAVEAT BEFORE TRUSTING A NUMBER
----------------------------------------
This is an ISOLATED sweep: it dispatches back-to-back copies of one matmul, so it sees more DRAM
bandwidth than the same matmul does inside a full decoder layer. ``tt/mlp.py`` documents a case
where that made an isolated sweep rank two *compute-kernel-config* candidates backwards. Grid and
blocking are much less sensitive to that than packer_l1_acc was, but the rule stands: use this to
RANK candidates, then confirm the winner in ``test_profile_single_layer_prefill``.

Run (T3K, 27B)::

    QWEN_MLP_SWEEP=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      python -m tracy -p -r -o /tmp/mlpsweep -m \\
      'pytest models/demos/blackhole/qwen36/tests/perf/test_mlp_prefill_matmul_sweep.py'

then rank with the per-op CSV (``DEVICE KERNEL DURATION [ns]``); the test prints the config order
it dispatched so rows map back to candidates. Skipped unless ``QWEN_MLP_SWEEP=1`` so a plain
``pytest tests/`` sweep never runs it.
"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

TILE = 32
SEQ = 2048  # production chunk-outer prefill length
ITERS = 3  # per candidate; take the median row in the CSV

# The compute config the Wormhole one-K-pass prefill arm actually uses (tt/mlp.py _CKC_MLP_KPASS1).
CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True)


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}.get(
        name, (1, max(1, min(ttnn.get_num_devices(), 2)))
    )


MESH_SHAPE = _mesh_shape()


def _subblock_w(per_core_n, sub_h, cap=8):
    """Widest out_subblock_w dividing per_core_n with sub_h*sub_w <= cap (the DST budget)."""
    for w in range(min(per_core_n, cap // sub_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _candidates(m, n, rows=8, max_cols=8):
    """(cols, in0_block_w, sub_h) candidates that are legal for this shape.

    out_block_w is pinned to per_core_N (one K pass) throughout -- the one-K-pass blocking is
    already established as the right regime for these matmuls on Wormhole; what is unswept is the
    grid width and the in0 block depth.
    """
    per_core_M = max(1, math.ceil(m / TILE / rows))
    out = []
    for cols in range(4, max_cols + 1):
        per_core_N = max(1, math.ceil(n / TILE / cols))
        for sub_h in (1, 2):
            if per_core_M % sub_h:
                continue
            sub_w = _subblock_w(per_core_N, sub_h)
            if sub_h > 1 and sub_w == 1:
                continue  # a 1-wide subblock is never the point of raising sub_h
            for in0_bw in (2, 4, 8):
                out.append((cols, in0_bw, sub_h, per_core_M, per_core_N, sub_w))
    return out


def _progcfg(k, cols, in0_bw, sub_h, per_core_M, per_core_N, sub_w, rows=8, act=None):
    k_tiles = math.ceil(k / TILE)
    if k_tiles % in0_bw:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_bw,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=per_core_N,  # one K pass
        transpose_mcast=False,
        fused_activation=act,
        fuse_batch=False,
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("proj", ["gate", "down"])
def test_mlp_prefill_matmul_sweep(mesh_device, device_params, proj):
    """Sweep prefill matmul grid/blocking for one projection at the real 27B TP shapes."""
    del device_params
    if os.environ.get("QWEN_MLP_SWEEP") != "1":
        pytest.skip("set QWEN_MLP_SWEEP=1 to run the MLP prefill matmul sweep")

    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=SEQ)
    tp = mesh_device.get_num_devices()
    hidden_tp = args.hidden_dim // tp

    # gate/up: [S, dim] x [dim, hidden/tp] with SILU fused, bfp4 weight (mlp.py w1/w3).
    # down:    [S, hidden/tp] x [hidden/tp, dim], bf8 activation and bf8 weight (mlp.py w2).
    # in0 is bf8 for BOTH on the 27B: ff_norm narrows its post-norm gather to bf8 (layer.py
    # _ff_gather_dtype), so gate/up receive bf8 and -- since ttnn.linear's output follows in0 --
    # emit bf8, which is what the SwiGLU multiply then hands the down-proj. Sweeping gate at bf16
    # would rank configs at a dtype the 27B no longer runs.
    in0_dt = ttnn.bfloat8_b if args.dim > 4096 else ttnn.bfloat16
    if proj == "gate":
        k, n, w_dt, x_dt, act = args.dim, hidden_tp, ttnn.bfloat4_b, in0_dt, ttnn.UnaryOpType.SILU
    else:
        k, n, w_dt, x_dt, act = hidden_tp, args.dim, ttnn.bfloat8_b, ttnn.bfloat8_b, None
    logger.info(f"{proj}: {SEQ}x{k}x{n}  tp={tp} dim={args.dim} hidden_dim={args.hidden_dim} (hidden/tp={hidden_tp})")

    torch.manual_seed(0)
    x_t = torch.randn(1, 1, SEQ, k, dtype=torch.bfloat16) * 0.1
    w_t = torch.randn(1, 1, k, n, dtype=torch.bfloat16) * 0.05
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    x = ttnn.from_torch(x_t, dtype=x_dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    w = ttnn.from_torch(w_t, dtype=w_dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    ref = x_t.to(torch.float32) @ w_t.to(torch.float32)
    if act is not None:
        ref = torch.nn.functional.silu(ref)

    mesh_device.enable_program_cache()
    cands = _candidates(SEQ, n)
    logger.info(f"{len(cands)} candidates; dispatch order below maps to MatmulDeviceOperation rows in the CSV")

    order = []
    for cols, in0_bw, sub_h, pcM, pcN, sub_w in cands:
        pc = _progcfg(k, cols, in0_bw, sub_h, pcM, pcN, sub_w, act=act)
        if pc is None:
            continue
        tag = f"{proj} cols={cols} cores={cols*8} in0_bw={in0_bw} sub={sub_h}x{sub_w} pcM={pcM} pcN={pcN}"
        try:
            out = ttnn.linear(x, w, compute_kernel_config=CKC, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except Exception as e:  # L1 / CB overflow etc: record and move on
            logger.warning(f"SKIP  {tag}: {str(e).splitlines()[0][:160]}")
            continue
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].to(torch.float32)
        _, pcc = comp_pcc(ref, got, 0.0)
        ttnn.deallocate(out)

        for _ in range(ITERS):  # the rows the profiler ranks; first (compile) call is above
            out = ttnn.linear(x, w, compute_kernel_config=CKC, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)
        order.append(tag)
        logger.info(f"RUN   [{len(order)-1:2d}] {tag}  pcc={pcc}")

    logger.info(f"dispatch order ({len(order)} candidates x {ITERS} profiled iters, after 1 unprofiled warmup each):")
    for i, tag in enumerate(order):
        logger.info(f"  [{i:2d}] {tag}")

    # Machine-readable dispatch order. Candidates that fail program creation (CB overflow at the
    # wide per_core_N values) are dropped mid-sweep, so a reader CANNOT reconstruct the mapping from
    # the candidate list alone -- it has to know which ones actually ran, in order.
    order_path = os.environ.get("QWEN_MLP_SWEEP_ORDER", f"/tmp/qwen_mlp_sweep_order_{proj}.json")
    with open(order_path, "w") as fh:
        json.dump({"proj": proj, "iters": ITERS, "rows_per_candidate": ITERS + 1, "order": order}, fh, indent=1)
    logger.info(f"dispatch order written to {order_path}")
