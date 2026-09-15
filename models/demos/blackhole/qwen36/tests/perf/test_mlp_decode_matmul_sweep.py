# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Core-count / fusion sweep for the TP MLP DECODE matmuls, ranked by DEVICE KERNEL DURATION.

WHY THIS EXISTS
---------------
The 1D decode progcfgs in ``model_config._init_tp_config`` all ask for ``num_cores=64`` on Wormhole,
but ``create_matmul_1d_decode_progcfg`` derives ``per_core_N = ceil(n_tiles / num_cores)`` and the op
then only lights up ``ceil(n_tiles / per_core_N)`` cores. At the 27B's TP=8 shapes that means:

    gate/up  n_tiles = 2176/32 = 68   -> per_core_N=2 -> 34 of 64 cores active
    down     n_tiles = 5120/32 = 160  -> per_core_N=3 -> 54 of 64 cores active

2 tiles/core is already the floor for 68 tiles on a 64-core grid (68 tiles cannot be spread 1-per-core
without 68 cores), so the idle cores are structural, not a misconfiguration -- which is why this sweep
also measures the one thing that CAN change the shape: fusing gate and up into a single N=4352 matmul,
so the same bytes are fetched by 46 cores in one pass with one in0 stream instead of two.

Also measures each projection against its weight bytes, because that decides whether tuning is even
worth attempting: at 199 GB/s the down-proj is already at the Wormhole DRAM roofline and only fewer
bytes could help it.

RESULTS (T3K, TP=8, 27B, M=32, device kernel duration; medians over 5 iters, repeated across two
runs to within 0.5us). THREE of the four findings are negative, which is the point of writing them
down -- the decode path is much closer to its floor than prefill was:

    gate  (SILU fused)   34 cores pcN=2   46.3 us     <- shipping config (num_cores=64 -> 34 active)
                         23 cores pcN=3   47.4 us
    up                   34 cores pcN=2   42.0 us     <- shipping config
                         23 cores pcN=3   41.0 us
    down                 54 cores pcN=3   58.9 us     <- shipping config, 11.84MB => 199 GB/s
                         40 cores pcN=4   58.3 us
    gate DRAM-sharded    64 cores         60.7 us     vs 46.3 us for 1D: +31%

  1. Core count is a wash: every legal (num_cores -> per_core_N) combination lands within 2-3% and
     the sign is not even consistent (23 cores wins for `up`, loses for `gate`). num_cores=64 stays.
  2. The DRAM-sharded weight layout the 1D path replaced is 31% SLOWER here, so model_config's 1D
     choice -- measured on the 9B -- holds at the 27B's shapes too. No re-tune needed.
  3. down is AT the DRAM roofline. Only fewer bytes would move it, and its bf8 is a deliberate
     accuracy choice (mlp.py load_mlp_weights: "gate/up bfloat4_b (bandwidth); down bfloat8_b").
  4. The ONE real lever -- fusing gate and up into a single N=2*2176=4352 matmul:

         gate + up separately   46.3 + 42.0 = 88.3 us
         gate_up fused          46 cores pcN=3  63.2 us     -28.4%

     The same weight bytes fetched by 46 cores in one in0 pass instead of 34 cores in two. Net of
     the SILU + split the fused form then needs (~5us: two slices on the 68-tile boundary plus a
     standalone silu, where today SILU is free in the gate matmul's packer) that is ~-18us/layer.
     NOT IMPLEMENTED: it needs a packed [gate|up] decode weight IN ADDITION to w1/w3, because
     prefill cannot share it (a fused prefill matmul gives per_core_N = ceil(136/8) = 17, and 17 is
     prime -> out_subblock_w collapses to 1). That is +6.27MB per layer per device, ~+401MB over 64
     layers, which competes with the KV cache for the same DRAM at long context. Worth doing only
     if that trade is acceptable at the target context length.

Run (T3K, 27B)::

    QWEN_MLP_SWEEP=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      python -m tracy -p -r -o /tmp/mlpdec -m \\
      'pytest models/demos/blackhole/qwen36/tests/perf/test_mlp_decode_matmul_sweep.py'

Candidate order is written to ``$QWEN_MLP_SWEEP_ORDER`` (default /tmp/qwen_mlp_decode_order.json).
Skipped unless ``QWEN_MLP_SWEEP=1``.
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
from models.demos.blackhole.qwen36.tt import tp_common as tpc

TILE = 32
M = 32  # one tile of tokens: every decode progcfg is built at M=1 and is batch-independent
ITERS = 5  # decode matmuls are tens of us, so average more of them


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}.get(
        name, (1, max(1, min(ttnn.get_num_devices(), 2)))
    )


MESH_SHAPE = _mesh_shape()

# The compute config the decode arm uses (mlp.py compute_kernel_config_decode).
CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)


def _bytes_per_elem(dt):
    return {ttnn.bfloat4_b: 0.5625, ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0}[dt]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_mlp_decode_matmul_sweep(mesh_device, device_params):
    """Sweep decode matmul core counts + the gate/up fusion at the real 27B TP shapes."""
    del device_params
    if os.environ.get("QWEN_MLP_SWEEP") != "1":
        pytest.skip("set QWEN_MLP_SWEEP=1 to run the MLP decode matmul sweep")

    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    args = Qwen36ModelArgs(mesh_device, max_batch_size=32, max_seq_len=128)
    tp = mesh_device.get_num_devices()
    hidden_tp = args.hidden_dim // tp
    logger.info(f"dim={args.dim} hidden_dim={args.hidden_dim} tp={tp} hidden/tp={hidden_tp}")

    # (label, K, N, weight dtype, fused activation)
    shapes = [
        ("gate", args.dim, hidden_tp, ttnn.bfloat4_b, ttnn.UnaryOpType.SILU),
        ("up", args.dim, hidden_tp, ttnn.bfloat4_b, None),
        # The fusion candidate: one column-parallel matmul emitting [gate | up] side by side. SILU
        # cannot be fused in the packer here (it would hit the `up` half too), so the caller would
        # slice and silu separately -- costed in the report, not in this matmul.
        ("gate_up_fused", args.dim, 2 * hidden_tp, ttnn.bfloat4_b, None),
        ("down", hidden_tp, args.dim, ttnn.bfloat8_b, None),
    ]
    core_counts = [16, 24, 32, 34, 40, 48, 56, 64]

    mesh_device.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    order = []

    for label, k, n, w_dt, act in shapes:
        torch.manual_seed(0)
        x_t = torch.randn(1, 1, M, k, dtype=torch.bfloat16) * 0.1
        w_t = torch.randn(1, 1, k, n, dtype=torch.bfloat16) * 0.05
        x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        w = ttnn.from_torch(w_t, dtype=w_dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        ref = x_t.to(torch.float32) @ w_t.to(torch.float32)
        if act is not None:
            ref = torch.nn.functional.silu(ref)
        wbytes = k * n * _bytes_per_elem(w_dt) / 1e6
        logger.info(f"--- {label}: {M}x{k}x{n} weight={w_dt} {wbytes:.2f} MB")

        seen = set()
        for nc in core_counts:
            pc = tpc.create_matmul_1d_decode_progcfg(M, k, n, num_cores=nc, fused_activation=act, grid_w=8)
            # Many num_cores values collapse to the same (grid, per_core_N) -- report once.
            grid = (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)
            key = (grid, pc.per_core_N, pc.in0_block_w, pc.out_subblock_w)
            if key in seen:
                logger.info(f"  skip num_cores={nc}: same config as an earlier one {key}")
                continue
            seen.add(key)
            active = min(grid[0] * grid[1], math.ceil(math.ceil(n / TILE) / pc.per_core_N))
            tag = (
                f"{label} num_cores={nc} grid={grid} "
                f"active~{active} pcN={pc.per_core_N} in0_bw={pc.in0_block_w} sub={pc.out_subblock_h}x{pc.out_subblock_w}"
            )
            try:
                out = ttnn.linear(
                    x, w, compute_kernel_config=CKC, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG
                )
            except Exception as e:
                logger.warning(f"SKIP  {tag}: {str(e).splitlines()[0][:150]}")
                continue
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].to(torch.float32)
            _, pcc = comp_pcc(ref, got, 0.0)
            ttnn.deallocate(out)
            for _ in range(ITERS):
                out = ttnn.linear(
                    x, w, compute_kernel_config=CKC, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                ttnn.deallocate(out)
            ttnn.synchronize_device(mesh_device)
            order.append(f"{tag} wbytes_mb={wbytes:.2f}")
            logger.info(f"RUN   [{len(order)-1:2d}] {tag}  pcc={pcc}")
        ttnn.deallocate(x)
        ttnn.deallocate(w)

    # The alternative the 1D path replaced: DRAM-WIDTH_SHARDED weights + the DRAM-sharded matmul.
    # model_config picks 1D because it measured faster on the 9B, and load_mlp_weights follows that
    # choice (the two weight layouts are not interchangeable). Worth re-checking at the 27B's shapes
    # because it is a config flip with no extra weight memory, unlike the gate/up fusion above.
    # gate only: the down-proj is already at the DRAM roofline (11.84MB / 58.4us = 199 GB/s), so no
    # weight-layout change can help it -- only fewer bytes could, and bf8 there is deliberate for
    # accuracy (see mlp.py load_mlp_weights).
    for label, k, n in [("gate_dramshard", args.dim, hidden_tp)]:
        torch.manual_seed(0)
        x_t = torch.randn(1, 1, M, k, dtype=torch.bfloat16) * 0.1
        w_t = torch.randn(1, 1, k, n, dtype=torch.bfloat16) * 0.05
        w_dt = ttnn.bfloat4_b if label.startswith("gate") else ttnn.bfloat8_b
        ref = x_t.to(torch.float32) @ w_t.to(torch.float32)
        act_memcfg = tpc.create_dram_sharded_mem_config(k, n)
        try:
            w = ttnn.from_torch(
                w_t,
                dtype=w_dt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=rep,
                memory_config=act_memcfg,
            )
            x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
            x_sh = ttnn.to_memory_config(x, args.act_shard_hidden if label.startswith("gate") else None)
        except Exception as e:
            logger.warning(f"SKIP {label} setup: {str(e).splitlines()[0][:150]}")
            continue
        pc = tpc.create_dram_sharded_matmul_program_config(M, k, n)
        wbytes = k * n * _bytes_per_elem(w_dt) / 1e6
        tag = f"{label} DRAM_SHARDED in0_bw={pc.in0_block_w} pcN={pc.per_core_N}"
        try:
            out = ttnn.linear(
                x_sh,
                w,
                compute_kernel_config=CKC,
                program_config=pc,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
        except Exception as e:
            logger.warning(f"SKIP  {tag}: {str(e).splitlines()[0][:150]}")
            continue
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].to(torch.float32)
        _, pcc = comp_pcc(ref, got, 0.0)
        ttnn.deallocate(out)
        for _ in range(ITERS):
            out = ttnn.linear(
                x_sh,
                w,
                compute_kernel_config=CKC,
                program_config=pc,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)
        order.append(f"{tag} wbytes_mb={wbytes:.2f}")
        logger.info(f"RUN   [{len(order)-1:2d}] {tag}  pcc={pcc}")

    order_path = os.environ.get("QWEN_MLP_SWEEP_ORDER", "/tmp/qwen_mlp_decode_order.json")
    with open(order_path, "w") as fh:
        json.dump({"proj": "decode", "iters": ITERS, "rows_per_candidate": ITERS + 1, "order": order}, fh, indent=1)
    logger.info(f"dispatch order written to {order_path}")
