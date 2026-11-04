# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-shape comparison: the matmul + reduce-scatter paths models run today vs matmul_reduce_scatter_prefill.

For every case of mrs_model_cases.CASES (one row-parallel linear of a model at its per-device shape) this runs the
model's current path, its other existing paths, and matmul_reduce_scatter_prefill (with the model's fused epilogue as
separate ops), on a Blackhole QuietBox 2 (1x4 ring) or a BH Galaxy (4x8, MRS_CMP_MESH=4x8). Each path:

  - eager call, checked against a host reference (PCC on a sample of row tiles; inputs are random, sharded on K)
  - captured in a trace with everything it dispatches (matmul, reduce-scatter, bias / addcmul / concat epilogue)
  - device time: real-time profiler over REPLAYS back-to-back replays; per replay and chip, the span from the first
    program's start to the last program's end (dispatch gaps between programs included), slowest chip, median
  - wall time: WALL_REPLAYS replays timed on the host, per replay

One JSON line per (case, path) goes to $MRS_CMP_OUT (default generated/mrs_model_compare.jsonl). A path that fails
(refusal, L1 clash, wrong output) is recorded with its error and the next path still runs; the test fails only when
ours fails.

  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/model_compare/ -m "not large"

Env: MRS_CMP_MESH mesh to open (default 1x4; 4x8 on a Galaxy), MRS_CMP_PAYLOAD fabric router payload bytes (default 8192, the tt_dit ring presets), MRS_CMP_PATHS comma list to
restrict paths (e.g. ours,prod).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import traceback

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.model_compare.mrs_model_cases import CASES, Case

TILE = 32
# The mesh the test opens: 1x4 (QuietBox 2, groups on axis 1) or 4x8 (BH Galaxy, TP4 groups on axis 0 and TP8 groups on
# axis 1, as the tt_dit Galaxy presets). Each case runs on the axis whose size is its tp, replicated over the other.
MESH_SHAPE = tuple(int(v) for v in os.environ.get("MRS_CMP_MESH", "1x4").lower().split("x"))


class G:
    """The group of the running case: its size and mesh axis (set by the test before anything is built)."""

    tp = 4
    axis = 1


NUM_LINKS = 2
REPLAYS = 10
WALL_REPLAYS = 10
SAMPLE_ROW_TILES = 8  # reference rows: this many row tiles spread over M (all rows when M is smaller)
TRACE_REGION_BYTES = 64 << 20
PAYLOAD = int(os.environ.get("MRS_CMP_PAYLOAD", "8192"))
OUT_PATH = os.environ.get("MRS_CMP_OUT", "generated/mrs_model_compare.jsonl")
ONLY_PATHS = {p for p in os.environ.get("MRS_CMP_PATHS", "").split(",") if p}

DTYPES = {"bf16": ttnn.bfloat16, "bf8b": ttnn.bfloat8_b, "bf4b": ttnn.bfloat4_b, "fp32": ttnn.float32}
FIDELITY = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
# PCC floors against the host reference of the device-quantized operands (the math fidelity is the main error)
PCC_FLOOR = {"LoFi": 0.98, "HiFi2": 0.995, "HiFi3": 0.997, "HiFi4": 0.998}

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
# The production BH Galaxy grids of the tt_dit paths: the fused op sees the 12x10 device grid, the unfused matmul
# the 11x10 power clamp (models/tt_dit/utils/matmul.py:485). The QuietBox's grid is logged.
DIT_DEVICE_GRID = ttnn.CoreCoord(12, 10)
DIT_MM_GRID = ttnn.CoreCoord(11, 10)


def _router_config():
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = PAYLOAD
    return config


DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": _router_config(),
    "require_exact_physical_num_devices": True,
    "trace_region_size": TRACE_REGION_BYTES,
}


# ---------------------------------------------------------------------------------------------------------------------
# Inputs and the host reference
# ---------------------------------------------------------------------------------------------------------------------


def quantized(t, dtype):
    """t as the device stores it in dtype, so the reference measures the op and not the block-float rounding."""
    if dtype in (ttnn.bfloat16, ttnn.float32):
        return t.to(torch.bfloat16).float() if dtype == ttnn.bfloat16 else t.float()
    flat = t.reshape(-1, t.shape[-1])
    return ttnn.to_torch(ttnn.from_torch(flat, dtype=dtype, layout=ttnn.TILE_LAYOUT)).reshape(t.shape).float()


def sample_rows(M):
    tiles = math.ceil(M / TILE)
    if tiles <= SAMPLE_ROW_TILES:
        return torch.arange(M)
    picks = sorted({round(i * (tiles - 1) / (SAMPLE_ROW_TILES - 1)) for i in range(SAMPLE_ROW_TILES)})
    rows = torch.cat([torch.arange(p * TILE, min((p + 1) * TILE, M)) for p in picks])
    return rows


def group_mapper(mesh_device, dim):
    """Shard dim over the group axis, replicate over the other mesh axis."""
    dims = (None, dim) if G.axis == 1 else (dim, None)
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)


def group_index(device_index, mesh_device):
    """The member index within its group of the device at row-major position device_index."""
    rows, cols = tuple(mesh_device.shape)
    r, c = divmod(device_index, cols)
    return c if G.axis == 1 else r


class Inputs:
    """Host tensors of a case (full, i.e. over the group) and their device copies."""

    def __init__(self, mesh_device, case: Case):
        torch.manual_seed(0)
        M, N = case.M, case.N
        self.case = case
        self.mesh_device = mesh_device
        act_dtype = DTYPES[case.act]
        w_dtype = DTYPES[case.weight]
        splits = case.k_split or (case.K,)
        # A per segment, full over the group: [1, 1, M, G.tp * k] sharded on dim 3; device g holds columns g*k:(g+1)*k
        self.a_full = [quantized(torch.randn(1, 1, M, G.tp * k), act_dtype) for k in splits]
        # W per device g: rows of segment s are W_s[g]; stacked [G.tp * K, N], sharded on dim 0
        w_seg = [quantized(torch.randn(G.tp, k, N) * (G.tp * case.K) ** -0.5, w_dtype) for k in splits]
        self.w_stacked = torch.cat(w_seg, dim=1).reshape(G.tp * case.K, N)
        self.w_seg = w_seg
        self.bias = quantized(torch.randn(1, N) * 0.1, ttnn.bfloat16) if case.bias else None
        self.dev_a = self.dev_b = None
        if case.addcmul:
            self.add_a = quantized(torch.randn(1, 1, M, N), ttnn.bfloat16)
            gate_rows = M if case.addcmul == "token" else 1
            self.add_b = quantized(torch.randn(1, 1, gate_rows, N), ttnn.bfloat16)

        shard3 = group_mapper(mesh_device, 3)
        shard0 = group_mapper(mesh_device, 0)

        def dev(t, dtype, mapper, mem=DRAM):
            return ttnn.from_torch(
                t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mem, mesh_mapper=mapper
            )

        self.dev = dev
        self.x = [dev(a, act_dtype, shard3) for a in self.a_full]
        self.w = dev(self.w_stacked, w_dtype, shard0)
        if case.bias:
            # pre-RS bias (RowParallelLinear / gpt_oss): device 0 holds the bias, the others zeros, [1, N] each
            pre = torch.zeros(G.tp, N)
            pre[0] = self.bias[0]
            self.bias_pre = dev(pre, ttnn.bfloat16, shard0)
            # post-RS bias for ours: device g holds bias[g*N/G.tp:(g+1)*N/G.tp]
            self.bias_post = dev(self.bias, ttnn.bfloat16, group_mapper(mesh_device, 1))
        if case.addcmul:
            self.dev_a = dev(self.add_a, ttnn.bfloat16, shard3)
            self.dev_b = dev(self.add_b, ttnn.bfloat16, shard3)

    def reference(self, rows):
        """Expected output rows for each device: [(device g, tensor [len(rows), N/G.tp])]."""
        case = self.case
        acc = torch.zeros(len(rows), case.N)
        for g in range(G.tp):
            for s, k in enumerate(case.k_split or (case.K,)):
                a_g = self.a_full[s][0, 0, rows, g * k : (g + 1) * k]
                acc += a_g @ self.w_seg[s][g]
        if self.bias is not None:
            acc += self.bias
        n = case.N // G.tp
        out = []
        for g in range(G.tp):
            r = acc[:, g * n : (g + 1) * n]
            if case.addcmul:
                gate = self.add_b[0, 0, rows if case.addcmul == "token" else slice(0, 1), g * n : (g + 1) * n]
                r = self.add_a[0, 0, rows, g * n : (g + 1) * n] + r * gate
            out.append(r)
        return out


def check_output(out, inputs: Inputs, case: Case):
    rows = sample_rows(case.M)
    expected = inputs.reference(rows)
    pccs = []
    for d, t in enumerate(ttnn.get_device_tensors(out)):
        g = group_index(d, inputs.mesh_device)
        got = ttnn.to_torch(t).float().reshape(-1, t.shape[-1])
        assert tuple(got.shape) == (case.M, case.N // G.tp), f"device {d}: shape {tuple(got.shape)}"
        got = got[rows]
        pcc = torch.corrcoef(torch.stack([expected[g].flatten(), got.flatten()]))[0, 1].item()
        pccs.append(float(pcc))
    return min(pccs)


# ---------------------------------------------------------------------------------------------------------------------
# Paths: each returns (run, info); run() dispatches the whole path and returns the scattered output
# ---------------------------------------------------------------------------------------------------------------------


def wh_ckc(mesh_device, case: Case):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=FIDELITY[case.fidelity],
        math_approx_mode=case.approx,
        fp32_dest_acc_en=case.fp32_acc,
        packer_l1_acc=True,
    )


class Env:
    """Per-test shared state: the tt_dit CCLManager supplies semaphores and persistent buffers to every path."""

    def __init__(self, mesh_device, case):
        from models.tt_dit.parallel.manager import CCLManager

        topo = ttnn.Topology.Ring if case.topology == "ring" else ttnn.Topology.Linear
        self.topology = topo
        self.ccl = CCLManager(mesh_device, num_links=NUM_LINKS, topology=topo)
        self.mesh_device = mesh_device


def rs_minimal(env: Env, x, *, hyper=None, persistent=False):
    """CCLManager.reduce_scatter (dim 3 over the group), optionally with the tt_dit hyperparameters."""
    if hyper is None:
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=env.ccl.get_rs_ping_pong_semaphore(G.axis),
            barrier_semaphore=env.ccl.get_barrier_semaphore(G.axis),
            num_links=NUM_LINKS,
            memory_config=DRAM,
            topology=env.topology,
            cluster_axis=G.axis,
        )
    return env.ccl.reduce_scatter(x, dim=3, mesh_axis=G.axis, use_persistent_buffer=persistent)


def path_mmrs_dit(env: Env, inp: Inputs, case: Case):
    from models.tt_dit.utils.matmul import get_fused_mmrs_config

    if case.model == "MiniMax-H3":
        from models.tt_dit.models.transformers.minimax_h3.mmrs_config import has_mmrs_config, register_mmrs_config

        assert has_mmrs_config(case.M, case.K, case.N, DIT_DEVICE_GRID), "H3 gates this shape off the fused path"
        register_mmrs_config(case.M, case.K, case.N, DIT_DEVICE_GRID)
    from models.tt_dit.utils.matmul import _resolve_fused_mmrs_config, default_fused_mmrs_config

    params = get_fused_mmrs_config(case.M, case.K, case.N, DIT_DEVICE_GRID, NUM_LINKS)
    adapted = None
    dev_grid = env.mesh_device.compute_with_storage_grid_size()
    resolved = _resolve_fused_mmrs_config(case.M, case.K, case.N, DIT_DEVICE_GRID, log=False)
    source = "table/rules" if resolved is not None else "default (warned)"
    resolved = resolved or default_fused_mmrs_config
    if resolved.compute_with_storage_grid_size.x > dev_grid.x:
        # The production config is resolved for the BH Galaxy 12x10 grid; a QuietBox 2 has 11 worker columns.
        # Keep the blocking, the window and the RS worker count; narrow only the matmul grid to the device width.
        mm_y = resolved.compute_with_storage_grid_size.y
        # An explicit RS worker count sized for the 12-wide RS zone may not fit the narrower one: cap it at the count
        # get_params derives for this zone (the zone's cores minus the muxes).
        zone_workers = (dev_grid.y - mm_y) * dev_grid.x // (2 * NUM_LINKS) - 1
        workers = resolved.num_workers_per_link
        capped = workers is not None and workers > zone_workers
        narrowed = resolved._replace(
            compute_with_storage_grid_size=ttnn.CoreCoord(dev_grid.x, mm_y),
            num_workers_per_link=zone_workers if capped else workers,
        )
        params = narrowed.get_params(ttnn.CoreCoord(dev_grid.x, dev_grid.y), NUM_LINKS, M=case.M)
        adapted = f"matmul grid {resolved.compute_with_storage_grid_size.x} -> {dev_grid.x} columns" + (
            f", RS workers/link {workers} -> {zone_workers}" if capped else ""
        )
    use_l1 = params["mm_window_blocks"] is not None
    ckc = wh_ckc(env.mesh_device, case)
    pre_rs_shape = (1, 1, case.M, case.N)
    rs_out_buf = env.ccl.get_rs_ping_pong_buffer(pre_rs_shape, 3, G.axis, return_intermediate=False)[1]
    x = inp.x if case.k_split else inp.x[0]
    info = {
        "config": repr(params["config"]),
        "mm_window_blocks": params["mm_window_blocks"],
        "num_workers_per_link": params["num_workers_per_link"],
        "rs_offset": params["reduce_scatter_core_grid_offset"].y,
        "source": source,
        "adapted": adapted,
    }

    def run():
        _, out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            input_tensor=x,
            weight_tensor=inp.w,
            dim=3,
            multi_device_global_semaphore=env.ccl.get_rs_ping_pong_semaphore(G.axis),
            **params,
            bias=inp.bias_pre if case.bias else None,
            memory_config_mm=L1 if use_l1 else DRAM,
            rs_intermediate_mem_config=DRAM,
            rs_output_mem_config=DRAM,
            topology=env.topology,
            cluster_axis=G.axis,
            compute_kernel_config=ckc,
            barrier_semaphore=None,
            using_persistent_buffers=True,
            optional_rs_output_tensor=rs_out_buf,
            fused_ternary_scalar=1.0,
            addcmul_input_tensor1=inp.dev_a,
            addcmul_input_tensor2=inp.dev_b,
            dtype=None,
            mm_progress_counters=env.ccl.get_mm_progress_counters_buffer(),
            mm_credit_counters=env.ccl.get_mm_credit_counters_buffer() if use_l1 else None,
        )
        return out

    return run, info


def path_unfused_dit(env: Env, inp: Inputs, case: Case):
    from models.tt_dit.utils.matmul import get_1d_matmul_config, get_matmul_config

    ckc = wh_ckc(env.mesh_device, case)
    bias = inp.bias_pre if case.bias else None
    if case.M <= 128 and not case.k_split:
        pc = get_1d_matmul_config(case.M, case.K, case.N, DIT_MM_GRID)
        info = {"matmul": "ttnn.linear 1D", "config": repr(pc)}

        def mm():
            return ttnn.linear(inp.x[0], inp.w, bias=bias, program_config=pc, compute_kernel_config=ckc)

    else:
        cfg = get_matmul_config(case.M, case.K, case.N, DIT_MM_GRID, None)
        info = {"matmul": "minimal_matmul", "config": repr(cfg)}
        x = list(inp.x) if case.k_split else inp.x[0]

        def mm():
            return ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=inp.w, bias_tensor=bias, config=cfg, compute_kernel_config=ckc
            )

    def run():
        out = rs_minimal(env, mm(), hyper=True, persistent=True)
        return ttnn.addcmul(inp.dev_a, out, inp.dev_b) if case.addcmul else out

    return run, info


def path_unfused_gptoss(env: Env, inp: Inputs, case: Case):
    # apply_output_projection: ttnn.matmul(bf8b x, o_proj, dtype=bf16), bias added in place; then the RS half of
    # mesh_config.allreduce (reduce_scatter_minimal_async, barrier, no persistent buffers).
    def run():
        out = ttnn.matmul(inp.x[0], inp.w, dtype=DTYPES[case.out_dtype])
        ttnn.add(out, inp.bias_pre, output_tensor=out)
        return rs_minimal(env, out)

    return run, {"matmul": "ttnn.matmul default"}


def path_unfused_m3(env: Env, inp: Inputs, case: Case):
    ckc = None if case.act == "bf8b" else wh_ckc(env.mesh_device, case)

    def run():
        out = ttnn.matmul(inp.x[0], inp.w, dtype=DTYPES[case.out_dtype], compute_kernel_config=ckc)
        return rs_minimal(env, out)

    return run, {"matmul": "ttnn.matmul default" if ckc is None else "ttnn.matmul HiFi3"}


def _muse_ckc(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def path_unfused_muse(env: Env, inp: Inputs, case: Case):
    # OptimizedDecoder._linear prefill ("prefill_kernel": "minimal") + MultichipDecoder._reduce (typecast to the ccl
    # dtype, reduce_scatter_minimal_async with barrier, no persistent buffers in prefill).
    m_tiles = case.M // TILE
    bm = min(4, m_tiles)
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=bm,
        K_block_size=4,  # MultichipDecoder.POLICY minimal_k
        N_block_size=32,  # minimal_n
        subblock_h=2 if bm % 2 == 0 else 1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
    )
    ckc = _muse_ckc(env.mesh_device)
    out_dtype, ccl_dtype = DTYPES[case.out_dtype], DTYPES[case.ccl_dtype]

    def run():
        out = ttnn.experimental.minimal_matmul(
            inp.x[0], inp.w, config=cfg, compute_kernel_config=ckc, memory_config=DRAM, dtype=out_dtype
        )
        if out_dtype != ccl_dtype:
            out = ttnn.typecast(out, ccl_dtype)
        return rs_minimal(env, out)

    return run, {"matmul": "minimal_matmul", "config": repr(cfg)}


class NotApplicable(Exception):
    """The path cannot express this case on this mesh (recorded as n/a, not as an error)."""


def require_line_mesh(env: Env):
    # matmul_reduce_scatter_async has no cluster_axis: it reduces over the whole mesh, so it needs a 1 x N / N x 1 mesh
    if 1 not in tuple(env.mesh_device.shape):
        raise NotApplicable("matmul_reduce_scatter_async has no cluster_axis; needs a 1-D mesh")


def path_mmrs_gptoss(env: Env, inp: Inputs, case: Case):
    # gpt_oss apply_output_projection_fused_rs (operations.py:141): _FUSED_MM_RS_CONFIGS[32] = (grid_y 4, M4, K4, N6,
    # chunk 2, subblock 2x2, 3 workers). Production runs it on Wormhole only; Blackhole gates it off (#46181).
    grid_y, m_block, k_block, n_block, chunk_width, sub_h, sub_w, workers = (4, 4, 4, 6, 2, 2, 2, 3)
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=k_block,
        N_block_size=n_block,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(8, grid_y),
    )
    ckc = wh_ckc(env.mesh_device, case)

    def run():
        _, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            inp.x[0],
            inp.w,
            3,
            env.ccl.get_rs_ping_pong_semaphore(G.axis),
            ttnn.CoreCoord(0, grid_y),
            compute_kernel_config=ckc,
            num_links=NUM_LINKS,
            memory_config_mm=DRAM,
            rs_output_mem_config=DRAM,
            topology=env.topology,
            cluster_axis=G.axis,
            bias=inp.bias_pre,
            config=cfg,
            barrier_semaphore=env.ccl.get_barrier_semaphore(G.axis),
            chunk_width_in_mm_blocks=chunk_width,
            num_workers_per_link=workers,
        )
        return rs_out

    return run, {"config": repr(cfg), "note": "Wormhole production config; gated off on Blackhole (#46181)"}


def path_mrs_async_muse(env: Env, inp: Inputs, case: Case):
    require_line_mesh(env)
    # MultichipDecoder._linear "fused_reduce" (multichip_decoder.py:494)
    m = case.M // TILE
    gy = min(m, 4)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, gy),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=math.ceil(m / gy),
        per_core_N=26,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    rep = ttnn.ReplicateTensorToMesh(env.mesh_device)
    interm = inp.dev(torch.zeros(1, 1, case.M, case.N), ttnn.bfloat16, rep)
    outbuf = inp.dev(torch.zeros(1, 1, case.M, case.N // G.tp), ttnn.bfloat16, rep)
    ckc = _muse_ckc(env.mesh_device)

    def run():
        _, y = ttnn.experimental.matmul_reduce_scatter_async(
            inp.x[0],
            inp.w,
            persistent_intermediate_buffer=interm,
            persistent_output_buffer=outbuf,
            dim=3,
            multi_device_global_semaphore=env.ccl.get_rs_ping_pong_semaphore(G.axis),
            reduce_scatter_core_grid_offset=(0, 6),
            num_links=NUM_LINKS,
            memory_config_rs=DRAM,
            memory_config_mm=DRAM,
            topology=env.topology,
            barrier_semaphore=env.ccl.get_barrier_semaphore(G.axis),
            program_config=pc,
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat16,
        )
        return y

    return run, {"program_config": repr(pc)}


def path_mrs_async_qwen36(env: Env, inp: Inputs, case: Case):
    require_line_mesh(env)
    # tp_common.matmul_reduce_scatter_prefill (the Python helper, tp_common.py:891)
    grid = (8, 8)
    m_tiles = math.ceil(case.M / TILE)
    k_tiles = case.K // TILE
    per_core_N = max(1, math.ceil(case.N / TILE / grid[0]))
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))})
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=min(4, max(1, k_tiles // grid[0])),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(m_tiles / grid[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=crs,
    )
    rep = ttnn.ReplicateTensorToMesh(env.mesh_device)
    act = DTYPES[case.act]
    interm = inp.dev(torch.zeros(1, 1, case.M, case.N), act, rep)
    outbuf = inp.dev(torch.zeros(1, 1, case.M, case.N // G.tp), act, rep)
    ckc = wh_ckc(env.mesh_device, case)

    def run():
        _, rs = ttnn.experimental.matmul_reduce_scatter_async(
            inp.x[0],
            inp.w,
            persistent_intermediate_buffer=interm,
            persistent_output_buffer=outbuf,
            dim=3,
            multi_device_global_semaphore=env.ccl.get_rs_ping_pong_semaphore(G.axis),
            reduce_scatter_core_grid_offset=(0, 8),
            barrier_semaphore=env.ccl.get_barrier_semaphore(G.axis),
            num_links=NUM_LINKS,
            memory_config_rs=DRAM,
            topology=env.topology,
            subdevice_id=None,
            memory_config_mm=DRAM,
            program_config=pc,
            compute_kernel_config=ckc,
        )
        return rs

    return run, {"program_config": repr(pc)}


def path_ours(env: Env, inp: Inputs, case: Case):
    ckc = ttnn.ComputeConfigDescriptor(math_fidelity=FIDELITY[case.fidelity], math_approx_mode=case.approx)
    x_in = inp.x
    info = {"epilogue": []}
    if case.act == "fp32":
        # the op takes bf16 / bf8b activations: hand it the activation in bf16 (outside the measured trace)
        x_in = [ttnn.typecast(t, ttnn.bfloat16) for t in inp.x]
        info["epilogue"].append("input cast fp32->bf16 (not timed)")
    if case.k_split:
        info["epilogue"].append("concat over K")
    if case.bias:
        info["epilogue"].append("bias add")
    if case.addcmul:
        info["epilogue"].append("addcmul")

    def run():
        x = ttnn.concat(x_in, dim=3) if case.k_split else x_in[0]
        out = ttnn.experimental.matmul_reduce_scatter_prefill(
            x,
            inp.w,
            cluster_axis=G.axis,
            scatter_dim=-1,
            topology=env.topology,
            num_links=NUM_LINKS,
            compute_kernel_config=ckc,
        )
        if case.bias:
            out = ttnn.add(out, inp.bias_post)
        if case.addcmul:
            out = ttnn.addcmul(inp.dev_a, out, inp.dev_b)
        return out

    return run, info


PATHS = {
    "mmrs_dit": path_mmrs_dit,
    "unfused_dit": path_unfused_dit,
    "unfused_gptoss": path_unfused_gptoss,
    "mmrs_gptoss": path_mmrs_gptoss,
    "unfused_m3": path_unfused_m3,
    "unfused_muse": path_unfused_muse,
    "mrs_async_muse": path_mrs_async_muse,
    "mrs_async_qwen36": path_mrs_async_qwen36,
    "ours": path_ours,
}


# ---------------------------------------------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------------------------------------------


def collect_window(mesh_device, run_fn, settle_s=0.15, timeout_s=3.0):
    """Run run_fn under the real-time profiler; every record as (chip, start, end, frequency)."""
    records, dropped = [], [0]

    def cb(batch):
        dropped[0] += int(batch.dropped)
        for r in batch.records:
            if float(r.frequency) > 0 and int(r.end_timestamp) > int(r.start_timestamp):
                records.append((int(r.chip_id), int(r.start_timestamp), int(r.end_timestamp), float(r.frequency)))

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(cb)
    try:
        run_fn()
        ttnn.synchronize_device(mesh_device)
        deadline, last, last_change = time.monotonic() + timeout_s, -1, time.monotonic()
        while time.monotonic() < deadline:
            if len(records) != last:
                last, last_change = len(records), time.monotonic()
            elif records and time.monotonic() - last_change >= settle_s:
                break
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    if dropped[0]:
        raise RuntimeError(f"real-time profiler dropped {dropped[0]} records")
    return records


def per_chip(records):
    chips = {}
    for chip, start, end, freq in records:
        chips.setdefault(chip, []).append((start, end, freq))
    for v in chips.values():
        v.sort()
    return chips


def measure_trace(mesh_device, trace_id):
    def replay():
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

    # programs per replay: one replay in its own window
    one = per_chip(collect_window(mesh_device, replay))
    num_chips = mesh_device.get_num_devices()
    assert len(one) == num_chips, f"records from {len(one)} chips, expected {num_chips}"
    programs = {len(v) for v in one.values()}
    assert len(programs) == 1, f"chips disagree on programs per replay: { {c: len(v) for c, v in one.items()} }"
    P = programs.pop()
    time.sleep(0.3)

    def replays():
        for _ in range(REPLAYS):
            replay()

    chips = per_chip(collect_window(mesh_device, replays))
    spans, sums = [], []
    for i in range(REPLAYS):
        span_i, sum_i = 0.0, 0.0
        for v in chips.values():
            assert len(v) >= REPLAYS * P, f"chip has {len(v)} records, expected {REPLAYS * P}"
            group = v[len(v) - REPLAYS * P + i * P :][:P]
            freq = group[0][2]
            span_i = max(span_i, (group[-1][1] - group[0][0]) / freq)
            sum_i = max(sum_i, sum((e - s) / f for s, e, f in group))
        spans.append(span_i)
        sums.append(sum_i)

    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(WALL_REPLAYS):
        replay()
    ttnn.synchronize_device(mesh_device)
    wall = (time.perf_counter() - t0) / WALL_REPLAYS * 1e9
    return {
        "programs": P,
        "device_ns": statistics.median(spans),
        "device_ns_min": min(spans),
        "device_ns_max": max(spans),
        "kernel_sum_ns": statistics.median(sums),
        "wall_ns": wall,
    }


def run_path(mesh_device, env, inputs, case, path_name):
    rec = {"case": case.name, "model": case.model, "path": path_name, "M": case.M, "K": case.K, "N": case.N}
    rec.update(mesh="x".join(map(str, MESH_SHAPE)), tp=G.tp, axis=G.axis, payload=PAYLOAD)
    rec["role"] = "ours" if path_name == "ours" else ("prod" if path_name == case.prod else "alt")
    trace_id, capturing, keep = None, False, []
    try:
        run, info = PATHS[path_name](env, inputs, case)
        rec["info"] = info
        eager = run()
        ttnn.synchronize_device(mesh_device)
        rec["pcc"] = check_output(eager, inputs, case)
        if path_name == "ours":
            try:
                plan = ttnn._ttnn.operations.experimental.ccl_experimental._matmul_reduce_scatter_prefill_last_plan()
                rec["plan"] = {str(k): (list(v) if isinstance(v, tuple) else v) for k, v in dict(plan).items()}
            except Exception as e:  # a debugging aid only
                rec["plan"] = f"unavailable: {e}"
        # Not deallocated: on paths with persistent output buffers the eager output IS the buffer the traced call
        # writes. It goes when the path's state does.
        keep.append(eager)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        capturing = True
        traced = run()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        capturing = False
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)  # first-replay jitter
        ttnn.synchronize_device(mesh_device)
        time.sleep(0.3)  # profiler records arrive asynchronously: let the warm-up's land before any window opens
        rec.update(measure_trace(mesh_device, trace_id))
        rec["pcc_traced"] = check_output(traced, inputs, case)
        floor = PCC_FLOOR[case.fidelity]
        rec["pcc_ok"] = rec["pcc"] >= floor and rec["pcc_traced"] >= floor
        rec["status"] = "ok" if rec["pcc_ok"] else "wrong_output"
    except NotApplicable as e:
        rec["status"] = "n/a"
        rec["error"] = str(e)
    except Exception as e:  # a refusal / L1 clash of one path must not hide the others
        rec["status"] = "error"
        logger.error(f"{case.name} {path_name}:\n{traceback.format_exc()}")
        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:400] if str(e) else ''}"
        if capturing:  # close the capture, or every later path fails on is_capturing_trace
            try:
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            except Exception:
                logger.error(f"could not end the trace capture:\n{traceback.format_exc()}")
    finally:
        if trace_id is not None:
            try:
                ttnn.release_trace(mesh_device, trace_id)
            except Exception:
                logger.error(f"could not release the trace:\n{traceback.format_exc()}")
        ttnn.synchronize_device(mesh_device)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    dev = rec.get("device_ns")
    logger.info(
        f"MRS-CMP {case.name} {path_name:17s} {rec['status']:12s} "
        f"device {dev / 1e3 if dev else float('nan'):9.1f} us  wall {rec.get('wall_ns', float('nan')) / 1e3:9.1f} us "
        f"programs {rec.get('programs', '-')}  pcc {rec.get('pcc', float('nan')):.5f} {rec.get('error', '')}"
    )
    return rec


def group_axis(tp):
    """The mesh axis of a group of tp devices, or None when no axis has that size (prefer axis 0 on a Galaxy)."""
    for axis in (0, 1):
        if MESH_SHAPE[axis] == tp:
            return axis
    return None


def _case_params():
    out = []
    for c in CASES:
        marks = [pytest.mark.large] if c.large else []
        if group_axis(c.tp) is None:
            marks.append(pytest.mark.skip(reason=f"needs a group of {c.tp}; the mesh is {MESH_SHAPE}"))
        out.append(pytest.param(c, id=c.name, marks=marks))
    return out


@pytest.mark.requires_host_iommu
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("case", _case_params())
def test_model_compare(mesh_device, case: Case):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("the real-time profiler must be active")
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(f"compute grid {grid.x}x{grid.y} (tt_dit configs are resolved for the BH Galaxy 12x10)")
    G.tp, G.axis = case.tp, group_axis(case.tp)
    inputs = Inputs(mesh_device, case)
    paths = [case.prod, *case.alts, "ours"]
    if ONLY_PATHS:
        paths = [p for p in paths if p in ONLY_PATHS or ("prod" in ONLY_PATHS and p == case.prod)]
    # each path gets its own CCLManager: its buffer cache is keyed by shape only, and the fused path's (output-only)
    # ping-pong entry would otherwise be handed to the unfused reduce-scatter
    results = {p: run_path(mesh_device, Env(mesh_device, case), inputs, case, p) for p in paths}
    ours = results.get("ours")
    if ours is not None:
        assert ours["status"] == "ok", f"ours: {ours['status']} {ours.get('error', '')} pcc {ours.get('pcc')}"
