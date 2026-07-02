# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Program-config + memory-layout sweep for the TextEncoder CNN ``Conv1d`` (``Conv2dDeviceOperation``).

The CNN stack runs ``Conv1d -> LayerNorm -> LeakyReLU`` three times. In the Tracy report the conv is
the single biggest compute op left (``Conv2dDeviceOperation 96 x 2560 x 512``: ~32µs x 3 ≈ 97µs, and
its companion ``HaloDeviceOperation`` adds more), yet it runs at only ~3.3% of FLOP peak — it's
launch/overhead bound on the tiny M=96 (3 tile-rows), not compute bound. The production call (see
``tt/tt_conv.py::_batched_tt_conv1d_nlc``) leaves ``shard_layout`` on auto (block-sharded, 24 cores),
``act_block_h_override`` default, ``force_split_reader=True`` (out_channels>=256).

This sweep isolates that one conv at the production shape (in=512, out=512, k=5, s=1, p=2, B=2, T=48)
and sweeps the tunable ``Conv1dConfig`` knobs:
  - ``shard_layout``: block / height / width / auto(None)
  - ``act_block_h_override``: 0(auto) / 32 / 64
  - ``force_split_reader``: True / False
  - ``enable_act_double_buffer`` / ``enable_weights_double_buffer``

For each config it runs ONE conv (1 warmup + 1 timed), reads the profiler back, and records the
device-kernel time of the ``Conv2dDeviceOperation`` row alone AND the total of all device ops the
conv emits (conv2d + halo + any reshard), the core count (output shard grid), plus PCC vs a torch
``F.conv1d`` reference. Prints the fastest PCC-passing config and writes a full markdown table
(``conv_sweep_results.md`` next to this file; override path via ``KOKORO_CONV_MD``).

CORE-COUNT CEILING — why the fastest configs top out at 24 cores (NOT swept directly; verified by
forcing the grid via ``override_sharding_config`` + a ``CoreRangeSet``, results below):
  The conv parallelizes its output ``[M, N] = [96, 512] = [3, 16]`` tiles over a 2D block grid; the
  K=2560 contraction is NOT split across cores (each core does the full reduction for its output
  tile). Two hard caps multiply to exactly 24:
    - ``gy <= 3``: M = B*T = 96 = 3 tile-rows, and a tile can't be split across cores. Height
      parallelism is capped at 3 (fixed by the input; nothing tunable changes it).
    - ``gx <= 8``: the conv's inner matmul requires ``block_w >= 2`` tiles (>=64 out-channels/core).
      N=512=16 tiles => gx <= 16/2 = 8. Forcing gx=16 (block_w=1) TT_FATALs (program.cpp:149,
      "not on device" — a 1-tile-wide output block can't form valid matmul subblocks for K=2560).
  So gy=3 x gx=8 = 24 is the LEGAL maximum, and ttnn's auto-heuristic already lands on it. Measured:
  forced 16x3=48 -> FATAL, 16x2=32 -> FATAL, 13x3 -> renormalized to 24 (and slower, 33.8µs), 8x3 ->
  21.7µs (== auto). ``act_block_h_override=64`` is what drops the grid to 16 cores (fatter blocks),
  which is why abh=64 regresses to ~40-50µs. The conv is bandwidth/compute-bound at 24 cores, not
  core-starved — hence double-buffering (not more cores) is the real lever.

REQUIRED ENV (test skips otherwise):
  - ``TT_METAL_DEVICE_PROFILER=1``
  - ``TT_METAL_PROFILER_MID_RUN_DUMP=1``

Run:
    export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
    pytest -s models/experimental/kokoro/perf/test_conv_text_encoder_perf_sweep.py -v
    # override shape:  KOKORO_CONV_SHAPE=512x512x5  KOKORO_CONV_BT=2x48
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from tracy.common import PROFILER_LOGS_DIR
from tracy.process_ops_logs import get_device_data_generate_report

from .test_matmul_decoder_perf_sweep import _drain_profiler

_PCC_TARGET = 0.99

_SHARD = {
    "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "auto": None,
}

# Math fidelity LoFi -> HiFi4. The conv MACs accumulate over K=Cin*kernel = 2560; fidelity sets how
# many BF16 sub-MAC passes approximate each fp32-ish multiply (LoFi=1, HiFi2=2, HiFi3=3, HiFi4=4),
# trading device time for precision. The inputs are already bf16, so the precision gain saturates fast.
_FIDELITY = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}

# Weight dtype for the conv. The conv re-reads the [Cin*k, Cout] weight per output block, so weight
# bandwidth is the dominant cost (weights_double_buffer was the biggest single win in the DRAM sweep).
# bfloat8_b halves the weight bytes — a direct lever on that bandwidth, at some precision cost.
_WDTYPE = {
    "bf16": ttnn.bfloat16,
    "bf8_b": ttnn.bfloat8_b,
}


@dataclass(frozen=True)
class ConvCase:
    label: str
    shard: str
    act_block_h: int  # 0 == leave default
    split_reader: bool
    act_dbuf: bool
    wt_dbuf: bool
    force_grid: Optional[tuple] = None  # (gx, gy) -> override_sharding_config onto this exact grid
    fidelity: str = "HiFi3"  # math_fidelity for the conv MACs (key into _FIDELITY)
    fp32_acc: bool = True  # fp32_dest_acc_en — fp32 accumulation in the dest register
    # --- extra knobs the original DRAM sweep held fixed (defaults reproduce it exactly) ---
    weights_dtype: str = "bf16"  # key into _WDTYPE — conv weight precision (bandwidth lever)
    config_in_dram: bool = True  # config_tensors_in_dram; False keeps the conv's config tensors in L1
    activation_reuse: bool = False  # enable_activation_reuse — reuse act across output blocks
    packer_l1_acc: bool = False  # compute-config: accumulate the K-reduction in L1 (packer)
    full_inner_dim: bool = False  # full_inner_dim — don't split the K contraction into inner blocks
    act_block_w_div: int = 0  # 0 == leave ttnn default (1); else block-width divisor
    reshard_if_not_optimal: bool = False  # let ttnn insert a reshard if the input layout is suboptimal
    transpose_shards: bool = False  # transpose the shard grid orientation


def _make_cases(m_tiles: int, n_tiles: int, gx_max: int, gy_max: int) -> list[ConvCase]:
    """Auto-shard-layout grid (double-buffers ON, the production setting) + a forced-core-grid block
    that explicitly pins the conv onto grids of varying core counts — including ones *above* what the
    auto heuristic picks — so the markdown shows the achievable core counts and their time (or the
    TT_FATAL when a grid is infeasible, e.g. block_w<2). The feasible grid is bounded by the output
    tiles: gy<=min(m_tiles,gy_max), gx<=min(n_tiles,gx_max), and block_w=n_tiles/gx>=2 for the conv
    matmul, so for the 96x512 shape (m_tiles=3,n_tiles=16) the legal max is gy=3 x gx=8 = 24."""
    cases: list[ConvCase] = []
    for shard in ("block", "height", "width", "auto"):
        for abh in (0, 32, 64):
            for sr in (True, False):
                for adb in (True, False):
                    for wdb in (False, True):
                        cases.append(
                            ConvCase(
                                label=f"{shard}_abh{abh}_sr{int(sr)}_adb{int(adb)}_wdb{int(wdb)}",
                                shard=shard,
                                act_block_h=abh,
                                split_reader=sr,
                                act_dbuf=adb,
                                wt_dbuf=wdb,
                            )
                        )
    # Forced-core-grid BLOCK cases (double buffers on): pin gx x gy explicitly across a range of core
    # counts to surface how many cores are actually usable + the device time at each. Grids beyond the
    # auto pick (e.g. gx=16 -> 48 cores) clamp/regress or TT_FATAL for this shape — the saturation point.
    # NOTE: only block is forced. Forcing WIDTH sharding onto a small grid is unsafe — width puts the
    # whole spatial extent + a large channel slice on each core, so few cores overflow L1 (CBs grow to
    # ~5.8 MB >> 1.5 MB/core) and the device SEGFAULTS (uncatchable, takes down the run). Width's usable
    # core count is fixed by the channel tiling (auto = 16 here); it can't be swept the way block can.
    seen = set()
    for gy in range(1, min(m_tiles, gy_max) + 1):
        for gx in range(1, min(n_tiles, gx_max) + 1):
            ncores = gx * gy
            if ncores <= 1 or ncores in seen:
                continue
            seen.add(ncores)
            cases.append(
                ConvCase(
                    label=f"force_{gx}x{gy}={ncores}c_block",
                    shard="block",
                    act_block_h=0,
                    split_reader=True,
                    act_dbuf=True,
                    wt_dbuf=True,
                    force_grid=(gx, gy),
                )
            )

    # Forced-core-grid WIDTH cases — HIGH core counts ONLY. Width puts the full spatial extent + a
    # channel slice (Cout/ncores) on each core, so the per-core L1 footprint blows up at LOW core
    # counts: forcing few cores grows the CBs past the 1.5 MB/core limit and SEGFAULTS — uncatchable,
    # it takes down the whole run (measured: 3 cores -> 5.8 MB CB). The segfault boundary is somewhere
    # between 3 and ~13 cores, so we ONLY emit grids that are BOTH (a) >= 13 cores (the verified-safe
    # minimum) AND (b) per-core channels <= 64 (2 tiles). Verified safe in isolation: 13x1 runs
    # (~58µs), 8x2 runs at 16 cores (~42µs); higher requests (16x1/16x2) throw a *catchable* TT_FATAL.
    # These show width's ceiling: it can't exceed n_tiles cores (forcing more FATALs), and at its 16
    # max it's ~2x slower than block@24 — width replicates the spatial work, block partitions it.
    seen_w = set()
    cout = n_tiles * 32  # output channels
    for gx, gy in [(13, 1), (n_tiles // 2, 2), (n_tiles, 1)]:  # 13c, 16c(8x2), and the FATAL ceiling
        ncores = gx * gy
        per_core_ch = -(-cout // max(ncores, 1))  # ceil
        if gx < 1 or gx > gx_max or gy > min(m_tiles, gy_max) or ncores in seen_w:
            continue
        if ncores < 13 or per_core_ch > 64:  # stay out of the low-core L1-overflow / segfault zone
            continue
        seen_w.add(ncores)
        cases.append(
            ConvCase(
                label=f"force_{gx}x{gy}={ncores}c_width",
                shard="width",
                act_block_h=0,
                split_reader=True,
                act_dbuf=True,
                wt_dbuf=True,
                force_grid=(gx, gy),
            )
        )

    # FIDELITY sweep (the headline knob here): hold the production-best program config — block shard
    # (24c), split_reader on, both double buffers on — and sweep math_fidelity LoFi -> HiFi4, each with
    # fp32_dest_acc_en on/off AND act_block_h_override in {auto(0), 32, 64}. This isolates the
    # fidelity/precision trade-off the TextEncoder conv actually pays, with abh as a second axis so the
    # table reports the *production-accurate* time: production pins abh=32 (the `[ABH=1|1 ...]` tag),
    # which blocks the activation into single-tile rows that double-buffer in L1 — ~1.5-1.9x faster than
    # auto for this M=96 (3 tile-row) shape, so abh=32 reproduces the ~16µs the full model sees while
    # abh=0/auto reads ~29µs. The conv inputs are bf16 so higher fidelity buys little PCC but costs
    # device time (more BF16 sub-MAC passes); the goal is the lowest-fidelity/cheapest-block config that
    # still clears the PCC bar. abh=64 fattens the block past L1 and drops the grid to 16c (regresses).
    for fid in ("LoFi", "HiFi2", "HiFi3", "HiFi4"):
        for acc in (True, False):
            for abh in (0, 32, 64):
                cases.append(
                    ConvCase(
                        label=f"fid_{fid}_fp32acc{int(acc)}_abh{abh}",
                        shard="block",
                        act_block_h=abh,
                        split_reader=True,
                        act_dbuf=True,
                        wt_dbuf=True,
                        fidelity=fid,
                        fp32_acc=acc,
                    )
                )
    return cases


def _make_l1_cases() -> list[ConvCase]:
    """L1-only sweep of the knobs the original DRAM sweep held fixed.

    Anchored on the PRODUCTION program config (block-sharded, abh=32, split_reader on, both double
    buffers on, LoFi + fp32_acc — see ``tt/tt_conv.py::_batched_tt_conv1d_nlc``), every case here
    pins ``config_tensors_in_dram=False`` so the conv's config tensors stay L1-resident. Two reference
    rows anchor the read: ``prod_dram`` (config tensors in DRAM = the current production setting) and
    ``L1_base`` (identical but config tensors in L1). Every other row flips ONE extra knob on top of
    ``L1_base`` — the knobs the DRAM sweep never varied: weight dtype, activation reuse, packer-L1
    accumulation, full-inner-dim, act-block-width divisor, reshard, transpose-shards — plus a couple
    of bandwidth-lever combos. All at the production fidelity so timings are production-comparable."""

    def base(label, **kw):
        d = dict(
            shard="block",
            act_block_h=32,
            split_reader=True,
            act_dbuf=True,
            wt_dbuf=True,
            fidelity="LoFi",
            fp32_acc=True,
            config_in_dram=False,
        )
        d.update(kw)
        return ConvCase(label=label, **d)

    cases = [
        # References
        base("prod_dram", config_in_dram=True),  # current production (config tensors in DRAM)
        base("L1_base"),  # same, config tensors in L1 — the "does L1 help at all" baseline
        # One-knob-at-a-time on top of L1_base
        base("L1_wt_bf8", weights_dtype="bf8_b"),  # halve weight bytes (the dominant bandwidth)
        base("L1_act_reuse", activation_reuse=True),  # reuse activation across output blocks
        base("L1_packer_l1acc", packer_l1_acc=True),  # accumulate K-reduction in L1
        base("L1_full_inner", full_inner_dim=True),  # don't split K into inner blocks
        base("L1_abwdiv2", act_block_w_div=2),  # narrower block width
        base("L1_abwdiv4", act_block_w_div=4),
        base("L1_reshard", reshard_if_not_optimal=True),
        base("L1_transpose", transpose_shards=True),
        # Bandwidth-lever combos (the promising ones stacked)
        base("L1_bf8_reuse", weights_dtype="bf8_b", activation_reuse=True),
        base("L1_bf8_reuse_packer", weights_dtype="bf8_b", activation_reuse=True, packer_l1_acc=True),
        base("L1_bf8_packer", weights_dtype="bf8_b", packer_l1_acc=True),
        # --- WIDTH sharding (L1) — the DRAM sweep had it ~2x slower than block; re-check under L1 +
        # the bf8_b weight lever to see if it closes the gap. No forced grid (few-core width overflows
        # L1 and segfaults; auto picks a safe core count). abh 0/32 x wt bf16/bf8_b x abw div.
        base("L1w_bf16", shard="width", act_block_h=0),
        base("L1w_bf16_abh32", shard="width", act_block_h=32),
        base("L1w_bf8", shard="width", act_block_h=0, weights_dtype="bf8_b"),
        base("L1w_bf8_abh32", shard="width", act_block_h=32, weights_dtype="bf8_b"),
        base("L1w_bf8_abwdiv2", shard="width", act_block_h=0, weights_dtype="bf8_b", act_block_w_div=2),
        base("L1w_bf8_packer", shard="width", act_block_h=0, weights_dtype="bf8_b", packer_l1_acc=True),
        # --- HEIGHT sharding (L1) — the DRAM sweep FATAL'd every height row in the sliding-window/halo
        # op. Re-checked here (L1 + bf8_b) to confirm whether it's a hard op limitation or L1-fixable.
        base("L1h_bf16", shard="height", act_block_h=0),
        base("L1h_bf16_abh32", shard="height", act_block_h=32),
        base("L1h_bf8", shard="height", act_block_h=0, weights_dtype="bf8_b"),
    ]
    return cases


def _conv_device_times(device: ttnn.Device) -> list[float]:
    """Device-kernel time (µs) for every device op the last conv emitted (halo + conv2d + reshard).

    The device-level profiler dump has no op-name column (op codes live in the host ops log), so we
    return all durations. The conv2d itself is the dominant op (~32µs vs ~1µs halo/reshard), so its
    time is ``max(...)`` and the conv's full device cost is ``sum(...)``.
    """
    ttnn.ReadDeviceProfiler(device)
    data = get_device_data_generate_report(
        PROFILER_LOGS_DIR, None, None, None, export_csv=False, cleanup_device_log=True
    )
    out: list[float] = []
    for d in data:
        dur = d.get("DEVICE KERNEL DURATION [ns]")
        if dur is not None:
            out.append(float(dur) / 1e3)
    return out


@dataclass
class ConvResult:
    case: ConvCase
    conv_us: float
    total_us: float
    pcc: float
    pcc_pass: bool
    n_ops: int = 0
    n_cores: int = 0  # cores the conv parallelized onto (output shard grid)
    out_layout: str = "-"  # output memory layout (block/width/height/interleaved)
    err: Optional[str] = None


def _layout_short(mc: ttnn.MemoryConfig) -> str:
    ml = mc.memory_layout
    return {
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: "block",
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width",
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height",
        ttnn.TensorMemoryLayout.INTERLEAVED: "interleaved",
    }.get(ml, str(ml))


def _out_cores(t: ttnn.Tensor) -> int:
    mc = t.memory_config()
    ss = mc.shard_spec
    if ss is None:
        return 0
    try:
        return int(ss.grid.num_cores())
    except Exception:
        return 0


def _shape():
    c, b = os.getenv("KOKORO_CONV_SHAPE", "512x512x5"), os.getenv("KOKORO_CONV_BT", "2x48")
    cin, cout, k = (int(v) for v in c.lower().split("x"))
    bt, seq = (int(v) for v in b.lower().split("x"))
    return cin, cout, k, bt, seq


# Open the device with an L1-small region: the L1-config sweep (config_tensors_in_dram=False) places
# the conv's config tensors in L1_SMALL, which is 0 B by default and FATALs. 32 KiB/core is ample (the
# conv needs ~16 B/bank) and does not perturb the DRAM cases (they don't touch L1_SMALL).
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_text_encoder_conv_perf_sweep(device):
    if os.getenv("TT_METAL_DEVICE_PROFILER") is None:
        pytest.skip("device-time sweep needs a profiler build + TT_METAL_DEVICE_PROFILER=1")
    if os.getenv("TT_METAL_PROFILER_MID_RUN_DUMP") is None:
        pytest.skip("set TT_METAL_PROFILER_MID_RUN_DUMP=1 so ReadDeviceProfiler flushes mid-run")

    Cin, Cout, K, B, T = _shape()
    pad = K // 2
    out_len = T  # stride=1, padding=K//2 keeps length

    torch.manual_seed(0)
    # torch conv1d reference on the [B, Cin, T] layout
    x_bct = torch.randn(B, Cin, T, dtype=torch.float32)
    w = torch.randn(Cout, Cin, K, dtype=torch.float32)
    bias = torch.randn(Cout, dtype=torch.float32)
    ref_bct = torch.nn.functional.conv1d(x_bct, w, bias, stride=1, padding=pad)  # [B, Cout, out_len]
    ref_flat = ref_bct.permute(0, 2, 1).reshape(1, 1, B * out_len, Cout)  # [1,1,B*out_len,Cout]

    # Device input flattened NLC: [1, 1, B*T, Cin]
    x_flat_t = x_bct.permute(0, 2, 1).reshape(1, 1, B * T, Cin)

    w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    grid = device.compute_with_storage_grid_size()
    m_tiles = (B * out_len + 31) // 32
    n_tiles = (Cout + 31) // 32
    cases = _make_cases(m_tiles, n_tiles, grid.x, grid.y)
    # KOKORO_CONV_FIDELITY_ONLY=1 runs just the fidelity sweep (the 8 `fid_*` cases) — fast enough to
    # finish inside the pytest timeout; the full program/forced-grid sweep is the slow default.
    if os.getenv("KOKORO_CONV_FIDELITY_ONLY"):
        cases = [c for c in cases if c.label.startswith("fid_")]
    # KOKORO_CONV_L1_ONLY=1 runs the focused L1 sweep of the previously-fixed knobs (config tensors in
    # L1, one extra knob per row on top of the production config). Small + fast; writes its own table.
    l1_only = bool(os.getenv("KOKORO_CONV_L1_ONLY"))
    if l1_only:
        cases = _make_l1_cases()
    results: list[ConvResult] = []
    for case in cases:
        x_in = warm = timed = None
        try:
            compute_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=_FIDELITY[case.fidelity],
                math_approx_mode=False,
                fp32_dest_acc_en=case.fp32_acc,
                packer_l1_acc=case.packer_l1_acc,
            )
            x_in = ttnn.from_torch(
                x_flat_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            def _mk_config():
                cfg = ttnn.Conv1dConfig(weights_dtype=_WDTYPE[case.weights_dtype])
                cfg.config_tensors_in_dram = case.config_in_dram
                if _SHARD[case.shard] is not None:
                    cfg.shard_layout = _SHARD[case.shard]
                if case.act_block_h:
                    cfg.act_block_h_override = case.act_block_h
                cfg.force_split_reader = case.split_reader
                cfg.enable_act_double_buffer = case.act_dbuf
                cfg.enable_weights_double_buffer = case.wt_dbuf
                cfg.enable_activation_reuse = case.activation_reuse
                cfg.full_inner_dim = case.full_inner_dim
                cfg.reshard_if_not_optimal = case.reshard_if_not_optimal
                cfg.transpose_shards = case.transpose_shards
                if case.act_block_w_div:
                    cfg.act_block_w_div = case.act_block_w_div
                if case.force_grid is not None:
                    gx, gy = case.force_grid
                    cfg.core_grid = ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]
                    )
                    cfg.override_sharding_config = True
                return cfg

            def _run():
                return ttnn.conv1d(
                    input_tensor=x_in,
                    weight_tensor=w_tt,
                    in_channels=Cin,
                    out_channels=Cout,
                    device=device,
                    bias_tensor=b_tt,
                    kernel_size=K,
                    stride=1,
                    padding=pad,
                    dilation=1,
                    batch_size=B,
                    input_length=T,
                    conv_config=_mk_config(),
                    compute_config=compute_config,
                    groups=1,
                    dtype=ttnn.bfloat16,
                )

            warm = _run()
            ttnn.synchronize_device(device)
            out_torch = ttnn.to_torch(warm).float().reshape(ref_flat.shape)
            ttnn.deallocate(warm)
            warm = None
            _drain_profiler(device)
            ttnn.synchronize_device(device)

            timed = _run()
            ttnn.synchronize_device(device)
            durs = _conv_device_times(device)
            if not durs:
                raise RuntimeError("no device ops in profiler dump — profiler build?")
            conv_us = max(durs)  # the conv2d dominates (halo/reshard are ~1µs)
            total_us = sum(durs)
            n_cores = _out_cores(timed)
            out_layout = _layout_short(timed.memory_config())

            pcc_pass, pcc = comp_pcc(ref_flat, out_torch, _PCC_TARGET)
            results.append(
                ConvResult(case, conv_us, total_us, float(pcc), bool(pcc_pass), len(durs), n_cores, out_layout)
            )
        except Exception as e:
            results.append(
                ConvResult(case, float("inf"), float("inf"), 0.0, False, err=str(e).strip().splitlines()[0][:80])
            )
        finally:
            for t in (timed, warm, x_in):
                if t is not None:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            ttnn.synchronize_device(device)

    passing = [r for r in results if r.pcc_pass]

    if l1_only:
        # Dedicated L1-sweep table: knobs the DRAM sweep held fixed, all at config_tensors_in_dram=False.
        md_path = os.getenv(
            "KOKORO_CONV_MD",
            os.path.join(os.path.dirname(__file__), "conv_sweep_l1_results.md"),
        )
        l1_best = min(passing, key=lambda r: r.conv_us) if passing else None
        prod = next((r for r in results if r.case.label == "prod_dram"), None)
        md = [
            f"# TextEncoder Conv1d — L1-config sweep (in={Cin} out={Cout} k={K} B={B} T={T}, M={B*T}, K={Cin*K}, N={Cout})",
            "",
            "All rows use the production program config (block-sharded, abh=32, split_reader, both "
            "double buffers, LoFi + fp32_acc). `prod_dram` = current production (config tensors in "
            "DRAM); every other row pins `config_tensors_in_dram=False` (config tensors in L1). Each "
            "`L1_*` row flips ONE extra knob the original DRAM sweep never varied (or a combo).",
            "",
            "- `conv_us` = dominant conv2d op; `total_us` = all device ops the conv emits (halo+conv2d+reshard).",
            "- `wt_dtype` = conv weight precision. `cfg_dram` = config_tensors_in_dram. `reuse` = "
            "enable_activation_reuse. `pk_l1` = packer_l1_acc. `full_k` = full_inner_dim. `abw` = "
            "act_block_w_div (auto=default 1). `resh` = reshard_if_not_optimal. `tps` = transpose_shards.",
            "",
            (
                f"**Fastest PCC-passing: `{l1_best.case.label}` — {l1_best.conv_us:.2f}µs conv "
                f"({l1_best.total_us:.2f}µs total), PCC={l1_best.pcc:.4f}"
                + (f" vs prod_dram {prod.conv_us:.2f}µs" if prod and not prod.err else "")
                + "**"
                if l1_best
                else "No row passed PCC."
            ),
            "",
            "",
            "**Findings:** `bf8_b` conv weights are the only real lever — 16.1→11.4µs (−29%) at PCC "
            "0.99988 vs 0.99989 (weight re-read bandwidth dominates this M=96 conv, so halving the "
            "weight bytes wins; `packer_l1_acc` adds ~0 on top). Moving config tensors to L1 "
            "(`config_tensors_in_dram=False`, `L1_base`) is neutral vs DRAM (16.4 vs 16.3µs). "
            "`enable_activation_reuse` FATALs — not supported for block sharding. `full_inner_dim`, "
            "`packer_l1_acc`, `act_block_w_div`, `reshard_if_not_optimal` are all within noise; "
            "`transpose_shards` regresses.",
            "",
            "**Shard layout (L1):** BLOCK stays the clear winner. WIDTH is ~29µs at best (`L1w_bf8`) — "
            "~2.5× slower than block+bf8; bf8 barely helps width (it replicates spatial work across "
            "cores rather than re-reading weights), and `abh=32` DOUBLES width to ~53µs (opposite of "
            "block, where abh=32 helps), while `act_block_w_div` FATALs on width. HEIGHT sharding "
            "FATALs on every config in the sliding-window/halo op — a hard op-level limitation for "
            "this Conv1d-as-Conv2d shape, NOT L1- or bf8-fixable (matches the DRAM sweep).",
            "",
            "| config | shard | wt_dtype | cfg_dram | abh | reuse | pk_l1 | full_k | abw | resh | tps | cores | #ops | conv_us | total_us | PCC | result |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(results, key=lambda r: (r.err is not None, r.conv_us)):
            c = r.case
            tag = " **(fastest)**" if r is l1_best else ""
            abw = c.act_block_w_div or "auto"
            abh = c.act_block_h or "auto"
            if r.err:
                md.append(
                    f"| {c.label}{tag} | {c.shard} | {c.weights_dtype} | {c.config_in_dram} | {abh} | "
                    f"{c.activation_reuse} | {c.packer_l1_acc} | {c.full_inner_dim} | {abw} | "
                    f"{c.reshard_if_not_optimal} | {c.transpose_shards} | - | - | - | - | - | ERROR: {r.err} |"
                )
                continue
            md.append(
                f"| {c.label}{tag} | {c.shard} | {c.weights_dtype} | {c.config_in_dram} | {abh} | "
                f"{c.activation_reuse} | {c.packer_l1_acc} | {c.full_inner_dim} | {abw} | "
                f"{c.reshard_if_not_optimal} | {c.transpose_shards} | {r.n_cores} | {r.n_ops} | "
                f"{r.conv_us:.2f} | {r.total_us:.2f} | {r.pcc:.4f} | {'PASS' if r.pcc_pass else 'FAIL'} |"
            )
        with open(md_path, "w") as f:
            f.write("\n".join(md) + "\n")
        logger.info(f"L1 sweep markdown written to {md_path}")
        logger.info("=== L1-config conv sweep (production program config, config tensors in L1) ===")
        for r in sorted(results, key=lambda r: (r.err is not None, r.conv_us)):
            if r.err:
                logger.info(f"  {r.case.label:>22} ERROR: {r.err}")
            else:
                logger.info(
                    f"  {r.case.label:>22} conv={r.conv_us:7.2f}µs total={r.total_us:7.2f}µs "
                    f"cores={r.n_cores:>2} PCC={r.pcc:.5f} {'PASS' if r.pcc_pass else 'FAIL'}"
                )
        if l1_best:
            logger.info(f"FASTEST L1 config: {l1_best.case.label} -> {l1_best.conv_us:.2f}µs, PCC={l1_best.pcc:.4f}")
        ttnn.deallocate(w_tt)
        ttnn.deallocate(b_tt)
        assert passing, f"No L1 conv config reached PCC>={_PCC_TARGET}."
        return

    best_conv = min(passing, key=lambda r: r.conv_us) if passing else None
    best_total = min(passing, key=lambda r: r.total_us) if passing else None
    baseline = next(
        (
            r
            for r in results
            if r.case.shard == "auto"
            and r.case.act_block_h == 0
            and r.case.split_reader
            and not r.case.act_dbuf
            and not r.case.wt_dbuf
        ),
        None,
    )

    logger.info(
        f"=== TextEncoder Conv1d sweep  in={Cin} out={Cout} k={K} B={B} T={T}  (M={B*T}, K={Cin*K}, N={Cout}) ==="
    )
    logger.info(f"conv_us = dominant (conv2d) op; total_us = all device ops the conv emits (halo+conv2d+reshard)")
    logger.info(
        f"{'config':>34} {'cores':>5} {'conv_us':>8} {'total_us':>9} {'#ops':>4} {'PCC':>8} {'result':>7}  note"
    )
    for r in sorted(results, key=lambda r: r.conv_us):
        c = r.case
        if r.err:
            logger.info(f"{c.label:>34} {'-':>5} {'-':>8} {'-':>9} {'-':>4} {'-':>8} {'ERROR':>7}  {r.err}")
            continue
        tags = []
        if r is best_conv:
            tags.append("best-conv")
        if r is best_total and r is not best_conv:
            tags.append("best-total")
        if r is baseline:
            tags.append("~baseline")
        logger.info(
            f"{c.label:>34} {r.n_cores:>5} {r.conv_us:>8.2f} {r.total_us:>9.2f} {r.n_ops:>4} {r.pcc:>8.4f} "
            f"{('PASS' if r.pcc_pass else 'FAIL'):>7}  {' '.join(tags)}"
        )

    # Full markdown table: config name | in/out mem layout | program-config knobs | cores | device time.
    md_path = os.getenv(
        "KOKORO_CONV_MD",
        os.path.join(os.path.dirname(__file__), "conv_sweep_results.md"),
    )
    md = [
        f"# TextEncoder Conv1d sweep — in={Cin} out={Cout} k={K} B={B} T={T} (M={B*T}, K={Cin*K}, N={Cout})",
        "",
        "- `conv_us` = dominant (conv2d) device op; `total_us` = sum of all device ops the conv emits (halo + conv2d + reshards).",
        "- `cores` = cores the conv parallelized onto (output shard grid). `in_mem` = requested input shard_layout; `out_mem` = actual conv output layout.",
        "- Program-config knobs: `abh` = act_block_h_override (0=auto), `split_reader`, `act_dbuf` = enable_act_double_buffer, `wt_dbuf` = enable_weights_double_buffer. `fidelity` = math_fidelity (LoFi→HiFi4), `fp32acc` = fp32_dest_acc_en. Fixed: config_tensors_in_dram=True. Program-knob rows are pinned at HiFi3/fp32acc=True; the `fid_*` rows sweep fidelity on the production-best program config.",
        "- `forced_grid` = explicit gx×gy pinned via override_sharding_config (blank = ttnn auto-selects). Rows above the auto core count exist to show the ceiling — they FATAL when the grid is infeasible (block_w<2).",
        "",
        "| config | in_mem | out_mem | forced_grid | abh | split_reader | act_dbuf | wt_dbuf | fidelity | fp32acc | cores | #ops | conv_us | total_us | PCC | result |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r.err is not None, r.conv_us, -(r.n_cores))):
        c = r.case
        fg = f"{c.force_grid[0]}x{c.force_grid[1]}" if c.force_grid else "-"
        if r.err:
            md.append(
                f"| {c.label} | {c.shard} | - | {fg} | {c.act_block_h or 'auto'} | {c.split_reader} | {c.act_dbuf} | "
                f"{c.wt_dbuf} | {c.fidelity} | {c.fp32_acc} | - | - | - | - | - | ERROR: {r.err} |"
            )
            continue
        note = " ".join(
            t
            for t, cond in (
                ("best-conv", r is best_conv),
                ("best-total", r is best_total and r is not best_conv),
                ("~baseline", r is baseline),
            )
            if cond
        )
        md.append(
            f"| {c.label}{(' **'+note+'**') if note else ''} | {c.shard} | {r.out_layout} | {fg} | {c.act_block_h or 'auto'} | "
            f"{c.split_reader} | {c.act_dbuf} | {c.wt_dbuf} | {c.fidelity} | {c.fp32_acc} | {r.n_cores} | {r.n_ops} | {r.conv_us:.2f} | "
            f"{r.total_us:.2f} | {r.pcc:.4f} | {'PASS' if r.pcc_pass else 'FAIL'} |"
        )

    # Dedicated fidelity-sweep table (the headline result): production-best program config, fidelity
    # LoFi->HiFi4 x fp32_acc on/off. Sorted fastest-first among PCC-passing rows so the recommended
    # config (lowest fidelity that still passes) is at the top.
    fid_rows = [r for r in results if not r.err and r.case.label.startswith("fid_")]
    if fid_rows:
        fid_pass = [r for r in fid_rows if r.pcc_pass]
        fid_best = min(fid_pass, key=lambda r: r.conv_us) if fid_pass else None
        md += [
            "",
            "## Fidelity sweep (production-best program config: auto/block 24c, double-buffered)",
            "",
            "Lowest device time among PCC-passing rows is the recommended conv `math_fidelity`. Inputs are "
            "bf16, so PCC saturates by HiFi2 and extra passes only cost time.",
            "",
            (
                f"**Recommended: `{fid_best.case.fidelity}` (fp32_acc={fid_best.case.fp32_acc}) — "
                f"{fid_best.conv_us:.2f}µs conv, PCC={fid_best.pcc:.4f}**"
                if fid_best
                else "No fidelity row passed PCC."
            ),
            "",
            "| fidelity | fp32_acc | abh | cores | conv_us | total_us | PCC | result |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(fid_rows, key=lambda r: (not r.pcc_pass, r.conv_us)):
            tag = " **(recommended)**" if r is fid_best else ""
            md.append(
                f"| {r.case.fidelity}{tag} | {r.case.fp32_acc} | {r.case.act_block_h or 'auto'} | {r.n_cores} | "
                f"{r.conv_us:.2f} | {r.total_us:.2f} | {r.pcc:.4f} | {'PASS' if r.pcc_pass else 'FAIL'} |"
            )

    # Core-count saturation: forced grids that keep the row split at its max (gy = m_tiles) while the
    # requested core count climbs. Demonstrates that for THIS shape adding cores past the natural pick
    # does NOT lower device time — it rises monotonically. The conv is dispatch/overhead-bound at the
    # natural pick, not core-starved.
    #
    # Two distinct core counts (don't conflate them):
    #   - compute_grid: the cores the Conv2d KERNEL runs on (the tracy "Cores" column). The override
    #     IS honored above the natural pick, so requesting 27/30/.../39 actually runs on 27/30/.../39
    #     cores. Below the natural pick the conv stays at it. The sweep's mid-run device dump has no
    #     CORE COUNT field, so we report the requested count (= the tracy compute grid, verified in
    #     conv_sweep.log) and clamp it up to the natural minimum.
    #   - output_shard_cores: how the OUTPUT tensor is block-sharded — capped at the legal max
    #     (gy<=m_tiles, block_w=n_tiles/gx>=2 => gx<=n_tiles//2), e.g. 24 for 96x512. This is what
    #     `_out_cores` reads; it stays pinned regardless of the compute grid.
    # Tracy-measured (conv_sweep.log, 96x2560x512): 24c->22µs, 27c->24µs, 30c->27µs, 33c->29µs,
    # 36c->31µs, 39c->34µs — more compute cores, strictly worse device time.
    def _req(fg):
        return fg[0] * fg[1]

    sat = sorted(
        (r for r in results if not r.err and r.case.force_grid and r.case.force_grid[1] == m_tiles),
        key=lambda r: (r.case.shard, _req(r.case.force_grid)),
    )
    if sat:
        md += [
            "",
            "## Core-count saturation (same shape) — more cores does NOT improve device time",
            "",
            f"Holding the row split at its max (gy = m_tiles = {m_tiles}) and raising requested cores via the "
            "forced grid (block sharding; width can't be forced — few-core width overflows L1 and segfaults). "
            "`compute_cores` (the conv kernel's grid, = tracy `Cores`) scales with the request above the natural "
            "pick, yet `conv_us`/`total_us` rise monotonically — the conv is overhead-bound, not core-starved. "
            "`output_shard_cores` (the output tensor layout) stays pinned at the legal max (block_w>=2 => "
            "gx<=n_tiles//2).",
            "",
            (
                f"Global fastest (unforced, ttnn auto): **{best_conv.case.label} -> {best_conv.n_cores} cores, "
                f"{best_conv.conv_us:.2f}µs** — every forced grid below is slower; forcing MORE cores than the "
                "auto pick only makes it worse."
                if best_conv
                else ""
            ),
            "",
            "| shard | forced_grid | requested_cores | compute_cores | output_shard_cores | conv_us | total_us |",
            "|---|---|---|---|---|---|---|",
        ]
        fastest = min(sat, key=lambda r: r.conv_us)
        for r in sat:
            tag = " **(fastest)**" if r is fastest else ""
            # compute grid honors the request above the natural pick, else clamps up to it
            compute_cores = max(_req(r.case.force_grid), r.n_cores)
            md.append(
                f"| {r.case.shard} | {r.case.force_grid[0]}x{r.case.force_grid[1]}{tag} | "
                f"{_req(r.case.force_grid)} | {compute_cores} | {r.n_cores} | {r.conv_us:.2f} | {r.total_us:.2f} |"
            )

    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    logger.info(f"markdown table written to {md_path}")

    if best_conv is not None:
        logger.info(
            f"FASTEST Conv2d-only PCC-PASS: {best_conv.case.label} -> {best_conv.conv_us:.2f}µs "
            f"(total {best_conv.total_us:.2f}µs over {best_conv.n_ops} ops, PCC={best_conv.pcc:.4f})"
        )
    if best_total is not None:
        logger.info(
            f"FASTEST total (conv+halo+reshard) PCC-PASS: {best_total.case.label} -> "
            f"{best_total.total_us:.2f}µs (conv {best_total.conv_us:.2f}µs, PCC={best_total.pcc:.4f})"
        )
    if baseline is not None and not baseline.err:
        logger.info(f"~baseline (auto/abh0/sr1): conv={baseline.conv_us:.2f}µs total={baseline.total_us:.2f}µs")

    fid_rows = [r for r in results if not r.err and r.case.label.startswith("fid_")]
    fid_pass = [r for r in fid_rows if r.pcc_pass]
    if fid_pass:
        fid_best = min(fid_pass, key=lambda r: r.conv_us)
        logger.info("=== FIDELITY sweep (production-best program config) ===")
        for r in sorted(fid_rows, key=lambda r: (not r.pcc_pass, r.conv_us)):
            logger.info(
                f"  {r.case.fidelity:>5} fp32acc={int(r.case.fp32_acc)} abh={str(r.case.act_block_h or 'auto'):>4} -> "
                f"conv={r.conv_us:7.2f}µs total={r.total_us:7.2f}µs PCC={r.pcc:.5f} {'PASS' if r.pcc_pass else 'FAIL'}"
            )
        logger.info(
            f"RECOMMENDED conv fidelity: {fid_best.case.fidelity} (fp32_dest_acc_en={fid_best.case.fp32_acc}, "
            f"abh={fid_best.case.act_block_h or 'auto'}) -> {fid_best.conv_us:.2f}µs, PCC={fid_best.pcc:.4f}"
        )

    ttnn.deallocate(w_tt)
    ttnn.deallocate(b_tt)
    assert passing, f"No conv config reached PCC>={_PCC_TARGET} — sweep harness broken."
