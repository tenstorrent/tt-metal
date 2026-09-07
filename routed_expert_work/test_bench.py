# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Measurement harness for moe_fused_swiglu (and the old unified_routed_expert_moe composite for
reference). One expert, Kimi K2.6 dims by default, x bf16 ROW_MAJOR, 11x8 grid.

Env knobs (all optional):
  BENCH_M          comma list of active token counts       default 64,128,256,512,1024,5120
  BENCH_WDTYPE     bfp4 | bfp8                              default bfp4
  BENCH_ITERS      dispatches per measurement (median)      default 3
  BENCH_OLD        1 -> also run the old composite op       default 0
  BENCH_PERF       0 -> skip RT-profiler timing             default 1
  BENCH_EMB / BENCH_HID  dims                               default 7168 / 2048
  BENCH_GRID       "11x8"                                   default 11x8
  BENCH_TAG        label written into the results jsonl     default "run"
  BENCH_APPROX     1 -> math_approx_mode on                 default 0 (model passes False)
  BENCH_XSCALE     stddev of x                              default 1.0
  BENCH_WSCALE     weight init std                          default 0.02
  BENCH_SEED       torch seed                               default 42
  BENCH_WSHARD     0 | 1 | g | d  weight DRAM placement       default 0 (interleaved)
                   1 = gate/up AND down DRAM ND-sharded, g = gate/up only, d = down only.
                   Shard is one TILE-ROW tall x per-core-N wide, on the full DRAM grid,
                   ROW_MAJOR (ROUND_ROBIN_1D), which is the shape commit 220f7ee87aa
                   measured: the height is what rotates DRAM banks across K, and the
                   width is what makes one core's K-row slice a single NoC request.
  BENCH_WSHARD_GU  override gate/up shard width in tiles      default ceil(hid_t/gx)
  BENCH_WSHARD_DN  override down shard width in tiles         default max split(emb_t, gx*gy)
  BENCH_DISTINCT_W 1 -> expert i's weights are scaled by (1 + i/4)  default 0
                   REQUIRED to test any cross-expert weight path: with identical weights per
                   expert, an off-by-one that hands expert i+1 expert i's weights is invisible.
                   Every expert writes the same output rows (offsets are all 0), so the graded
                   output is the LAST expert's -- which is the one a prefetch chain feeds last.
  BENCH_EXPERTS    local experts in the ONE program            default 1
                   Every expert gets its own weight tensors (own DRAM addresses) and its own
                   count, so N experts is N sequential weight reads inside one dispatch --
                   which is what a cross-expert weight prefetch would overlap.
  BENCH_WSHARD_H   shard height in TILE-ROWS                default 1 (banks rotate per K-row)
  BENCH_WSHARD_VIA reshard | from_torch  how the shard is built  default reshard
                   `reshard` builds DRAM-interleaved and then to_memory_config()s onto the
                   shard, which MOVES the bytes: the A/B then differs only in placement.
                   Handing from_torch the ND config instead RE-QUANTISES bfp4 -- measured
                   0.74 % of values off by one quantum (maxabs 0.015625, rel-L2 3.1e-2) --
                   which leaks into the output and makes a perf A/B a numerics A/B too.

Each case appends one JSON line to routed_expert_work/results/<tag>.jsonl and logs a BENCH line.
Run through the device lock:
  scripts/run_safe_pytest.sh routed_expert_work/test_bench.py
"""
import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, TorchExpert
from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

RESULTS_DIR = Path(__file__).parent / "results"
ALLOCATED_TOKENS = 5120

_WDTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


def _env(name, default):
    return os.environ.get(name, default)


def _m_list():
    return [int(v) for v in _env("BENCH_M", "64,128,256,512,1024,5120").split(",") if v]


def _grid():
    x, y = _env("BENCH_GRID", "11x8").split("x")
    return ttnn.CoreCoord(int(x), int(y))


def _split_sizes(total: int, groups: int):
    """`base + (i < rem)` split, the same one moe_fused_swiglu_geometry.cpp uses."""
    base, rem = total // groups, total % groups
    return [base + (1 if i < rem else 0) for i in range(groups)]


def _preferred_shard_widths(emb: int, hidden: int, grid) -> tuple[int, int]:
    """(gate_up, down) shard width in TILES: exactly the N slice one core reads per K-row.

    gate/up — the hidden axis is split across grid COLUMNS, so column x owns
    ceil(hid_t / gx) tiles (`Blocking::choose_hn_pad`'s `floor` candidate, which is the one
    it takes for these dims).
    down    — the emb-output axis is split across ALL cores, so `ec_max` is the widest slice.
    """
    hid_t, emb_t = hidden // ttnn.TILE_SIZE, emb // ttnn.TILE_SIZE
    hn_pad = (hid_t + grid.x - 1) // grid.x
    ec_max = max(_split_sizes(emb_t, grid.x * grid.y))
    return hn_pad, ec_max


def _nd_shard_config(device, n_tiles: int, height_tiles: int = 1):
    """DRAM ND shard, one tile-row tall and `n_tiles` wide, spread over every DRAM bank.

    Height of exactly one tile-row is load-bearing: shards distribute ROUND_ROBIN_1D, so
    consecutive K-rows land in DIFFERENT banks. A core pinned to one bank saturates near
    30 GB/s no matter how large the request, while the same bytes with the bank rotating
    reach ~370 GB/s -- so a taller shard (one request per K-BLOCK) measures no faster than
    interleaved. The op reads the width off the tensor (`geometry::nd_shard_n_tiles`) and
    coalesces each read up to the shard boundary.
    """
    dram = device.dram_grid_size()
    return ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([height_tiles * ttnn.TILE_SIZE, n_tiles * ttnn.TILE_SIZE]),
            grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))]),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _weight_mem_configs(device, emb: int, hidden: int, grid):
    """(gate_up, down) weight memory configs for BENCH_WSHARD; None == DRAM interleaved."""
    which = _env("BENCH_WSHARD", "0")
    if which == "0":
        return None, None, (0, 0)
    gu_w, dn_w = _preferred_shard_widths(emb, hidden, grid)
    gu_w = int(_env("BENCH_WSHARD_GU", str(gu_w)))
    dn_w = int(_env("BENCH_WSHARD_DN", str(dn_w)))
    # BENCH_WSHARD_H > 1 pins a core to ONE bank for that many K-rows instead of rotating
    # every row, which is the shape the preferred config deliberately rejects.
    height = int(_env("BENCH_WSHARD_H", "1"))
    gu_mc = _nd_shard_config(device, gu_w, height)
    dn_mc = _nd_shard_config(device, dn_w, height)
    if which == "g":
        return gu_mc, None, (gu_w, 0)
    if which == "d":
        return None, dn_mc, (0, dn_w)
    return gu_mc, dn_mc, (gu_w, dn_w)


def _compute_config(approx: bool):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=approx,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _old_compute_config():
    # COMPUTE_KERNEL_CONFIG_LOFI from tt_routed_expert.py: what the model hands the composite.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _err_metrics(ref: torch.Tensor, out: torch.Tensor):
    ref = ref.float()
    out = out.float()
    _, pcc = comp_pcc(ref, out)
    diff = out - ref
    rel_rms = (diff.norm() / ref.norm()).item()
    max_abs = diff.abs().max().item()
    return {"pcc": float(pcc), "rel_rms": rel_rms, "max_abs": max_abs}


def _median_ns(device, run_fn, kernel_dir: str, iters: int):
    def run_all():
        for _ in range(iters):
            run_fn()

    _, per_program = profile_realtime_program_merged(device, run_all)
    matched = [
        e["duration_ns"]
        for e in per_program.values()
        if any(kernel_dir in s.replace("\\", "/") for s in e["kernel_sources"])
    ]
    if len(matched) != iters:
        for seq, (rid, e) in enumerate(per_program.items()):
            logger.error(
                f"  [{seq}] runtime_id={rid} ns={e['duration_ns']:.0f} "
                f"kernels={sorted({s.rsplit('/', 1)[-1] for s in e['kernel_sources']})}"
            )
        raise AssertionError(f"expected {iters} programs matching {kernel_dir}, got {len(matched)}")
    return statistics.median(matched), matched


@pytest.mark.parametrize("active_tokens", _m_list(), ids=lambda m: f"M{m}")
def test_bench(device, active_tokens):
    emb = int(_env("BENCH_EMB", "7168"))
    hidden = int(_env("BENCH_HID", "2048"))
    wdtype_name = _env("BENCH_WDTYPE", "bfp4")
    wdtype = _WDTYPES[wdtype_name]
    iters = int(_env("BENCH_ITERS", "3"))
    # NOT `run_old`: `def run_old()` below would shadow the flag and the truthy
    # function object would make `if run_old:` fire on every run.
    also_run_old = _env("BENCH_OLD", "0") == "1"
    do_perf = _env("BENCH_PERF", "1") == "1"
    tag = _env("BENCH_TAG", "run")
    approx = _env("BENCH_APPROX", "0") == "1"
    xscale = float(_env("BENCH_XSCALE", "1.0"))
    wscale = float(_env("BENCH_WSCALE", "0.02"))
    seed = int(_env("BENCH_SEED", "42"))
    grid = _grid()

    torch.manual_seed(seed)
    n_experts = int(_env("BENCH_EXPERTS", "1"))
    distinct_w = _env("BENCH_DISTINCT_W", "0") == "1"
    base_weights = {
        "gate_proj": torch.randn(hidden, emb, dtype=torch.float32) * wscale,
        "up_proj": torch.randn(hidden, emb, dtype=torch.float32) * wscale,
        "down_proj": torch.randn(emb, hidden, dtype=torch.float32) * wscale,
    }

    def expert_scale(e):
        return (1.0 + e / 4.0) if distinct_w else 1.0

    # The reference is the LAST expert's: every expert writes the same rows, so it wins.
    weights = {k: v * expert_scale(n_experts - 1) for k, v in base_weights.items()}
    torch_active = torch.randn(active_tokens, emb, dtype=torch.float32) * xscale
    if _env("BENCH_SPIKY", "0") == "1":
        # heavy-tailed activations: 1% of the positions carry 16x outliers, plus 8 outlier channels
        # (32x) shared by every token -- the block-float exponent-sharing worst case
        mask = torch.rand(active_tokens, emb) < 0.01
        torch_active = torch_active * (1 + 15 * mask.float())
        torch_active[:, torch.randperm(emb)[:8]] *= 32
    torch_input = torch.zeros(ALLOCATED_TOKENS, emb, dtype=torch.float32)
    torch_input[:active_tokens] = torch_active

    with torch.no_grad():
        if _env("BENCH_IDENTITY", "0") == "1":
            # pairs with MOE_FUSED_SWIGLU_DEFINES=MOE_DEBUG_NO_SILU: h = gate * up, no activation
            g = torch_active @ weights["gate_proj"].T
            u = torch_active @ weights["up_proj"].T
            ref = (g * u) @ weights["down_proj"].T
        else:
            ref = TorchExpert(emb, hidden, weights, activation=ACTIVATION_SILU)(torch_active)

    def to_device(t, dtype, layout, memory_config=None):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    gu_mc, dn_mc, shard_widths = _weight_mem_configs(device, emb, hidden, grid)
    via_reshard = _env("BENCH_WSHARD_VIA", "reshard") == "reshard"

    def weight(t, memory_config):
        if memory_config is None:
            return to_device(t, wdtype, ttnn.TILE_LAYOUT)
        if via_reshard:
            return ttnn.to_memory_config(to_device(t, wdtype, ttnn.TILE_LAYOUT), memory_config)
        return to_device(t, wdtype, ttnn.TILE_LAYOUT, memory_config)

    # Distinct tensors per expert so each carries its own DRAM base: the point of the
    # multi-expert case is N sequential weight reads, not N reads of one cached address.
    w_gate = [weight(base_weights["gate_proj"].T * expert_scale(e), gu_mc) for e in range(n_experts)]
    w_up = [weight(base_weights["up_proj"].T * expert_scale(e), gu_mc) for e in range(n_experts)]
    w_down = [weight(base_weights["down_proj"].T * expert_scale(e), dn_mc) for e in range(n_experts)]
    x = to_device(torch_input.reshape(1, 1, ALLOCATED_TOKENS, emb), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    def idx_tensor(values):
        return to_device(torch.tensor(values, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    # Every local expert points at a DIFFERENT global id, and each carries the same count, so
    # the dispatch does n_experts equal-sized passes.
    idx = idx_tensor(list(range(n_experts)))
    counts = idx_tensor([active_tokens] * n_experts)
    offsets = idx_tensor([0] * n_experts)

    def run_fused():
        return ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            x,
            w_gate,
            w_up,
            w_down,
            counts,
            idx,
            input_m_tiles=ALLOCATED_TOKENS // 32,
            core_grid=grid,
            compute_kernel_config=_compute_config(approx),
            dtype=ttnn.bfloat16 if _env("BENCH_OUT_BF16", "0") == "1" else ttnn.bfloat8_b,
            intermediate_dtype=ttnn.bfloat16 if _env("BENCH_INTERMEDIATE", "bfp8") == "bf16" else ttnn.bfloat8_b,
        )

    def run_old():
        return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x,
            offsets,
            counts,
            idx,
            w_gate,
            w_up,
            w_down,
            max_dispatched_tokens_per_expert=ALLOCATED_TOKENS,
            compute_kernel_config=_old_compute_config(),
        )

    record = {
        "tag": tag,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "M": active_tokens,
        "emb": emb,
        "hidden": hidden,
        "wdtype": wdtype_name,
        "grid": f"{grid.x}x{grid.y}",
        "approx": approx,
        "xscale": xscale,
        "wscale": wscale,
        "wshard": _env("BENCH_WSHARD", "0"),
        "shard_gu_tiles": shard_widths[0],
        "shard_dn_tiles": shard_widths[1],
        "shard_via": _env("BENCH_WSHARD_VIA", "reshard"),
        "shard_h": int(_env("BENCH_WSHARD_H", "1")),
        "experts": n_experts,
        "distinct_w": distinct_w,
    }

    # correctness first (also warms the JIT cache so the timed window is pure dispatch)
    out = ttnn.to_torch(run_fused())[0, 0][:active_tokens]
    record["fused"] = _err_metrics(ref, out)
    if _env("BENCH_SAVE", ""):
        torch.save({"ref": ref, "out": out.float()}, _env("BENCH_SAVE", ""))
    if also_run_old:
        out_old = ttnn.to_torch(run_old())[0, 0][:active_tokens]
        record["old"] = _err_metrics(ref, out_old)
    ttnn.synchronize_device(device)

    if do_perf:
        med, samples = _median_ns(device, run_fused, "/moe_fused_swiglu/", iters)
        record["fused"]["ns"] = med
        record["fused"]["samples_ns"] = samples
        if also_run_old:
            med_old, samples_old = _median_ns(device, run_old, "/unified_routed_expert_ffn/", iters)
            record["old"]["ns"] = med_old
            record["old"]["samples_ns"] = samples_old

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{tag}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    def fmt(d):
        s = f"pcc={d['pcc']:.6f} rel_rms={d['rel_rms']:.5f}"
        if "ns" in d:
            s += f" ns={d['ns']:.0f}"
        return s

    placement = record["wshard"]
    if placement != "0":
        placement += f"(gu={shard_widths[0]},dn={shard_widths[1]})"
    line = f"BENCH tag={tag} M={active_tokens} w={wdtype_name} wshard={placement} " f"fused[{fmt(record['fused'])}]"
    if also_run_old:
        line += f" old[{fmt(record['old'])}]"
    logger.info(line)
    print(line, flush=True)
