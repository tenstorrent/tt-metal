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
    run_old = _env("BENCH_OLD", "0") == "1"
    do_perf = _env("BENCH_PERF", "1") == "1"
    tag = _env("BENCH_TAG", "run")
    approx = _env("BENCH_APPROX", "0") == "1"
    xscale = float(_env("BENCH_XSCALE", "1.0"))
    wscale = float(_env("BENCH_WSCALE", "0.02"))
    seed = int(_env("BENCH_SEED", "42"))
    grid = _grid()

    torch.manual_seed(seed)
    weights = {
        "gate_proj": torch.randn(hidden, emb, dtype=torch.float32) * wscale,
        "up_proj": torch.randn(hidden, emb, dtype=torch.float32) * wscale,
        "down_proj": torch.randn(emb, hidden, dtype=torch.float32) * wscale,
    }
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

    def to_device(t, dtype, layout):
        return ttnn.from_torch(
            t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    w_gate = to_device(weights["gate_proj"].T, wdtype, ttnn.TILE_LAYOUT)
    w_up = to_device(weights["up_proj"].T, wdtype, ttnn.TILE_LAYOUT)
    w_down = to_device(weights["down_proj"].T, wdtype, ttnn.TILE_LAYOUT)
    x = to_device(torch_input.reshape(1, 1, ALLOCATED_TOKENS, emb), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    def idx_tensor(values):
        return to_device(torch.tensor(values, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    idx = idx_tensor([0])
    counts = idx_tensor([active_tokens])
    offsets = idx_tensor([0])

    def run_fused():
        return ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            x,
            [w_gate],
            [w_up],
            [w_down],
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
            [w_gate],
            [w_up],
            [w_down],
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
    }

    # correctness first (also warms the JIT cache so the timed window is pure dispatch)
    out = ttnn.to_torch(run_fused())[0, 0][:active_tokens]
    record["fused"] = _err_metrics(ref, out)
    if _env("BENCH_SAVE", ""):
        torch.save({"ref": ref, "out": out.float()}, _env("BENCH_SAVE", ""))
    if run_old:
        out_old = ttnn.to_torch(run_old())[0, 0][:active_tokens]
        record["old"] = _err_metrics(ref, out_old)
    ttnn.synchronize_device(device)

    if do_perf:
        med, samples = _median_ns(device, run_fused, "/moe_fused_swiglu/", iters)
        record["fused"]["ns"] = med
        record["fused"]["samples_ns"] = samples
        if run_old:
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

    line = f"BENCH tag={tag} M={active_tokens} w={wdtype_name} fused[{fmt(record['fused'])}]"
    if run_old:
        line += f" old[{fmt(record['old'])}]"
    logger.info(line)
    print(line, flush=True)
