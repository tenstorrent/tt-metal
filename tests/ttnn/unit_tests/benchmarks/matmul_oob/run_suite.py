# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Benchmark matmul's default (no program_config) selection over the shape suite in suite.py.

For every case and mode this records the config matmul auto-selected, the on-device time per call (summed
over every program the call launches, so manual transposes and post-processed bias/activation are included;
kernel duration is the headline number, FW and per-RISC kernel durations and the core count are recorded too),
estimated utilization, and PCC against torch. util_pct is against peak math, dram_util_pct against reading/writing
each DRAM tensor once at peak bandwidth, and roofline_pct against whichever of the two is the larger bound. Rows are appended to the CSV as they finish, so an
interrupted run can be continued with --resume (after a hang, reset the device first).

Usage (from the repo root, with python_env active):
  python tests/ttnn/unit_tests/benchmarks/matmul_oob/run_suite.py --out generated/matmul_oob/wh_baseline.csv
  python .../run_suite.py --tiers issues --filter i56976 --modes oob
  python .../run_suite.py --list

Device time comes from the in-process profiler API (TT_METAL_DEVICE_PROFILER etc. are set below), so this
needs a build with the profiler enabled (the default for build_metal.sh).
"""

import argparse
import contextlib
import csv
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

for _var in (
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_PROFILER_MID_RUN_DUMP",
    "TT_METAL_PROFILER_CPP_POST_PROCESS",
    "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES",
):
    os.environ.setdefault(_var, "1")

import torch  # noqa: E402

import ttnn  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from suite import TIERS, cases_from_csv, get_cases  # noqa: E402

DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
# Ideal cycles for one 32x32x32 tile matmul per core, by fidelity
CYCLES_PER_TILE = {"LoFi": 16, "HiFi2": 32, "HiFi3": 48, "HiFi4": 64}
# Nominal AICLK in GHz (actual clock can be lower under throttling, making util_pct conservative)
FREQ_GHZ = {"wormhole_b0": 1.0, "blackhole": 1.35}
# Nominal peak DRAM bandwidth per chip in GB/s (= bytes/ns): n150/n300 chip and p150
DRAM_GBPS = {"wormhole_b0": 288, "blackhole": 512}
TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "bfp4": 576, "fp32": 4096}
# Cases whose L1-resident tensors need more than this fraction of the grid's L1 are skipped as infeasible
L1_BYTES_PER_CORE = 1536 * 1024
L1_FEASIBLE_FRACTION = 0.6
# PCC floor for cases with bfp4 operands (the default --pcc-threshold applies otherwise)
BFP4_PCC_THRESHOLD = 0.97
SHARD_STRATEGIES = {
    "l1_height": ttnn.ShardStrategy.HEIGHT,
    "l1_width": ttnn.ShardStrategy.WIDTH,
    "l1_block": ttnn.ShardStrategy.BLOCK,
}
OUT_SHARDED_MEMS = {
    "l1_height": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    "l1_width": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    "l1_block": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
}
DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
# Extra per-call durations recorded alongside the kernel duration (CSV column -> profiler analysis)
EXTRA_DURATIONS = {
    "fw_ns": "DEVICE FW DURATION [ns]",
    "brisc_ns": "DEVICE BRISC KERNEL DURATION [ns]",
    "ncrisc_ns": "DEVICE NCRISC KERNEL DURATION [ns]",
    "trisc0_ns": "DEVICE TRISC0 KERNEL DURATION [ns]",
    "trisc1_ns": "DEVICE TRISC1 KERNEL DURATION [ns]",
    "trisc2_ns": "DEVICE TRISC2 KERNEL DURATION [ns]",
}
last_auto_config = ttnn._ttnn.operations.matmul.matmul_last_auto_program_config
# v2 mode: whether the selection fell back to the legacy one (inputs v2 doesn't handle yet)
last_auto_fell_back = getattr(ttnn._ttnn.operations.matmul, "matmul_last_auto_config_fell_back", lambda: False)
PCC_SAMPLE = 1 << 24  # max output elements used for PCC


# Selector modes. Each is a context manager active while the case runs; new selectors hook in here.
MODES = {
    "oob": contextlib.nullcontext,
    "v2": lambda: ttnn.manage_config("matmul_auto_config_v2", True),
}

FIELDS = [
    "case",
    "tier",
    "source",
    "tags",
    "mode",
    "status",
    "error",
    "config_type",
    "config",
    "fallback",
    "device_ns",
    "device_ns_min",
    "fw_ns",
    "brisc_ns",
    "ncrisc_ns",
    "trisc0_ns",
    "trisc1_ns",
    "trisc2_ns",
    "cores",
    "programs_per_call",
    "util_pct",
    "dram_util_pct",
    "roofline_pct",
    "bound",
    "tflops",
    "pcc",
    "batch",
    "M",
    "K",
    "N",
    "a_shape",
    "b_shape",
    "a_dtype",
    "b_dtype",
    "out_dtype",
    "a_mem",
    "b_mem",
    "out_mem",
    "a_shard",
    "b_shard",
    "out_shard",
    "transpose_a",
    "transpose_b",
    "op",
    "bias",
    "activation",
    "core_grid",
    "fidelity",
    "fp32_acc",
    "packer_l1_acc",
    "arch",
    "grid",
    "git",
]


def git_rev():
    try:
        root = Path(__file__).resolve().parents[5]
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=root) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def short_error(e):
    msg = str(e).strip()
    # TT_THROW/TT_FATAL messages carry "info:\n<message>\nbacktrace:"; keep just the message
    m = re.search(r"info:\s*\n(.*?)(?:\nbacktrace:|$)", msg, re.S)
    if m:
        msg = m.group(1)
    return " ".join(msg.split())[:300]


SHARD_LAYOUTS = {
    "l1_height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "l1_width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "l1_block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
}


def explicit_sharded_memory_config(mem, shard):
    ranges, shape, orientation = shard
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in ranges]
    )
    spec = ttnn.ShardSpec(
        grid,
        list(shape),
        ttnn.ShardOrientation.COL_MAJOR if orientation == "col" else ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(SHARD_LAYOUTS[mem], ttnn.BufferType.L1, spec)


def shard_fits(shard, grid):
    return shard is None or all(x1 < grid.x and y1 < grid.y for _, _, x1, y1 in shard[0])


def input_memory_config(mem, shape, grid, shard=None):
    if mem == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if mem == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if shard is not None:
        return explicit_sharded_memory_config(mem, shard)
    return ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        strategy=SHARD_STRATEGIES[mem],
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def output_memory_config(mem, shard=None):
    if mem == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if mem == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if shard is not None:
        return explicit_sharded_memory_config(mem, shard)
    return OUT_SHARDED_MEMS[mem]


def pcc(golden, actual):
    g = golden.flatten().double()
    a = actual.flatten().double()
    if g.numel() > PCC_SAMPLE:
        idx = torch.randint(0, g.numel(), (PCC_SAMPLE,), generator=torch.Generator().manual_seed(0))
        g, a = g[idx], a[idx]
    g = g - g.mean()
    a = a - a.mean()
    denom = g.norm() * a.norm()
    if denom == 0:
        return 1.0 if torch.equal(g, a) else 0.0
    return float((g @ a) / denom)


def new_program_entries(device_id, seen):
    """Profiler entries for programs not seen before, ordered by runtime id."""
    data = ttnn.get_all_programs_perf_data().get(device_id, [])
    fresh = [p for p in data if p.program_execution_uid.runtime_id not in seen]
    seen.update(p.program_execution_uid.runtime_id for p in fresh)
    return sorted(fresh, key=lambda p: p.program_execution_uid.runtime_id)


def tensor_bytes(shape, dtype):
    """Size of a tiled tensor (padded to 32x32 tiles)."""
    lead = math.prod(shape[:-2])
    return lead * math.ceil(shape[-2] / 32) * math.ceil(shape[-1] / 32) * TILE_BYTES[dtype]


def placement_bytes(case):
    """(bytes resident in L1, bytes moved to/from DRAM once) for A, B and the output."""
    batch, M, _, N = case.mkn
    sizes = [
        (case.a_mem, tensor_bytes(case.a_shape, case.a_dtype)),
        (case.b_mem, tensor_bytes(case.b_shape, case.b_dtype)),
        (case.out_mem, tensor_bytes((batch, M, N), case.out_dtype or case.a_dtype)),
    ]
    l1 = sum(size for mem, size in sizes if mem != "dram")
    dram = sum(size for mem, size in sizes if mem == "dram")
    return l1, dram


class CaseRun:
    """Input tensors for one case, reusable across modes and explicit program configs.

    Construction raises Infeasible or SetupError; measure() returns a result row.
    """

    class Infeasible(Exception):
        pass

    class SetupError(Exception):
        pass

    def __init__(self, case, device, args, seen_programs):
        self.case, self.device, self.args, self.seen_programs = case, device, args, seen_programs
        self.grid = device.compute_with_storage_grid_size()
        self.arch = str(device.arch()).split(".")[-1].lower()
        self.device_id = device.get_device_ids()[0] if hasattr(device, "get_device_ids") else device.id()
        self.tensors = []
        self.golden = None
        batch, M, K, N = case.mkn

        l1_bytes, self.dram_bytes = placement_bytes(case)
        l1_capacity = L1_FEASIBLE_FRACTION * L1_BYTES_PER_CORE * self.grid.x * self.grid.y
        if not all(shard_fits(s, self.grid) for s in (case.a_shard, case.b_shard, case.out_shard)):
            raise CaseRun.Infeasible("shard grid exceeds the device grid")
        if l1_bytes > l1_capacity:
            raise CaseRun.Infeasible(
                f"L1-resident tensors need {l1_bytes / 2**20:.0f} MiB > {l1_capacity / 2**20:.0f} MiB"
            )
        try:
            torch.manual_seed(0)
            # --configs-only needs shapes only: meta tensors, no host data
            rand = (
                (lambda shape, dtype: torch.empty(shape, dtype=dtype, device="meta"))
                if getattr(args, "configs_only", False)
                else torch.randn
            )
            a_t = rand(case.a_shape, dtype=torch.bfloat16)
            b_t = rand(case.b_shape, dtype=torch.bfloat16) / math.sqrt(K)
            self.a = self._to_device(
                a_t, case.a_dtype, input_memory_config(case.a_mem, case.a_shape, self.grid, case.a_shard)
            )
            self.b = self._to_device(
                b_t, case.b_dtype, input_memory_config(case.b_mem, case.b_shape, self.grid, case.b_shard)
            )
            self.bias = None
            if case.bias:
                bias_t = rand((1, N), dtype=torch.bfloat16)
                self.bias = self._to_device(bias_t, case.b_dtype, ttnn.DRAM_MEMORY_CONFIG)
        except Exception as e:
            self.close()
            raise CaseRun.SetupError(short_error(e)) from e

        self.kwargs = dict(
            transpose_a=case.transpose_a,
            transpose_b=case.transpose_b,
            memory_config=output_memory_config(case.out_mem, case.out_shard),
            dtype=DTYPES[case.out_dtype] if case.out_dtype else None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=FIDELITIES[case.fidelity],
                math_approx_mode=False,
                fp32_dest_acc_en=case.fp32_acc,
                packer_l1_acc=case.packer_l1_acc,
            ),
        )
        if case.core_grid == "device":
            self.kwargs["core_grid"] = ttnn.CoreGrid(y=self.grid.y, x=self.grid.x)
        elif case.core_grid is not None:
            self.kwargs["core_grid"] = ttnn.CoreGrid(x=case.core_grid[0], y=case.core_grid[1])

    def _to_device(self, t, dtype, memory_config):
        if getattr(self.args, "configs_only", False):
            # Only the shapes, dtypes and placements matter: allocate on the device without host data
            tensor = ttnn.empty(
                ttnn.Shape(list(t.shape)),
                dtype=DTYPES[dtype],
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=memory_config,
            )
            self.tensors.append(tensor)
            return tensor
        tensor = ttnn.from_torch(
            t, dtype=DTYPES[dtype], layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=memory_config
        )
        self.tensors.append(tensor)
        return tensor

    def _call(self, program_config):
        kwargs = dict(self.kwargs)
        if program_config is not None:
            kwargs.pop("core_grid", None)  # core_grid and program_config are mutually exclusive
            kwargs["program_config"] = program_config
        if self.case.op == "linear":
            return ttnn.linear(self.a, self.b, bias=self.bias, activation=self.case.activation, **kwargs)
        return ttnn.matmul(self.a, self.b, **kwargs)

    def _poison_output_location(self):
        """Allocate and free a NaN tensor the size of the output in the output's memory, so output tiles the
        kernel never writes read back as NaN (and fail PCC) instead of showing a previous run's result."""
        case = self.case
        if case.out_mem not in ("dram", "l1") and case.out_shard is None:
            return  # a sharded output without a spec gets its shard grid from the program config
        batch, M, _, N = case.mkn
        try:
            t = ttnn.from_torch(
                torch.full((batch, M, N), float("nan"), dtype=torch.bfloat16),
                dtype=DTYPES[case.out_dtype or case.a_dtype],
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=output_memory_config(case.out_mem, case.out_shard),
            )
            ttnn.deallocate(t)
        except Exception:
            pass  # best effort (e.g. no room in L1)

    def measure(self, mode="oob", program_config=None):
        """Run warmup + timed iterations; returns a row with timing, config and PCC."""
        case, args, grid = self.case, self.args, self.grid
        batch, M, K, N = case.mkn
        row = {"status": "ok", "error": ""}
        out = None
        try:
            with MODES[mode]():
                last_auto_config(reset=True)
                out = self._call(program_config)  # compile
                ttnn.synchronize_device(self.device)
                fell_back = program_config is None and last_auto_fell_back()
                config = repr(program_config) if program_config is not None else last_auto_config(reset=True)
                row["fallback"] = int(fell_back)
                row["config"] = config or ""
                row["config_type"] = config.split("(", 1)[0] if config else ""
                if getattr(
                    args, "configs_only", False
                ):  # the chosen config is all that's needed: no timing, profiler read or PCC
                    ttnn.deallocate(out)
                    return row
                for _ in range(args.warmup):
                    ttnn.deallocate(out)  # each call sees the same L1 state (one output at a time)
                    out = self._call(program_config)
                ttnn.synchronize_device(self.device)
                ttnn.ReadDeviceProfiler(self.device)
                new_program_entries(self.device_id, self.seen_programs)  # discard compile/warmup

                for _ in range(args.iters):
                    ttnn.deallocate(out)
                    out = self._call(program_config)
                ttnn.synchronize_device(self.device)
                ttnn.ReadDeviceProfiler(self.device)
                entries = new_program_entries(self.device_id, self.seen_programs)

                # One more call for the PCC check, into a location freshly filled with NaN
                ttnn.deallocate(out)
                self._poison_output_location()
                out = self._call(program_config)
                ttnn.synchronize_device(self.device)
                ttnn.ReadDeviceProfiler(self.device)
                new_program_entries(self.device_id, self.seen_programs)  # not timed
        except Exception as e:
            if out is not None:
                ttnn.deallocate(out)
            return {**row, "status": "error", "error": short_error(e)}

        if not entries or len(entries) % args.iters != 0:
            row.update(status="perf_error", error=f"{len(entries)} profiler entries for {args.iters} iterations")
        else:
            per_call = len(entries) // args.iters
            calls = [entries[i * per_call : (i + 1) * per_call] for i in range(args.iters)]

            def total(call_entries, key):
                results = [p.program_analyses_results.get(key) for p in call_entries]
                return sum(r.duration for r in results if r is not None)

            # Report the median call by kernel duration; the other durations come from that same call
            calls.sort(key=lambda c: total(c, DURATION_KEY))
            median_call = calls[len(calls) // 2]
            device_ns = total(median_call, DURATION_KEY)
            row.update({col: total(median_call, key) for col, key in EXTRA_DURATIONS.items()})
            row["cores"] = max(p.core_count for p in median_call)
            tiles = batch * math.ceil(M / 32) * math.ceil(K / 32) * math.ceil(N / 32)
            ideal_ns = tiles * CYCLES_PER_TILE[case.fidelity] / (grid.x * grid.y) / FREQ_GHZ.get(self.arch, 1.0)
            # DRAM ideal: every DRAM-resident tensor read/written exactly once at peak bandwidth
            dram_ideal_ns = self.dram_bytes / DRAM_GBPS.get(self.arch, DRAM_GBPS["wormhole_b0"])
            row.update(
                device_ns=device_ns,
                device_ns_min=total(calls[0], DURATION_KEY),
                programs_per_call=per_call,
                util_pct=round(100 * ideal_ns / device_ns, 2),
                dram_util_pct=round(100 * dram_ideal_ns / device_ns, 2),
                roofline_pct=round(100 * max(ideal_ns, dram_ideal_ns) / device_ns, 2),
                bound="dram" if dram_ideal_ns > ideal_ns else "math",
                tflops=round(2 * batch * M * K * N / device_ns / 1e3, 3),
            )

        self._check(row, out)
        ttnn.deallocate(out)
        return row

    def _golden(self):
        if self.golden is None:
            case = self.case
            ga = ttnn.to_torch(self.a).float()
            gb = ttnn.to_torch(self.b).float()
            if case.transpose_a:
                ga = ga.transpose(-1, -2)
            if case.transpose_b:
                gb = gb.transpose(-1, -2)
            golden = ga @ gb
            if self.bias is not None:
                golden = golden + ttnn.to_torch(self.bias).float()
            if case.activation == "silu":
                golden = torch.nn.functional.silu(golden)
            elif case.activation == "relu":
                golden = torch.relu(golden)
            elif case.activation == "gelu":
                golden = torch.nn.functional.gelu(golden)
            elif case.activation == "gelu_approx":
                golden = torch.nn.functional.gelu(golden, approximate="tanh")
            self.golden = golden
        return self.golden

    def _check(self, row, out):
        case, args = self.case, self.args
        batch, M, K, N = case.mkn
        row["pcc"] = ""
        if args.pcc_max_flops and 2 * batch * M * K * N > args.pcc_max_flops:
            return
        try:
            golden = self._golden()
            actual = ttnn.to_torch(out).float()
            row["pcc"] = round(pcc(golden, actual.reshape(golden.shape)), 6)
            threshold = BFP4_PCC_THRESHOLD if "bfp4" in (case.a_dtype, case.b_dtype) else args.pcc_threshold
            if row["status"] == "ok" and not row["pcc"] >= threshold:  # NaN (unwritten output) fails too
                row["status"] = "pcc_fail"
        except Exception as e:
            row["error"] = (row["error"] + " | pcc: " + short_error(e)).strip(" |")

    def close(self):
        for t in self.tensors:
            ttnn.deallocate(t)
        self.tensors = []
        self.golden = None


def run_case(case, mode, device, args, seen_programs):
    try:
        run = CaseRun(case, device, args, seen_programs)
    except CaseRun.Infeasible as e:
        return {"status": "infeasible", "error": str(e)}
    except CaseRun.SetupError as e:
        return {"status": "setup_error", "error": str(e)}
    try:
        return run.measure(mode)
    finally:
        run.close()


def shard_str(shard):
    if shard is None:
        return ""
    ranges, shape, orientation = shard
    grid = "+".join(f"({x0},{y0})-({x1},{y1})" for x0, y0, x1, y1 in ranges)
    return f"{grid}:{shape[0]}x{shape[1]}:{orientation}"


def case_fields(case, arch, grid, git):
    batch, M, K, N = case.mkn
    return dict(
        case=case.name,
        tier=case.tier,
        source=case.source,
        tags=";".join(case.tags),
        batch=batch,
        M=M,
        K=K,
        N=N,
        a_shape="x".join(map(str, case.a_shape)),
        b_shape="x".join(map(str, case.b_shape)),
        a_dtype=case.a_dtype,
        b_dtype=case.b_dtype,
        out_dtype=case.out_dtype or "",
        a_mem=case.a_mem,
        b_mem=case.b_mem,
        out_mem=case.out_mem,
        a_shard=shard_str(case.a_shard),
        b_shard=shard_str(case.b_shard),
        out_shard=shard_str(case.out_shard),
        transpose_a=int(case.transpose_a),
        transpose_b=int(case.transpose_b),
        op=case.op,
        bias=int(case.bias),
        activation=case.activation or "",
        core_grid="" if case.core_grid is None else str(case.core_grid),
        fidelity=case.fidelity,
        fp32_acc=int(case.fp32_acc),
        packer_l1_acc=int(case.packer_l1_acc),
        arch=arch,
        grid=f"{grid.x}x{grid.y}",
        git=git,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="generated/matmul_oob/results.csv")
    parser.add_argument("--tiers", nargs="+", choices=list(TIERS), default=None)
    parser.add_argument("--cases-csv", default=None, help="take the cases from an earlier results CSV instead")
    parser.add_argument("--filter", default=None, help="regex on case name")
    parser.add_argument("--exclude-tags", nargs="*", default=[], help="skip cases with any of these tags")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=["oob"])
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="record each case's selected config only (one call per case: no timing, profiler read or PCC)",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pcc-threshold", type=float, default=0.99)
    parser.add_argument(
        "--pcc-max-flops", type=float, default=2e12, help="skip the torch golden above this many flops (0: never)"
    )
    parser.add_argument("--resume", action="store_true", help="skip (case, mode) pairs already in --out")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--list", action="store_true", help="print the selected cases and exit")
    args = parser.parse_args()

    cases = cases_from_csv(args.cases_csv) if args.cases_csv else get_cases(args.tiers)
    if args.cases_csv and args.tiers:
        cases = [c for c in cases if c.tier in args.tiers]
    if args.filter:
        cases = [c for c in cases if re.search(args.filter, c.name)]
    if args.exclude_tags:
        cases = [c for c in cases if not set(c.tags) & set(args.exclude_tags)]
    if args.list:
        for c in cases:
            print(f"{c.name:48s} {c.tier:8s} bMKN={c.mkn} {c.a_dtype}x{c.b_dtype} {c.a_mem}/{c.b_mem}->{c.out_mem}")
        print(f"{len(cases)} cases")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            done = {(r["case"], r["mode"]) for r in csv.DictReader(f)}
    elif out_path.exists():
        sys.exit(f"{out_path} exists; pass --resume to continue it or choose another --out")

    todo = [(c, m) for c in cases for m in args.modes if (c.name, m) not in done]
    print(f"{len(todo)} runs ({len(cases)} cases x {len(args.modes)} modes, {len(done)} already done) -> {out_path}")

    git = git_rev()
    device = ttnn.open_device(device_id=args.device_id)
    arch = str(device.arch()).split(".")[-1].lower()
    grid = device.compute_with_storage_grid_size()
    seen_programs = set()
    write_header = not out_path.exists()
    try:
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            for i, (case, mode) in enumerate(todo):
                t0 = time.time()
                row = run_case(case, mode, device, args, seen_programs)
                row.update(case_fields(case, arch, grid, git), mode=mode)
                writer.writerow(row)
                f.flush()
                device.clear_program_cache()
                us = (
                    f"{row['device_ns'] / 1e3:9.1f}us (fw {row['fw_ns'] / 1e3:9.1f}us)"
                    if row.get("device_ns")
                    else " " * 29
                )
                util = f"{row['roofline_pct']:5.1f}%" if row.get("roofline_pct") is not None else "      "
                print(
                    f"[{i + 1}/{len(todo)}] {case.name:48s} {mode:6s} {row['status']:11s} {us} {util} "
                    f"{row.get('config_type', ''):46s} ({time.time() - t0:.1f}s) {row['error'][:120]}",
                    flush=True,
                )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
