# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact weighted-tail interface and complete compact-SiLU adapter diagnostic.

CPU preparation by default; --run-device is parent-owned. No production edits.
"""

import argparse
import ast
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_gated_norm_tail_fusion_probe as norm
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_gated_norm_tail_timing_probe as timing
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_silu_compact_lookup_probe as local
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

E = norm.E
LOCAL_SHA = "c546d954f88308658cf4af2b0f9ff481397e583b8d5e86e3cd30899d0a7b271c"
TIMING_SHA = "c0af8685423e6688a3362aa04a99c6db4d66999bebe1a5f9a890dc000f75994c"


def weighted_kernel_source():
    original = norm.kernel_source()
    edits = [
        ("gamma(2), gate(3), output(16)", "gamma(2), output(16)"),
        (" gate.reserve_back(4); gate.push_back(4);\n", ""),
        (" gamma.wait_front(4); gate.wait_front(4);", " gamma.wait_front(4);"),
        (" copy_tile(3,tile,3);", ""),
        ("   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::vFloat g=sfpi::dst_reg[96]; sfpi::dst_reg[0]=x*g; }\n", ""),
        ("   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[0]=round_bf16(x); }\n", ""),
        (" gamma.pop_front(4); gate.pop_front(4);", " gamma.pop_front(4);"),
    ]
    result = original
    for before, after in edits:
        assert result.count(before) == 1, before
        result = result.replace(before, after)
    # The entire numerical prefix through weighted BF16 rounding is identical.
    prefix_start = "  _llk_math_eltwise_sfpu_start_(1);"
    prefix_end = "   { sfpi::vFloat x=sfpi::dst_reg[0]; x=round_bf16(x); sfpi::dst_reg[0]=x; if constexpr(debug) sfpi::dst_reg[160]=x; }"
    assert original.split(prefix_start)[1].split(prefix_end)[0] == result.split(prefix_start)[1].split(prefix_end)[0]
    assert "gate(" not in result and "dst_reg[96]" not in result
    return result


WEIGHTED_NORM_COMPUTE_SOURCE = weighted_kernel_source()


class WeightedNormTail:
    """Preserve explicit post-sum arithmetic, returning BF16-rounded FP32 weights."""

    def __init__(self, t, mesh, gamma, debug=False):
        self.t, self.mesh, self.debug = t, mesh, debug
        self.cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(0, 0), t.CoreCoord(0, 0))])
        self.memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 128), t.ShardOrientation.ROW_MAJOR),
        )
        self.sum_memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 32), t.ShardOrientation.ROW_MAJOR),
        )
        expanded_gamma = gamma.float().reshape(1, 1, 128).expand(1, 32, 128).contiguous()
        self.gamma = t.from_torch(
            expanded_gamma, dtype=t.float32, layout=t.TILE_LAYOUT, device=mesh, memory_config=self.memory
        )
        self.config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        self.config.unpack_to_dest_mode = modes
        self.source = WEIGHTED_NORM_COMPUTE_SOURCE

    def __call__(self, core, sums):
        t = self.t
        assert tuple(core.shape) == (1, 32, 128) and tuple(sums.shape) == (1, 32, 1)
        assert core.dtype == sums.dtype == t.float32
        core = t.to_memory_config(core, self.memory)
        sums = t.to_memory_config(sums, self.sum_memory)
        output = t.empty(
            (1, 32, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.memory
        )
        bindings = [(0, core), (1, sums), (2, self.gamma), (16, output)]
        debug_outputs = {}
        if self.debug:
            for index, name in ((17, "normalized_fp32"), (18, "weighted_fp32_carrier"), (19, "inverse_repeated")):
                value = t.empty(
                    (1, 32, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.memory
                )
                bindings.append((index, value))
                debug_outputs[name] = value
        descriptor = t.KernelDescriptor(
            kernel_source=self.source,
            source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self.cores,
            compile_time_args=[int(self.debug)],
            runtime_args=[],
            config=self.config,
        )
        program = t.ProgramDescriptor(
            kernels=[descriptor],
            semaphores=[],
            cbs=[t.cb_descriptor_from_sharded_tensor(i, value) for i, value in bindings],
        )
        t.generic_op([value for _, value in bindings], program)
        return output, debug_outputs


class CombinedAdapter:
    def __init__(self, t, mesh, gamma):
        self.t = t
        self.compute = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.weighted = WeightedNormTail(t, mesh, gamma, debug=False)
        self.lookup = local.LocalSiluMultiply(t, mesh, debug=False)

    def prepare_core(self, core):
        t = self.t
        core_bf16 = t.typecast(core, t.bfloat16)
        core_rows = t.reshape(core_bf16, (1, 32, 128))
        wide = t.typecast(core_rows, t.float32, memory_config=t.L1_MEMORY_CONFIG)
        squares = t.multiply(
            wide, wide, dtype=t.float32, fast_and_approximate_mode=False, memory_config=t.L1_MEMORY_CONFIG
        )
        sums = t.sum(
            squares, dim=-1, keepdim=True, compute_kernel_config=self.compute, memory_config=t.L1_MEMORY_CONFIG
        )
        return wide, sums

    def __call__(self, core, z):
        wide, sums = self.prepare_core(core)
        weighted, _ = self.weighted(wide, sums)
        merged, _ = self.lookup(weighted, z)
        return merged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--time", action="store_true")
    p.add_argument("--random-cases", type=int, default=8)
    p.add_argument("--position", type=int, default=3271)
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trace-region-size", type=int, default=16777216)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / norm.REVISION,
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists(), "Preserve earlier evidence"
    assert args.replays >= 100 and args.samples > 0 and args.warmup > 0
    assert verify_source(local.__file__, LOCAL_SHA) and verify_source(timing.__file__, TIMING_SHA)
    torch.set_num_threads(4)
    gamma, fixtures, timing_fixture, old_result = timing.prepare(args)
    result = dict(
        classification="Weighted FP32 interface correctness and complete normalization/compact-SiLU adapter diagnostic",
        acceptance_eligible=False,
        oracle_assisted=False,
        status="cpu_setup_pass",
        device_closed=False,
        source_sha256=norm.sha(__file__),
        local_source_sha256=raw_sha256(local.__file__),
        local_source_historical_sha256=LOCAL_SHA,
        norm_source_sha256=raw_sha256(norm.__file__),
        norm_source_historical_sha256=timing.EXPECTED_HELPER_SHA256,
        timing_source_sha256=raw_sha256(timing.__file__),
        timing_source_historical_sha256=TIMING_SHA,
        kernel_sha256=hashlib.sha256(WEIGHTED_NORM_COMPUTE_SOURCE.encode()).hexdigest(),
        weighted_arithmetic_prefix_identical=True,
        timing_scope=old_result["timing_scope"],
        fixture_names=[f["name"] for f in fixtures],
        correctness=[],
        timing=[],
        placement="weighted output remains single-core HEIGHT_SHARDED FP32; local lookup TensorAccessor reads that layout directly",
        command=sys.argv,
    )
    tree = ast.parse(Path(__file__).read_text())
    for class_name in ("WeightedNormTail", "CombinedAdapter"):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        for method in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != "__init__"]:
            calls = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
            assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in calls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, content in (
        ("source.py.snapshot", Path(__file__).read_text()),
        ("compute.cpp.snapshot", WEIGHTED_NORM_COMPUTE_SOURCE),
    ):
        Path(str(args.output) + "." + suffix).write_text(content)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_WEIGHTED_COMPACT_SILU_SETUP_PASS", len(fixtures))
        return
    if args.time:
        for variable in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
            assert os.environ.get(variable) in (None, "", "0")
    from ttnn.tools import trace_allocation_tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(args.output)
    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=args.trace_region_size
        )
        baseline = timing.CompleteAdapter(t, mesh, gamma)
        candidate = CombinedAdapter(t, mesh, gamma)
        frozen_debug = norm.GatedNormTail(t, mesh, gamma, debug=True)
        weighted_debug = WeightedNormTail(t, mesh, gamma, debug=True)

        def upload(fixture):
            return (
                t.from_torch(
                    fixture["core_input"],
                    dtype=t.float32,
                    layout=t.TILE_LAYOUT,
                    device=mesh,
                    memory_config=t.L1_MEMORY_CONFIG,
                ),
                t.from_torch(
                    fixture["z_input"],
                    dtype=t.bfloat16,
                    layout=t.TILE_LAYOUT,
                    device=mesh,
                    memory_config=t.DRAM_MEMORY_CONFIG,
                ),
            )

        tensors = {}
        for fixture in fixtures:
            core, z = upload(fixture)
            with device_only():
                wide, sums = candidate.prepare_core(core)
                gate = norm.device_lookup(t, z, baseline.table)
                frozen_output, frozen = frozen_debug(wide, sums, gate)
                weighted_output, weighted = weighted_debug(wide, sums)
                expected = baseline(core, z, "explicit")
                actual = candidate(core, z)
            host = {k: t.to_torch(v).clone() for k, v in weighted.items()}
            old = {k: t.to_torch(v).clone() for k, v in frozen.items()}
            host["weighted_output"] = t.to_torch(weighted_output).clone()
            checks = {k: local.compare(old[k], host[k]) for k in old}
            checks["weighted_output"] = local.compare(old["weighted_fp32_carrier"], host["weighted_output"])
            checks["complete_adapter"] = local.compare(t.to_torch(expected), t.to_torch(actual))
            result["correctness"].append(dict(name=fixture["name"], checks=checks))
            tensors[fixture["name"]] = dict(candidate=host, frozen=old)
            save()
            print(fixture["name"], {k: v["bit_differences"] for k, v in checks.items()}, flush=True)
            assert all(check["bit_differences"] == 0 for check in checks.values()), (fixture["name"], checks)
            del core, z, wide, sums, gate, frozen_output, frozen, weighted_output, weighted, expected, actual
        tensor_path = Path(str(args.output) + ".tensors.pt")
        torch.save(tensors, tensor_path)
        result["tensor_artifact"] = dict(path=str(tensor_path), sha256=norm.sha(tensor_path))
        result["weighted_and_complete_adapter_bitexact"] = True
        del frozen_debug, weighted_debug
        if args.time:
            core, z = upload(timing_fixture)
            reference = t.to_torch(baseline(core, z, "explicit")).clone()
            for name, operation in (
                ("explicit", lambda c, z: baseline(c, z, "explicit")),
                ("fused_tail", lambda c, z: baseline(c, z, "fused_tail")),
                ("weighted_compact", candidate),
            ):
                for _ in range(args.warmup):
                    with device_only():
                        output = operation(core, z)
                    t.synchronize_device(mesh)
                    del output
                trace_id, capturing = None, False
                try:
                    trace_id = t.begin_trace_capture(mesh, cq_id=0)
                    capturing = True
                    with device_only():
                        output = operation(core, z)
                    t.end_trace_capture(mesh, trace_id, cq_id=0)
                    capturing = False
                    trace_allocation_tracker.acknowledge_corruptible(output)
                    for _ in range(args.warmup):
                        t.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    first = t.to_torch(output).clone()
                    assert local.compare(reference, first)["bit_differences"] == 0
                    samples = []
                    for _ in range(args.samples):
                        t.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            t.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                        t.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = t.to_torch(output).clone()
                    assert local.compare(first, last)["bit_differences"] == 0
                    result["timing"].append(
                        dict(
                            variant=name,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            replays_per_sample=args.replays,
                            first_vs_explicit=local.compare(reference, first),
                            first_vs_last=local.compare(first, last),
                            output_memory_config=str(output.memory_config()),
                        )
                    )
                    save()
                    print(name, statistics.median(samples), "ms", flush=True)
                finally:
                    if trace_id is not None:
                        if capturing:
                            t.end_trace_capture(mesh, trace_id, cq_id=0)
                        t.synchronize_device(mesh)
                        t.release_trace(mesh, trace_id)
                    del output
        result["status"] = "completed_diagnostic"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


if __name__ == "__main__":
    main()
