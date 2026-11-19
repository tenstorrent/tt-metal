# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Preserve native reduction/typecast and fold final DRAM placement into typecast. Hardware opt-in."""

import argparse
import ast
import copy
import json
import os
import statistics
import sys
import time
import traceback
import types
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_streaming_projection_scan_probe as stream

E, REPO, REVISION = stream.E, stream.REPO, stream.REVISION
sha, bits = stream.sha, stream.bits
BASE = E / "v6_corrected_fma_full_v5_streaming_scan_norm_tail_integration.py.snapshot"
BASE_SHA = "700c10db90b23fac1d40427c9bbcd82a6924c7704699d54768a0160c2b641e05"
SETUP_SOURCE = E / "v6_inline_router_silu_chunk64_output_mkl_explicit_norm_v2.py.snapshot"


def sources():
    assert sha(BASE) == BASE_SHA
    tree = ast.parse(BASE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    baseline = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "linear")
    candidate = copy.deepcopy(baseline)
    for branch in candidate.body[1:3]:
        old = branch.body[-1]
        assert isinstance(old, ast.Return) and ast.unparse(old.value.func) == "ttnn.to_memory_config"
        assert ast.unparse(old.value.args[1]) == "DRAM"
        direct = ast.Return(copy.deepcopy(old.value.args[0]))
        assert ast.unparse(direct.value.func) == "ttnn.reshape"
        cast = direct.value.args[0]
        assert ast.unparse(cast.func) == "ttnn.typecast" and not cast.keywords
        cast.keywords.append(ast.keyword("memory_config", ast.Name("DRAM", ast.Load())))
        branch.body[-1:] = [ast.If(ast.Name("terminal_scan", ast.Load()), [direct], []), old]
    restored = copy.deepcopy(candidate)
    for branch in restored.body[1:3]:
        assert isinstance(branch.body[-2], ast.If) and ast.unparse(branch.body[-2].test) == "terminal_scan"
        del branch.body[-2]
    assert ast.dump(restored) == ast.dump(baseline)
    assert ast.dump(candidate.body[0]) == ast.dump(baseline.body[0])
    setup = stream.extract(SETUP_SOURCE)["setup"]
    return dict(
        baseline=ast.unparse(baseline), candidate=ast.unparse(ast.fix_missing_locations(candidate)), setup=setup
    )


def rounding_fixture():
    # Exact FP32 values immediately below/at/above BF16 half-way boundaries.
    codes = []
    for sign in (0, 0x80000000):
        for mantissa in range(42):
            midpoint = sign | (127 << 23) | (mantissa << 16) | 0x8000
            codes.extend([midpoint - 1, midpoint, midpoint + 1])
    codes += [0, 0x80000000, 0x3F800000, 0xBF800000]
    assert len(codes) == 256
    values = torch.tensor(codes, dtype=torch.int64).int().view(torch.float32)
    partial = torch.zeros((4, 1, 1, 256), dtype=torch.float32)
    partial[0, 0, 0] = values
    return partial


def run(args):
    torch.set_num_threads(8)
    assert not args.output.exists(), "Preserve earlier evidence"
    code = sources()
    state, cases, captures = stream.fixtures(args.checkpoint, args.random_cases)
    for case in cases:
        if case["x"].shape[-2] == 1:
            case["kwargs"]["terminal_scan"] = True
    rounding = rounding_fixture()
    result = dict(
        status="cpu_source_setup_pass",
        device_executed=False,
        classification="native typecast final-placement component; integrated performance unmeasured",
        script_sha256=sha(__file__),
        base=str(BASE),
        base_sha256=BASE_SHA,
        streaming_helper=str(Path(stream.__file__)),
        streaming_helper_sha256=sha(stream.__file__),
        setup_snapshot=str(SETUP_SOURCE),
        setup_snapshot_sha256=sha(SETUP_SOURCE),
        captures=captures,
        fixture_cases=len(cases),
        seed=603636,
        full_linear_ast_restoration=True,
        full_attention_guard_unchanged=True,
        default_prefill_tail_unchanged=True,
        changed_arithmetic="None: native FP32 reduction and explicit typecast retained; typecast output placement becomes DRAM for router/out",
        native_order="Direct FP32 unpack, copy block0 to dst0, then add_binary_tile(dst0,dst1,dst0) in ascending block order",
        precision_limit="Same native typecast integer RNE; placement change still requires exact device checks",
        cases=[],
        timing=[],
    )
    native_dir = REPO / "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device"
    dependencies = [
        native_dir / name
        for name in [
            "fast_reduce_nc_program_factory.cpp",
            "fast_reduce_nc_device_operation.cpp",
            "kernels/reduce_nc.cpp",
            "kernels/reader_reduce_nc.cpp",
        ]
    ]
    dependencies += [
        REPO / "ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp",
        REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h",
        REPO / "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp",
    ]
    result["dependencies"] = [dict(path=str(p), sha256=sha(p)) for p in dependencies]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, text in code.items():
        Path(str(args.output) + f".{name}.snapshot").write_text(text)
    fixture_path = Path(str(args.output) + ".fixtures.pt")
    torch.save(dict(state=state, cases=cases, rounding=rounding), fixture_path)
    result["fixtures"] = dict(path=str(fixture_path), sha256=sha(fixture_path))

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_SOURCE_SETUP_PASS", len(cases), "256 rounding lanes")
        return
    assert os.environ.get("TT_METAL_DEVICE_PROFILER") in (None, "", "0")
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=67108864
        )
        result.update(status="running", device_executed=True, device_closed=False)
        obj = types.SimpleNamespace(
            streaming_projection_scan=stream.StreamingProjectionScan(ttnn, mesh),
            mesh_device=mesh,
            kind="linear_attention",
            decode_linear={},
            w={},
        )
        obj.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        exec(
            compile(code["setup"], str(args.output) + ":setup", "exec"),
            dict(
                self=obj,
                torch=torch,
                ttnn=ttnn,
                mesh_device=mesh,
                upload=upload,
                DRAM=ttnn.DRAM_MEMORY_CONFIG,
                state_dict=state,
                pieces=[state[f"linear_attn.in_proj_{s}.weight"] for s in ("qkv", "z", "a", "b")],
            ),
        )
        calls = {}
        for label in ("baseline", "candidate"):
            ns = dict(ttnn=ttnn, DRAM=ttnn.DRAM_MEMORY_CONFIG)
            exec(compile(code[label], str(args.output) + ":" + label, "exec"), ns)
            calls[label] = types.MethodType(ns["linear"], obj)

        def call(label, dx, case, dtype):
            obj.linear = calls[label]
            with device_only():
                return obj.linear(dx, case["key"], dtype, **case["kwargs"])

        dp = upload(rounding, ttnn.float32)
        original = ttnn.empty(
            (1, 1, 1, 256),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        original = ttnn.experimental.fast_reduce_nc(
            dp, dims=[0], output=original, compute_kernel_config=obj.compute, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        expected = ttnn.typecast(original, ttnn.bfloat16)
        packed = ttnn.typecast(original, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert packed.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        torch.save(
            dict(
                expected=ttnn.to_torch(expected), packed=ttnn.to_torch(packed), reduction_fp32=ttnn.to_torch(original)
            ),
            Path(str(args.output) + ".rounding.pt"),
        )
        result["rounding_midpoints_bf16"] = bits(ttnn.to_torch(expected), ttnn.to_torch(packed))
        save()
        assert result["rounding_midpoints_bf16"]["exact"]
        del dp, original, expected, packed
        payload = {}
        for case in cases:
            if args.case_filter and not any(s in case["name"] for s in args.case_filter):
                continue
            dx = upload(case["x"])
            row = dict(name=case["name"], shape=list(case["x"].shape), outputs={})
            payload[case["name"]] = {}
            for dtype, label in [(ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")]:
                old = call("baseline", dx, case, dtype)
                hn = ttnn.to_torch(old)
                placement = old.memory_config()
                del old
                new = call("candidate", dx, case, dtype)
                assert new.memory_config() == placement
                hf = ttnn.to_torch(new)
                del new
                row["outputs"][label] = bits(hn, hf)
                payload[case["name"]][label] = dict(native=hn, packed=hf)
            result["cases"].append(row)
            save()
            assert all(v["exact"] for v in row["outputs"].values()), case["name"]
            if args.timing and case["name"] in ("actual_227/router", "actual_227/out"):
                for variant in ("baseline", "candidate"):
                    for _ in range(3):
                        output = call(variant, dx, case, ttnn.bfloat16)
                    reference = ttnn.to_torch(output)
                    eager = []
                    for _ in range(5):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        output = call(variant, dx, case, ttnn.bfloat16)
                        ttnn.synchronize_device(mesh)
                        eager.append((time.perf_counter() - start) * 1000)
                    output = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    output = call(variant, dx, case, ttnn.bfloat16)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    try:
                        samples = []
                        for _ in range(10):
                            ttnn.synchronize_device(mesh)
                            start = time.perf_counter()
                            for _ in range(10):
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            samples.append((time.perf_counter() - start) * 1000 / 10)
                        repeated = bits(reference, ttnn.to_torch(output))
                        assert repeated["exact"]
                        result["timing"].append(
                            dict(
                                name=case["name"],
                                variant=variant,
                                median_ms=statistics.median(samples),
                                samples_ms=samples,
                                warmed_eager_median_ms=statistics.median(eager),
                                warmed_eager_samples_ms=eager,
                                after_100_replays=repeated,
                            )
                        )
                    finally:
                        ttnn.release_trace(mesh, trace)
                    del output
                save()
        assert result["cases"]
        torch.save(payload, Path(str(args.output) + ".device.pt"))
        result["status"] = "component_pass"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--random-cases", type=int, default=1)
    p.add_argument("--case-filter", nargs="+")
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--timing", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
