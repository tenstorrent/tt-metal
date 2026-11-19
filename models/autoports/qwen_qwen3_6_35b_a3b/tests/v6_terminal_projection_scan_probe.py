# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generic terminal native projection scan. CPU/source setup by default; hardware opt-in."""

import argparse
import ast
import copy
import json
import math
import os
import statistics
import sys
import time
import traceback
import types
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_ordered_router_fusion_probe as router

E, REPO, REVISION = router.E, router.REPO, router.REVISION
sha, bits = router.sha, router.bits
READER = (
    router.READER.replace(
        "const uint32_t run=get_arg_val<uint32_t>(1);",
        "const uint32_t start=get_arg_val<uint32_t>(1), runs=get_arg_val<uint32_t>(4);",
    )
    .replace(
        "for(uint32_t h=0;h<ht;++h) {",
        "for(uint32_t run=start;run<start+runs;++run) { for(uint32_t h=0;h<ht;++h) {",
    )
    .replace("noc.async_read_barrier();buffer.push_back(1);\n }", "noc.async_read_barrier();buffer.push_back(1);\n }}")
)
COMPUTE = router.COMPUTE.replace(
    "for(uint32_t j=0;j<count;++j) {",
    "const uint32_t runs=get_arg_val<uint32_t>(0);\n for(uint32_t run=0;run<runs;++run) { for(uint32_t j=0;j<count;++j) {",
).replace("tile_regs_release();\n }", "tile_regs_release();\n }}")
WRITER = (
    router.WRITER.replace(
        "const uint32_t id=get_arg_val<uint32_t>(1);",
        "const uint32_t start=get_arg_val<uint32_t>(1), runs=get_arg_val<uint32_t>(2);",
    )
    .replace(
        "final_tile.wait_front(1);row.reserve_back(1);",
        "for(uint32_t id=start;id<start+runs;++id) {\n final_tile.wait_front(1);row.reserve_back(1);",
    )
    .replace(
        "noc.async_write_barrier();row.pop_front(1);final_tile.pop_front(1);",
        "noc.async_write_barrier();row.pop_front(1);final_tile.pop_front(1);\n }",
    )
)


def schedule(total, grid_x, grid_y):
    """Balanced contiguous run partition, with no inactive CB consumers."""
    active = min(total, grid_x * grid_y)
    base, extra = divmod(total, active)
    start, result = 0, []
    for i in range(active):
        count = base + (i < extra)
        result.append((i % grid_x, i // grid_x, start, count))
        start += count
    assert start == total
    return result


class TerminalProjectionScan:
    """Keep native Default cumsum arithmetic and emit one row per leading batch."""

    def __init__(self, ttnn, mesh):
        self.ttnn, self.mesh = ttnn, mesh
        self.grid = mesh.compute_with_storage_grid_size()

    def __call__(self, products):
        ttnn = self.ttnn
        shape = tuple(products.shape)
        assert len(shape) >= 2 and shape[-2] in (512, 1024)
        assert shape[-1] > 0 and shape[-1] % 32 == 0
        assert products.dtype == ttnn.float32 and products.layout == ttnn.TILE_LAYOUT
        assert products.memory_config() in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
        ht, wt = shape[-2] // 32, shape[-1] // 32
        runs = math.prod(shape[:-2]) * wt
        partition = schedule(runs, self.grid.x, self.grid.y)
        cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y, _, _ in partition]
        )
        output = ttnn.empty(
            shape[:-2] + (1, shape[-1]),
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=products.memory_config(),
        )
        rd, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for x, y, start, count in partition:
            rd[x][y] = [products.buffer_address(), start, ht, wt, count]
            wr[x][y] = [output.buffer_address(), start, count]
            cr[x][y] = [count]
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes

        def kernel(source, conf, compile_args, runtime):
            return ttnn.KernelDescriptor(
                kernel_source=source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=compile_args,
                runtime_args=runtime,
            )

        program = ttnn.ProgramDescriptor(
            kernels=[
                kernel(
                    READER, ttnn.ReaderConfigDescriptor(), ttnn.TensorAccessorArgs(products).get_compile_time_args(), rd
                ),
                kernel(COMPUTE, config, [ht], cr),
                kernel(
                    WRITER, ttnn.WriterConfigDescriptor(), ttnn.TensorAccessorArgs(output).get_compile_time_args(), wr
                ),
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.float32, 4096)],
                )
                for index, pages in [(0, 2), (16, 1), (17, 1)]
            ],
        )
        ttnn.generic_op([products, output], program)
        return output


def extract(snapshot):
    tree = ast.parse(snapshot.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    baseline = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "linear")
    candidate = copy.deepcopy(baseline)
    replacements = []
    for branch in candidate.body[:3]:
        assert isinstance(branch, ast.If)
        i = next(
            i
            for i, n in enumerate(branch.body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "prefixes"
        )
        old = branch.body[i : i + 6]
        assert [ast.unparse(n).split(" = ")[0] for n in old] == [
            "prefixes",
            "del products",
            "tail",
            "del prefixes",
            "partial",
            "del tail",
        ]
        assert "disable_compensated_sum=True" in ast.unparse(old[0])
        replacements.append((i, copy.deepcopy(old)))
        branch.body[i : i + 6] = ast.parse("partial = self.terminal_projection_scan(products)\ndel products").body
    restored = copy.deepcopy(candidate)
    for branch, (i, old) in zip(restored.body, replacements):
        branch.body[i : i + 2] = old
    assert ast.dump(restored) == ast.dump(baseline)
    products = copy.deepcopy(baseline)
    products.body = products.body[:3]
    for branch, (i, _) in zip(products.body, replacements):
        branch.body = branch.body[:i] + [ast.Return(ast.Name("products", ast.Load()))]
    products.body += ast.parse("raise AssertionError('products helper requires one selected row')").body
    setup = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_state_dict")
    targets = {
        "order",
        "pair32",
        "self.attention_pair32_indices",
        "state_weight",
        "self.scan_attention_weight",
        "self.scan_output_weight",
        "output_order",
        "self.scan_attention_out_weight",
        "output_pair32",
        "self.output_pair32_indices",
        "router_order",
        "self.scan_router_weight",
        "router_pair32",
        "self.router_pair32_indices",
    }
    assignments = sorted(
        [n for n in ast.walk(setup) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in targets],
        key=lambda n: n.lineno,
    )
    assert len(assignments) == len(targets)
    setup_module = ast.Module(copy.deepcopy(assignments), [])
    return {
        k: ast.unparse(ast.fix_missing_locations(v))
        for k, v in [("baseline", baseline), ("candidate", candidate), ("products", products), ("setup", setup_module)]
    }


def fixtures(checkpoint, random_cases):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for name in [f"linear_attn.in_proj_{s}.weight" for s in ("qkv", "z", "a", "b")] + [
        "linear_attn.out_proj.weight",
        "mlp.gate.weight",
    ]:
        key = "model.language_model.layers.34." + name
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as f:
            state[name] = f.get_tensor(key)
    inputs, captures = {}, []
    for pos in (227, 738, 5799):
        path = E / f"v6_inline_precision_stream_probe.json.pos{pos}.pt"
        cap = torch.load(path, map_location="cpu", weights_only=False)
        assert cap["layer"] == 34 and cap["position"] == pos
        inputs[f"actual_{pos}"] = {
            name: cap["tt_values"][key].bfloat16().clone()
            for name, key in [("router", "post_norm"), ("out", "gated_core"), ("attention", "input_norm")]
        }
        captures.append(dict(path=str(path), sha256=sha(path)))
    g = torch.Generator().manual_seed(603636)
    for i in range(random_cases):
        inputs[f"random_{i}"] = {
            name: (torch.randn((1, 1, k), generator=g) * (2.0 ** (i % 5 - 2))).bfloat16()
            for name, k in [("router", 2048), ("out", 4096), ("attention", 2048)]
        }
    inputs["zero"] = {
        name: torch.zeros((1, 1, k), dtype=torch.bfloat16)
        for name, k in [("router", 2048), ("out", 4096), ("attention", 2048)]
    }
    cases = []
    for name, values in inputs.items():
        for label, key, group, kwargs in [
            ("router", "mlp.gate.weight", "router", dict(ordered_router=True)),
            ("out", "linear_attn.out_proj.weight", "out", dict(ordered_out=True)),
            ("attention_k512", "attention_packed", "attention", dict(projection_k=512)),
            ("output_k512", "output_only_packed", "attention", dict(projection_k=512)),
            ("attention_k1024", "attention_packed", "attention", dict(projection_k=1024)),
            ("output_k1024", "output_only_packed", "attention", dict(projection_k=1024)),
        ]:
            cases.append(dict(name=name + "/" + label, x=values[group], key=key, kwargs=kwargs))
    stacked = torch.cat([inputs[f"actual_{p}"]["attention"] for p in (227, 738, 5799)], dim=1)
    for key in ("attention_packed", "output_only_packed"):
        cases.append(dict(name="stacked_actual_three_row_prefill/" + key, x=stacked, key=key, kwargs={}))
    return state, cases, captures


def run(args):
    torch.set_num_threads(8)
    assert not args.output.exists(), "Preserve previous evidence"
    sources = extract(args.snapshot)
    state, cases, captures = fixtures(args.checkpoint, args.random_cases)
    order = torch.arange(2048).reshape(1024, 2).flip(1).flatten()
    actual_x = cases[0]["x"].flatten()[order].float().reshape(4, 1, 512, 1)
    actual_w = state["mlp.gate.weight"].T[order].float().reshape(4, 1, 512, 256)
    actual_products = actual_x * actual_w
    leading_fixture = torch.stack((actual_products[:3, 0], actual_products[1:4, 0]))
    # CPU traversal proof includes unequal work groups, multiple leading axes, all widths/Ks.
    checks = []
    for shape in [
        (4, 1, 512, 256),
        (8, 1, 512, 2048),
        (4, 1, 512, 6208),
        (4, 1, 512, 6144),
        (2, 1, 1024, 6208),
        (2, 1, 1024, 6144),
        (2, 3, 512, 256),
    ]:
        ht, wt = shape[-2] // 32, shape[-1] // 32
        total = math.prod(shape[:-2]) * wt
        for gx, gy in [(8, 8), (11, 10)]:
            partition = schedule(total, gx, gy)
            runs = [r for _, _, start, count in partition for r in range(start, start + count)]
            assert runs == list(range(total))
            tiles = [(r // wt) * ht * wt + h * wt + r % wt for r in runs for h in range(ht)]
            assert sorted(tiles) == list(range(math.prod(shape) // 1024))
        checks.append(dict(shape=shape, runs=total, input_tiles=len(tiles), traversal_exact=True))
    result = dict(
        status="cpu_source_setup_pass",
        device_executed=False,
        captures=captures,
        classification="component diagnostic; no decoder acceptance or performance claim",
        revision=REVISION,
        script_sha256=sha(__file__),
        frozen_router_sha256=sha(router.__file__),
        snapshot_sha256=sha(args.snapshot),
        snapshot=str(args.snapshot),
        random_seed=603636,
        fixture_cases=len(cases),
        scheduling_checks=checks,
        ast_restoration_exact=True,
        changed_scan_sequences=3,
        source_runtime="native FP32 cumsum; raw final-row relocation; no host conversion",
        limits="CPU setup proves source scope/scheduling only; native SFPU arithmetic and performance require opt-in device checks",
        cases=[],
        timing=[],
    )
    result["dependencies"] = [
        dict(path=str(p), sha256=sha(p))
        for p in [
            REPO
            / "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_tile_axis_compute.cpp",
            REPO / "ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp",
            REPO / "tt_metal/hw/inc/api/compute/cumsum.h",
            REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h",
        ]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, source in dict(reader=READER, compute=COMPUTE, writer=WRITER, **sources).items():
        Path(str(args.output) + f".{label}.snapshot").write_text(source)
    fixture_path = Path(str(args.output) + ".fixtures.pt")
    torch.save(dict(state=state, cases=cases, leading_fixture=leading_fixture), fixture_path)
    result["fixtures"] = dict(path=str(fixture_path), sha256=sha(fixture_path))

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_SOURCE_SETUP_PASS", len(cases))
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
            terminal_projection_scan=TerminalProjectionScan(ttnn, mesh),
            mesh_device=mesh,
            kind="linear_attention",
            decode_linear={},
            w={},
        )
        result["grid"] = [obj.terminal_projection_scan.grid.x, obj.terminal_projection_scan.grid.y]
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
            compile(sources["setup"], str(args.output) + ":setup", "exec"),
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
        for label in ("baseline", "candidate", "products"):
            namespace = dict(ttnn=ttnn, DRAM=ttnn.DRAM_MEMORY_CONFIG)
            exec(compile(sources[label], str(args.output) + ":" + label, "exec"), namespace)
            calls[label] = types.MethodType(namespace["linear"], obj)

        def call(label, dx, case, dtype):
            obj.linear = calls[label]
            with device_only():
                return obj.linear(dx, case["key"], dtype, **case["kwargs"])

        payload = {}
        leading_device = upload(leading_fixture, ttnn.float32)
        native_prefix = ttnn.cumsum(leading_device, -2, dtype=ttnn.float32, disable_compensated_sum=True)
        native_tail = native_prefix[:, :, 480:512, :]
        native_row = native_tail[:, :, 31:32, :]
        with device_only():
            generic_row = obj.terminal_projection_scan(leading_device)
        result["multiple_leading_axes_fp32"] = bits(ttnn.to_torch(native_row), ttnn.to_torch(generic_row))
        save()
        assert result["multiple_leading_axes_fp32"]["exact"]
        del leading_device, native_prefix, native_tail, native_row, generic_row
        for case in cases:
            if args.case_filter and not any(s in case["name"] for s in args.case_filter):
                continue
            dx = upload(case["x"])
            row = dict(name=case["name"], input_shape=list(case["x"].shape), outputs={})
            payload[case["name"]] = {}
            if case["x"].shape[-2] == 1:
                products = call("products", dx, case, ttnn.float32)
                k = products.shape[-2]
                prefixes = ttnn.cumsum(
                    products, -2, dtype=ttnn.float32, disable_compensated_sum=True, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                tail = prefixes[:, :, k - 32 : k, :]
                partial = tail[:, :, 31:32, :]
                with device_only():
                    terminal = obj.terminal_projection_scan(products)
                hn, hf = ttnn.to_torch(partial), ttnn.to_torch(terminal)
                row["partial_fp32"] = bits(hn, hf)
                del products, prefixes, tail, partial, terminal
            for dtype, label in [(ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")]:
                native = call("baseline", dx, case, dtype)
                hn = ttnn.to_torch(native)
                del native
                candidate = call("candidate", dx, case, dtype)
                hf = ttnn.to_torch(candidate)
                del candidate
                row["outputs"][label] = bits(hn, hf)
                payload[case["name"]][label] = dict(native=hn, candidate=hf)
            result["cases"].append(row)
            save()
            assert row.get("partial_fp32", {"exact": True})["exact"] and all(
                v["exact"] for v in row["outputs"].values()
            ), case["name"]
            if args.timing and (case["name"].startswith("actual_227/") or case["name"].startswith("stacked_")):
                for variant in ("baseline", "candidate"):
                    for _ in range(3):
                        out = call(variant, dx, case, ttnn.bfloat16)
                    reference = ttnn.to_torch(out)
                    eager_samples = []
                    for _ in range(5):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        out = call(variant, dx, case, ttnn.bfloat16)
                        ttnn.synchronize_device(mesh)
                        eager_samples.append((time.perf_counter() - start) * 1000)
                    assert bits(reference, ttnn.to_torch(out))["exact"]
                    out = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    out = call(variant, dx, case, ttnn.bfloat16)
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
                        repeated = bits(reference, ttnn.to_torch(out))
                        assert repeated["exact"]
                        result["timing"].append(
                            dict(
                                name=case["name"],
                                variant=variant,
                                median_ms=statistics.median(samples),
                                samples_ms=samples,
                                warmed_eager_median_ms=statistics.median(eager_samples),
                                warmed_eager_samples_ms=eager_samples,
                                after_100_replays=repeated,
                            )
                        )
                    finally:
                        ttnn.release_trace(mesh, trace)
                    del out
                save()
        assert result["cases"], "No cases selected"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=E / "v6_inline_router_silu_chunk64_output_mkl_explicit_norm_v2.py.snapshot"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-cases", type=int, default=1)
    parser.add_argument(
        "--case-filter", nargs="+", help="Optional case-name substring selection; recorded cases show executed scope"
    )
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--timing", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
