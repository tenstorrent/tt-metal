# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host source proof; opt-in exact complete-adapter checks/timing for transport cleanup.

No device opens unless --run-device. Default imports neither Torch nor TTNN.
This is component evidence; full decoder endurance and performance are separate.
"""

import argparse
import ast
import copy
import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_build_transport_cleanup as builder

REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def source_check():
    result = {}
    for variant in ("placement", "indices", "combined"):
        expected, proof = builder.build(variant)
        path = builder.E / f"v6_ab28_transport_{variant}.py.snapshot"
        assert path.read_text() == expected
        result[variant] = dict(path=str(path), sha256=builder.sha(path), proof=proof)
    return result


def load_snapshot(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def adapter_source(path, kind):
    """Extract the real boundary statements, with no replacement arithmetic."""
    tree = ast.parse(Path(path).read_text())
    cls = builder.named(tree, "FusedDecoder")
    if kind == "norm":
        fn = builder.named(cls, "_forward")
        start = next(
            i
            for i, node in enumerate(fn.body)
            if i > 0 and isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "mlp_input"
        )
        body = copy.deepcopy(fn.body[start:-1])
        body += ast.parse("return mlp_input").body
        signature = "def adapter(self, x, decode):\n    pass"
    else:
        fn = builder.named(cls, "_moe")
        selection = next(
            node
            for node in fn.body[1:]
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "s == 1"
            and isinstance(node.body[0], ast.Assign)
            and ast.unparse(node.body[0].value) == "self.inline_selector(logits)"
        )
        body = copy.deepcopy(selection.body)
        index_consumer = next(
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "ids"
            and ast.unparse(node.value) == "ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, 8))"
        )
        body += [copy.deepcopy(index_consumer)] + ast.parse("return scores, ids").body
        signature = "def adapter(self, logits):\n    pass"
    function = ast.parse(signature).body[0]
    function.body = body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    return module


def run_device(args, report, save):
    import torch
    from safetensors import safe_open
    from ttnn.tools import trace_allocation_tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    torch.set_num_threads(4)
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        if args.time:
            assert os.environ.get(key) in (None, "", "0"), "Timing must be separate from watcher/profiler"
    index_path = args.checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    key = f"model.language_model.layers.{args.layer}.post_attention_layernorm.weight"
    assert key in index
    weight_path = args.checkpoint / index[key]
    with safe_open(weight_path, framework="pt", device="cpu") as source:
        gamma = (source.get_tensor(key).float() + 1).reshape(1, 1, 2048)
    torch.manual_seed(154)
    fixtures = []
    for length in args.lengths:
        assert length > 0
        fixtures.append((f"prefill_s{length}", torch.randn(1, length, 2048).bfloat16(), False))
    for number, scale in enumerate((0.0, 0.01, 1.0, 10.0)):
        fixtures.append((f"decode{number}", (torch.randn(1, 1, 2048) * scale).bfloat16(), True))
    logit_fixtures = [torch.randn(1, 256).bfloat16() for _ in range(4)]
    logit_fixtures += [
        torch.zeros(1, 256, dtype=torch.bfloat16),
        (torch.arange(256).reshape(1, 256) % 8).bfloat16(),
        torch.arange(256).reshape(1, 256).bfloat16(),
    ]
    input_path = Path(str(args.output) + ".inputs.pt")
    assert not input_path.exists()
    torch.save(dict(gamma=gamma, norm=fixtures, logits=logit_fixtures), input_path)
    report.update(
        checkpoint_revision=REVISION,
        checkpoint_weight_key=key,
        checkpoint_index_sha256=builder.sha(index_path),
        inputs=str(input_path),
        inputs_sha256=builder.sha(input_path),
        fixture_scope="Actual tensor shapes and checkpoint gamma; synthetic seeded operands, not captured model states",
    )
    modules = {
        "base": load_snapshot(builder.BASE, "_transport_base"),
        "candidate": load_snapshot(Path(report["sources"]["combined"]["path"]), "_transport_candidate"),
    }
    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=args.trace_region_size
        )
        compute = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        cores = t.num_cores_to_corerangeset(32, mesh.compute_with_storage_grid_size(), row_wise=True)
        shard = t.create_sharded_memory_config(
            (32, 64),
            cores,
            t.ShardStrategy.WIDTH,
            t.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        def upload(value, dtype):
            return t.from_torch(
                value.contiguous(), device=mesh, dtype=dtype, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
            )

        weight = upload(gamma, t.float32)
        environment = {}
        adapters = {}
        for variant, module in modules.items():
            obj = SimpleNamespace(
                kind="linear_attention",
                mesh_device=mesh,
                compute=compute,
                config=SimpleNamespace(rms_norm_eps=1e-6),
                norm_memory=shard,
                w={"post_attention_layernorm.weight": weight},
                round_widen_chain=[
                    t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value)),
                    t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value)),
                ],
                inline_selector=module.InlineSelector(t, mesh, module.PORTABLE_CPP),
            )
            obj.norm = lambda x, key, obj=obj, module=module: module.FusedDecoder.norm(obj, x, key)
            environment[variant] = obj
            path = builder.BASE if variant == "base" else Path(report["sources"]["combined"]["path"])
            for kind in ("norm", "selector"):
                scope = dict(module.__dict__)
                exec(compile(adapter_source(path, kind), f"<exact-{variant}-{kind}-boundary>", "exec"), scope)
                adapters[variant, kind] = scope["adapter"]

        def run(variant, kind, tensor, decode=False):
            with device_only():
                if kind == "norm":
                    outputs = (adapters[variant, kind](environment[variant], tensor, decode),)
                else:
                    outputs = adapters[variant, kind](environment[variant], tensor)
            for output in outputs:
                assert output.memory_config() == t.DRAM_MEMORY_CONFIG
            if kind == "selector":
                assert outputs[1].layout == t.ROW_MAJOR_LAYOUT and tuple(outputs[1].shape) == (1, 1, 1, 8)
                assert outputs[1].dtype == t.uint16
            return outputs

        def read(outputs):
            return [t.to_torch(value).contiguous().clone() for value in outputs]

        def differences(left, right):
            assert len(left) == len(right)
            counts = []
            for a, b in zip(left, right):
                assert a.shape == b.shape and a.dtype == b.dtype
                counts.append(int((a.view(torch.int16) != b.view(torch.int16)).sum()))
            return counts

        timing_inputs = []
        for name, cpu, decode in fixtures:
            tensor = upload(cpu, t.bfloat16)
            left = read(run("base", "norm", tensor, decode))
            right = read(run("candidate", "norm", tensor, decode))
            delta = differences(left, right)
            report["correctness"].append(dict(kind="norm", name=name, differences=delta, decode=decode))
            save()
            assert not any(delta), (name, delta)
            if name == "decode2":
                timing_inputs.append(("norm", tensor, decode, left))
            else:
                del tensor
        for number, cpu in enumerate(logit_fixtures):
            tensor = upload(cpu, t.bfloat16)
            left = read(run("base", "selector", tensor))
            right = read(run("candidate", "selector", tensor))
            delta = differences(left, right)
            report["correctness"].append(dict(kind="selector", name=f"logits{number}", differences=delta))
            save()
            assert not any(delta), (number, delta)
            if number == 0:
                timing_inputs.append(("selector", tensor, False, left))
            else:
                del tensor
        if args.time:
            for kind, tensor, decode, reference in timing_inputs:
                for variant in ("base", "candidate"):
                    for _ in range(args.warmup):
                        output = run(variant, kind, tensor, decode)
                        t.synchronize_device(mesh)
                        del output
                    eager = []
                    for _ in range(args.samples):
                        t.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.eager_repeats):
                            output = run(variant, kind, tensor, decode)
                            t.synchronize_device(mesh)
                            del output
                        eager.append((time.perf_counter() - start) * 1000 / args.eager_repeats)
                    trace = t.begin_trace_capture(mesh, cq_id=0)
                    output = run(variant, kind, tensor, decode)
                    t.end_trace_capture(mesh, trace, cq_id=0)
                    try:
                        for value in output:
                            trace_allocation_tracker.acknowledge_corruptible(value)
                        for _ in range(args.warmup):
                            t.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        traced = []
                        for _ in range(args.samples):
                            t.synchronize_device(mesh)
                            start = time.perf_counter()
                            for _ in range(args.replays):
                                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            t.synchronize_device(mesh)
                            traced.append((time.perf_counter() - start) * 1000 / args.replays)
                            assert not any(differences(reference, read(output)))
                        report["timing"].append(
                            dict(
                                kind=kind,
                                variant=variant,
                                eager_ms=eager,
                                traced_ms=traced,
                                eager_median_ms=statistics.median(eager),
                                traced_median_ms=statistics.median(traced),
                                checked_replays=args.samples * args.replays,
                            )
                        )
                        save()
                    finally:
                        t.release_trace(mesh, trace)
                        del output
        report["status"] = "completed_exact_component"
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            report["device_closed"] = True
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 33, 65, 127])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--eager-repeats", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--trace-region-size", type=int, default=16777216)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert args.samples > 0 and args.warmup > 0 and args.eager_repeats > 0 and args.replays >= 100
    sources = source_check()
    for kind in ("norm", "selector"):
        compile(adapter_source(builder.BASE, kind), "<base-adapter-check>", "exec")
        compile(adapter_source(sources["combined"]["path"], kind), "<candidate-adapter-check>", "exec")
    report = dict(
        status="source_setup_pass",
        acceptance_eligible=False,
        device_closed=False,
        script_sha256=builder.sha(__file__),
        builder_sha256=builder.sha(builder.__file__),
        base_sha256=builder.BASE_SHA,
        sources=sources,
        command=sys.argv,
        correctness=[],
        timing=[],
        norm_timing_scope="BF16 DRAM residual [1,1,2048] through exact postnorm and all placement requests to BF16 DRAM output; excludes residual add and later gate projection",
        selector_timing_scope="BF16 DRAM logits [1,256] through unchanged selector, FP32 stable softmax, BF16 scores and actual score/index reshapes to scores plus UINT16 RM single-stick ids; excludes sparse matmul",
        limitation="Synthetic actual-shape components cannot replace full-model fresh endurance, context, or complete decoder latency gates",
    )
    snapshot = Path(str(args.output) + ".source.py.snapshot")
    assert not snapshot.exists()
    snapshot.write_bytes(Path(__file__).read_bytes())
    report["source_snapshot"] = str(snapshot)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules and "torch" not in sys.modules
        print("SOURCE_SETUP_PASS")
        return
    try:
        run_device(args, report, save)
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
