# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static TP4 scatter A/B with no model weights or production edits.

The baseline method is compiled from the current generator's AST or an earlier
result's preserved baseline_method when --baseline-artifact is supplied.
--inspect-source loads only Python's standard library and never imports TTNN.
Device execution requires the coordinating agent's exclusive hardware window.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[1]
REPO = MODEL_DIR.parents[2]
DEFAULT_CONFIG = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots"
    / "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json"
)


def baseline_ast(path):
    source = path.read_text()
    cls = next(
        node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Qwen36Generator"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_scatter_slot_logits"
    )
    if method.decorator_list:
        raise ValueError("Baseline acquired decorators; review extraction before running")
    return ast.Module(body=[method], type_ignores=[]), ast.get_source_segment(source, method)


def candidate_method(ttnn):
    def scatter_slot_logits(self, logits, slot: int, batch: int):
        if batch == 1:
            return logits
        empty = ttnn.zeros_like(logits)
        widened = ttnn.concat(
            [logits if row == slot else empty for row in range(batch)],
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(empty)
        ttnn.deallocate(logits)
        return widened

    return scatter_slot_logits


def check_batch1_without_ttnn(tree, baseline_path):
    class ForbiddenTTNN:
        def __getattribute__(self, name):
            raise AssertionError(f"Batch-1 fast return accessed TTNN: {name}")

    forbidden = ForbiddenTTNN()
    namespace = {"ttnn": forbidden}
    exec(compile(tree, str(baseline_path), "exec"), namespace)
    marker, owner = object(), object()
    checks = {}
    for name, method in (
        ("original", namespace["_scatter_slot_logits"]),
        ("device_fill", candidate_method(forbidden)),
    ):
        assert method(owner, marker, 0, 1) is marker
        checks[name] = {"same_object": True, "ttnn_attribute_accesses": 0}
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--baseline-artifact", type=Path, help="Read the original baseline_method from a saved probe JSON"
    )
    parser.add_argument(
        "--local-vocab", type=int, help="Explicit alternate width; default derives from checkpoint config"
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--inspect-source", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/slot_logits_scatter.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    vocab = int(config.get("text_config", config)["vocab_size"])
    padded_vocab = math.ceil(vocab / (4 * 32)) * (4 * 32)
    local_vocab = args.local_vocab or padded_vocab // 4
    if local_vocab <= 0 or local_vocab % 32:
        raise ValueError("Local vocabulary width must be a positive tile multiple")
    if args.samples < 3 or args.warmups < 1:
        raise ValueError("Require at least three measured samples and one discarded warmup")
    baseline_path = MODEL_DIR / "tt/generator.py"
    if args.baseline_artifact is None:
        tree, method_source = baseline_ast(baseline_path)
        baseline_origin = {"kind": "current_generator_ast", "path": str(baseline_path)}
    else:
        if args.baseline_artifact.resolve() == args.output.resolve():
            raise ValueError("Use a different output path to preserve the historical baseline artifact")
        baseline_bytes = args.baseline_artifact.read_bytes()
        baseline_result = json.loads(baseline_bytes)
        method_source = baseline_result["baseline_method"]
        tree = ast.parse(method_source)
        if (
            len(tree.body) != 1
            or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != "_scatter_slot_logits"
            or tree.body[0].decorator_list
        ):
            raise ValueError("Saved baseline must contain exactly one undecorated _scatter_slot_logits method")
        baseline_origin = {
            "kind": "preserved_artifact_method",
            "path": str(args.baseline_artifact.resolve()),
            "artifact_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "recorded_generator_sha256": baseline_result["source_sha256"].get(str(baseline_path)),
        }
    source_paths = [
        Path(__file__).resolve(),
        baseline_path,
        REPO / "ttnn/cpp/ttnn/operations/creation/creation.cpp",
        REPO / "tt_metal/impl/tensor/host_tensor_factory.cpp",
        REPO / "tt_metal/impl/tensor/tensor_impl.cpp",
        args.config,
    ]
    result = {
        "invocation": sys.argv,
        "status": "source_prepared_no_device_execution",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "baseline_method": method_source,
        "baseline_origin": baseline_origin,
        "baseline_method_sha256": hashlib.sha256(method_source.encode()).hexdigest(),
        "vocab_size": vocab,
        "padded_vocab_size": padded_vocab,
        "local_vocab": local_vocab,
        "width_matches_checkpoint": local_vocab == padded_vocab // 4,
        "logical_input_shape_per_rank": [1, 1, 1, local_vocab],
        "padded_input_shape_per_rank": [1, 1, 32, local_vocab],
        "dtype": "BFLOAT8_B",
        "batch": 32,
        "slots": [0, 7, 31],
        "mesh": [1, 4],
        "weights_loaded": False,
        "trace_capture": False,
        "logical_zero_float_bytes_baseline": 31 * local_vocab * 4,
        "padded_zero_float_bytes_baseline": 31 * 32 * local_vocab * 4,
        "measurement": "Host method plus device completion; input cloning, readback and output release excluded equally",
        "correctness": {},
        "timing": {},
        "batch1_host_contract": check_batch1_without_ttnn(tree, baseline_path),
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    if args.inspect_source:
        save()
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("status", "vocab_size", "local_vocab", "padded_zero_float_bytes_baseline")
                }
            )
        )
        return

    import torch

    import ttnn

    def digest(host):
        return hashlib.sha256(host.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1" or ttnn.TRACE_ALLOC_TRACKING:
        raise ValueError("Run this static host-latency probe without device profiler or trace allocation tracking")
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    namespace = {"ttnn": ttnn}
    exec(compile(tree, str(baseline_path), "exec"), namespace)
    mesh = canonical = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
        owners = {name: types.SimpleNamespace(mesh_device=mesh) for name in ("original", "device_fill")}
        methods = {
            "original": types.MethodType(namespace["_scatter_slot_logits"], owners["original"]),
            "device_fill": types.MethodType(candidate_method(ttnn), owners["device_fill"]),
        }
        random = torch.Generator().manual_seed(20260911)
        host = torch.randn((1, 1, 1, 4 * local_vocab), generator=random, dtype=torch.float32) * 4
        canonical = ttnn.from_torch(
            host,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        del host

        def binding(tensor):
            return {
                "shape": list(tensor.shape),
                "padded_shape": list(tensor.padded_shape),
                "dtype": str(tensor.dtype),
                "layout": str(tensor.layout),
                "memory_config": str(tensor.memory_config()),
                "ranks": [
                    {"buffer_unique_id": shard.buffer_unique_id(), "buffer_address": shard.buffer_address()}
                    for shard in ttnn.get_device_tensors(tensor)
                ],
            }

        def read_ranks(tensor):
            return [ttnn.to_torch(shard).contiguous() for shard in ttnn.get_device_tensors(tensor)]

        original_binding = binding(canonical)
        oracle_rows = read_ranks(canonical)
        if tuple(canonical.shape) != (1, 1, 1, local_vocab):
            raise AssertionError(f"Unexpected local logits geometry: {canonical.shape}")
        if not all(torch.isfinite(row).all().item() for row in oracle_rows):
            raise AssertionError("Synthetic BF8 input is not finite")
        input_digests = [digest(row) for row in oracle_rows]
        result["canonical_input"] = original_binding
        result["input_sha256_by_rank"] = input_digests
        result["distinct_rank_inputs"] = len(set(input_digests)) == 4
        assert result["distinct_rank_inputs"]
        result["batch1_device_contract"] = {}
        for name, method in methods.items():
            before = mesh.num_program_cache_entries()
            assert method(canonical, 0, 1) is canonical
            assert binding(canonical) == original_binding
            assert mesh.num_program_cache_entries() == before
            result["batch1_device_contract"][name] = {
                "same_object": True,
                "input_owner_preserved": True,
                "program_count_unchanged": True,
            }

        def invoke(name, slot):
            # Both methods consume their input. Only this private clone is
            # transferred; the shared canonical owner remains live and intact.
            working = ttnn.clone(canonical)
            ttnn.synchronize_device(mesh)
            before = mesh.num_program_cache_entries()
            started = time.perf_counter()
            output = methods[name](working, slot, 32)
            ttnn.synchronize_device(mesh)
            elapsed = (time.perf_counter() - started) * 1000
            if working.is_allocated():
                raise AssertionError(f"{name} changed the consumed-input ownership contract")
            if not canonical.is_allocated() or binding(canonical) != original_binding:
                raise AssertionError(f"{name} invalidated the shared input owner")
            return output, elapsed, before, mesh.num_program_cache_entries()

        for slot in result["slots"]:
            result["correctness"][str(slot)] = {}
            baseline_rows = None
            for name in methods:
                output, _, _, _ = invoke(name, slot)
                output_binding = binding(output)
                rows = read_ranks(output)
                assert tuple(output.shape) == (1, 32, 1, local_vocab)
                exact_oracle, exact_baseline = [], []
                for rank, host_rows in enumerate(rows):
                    expected = torch.zeros_like(host_rows)
                    expected[:, slot : slot + 1, :, :] = oracle_rows[rank]
                    exact_oracle.append(torch.equal(host_rows, expected))
                    if baseline_rows is not None:
                        exact_baseline.append(torch.equal(host_rows, baseline_rows[rank]))
                assert [digest(row) for row in read_ranks(canonical)] == input_digests
                result["correctness"][str(slot)][name] = {
                    "all_rows_exact_to_oracle_by_rank": exact_oracle,
                    "all_rows_exact_to_baseline_by_rank": exact_baseline or None,
                    "input_owner_and_values_preserved": True,
                    "output": output_binding,
                    "row_sha256_by_rank": [[digest(row[:, index : index + 1]) for index in range(32)] for row in rows],
                }
                if not all(exact_oracle) or not all(exact_baseline):
                    raise AssertionError(f"{name} slot {slot} changed rank-local rows")
                if baseline_rows is None:
                    baseline_rows = rows
                ttnn.deallocate(output)
                save()

            # Warm each slot-dependent concat shape, then alternate arm order.
            for _ in range(args.warmups):
                for name in methods:
                    output, _, _, _ = invoke(name, slot)
                    ttnn.deallocate(output)
            samples = {name: [] for name in methods}
            for index in range(args.samples):
                order = ("original", "device_fill") if index % 2 == 0 else ("device_fill", "original")
                for name in order:
                    output, elapsed, before, after = invoke(name, slot)
                    samples[name].append(
                        {"ms": elapsed, "program_entries_before": before, "program_entries_after": after}
                    )
                    ttnn.deallocate(output)
                    if before != after:
                        raise AssertionError(f"{name} slot {slot} compiled during warmed measurement")
            medians = {name: statistics.median(item["ms"] for item in values) for name, values in samples.items()}
            result["timing"][str(slot)] = {
                "discarded_warmups_per_arm": args.warmups,
                "samples": samples,
                "median_ms": medians,
                "speedup": medians["original"] / medians["device_fill"],
            }
            print("SCATTER_RESULT", slot, json.dumps(result["timing"][str(slot)]), flush=True)
            save()
        assert [digest(row) for row in read_ranks(canonical)] == input_digests
        result["status"] = "passed_static_tp4_correctness_and_timing"
        save()
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        save()
        raise
    finally:
        if mesh is not None:
            if canonical is not None and canonical.is_allocated():
                ttnn.deallocate(canonical)
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
