# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small grouped packed-lookup device proof, including live trace refresh.

Tests exactly the quantizer settings used by probe_packed_embedding_cpu. The
table is deliberately compact and is ONLY an operator diagnostic, never full
PLE or decoder acceptance. Optional checkpoint rows span all128 source shards.
Parent executes this hardware runner serially under the device-usage workflow.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")

import torch
from safetensors import safe_open
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_packed_embedding_cpu import independent_quantize
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding_cpu import _quantize_rows, _round32, quantization_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--codebook-report", type=Path)
    parser.add_argument("--spec", type=Path, help="One exact quantizer/spec hypothesis or completed cache manifest")
    parser.add_argument(
        "--small-cache-dir",
        type=Path,
        help="Serialize/reload the compact operator fixture before upload; never a full-domain cache",
    )
    parser.add_argument("--groups", default="20,160,80,40,32,16")
    parser.add_argument("--lengths", default="1,7,31,32")
    args = parser.parse_args()
    groups = [int(item) for item in args.groups.split(",")]
    lengths = [int(item) for item in args.lengths.split(",")]
    if not lengths or not all(1 <= value <= 32 for value in lengths):
        parser.error("--lengths must be physical hash lengths1..32")
    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        parser.error("TT_METAL_TRACE_ALLOC_TRACKING must be enabled")
    torch.set_num_threads(8)
    generator = torch.Generator().manual_seed(3031)
    weight = torch.cat((torch.zeros(1, 160), torch.randn(63, 160, generator=generator) * 0.02)).bfloat16()
    if args.snapshot:
        index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        real = []
        for shard in range(128):
            key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
            with safe_open(args.snapshot / index[key], framework="pt") as f:
                real.append(f.get_slice(key)[1017:1019, :].clone())
        weight = torch.cat((weight, *real))
    if args.spec is None and args.codebook_report is None:
        parser.error("Provide --spec or --codebook-report")
    books = json.loads(args.codebook_report.read_text())["training"]["fixed_codebooks"] if args.codebook_report else {}
    specs = []
    if args.spec:
        candidate = json.loads(args.spec.read_text())
        specs.append(
            quantization_spec(
                **{
                    key: candidate[key]
                    for key in ("quantization", "group_size", "codebook_values", "scale_codebook_values")
                    if key in candidate
                }
            )
        )
    for size in [] if args.spec else groups:
        if size == 160:
            specs.append(quantization_spec())
            continue
        for method in ("absmax", "lsq"):
            specs.append(quantization_spec(quantization=method, group_size=size))
            if f"group{size}_affineFalse" in books:
                specs.append(
                    quantization_spec(
                        quantization=method, group_size=size, codebook_values=books[f"group{size}_affineFalse"]
                    )
                )
    report = {
        "scope": "compact packed lookup operator; not full PLE or decoder acceptance",
        "rows": weight.shape[0],
        "real_rows": 256 if args.snapshot else 0,
        "real_shards": 128 if args.snapshot else 0,
        "snapshot": str(args.snapshot) if args.snapshot else None,
        "codebook_report": str(args.codebook_report),
        "spec_file": str(args.spec) if args.spec else None,
        "small_cache_dir": str(args.small_cache_dir) if args.small_cache_dir else None,
        "trace_allocation_tracking": True,
        "runtime_audit": "forbid torch functions/conversions and throw on fallback around lookup and replay",
        "cases": [],
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def compare(expected, actual):
        return {
            "exact_equal": torch.equal(expected, actual),
            "max_abs_error": (expected.float() - actual.float()).abs().max().item(),
        }

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace_id = None
    try:
        for spec in specs:
            options = {
                key: spec[key]
                for key in ("quantization", "group_size", "codebook_values", "scale_codebook_values")
                if key in spec
            }
            if args.small_cache_dir:
                import numpy as np

                identity = spec["codebook_sha256"] + "_" + spec.get("scale_codebook_sha256", str(spec["group_size"]))
                directory = args.small_cache_dir / identity
                directory.mkdir(parents=True, exist_ok=True)
                payload = {}
                for name, values in zip(("words", "scales"), _quantize_rows(weight, **options)):
                    padded = torch.zeros(32 * _round32((values.numel() + 31) // 32), dtype=values.dtype)
                    padded[: values.numel()] = values
                    striped = padded.reshape(-1, 32).T.contiguous()
                    array = striped.view(torch.uint16).numpy() if values.dtype == torch.bfloat16 else striped.numpy()
                    path = directory / f"{name}.bin"
                    array.tofile(path)
                    saved = torch.from_numpy(np.memmap(path, mode="c", dtype=array.dtype, shape=array.shape))
                    payload[name] = saved.view(torch.bfloat16) if values.dtype == torch.bfloat16 else saved
                (directory / "fixture_only.json").write_text(
                    json.dumps(
                        {
                            "scope": "compact selected-row operator fixture; invalid as full-domain checkpoint cache",
                            "rows": weight.shape[0],
                            "spec": spec,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                lookup = PackedEmbedding._from_stripes(
                    payload["words"], payload["scales"], mesh_device=mesh, rows=weight.shape[0], **options
                )
            else:
                lookup = PackedEmbedding.from_torch(weight, mesh_device=mesh, **options)
            expected_weight = independent_quantize(weight, spec)
            # Prove high-bit scale arithmetic without allocating the complete
            # table: group20/16 linear indices exceed signed INT32 near the end.
            large_rows = torch.tensor(
                [0, 1, 7, 8, 31, 32, 2500011, 2500012, 268435455, 268435456, 300000001, 320001535]
            ).reshape(-1, 1)
            address_input = ttnn.from_torch(large_rows, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)
            with runtime_audit():
                columns, stripes = lookup.scale_addresses(address_input)
            scale_storage_per_row = spec.get("scale_words_per_row", spec["scales_per_row"])
            linear = large_rows * scale_storage_per_row + torch.arange(scale_storage_per_row)
            address_exact = torch.equal(ttnn.to_torch(columns).long(), linear >> 5) and torch.equal(
                ttnn.to_torch(stripes).long(), linear & 31
            )
            for tensor in (address_input, columns, stripes):
                ttnn.deallocate(tensor)
            if not address_exact:
                raise AssertionError(f"Grouped global scale addressing failed: {spec}")
            for length in lengths:
                ids = torch.randint(0, weight.shape[0], (1, 1, length, 16), generator=generator)
                ids.flatten()[:8] = torch.tensor([0, 1, 7, 8, 31, 32, weight.shape[0] - 2, weight.shape[0] - 1])
                changed = ids.flip(-1).contiguous()
                expected = expected_weight[ids.flatten()].reshape(1, 1, length, 2560)
                expected_changed = expected_weight[changed.flatten()].reshape(expected.shape)
                device_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                host_changed = ttnn.from_torch(changed, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                with runtime_audit():
                    warm = lookup(device_ids)
                eager = ttnn.to_torch(warm)
                ttnn.deallocate(warm)
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                with runtime_audit():
                    output = lookup(device_ids)
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                trace_allocation_tracker.acknowledge_corruptible(output)
                ttnn.to_torch(output)
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                replay = ttnn.to_torch(output)
                ttnn.copy_host_to_device_tensor(host_changed, device_ids)
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                changed_output = ttnn.to_torch(output)
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                repeated = ttnn.to_torch(output)
                entry = {
                    "quantizer": spec,
                    "sequence": length,
                    "full_domain_scale_addresses_exact": address_exact,
                    "eager": compare(expected, eager),
                    "trace": compare(expected, replay),
                    "changed_input": compare(expected_changed, changed_output),
                    "repeated_input_exact": torch.equal(changed_output, repeated),
                    "captures": 1,
                    "replays": 3,
                    "acknowledgment": "exact trace output only; consumed before every replay",
                }
                entry["passed"] = (
                    all(entry[key]["exact_equal"] for key in ("eager", "trace", "changed_input"))
                    and entry["repeated_input_exact"]
                )
                report["cases"].append(entry)
                save()
                print("GROUPED_EMBEDDING_CASE", json.dumps(entry), flush=True)
                ttnn.release_trace(mesh, trace_id)
                trace_id = None
                ttnn.deallocate(output)
                ttnn.deallocate(device_ids)
                if not entry["passed"]:
                    raise AssertionError("Grouped reconstruction differs from independent CPU reference")
            for name in ("table", "scales", "codebook", "lanes", "scale_lanes", "scale_indices", "scale_codebook"):
                value = getattr(lookup, name, None)
                if value is not None:
                    ttnn.deallocate(value)
        report["passed"] = bool(report["cases"]) and all(case["passed"] for case in report["cases"])
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        save()
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
