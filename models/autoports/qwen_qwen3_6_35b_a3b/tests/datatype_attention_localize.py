# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact-input attention controls for resume localization; not a model gate.

Construct one real baseline decoder, without embeddings, LM head, generator,
or traces. Compare existing projection helpers on identical captured operands.
Replay the existing singleton attention branch to observe its actual gated
normalization operands, then compare native SFPU and LUT SiLU on those operands.
This program must run serially under the device owner's bounded timeout.
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--position", type=int, default=2559)
    parser.add_argument("--phases", nargs="+", choices=("fresh", "prefix", "suffix"), default=["fresh", "suffix"])
    parser.add_argument(
        "--capture-products", action="store_true", help="Retain large FP32 products as well as partials"
    )
    parser.add_argument(
        "--skip-gating", action="store_true", help="Projection-only control; explicitly marks gating untested"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    assert not result_path.exists(), "Use a fresh output directory"
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1"
    assert not os.environ.get("TT_METAL_WATCHER"), "Watcher requires a separate job"
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

    import torch

    torch.set_num_threads(8)
    metadata_path = args.captures_dir / "capture_metadata.json"
    statistics_path = args.captures_dir / "statistics.json"
    capture_metadata = json.loads(metadata_path.read_text())
    localization = json.loads(statistics_path.read_text())
    captures = torch.load(args.captures_dir / "captures.pt", map_location="cpu", weights_only=False)
    assert localization["precision_config"] == "baseline" and localization["gate_modified"] is False
    assert localization["start"] == 2559 or args.skip_gating, "Current gate replay requires singleton suffix"
    sources = [Path(__file__), metadata_path, statistics_path, *sorted((ROOT / "tt").glob("*.py"))]
    result = dict(
        status="running",
        command=sys.argv,
        time_unix=time.time(),
        mesh=[1, 4],
        layer=args.layer,
        position=args.position,
        captures_dir=str(args.captures_dir.resolve()),
        scope="Same-input kernel isolation only; no full-model accuracy, performance, or gate waiver",
        original_gate=localization.get("gate"),
        gate_modified=False,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        construction="One MultichipDecoder; same baseline/full_model_optimized arguments as QwenModel",
        projection_cases=[],
        projection_comparisons=[],
        gating=dict(status="not_run"),
    )
    saved = {}
    mesh = layer = None

    def save():
        torch.save(saved, args.output_dir / "operands_outputs.pt")
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def metrics(left, right):
        a, b = left.double().flatten(), right.double().flatten()
        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
        delta = a - b
        ac, bc = a - a.mean(), b - b.mean()
        denominator = ac.norm() * bc.norm()
        equal = torch.equal(a, b)
        return dict(
            bit_equal=equal,
            pcc=1.0 if equal else float((ac @ bc) / denominator) if denominator > 0 else None,
            max_abs=float(delta.abs().max()),
            rms_difference=float(delta.square().mean().sqrt()),
            relative_l2=float(delta.norm() / b.norm()) if b.norm() > 0 else None,
            changed_elements=int(torch.count_nonzero(delta)),
            elements=a.numel(),
        )

    def normalized(phase):
        found = [
            (key, value)
            for key, value in capture_metadata.items()
            if value["phase"] == phase
            and value["layer"] == args.layer
            and value["event"] == "normalized.input_layernorm.weight"
            and args.position in value.get("positions", [])
        ]
        assert len(found) == 1, (phase, args.position, [key for key, _ in found])
        key, meta = found[0]
        row = captures[key][meta["positions"].index(args.position)].reshape(1, 1, 2048)
        assert row.dtype == torch.bfloat16 and torch.isfinite(row).all()
        return row, key, meta

    try:
        import ttnn
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_REVISION, Checkpoint
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.precision import layer_precision, resolve_precision

        def host(tensor):
            parts = [ttnn.to_torch(part).detach().clone() for part in ttnn.get_device_tensors(tensor)]
            assert len(parts) == 4 and all(torch.isfinite(part).all() for part in parts)
            return parts

        def upload(value, dtype, *, dim=None, layout=None, memory=None):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT if layout is None else layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if memory is None else memory,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if dim is None else ttnn.ShardTensorToMesh(mesh, dim=dim),
            )

        checkpoint = Checkpoint()
        assert checkpoint.config.layer_types[args.layer] == "linear_attention"
        result["checkpoint"] = dict(path=str(checkpoint.path), revision=HF_REVISION)
        state_dict = checkpoint.select(f"model.language_model.layers.{args.layer}.")
        # Hash the actual attention tensors rather than multi-GB shard files.
        result["attention_weight_sha256"] = {
            key: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for key, value in state_dict.items()
            if key.startswith("linear_attn.")
        }
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        layer = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=checkpoint.config,
            layer_idx=args.layer,
            mesh_device=mesh,
            full_model_optimized=True,
            precision_config=layer_precision(resolve_precision("baseline"), args.layer),
        )
        del state_dict
        assert layer.pack_linear and type(layer.direct_coefficient_scan).__name__ == "CachedCoefficientScan"
        result["runtime_precision"] = layer.precision_summary()
        result["execution_geometry"] = layer.execution_geometry
        weights = host(layer.scan_combined_weight)
        saved["projection/weights_pair_swapped"] = weights
        weight_rows = [w.reshape(2048, 3136) for w in weights]
        pair_order = torch.arange(2048).reshape(-1, 2).flip(1).flatten()
        result["cpu_control"] = {
            "projection": "FP64 dot of actual BF16 device weights and BF16 x, both in paired K order",
            "grouped_fp32": "CPU sequential FP32 products/adds within K groups; diagnostic, not a bit-exact TT cumsum emulator",
            "gating": "FP64 sigmoid/SiLU then FP32 product and BF16 rounding; separate production FP32 LUT control",
        }
        for phase in args.phases:
            x_cpu, key, meta = normalized(phase)
            saved[f"projection/{phase}/x"] = x_cpu
            paired = x_cpu.flatten()[pair_order].float()
            cpu_dot = [paired.double() @ w.double() for w in weight_rows]
            saved[f"projection/{phase}/cpu_fp64"] = cpu_dot
            x = upload(x_cpu, ttnn.bfloat16)
            branch_outputs = {}
            for name, k, helper in (
                ("cached512", 512, layer.direct_coefficient_scan),
                ("narrow512", 512, layer.direct_coefficient_scan.narrow),
                ("narrow1024", 1024, layer.direct_coefficient_scan.narrow),
            ):
                weight = ttnn.reshape(layer.scan_combined_weight, (2048 // k, 1, k, 3136))
                assert helper.supports(weight, x), (name, weight.shape, x.shape)
                prefix = f"projection/{phase}/{name}"
                partial = helper(weight, x)
                saved[prefix + "/partial_fp32"] = host(partial)
                total = ttnn.experimental.fast_reduce_nc(
                    partial, dims=[0], compute_kernel_config=layer.compute, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                total = ttnn.reshape(total, (1, 1, 3136), (1, 32, 3136))
                saved[prefix + "/total_fp32"] = host(total)
                final = ttnn.typecast(total, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                final_host = host(final)
                saved[prefix + "/final_bf16"] = final_host
                branch_outputs[name] = final_host
                coefficients = helper(weight, x, coefficients_only=True)
                coefficient_host = host(coefficients)
                saved[prefix + "/coefficients"] = coefficient_host
                expected = paired.reshape(2048 // k, 1, k, 1).bfloat16()
                coefficient_checks = [metrics(value, expected) for value in coefficient_host]
                assert all(item["bit_equal"] for item in coefficient_checks), (phase, name, coefficient_checks)
                if args.capture_products:
                    products = helper(weight, x, products_only=True)
                    saved[prefix + "/products_fp32"] = host(products)
                    ttnn.deallocate(products)
                grouped = []
                for w in weight_rows:
                    blocks = []
                    for offset in range(0, 2048, k):
                        accumulator = torch.zeros(3136, dtype=torch.float32)
                        for row in range(offset, offset + k):
                            accumulator += paired[row] * w[row].float()
                        blocks.append(accumulator)
                    grouped.append(torch.stack(blocks).reshape(2048 // k, 1, 1, 3136))
                saved[prefix + "/cpu_sequential_partial_fp32"] = grouped
                result["projection_cases"].append(
                    dict(
                        phase=phase,
                        branch=name,
                        projection_k=k,
                        helper=type(helper).__name__,
                        capture_key=key,
                        input_metadata=meta,
                        coefficients=coefficient_checks,
                        partial_vs_cpu_sequential=[
                            metrics(a, b) for a, b in zip(saved[prefix + "/partial_fp32"], grouped)
                        ],
                        final_vs_cpu_fp64=[metrics(a.flatten(), b) for a, b in zip(final_host, cpu_dot)],
                        final_vs_cpu_fp64_rounded_bf16=[
                            metrics(a.flatten(), b.bfloat16()) for a, b in zip(final_host, cpu_dot)
                        ],
                    )
                )
                for tensor in (partial, total, final, coefficients):
                    ttnn.deallocate(tensor)
                print("PROJECTION_CONTROL", phase, name, flush=True)
            for left, right in itertools.combinations(branch_outputs, 2):
                result["projection_comparisons"].append(
                    dict(
                        phase=phase,
                        left=left,
                        right=right,
                        final_per_rank=[metrics(a, b) for a, b in zip(branch_outputs[left], branch_outputs[right])],
                        # K512 reader controls share the same partial shape.
                        partial_per_rank=(
                            [
                                metrics(a, b)
                                for a, b in zip(
                                    saved[f"projection/{phase}/{left}/partial_fp32"],
                                    saved[f"projection/{phase}/{right}/partial_fp32"],
                                )
                            ]
                            if (left, right) == ("cached512", "narrow512")
                            else None
                        ),
                    )
                )
            ttnn.deallocate(x)
            save()

        if not args.skip_gating:
            x_cpu, key, meta = normalized("suffix")
            assert meta["context"]["public_length"] == 1 and meta["context"]["decode"] is True
            state = {}
            state_sources = {}
            for name, dtype, dim, layout in (
                ("recurrent", ttnn.float32, 1, ttnn.TILE_LAYOUT),
                ("conv", ttnn.bfloat16, 2, ttnn.ROW_MAJOR_LAYOUT),
            ):
                state_key = f"suffix/l{args.layer:02}/state.public_begin.after_bf16_cast/{name}"
                parts = captures[state_key]
                assert len(parts) == 4
                saved[f"gating/state_before/{name}"] = parts
                state[name] = upload(torch.cat(parts, dim=dim), dtype, dim=dim, layout=layout)
                state_sources[name] = state_key
            x = upload(x_cpu, ttnn.bfloat16)
            original_multiply, original_linear = ttnn.multiply, layer.linear
            observed = {}

            def multiply(a, b, *positional, **kwargs):
                is_gate = kwargs.get("input_tensor_b_activations") == [ttnn.UnaryOpType.SILU]
                if is_gate and tuple(a.shape) == (1, 8, 128) and tuple(b.shape) == (1, 8, 128):
                    assert "weighted" not in observed, "Expected exactly one singleton SiLU gate"
                    observed["weighted"] = host(a)
                    observed["z_fp32_rows"] = host(b)
                    observed["multiply_kwargs"] = {k: str(v) for k, v in kwargs.items()}
                    value = original_multiply(a, b, *positional, **kwargs)
                    observed["native_original"] = host(value)
                    return value
                return original_multiply(a, b, *positional, **kwargs)

            def linear(value, name, *positional, **kwargs):
                out = original_linear(value, name, *positional, **kwargs)
                if name == "attention_combined":
                    observed["combined_projection"] = host(out)
                return out

            with patch.object(ttnn, "multiply", multiply), patch.object(layer, "linear", linear):
                out = layer._linear_attention(x, state, decode=True, cache_boundary=False, projection_k=512)
            assert "weighted" in observed and "combined_projection" in observed
            saved["gating/attention_local_output_fp32"] = host(out)
            for name in ("weighted", "z_fp32_rows", "native_original", "combined_projection"):
                saved["gating/" + name] = observed[name]
            weighted_parts, z_parts = observed["weighted"], []
            for weighted, z, projected in zip(weighted_parts, observed["z_fp32_rows"], observed["combined_projection"]):
                assert torch.equal(weighted, weighted.bfloat16().float()), "Expected BF16-valued FP32 weighted input"
                z_bf16 = projected[:, :, 2112:3136]
                assert torch.equal(z.flatten(), z_bf16.float().flatten())
                z_parts.append(z_bf16)
            saved["gating/projection_z_bf16"] = z_parts
            # Both branches start from exactly these same retained operands.
            weighted = upload(torch.cat(weighted_parts, dim=1), ttnn.float32, dim=1, memory=ttnn.L1_MEMORY_CONFIG)
            z = upload(torch.cat(z_parts, dim=2), ttnn.bfloat16, dim=2)
            native = original_multiply(
                weighted,
                ttnn.reshape(ttnn.typecast(z, ttnn.float32), (1, 8, 128)),
                input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                fast_and_approximate_mode=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            indices = ttnn.reshape(ttnn.to_layout(ttnn.bitcast(z, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), (1, 1024))
            lut = ttnn.to_layout(ttnn.reshape(ttnn.gather(layer.silu_table, 1, indices), (1, 8, 128)), ttnn.TILE_LAYOUT)
            lookup = original_multiply(
                weighted,
                lut,
                dtype=ttnn.bfloat16,
                fast_and_approximate_mode=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            native_host, lookup_host = host(native), host(lookup)
            saved["gating/native_replay"] = native_host
            saved["gating/lookup_replay"] = lookup_host
            saved["gating/lookup_values_fp32"] = host(lut)
            cpu64, cpu_lut = [], []
            for w, z in zip(weighted_parts, z_parts):
                z64 = z.double().reshape(1, 8, 128)
                cpu64.append((w.double() * (z64 * torch.sigmoid(z64))).float().bfloat16())
                cpu_lut.append((w * torch.nn.functional.silu(z.float().reshape(1, 8, 128))).bfloat16())
            saved["gating/cpu_fp64_rounded"] = cpu64
            saved["gating/cpu_production_lut"] = cpu_lut
            result["gating"] = dict(
                status="completed",
                capture_key=key,
                input_metadata=meta,
                state_sources=state_sources,
                attention_kwargs=dict(decode=True, cache_boundary=False, projection_k=512),
                observed_native_kwargs=observed["multiply_kwargs"],
                same_operand_replay=[metrics(a, b) for a, b in zip(native_host, observed["native_original"])],
                native_vs_lookup=[metrics(a, b) for a, b in zip(native_host, lookup_host)],
                native_vs_cpu_fp64=[metrics(a, b) for a, b in zip(native_host, cpu64)],
                lookup_vs_cpu_fp64=[metrics(a, b) for a, b in zip(lookup_host, cpu64)],
                lookup_vs_cpu_production_lut=[metrics(a, b) for a, b in zip(lookup_host, cpu_lut)],
                limitation="Singleton replay isolates gate operands; it does not emulate the fresh chunk recurrence or prove a full-model fix",
            )
            assert all(
                item["bit_equal"] for item in result["gating"]["same_operand_replay"]
            ), "Native same-input replay changed"
            print("GATING_CONTROL_COMPLETED", flush=True)
        else:
            result["gating"] = dict(status="skipped_by_command", reason="--skip-gating")
        ttnn.synchronize_device(mesh)
        result["status"] = "completed_diagnostic_not_gate"
    except BaseException:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            import ttnn

            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
            save()


if __name__ == "__main__":
    main()
