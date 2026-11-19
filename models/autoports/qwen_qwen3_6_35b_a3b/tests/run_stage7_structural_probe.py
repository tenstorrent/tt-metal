# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated structural controls, serialized by the caller under timeout.

Integrated mode reuses run_multichip_decoder._run_side twice on the same1x4
policy, requiring exact prefill, evolving decode and per-rank state equality.
Component mode checks the affected reduction interfaces before integrated work.
All promoted current defaults remain enabled. No candidate is installed globally
outside this process or adopted by the production model.
"""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path
from types import MethodType

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests import run_multichip_decoder as runner
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_stage7_scan_probe import VARIANTS as SCAN_VARIANTS
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_structural_probe import install
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder


def exact_tree(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        if (
            a.shape != b.shape
            or a.dtype != b.dtype
            or not torch.equal(a, b)
            or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
        ):
            raise AssertionError(f"Exact tensor mismatch at {path}; shapes {a.shape}/{b.shape}")
        return 1
    if isinstance(a, dict):
        if set(a) != set(b):
            raise AssertionError(f"Dictionary keys differ at {path}")
        return sum(exact_tree(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"Sequence lengths differ at {path}")
        return sum(exact_tree(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
    if a != b:
        raise AssertionError(f"Value mismatch at {path}")
    return 0


def integrated(args, result):
    c, sd = config(), weights(args.layer)
    inputs = runner._inputs(args, c)
    print("STRUCTURAL_BASELINE_START", flush=True)
    baseline = runner._run_side(args, c, sd, inputs, multichip=True)
    descriptor = MultichipDecoder.__dict__["from_state_dict"]
    constructor = descriptor.__func__
    installations = []
    result["source_edits"] = installations

    def construct(cls, state_dict, **kwargs):
        if args.production:
            kwargs["full_model_optimized"] = True
        dec = constructor(cls, state_dict, **kwargs)
        evidence = {"layer": kwargs.get("layer_idx"), "kind": dec.kind}
        if args.production:
            evidence["execution_geometry"] = dec.execution_geometry
        if not args.production and (args.recurrent_permute or args.collective_padding):
            install(dec, recurrent_permute=args.recurrent_permute, collective_padding=args.collective_padding)
            evidence["structural"] = dec.stage7_structural_probe
        if args.direct_recurrent_sum and not args.production:
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.direct_recurrent_sum_probe import install as install_sum

            install_sum(dec)
            evidence["direct_recurrent_sum"] = dec.kind == "linear_attention"
            if args.recurrent_permute and dec.kind == "linear_attention":
                # Structural installation copied the old final contraction.
                # Explicitly route its head-row interface through the new op;
                # otherwise the second contraction silently stays baseline.
                original_head_rows = dec._recurrent_sum_head_rows

                def head_rows(self, products):
                    if tuple(products.shape) != (1, 8, 128, 128):
                        return original_head_rows(products)
                    return self.direct_recurrent_sum_probe(products, terminal_rows=True)

                dec._recurrent_sum_head_rows = MethodType(head_rows, dec)
                evidence["direct_sum_terminal_rows"] = True
        if args.scan_variant:
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.scan_terminal_probe import ScanTerminalProbe

            selective_cache = args.scan_variant == "cache32_selective"
            read_bytes, stores = (32, "baseline") if selective_cache else SCAN_VARIANTS[args.scan_variant]
            narrow = ScanTerminalProbe(ttnn, dec.mesh_device, read_bytes=read_bytes, stores=stores)
            dec.direct_coefficient_scan = narrow
            if selective_cache:
                from models.autoports.qwen_qwen3_6_35b_a3b.tt.scan_coefficient_cache_probe import CachedCoefficientScan

                cached = CachedCoefficientScan(ttnn, dec.mesh_device, read_bytes=32, padding="none")

                def selected_scan(weight, activation, **options):
                    selected = cached if tuple(weight.shape) in ((4, 1, 512, 3136), (2, 1, 512, 2048)) else narrow
                    return selected(weight, activation, **options)

                selected_scan.reader_source = cached.reader_source
                selected_scan.compute_source = cached.compute_source
                dec.direct_coefficient_scan = selected_scan
            evidence["scan"] = {
                "variant": args.scan_variant,
                "read_bytes": read_bytes,
                "stores": stores,
                "reader_sha256": hashlib.sha256(dec.direct_coefficient_scan.reader_source.encode()).hexdigest(),
                "compute_sha256": hashlib.sha256(dec.direct_coefficient_scan.compute_source.encode()).hexdigest(),
            }
        if args.product_recurrent_sum:
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.product_recurrent_sum_probe import (
                install as install_product_sum,
            )

            install_product_sum(dec)
            evidence["product_recurrent_sum"] = dec.kind == "linear_attention"
        if args.state_slab:
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.state_slab_recurrence_probe import (
                install as install_state_slab,
            )

            install_state_slab(dec)
            evidence["state_slab"] = dec.kind == "linear_attention"
        installations.append(evidence)
        return dec

    MultichipDecoder.from_state_dict = classmethod(construct)
    try:
        print("STRUCTURAL_CANDIDATE_START", flush=True)
        candidate = runner._run_side(args, c, sd, inputs, multichip=True)
    finally:
        MultichipDecoder.from_state_dict = descriptor
    tensor_path = str(Path(args.output).with_suffix(".pt"))
    torch.save(dict(inputs=inputs, baseline=baseline, candidate=candidate), tensor_path)
    result.update(
        baseline=baseline["summary"],
        candidate=candidate["summary"],
        source_edits=installations,
        tensor_artifact=tensor_path,
    )
    comparisons = {}
    for key in ("prefill", "decode", "states", "prefill_states"):
        comparisons[key] = exact_tree(baseline[key], candidate[key], key)
    exact_tree(baseline["summary"]["precision_policy"], candidate["summary"]["precision_policy"], "policy")
    result["exact_tensor_comparisons"] = comparisons
    result["decode_speedup"] = (
        baseline["summary"]["decode_traced_warmed_ms"] / candidate["summary"]["decode_traced_warmed_ms"]
    )
    result["gate_boundary"] = (
        "Existing runner gates retained: same-rank replicas, deterministic prefill, every-step eager/trace, "
        "repeated evolving replay, live-trace prefill/replay, runtime device-only audit; additionally exact "
        "selected-mesh baseline/candidate outputs and all physical states. This is not a new HF accuracy run."
    )


def components(args, result):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
    try:
        dec = install(
            MultichipDecoder.from_state_dict(
                weights(args.layer), hf_config=config(), layer_idx=args.layer, mesh_device=mesh
            ),
            recurrent_permute=args.recurrent_permute,
            collective_padding=args.collective_padding,
        )
        result["source_edits"] = [dec.stage7_structural_probe]
        if args.collective_padding and not dec.batch_direct_ar:
            raise AssertionError("R>1 component controls require the selected batch_direct_ar=True")
        cases = []
        if args.recurrent_permute and dec.kind == "linear_attention":
            cases.append(
                (
                    "recurrent_final_head_rows",
                    (1, 32, 128, 128),
                    1,
                    ttnn.float32,
                    lambda x: ttnn.permute(dec._structural_original_sum(x), (0, 2, 1, 3)),
                    dec._recurrent_sum_head_rows,
                )
            )
        if args.collective_padding:
            for rows in (1, 3, 31, 32):
                for dtype in (ttnn.bfloat16, ttnn.float32):
                    cases.append(
                        (
                            f"collective_rows{rows}_{dtype}",
                            (1, rows, 8192),
                            -1,
                            dtype,
                            lambda x: dec._structural_original_reduce(x, batched=True),
                            lambda x: dec._reduce(x, batched=True),
                        )
                    )
        observations = []
        for name, shape, dim, dtype, baseline, candidate in cases:
            traces = []
            try:
                sources = []
                poison_padding = name.startswith("collective_") and shape[-2] < 32
                for step in range(args.component_steps):
                    torch.manual_seed(args.seed + step)
                    source = torch.randn(shape, dtype=torch.float32) * 0.05
                    sources.append(
                        ttnn.from_torch(
                            source,
                            device=mesh,
                            dtype=dtype,
                            layout=ttnn.TILE_LAYOUT,
                            pad_value=37.0 if poison_padding else 0.0,
                            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
                        )
                    )
                stable = ttnn.clone(sources[0])
                destinations = []
                for forward in (baseline, candidate):
                    with device_only():
                        value = forward(stable)
                    destination = ttnn.empty_like(value)
                    ttnn.copy(value, destination)
                    # Read every per-rank output layout before capture. A first
                    # read can compile untilize/unpad programs in some layouts.
                    for part in ttnn.get_device_tensors(destination):
                        ttnn.to_torch(part)
                    destinations.append(destination)
                    del value
                # Warm every copy signature and forward before the first capture.
                for source in sources:
                    ttnn.copy(source, stable)
                    if poison_padding:
                        # Expose existing physical rows with a metadata-only
                        # tiled view. No padding write or layout conversion.
                        physical = ttnn.reshape(stable, stable.padded_shape, stable.padded_shape)
                        if physical.buffer_address() != stable.buffer_address():
                            raise AssertionError("Padding inspection must remain a metadata-only view")
                        for part in ttnn.get_device_tensors(physical):
                            padded = ttnn.to_torch(part)
                            if not torch.all(padded[..., shape[-2] :, :] == 37):
                                raise AssertionError("Poison padding was not preserved by upload/copy")
                        del physical
                ttnn.synchronize_device(mesh)
                for forward, destination in zip((baseline, candidate), destinations):
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    try:
                        with device_only():
                            ttnn.copy(forward(stable), destination)
                    finally:
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        traces.append(trace)
                for step, source in enumerate(sources):
                    ttnn.copy(source, stable)
                    for trace in traces:
                        with device_only():
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    outputs = [
                        [ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(dst)] for dst in destinations
                    ]
                    comparisons = exact_tree(outputs[0], outputs[1], f"{name}.step{step}")
                    observations.append(
                        dict(
                            case=name,
                            step=step,
                            rank_tensors_equal=comparisons,
                            nonzero_physical_padding_verified=poison_padding,
                            logical_input_shape=list(stable.shape),
                            padded_input_shape=list(stable.padded_shape),
                        )
                    )
                print("STRUCTURAL_COMPONENT_PASS", name, flush=True)
            finally:
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
        result["components"] = observations
        result[
            "scope"
        ] = "Changed-input reduction-interface trace equality; recurrent state evolution is covered by integrated mode"
    finally:
        ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("component", "integrated"), default="integrated")
    parser.add_argument("--recurrent-permute", action="store_true")
    parser.add_argument("--collective-padding", action="store_true")
    parser.add_argument("--direct-recurrent-sum", action="store_true", help="Integrated mode only")
    parser.add_argument(
        "--scan-variant",
        choices=[*SCAN_VARIANTS, "cache32_selective"],
        help="Integrated mode only; current scan oracle",
    )
    parser.add_argument("--production", action="store_true", help="Compare explicit optimized full-model geometry")
    parser.add_argument("--product-recurrent-sum", action="store_true", help="Isolated FP32 product/contraction fusion")
    parser.add_argument("--state-slab", action="store_true", help="Isolated exact full-state recurrence fusion")
    parser.add_argument("--component-steps", type=int, default=3)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=33)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--stack-depth", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--prefill-repeats", type=int, default=2)
    parser.add_argument("--stress-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--trace-region-size", type=int, default=67108864)
    parser.add_argument("--activation-file")
    parser.add_argument("--functional-sequence", action="store_true")
    parser.add_argument("--functional-repeats", type=int, default=3)
    parser.add_argument("--timing", action="store_true", help="Integrated timing only; requires diagnostics disabled")
    args = parser.parse_args()
    if not (
        args.recurrent_permute
        or args.collective_padding
        or args.direct_recurrent_sum
        or args.scan_variant
        or args.production
        or args.product_recurrent_sum
        or args.state_slab
    ):
        parser.error("Select at least one structural, direct recurrent sum, or scan hypothesis")
    if args.mode == "component" and (
        args.direct_recurrent_sum
        or args.scan_variant
        or args.production
        or args.product_recurrent_sum
        or args.state_slab
    ):
        parser.error(
            "Direct recurrent sum and scan hypotheses require integrated mode; use their focused component drivers"
        )
    if (
        args.context < args.length + args.decode_steps
        or min(
            args.batch,
            args.decode_steps,
            args.repeats,
            args.prefill_repeats,
            args.stress_repeats,
            args.stack_depth,
            args.component_steps,
        )
        < 1
    ):
        parser.error("Invalid context or nonpositive workload")
    if args.component_steps < 2 or args.decode_steps < 2:
        parser.error("Changing-input checks require at least two steps")
    if args.timing:
        if args.mode != "integrated":
            parser.error("--timing applies only to integrated mode")
        if any(
            os.environ.get(k) not in (None, "", "0")
            for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER")
        ):
            parser.error("--timing requires allocation tracking, watcher and profiler disabled")
    elif (
        os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1"
        or os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") != "0"
    ):
        parser.error("Correctness mode requires TRACKING=1 and SKIP_PROGRAM_CACHE=0 explicitly")
    args.mesh_build_kwargs = args.baseline_build_kwargs = "{}"
    args.mesh_projection_geometry = None
    args.profile = False
    if args.functional_sequence and (
        (args.length, args.decode_steps, args.batch, args.stack_depth) != (128, 128, 1, 1)
        or args.layer not in (0, 3)
        or args.functional_repeats < 2
    ):
        parser.error("Functional timing requires layer0/3,128-in/128-out,B1,depth1,>=2 repeats")
    args.functional_repeats = 3
    args.save_prefill_states = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    record(output)
    provenance_path = Path(str(output) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance["device"] = "Blackhole 1x4, serial selected-mesh baseline and candidate"
    repo = Path(__file__).resolve().parents[4]
    header = repo / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h"
    provenance.setdefault("native_source_sha256", {})[str(header.relative_to(repo))] = hashlib.sha256(
        header.read_bytes()
    ).hexdigest()
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    result = dict(
        passed=False,
        args=vars(args),
        trace_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
        selected_defaults_preserved=True,
    )
    try:
        (components if args.mode == "component" else integrated)(args, result)
        result["passed"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(result, indent=2) + "\n")
    print("STRUCTURAL_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
