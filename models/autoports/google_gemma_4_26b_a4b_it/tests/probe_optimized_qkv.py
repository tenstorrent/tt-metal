# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare QKV policies on the exact activation from a specified traced step.

The normal parity harness supplies real inputs, checkpoint weights, and cache
history. Diagnostic readbacks and component timing happen after a completed
layer replay, outside its audited forward. Instrumented layer time is not a
performance result. This probe does not change the harness acceptance threshold.
"""

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import BroadcastQKV, FusedDecoder, TiedQKV
from models.autoports.google_gemma_4_26b_a4b_it.tt.routing_precision import QKVLinear


def metrics(expected, actual):
    expected, actual = expected.double().flatten(), actual.double().flatten()
    delta = actual - expected
    left, right = expected - expected.mean(), actual - actual.mean()
    denominator = left.norm() * right.norm()
    return dict(
        pcc=float((left @ right) / denominator) if denominator else None,
        max_abs=float(delta.abs().max()),
        rms_error=float(delta.square().mean().sqrt()),
        relative_l2=float(delta.norm() / expected.norm()),
        finite=bool(actual.isfinite().all()),
        exact_equal=torch.equal(expected, actual),
    )


def split_projection(
    value,
    source,
    compute,
    terms,
    *,
    program=None,
    packed=False,
    evidence=None,
    operand_float32=False,
    weight=None,
    partitions=None,
    weight_components=None,
    lane_mask=None,
):
    residual = value
    pieces = []
    for index in range(terms):
        piece = ttnn.typecast(residual, ttnn.bfloat16)
        pieces.append(piece)
        if index + 1 < terms:
            residual = ttnn.subtract(residual, ttnn.typecast(piece, ttnn.float32))
    kwargs = dict(
        dtype=ttnn.float32,
        compute_kernel_config=compute,
        memory_config=source.decode_memory,
    )
    if program is not None:
        kwargs["program_config"] = program
    weight = source.weight if weight is None else weight
    operands = [ttnn.typecast(piece, ttnn.float32) for piece in pieces] if operand_float32 else pieces
    if lane_mask is not None:
        assert not operand_float32 and partitions is None and weight_components is None
        operands = [ttnn.mul(lane_mask, piece) for piece in operands]
        packed = True
    if weight_components is not None:
        assert not packed and partitions is None
        products = []
        for operand in operands:
            accumulated = None
            for component_weight in weight_components:
                partial = ttnn.linear(operand, component_weight, **kwargs)
                accumulated = (
                    partial
                    if accumulated is None
                    else ttnn.add(accumulated, partial, memory_config=source.decode_memory)
                )
            products.append(accumulated)
        result = products[0]
        for product in products[1:]:
            result = ttnn.add(result, product, memory_config=source.decode_memory)
    elif partitions is not None:
        assert not packed and program is None
        products = []
        for operand in operands:
            accumulated = None
            for start, end, partial_weight in partitions:
                partial = ttnn.linear(operand[..., start:end], partial_weight, **kwargs)
                accumulated = (
                    partial
                    if accumulated is None
                    else ttnn.add(accumulated, partial, memory_config=source.decode_memory)
                )
            products.append(accumulated)
        result = products[0]
        for product in products[1:]:
            result = ttnn.add(result, product, memory_config=source.decode_memory)
    elif packed:
        products = [ttnn.linear(ttnn.concat(operands, dim=-2), weight, **kwargs)]
        result = ttnn.sum(products[0], dim=-2, keepdim=True)
    else:
        products = [ttnn.linear(piece, weight, **kwargs) for piece in operands]
        result = products[0]
        for product in products[1:]:
            result = ttnn.add(result, product, memory_config=source.decode_memory)
    if evidence is not None:
        evidence.update(
            input_dtype=str(value.dtype),
            weight_dtype=str(weight.dtype),
            component_dtypes=[str(piece.dtype) for piece in pieces],
            matmul_input_dtypes=[str(piece.dtype) for piece in operands],
            matmul_output_dtypes=[str(product.dtype) for product in products],
            output_dtype=str(result.dtype),
            compute_config={
                field: str(getattr(compute, field))
                for field in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc")
            },
            program_config=str(program),
            components=pieces,
            products=products,
        )
        if partitions is not None:
            evidence.update(
                external_k=partitions[0][1] - partitions[0][0],
                external_partitions=len(partitions),
                external_weight_dtypes=sorted({str(part[2].dtype) for part in partitions}),
                external_partial_output_dtype=str(partial.dtype),
            )
        if weight_components is not None:
            evidence["weight_component_dtypes"] = [str(component.dtype) for component in weight_components]
            evidence["weight_partial_output_dtype"] = str(partial.dtype)
        if lane_mask is not None:
            evidence.update(
                lane_count=lane_mask.shape[-2],
                lane_mask_dtype=str(lane_mask.dtype),
                packed_matmul_rows=terms * lane_mask.shape[-2],
                lane_partition_rule="Row r retains K indices k with k % lane_count == r",
            )
    if hasattr(source, "kv_width"):
        result = ttnn.concat(
            (result, result[..., source.width - source.kv_width :]),
            dim=-1,
            memory_config=source.decode_memory,
        )
    return result


def load_saved_projection(path, layer_idx, mesh):
    """Upload a saved activation and only its real QKV checkpoint weights."""
    saved = torch.load(path, weights_only=True)
    if "weight" in saved:
        matrix = saved["weight"]
        kv_width = saved.get("kv_width")
        if kv_width is not None:
            matrix = torch.cat((matrix, matrix[..., -kv_width:]), dim=-1)
    else:
        config = run_decoder.AutoConfig.from_pretrained(Path(__file__).parent).text_config
        tied = config.layer_types[layer_idx] == "full_attention" and config.attention_k_eq_v
        prefix = f"model.language_model.layers.{layer_idx}.self_attn."
        names = ["q_proj.weight", "k_proj.weight"] + ([] if tied else ["v_proj.weight"])
        index = json.loads(
            Path(
                run_decoder.hf_hub_download(
                    run_decoder.MODEL, "model.safetensors.index.json", revision=run_decoder.REVISION
                )
            ).read_text()
        )
        weights = {}
        for shard in sorted({index["weight_map"][prefix + name] for name in names}):
            checkpoint = run_decoder.hf_hub_download(run_decoder.MODEL, shard, revision=run_decoder.REVISION)
            with run_decoder.safe_open(checkpoint, framework="pt", device="cpu") as handle:
                for name in names:
                    if index["weight_map"][prefix + name] == shard:
                        weights[name] = handle.get_tensor(prefix + name)
        ordered = [weights["q_proj.weight"], weights["k_proj.weight"]]
        ordered.append(weights["k_proj.weight"] if tied else weights["v_proj.weight"])
        matrix = torch.cat([weight.transpose(-2, -1) for weight in ordered], dim=-1)[None, None]
        kv_width = weights["k_proj.weight"].shape[0] if tied else None
    weight = ttnn.from_torch(matrix, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    source = BroadcastQKV(QKVLinear(weight, mesh), group_size=16384)
    if kv_width is not None:
        source = TiedQKV(source, kv_width)
    source.decode_memory = ttnn.L1_MEMORY_CONFIG
    activation = ttnn.from_torch(saved["activation"], device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    return source, activation, saved["position"]


def small_product_controls(host_value, host_weight, source, mesh):
    """Localize errors with one populated K lane and one-face sums at real geometry."""
    value = host_value.bfloat16().float()
    width = value.shape[-1]
    maximum = int(value.flatten().abs().argmax())
    compute = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    rows = []

    def compare(name, operand, host_matrix=host_weight, weight=None, config=compute):
        device_operand = ttnn.from_torch(operand, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        matrix = source.weight if weight is None else weight
        output = ttnn.linear(
            device_operand,
            matrix,
            dtype=ttnn.float32,
            compute_kernel_config=config,
            memory_config=source.decode_memory,
        )
        expected = operand.double() @ host_matrix.double()
        actual = ttnn.to_torch(output).float()
        delta = actual.double() - expected
        worst = int(delta.abs().flatten().argmax())
        row = dict(
            control=name,
            input_shape=list(device_operand.shape),
            input_dtype=str(device_operand.dtype),
            weight_dtype=str(matrix.dtype),
            output_dtype=str(output.dtype),
            fidelity=str(config.math_fidelity),
            nonzero_k=int(operand.count_nonzero()),
            comparison=metrics(expected, actual),
            worst_output_index=worst,
            worst_expected=float(expected.flatten()[worst]),
            worst_actual=float(actual.flatten()[worst]),
        )
        rows.append(row)
        print("QKV_SMALL_CONTROL", row, flush=True)
        device_operand.deallocate(True)
        output.deallocate(True)

    for index in sorted({0, 15, 16, 31, 32, maximum, width - 1}):
        for amplitude_name, amplitude in (("unit", 1.0), ("real", float(value.flatten()[index]))):
            operand = torch.zeros_like(value)
            operand[..., index] = amplitude
            compare(f"onehot_{index}_{amplitude_name}", operand)
    for start in sorted({0, maximum // 32 * 32}):
        for count in (2, 4, 8, 16, 32):
            operand = torch.zeros_like(value)
            operand[..., start : start + count] = value[..., start : start + count]
            compare(f"contiguous_{start}_{count}", operand)
    # SrcB's first fidelity phase retains six mantissa bits; SrcA's retains four.
    limited_value = (value.contiguous().view(torch.int32) & (-(1 << 17))).view(torch.float32)
    limited_weight = (host_weight.contiguous().view(torch.int32) & (-(1 << 19))).view(torch.float32)
    device_weight = ttnn.from_torch(limited_weight, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    for fidelity in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4):
        config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        compare(f"phase0_only_{str(fidelity).split('.')[-1]}", limited_value, limited_weight, device_weight, config)
    device_weight.deallocate(True)
    return rows


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-position", type=int, default=4149)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--activation-output", type=Path)
    parser.add_argument("--activation-input", type=Path)
    parser.add_argument("--qkv-terms", default="2,3")
    parser.add_argument("--qkv-blocks", default="1,4,8,11,22")
    parser.add_argument("--qkv-grids", default="8x8")
    parser.add_argument("--qkv-packed-rows", action="store_true")
    parser.add_argument("--qkv-product-details", action="store_true")
    parser.add_argument("--qkv-float32-operands", action="store_true")
    parser.add_argument("--qkv-external-k", default="", help="Comma-separated tile-aligned K partition widths")
    parser.add_argument("--qkv-weight-mantissa", default="", help="Retained high-weight mantissa bits; exact residual")
    parser.add_argument("--qkv-fidelity-controls", default="", help="Additional original-weight fidelity controls")
    parser.add_argument("--qkv-product-output", type=Path, help="Save exact host operands and component outputs")
    parser.add_argument("--qkv-small-controls", action="store_true")
    parser.add_argument("--qkv-controls-only", action="store_true")
    parser.add_argument("--qkv-lanes", default="", help="Partition dot-product lanes into independent M rows")
    parser.add_argument("--qkv-timing-repeats", type=int, default=10)
    parser.add_argument("--harness-qkv", choices=("broadcast", "split2"), default="split2")
    args, remaining = parser.parse_known_args()
    if "--profile" in remaining or "--diagnostic" in remaining:
        parser.error("QKV instrumentation needs ordinary per-position correctness readbacks")
    terms = [int(item) for item in args.qkv_terms.split(",")]
    blocks = [int(item) for item in args.qkv_blocks.split(",") if item]
    external_widths = [int(item) for item in args.qkv_external_k.split(",") if item]
    weight_mantissas = [int(item) for item in args.qkv_weight_mantissa.split(",") if item]
    lane_counts = [int(item) for item in args.qkv_lanes.split(",") if item]
    grids = [tuple(map(int, item.split("x"))) for item in args.qkv_grids.split(",")]
    if any(term < 1 for term in terms) or args.qkv_timing_repeats < 1:
        parser.error("Terms and timing repetitions must be positive")
    if "--decoder" not in remaining:
        remaining = ["--decoder", "fused", *remaining]
    sys.argv = [sys.argv[0], *remaining]
    references, captured = {}, {}
    report = dict(
        position=args.probe_position,
        harness_qkv=args.harness_qkv,
        scope="Same-input component comparison; instrumented layer timing is not a performance claim",
        candidates=[],
        captured=False,
    )
    original_factory = FusedDecoder.from_state_dict.__func__
    original_load = run_decoder.load_layer
    original_compare = run_decoder.comp_pcc
    next_position = 0

    def write_report():
        args.probe_output.write_text(json.dumps(report, indent=2) + "\n")

    def load(*a, **kw):
        layer = original_load(*a, **kw)
        original_forward = layer.forward

        def forward(value, *a, **kw):
            nonlocal next_position
            result = original_forward(value, *a, **kw)
            if value.shape[-2] > 1:
                next_position = value.shape[-2]
            else:
                references[result.data_ptr()] = next_position
                next_position += 1
            return result

        layer.forward = forward
        return layer

    def factory(cls, *a, **kw):
        result = original_factory(cls, *a, **kw)
        attention = result.layer.self_attn.source
        source = attention.weights.wqkv
        captured.update(source=source, mesh=kw["mesh_device"])

        class Projection:
            def __call__(self, value, compute_kernel_config=None, out_memory_config=None):
                if value.shape[-2] != 1:
                    return source(value, compute_kernel_config, out_memory_config)
                # This persistent device buffer is rewritten by each trace replay.
                captured["activation"] = ttnn.clone(value)
                if args.harness_qkv == "split2":
                    return split_projection(value, source, source.compute, 2)
                return source(value)

        attention.weights = replace(attention.weights, wqkv=Projection())
        return result

    def investigate():
        source, mesh = captured["source"], captured["mesh"]
        value = captured["activation"]
        host_value = ttnn.to_torch(value).float()
        host_weight = ttnn.to_torch(source.weight).float()
        expected = host_value.double() @ host_weight.double()
        if hasattr(source, "kv_width"):
            expected = torch.cat((expected, expected[..., source.width - source.kv_width :]), dim=-1)
        baseline = source(value)
        broadcast = ttnn.to_torch(baseline).float()
        baseline.deallocate(True)
        report.update(
            captured=True,
            activation_shape=list(value.shape),
            activation_dtype=str(value.dtype),
            weight_shape=list(source.weight.shape),
            weight_dtype=str(source.weight.dtype),
            weight_values_exactly_bf16=torch.equal(host_weight, host_weight.bfloat16().float()),
            output_memory=str(source.decode_memory),
            broadcast_vs_cpu_fp64=metrics(expected, broadcast),
        )
        if args.activation_output:
            torch.save(
                dict(
                    position=report["position"],
                    activation=host_value,
                    weight=host_weight.bfloat16(),
                    kv_width=getattr(source, "kv_width", None),
                ),
                args.activation_output,
            )
        write_report()
        if args.qkv_small_controls:
            report["small_product_controls"] = small_product_controls(host_value, host_weight, source, mesh)
            write_report()
        if args.qkv_controls_only:
            assert args.qkv_small_controls, "Controls-only requires --qkv-small-controls"
            return
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        configurations = [("default", None)]
        for gx, gy in grids:
            for block in blocks:
                assert value.shape[-1] // 32 % block == 0, "K block must divide 88 tiles"
                per_core_n = math.ceil(source.weight.shape[-1] / 32 / (gx * gy))
                program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    in0_block_w=block,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=per_core_n,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                configurations.append((f"1d_{gx}x{gy}_k{block}", program))
        wide_weight = ttnn.typecast(source.weight, ttnn.float32) if args.qkv_float32_operands else None
        lane_masks = []
        for lanes in lane_counts:
            assert lanes > 0 and value.shape[-1] % lanes == 0
            mask = (torch.arange(value.shape[-1])[None, :] % lanes) == torch.arange(lanes)[:, None]
            device_mask = ttnn.from_torch(
                mask[None, None].bfloat16(), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            lane_masks.append((lanes, device_mask))
        weight_component_sets = []
        for bits in weight_mantissas:
            assert 0 <= bits <= 7
            high_weight = (host_weight.contiguous().view(torch.int32) & (-(1 << (23 - bits)))).view(torch.float32)
            low_weight = host_weight - high_weight
            assert torch.equal(high_weight.double() + low_weight.double(), host_weight.double())
            assert torch.equal(high_weight, high_weight.bfloat16().float())
            assert torch.equal(low_weight, low_weight.bfloat16().float())
            components = tuple(
                ttnn.from_torch(part, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                for part in (high_weight, low_weight)
            )
            weight_component_sets.append((bits, components))
        fidelity_controls = []
        for fidelity in args.qkv_fidelity_controls.split(","):
            if fidelity:
                fidelity_controls.append(
                    (
                        fidelity,
                        ttnn.init_device_compute_kernel_config(
                            mesh.arch(),
                            math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                            math_approx_mode=False,
                            fp32_dest_acc_en=True,
                            packer_l1_acc=False,
                        ),
                    )
                )
        partition_sets = []
        for width in external_widths:
            assert width > 0 and width % 32 == 0 and value.shape[-1] % width == 0
            partitions = tuple(
                (start, start + width, source.weight[..., start : start + width, :])
                for start in range(0, value.shape[-1], width)
            )
            partition_sets.append((width, partitions))

        def plain():
            result = ttnn.linear(
                value,
                source.weight,
                dtype=ttnn.float32,
                compute_kernel_config=compute,
                memory_config=source.decode_memory,
            )
            if hasattr(source, "kv_width"):
                result = ttnn.concat(
                    (result, result[..., source.width - source.kv_width :]),
                    dim=-1,
                    memory_config=source.decode_memory,
                )
            return result

        candidates = [("broadcast", lambda evidence=None: source(value)), ("plain_fp32", lambda evidence=None: plain())]
        for count in terms:
            for config_name, program in configurations:
                for packed in [False, True] if args.qkv_packed_rows else [False]:
                    name = f"split{count}_{config_name}" + ("_packed" if packed else "")

                    def candidate(evidence=None, count=count, program=program, packed=packed):
                        return split_projection(
                            value, source, compute, count, program=program, packed=packed, evidence=evidence
                        )

                    candidates.append((name, candidate))
            if args.qkv_float32_operands:
                for weight_name, weight in (("bf16_w", source.weight), ("fp32_w", wide_weight)):

                    def widened(evidence=None, count=count, weight=weight):
                        return split_projection(
                            value, source, compute, count, evidence=evidence, operand_float32=True, weight=weight
                        )

                    candidates.append((f"split{count}_fp32_x_{weight_name}", widened))
            for width, partitions in partition_sets:

                def external(evidence=None, count=count, partitions=partitions):
                    return split_projection(value, source, compute, count, evidence=evidence, partitions=partitions)

                candidates.append((f"split{count}_external_k{width}", external))
            for bits, weight_components in weight_component_sets:

                def weight_split(evidence=None, count=count, weight_components=weight_components, bits=bits):
                    result = split_projection(
                        value, source, compute, count, evidence=evidence, weight_components=weight_components
                    )
                    if evidence is not None:
                        evidence.update(weight_high_mantissa_bits=bits, weight_reconstruction_exact=True)
                    return result

                candidates.append((f"split{count}_weight_mantissa{bits}", weight_split))
            for fidelity, control_compute in fidelity_controls:

                def fidelity_control(evidence=None, count=count, control_compute=control_compute):
                    return split_projection(value, source, control_compute, count, evidence=evidence)

                candidates.append((f"split{count}_{fidelity}", fidelity_control))
            for lanes, lane_mask in lane_masks:

                def lane_partition(evidence=None, count=count, lane_mask=lane_mask):
                    return split_projection(value, source, compute, count, evidence=evidence, lane_mask=lane_mask)

                candidates.append((f"split{count}_lanes{lanes}", lane_partition))
        product_snapshot = dict(
            position=report["position"], activation=host_value, weight=host_weight.bfloat16(), candidates=[]
        )
        for name, candidate in candidates:
            evidence = {}
            with device_only():
                warm = candidate(evidence)
            actual = ttnn.to_torch(warm).float()
            evidence.setdefault("input_dtype", str(value.dtype))
            evidence.setdefault("weight_dtype", str(source.weight.dtype))
            evidence.setdefault("output_dtype", str(warm.dtype))
            components = evidence.pop("components", [])
            products = evidence.pop("products", [])
            if components:
                host_components = [ttnn.to_torch(piece).double() for piece in components]
                reconstruction = sum(host_components)
                evidence["activation_reconstruction"] = metrics(host_value, reconstruction)
                if args.qkv_product_details or args.qkv_product_output:
                    host_products = [ttnn.to_torch(product).float() for product in products]
                    if len(host_products) == 1 and host_products[0].shape[-2] > 1:
                        group_rows = evidence.get("lane_count", 1)
                        host_products = [
                            group.sum(dim=-2, keepdim=True) for group in host_products[0].split(group_rows, dim=-2)
                        ]
                    evidence["product_details"] = []
                    for component, product in zip(host_components, host_products):
                        expected_product = component @ host_weight.double()
                        expected_fp32 = expected_product.float()
                        truncated = (expected_fp32.view(torch.int32) & -8192).view(torch.float32)
                        bits = product.contiguous().view(torch.int32)
                        evidence["product_details"].append(
                            dict(
                                versus_cpu=metrics(expected_product, product),
                                versus_cpu_tf32_output=metrics(truncated, product),
                                zero_low_13_bit_fraction=float(((bits & 8191) == 0).float().mean()),
                                mean_signed_error=float((product.double() - expected_product).mean()),
                            )
                        )
                    if args.qkv_product_output:
                        product_snapshot["candidates"].append(
                            dict(candidate=name, components=host_components, products=host_products, result=actual)
                        )
                        torch.save(product_snapshot, args.qkv_product_output)
            components.clear()
            products.clear()
            warm.deallocate(True)
            row = dict(
                candidate=name,
                versus_cpu_fp64=metrics(expected, actual),
                versus_broadcast=metrics(broadcast, actual),
                **evidence,
            )
            report["candidates"].append(row)
            write_report()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with device_only():
                traced = candidate()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            try:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                replay = ttnn.to_torch(traced).float()
                row["replay_equal"] = torch.equal(actual, replay)
                times = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    for _ in range(args.qkv_timing_repeats):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter_ns() - start) / (1000 * args.qkv_timing_repeats))
                row["traced_component_host_us"] = statistics.median(times)
                row["traced_component_samples_us"] = times
            finally:
                ttnn.release_trace(mesh, trace)
                traced.deallocate(True)
            write_report()
            print("QKV_PROBE", row, flush=True)

    def compare(reference, actual, *a, **kw):
        result = original_compare(reference, actual, *a, **kw)
        if references.get(reference.data_ptr()) == args.probe_position and not report["captured"]:
            report["layer_check"] = dict(passed=bool(result[0]), pcc=float(result[1]))
            investigate()
        return result

    try:
        if args.activation_input:
            saved_parser = argparse.ArgumentParser(add_help=False)
            saved_parser.add_argument("--layer", type=int, default=0)
            saved_args, _ = saved_parser.parse_known_args(remaining)
            torch.set_num_threads(8)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
            try:
                source, value, position = load_saved_projection(args.activation_input, saved_args.layer, mesh)
                report.update(
                    position=position,
                    activation_input=str(args.activation_input),
                    standalone_component_only=True,
                    layer=saved_args.layer,
                )
                captured.update(source=source, activation=value, mesh=mesh)
                investigate()
            finally:
                ttnn.close_mesh_device(mesh)
            return
        with (
            patch.object(run_decoder, "load_layer", load),
            patch.object(run_decoder, "comp_pcc", compare),
            patch.object(FusedDecoder, "from_state_dict", classmethod(factory)),
        ):
            run_decoder.main()
        assert report["captured"], "The requested position was not reached by the harness"
    except BaseException as exc:
        report["harness_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_report()


if __name__ == "__main__":
    main()
