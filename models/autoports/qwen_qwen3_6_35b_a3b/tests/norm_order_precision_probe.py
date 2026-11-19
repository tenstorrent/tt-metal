# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured broad same-input norm-order diagnostic; no decoder fallback."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def cpu_schedule(squares):
    blocks = squares.reshape(-1, 4, 16, 32)
    lanes = blocks[:, :, 0]
    for i in range(1, 16):
        lanes = lanes + blocks[:, :, i]
    total = lanes[:, 0]
    for i in range(1, 4):
        total = total + lanes[:, i]
    groups = total.reshape(-1, 4, 8)
    eight = groups[:, 0]
    for i in range(1, 4):
        eight = eight + groups[:, i]
    value = eight[:, 0]
    for i in range(1, 8):
        value = value + eight[:, i]
    return value.reshape(-1, 1) * (1 / 2048)


def initial_metrics():
    return dict(
        rows=0,
        mean_different_codes=0,
        mean_max_abs=0.0,
        mean_max_ulp=0,
        inverse_different_codes=0,
        inverse_max_ulp=0,
        first_inverse_different_positions=[],
        norm_different_codes=0,
        norm_affected_rows=0,
        norm_min_row_pcc=1.0,
        first_mean_different_positions=[],
        first_norm_different_positions=[],
        device_wall_seconds=0.0,
        cpu_wall_seconds=0.0,
    )


def accumulate(metrics, positions, mean_ref, mean_actual, norm_ref, norm_actual, inverse_ref=None, inverse_actual=None):
    if inverse_ref is not None:
        ir, ia = inverse_ref.float().contiguous().view(torch.int32).reshape(
            -1
        ), inverse_actual.float().contiguous().view(torch.int32).reshape(-1)
        different = ir != ia
        metrics["inverse_different_codes"] += int(different.sum())
        metrics["inverse_max_ulp"] = max(metrics["inverse_max_ulp"], int((ir.long() - ia.long()).abs().max()))
        metrics["first_inverse_different_positions"] = (
            metrics["first_inverse_different_positions"]
            + [int(p) for p, failed in zip(positions, different.tolist()) if failed]
        )[:20]
    mr, ma = mean_ref.float().reshape(-1), mean_actual.float().reshape(-1)
    nr, na = norm_ref.bfloat16().reshape(-1, 2048), norm_actual.bfloat16().reshape(-1, 2048)
    md = mr.contiguous().view(torch.int32) != ma.contiguous().view(torch.int32)
    nd = nr != na
    affected = nd.any(-1)
    metrics["rows"] += len(positions)
    metrics["mean_different_codes"] += int(md.sum())
    metrics["mean_max_abs"] = max(metrics["mean_max_abs"], float((mr - ma).abs().max()))
    metrics["mean_max_ulp"] = max(
        metrics["mean_max_ulp"], int((mr.view(torch.int32).long() - ma.view(torch.int32).long()).abs().max())
    )
    metrics["norm_different_codes"] += int(nd.sum())
    metrics["norm_affected_rows"] += int(affected.sum())
    a, b = nr.double(), na.double()
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    pcc = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
    metrics["norm_min_row_pcc"] = min(metrics["norm_min_row_pcc"], float(pcc.min()))
    for key, mask in (("first_mean_different_positions", md), ("first_norm_different_positions", affected)):
        metrics[key] = (metrics[key] + [int(p) for p, failed in zip(positions, mask.tolist()) if failed])[:20]


def main(args):
    torch.set_num_threads(8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    snapshot = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.0.input_layernorm.weight"
    with safe_open(snapshot / index[key], framework="pt", device="cpu") as f:
        gamma = 1 + f.get_tensor(key).float()
    eps = json.loads((snapshot / "config.json").read_text())["text_config"]["rms_norm_eps"]
    saved = torch.load(args.stream, map_location="cpu", weights_only=False)
    stream = torch.cat([saved["prefix"], saved["tokens"]], dim=1).reshape(-1, 2048)
    positions = list(range(len(stream))) if args.all_stream else sorted(set(range(128)) | set(args.positions))
    if min(positions) < 0 or max(positions) >= len(stream):
        raise ValueError("requested position outside saved stream")
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    result = dict(
        checkpoint_revision=REVISION,
        stream_sha256=hashlib.sha256(Path(args.stream).read_bytes()).hexdigest(),
        purpose="Same-input reduction-order diagnostic; wall time includes cold dispatch/compilation, not warmed decoder latency",
        execution="CPU-only" if args.oracle_only else "serial physical chip 0",
        rows=len(positions),
        chunk_size=args.chunk_size,
        positions="all stream" if args.all_stream else positions,
        cpu_schedule=initial_metrics(),
        variants={},
    )
    # Independent CPU schedule validation runs before opening a device.
    for start in range(0, len(positions), args.chunk_size):
        pos = positions[start : start + args.chunk_size]
        x = stream[pos].float()
        ref_mean = x.square().mean(-1, keepdim=True)
        ref_norm = (x * torch.rsqrt(ref_mean + eps) * gamma).bfloat16()
        begin = time.perf_counter()
        mean = cpu_schedule(x.square())
        norm = (x * torch.rsqrt(mean + eps) * gamma).bfloat16()
        result["cpu_schedule"]["cpu_wall_seconds"] += time.perf_counter() - begin
        accumulate(result["cpu_schedule"], pos, ref_mean, mean, ref_norm, norm)
    assert result["cpu_schedule"]["mean_different_codes"] == 0, result["cpu_schedule"]
    assert result["cpu_schedule"]["norm_different_codes"] == 0, result["cpu_schedule"]
    if args.oracle_only:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return

    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(value):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        tg = upload(gamma.reshape(1, 1, 2048))

        def tt_schedule(squares, count):
            blocks = ttnn.reshape(squares, (count, 4, 16, 32))
            lanes = blocks[:, :, :1, :]
            for i in range(1, 16):
                lanes = ttnn.add(lanes, blocks[:, :, i : i + 1, :])
            total = lanes[:, :1, :, :]
            for i in range(1, 4):
                total = ttnn.add(total, lanes[:, i : i + 1, :, :])
            groups = ttnn.reshape(total, (count, 1, 4, 8))
            eight = groups[:, :, :1, :]
            for i in range(1, 4):
                eight = ttnn.add(eight, groups[:, :, i : i + 1, :])
            value = eight[:, :, :, :1]
            for i in range(1, 8):
                value = ttnn.add(value, eight[:, :, :, i : i + 1])
            return ttnn.reshape(ttnn.multiply(value, 1 / 2048), (1, count, 1))

        def variant_key(reduction, inverse):
            return reduction if args.inverse_variants == ["rsqrt"] else reduction + "/" + inverse

        def details(expected, actual):
            e, a = expected.float().contiguous().reshape(-1), actual.float().contiguous().reshape(-1)
            eb, ab = e.view(torch.int32), a.view(torch.int32)
            different = torch.where(eb != ab)[0].tolist()
            selected = list(dict.fromkeys(different[:12] + ([239] if e.numel() > 239 else [0])))
            return dict(
                different_codes=len(different),
                max_abs=float((e - a).abs().max()),
                samples=[
                    dict(
                        index=i,
                        expected=float(e[i]),
                        actual=float(a[i]),
                        expected_bits=int(eb[i]),
                        actual_bits=int(ab[i]),
                    )
                    for i in selected
                ],
            )

        result["intermediates"] = []
        for variant in args.variants:
            for inverse_variant in args.inverse_variants:
                result["variants"][variant_key(variant, inverse_variant)] = initial_metrics()
        for start in range(0, len(positions), args.chunk_size):
            pos = positions[start : start + args.chunk_size]
            x = stream[pos].float()
            ref_squares = x.square()
            ref_mean = ref_squares.mean(-1, keepdim=True)
            ref_shifted = ref_mean + eps
            ref_inv = torch.rsqrt(ref_shifted)
            ref_normalized = x * ref_inv
            ref_weighted = ref_normalized * gamma
            ref_norm = ref_weighted.bfloat16()
            tx = upload(x.reshape(1, len(pos), 2048))
            for variant in args.variants:
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                squares = ttnn.multiply(tx, tx)
                if variant == "cpu8":
                    mean = tt_schedule(squares, len(pos))
                elif variant == "two_level":
                    tiles = ttnn.reshape(squares, (len(pos), 1, 64, 32))
                    partial = ttnn.sum(tiles, dim=-1, keepdim=True, compute_kernel_config=compute)
                    summed = ttnn.sum(partial, dim=-2, keepdim=True, compute_kernel_config=compute)
                    mean = ttnn.reshape(ttnn.multiply(summed, 1 / 2048), (1, len(pos), 1))
                else:
                    mean = ttnn.mean(squares, dim=-1, keepdim=True, compute_kernel_config=compute)
                shifted = ttnn.add(mean, eps)
                ttnn.synchronize_device(mesh)
                mean_seconds = time.perf_counter() - begin
                got_mean = ttnn.to_torch(mean)
                selected = [(i, p) for i, p in enumerate(pos) if p in args.intermediate_positions]
                for inverse_variant in args.inverse_variants:
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    if inverse_variant == "rsqrt":
                        inv = ttnn.rsqrt(shifted, fast_and_approximate_mode=False)
                    elif inverse_variant == "reciprocal_sqrt":
                        inv = ttnn.reciprocal(ttnn.sqrt(shifted, fast_and_approximate_mode=False))
                    elif inverse_variant == "newton_rsqrt":
                        estimate = ttnn.rsqrt(shifted, fast_and_approximate_mode=False)
                        correction = ttnn.add(
                            ttnn.multiply(ttnn.multiply(shifted, ttnn.multiply(estimate, estimate)), -0.5), 1.5
                        )
                        inv = ttnn.multiply(estimate, correction)
                    elif inverse_variant == "split_product":

                        def split(value):
                            high = ttnn.typecast(ttnn.typecast(value, ttnn.bfloat16), ttnn.float32)
                            return high, ttnn.subtract(value, high)

                        root = ttnn.sqrt(shifted, fast_and_approximate_mode=False)
                        high, low = split(root)
                        residual = ttnn.subtract(
                            ttnn.subtract(
                                ttnn.subtract(shifted, ttnn.multiply(high, high)),
                                ttnn.multiply(ttnn.multiply(high, 2.0), low),
                            ),
                            ttnn.multiply(low, low),
                        )
                        root = ttnn.add(root, ttnn.multiply(residual, ttnn.multiply(ttnn.reciprocal(root), 0.5)))
                        inv = ttnn.reciprocal(root)
                        rh, rl = split(root)
                        ih, il = split(inv)
                        ones = ttnn.ones_like(shifted)
                        residual = ttnn.subtract(
                            ttnn.subtract(
                                ttnn.subtract(ttnn.subtract(ones, ttnn.multiply(rh, ih)), ttnn.multiply(rh, il)),
                                ttnn.multiply(rl, ih),
                            ),
                            ttnn.multiply(rl, il),
                        )
                        inv = ttnn.add(inv, ttnn.multiply(inv, residual))
                    else:
                        # Probe the native MAD residual; cancellation accuracy is measured separately.
                        root = ttnn.sqrt(shifted, fast_and_approximate_mode=False)
                        residual = ttnn.addcmul(shifted, ttnn.neg(root), root)
                        root = ttnn.add(root, ttnn.multiply(residual, ttnn.multiply(ttnn.reciprocal(root), 0.5)))
                        inv = ttnn.reciprocal(root)
                        if inverse_variant == "refined_sqrt_reciprocal":
                            ones = ttnn.ones_like(shifted)
                            reciprocal_residual = ttnn.addcmul(ones, ttnn.neg(root), inv)
                            inv = ttnn.add(inv, ttnn.multiply(inv, reciprocal_residual))
                    normalized = ttnn.multiply(tx, inv)
                    weighted = ttnn.multiply(normalized, tg)
                    norm = ttnn.typecast(weighted, ttnn.bfloat16)
                    ttnn.synchronize_device(mesh)
                    label = variant_key(variant, inverse_variant)
                    result["variants"][label]["device_wall_seconds"] += mean_seconds + time.perf_counter() - begin
                    got_inv, got_norm = ttnn.to_torch(inv), ttnn.to_torch(norm)
                    accumulate(result["variants"][label], pos, ref_mean, got_mean, ref_norm, got_norm, ref_inv, got_inv)
                    if selected:
                        tensors = dict(
                            squares=(ref_squares, ttnn.to_torch(squares)),
                            shifted=(ref_shifted, ttnn.to_torch(shifted)),
                            inverse=(ref_inv, got_inv),
                            normalized=(ref_normalized, ttnn.to_torch(normalized)),
                            weighted=(ref_weighted, ttnn.to_torch(weighted)),
                        )
                        for i, position in selected:
                            stages = {
                                name: details(expected.reshape(len(pos), -1)[i], actual.reshape(len(pos), -1)[i])
                                for name, (expected, actual) in tensors.items()
                            }
                            result["intermediates"].append(
                                dict(
                                    position=position,
                                    variant=label,
                                    stages=stages,
                                    first_difference=next(
                                        (name for name, value in stages.items() if value["different_codes"]), None
                                    ),
                                )
                            )
                    del inv, normalized, weighted, norm
                del squares, mean, shifted
            del tx
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print("NORM_ORDER", start + len(pos), len(positions), json.dumps(result["variants"]), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default=str(DOC / "endurance/mlp_norm_shard_8192_0_results.json.tensors.pt"))
    parser.add_argument("--positions", nargs="+", type=int, default=[128, 130, 3032, 3979, 5011, 6595, 7703])
    parser.add_argument("--all-stream", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--variants", nargs="+", choices=["cpu8", "mean", "two_level"], default=["cpu8", "mean", "two_level"]
    )
    parser.add_argument(
        "--inverse-variants",
        nargs="+",
        choices=[
            "rsqrt",
            "reciprocal_sqrt",
            "newton_rsqrt",
            "refined_sqrt",
            "refined_sqrt_reciprocal",
            "split_product",
        ],
        default=["rsqrt"],
    )
    parser.add_argument("--intermediate-positions", nargs="*", type=int, default=[])
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
