# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured same-input input-norm/QK-L2 diagnostic; no decoder fallback."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def compare(expected, actual):
    actual = actual.reshape(expected.shape)
    a, b = expected.double().flatten(), actual.double().flatten()
    ac, bc = a - a.mean(), b - b.mean()
    return dict(
        pcc=1.0 if torch.equal(a, b) else float(ac @ bc / (ac.norm() * bc.norm())),
        relative_l2=float((a - b).norm() / a.norm()),
        max_abs=float((a - b).abs().max()),
        bf16_different_codes=int((expected.bfloat16() != actual.bfloat16()).sum()),
        elements=a.numel(),
        actual_dtype=str(actual.dtype),
    )


def load_cpu_cases(args):
    snapshot = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    names = ("input_layernorm.weight", "linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")
    weights = {}
    for key in names:
        name = "model.language_model.layers.0." + key
        with safe_open(snapshot / index[name], framework="pt", device="cpu") as f:
            weights[key] = f.get_tensor(name)
    eps = json.loads((snapshot / "config.json").read_text())["text_config"]["rms_norm_eps"]
    gamma = 1 + weights["input_layernorm.weight"].float()
    saved = torch.load(args.stream, map_location="cpu", weights_only=False)
    stream = torch.cat([saved["prefix"], saved["tokens"]], dim=1)
    cases = []
    for position in args.positions:
        if position < 3 or position >= stream.shape[1]:
            raise ValueError(f"position {position} outside four-token stream window")
        window = stream[:, position - 3 : position + 1]
        xf = window.float()
        norm = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * gamma).bfloat16()
        # Four independently supplied external tokens give the exact depthwise conv history.
        projected = torch.cat(
            [
                torch.nn.functional.linear(norm[:, i : i + 1], weights["linear_attn.in_proj_qkv.weight"])
                for i in range(4)
            ],
            dim=1,
        )
        preactivation = torch.nn.functional.conv1d(
            projected.transpose(1, 2), weights["linear_attn.conv1d.weight"], groups=8192
        )
        conv = torch.nn.functional.silu(preactivation).transpose(1, 2)
        qk = torch.cat([conv[..., :2048], conv[..., 2048:4096]], dim=0).reshape(2, 1, 16, 128)
        squares = qk * qk
        sums = squares.sum(-1, keepdim=True)
        shifted = sums + 1e-6
        inverse = torch.rsqrt(shifted)
        output = qk * inverse
        case = dict(
            position=position,
            input=window[:, -1:].contiguous(),
            norm=norm[:, -1:].contiguous(),
            qk=qk,
            conv_preactivation=preactivation.transpose(1, 2).contiguous(),
            conv_silu=conv.contiguous(),
            l2=dict(squares=squares, sums=sums, shifted=shifted, inverse=inverse, output=output),
            norm_intermediates=dict(
                squares=xf[:, -1:].square(),
                variance=xf[:, -1:].square().mean(-1, keepdim=True),
                shifted=xf[:, -1:].square().mean(-1, keepdim=True) + eps,
                inverse=torch.rsqrt(xf[:, -1:].square().mean(-1, keepdim=True) + eps),
                normalized=xf[:, -1:] * torch.rsqrt(xf[:, -1:].square().mean(-1, keepdim=True) + eps),
                weighted=xf[:, -1:] * torch.rsqrt(xf[:, -1:].square().mean(-1, keepdim=True) + eps) * gamma,
            ),
            cpu_validation={},
        )
        capture = DOC / f"endurance_state_probe.json.pos{position}.pt"
        if capture.exists():
            probe = torch.load(capture, map_location="cpu", weights_only=False)
            case["cpu_validation"]["external_token"] = compare(probe["token"], case["input"])
            case["cpu_validation"]["input_norm"] = compare(probe["hf_values"]["input_norm"], case["norm"])
            reference_qk = torch.cat(
                [probe["hf_values"]["delta_q"][:, :, ::2], probe["hf_values"]["delta_k"][:, :, ::2]], dim=0
            )
            case["cpu_validation"]["qk_l2"] = compare(reference_qk, output)
            assert all(
                case["cpu_validation"][key]["bf16_different_codes"] == 0 for key in ("external_token", "input_norm")
            ), case["cpu_validation"]
            # The four-token conv has a different CPU kernel shape from HF's cache4+token
            # conv; record any rounding differences rather than replacing the oracle.
            assert case["cpu_validation"]["qk_l2"]["pcc"] > 0.9999, case["cpu_validation"]
        cases.append(case)
    return cases, gamma, eps


def run(args):
    torch.set_num_threads(4)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    cases, gamma, eps = load_cpu_cases(args)
    result = dict(
        purpose="same-input numerical microprobe; no timing or full-decoder correctness claim",
        checkpoint_revision=REVISION,
        stream=str(Path(args.stream).resolve()),
        stream_sha256=hashlib.sha256(Path(args.stream).read_bytes()).hexdigest(),
        execution="CPU oracle only" if args.oracle_only else "CPU oracle plus serial physical device 0",
        qk_oracle_note="Four-token CPU convolution; any difference from captured HF cache4+token convolution is reported. All L2 candidates receive the identical reconstructed BF16 input.",
        cases=[],
    )
    if args.oracle_only:
        result["cases"] = [dict(position=c["position"], cpu_validation=c["cpu_validation"]) for c in cases]
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    import ttnn

    norm_diagnostics = []
    diagnostic_output = args.norm_diagnostic_output or args.output + ".norm_diagnostics.json"
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        compute = ttnn.init_device_compute_kernel_config(
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

        def read(x):
            return ttnn.to_torch(x)

        def round_widen(x):
            return ttnn.typecast(ttnn.typecast(x, ttnn.bfloat16), ttnn.float32)

        tg = upload(gamma.reshape(1, 1, -1), ttnn.float32)
        l2weight = upload(torch.full((1, 1, 128), 128**-0.5), ttnn.float32)
        cores = ttnn.num_cores_to_corerangeset(32, mesh.compute_with_storage_grid_size(), row_wise=True)
        shard = ttnn.create_sharded_memory_config(
            (32, 64),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        for case in cases:
            row = dict(
                position=case["position"],
                cpu_validation=case["cpu_validation"],
                input_norm={},
                qk_l2={},
                conv_silu={},
                isolated_multiply={},
            )
            x = upload(case["input"])
            for label, inp in (("native_interleaved", x), ("native_width_sharded", ttnn.to_memory_config(x, shard))):
                out = ttnn.rms_norm(
                    inp, weight=tg, epsilon=eps, compute_kernel_config=compute, memory_config=inp.memory_config()
                )
                row["input_norm"][label] = compare(case["norm"], read(out))
            x32 = ttnn.typecast(x, ttnn.float32)
            norm_squares = ttnn.multiply(x32, x32)
            variance = ttnn.mean(norm_squares, dim=-1, keepdim=True, compute_kernel_config=compute)
            norm_shifted = ttnn.add(variance, eps)
            inv = ttnn.rsqrt(norm_shifted)
            normalized = ttnn.multiply(x32, inv)
            weighted = ttnn.multiply(normalized, tg)
            exact = ttnn.typecast(weighted, ttnn.bfloat16)
            row["input_norm"]["explicit_fp32_sfpu"] = compare(case["norm"], read(exact))
            if args.norm_diagnostic:

                def detail(expected, actual):
                    actual = actual.reshape(expected.shape).float().contiguous()
                    expected = expected.float().contiguous()
                    eb, ab = expected.view(torch.int32).flatten(), actual.view(torch.int32).flatten()
                    indices = torch.where(eb != ab)[0].tolist()
                    details = dict(compare(expected, actual), float32_different_codes=len(indices))
                    selected = list(dict.fromkeys(indices[:16] + ([239] if expected.numel() > 239 else [0])))
                    details["samples"] = [
                        dict(
                            index=i,
                            expected=float(expected.flatten()[i]),
                            actual=float(actual.flatten()[i]),
                            expected_bits=int(eb[i]),
                            actual_bits=int(ab[i]),
                        )
                        for i in selected
                    ]
                    return details

                intermediates = dict(
                    squares=norm_squares,
                    variance=variance,
                    shifted=norm_shifted,
                    inverse=inv,
                    normalized=normalized,
                    weighted=weighted,
                )
                diag = dict(
                    position=case["position"],
                    baseline_intermediates={
                        key: detail(case["norm_intermediates"][key], read(value))
                        for key, value in intermediates.items()
                    },
                    alternatives={},
                )
                diag["first_fp32_difference"] = next(
                    (key for key, value in diag["baseline_intermediates"].items() if value["float32_different_codes"]),
                    None,
                )
                sum_mean = ttnn.multiply(
                    ttnn.sum(norm_squares, dim=-1, keepdim=True, compute_kernel_config=compute), 1 / 2048
                )
                # Explicit two-level SFPU reduction changes summation order without FPU narrowing.
                tiles = ttnn.reshape(norm_squares, (1, 1, 64, 32))
                partial = ttnn.sum(tiles, dim=-1, keepdim=True, compute_kernel_config=compute)
                tiled_sum = ttnn.sum(partial, dim=-2, keepdim=True, compute_kernel_config=compute)
                tiled_mean = ttnn.reshape(ttnn.multiply(tiled_sum, 1 / 2048), (1, 1, 1))
                for label, candidate_variance in (("sum_then_scale", sum_mean), ("two_level_64x32", tiled_mean)):
                    shifted = ttnn.add(candidate_variance, eps)
                    candidate_inv = ttnn.rsqrt(shifted, fast_and_approximate_mode=False)
                    candidate_weighted = ttnn.multiply(ttnn.multiply(x32, candidate_inv), tg)
                    diag["alternatives"][label] = dict(
                        variance=detail(case["norm_intermediates"]["variance"], read(candidate_variance)),
                        inverse=detail(case["norm_intermediates"]["inverse"], read(candidate_inv)),
                        weighted=detail(case["norm_intermediates"]["weighted"], read(candidate_weighted)),
                        output=compare(case["norm"], read(ttnn.typecast(candidate_weighted, ttnn.bfloat16))),
                    )
                # Uploaded CPU shifted variance is a labeled same-input unary microprobe only.
                cpu_shifted = upload(case["norm_intermediates"]["shifted"], ttnn.float32)
                inverse_options = {
                    "reciprocal_sqrt": ttnn.reciprocal(ttnn.sqrt(norm_shifted, fast_and_approximate_mode=False)),
                    "rsqrt_on_cpu_shifted": ttnn.rsqrt(cpu_shifted, fast_and_approximate_mode=False),
                    "reciprocal_sqrt_on_cpu_shifted": ttnn.reciprocal(
                        ttnn.sqrt(cpu_shifted, fast_and_approximate_mode=False)
                    ),
                }
                for label, candidate_inv in inverse_options.items():
                    candidate_weighted = ttnn.multiply(ttnn.multiply(x32, candidate_inv), tg)
                    diag["alternatives"][label] = dict(
                        inverse=detail(case["norm_intermediates"]["inverse"], read(candidate_inv)),
                        weighted=detail(case["norm_intermediates"]["weighted"], read(candidate_weighted)),
                        output=compare(case["norm"], read(ttnn.typecast(candidate_weighted, ttnn.bfloat16))),
                    )
                norm_diagnostics.append(diag)
                Path(diagnostic_output).parent.mkdir(parents=True, exist_ok=True)
                Path(diagnostic_output).write_text(
                    json.dumps(
                        dict(
                            parent_result=args.output,
                            purpose="Unmeasured FP32 rounding localization; CPU-uploaded controls are labeled",
                            cases=norm_diagnostics,
                        ),
                        indent=2,
                    )
                    + "\n"
                )

            pre = upload(case["conv_preactivation"])
            row["conv_silu"]["native_bf16"] = compare(case["conv_silu"], read(ttnn.silu(pre)))
            pre32 = ttnn.typecast(pre, ttnn.float32)
            precise_silu = ttnn.typecast(ttnn.silu(pre32), ttnn.bfloat16)
            row["conv_silu"]["fp32_then_explicit_bf16"] = compare(case["conv_silu"], read(precise_silu))
            qk = upload(case["qk"])
            qkf = ttnn.typecast(qk, ttnn.float32)
            native = ttnn.rms_norm(qkf, weight=l2weight, epsilon=1e-6 / 128, compute_kernel_config=compute)
            row["qk_l2"]["native_rms_float32"] = compare(case["l2"]["output"], read(native))
            # Isolate output rounding: identical BF16 operands, including canonical inverse.
            cpu_inverse = upload(case["l2"]["inverse"])
            for label, rhs, expected in (
                ("square", qk, case["l2"]["squares"]),
                ("qk_times_inverse", cpu_inverse, case["l2"]["output"]),
            ):
                row["isolated_multiply"][label] = {
                    "default": compare(expected, read(ttnn.multiply(qk, rhs))),
                    "fast_and_approximate_false": compare(
                        expected, read(ttnn.multiply(qk, rhs, fast_and_approximate_mode=False))
                    ),
                    "fp32_then_explicit_bf16": compare(
                        expected,
                        read(ttnn.typecast(ttnn.multiply(qkf, ttnn.typecast(rhs, ttnn.float32)), ttnn.bfloat16)),
                    ),
                }
            # Same primitive BF16 boundaries as the prior full-stream L2 control.
            sq = ttnn.multiply(qk, qk)
            sums = ttnn.sum(sq, dim=-1, keepdim=True, compute_kernel_config=compute)
            shifted = ttnn.add(sums, 1e-6)
            inv = ttnn.rsqrt(shifted)
            out = ttnn.multiply(qk, inv)
            row["qk_l2"]["primitive_bf16"] = {
                key: compare(case["l2"][key], read(value))
                for key, value in dict(squares=sq, sums=sums, shifted=shifted, inverse=inv, output=out).items()
            }
            # FP32 SFPU arithmetic, explicitly reproducing every canonical BF16 result boundary.
            sq = round_widen(ttnn.multiply(qkf, qkf))
            sums = round_widen(ttnn.sum(sq, dim=-1, keepdim=True, compute_kernel_config=compute))
            shifted = round_widen(ttnn.add(sums, 1e-6))
            inv = round_widen(ttnn.rsqrt(shifted))
            out = round_widen(ttnn.multiply(qkf, inv))
            row["qk_l2"]["sfpu_explicit_bf16_boundaries"] = {
                key: compare(case["l2"][key], read(value))
                for key, value in dict(squares=sq, sums=sums, shifted=shifted, inverse=inv, output=out).items()
            }
            result["cases"].append(row)
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print("FRONTEND_PRECISION", json.dumps(row), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default=str(DOC / "endurance/mlp_norm_shard_8192_0_results.json.tensors.pt"))
    parser.add_argument("--positions", type=int, nargs="+", default=[128, 3032, 3979, 5011, 6595, 7703])
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--norm-diagnostic", action="store_true")
    parser.add_argument("--norm-diagnostic-output")
    run(parser.parse_args())
