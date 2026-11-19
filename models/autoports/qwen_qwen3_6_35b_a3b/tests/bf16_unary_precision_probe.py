# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive BF16-domain unary diagnostic; --oracle-only never imports TTNN."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record


def codes(x):
    return x.bfloat16().contiguous().view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).reshape(-1)


def json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def compare(inputs, reference, actual, limit):
    reference, actual = reference.bfloat16().reshape(-1), actual.bfloat16().reshape(-1)
    assert reference.shape == actual.shape == inputs.shape
    ic, rc, ac = codes(inputs), codes(reference), codes(actual)
    different = rc != ac
    valid = torch.isfinite(reference) & torch.isfinite(actual)
    normal_input = inputs.float().abs() >= torch.finfo(torch.bfloat16).tiny
    zero_input = inputs == 0
    subnormal_input = ~normal_input & ~zero_input
    numeric_different = valid & (reference != actual)

    # Monotonic BF16 magnitude codes, identifying +0/-0 for ULP distance only.
    def ordered(c):
        return torch.where(c.bitwise_and(0x8000) != 0, 0x8000 - c.bitwise_and(0x7FFF), 0x8000 + c)

    def examples(mask):
        result = []
        for i in mask.nonzero().reshape(-1)[:limit].tolist():
            result.append(
                dict(
                    input_code=f"0x{int(ic[i]):04x}",
                    input=json_number(inputs[i]),
                    reference_code=f"0x{int(rc[i]):04x}",
                    reference=json_number(reference[i]),
                    actual_code=f"0x{int(ac[i]):04x}",
                    actual=json_number(actual[i]),
                )
            )
        return result

    error = (reference[valid].double() - actual[valid].double()).abs()
    distance = (ordered(rc[valid]) - ordered(ac[valid])).abs()
    return dict(
        inputs=inputs.numel(),
        exact_code_matches=int((~different).sum()),
        exact_code_differences=int(different.sum()),
        numeric_comparisons=int(valid.sum()),
        numeric_differences=int(numeric_different.sum()),
        excluded_nonfinite_outputs=int((~valid).sum()),
        reference_nan_outputs=int(torch.isnan(reference).sum()),
        actual_nan_outputs=int(torch.isnan(actual).sum()),
        reference_inf_outputs=int(torch.isinf(reference).sum()),
        actual_inf_outputs=int(torch.isinf(actual).sum()),
        signed_zero_only_differences=int((different & valid & (reference == 0) & (actual == 0)).sum()),
        max_absolute_error=float(error.max()) if error.numel() else None,
        max_bf16_ulp_distance=int(distance.max()) if distance.numel() else None,
        input_class_counts={
            name: dict(
                inputs=int(mask.sum()),
                code_differences=int((different & mask).sum()),
                numeric_differences=int((numeric_different & mask).sum()),
            )
            for name, mask in (("normal", normal_input), ("subnormal", subnormal_input), ("zero", zero_input))
        },
        first_differences=examples(different),
        first_normal_input_differences=examples(different & normal_input),
    )


def main(args):
    if args.first_differences < 1:
        raise ValueError("--first-differences must be positive")
    torch.set_num_threads(8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    all_inputs = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = all_inputs[torch.isfinite(all_inputs)]
    positive = finite[finite > 0]
    assert finite.numel() == 65280 and positive.numel() == 32639
    domains = dict(silu=finite, sigmoid=finite, rsqrt=positive)
    references = dict(silu=F.silu(finite), sigmoid=torch.sigmoid(finite), rsqrt=torch.rsqrt(positive))
    result = dict(
        purpose="Exhaustive finite-BF16 input codes; FP32 device unary followed by explicit BF16 cast",
        execution="CPU-only" if args.oracle_only else "serial physical chip 0",
        torch_version=torch.__version__,
        torch_git_version=torch.version.git_version,
        input_codes=65536,
        finite_input_codes=finite.numel(),
        excluded_nan_or_inf_input_codes=256,
        rsqrt_positive_input_codes=positive.numel(),
        rsqrt_excluded_nonpositive_finite_codes=finite.numel() - positive.numel(),
        canonical_reference="Direct torch unary on BF16 input, including direct torch.rsqrt for both rsqrt variants",
        cpu_fp32_then_bf16={},
        device={},
    )
    cpu_ops = dict(
        silu=F.silu, sigmoid=torch.sigmoid, rsqrt=torch.rsqrt, sqrt_reciprocal=lambda x: torch.reciprocal(torch.sqrt(x))
    )
    for name, op in cpu_ops.items():
        domain = "rsqrt" if name == "sqrt_reciprocal" else name
        result["cpu_fp32_then_bf16"][name] = compare(
            domains[domain], references[domain], op(domains[domain].float()).bfloat16(), args.first_differences
        )

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if args.oracle_only:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "cpu_code_differences": {
                        name: row["exact_code_differences"] for name, row in result["cpu_fp32_then_bf16"].items()
                    },
                }
            )
        )
        return

    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        # Safe positive padding avoids introducing invalid-domain rsqrt work.
        def upload(x):
            padded = torch.ones(math.ceil(x.numel() / 1024) * 1024, dtype=torch.float32)
            padded[: x.numel()] = x.float()
            return ttnn.from_torch(
                padded.reshape(1, 1, -1, 32),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        full_input, positive_input = upload(finite), upload(positive)
        device_ops = dict(
            silu=ttnn.silu,
            sigmoid=ttnn.sigmoid,
            rsqrt=lambda x: ttnn.rsqrt(x, fast_and_approximate_mode=False),
            sqrt_reciprocal=lambda x: ttnn.reciprocal(ttnn.sqrt(x, fast_and_approximate_mode=False)),
        )
        for name, op in device_ops.items():
            domain = "rsqrt" if name == "sqrt_reciprocal" else name
            x = positive_input if domain == "rsqrt" else full_input
            fp32_result = op(x)
            assert fp32_result.dtype == ttnn.float32, f"{name} unexpectedly changed FP32 output dtype"
            rounded = ttnn.typecast(fp32_result, ttnn.bfloat16)
            actual = ttnn.to_torch(rounded).reshape(-1)[: domains[domain].numel()].clone()
            result["device"][name] = compare(domains[domain], references[domain], actual, args.first_differences)
            save()
            print(
                json.dumps(
                    {
                        "variant": name,
                        "code_differences": result["device"][name]["exact_code_differences"],
                        "numeric_differences": result["device"][name]["numeric_differences"],
                    }
                ),
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-only", action="store_true", help="Validate CPU domains/reference without importing TTNN"
    )
    parser.add_argument("--first-differences", type=int, default=20)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance/bf16_unary_precision.json"),
    )
    main(parser.parse_args())
