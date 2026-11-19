# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded unary-chain diagnostic; --oracle-only never imports TTNN or opens a device."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder"
CASES = ("conv", "round_widen", "beta", "qk_rsqrt")


def number(x):
    x = float(x)
    return x if math.isfinite(x) else str(x)


def codes(x):
    x = x.contiguous().reshape(-1)
    if x.dtype == torch.bfloat16:
        return x.view(torch.int16).to(torch.int64).bitwise_and(0xFFFF)
    assert x.dtype == torch.float32
    return x.view(torch.int32).to(torch.int64).bitwise_and(0xFFFFFFFF)


def compare(inputs, reference, actual, limit):
    assert reference.dtype == actual.dtype and reference.shape == actual.shape == inputs.shape
    rc, ac, ic = codes(reference), codes(actual), codes(inputs)
    different = rc != ac
    finite = torch.isfinite(reference) & torch.isfinite(actual)
    numeric = finite & (reference != actual)
    error = torch.zeros(inputs.numel(), dtype=torch.float64)
    error[finite] = (reference[finite].double() - actual[finite].double()).abs()
    width = 4 if reference.dtype == torch.bfloat16 else 8

    def example(i):
        return dict(
            index=i,
            input=number(inputs[i]),
            input_fp32_code=f"0x{int(ic[i]):08x}",
            reference=number(reference[i]),
            actual=number(actual[i]),
            reference_code=f"0x{int(rc[i]):0{width}x}",
            actual_code=f"0x{int(ac[i]):0{width}x}",
        )

    def classes(x):
        return dict(
            finite=int(torch.isfinite(x).sum()),
            nan=int(torch.isnan(x).sum()),
            positive_inf=int(torch.isposinf(x).sum()),
            negative_inf=int(torch.isneginf(x).sum()),
        )

    masks = dict(
        zero=inputs == 0,
        below_bf16_normal=(inputs != 0) & (inputs.abs() < torch.finfo(torch.bfloat16).tiny),
        ordinary=(inputs.abs() >= 1e-20) & (inputs.abs() <= 1e20),
        outside_ordinary=(inputs != 0) & ((inputs.abs() < 1e-20) | (inputs.abs() > 1e20)),
    )
    return dict(
        inputs=inputs.numel(),
        output_dtype=str(reference.dtype),
        exact_code_differences=int(different.sum()),
        finite_pair_code_differences=int((different & finite).sum()),
        finite_numeric_differences=int(numeric.sum()),
        finite_numeric_comparisons=int(finite.sum()),
        excluded_nonfinite_pairs=int((~finite).sum()),
        nonfinite_code_differences=int((different & ~finite).sum()),
        nonfinite_class_differences=int(
            (
                (torch.isnan(reference) != torch.isnan(actual))
                | (torch.isposinf(reference) != torch.isposinf(actual))
                | (torch.isneginf(reference) != torch.isneginf(actual))
            ).sum()
        ),
        signed_zero_only_differences=int((different & finite & (reference == 0) & (actual == 0)).sum()),
        reference_classes=classes(reference),
        actual_classes=classes(actual),
        max_absolute_error=float(error.max()) if finite.any() else None,
        max_absolute_error_example=example(int(error.argmax())) if numeric.any() else None,
        diagnostic_input_classes={
            name: dict(inputs=int(mask.sum()), code_differences=int((different & mask).sum()))
            for name, mask in masks.items()
        },
        first_differences=[example(i) for i in different.nonzero().flatten()[:limit].tolist()],
        first_ordinary_differences=[
            example(i) for i in (different & masks["ordinary"]).nonzero().flatten()[:limit].tolist()
        ],
    )


def cpu_reference(name, x):
    if name == "conv":
        return F.silu(x.bfloat16().float()).bfloat16()
    if name == "round_widen":
        return x.bfloat16().float()
    if name == "beta":
        return torch.sigmoid(x).bfloat16().float()
    if name == "qk_rsqrt":
        return torch.rsqrt(x).bfloat16().float()
    raise ValueError(name)


def build_groups(args):
    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = raw[torch.isfinite(raw)].float()
    assert finite.numel() == 65280
    # Every adjacent nonnegative finite BF16 pair, both signs, exact midpoint and its FP32 neighbors.
    positive = raw[:0x7F80].float()
    mid = ((positive[:-1].double() + positive[1:].double()) * 0.5).float()
    around = torch.stack(
        (
            torch.nextafter(mid, torch.full_like(mid, -torch.inf)),
            mid,
            torch.nextafter(mid, torch.full_like(mid, torch.inf)),
        ),
        dim=1,
    ).flatten()
    midpoints = torch.cat((around, -around))
    groups = dict(
        all_finite_bf16=finite,
        bf16_midpoints_and_neighbors=midpoints,
        fp32_edges=torch.tensor(
            [
                0.0,
                -0.0,
                torch.finfo(torch.float32).tiny,
                -torch.finfo(torch.float32).tiny,
                torch.finfo(torch.float32).max,
                -torch.finfo(torch.float32).max,
                2.0**-149,
                -(2.0**-149),
            ],
            dtype=torch.float32,
        ),
    )
    provenance = []
    for position in args.positions:
        path = Path(f"{args.capture_prefix}.pos{position}.pt")
        if not path.exists():
            raise FileNotFoundError(
                f"Required actual-value capture missing: {path}; choose --capture-prefix/--positions"
            )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        for branch in ("tt_values", "hf_values"):
            values = saved.get(branch, {})
            for key in (
                "input_norm",
                "post_norm",
                "projection_packed",
                "projection_qkv",
                "projection_a",
                "projection_b",
            ):
                if key not in values:
                    continue
                tensor = values[key].float().reshape(-1)
                assert torch.isfinite(tensor).all(), f"Nonfinite actual input: {path}:{branch}/{key}"
                groups[f"pos{position}_{branch}_{key}"] = tensor
        # Real token-derived FP32 RMS intermediates, explicitly unweighted; no checkpoint or host oracle in runtime.
        token = saved["token"].float()
        denominator = token.square().mean(dim=-1, keepdim=True) + 1e-6
        groups[f"pos{position}_token_rms_denominator"] = denominator.flatten()
        groups[f"pos{position}_token_unweighted_rms_norm_fp32"] = (token * denominator.rsqrt()).flatten()
        # Q/K pre-L2 products captured on device supply realistic positive reduction inputs.
        for key in ("delta_q", "delta_k"):
            if key in saved.get("tt_values", {}):
                qk = saved["tt_values"][key].float()
                groups[f"pos{position}_{key}_squared_sum_plus_eps"] = (qk.square().sum(-1) + 1e-6).flatten()
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return groups, provenance


def run_device(ttnn, mesh, x, case, chunk_size):
    # Constructor is the bound (UnaryOpType, float, float) overload in activation.cpp:37–41.
    def cast(a, b):
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(a.value), float(b.value))

    round_op = cast(ttnn.float32, ttnn.bfloat16)
    widen_op = cast(ttnn.bfloat16, ttnn.float32)
    chains = dict(
        conv=[round_op, widen_op, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), round_op],
        round_widen=[round_op, widen_op],
        beta=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID), round_op, widen_op],
        qk_rsqrt=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0), round_op, widen_op],
    )
    separate_parts, chain_parts = [], []
    for start in range(0, x.numel(), chunk_size):
        part = x[start : start + chunk_size]
        padded = torch.ones(math.ceil(part.numel() / 1024) * 1024, dtype=torch.float32)
        padded[: part.numel()] = part
        live = []

        def track(t):
            live.append(t)
            return t

        try:
            device_input = track(
                ttnn.from_torch(
                    padded.reshape(1, 1, -1, 32),
                    device=mesh,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )

            def round_tensor(t):
                return track(ttnn.typecast(t, ttnn.bfloat16))

            def widen_tensor(t):
                return track(ttnn.typecast(t, ttnn.float32))

            if case == "conv":
                separate = round_tensor(track(ttnn.silu(widen_tensor(round_tensor(device_input)))))
            elif case == "round_widen":
                separate = widen_tensor(round_tensor(device_input))
            elif case == "beta":
                separate = widen_tensor(round_tensor(track(ttnn.sigmoid(device_input))))
            else:
                separate = widen_tensor(round_tensor(track(ttnn.rsqrt(device_input, fast_and_approximate_mode=False))))
            chain = track(ttnn.unary_chain(device_input, chains[case]))
            expected_dtype = ttnn.bfloat16 if case == "conv" else ttnn.float32
            assert separate.dtype == chain.dtype == expected_dtype
            separate_parts.append(ttnn.to_torch(separate).reshape(-1)[: part.numel()].clone())
            chain_parts.append(ttnn.to_torch(chain).reshape(-1)[: part.numel()].clone())
        finally:
            for tensor in reversed(live):
                ttnn.deallocate(tensor)
    return torch.cat(separate_parts), torch.cat(chain_parts)


def main(args):
    if args.chunk_size < 1024 or args.chunk_size > 262144 or args.first_differences < 1:
        raise ValueError("Require 1024 <= --chunk-size <= 262144 and positive --first-differences")
    torch.set_num_threads(8)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Preserving existing report {output}; choose another --output or explicit --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    groups, captures = build_groups(args)
    all_inputs = torch.cat(list(groups.values()))
    assert torch.isfinite(all_inputs).all()
    result = dict(
        classification="diagnostic only; no acceptance waiver",
        execution="CPU-only" if args.oracle_only else "serial physical chip 0",
        status="in_progress",
        torch_version=torch.__version__,
        torch_git_version=torch.version.git_version,
        captures=captures,
        inputs=all_inputs.numel(),
        groups={k: v.numel() for k, v in groups.items()},
        chunk_size=args.chunk_size,
        cases={},
        reference="Torch FP32 arithmetic with the explicit BF16 boundaries in each case; not direct BF16 unary",
        input_notes="All 65280 finite BF16 codes; all signed adjacent-BF16 midpoint neighborhoods; real captures and labeled token-derived RMS intermediates. Diagnostic ordinary range is 1e-20..1e20; totals are never filtered by it.",
        formulas=dict(
            conv="silu(x.bfloat16().float()).bfloat16()",
            round_widen="x.bfloat16().float()",
            beta="sigmoid(x).bfloat16().float()",
            qk_rsqrt="rsqrt(x).bfloat16().float(); x > 0",
        ),
    )

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = None
    if not args.oracle_only:
        import ttnn

        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        for case in CASES:
            domain_groups = {
                name: values[values > 0] if case == "qk_rsqrt" else values for name, values in groups.items()
            }
            x = torch.cat(list(domain_groups.values()))
            reference = cpu_reference(case, x)
            row = dict(
                inputs=x.numel(),
                excluded_invalid_domain_inputs=all_inputs.numel() - x.numel(),
                cpu_reference_classes=compare(x, reference, reference, args.first_differences),
                groups={},
            )
            if args.oracle_only:
                row["device_execution"] = "not run"
            else:
                separate, chain = run_device(ttnn, mesh, x, case, args.chunk_size)
                pairs = dict(
                    chain_vs_separate=(separate, chain),
                    separate_vs_cpu=(reference, separate),
                    chain_vs_cpu=(reference, chain),
                )
                row.update({key: compare(x, a, b, args.first_differences) for key, (a, b) in pairs.items()})
                offset = 0
                for name, values in domain_groups.items():
                    size = values.numel()
                    if size:
                        row["groups"][name] = {
                            key: compare(
                                values, a[offset : offset + size], b[offset : offset + size], args.first_differences
                            )
                            for key, (a, b) in pairs.items()
                        }
                    offset += size
            result["cases"][case] = row
            save()
            print(
                json.dumps(
                    dict(
                        case=case,
                        inputs=x.numel(),
                        chain_vs_separate_code_differences=row.get("chain_vs_separate", {}).get(
                            "exact_code_differences"
                        ),
                    )
                ),
                flush=True,
            )
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    result["status"] = "complete"
    save()
    print(json.dumps(dict(output=str(output), classification=result["classification"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--capture-prefix", default=str(DOC / "endurance_precise_probe.json"))
    parser.add_argument("--positions", nargs="+", type=int, default=[128, 7703])
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--first-differences", type=int, default=12)
    parser.add_argument("--output", default=str(DOC / "endurance/unary_chain_precision_probe.json"))
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
