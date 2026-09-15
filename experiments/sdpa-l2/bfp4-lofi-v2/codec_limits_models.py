# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU representation-only codec comparisons, NOT Sage3 or hardware emulation.

Original BF16 inputs; exact FP64 QK/softmax/PV/state, unquantized P.
All main rows use native TT group16 along D for BOTH K and V, to isolate
codebook/scaling. Optional NVIDIA-like V grouping along N is separately labeled.
Continuous absmax scales are optimistic representability comparisons, not
rigorous lower bounds and not globally/per-group MSE-optimal scales.
"""

import argparse
import ast
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROBE = HERE.parent / "bfp4-lofi-v1/probe.py"
REPRO = ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"
ADAPT = HERE / "adaptive_bfp4_round.py"
B4 = HERE / "bfp4_round.py"


def functions(path, names, **extra):
    """Reuse exact pinned CPU function ASTs without importing TTNN/device code."""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    namespace = dict(torch=torch, math=math, **extra)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


QUALIFIED = functions(B4, ("validate_input", "host_rne_bfp4"))
TT = functions(PROBE, ("quantize",))
ADAPTIVE = functions(ADAPT, ("validate_input", "ftz", "score_tree", "oracle"), QUALIFIED=QUALIFIED)
REF = functions(REPRO, ("make_inputs", "reference", "metrics"))
CODECS = (
    "tt_native",
    "tt_rne",
    "tt_adaptive_pm",
    "uniform7_continuous",
    "e2m1_continuous",
    "e2m1_power2_g16",
    "nvfp4_e4m3",
)


def e4m3_positive_table():
    # E4M3FN positive finite encodings0..126;127 is NaN, maximum448.
    return torch.tensor(
        [
            m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
            for e in range(16)
            for m in range(8)
            if e < 15 or m < 7
        ],
        dtype=torch.float64,
    )


E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
E4M3 = e4m3_positive_table()


def nearest_even_positive(x, table):
    """Saturating nearest level, ties to even encoded significand/code bit."""
    assert bool((x >= 0).all()) and bool(torch.isfinite(x).all())
    mids = (table[:-1] + table[1:]) * 0.5
    index = torch.bucketize(x.contiguous(), mids)
    at_mid = (index < mids.numel()) & (x == mids[index.clamp_max(mids.numel() - 1)])
    index = index + (at_mid & ((index & 1) != 0)).long()
    return table[index]


def quantize(x, codec, axis="D"):
    assert x.dtype == torch.bfloat16 and x.ndim == 2
    oriented = x if axis == "D" else x.T.contiguous()
    groups = oriented.double().reshape(-1, 16)
    maximum = groups.abs().amax(-1, keepdim=True)
    stats = dict(codec=codec, group_size=16, group_axis=axis)
    if codec == "tt_native":
        output = TT.quantize(oriented, 3, "device").double()
        stats["packing"] = "qualified default device E8M6 RNA -> shared B8 RNA -> trunc3"
    elif codec == "tt_rne":
        output = QUALIFIED.host_rne_bfp4(oriented).double()
        stats["packing"] = "qualified shared3-magnitude-bit RNE plus saturation"
    elif codec == "tt_adaptive_pm":
        output, search = ADAPTIVE.oracle(oriented, "pm")
        output = output.double()
        stats["search"] = search
    else:
        if codec == "uniform7_continuous":
            scale = maximum / 7
            scale = torch.where(scale == 0, 1.0, scale)
            codes = (groups.abs() / scale).round().clamp_max(7)
        else:
            if codec == "e2m1_continuous":
                scale = maximum / 6
                stats["scaling"] = "FP64 absmax/6; unencoded continuous scale, NOT MSE-optimal"
            elif codec == "e2m1_power2_g16":
                # Controlled power2 codebook comparison, NOT claimed as a
                # complete MXFP4 recipe (which also uses group32).
                exponent = torch.frexp(maximum)[1] - 1
                scale = torch.ldexp(torch.ones_like(maximum), exponent - 2)
                stats["scaling"] = "2^(floor(log2(group_absmax))-2), E2M1 saturation"
            elif codec == "nvfp4_e4m3":
                # Representative absmax hierarchical NVFP4 recipe.
                # Explicit FP32 global and desired-scale computation, then
                # RNE E4M3FN block-scale encoding. Reconstruct encoded values
                # with FP64 arithmetic to isolate representation error.
                global_scale = (maximum.max().float() / (6 * 448)).float()
                if float(global_scale) == 0:
                    global_scale = torch.tensor(1.0, dtype=torch.float32)
                desired = (maximum.float() / (6 * global_scale)).double()
                local = nearest_even_positive(desired, E4M3)
                scale = local * global_scale.double()
                stats.update(
                    global_scale=float(global_scale),
                    block_scale_zero_groups=int((local == 0).sum()),
                    scaling="E4M3FN RNE(absmax/(6*global)); global=FP32(tensor_absmax/(6*448))",
                )
            else:
                raise ValueError(codec)
            scale = torch.where(maximum == 0, 1.0, scale)
            # If a very small block scale rounds to zero, all its values
            # reconstruct to zero; avoid inventing a nonzero rescue scale.
            divisor = torch.where(scale == 0, 1.0, scale)
            codes = nearest_even_positive(groups.abs() / divisor, E2M1)
        output = (codes * scale * groups.sign()).reshape_as(oriented)
        stats["saturated_value_count"] = int(
            (groups.abs() > (7 if codec == "uniform7_continuous" else 6) * scale).sum()
        )
    if axis == "N":
        output = output.T.contiguous()
    assert output.shape == x.shape and bool(torch.isfinite(output).all())
    stats.update(
        reconstruction_l2_pct=float(100 * (output - x.double()).norm() / x.double().norm()),
        relative_mean_error=float((output - x.double()).mean() / x.double().square().mean().sqrt()),
        zero_fraction=float((output == 0).double().mean()),
    )
    return output, stats


def q7(q):
    raw = q.float().contiguous().view(torch.int32)
    shift = 17  # seven significant bits, including hidden bit
    return ((raw + (1 << (shift - 1)) - 1 + ((raw >> shift) & 1)) & ~((1 << shift) - 1)).view(torch.float32)


def self_test():
    assert torch.equal(nearest_even_positive(E2M1, E2M1), E2M1)
    assert torch.equal(nearest_even_positive(E4M3, E4M3), E4M3)
    for table in (E2M1, E4M3):
        mids = (table[:-1] + table[1:]) / 2
        expected_index = torch.arange(mids.numel())
        expected_index += expected_index & 1
        assert torch.equal(nearest_even_positive(mids, table), table[expected_index])
    assert float(E4M3[-1]) == 448 and float(E4M3[1]) == 2.0**-9
    generator = torch.Generator().manual_seed(10)
    x = torch.randn((32, 128), generator=generator).bfloat16()
    for name in CODECS:
        for axis in ("D", "N"):
            value, _ = quantize(x, name, axis)
            assert bool(torch.isfinite(value).all())
    # Continuous uniform codebook must represent equally spaced integer data.
    exact = torch.arange(-7, 9).clamp_max(7).bfloat16().repeat(32, 8)
    value, _ = quantize(exact, "uniform7_continuous")
    assert torch.equal(value, exact.double())
    return dict(codebook_roundtrip=True, all_midpoint_ties_even=True, finite_both_axes=True, uniform_integer_exact=True)


def source_files():
    return [Path(__file__).resolve(), PROBE, REPRO, ADAPT, B4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096])
    parser.add_argument("--q-rows", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1240])
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=("normal", "outliers", "channel_v", "common_v"),
        default=["normal", "outliers"],
    )
    parser.add_argument("--codecs", nargs="+", choices=CODECS, default=list(CODECS))
    parser.add_argument("--q-modes", nargs="+", choices=("bf16", "q7"), default=["bf16", "q7"])
    parser.add_argument(
        "--nv-v-axis-n", action="store_true", help="Add NVFP4 V grouping along sequence, separately labeled"
    )
    parser.add_argument(
        "--v-axes",
        nargs="+",
        choices=("D", "N"),
        default=["D"],
        help="Grouping axis for V, including TT codecs; K always groups along D",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    assert args.q_rows > 0 and args.q_rows % 32 == 0 and args.threads > 0
    assert all(n > 0 and n % 32 == 0 for n in args.lengths)
    path = HERE / (args.label + ".jsonl")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
    tested = self_test()
    started = time.monotonic()
    with path.open("x") as stream:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                args=vars(args),
                source_sha256=pins,
                self_tests=tested,
                contract=__doc__,
                hostname=platform.node(),
                torch_version=torch.__version__,
                execution="CPU-only, no TTNN import or device calls",
                output_rounding="reported both FP64 and BF16",
            )
        )
        for n in args.lengths:
            for seed in args.seeds:
                for distribution in args.distributions:
                    inputs = REF.make_inputs(
                        1, args.q_rows, n, 128, seed, "normal" if distribution == "channel_v" else distribution
                    )
                    if distribution == "channel_v":
                        # Eight persistent feature outliers; each D-group16
                        # has one32x channel, while N-group16 stays per-feature.
                        inputs[2][..., ::16] *= 32
                    q, k, v = [x.squeeze(0).squeeze(0).contiguous() for x in inputs]
                    reference = REF.reference(q, k, v)
                    context = dict(length=n, q_rows=args.q_rows, seed=seed, distribution=distribution)
                    emit(
                        dict(
                            kind="inputs",
                            **context,
                            sha256={
                                name: hashlib.sha256(x.view(torch.int16).numpy().tobytes()).hexdigest()
                                for name, x in zip(("Q", "K", "V"), (q, k, v))
                            },
                        )
                    )
                    variants = [(name, axis) for name in args.codecs for axis in args.v_axes]
                    if args.nv_v_axis_n and ("nvfp4_e4m3", "N") not in variants:
                        variants.append(("nvfp4_e4m3", "N"))
                    for q_mode in args.q_modes:
                        qe = q.double() if q_mode == "bf16" else q7(q).double()
                        base_p = torch.softmax(qe @ k.double().T / math.sqrt(128), -1)
                        emit(
                            dict(
                                kind="attention",
                                **context,
                                q_mode=q_mode,
                                codec="unquantized",
                                v_axis="D",
                                scope="control",
                                fp64_output=REF.metrics(base_p @ v.double(), reference),
                                bf16_output=REF.metrics((base_p @ v.double()).bfloat16(), reference),
                            )
                        )
                        for codec, v_axis in variants:
                            ke, ks = quantize(k, codec, "D")
                            ve, vs = quantize(v, codec, v_axis)
                            pe = torch.softmax(qe @ ke.T / math.sqrt(128), -1)
                            for scope, actual in (
                                ("K_only", pe @ v.double()),
                                ("V_only", base_p @ ve),
                                ("KV", pe @ ve),
                            ):
                                extra = {}
                                if distribution == "channel_v":
                                    quiet = torch.arange(128) % 16 != 0
                                    extra["quiet_channel_fp64_output"] = REF.metrics(
                                        actual[..., quiet], reference[..., quiet]
                                    )
                                    extra["quiet_V_reconstruction_l2_pct"] = float(
                                        100
                                        * (ve[..., quiet] - v.double()[..., quiet]).norm()
                                        / v.double()[..., quiet].norm()
                                    )
                                if distribution == "common_v":
                                    offset = v.double().mean(0, keepdim=True)
                                    extra["centered_fp64_output"] = REF.metrics(actual - offset, reference - offset)
                                    extra["centered_bf16_output"] = REF.metrics(
                                        actual.bfloat16().double() - offset, reference - offset
                                    )
                                emit(
                                    dict(
                                        kind="attention",
                                        **context,
                                        q_mode=q_mode,
                                        codec=codec,
                                        v_axis=v_axis,
                                        scope=scope,
                                        K=ks,
                                        V=vs,
                                        fp64_output=REF.metrics(actual, reference),
                                        bf16_output=REF.metrics(actual.bfloat16(), reference),
                                        **extra,
                                    )
                                )
        assert pins == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
        emit(dict(kind="complete", sources_unchanged=True, elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
