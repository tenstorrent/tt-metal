# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only idealized exp/order mechanism; NOT bit-exact device emulation.

Original normal BF16 Q/K/V -> FP64 scores, online state, PV and exact rescale.
Only score-exp family and running/global maximum schedule vary. No TTNN,
Q7/BFP quantization, integer exp grid, FPU alignment or BF16 internal spills.
Global maxima are an oracle diagnostic, not a free production implementation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHUNK = 512
DIM = 128
NATIVE_B = 32500.818359375
NATIVE_C = (32512.0 - NATIVE_B) / 256.0
CONSTANT_BIAS = 2.0 ** (-NATIVE_C)
RTOL = 1e-11
ATOL = 1e-12


def source_hashes():
    paths = [
        Path(__file__).resolve(),
        HERE / "exp_native.hpp",
        HERE / "streaming/compute_streaming.hpp",
        ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h",
        ROOT
        / "experiments/sdpa-l2/hybrid-mixed-v1/candidate/tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def tensor_hash(x):
    # Byte view includes signed zero and avoids NumPy BF16 dtype limitations.
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(actual, reference):
    a, b = actual.double(), reference.double()
    delta = a - b
    norm = b.norm()
    ac, bc = a - a.mean(), b - b.mean()
    pcc_denom = ac.norm() * bc.norm()
    return dict(
        l2_pct=float(100 * delta.norm() / norm) if norm else None,
        pcc=float((ac * bc).sum() / pcc_denom) if pcc_denom else None,
        absolute_error_max=float(delta.abs().max()),
        absolute_error_rms=float(delta.square().mean().sqrt()),
        reference_rms=float(b.square().mean().sqrt()),
        gain=float((a * b).sum() / b.square().sum()) if norm else None,
    )


def score_exp(x, family):
    if family == "exact":
        return torch.exp(x)
    if family == "constant_bias":
        return CONSTANT_BIAS * torch.exp(x)
    assert family == "continuous_native"
    # Continuous Schraudolph linear-float surrogate. No integer grid, clamp,
    # native MAD rounding, output packer rounding or consumed-P truncation.
    t = x / math.log(2.0) - NATIVE_C
    exponent = torch.floor(t)
    return torch.exp2(exponent) * (1.0 + t - exponent)


def online(scores, values, order, family, schedule, check_algebra):
    rows, length = scores.shape
    global_max = scores.amax(dim=1)
    maximum = global_max.clone() if schedule == "global_max" else torch.full_like(global_max, -math.inf)
    numerator = torch.zeros((rows, DIM), dtype=torch.float64)
    denominator = torch.zeros(rows, dtype=torch.float64)
    max_updates = torch.zeros(rows, dtype=torch.int64)
    anchored_weights = torch.empty_like(scores) if check_algebra else None
    for block in order:
        start, stop = block * CHUNK, (block + 1) * CHUNK
        local = scores[:, start:stop]
        next_max = global_max if schedule == "global_max" else torch.maximum(maximum, local.amax(dim=1))
        if schedule == "running_max":
            max_updates += next_max > maximum
        # Exact FP64 exp for rescaling, regardless of the score-exp family.
        alpha = torch.exp(maximum - next_max)
        probability = score_exp(local - next_max[:, None], family)
        numerator = alpha[:, None] * numerator + probability @ values[start:stop]
        denominator = alpha * denominator + probability.sum(dim=1)
        if check_algebra:
            # Independent dense evaluation of the telescoped insertion weights.
            # Place blocks back at original positions, independent of loop order.
            anchored_weights[:, start:stop] = probability * torch.exp(next_max - global_max)[:, None]
        maximum = next_max
    assert torch.equal(maximum, global_max), "Final maximum must be permutation-invariant"
    assert bool((denominator > 0).all())
    output = numerator / denominator[:, None]
    assert bool(torch.isfinite(output).all())
    diagnostics = dict(
        final_max_matches_global=True,
        running_max_updates_including_initial=max_updates.tolist(),
        positive_denominator=True,
    )
    if check_algebra:
        direct = (anchored_weights @ values) / anchored_weights.sum(dim=1, keepdim=True)
        assert torch.allclose(output, direct, rtol=RTOL, atol=ATOL), "Telescoped-weight algebra failed"
        diagnostics["telescoped_weight_identity"] = dict(passed=True, **metrics(output, direct))
    return output, diagnostics


def scalar_algebra_checks():
    # Three cases with a nontrivial mantissa-phase shift, a whole-octave shift,
    # and a small shift. These are mechanism examples, not measured chip values.
    examples = []
    for a, b in [(-0.3, -0.4), (-0.3, -math.log(2.0)), (-1.2, -0.05)]:
        ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
        for family in ["exact", "constant_bias", "continuous_native"]:
            lhs = score_exp(ta, family) * torch.exp(tb)
            rhs = score_exp(ta + tb, family)
            ratio = float(lhs / rhs)
            if family != "continuous_native" or b == -math.log(2.0):
                assert math.isclose(ratio, 1.0, rel_tol=1e-12, abs_tol=1e-12)
            examples.append(dict(a=a, b=b, family=family, product_to_direct_ratio=ratio))
    assert any(abs(x["product_to_direct_ratio"] - 1) > 0.01 for x in examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, choices=[4096, 32768], default=[4096, 32768])
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--sample-rows", type=int, default=64)
    args = parser.parse_args()
    assert Path(args.label).name == args.label and args.label not in (".", "..")
    assert 0 < args.sample_rows <= 128 and len(set(args.lengths)) == len(args.lengths)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    pins = source_hashes()
    started = time.perf_counter()
    with (HERE / (args.label + ".jsonl")).open("x") as stream:

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
                execution="CPU-only idealized mechanism; NOT device emulation or SageAttention execution",
                python=sys.version,
                torch_version=torch.__version__,
                platform=platform.platform(),
                cpu_threads=4,
                interop_threads=1,
                device="cpu",
                heads=1,
                head_dim=DIM,
                k_chunk=CHUNK,
                assumptions=[
                    "FP64 scores, state, PV, normalization and exp rescale; finite CPU arithmetic, not symbolic exactness",
                    "No Q7/KV5/BFP quantization, native exp integer grid, BF16 rowmax spill, FPU alignment or packer model",
                    "Continuous native surrogate uses the source B constant but ideal real-valued A/log2 factor",
                    "Global-max schedule requires prior full-score maximum; oracle control with unmodeled implementation cost",
                    "Reference is exact-exp FP64 attention on original BF16 inputs; BF16 cast reported separately",
                    "All output metrics cover the explicit sampled Q rows and every KV token; not all N query rows",
                    "Runtime is CPU experiment wall time, not attention kernel performance",
                ],
                native_surrogate=dict(B=NATIVE_B, c=NATIVE_C, constant_bias=CONSTANT_BIAS),
                algebra_tolerance=dict(rtol=RTOL, atol=ATOL),
                scalar_examples=scalar_algebra_checks(),
            )
        )
        cases = 0
        for length in args.lengths:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            originals = [
                torch.randn((length, DIM), generator=generator, dtype=torch.float32).bfloat16() for _ in range(3)
            ]
            input_hashes = [tensor_hash(x) for x in originals]
            query_rows = torch.linspace(0, length - 1, args.sample_rows).long().unique()
            q, k, v = originals[0][query_rows].double(), originals[1].double(), originals[2].double()
            scores = (q @ k.T) / math.sqrt(DIM)
            reference = torch.softmax(scores, dim=1) @ v
            reference_bf16 = reference.bfloat16()
            assert bool(torch.isfinite(reference).all())
            block_rms = k.reshape(length // CHUNK, CHUNK, DIM).square().mean(dim=(1, 2)).sqrt()
            orders = [
                ("identity", list(range(length // CHUNK))),
                ("reverse", list(reversed(range(length // CHUNK)))),
                ("k_rms_descending", torch.argsort(block_rms, descending=True, stable=True).tolist()),
            ]
            agreement = {}
            for name, order in orders:
                assert sorted(order) == list(range(length // CHUNK))
                indices = torch.cat([torch.arange(b * CHUNK, (b + 1) * CHUNK) for b in order])
                # Joint block permutation and its inverse recover original bits.
                inverse = torch.argsort(indices)
                assert torch.equal(originals[1][indices][inverse].view(torch.int16), originals[1].view(torch.int16))
                assert torch.equal(originals[2][indices][inverse].view(torch.int16), originals[2].view(torch.int16))
                reordered_ref = torch.softmax(scores[:, indices], dim=1) @ v[indices]
                assert torch.allclose(reordered_ref, reference, rtol=RTOL, atol=ATOL)
                agreement[name] = dict(passed=True, **metrics(reordered_ref, reference))
            global_direct = {}
            for family in ["exact", "constant_bias", "continuous_native"]:
                p = score_exp(scores - scores.amax(dim=1, keepdim=True), family)
                global_direct[family] = (p @ v) / p.sum(dim=1, keepdim=True)
                for schedule in ["running_max", "global_max"]:
                    identity = None
                    for name, order in orders:
                        output, algebra = online(scores, v, order, family, schedule, True)
                        replay, _ = online(scores, v, order, family, schedule, False)
                        assert tensor_hash(output) == tensor_hash(replay), "CPU deterministic replay failed"
                        if identity is None:
                            assert name == "identity"
                            identity = output.clone()
                        exact_control = family in ("exact", "constant_bias")
                        if exact_control:
                            assert torch.allclose(
                                output, reference, rtol=RTOL, atol=ATOL
                            ), "Exact/constant-bias control failed"
                        if schedule == "global_max":
                            assert torch.allclose(output, global_direct[family], rtol=RTOL, atol=ATOL)
                            assert torch.allclose(
                                output, identity, rtol=RTOL, atol=ATOL
                            ), "Fixed-global-max order invariance failed"
                        assert [tensor_hash(x) for x in originals] == input_hashes, "Original BF16 input mutated"
                        assert source_hashes() == pins, "Pinned source changed during CPU experiment"
                        rounded = output.bfloat16()
                        emit(
                            dict(
                                kind="result",
                                execution="CPU idealized, not device",
                                length=length,
                                heads=1,
                                head_dim=DIM,
                                seed=args.seed,
                                sampled_query_rows=query_rows.tolist(),
                                family=family,
                                maximum_schedule=schedule,
                                order=name,
                                block_indices=order,
                                original_input_sha256=input_hashes,
                                sampled_score_sha256=tensor_hash(scores),
                                reference_output_sha256=tensor_hash(reference),
                                k_block_rms=block_rms.tolist(),
                                accuracy_fp64=metrics(output, reference),
                                accuracy_bf16_output=metrics(rounded, reference),
                                bf16_output_rounding_floor=metrics(reference_bf16, reference),
                                interorder_fp64=metrics(output, identity),
                                interorder_bf16=metrics(rounded, identity.bfloat16()),
                                versus_same_family_global_max=metrics(output, global_direct[family]),
                                reference_permutation_agreement=agreement[name],
                                algebra_checks=algebra,
                                exact_or_constant_bias_matches_reference=True if exact_control else None,
                                global_schedule_order_invariant=True if schedule == "global_max" else None,
                                output_sha256=tensor_hash(output),
                                bf16_output_sha256=tensor_hash(rounded),
                                deterministic_cpu_replays=1,
                                replay_bitwise_equal=True,
                                inputs_unchanged=True,
                                sources_unchanged=True,
                                all_sampled_output_finite=True,
                            )
                        )
                        cases += 1
            del originals, q, k, v, scores, reference, global_direct
        assert cases == len(args.lengths) * 3 * 2 * 3
        emit(
            dict(
                kind="complete",
                cases=cases,
                sources_unchanged=source_hashes() == pins,
                cpu_wall_seconds=time.perf_counter() - started,
                device_jobs=0,
            )
        )


if __name__ == "__main__":
    main()
