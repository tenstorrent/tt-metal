# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only ln(2)-lattice maximum diagnostic, NOT device/Sage emulation.

Imports the frozen continuous-exp model; no changes to that measured producer.
Lattice maxima are calculated in FP64. A separate BF16-spill schedule shows
why simply storing snapped real maxima in BF16 need not preserve the identity.
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

import exp_order_cpu_mechanism as M

HERE = Path(__file__).resolve().parent
LN2 = math.log(2.0)


def pins():
    result = M.source_hashes()
    for path in [Path(__file__).resolve(), HERE / "exp-order-cpu-mechanism-v1.jsonl"]:
        result[str(path.relative_to(M.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def snapped(x, bf16_spill):
    value = torch.ceil(x / LN2) * LN2
    return value.bfloat16().double() if bf16_spill else value


def lattice_online(scores, values, order, family, bf16_spill, check):
    final = snapped(scores.amax(dim=1), bf16_spill)
    current = torch.full_like(final, -math.inf)
    numerator = torch.zeros((scores.shape[0], M.DIM), dtype=torch.float64)
    denominator = torch.zeros_like(final)
    updates = torch.zeros_like(final, dtype=torch.int64)
    anchored = torch.empty_like(scores) if check else None
    phase_error = 0.0
    positive_exp_argument_max = 0.0
    for block in order:
        start, stop = block * M.CHUNK, (block + 1) * M.CHUNK
        local = scores[:, start:stop]
        candidate = snapped(local.amax(dim=1), bf16_spill)
        following = torch.maximum(current, candidate)
        updates += following > current
        alpha = torch.exp(current - following)
        argument = local - following[:, None]
        probability = M.score_exp(argument, family)
        numerator = alpha[:, None] * numerator + probability @ values[start:stop]
        denominator = alpha * denominator + probability.sum(dim=1)
        if check:
            anchored[:, start:stop] = probability * torch.exp(following - final)[:, None]
        coordinate = following / LN2
        phase_error = max(phase_error, float((coordinate - coordinate.round()).abs().max()))
        positive_exp_argument_max = max(positive_exp_argument_max, float(argument.max()))
        current = following
    assert torch.equal(current, final)
    assert bool((denominator > 0).all())
    output = numerator / denominator[:, None]
    assert bool(torch.isfinite(output).all())
    diagnostic = dict(
        final_max_matches_snapped_global=True,
        max_updates_including_initial=updates.tolist(),
        maximum_off_lattice_log2_distance=phase_error,
        positive_exp_argument_max=positive_exp_argument_max,
        positive_denominator=True,
    )
    if check:
        direct = (anchored @ values) / anchored.sum(dim=1, keepdim=True)
        assert torch.allclose(output, direct, rtol=M.RTOL, atol=M.ATOL)
        fixed_p = M.score_exp(scores - final[:, None], family)
        fixed_output = (fixed_p @ values) / fixed_p.sum(dim=1, keepdim=True)
        if not bf16_spill:
            assert torch.allclose(output, fixed_output, rtol=M.RTOL, atol=M.ATOL)
        diagnostic.update(
            telescoped_weight_identity=dict(passed=True, **M.metrics(output, direct)),
            versus_fixed_snapped_global=M.metrics(output, fixed_output),
            lattice_translation_identity_required=not bf16_spill,
            lattice_translation_identity_passed=True if not bf16_spill else None,
        )
    return output, diagnostic


def scalar_checks():
    x = torch.linspace(-12.0, 0.0, 257, dtype=torch.float64)
    checks = []
    for shift in [1, 2, 7]:
        lhs = M.score_exp(x - shift * LN2, "continuous_native")
        rhs = math.ldexp(1.0, -shift) * M.score_exp(x, "continuous_native")
        error = float(((lhs - rhs) / rhs).abs().max())
        assert error < 1e-12
        checks.append(dict(octave_shift=shift, maximum_relative_translation_error=error))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, choices=[4096, 32768], default=[4096, 32768])
    args = parser.parse_args()
    assert Path(args.label).name == args.label and args.label not in (".", "..")
    assert len(set(args.lengths)) == len(args.lengths)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    prior = [json.loads(line) for line in (HERE / "exp-order-cpu-mechanism-v1.jsonl").read_text().splitlines()]
    assert prior[-1]["kind"] == "complete" and prior[-1]["cases"] == 36
    controls = {
        (row["length"], row["family"], row["maximum_schedule"], row["order"]): row
        for row in prior
        if row["kind"] == "result"
    }
    source_pins = pins()
    start_time = time.perf_counter()
    count = 0
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
                source_sha256=source_pins,
                execution="CPU-only idealized continuous-exp mechanism, NOT device/Sage execution",
                python=sys.version,
                torch_version=torch.__version__,
                platform=platform.platform(),
                cpu_threads=4,
                interop_threads=1,
                seed=1240,
                heads=1,
                head_dim=M.DIM,
                sample_rows=64,
                k_chunk=M.CHUNK,
                scalar_translation_checks=scalar_checks(),
                assumptions=[
                    "Same BF16 normal generator and sampled Q rows as pinned v1; exact control output hashes must match",
                    "FP64 scores, PV, recurrent state, normalization and exact exp rescale",
                    "Native F is the continuous surrogate: no native integer grid, Q7/BFP, FPU alignment or score spill",
                    "Lattice running max uses ceil(local max / ln2) * ln2 before online max; no global prepass needed",
                    "BF16-spill variant rounds this snapped real max only; not a complete silicon emulation",
                    "Fixed absolute lattice can sacrifice common-score-shift invariance; separate identity-order shift controls included",
                    "Fixed global max is an oracle control with unmodeled prepass cost",
                    "All metrics use 64 sampled Q rows and every KV token; final BF16 cast is separate",
                    "No inference about device accuracy/performance or model-quality follows from CPU wall time",
                ],
            )
        )
        for length in args.lengths:
            generator = torch.Generator(device="cpu").manual_seed(1240)
            originals = [
                torch.randn((length, M.DIM), generator=generator, dtype=torch.float32).bfloat16() for _ in range(3)
            ]
            hashes = [M.tensor_hash(x) for x in originals]
            rows = torch.linspace(0, length - 1, 64).long().unique()
            q, k, v = originals[0][rows].double(), originals[1].double(), originals[2].double()
            scores = (q @ k.T) / math.sqrt(M.DIM)
            reference = torch.softmax(scores, dim=1) @ v
            rms = k.reshape(length // M.CHUNK, M.CHUNK, M.DIM).square().mean(dim=(1, 2)).sqrt()
            orders = [
                ("identity", list(range(length // M.CHUNK))),
                ("reverse", list(reversed(range(length // M.CHUNK)))),
                ("k_rms_descending", torch.argsort(rms, descending=True, stable=True).tolist()),
            ]
            for family in ["exact", "continuous_native"]:
                for schedule in ["running_max", "global_max", "lattice_running", "lattice_running_bf16_spill"]:
                    identity = None
                    for order_name, order in orders:
                        assert sorted(order) == list(range(length // M.CHUNK))
                        indices = torch.cat([torch.arange(b * M.CHUNK, (b + 1) * M.CHUNK) for b in order])
                        inverse = torch.argsort(indices)
                        for original in originals[1:]:
                            assert torch.equal(original[indices][inverse].view(torch.int16), original.view(torch.int16))
                        permuted_reference = torch.softmax(scores[:, indices], dim=1) @ v[indices]
                        assert torch.allclose(permuted_reference, reference, rtol=M.RTOL, atol=M.ATOL)
                        if schedule in ("running_max", "global_max"):
                            output, diagnostic = M.online(scores, v, order, family, schedule, True)
                            replay, _ = M.online(scores, v, order, family, schedule, False)
                        else:
                            spill = schedule.endswith("bf16_spill")
                            output, diagnostic = lattice_online(scores, v, order, family, spill, True)
                            replay, _ = lattice_online(scores, v, order, family, spill, False)
                        old = controls[
                            (length, family, "running_max" if schedule.startswith("lattice") else schedule, order_name)
                        ]
                        assert hashes == old["original_input_sha256"]
                        assert M.tensor_hash(scores) == old["sampled_score_sha256"]
                        assert M.tensor_hash(reference) == old["reference_output_sha256"]
                        assert rows.tolist() == old["sampled_query_rows"] and order == old["block_indices"]
                        prior_output_checked = not schedule.startswith("lattice")
                        if prior_output_checked:
                            assert M.tensor_hash(output) == old["output_sha256"]
                        assert M.tensor_hash(output) == M.tensor_hash(replay)
                        if identity is None:
                            identity = output.clone()
                        invariant = schedule in ("global_max", "lattice_running") or family == "exact"
                        if invariant:
                            assert torch.allclose(output, identity, rtol=M.RTOL, atol=M.ATOL)
                        if family == "exact":
                            assert torch.allclose(output, reference, rtol=M.RTOL, atol=M.ATOL)
                        assert [M.tensor_hash(x) for x in originals] == hashes
                        assert pins() == source_pins
                        emit(
                            dict(
                                kind="result",
                                execution="CPU idealized, not device",
                                length=length,
                                heads=1,
                                seed=1240,
                                head_dim=M.DIM,
                                sampled_query_rows=rows.tolist(),
                                family=family,
                                maximum_schedule=schedule,
                                order=order_name,
                                block_indices=order,
                                original_input_sha256=hashes,
                                sampled_score_sha256=M.tensor_hash(scores),
                                reference_output_sha256=M.tensor_hash(reference),
                                output_sha256=M.tensor_hash(output),
                                bf16_output_sha256=M.tensor_hash(output.bfloat16()),
                                accuracy_fp64=M.metrics(output, reference),
                                accuracy_bf16_output=M.metrics(output.bfloat16(), reference),
                                bf16_output_rounding_floor=M.metrics(reference.bfloat16(), reference),
                                interorder_fp64=M.metrics(output, identity),
                                interorder_bf16=M.metrics(output.bfloat16(), identity.bfloat16()),
                                algebra_checks=diagnostic,
                                reference_permutation_check=True,
                                prior_v1_input_score_reference_rows_order_match=True,
                                prior_v1_control_output_bitwise_match=True if prior_output_checked else None,
                                required_order_invariance_passed=True if invariant else None,
                                exact_exp_matches_reference=True if family == "exact" else None,
                                replay_bitwise_equal=True,
                                deterministic_cpu_replays=1,
                                inputs_unchanged=True,
                                sources_unchanged=True,
                                all_sampled_output_finite=True,
                            )
                        )
                        count += 1
            # Orthogonal invariance tradeoff: add one common score offset to
            # every token, retaining ORIGINAL unshifted exact attention as ref.
            # These are identity-order controls, not a full shift x order grid.
            for family in ["exact", "continuous_native"]:
                for schedule in ["running_max", "global_max", "lattice_running"]:
                    unshifted = None
                    for offset_octaves in [0.0, 0.25, 0.5]:
                        shifted = scores + offset_octaves * LN2
                        shifted_reference = torch.softmax(shifted, dim=1) @ v
                        assert torch.allclose(shifted_reference, reference, rtol=M.RTOL, atol=M.ATOL)
                        order = orders[0][1]
                        if schedule == "lattice_running":
                            output, diagnostic = lattice_online(shifted, v, order, family, False, True)
                            replay, _ = lattice_online(shifted, v, order, family, False, False)
                        else:
                            output, diagnostic = M.online(shifted, v, order, family, schedule, True)
                            replay, _ = M.online(shifted, v, order, family, schedule, False)
                        if unshifted is None:
                            unshifted = output.clone()
                        require_shift_invariance = family == "exact" or schedule != "lattice_running"
                        if require_shift_invariance:
                            assert torch.allclose(output, unshifted, rtol=M.RTOL, atol=M.ATOL)
                        if family == "exact":
                            assert torch.allclose(output, reference, rtol=M.RTOL, atol=M.ATOL)
                        assert M.tensor_hash(output) == M.tensor_hash(replay)
                        assert [M.tensor_hash(x) for x in originals] == hashes and pins() == source_pins
                        emit(
                            dict(
                                kind="shift_result",
                                execution="CPU idealized, not device",
                                length=length,
                                heads=1,
                                seed=1240,
                                head_dim=M.DIM,
                                sampled_query_rows=rows.tolist(),
                                family=family,
                                maximum_schedule=schedule,
                                order="identity",
                                common_score_offset_octaves=offset_octaves,
                                common_score_offset=offset_octaves * LN2,
                                original_input_sha256=hashes,
                                original_sampled_score_sha256=M.tensor_hash(scores),
                                shifted_sampled_score_sha256=M.tensor_hash(shifted),
                                original_reference_output_sha256=M.tensor_hash(reference),
                                output_sha256=M.tensor_hash(output),
                                bf16_output_sha256=M.tensor_hash(output.bfloat16()),
                                accuracy_fp64=M.metrics(output, reference),
                                accuracy_bf16_output=M.metrics(output.bfloat16(), reference),
                                versus_unshifted_fp64=M.metrics(output, unshifted),
                                versus_unshifted_bf16=M.metrics(output.bfloat16(), unshifted.bfloat16()),
                                exact_attention_shift_check=dict(
                                    passed=True, **M.metrics(shifted_reference, reference)
                                ),
                                algebra_checks=diagnostic,
                                required_shift_invariance_passed=True if require_shift_invariance else None,
                                replay_bitwise_equal=True,
                                inputs_unchanged=True,
                                sources_unchanged=True,
                                all_sampled_output_finite=True,
                            )
                        )
                        count += 1
        assert count == len(args.lengths) * (2 * 4 * 3 + 2 * 3 * 3)
        emit(
            dict(
                kind="complete",
                cases=count,
                source_pins_unchanged=pins() == source_pins,
                cpu_wall_seconds=time.perf_counter() - start_time,
                device_jobs=0,
            )
        )


if __name__ == "__main__":
    main()
