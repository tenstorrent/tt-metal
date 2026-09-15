# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only attention representation/noise-budget study; not a hardware emulator.

Original normal BF16 Q/K/V, exact FP64 QK/softmax/PV, unquantized P.
Mean correction subtracts mean(Vq-V) AFTER attention, without re-quantizing V.
Analytical estimates are heuristics, not lower bounds or model-quality claims.
Run only when the parent has reserved a CPU-safe gap in device measurements.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch

import codec_limits_models as C

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODECS = ("tt_rne", "nvfp4_e4m3")
SCOPES = ("control", "K_only", "V_only", "KV", "V_only_mean_corrected", "KV_mean_corrected")


def digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sampled_rows(length, count):
    assert 0 < count <= length
    return [0] if count == 1 else [i * (length - 1) // (count - 1) for i in range(count)]


def make_inputs(length, count, seed):
    # Generate a full square BF16 input first, then retain only explicit sampled Q.
    # K/V and Q use the existing independent seeded generators; no Q7 preprocessing.
    q, k, v = C.REF.make_inputs(1, length, length, 128, seed, "normal")
    rows = sampled_rows(length, count)
    return q[0, 0, rows].contiguous(), k[0, 0], v[0, 0], rows


def representation(original, represented, stats):
    x, delta = original.double(), represented - original.double()
    rms = x.square().mean().sqrt()
    centered_x, centered_delta = x - x.mean(), delta - delta.mean()
    denominator = centered_x.norm() * centered_delta.norm()
    mean_error = delta.mean(dim=0)
    return dict(
        **stats,
        error_rms=float(delta.square().mean().sqrt()),
        value_rms=float(rms),
        sigma_relative=float(delta.norm() / x.norm()),
        quantization_gain=float((represented * x).sum() / x.square().sum()),
        error_input_correlation=float((centered_x * centered_delta).sum() / denominator) if denominator else None,
        token_mean_error_rms=float(mean_error.square().mean().sqrt()),
        token_mean_error_max_abs=float(mean_error.abs().max()),
    )


def compare_error(predicted, measured):
    denominator = float(measured.norm())
    cos_den = float(predicted.norm() * measured.norm())
    return dict(
        predicted_error_rms=float(predicted.square().mean().sqrt()),
        measured_error_rms=float(measured.square().mean().sqrt()),
        discrepancy_rms=float((predicted - measured).square().mean().sqrt()),
        discrepancy_l2_pct=100 * float((predicted - measured).norm()) / denominator if denominator else None,
        cosine=float((predicted * measured).sum()) / cos_den if cos_den else None,
        relative_comparison_undefined=not bool(denominator),
    )


def calculate(q, k, v, kq, vq, query_batch=8):
    """Bound query-by-key scratch without changing full-KV FP64 attention semantics."""
    kd, vd, qd = k.double(), v.double(), q.double()
    delta_v = vq - vd
    mean_error = delta_v.mean(dim=0, keepdim=True)
    values = {name: torch.empty_like(qd) for name in SCOPES}
    linear_k, linear_kv, linear_kv_cross = [torch.empty_like(qd) for _ in range(3)]
    statistics = {
        key: []
        for key in (
            "sum_p_squared",
            "max_p",
            "entropy",
            "logit_variance",
            "delta_logit_rms",
            "p_weighted_delta_logit_variance",
        )
    }
    for start in range(0, q.shape[0], query_batch):
        qb = qd[start : start + query_batch]
        scores = (qb @ kd.T) / math.sqrt(128)
        quant_scores = (qb @ kq.T) / math.sqrt(128)
        p, pq = scores.softmax(-1), quant_scores.softmax(-1)
        ds = quant_scores - scores
        weighted_ds = (p * ds).sum(-1, keepdim=True)
        dp = p * (ds - weighted_ds)  # Exact softmax Jacobian applied to the measured logit perturbation.
        reference = p @ vd
        outputs = dict(control=reference, K_only=pq @ vd, V_only=p @ vq, KV=pq @ vq)
        outputs["V_only_mean_corrected"] = outputs["V_only"] - mean_error
        outputs["KV_mean_corrected"] = outputs["KV"] - mean_error
        for name, output in outputs.items():
            values[name][start : start + qb.shape[0]] = output
        k_prediction = dp @ vd
        v_error = outputs["V_only"] - reference
        linear_k[start : start + qb.shape[0]] = k_prediction
        linear_kv[start : start + qb.shape[0]] = k_prediction + v_error
        linear_kv_cross[start : start + qb.shape[0]] = k_prediction + v_error + dp @ delta_v
        statistics["sum_p_squared"].append(p.square().sum(-1))
        statistics["max_p"].append(p.amax(-1))
        statistics["entropy"].append(-(p * p.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(-1))
        statistics["logit_variance"].append(scores.var(-1, correction=0))
        statistics["delta_logit_rms"].append(ds.square().mean(-1).sqrt())
        statistics["p_weighted_delta_logit_variance"].append((p * (ds - weighted_ds).square()).sum(-1))
    vectors = {key: torch.cat(parts) for key, parts in statistics.items()}
    stats = {key: dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max())) for key, x in vectors.items()}
    concentration = k.shape[0] * float(vectors["sum_p_squared"].mean())
    stats.update(
        n_times_mean_sum_p_squared=concentration,
        measured_weight_centering_factor=math.sqrt(max(0.0, 1 - 1 / concentration)),
        mean_query_variance_proxy=float(qd.square().mean()),
        gaussian_weight_concentration_proxy=float(qd.square().mean(-1).exp().mean()),
    )
    predictions = dict(
        K_only=linear_k,
        KV=linear_kv,
        KV_with_linearized_P_cross_term=linear_kv_cross,
        KV_mean_corrected=linear_kv - mean_error,
    )
    return values, predictions, stats, mean_error


def heuristics(kstats, vstats, weights, uniform):
    sk, sv = kstats["sigma_relative"], vstats["sigma_relative"]
    if uniform:
        sk = 0.0
    factor = weights["measured_weight_centering_factor"]
    return dict(
        interpretation="Independent additive noise, diffuse Gaussian logits; heuristic, NOT a lower bound",
        representation_sigma_k=kstats["sigma_relative"],
        representation_sigma_v=sv,
        K_only_l2_pct=100 * sk,
        V_only_l2_pct=100 * sv,
        KV_l2_pct=100 * math.hypot(sk, sv),
        V_mean_corrected_measured_weight_proxy_l2_pct=100 * sv * factor,
        KV_mean_corrected_measured_weight_proxy_l2_pct=100 * math.hypot(sk, sv * factor),
        unit_gaussian_centering_factor=math.sqrt(1 - 1 / math.e),
        unit_gaussian_V_mean_corrected_l2_pct=100 * sv * math.sqrt(1 - 1 / math.e) if not uniform else 0.0,
        unit_gaussian_KV_mean_corrected_l2_pct=(
            100 * math.sqrt(sk**2 + sv**2 * (1 - 1 / math.e)) if not uniform else 0.0
        ),
        assumptions="Q/K/V approximately independent unit-variance Gaussian; noise weak, centered, independent across tokens and from weights/values; finite-N ratios concentrate",
        caveats="Shared-scale correlations, clipping/gain bias, K-dependent softmax weights, concentrated logits and finite-sample cancellation violate assumptions",
    )


def self_tests():
    generator = torch.Generator().manual_seed(777)
    q = torch.randn(4, 128, generator=generator).bfloat16()
    k = torch.randn(32, 128, generator=generator).bfloat16()
    v = torch.randn(32, 128, generator=generator).bfloat16()
    kq, _ = C.quantize(k, "tt_rne")
    vq, _ = C.quantize(v, "tt_rne")
    values, predictions, _, mean = calculate(q, k, v, kq, vq, 2)
    dense = ((q.double() @ k.double().T) / math.sqrt(128)).softmax(-1) @ v.double()
    torch.testing.assert_close(values["control"], dense, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(values["V_only_mean_corrected"], values["V_only"] - mean, rtol=0, atol=0)
    uniform, uniform_predictions, _, _ = calculate(torch.zeros_like(q), k, v, kq, vq, 2)
    for name in ("K_only", "V_only_mean_corrected", "KV_mean_corrected"):
        torch.testing.assert_close(uniform[name], uniform["control"], rtol=1e-13, atol=1e-14)
    assert torch.count_nonzero(uniform_predictions["K_only"]) == 0
    # A small finite perturbation tests the Jacobian independently of coarse quantization.
    perturbation = torch.randn(k.shape, generator=generator, dtype=torch.float64)
    epsilon = 1e-5
    tiny, tiny_predictions, _, _ = calculate(q, k, v, k.double() + epsilon * perturbation, v.double(), 2)
    error = tiny["K_only"] - tiny["control"]
    discrepancy = float((tiny_predictions["K_only"] - error).norm() / error.norm())
    assert discrepancy < 1e-3
    assert sampled_rows(1024, 3) == [0, 511, 1023]
    return dict(
        dense_reference=True,
        mean_correction_identity=True,
        uniform_K_invariance=True,
        uniform_mean_correction=True,
        softmax_jacobian_finite_difference=True,
        finite_difference_relative_discrepancy=discrepancy,
    )


def source_hashes():
    files = set(C.source_files()) | {Path(__file__).resolve(), Path(C.__file__).resolve()}
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768, 262144])
    parser.add_argument("--q-rows", type=int, choices=(64, 128), default=64)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1240, 1241])
    parser.add_argument(
        "--long-single-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only first seed for N>32768 to bound the study",
    )
    parser.add_argument("--codecs", nargs="+", choices=CODECS, default=["tt_rne", "nvfp4_e4m3"])
    parser.add_argument("--self-test-only", action="store_true", help="Tiny CPU checks, no result file")
    args = parser.parse_args()
    assert Path(args.label).name == args.label and args.label not in (".", "..")
    assert all(n >= args.q_rows and n % 32 == 0 for n in args.lengths)
    assert args.query_batch > 0 and len(args.seeds) == len(set(args.seeds)) and args.seeds
    assert len(args.codecs) == len(set(args.codecs))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    pins = source_hashes()
    tests = self_tests()
    if args.self_test_only:
        print(json.dumps(dict(self_tests=tests, source_sha256=pins, execution="CPU_ONLY"), allow_nan=False))
        return
    path = HERE / (args.label + ".jsonl")
    assert not path.exists(), "Use a fresh label"
    plan = [
        (n, seed)
        for n in args.lengths
        for seed in (args.seeds[:1] if args.long_single_seed and n > 32768 else args.seeds)
    ]
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
                self_tests=tests,
                execution="CPU-only; no TTNN imports/device calls",
                threads=4,
                torch_version=str(torch.__version__),
                hostname=platform.node(),
                planned_length_seeds=plan,
                scope=__doc__,
                reference="Original BF16 inputs; FP64 QK/softmax/PV/state; no P quantization",
                uniform_control="Replace only sampled Q by exact BF16 zeros; same original K/V",
                grouping="K and V shared groups16 along D for BOTH codecs, no N-axis transpose",
                timing_scope="CPU elapsed time is bookkeeping only, not kernel throughput or model performance",
            )
        )
        cases = 0
        for length, seed in plan:
            q, k, v, rows = make_inputs(length, args.q_rows, seed)
            input_hashes = {name: digest(x) for name, x in zip(("sampled_Q", "K", "V"), (q, k, v))}
            context = dict(
                length=length,
                seed=seed,
                heads=1,
                head_dim=128,
                sampled_query_rows=rows,
                original_input_sha256=input_hashes,
            )
            emit(dict(kind="inputs", **context))
            for codec in args.codecs:
                kq, ks = C.quantize(k, codec, "D")
                vq, vs = C.quantize(v, codec, "D")
                kstats, vstats = representation(k, kq, ks), representation(v, vq, vs)
                for uniform in (False, True):
                    query = torch.zeros_like(q) if uniform else q
                    values, predictions, weights, mean = calculate(query, k, v, kq, vq, args.query_batch)
                    reference = values["control"]
                    # Independent existing online FP64 implementation; this is not a second GPU path.
                    independent_reference = C.REF.reference(query, k, v)
                    torch.testing.assert_close(reference, independent_reference, rtol=1e-11, atol=1e-13)
                    outputs = {
                        name: dict(
                            fp64_output=C.REF.metrics(value, reference),
                            bf16_output=C.REF.metrics(value.bfloat16(), reference),
                        )
                        for name, value in values.items()
                    }
                    predicted = {}
                    for name, error in predictions.items():
                        actual_name = "KV" if name == "KV_with_linearized_P_cross_term" else name
                        predicted[name] = dict(
                            output_vs_original_reference=C.REF.metrics(reference + error, reference),
                            error_agreement=compare_error(error, values[actual_name] - reference),
                        )
                    if uniform:
                        for name in ("K_only", "V_only_mean_corrected", "KV_mean_corrected"):
                            assert outputs[name]["fp64_output"]["l2_pct"] < 1e-8
                    k_error = values["K_only"] - reference
                    v_error = values["V_only"] - reference
                    cross_error = values["KV"] - values["K_only"] - values["V_only"] + reference
                    error_cos_den = float(k_error.norm() * v_error.norm())
                    decomposition = dict(
                        K_V_error_cosine=float((k_error * v_error).sum()) / error_cos_den if error_cos_den else None,
                        measured_K_V_root_sum_squares_l2_pct=math.hypot(
                            outputs["K_only"]["fp64_output"]["l2_pct"], outputs["V_only"]["fp64_output"]["l2_pct"]
                        ),
                        exact_nonlinear_cross_l2_pct=float(100 * cross_error.norm() / reference.norm()),
                        cross_definition="(Pq-P) @ (Vq-V); KV error = K-only error + V-only error + cross",
                        linearized_cross_definition="dp_linear @ (Vq-V); second order in joint K/V noise, not a purely first-order KV prediction",
                    )
                    emit(
                        dict(
                            kind="noise_budget",
                            **context,
                            codec=codec,
                            query_mode="uniform_zero" if uniform else "normal_bf16",
                            query_sha256=digest(query),
                            representations=dict(K=kstats, V=vstats),
                            measured_weights=weights,
                            heuristic=heuristics(kstats, vstats, weights, uniform),
                            outputs=outputs,
                            first_order=predicted,
                            exact_error_decomposition=decomposition,
                            original_fp64_reference_sha256=digest(reference),
                            original_fp64_reference_rms=float(reference.square().mean().sqrt()),
                            online_reference_agreement=C.REF.metrics(reference, independent_reference),
                            correction_mean_error_rms=float(mean.square().mean().sqrt()),
                            correction="Subtract FP64 token mean(Vq-V) from full FP64 attention output before final BF16 round; original V is NOT centered/requantized",
                            all_outputs_finite=all(bool(torch.isfinite(x).all()) for x in values.values()),
                        )
                    )
                    cases += 1
                del kq, vq
            assert input_hashes == {name: digest(x) for name, x in zip(("sampled_Q", "K", "V"), (q, k, v))}
            assert source_hashes() == pins
        emit(
            dict(
                kind="complete",
                cases=cases,
                original_inputs_unchanged=True,
                sources_unchanged=source_hashes() == pins,
                elapsed_cpu_seconds=time.monotonic() - started,
                device_jobs=0,
            )
        )


if __name__ == "__main__":
    main()
