# SPDX-License-Identifier: Apache-2.0
"""Host-only experiment, NOT a runtime sampler or approved exact replacement."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_PLUGINS"] = ""
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample


def candidate_filter(logits, p, candidate_count=2048, guard=0.0):
    """Real-arithmetic nucleus with optional empirical numerical guard.

    No claim of bit-exact upstream floating-point support: the full upstream
    sort changes softmax reduction order. guard=0 exposes that distinction.
    """
    batch, vocab = logits.shape
    if candidate_count >= vocab:
        return apply_top_k_top_p(logits.clone(), None, p), {"fallback_rows": batch, "candidate_rows": 0}
    values, indices = torch.topk(logits, candidate_count, dim=-1, sorted=True)
    # Double precision full normalization, never renormalize just candidates.
    normalizer = torch.logsumexp(logits.double(), dim=-1, keepdim=True)
    probs = (values.double() - normalizer).exp()
    candidate_mass = probs.sum(-1)
    # Ascending mass = omitted lower mass plus candidate prefix mass.
    ascending_mass = (1 - candidate_mass).unsqueeze(1) + probs.flip(-1).cumsum(-1)
    threshold = (1 - p).double().unsqueeze(1)
    keep = (ascending_mass > threshold).flip(-1)
    keep[:, 0] = True
    unsafe = (candidate_mass < p.double()) | (p <= 0) | (p >= 1) | (~torch.isfinite(logits).all(-1))
    # Repeated equal logits at a partially included cutoff must retain the
    # installed unstable sort's particular index ordering: fall back.
    split = keep[:, :-1] != keep[:, 1:]
    tied_cutoff = ((values[:, :-1] == values[:, 1:]) & split).any(-1)
    close = (ascending_mass - threshold).abs().amin(-1) <= guard
    if guard > 0:
        unsafe |= tied_cutoff | close
    output = torch.full_like(logits, -float("inf"))
    output.scatter_(1, indices, values.masked_fill(~keep, -float("inf")))
    if unsafe.any():
        output[unsafe] = apply_top_k_top_p(logits[unsafe].clone(), None, p[unsafe])
    return output, dict(
        fallback_rows=int(unsafe.sum()),
        candidate_rows=int((~unsafe).sum()),
        tied_cutoff_rows=int(tied_cutoff.sum()),
        near_cutoff_rows=int(close.sum()),
    )


def compare(logits, p, k, guard):
    reference = apply_top_k_top_p(logits.clone(), None, p)
    actual, info = candidate_filter(logits, p, k, guard)
    support_mismatch = (torch.isfinite(reference) != torch.isfinite(actual)).any(-1)
    reference_probs = reference.softmax(-1)
    actual_probs = actual.softmax(-1)
    info.update(
        support_mismatch_rows=int(support_mismatch.sum()),
        max_probability_abs_error=float((reference_probs - actual_probs).abs().max()),
    )
    return info


def main():
    torch.manual_seed(42)
    out = Path(__file__).parents[1] / "doc/benchmark/recovery/top_p_preselection.json"
    cases = []
    # Float32 threshold chosen from upstream's own cumulative mass, including
    # a boundary where rounded cumulative mass differs from real arithmetic.
    for name, logits, p, k in [
        ("tied_cutoff", torch.cat([torch.zeros(1, 80), torch.full((1, 944), -30.0)], 1), torch.tensor([0.5]), 128),
        ("all_equal", torch.zeros(4, 1024), torch.tensor([0.0, 0.5, 0.95, 1.0]), 128),
        ("peaked", torch.randn(4, 4096) * 8, torch.full((4,), 0.95), 256),
        ("diffuse", torch.randn(4, 4096), torch.full((4,), 0.95), 256),
        ("nonfinite", torch.tensor([[0.0, -float("inf"), -float("inf"), -float("inf")]]), torch.tensor([0.95]), 2),
    ]:
        for guard in [0.0, 1e-5]:
            cases.append(dict(case=name, guard=guard, **compare(logits, p, k, guard)))
    # A sweep near exact upstream cutoff exposes numerical support changes.
    near = torch.randn(64, 4096) * 6
    ordered = near.sort(-1).values
    cumul = ordered.softmax(-1).cumsum(-1)
    thresholds = cumul[:, -32]
    for direction in [-1, 0, 1]:
        p = 1 - thresholds
        if direction:
            p = torch.nextafter(p, torch.full_like(p, float("inf") if direction > 0 else -float("inf")))
        for guard in [0.0, 1e-5]:
            cases.append(dict(case="cutoff_boundary", direction=direction, guard=guard, **compare(near, p, 512, guard)))
    # Exact installed host sampler baseline, loaded without importing TT plugin.
    path = Path("/home/ttuser/k2-horizon-mova/vllm-tt-plugin/src/vllm_tt_plugin/host_sampler.py")
    spec = importlib.util.spec_from_file_location("k2_sampling_experiment", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    timings = []
    for sigma in [1.0, 3.0, 5.0, 8.0]:
        logits = torch.randn(32, 250624) * sigma
        p = torch.full((32,), 0.95)
        correctness = compare(logits, p, 2048, 1e-5)
        sampler = mod.TTTopKTopPSampler()
        values = []
        for mode in ["upstream_actual_sampler", "candidate_guarded_sampler"]:
            elapsed = []
            for _ in range(2):
                x = logits.clone()
                start = time.perf_counter()
                if mode == "upstream_actual_sampler":
                    token, _ = sampler.forward_native(x, {}, None, p)
                else:
                    filtered, _ = candidate_filter(x, p, 2048, 1e-5)
                    token = random_sample(filtered.softmax(-1, dtype=torch.float32), {})
                elapsed.append(time.perf_counter() - start)
                assert token.numel() == 32
            values.append(dict(mode=mode, seconds=elapsed))
        timings.append(
            dict(
                synthetic_logit_sigma=sigma,
                correctness=correctness,
                timings=values,
                second_call_speedup=values[0]["seconds"][1] / values[1]["seconds"][1],
            )
        )
    rng_checks = []
    logits = torch.randn(32, 4096) * 8
    p = torch.full((32,), 0.95)
    reference = apply_top_k_top_p(logits.clone(), None, p).softmax(-1)
    actual, _ = candidate_filter(logits, p, 512, 1e-5)
    actual = actual.softmax(-1)
    assert torch.equal(reference, actual)
    for count in [0, 7, 32]:
        global_state = torch.random.get_rng_state()
        first = {i: torch.Generator().manual_seed(i) for i in range(count)}
        second = {i: torch.Generator().manual_seed(i) for i in range(count)}
        tokens_a = random_sample(reference.clone(), first)
        next_global = torch.random.get_rng_state()
        torch.random.set_rng_state(global_state)
        tokens_b = random_sample(actual.clone(), second)
        rng_checks.append(
            dict(
                explicit_generators=count,
                tokens_identical=bool(torch.equal(tokens_a, tokens_b)),
                global_rng_state_identical=bool(torch.equal(next_global, torch.random.get_rng_state())),
                all_request_rng_states_identical=all(
                    torch.equal(first[i].get_state(), second[i].get_state()) for i in first
                ),
            )
        )
    assert not any(n == "ttnn" or n.startswith("ttnn.") for n in sys.modules)
    result = dict(
        torch_threads=torch.get_num_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        batch=32,
        vocab=250624,
        top_p=0.95,
        top_k=None,
        candidate_count=2048,
        ttnn_imported=False,
        cases=cases,
        timings=timings,
        rng_checks=rng_checks,
        verdict="Not approved as an exact upstream replacement. Real-arithmetic proof does not guarantee upstream floating-point cutoff/tie behavior; empirical numerical guard is not a proven error bound. Speed is conditional on peaked logits and is unverified on actual model logits.",
        rng="The experiment retains full-vocabulary random_sample, including its global/optional row-generator semantics. No candidate-only RNG or changed sample-space draw count.",
        guard="1e-5 absolute cumulative-probability proximity triggers full upstream fallback; tie-cutoff and insufficient mass also fallback. This tolerance is empirical, not a correctness proof.",
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
