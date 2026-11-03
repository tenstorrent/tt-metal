# SPDX-License-Identifier: Apache-2.0
"""Host-only sampling component and categorical inverse-CDF experiment."""
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_PLUGINS"] = ""
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample, sample_with_exponential_noise


def cdf_sample(probs, generators):
    cumulative = probs.double().cumsum(-1)
    batch, vocab = probs.shape
    if len(generators) != batch:
        uniform = torch.rand((batch, 1), dtype=torch.float64)
    else:
        uniform = torch.empty((batch, 1), dtype=torch.float64)
    for row, gen in generators.items():
        uniform[row] = torch.rand((1,), dtype=torch.float64, generator=gen)
    total = cumulative[:, -1:]
    target = torch.minimum(uniform * total, torch.nextafter(total, torch.zeros_like(total)))
    return torch.searchsorted(cumulative, target, right=True).flatten()


def measured(function, repeats=2):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = function()
        times.append(time.perf_counter() - start)
    return times


def main():
    torch.set_num_threads(8)
    torch.manual_seed(891)
    results = []
    for sigma in [1.0, 5.0, 8.0]:
        logits = torch.randn(32, 250624) * sigma
        p = torch.full((32,), 0.95)
        filtered = apply_top_k_top_p(logits.clone(), None, p)
        probs = filtered.softmax(-1)
        noise = torch.empty_like(probs).exponential_()
        components = {
            "upstream_top_p_filter_including_logits_clone": measured(
                lambda: apply_top_k_top_p(logits.clone(), None, p)
            ),
            "filtered_softmax": measured(lambda: filtered.softmax(-1)),
            "exponential_noise_generation": measured(lambda: torch.empty_like(probs).exponential_()),
            "division_argmax_including_probs_clone": measured(
                lambda: sample_with_exponential_noise(probs.clone(), noise)
            ),
            "upstream_random_sample_including_probs_clone": measured(lambda: random_sample(probs.clone(), {})),
            "cdf_double_cumsum": measured(lambda: probs.double().cumsum(-1)),
            "cdf_complete_sampler": measured(lambda: cdf_sample(probs, {})),
        }
        cumulative = probs.double().cumsum(-1)
        target = torch.rand((32, 1), dtype=torch.float64) * cumulative[:, -1:]
        components["cdf_searchsorted_only"] = measured(lambda: torch.searchsorted(cumulative, target, right=True))

        def end_to_end(sampler):
            mask = apply_top_k_top_p(logits.clone(), None, p)
            return sampler(mask.softmax(-1), {})

        components["upstream_end_to_end"] = measured(lambda: end_to_end(random_sample))
        components["cdf_end_to_end"] = measured(lambda: end_to_end(cdf_sample))
        # Differences between ideal inverse-CDF intervals and normalized
        # upstream float32 probabilities, evaluated in double precision.
        widths = torch.diff(torch.cat([torch.zeros((32, 1), dtype=torch.float64), cumulative], 1), dim=-1)
        desired = probs.double() / probs.double().sum(-1, keepdim=True)
        actual = widths / cumulative[:, -1:]
        results.append(
            dict(
                logit_sigma=sigma,
                retained_support=[int(x) for x in (probs > 0).sum(-1)],
                components_seconds=components,
                second_call_e2e_speedup=components["upstream_end_to_end"][1] / components["cdf_end_to_end"][1],
                ideal_cdf_probability_max_abs_error=float((desired - actual).abs().max()),
                cdf_zero_interval_with_positive_probability=int(((widths == 0) & (probs > 0)).sum()),
            )
        )
    # Adversarial support and small distribution smoke: 0-probability entries
    # must never be drawn, including the leading and trailing entries.
    expected = torch.tensor([[0.0, 0.1, 0.0, 0.2, 0.7, 0.0]], dtype=torch.float32)
    expanded = expected.expand(100000, -1)
    draws = cdf_sample(expanded, {})
    frequency = torch.bincount(draws, minlength=6).double() / len(draws)
    ownership = []
    for count in [0, 7, 32]:
        generators = {i: torch.Generator().manual_seed(i) for i in range(count)}
        before = {i: g.get_state().clone() for i, g in generators.items()}
        before_global = torch.random.get_rng_state().clone()
        token = cdf_sample(torch.tensor([[0.0, 1.0, 0.0]]).expand(32, -1), generators)
        ownership.append(
            dict(
                explicit_generators=count,
                all_tokens_in_positive_support=bool((token == 1).all()),
                supplied_generators_advanced=all(
                    not torch.equal(before[i], generators[i].get_state()) for i in generators
                ),
                global_generator_advanced=not torch.equal(before_global, torch.random.get_rng_state()),
            )
        )
    assert not any(n == "ttnn" or n.startswith("ttnn.") for n in sys.modules)
    output = dict(
        torch_threads=torch.get_num_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        batch=32,
        vocab=250624,
        top_p=0.95,
        temperature=1.0,
        top_k=None,
        ttnn_imported=False,
        components=results,
        small_distribution_test=dict(
            samples=len(draws),
            expected=expected.flatten().tolist(),
            observed=frequency.tolist(),
            masked_tokens_drawn=int(((draws == 0) | (draws == 2) | (draws == 5)).sum()),
        ),
        generator_ownership=ownership,
        mathematical_contract="In real arithmetic, inverse CDF with independent U uniform[0,1) samples category i with probability w_i/sum(w). Full upstream filter remains unchanged, preserving its cutoff and tie decisions. Double cumulative sums introduce their own roundoff; this is not bitwise identity with float32 exponential racing.",
        rng_contract="Global RNG supplies batch uniforms when any requests lack private generators; supplied row generators overwrite their own uniforms; all-private batch avoids global RNG. One float64 uniform per sampled row replaces vocabulary-sized exponential noise, so RNG stream advancement and generated sequences CHANGE even for same seeds. No seeding policy change is required for unseeded requests.",
        limitations="Synthetic logits, two repetitions per component. Component timings include listed clones and do not add exactly to end-to-end measurements. No actual model logits or serving measurement. No runtime integration installed.",
    )
    path = Path(__file__).parents[1] / "doc/benchmark/recovery/sampling_components.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
