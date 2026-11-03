# SPDX-License-Identifier: Apache-2.0
"""Host-only thread/sampler comparison; full upstream top-p filter retained."""
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_PLUGINS"] = ""
import numpy as np
import torch
from benchmark_cdf_sampler import cpu_cdf_sample, exact_cdf_or_none
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample


def measure(fn):
    samples = []
    for _ in range(2):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return samples


def main():
    torch.manual_seed(717)
    logits_by_sigma = {s: torch.randn(32, 250624) * s for s in [1.0, 5.0, 8.0]}
    p = torch.full((32,), 0.95)
    rows = []
    for threads in [4, 8, 16]:
        torch.set_num_threads(threads)
        for sigma, logits in logits_by_sigma.items():
            filtered = apply_top_k_top_p(logits.clone(), None, p)
            probs = filtered.softmax(-1)

            def sample(sampler):
                values = apply_top_k_top_p(logits.clone(), None, p).softmax(-1)
                return sampler(values)

            timings = {
                "upstream_e2e": measure(lambda: sample(lambda q: random_sample(q, {}))),
                "cdf_exact_intervals_e2e": measure(lambda: sample(lambda q: cpu_cdf_sample(q, {}))),
                "multinomial_one_e2e": measure(lambda: sample(lambda q: torch.multinomial(q, 1, replacement=True))),
                "multinomial_two_take_first_e2e": measure(
                    lambda: sample(lambda q: torch.multinomial(q, 2, replacement=True)[:, 0])
                ),
                "cdf_exact_intervals_draw_only": measure(lambda: cpu_cdf_sample(probs, {})),
                "multinomial_one_draw_only": measure(lambda: torch.multinomial(probs, 1, replacement=True)),
                "multinomial_two_draw_only": measure(lambda: torch.multinomial(probs, 2, replacement=True)[:, 0]),
            }
            rows.append(
                dict(
                    threads=threads,
                    sigma=sigma,
                    exact_intervals_accepted=exact_cdf_or_none(probs) is not None,
                    timings_seconds=timings,
                    second_call_speedups={
                        name: timings["upstream_e2e"][1] / times[1]
                        for name, times in timings.items()
                        if name.endswith("_e2e") and name != "upstream_e2e"
                    },
                )
            )
    # Sorting-only NumPy baseline: does not preserve upstream tie decisions or
    # floating softmax/cumulative calculations, so not a drop-in implementation.
    logits = logits_by_sigma[1.0].numpy()
    numpy_sort = measure(lambda: np.argsort(logits, axis=-1))
    assert not any(n == "ttnn" or n.startswith("ttnn.") for n in sys.modules)
    result = dict(
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        batch=32,
        vocab=250624,
        p=0.95,
        k=None,
        repeats=2,
        ttnn_imported=False,
        rows=rows,
        numpy_argsort_only_seconds=numpy_sort,
        notes=[
            "Original upstream filter used for every end-to-end row.",
            "Two-draw multinomial keeps the first categorical draw and discards the second; RNG advancement changes and private generators need per-row calls.",
            "CDF helper falls back before random draws unless interval widths exactly equal input float32 probability weights.",
            "Each synthetic logits tensor is reused across thread configurations; no live server/hardware.",
            "NumPy sorting is timing-only; not an accepted replacement for upstream floating cutoff or tie ordering.",
        ],
    )
    path = Path(__file__).parents[1] / "doc/benchmark/recovery/cdf_threads.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
