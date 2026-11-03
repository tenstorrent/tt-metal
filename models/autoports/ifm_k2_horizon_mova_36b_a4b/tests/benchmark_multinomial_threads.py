# SPDX-License-Identifier: Apache-2.0
"""Measure actual TT sampler versus experimental checked multinomial wrapper."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_PLUGINS"] = ""
import torch
from benchmark_multinomial_sampler import ExperimentalMultinomialTopKTopPSampler


def main():
    path = Path("/home/ttuser/k2-horizon-mova/vllm-tt-plugin/src/vllm_tt_plugin/host_sampler.py")
    spec = importlib.util.spec_from_file_location("k2_multinomial_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    torch.manual_seed(773)
    logits_by_sigma = {s: torch.randn(32, 250624) * s for s in [1.0, 5.0, 8.0]}
    p = torch.full((32,), 0.95)
    rows = []
    for threads in [8, 16]:
        torch.set_num_threads(threads)
        for sigma, logits in logits_by_sigma.items():
            for generator_count in [0, 7, 32]:
                results = {}
                for name, cls in [
                    ("installed_tt_sampler", mod.TTTopKTopPSampler),
                    ("experimental_checked_multinomial", ExperimentalMultinomialTopKTopPSampler),
                ]:
                    sampler = cls()
                    gens = {i: torch.Generator().manual_seed(i) for i in range(generator_count)}
                    times = []
                    for _ in range(2):
                        values = logits.clone()
                        start = time.perf_counter()
                        token, _ = sampler.forward_native(values, gens, None, p)
                        times.append(time.perf_counter() - start)
                        assert token.numel() == 32
                    results[name] = times
                    if getattr(sampler, "_sampling_pool", None) is not None:
                        sampler._sampling_pool.shutdown(wait=True)
                rows.append(
                    dict(
                        threads=threads,
                        sigma=sigma,
                        explicit_generator_count=generator_count,
                        times_seconds=results,
                        second_call_speedup=results["installed_tt_sampler"][1]
                        / results["experimental_checked_multinomial"][1],
                    )
                )
    assert not any(n == "ttnn" or n.startswith("ttnn.") for n in sys.modules)
    result = dict(
        batch=32,
        vocab=250624,
        top_p=0.95,
        top_k=None,
        temperature=1.0,
        repeats=2,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        rows=rows,
        ttnn_imported=False,
        scope="Whole sampler call including unchanged upstream full filter, softmax, validity checks, global/private draw ownership and output selection; input cloning outside timed call. No server/device work.",
        distribution="PyTorch multinomial replacement draws have categorical marginal w_i/sum(w); discarding second draw does not alter first marginal. Upstream filter and its finite cutoff/tie support remain identical. Finite RNG and accumulation algorithm differ from exponential racing; no bitwise token/RNG equivalence claimed.",
        edgecases="Invalid/nonfinite/zero/negative rows and requested fp64 Gumbel use original random_sample before any replacement RNG draw. Empty batch returns empty. Valid finite extreme weights rely on standard torch.multinomial numerical behavior.",
        rng="No seeding policy change. For any unseeded rows, global RNG draws batch of two categorical samples per row; private row generators overwrite their own rows. All-private batches avoid global RNG. Sequence advancement changes from vocabulary-sized exponential draws to two categorical draws.",
        deployment="Experimental model-local helper only; no installed plugin/runtime changes. Host timings do not establish model serving throughput or full-stage completion.",
    )
    path = Path(__file__).parents[1] / "doc/benchmark/recovery/multinomial_threads.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
