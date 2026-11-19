# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TT-Metal project
"""Short CPU-only P2 sampler benchmark; no device transfer or serving claims.

Each thread setting runs in a fresh subprocess using the plugin's official
hardware-blocking ttnn stub. Only that child's PyTorch thread setting changes.
Synthetic FP32 logits are restored outside timing because sampling mutates them.
The full installed create_host_sampler() call is timed, with persistent seeded
generators and its normal noise-prefetch pool. This is not a P2 acceptance run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    # Import the official stub first, before importing any vLLM/plugin module.
    stub = args.plugin / "tests/host-stubs"
    sys.path.insert(0, str(stub))
    os.environ["VLLM_PLUGINS"] = "tt"
    import ttnn

    assert Path(ttnn.__file__).resolve() == (stub / "ttnn/__init__.py").resolve()
    import torch

    torch.set_num_threads(args.worker_threads)
    torch.set_num_interop_threads(1)
    import vllm.v1.sample.ops.topk_topp_sampler as upstream
    import vllm.v1.sample.sampler as upstream_sampler
    import vllm_tt_plugin.host_sampler as plugin_sampler
    from vllm.platforms import current_platform
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    assert type(current_platform).__name__ == "TTPlatform"
    assert not any(name == "_ttnn" or name.startswith("ttnn._ttnn") for name in sys.modules)
    profiles = json.loads(args.profiles.read_text())
    profile = next(p for p in profiles["profiles"] if p["name"] == "lm_eval_reasoning")
    request = profile["request"]
    # Fail if the installed profile changes instead of silently timing stale args.
    assert request == {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "seed": 42, "n": 1, "max_tokens": 32768}
    results = []
    for rows in (1, 32):
        source = torch.randn(rows, 248320, generator=torch.Generator().manual_seed(12345))
        logits = torch.empty_like(source)
        generators = {row: torch.Generator().manual_seed(request["seed"]) for row in range(rows)}
        metadata = SamplingMetadata(
            temperature=torch.full((rows,), request["temperature"]),
            all_greedy=False,
            all_random=True,
            top_p=torch.full((rows,), request["top_p"]),
            top_k=torch.full((rows,), request["top_k"], dtype=torch.int32),
            generators=generators,
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(rows),
            presence_penalties=torch.zeros(rows),
            repetition_penalties=torch.ones(rows),
            output_token_ids=[[] for _ in range(rows)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        sampler = plugin_sampler.create_host_sampler()
        durations = []
        tokens = []
        start_all = time.perf_counter()
        try:
            with torch.inference_mode():
                for iteration in range(args.warmups + args.iterations):
                    # Not a transfer measurement; emulate fresh host-owned logits.
                    logits.copy_(source)
                    start = time.perf_counter_ns()
                    output = sampler(logits=logits, sampling_metadata=metadata)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if iteration >= args.warmups:
                        durations.append(elapsed)
                        tokens.append(output.sampled_token_ids.flatten().tolist())
            noise = sampler.topk_topp_sampler
            result = {
                "rows": rows,
                "vocab": source.shape[1],
                "dtype": str(source.dtype),
                "sample_ms": durations,
                "p50_ms": float(torch.tensor(durations).median()),
                "p90_ms": float(torch.tensor(durations).quantile(0.9)),
                "mean_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "sampled_token_ids": tokens,
                "final_generator_state_sha256": [
                    hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest() for g in generators.values()
                ],
                "noise_cache_bytes": noise._cached_bytes,
                "pool_max_workers": noise._sampling_pool._max_workers,
                "sampling_row_workers": 1
                if rows == 1
                else max(1, min(4, len(os.sched_getaffinity(0)) // torch.get_num_threads(), rows)),
                "host_logits_bytes": source.numel() * source.element_size(),
            }
        finally:
            if sampler.topk_topp_sampler._sampling_pool is not None:
                sampler.topk_topp_sampler._sampling_pool.shutdown(wait=True)
        result["total_case_seconds_including_restore_warmups_and_pool_drain"] = time.perf_counter() - start_all
        results.append(result)
    sources = [
        Path(plugin_sampler.__file__),
        Path(upstream.__file__),
        Path(upstream_sampler.__file__),
        args.profiles,
        Path(ttnn.__file__),
        Path(__file__),
    ]
    return {
        "scope": "Installed full host sampler only; synthetic logits, no transfer, model, scheduler, HTTP, or P2 acceptance",
        "profile": profile,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpu_model": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "load_average": list(os.getloadavg()),
        "torch_version": torch.__version__,
        "python": sys.version,
        "platform": type(current_platform).__name__,
        "stub": str(ttnn.__file__),
        "source_sha256": {str(path): digest(path) for path in sources},
        "plugin_commit": subprocess.check_output(
            ["git", "-C", str(args.plugin), "rev-parse", "HEAD"], text=True
        ).strip(),
        "plugin_dirty": subprocess.check_output(
            ["git", "-C", str(args.plugin), "status", "--porcelain"], text=True
        ).splitlines(),
        "warmups": args.warmups,
        "iterations": args.iterations,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, default=Path("/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin"))
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(
            "/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/eval_request_profiles.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--worker-threads", type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmups < 1 or args.iterations < 2:
        parser.error("Require at least one warmup and two measured calls")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker_threads:
        args.output.write_text(json.dumps(worker(args), indent=2) + "\n")
        return
    runs = []
    for threads in (1, 2, 4):
        child_path = args.output.with_name(f"{args.output.stem}.threads{threads}.json")
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), VLLM_PLUGINS="tt")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--plugin",
                str(args.plugin),
                "--profiles",
                str(args.profiles),
                "--output",
                str(child_path),
                "--warmups",
                str(args.warmups),
                "--iterations",
                str(args.iterations),
                "--worker-threads",
                str(threads),
            ],
            env=env,
            check=True,
            timeout=120,
        )
        runs.append(json.loads(child_path.read_text()))
    comparisons = []
    for run in runs:
        for reference, case in zip(runs[0]["cases"], run["cases"]):
            comparisons.append(
                {
                    "threads": run["threads"],
                    "rows": case["rows"],
                    "p50_ms": case["p50_ms"],
                    "speedup_vs_one_thread": reference["p50_ms"] / case["p50_ms"],
                    "tokens_equal_one_thread": case["sampled_token_ids"] == reference["sampled_token_ids"],
                    "rng_equal_one_thread": case["final_generator_state_sha256"]
                    == reference["final_generator_state_sha256"],
                }
            )
    report = {
        "limitations": [
            "CPU microbenchmark only: does not waive P2 or demonstrate a serving fix.",
            "Synthetic fixed normal logits restored between calls; not model logits or a model trajectory.",
            "No device readback, row selection, metadata construction, decode, scheduler, networking, or server contention isolation is timed.",
            "Continuous sampling includes normal speculative noise prefetch; final speculative pool drain is outside per-call timing.",
            "Child-process thread settings only; no plugin/runtime edits or live-process/global environment changes.",
            "Short sequential samples on a shared machine: ordering, caches and other workloads can affect timings.",
        ],
        "runs": runs,
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))
    assert all(item["tokens_equal_one_thread"] and item["rng_equal_one_thread"] for item in comparisons)


if __name__ == "__main__":
    main()
