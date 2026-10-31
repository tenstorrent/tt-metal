"""Full-stack output and exact workload evidence on an already prepared generator."""

import json
import os
import time
from unittest.mock import patch


def qualitative(gen, root):
    controls = json.loads((root / "qualitative_hf.json").read_text())
    if len(controls) != 6:
        raise RuntimeError("Wait for all six pinned HF controls before the shared suite")
    outputs = []
    for control in controls:
        print("TT_QUALITATIVE", control["id"], flush=True)
        count = len(control["generated_tokens"])
        generated = gen.generate(control["prompt_tokens"], count)
        outputs.append(
            dict(
                id=control["id"],
                generated_tokens=generated,
                completion=gen.tokenizer.decode(generated),
                hf_completion=control["completion"],
                first_divergence=next(
                    (i for i, (a, b) in enumerate(zip(generated, control["generated_tokens"])) if a != b), None
                ),
            )
        )
        (root / "qualitative_tt.json").write_text(json.dumps(outputs, indent=2) + "\n")
        print(outputs[-1]["completion"], flush=True)
    return outputs


def performance(gen, root, *, repeats=3):
    # Performance-only prompt: valid chat-template IDs expanded to exactly4096.
    # This is not a qualitative or accuracy sample.
    prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
    prompt = (prompt * ((4096 + len(prompt) - 1) // len(prompt)))[:4096]
    run = (
        (lambda: gen.generate(prompt, 128, ignore_eos=True, buffered=False))
        if gen.batch == 1
        else (lambda: gen.generate_batch([prompt] * gen.batch, 128, ignore_eos=True, buffered=False))
    )
    run()  # workload warmup, including request page ownership
    samples = []
    for repeat in range(repeats):
        first = []
        original = gen.prefill_forward

        def timed_prefill(*args, **kwargs):
            result = original(*args, **kwargs)
            first.append(time.perf_counter())
            return result

        counters = dict(gen.counters)
        with patch.object(gen, "prefill_forward", side_effect=timed_prefill):
            start = time.perf_counter()
            output = run()
            end = time.perf_counter()
        assert len(first) == 1
        assert gen.counters["token_refreshes"] == counters["token_refreshes"]
        assert gen.counters["position_refreshes"] - counters["position_refreshes"] == 1
        assert gen.counters["page_refreshes"] == counters["page_refreshes"]
        assert gen.counters["model_replays"] - counters["model_replays"] == 127
        assert len(output) == (128 if gen.batch == 1 else gen.batch)
        samples.append(
            dict(
                ttft_ms=(first[0] - start) * 1000,
                decode_tps=127 / (end - first[0]),
                decode_ms=(end - first[0]) * 1000 / 127,
                total_seconds=end - start,
                counters={k: gen.counters[k] - v for k, v in counters.items()},
            )
        )
        print("FULL_PERFORMANCE", gen.batch, repeat, samples[-1], flush=True)
    import statistics

    result = dict(
        scope="full60layers",
        diagnostics={
            key: os.environ.get(key, "0")
            for key in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_WATCHER",
                "TT_METAL_DPRINT_CORES",
            )
        },
        source="autoregressive",
        input_tokens=4096,
        generated_tokens=128,
        batch_size=gen.batch,
        concurrent_requests=gen.batch,
        ttft_ms=statistics.median(s["ttft_ms"] for s in samples),
        decode_tps=statistics.median(s["decode_tps"] for s in samples),
        decode_ms=statistics.median(s["decode_ms"] for s in samples),
        timing="host end-to-end; first token includes request reset, prefill and traced sampling; decode includes traced model, sampling and caller token readback",
        samples=samples,
    )
    (root / f"performance_b{gen.batch}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
