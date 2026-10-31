"""End-to-end fixed-step token-out timing; first-token and final reads only."""
import json
import os
import statistics
import time
from unittest.mock import patch


def performance(gen, root, *, repeats=3):
    prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
    prompt = (prompt * ((4096 + len(prompt) - 1) // len(prompt)))[:4096]
    prompts = [prompt] * gen.batch
    expected = gen.generate_batch(prompts, 128, ignore_eos=True, buffered=False)
    actual = gen.generate_buffered(prompts, 128)
    assert actual == expected, "Buffered output must equal streaming at exact benchmark workload"
    samples = []
    for repeat in range(repeats):
        first = []
        original = gen.prefill_forward

        def timed_prefill(*args, **kwargs):
            out = original(*args, **kwargs)
            # One first-token read establishes real host TTFT. No read occurs
            # inside the following 127-step fixed device-feedback loop.
            gen._read_tokens()
            first.append(time.perf_counter())
            return out

        before = dict(gen.counters)
        with patch.object(gen, "prefill_forward", side_effect=timed_prefill):
            start = time.perf_counter()
            output = gen.generate_buffered(prompts, 128)
            end = time.perf_counter()
        counts = {k: gen.counters[k] - v for k, v in before.items()}
        assert output == expected
        assert counts["model_replays"] == 127 and counts["sample_replays"] == 128
        assert counts["readbacks"] == 2 and counts["position_refreshes"] == 1
        assert counts["token_refreshes"] == counts["page_refreshes"] == counts["captures"] == 0
        samples.append(
            dict(
                ttft_ms=(first[0] - start) * 1000,
                decode_tps=127 / (end - first[0]),
                decode_ms=(end - first[0]) * 1000 / 127,
                counters=counts,
            )
        )
        print("OPTIMIZED_PERFORMANCE", gen.batch, repeat, samples[-1], flush=True)
    result = dict(
        scope="full60layers",
        source="autoregressive",
        input_tokens=4096,
        generated_tokens=128,
        batch_size=gen.batch,
        concurrent_requests=gen.batch,
        ttft_ms=statistics.median(s["ttft_ms"] for s in samples),
        decode_tps=statistics.median(s["decode_tps"] for s in samples),
        decode_ms=statistics.median(s["decode_ms"] for s in samples),
        samples=samples,
        streaming_exact=True,
        diagnostics={
            k: os.environ.get(k, "0")
            for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER")
        },
        timing="Host TTFT through first token read; decode includes 127 nonblocking model/buffered-sampling replays and one final history read; no per-token host boundary",
    )
    (root / f"performance_b{gen.batch}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
