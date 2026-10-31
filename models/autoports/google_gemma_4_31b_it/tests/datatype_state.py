"""Repeated sampled and greedy trace reuse through selected precision construction."""

import json
import os


def run(gen, root):
    prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
    cases = []
    for length, sampled in ((161, False), (129, True), (1025, False), (161, False)):
        expanded = (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
        prompts = [expanded if i % 3 != 2 else None for i in range(gen.batch)]
        params = dict(temperature=0.7, top_k=16, top_p=0.9, seed=123) if sampled else None
        expected = gen.generate_batch(prompts, 16, sampling_params=params, ignore_eos=True, buffered=False)
        before = dict(gen.counters)
        actual = gen.generate_buffered(prompts, 16, sampling_params=params)
        assert actual == expected
        delta = {k: gen.counters[k] - v for k, v in before.items()}
        assert delta["captures"] == 0 and delta["readbacks"] == 1 and delta["token_refreshes"] == 0
        cases.append(
            {
                "length": length,
                "sampled": sampled,
                "exact": True,
                "counters": delta,
                "trace_ids": [str(t) for t in (gen.model_trace, gen.sample_trace, gen.buffered_sample_trace)],
            }
        )
    assert all(c["trace_ids"] == cases[0]["trace_ids"] for c in cases)
    (root / "state.json").write_text(
        json.dumps(
            {
                "scope": "full60layers",
                "batch": gen.batch,
                "cases": cases,
                "diagnostics": {
                    k: os.environ.get(k)
                    for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE")
                },
            },
            indent=2,
        )
        + "\n"
    )
