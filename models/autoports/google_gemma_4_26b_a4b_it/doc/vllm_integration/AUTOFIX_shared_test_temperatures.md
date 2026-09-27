# Shared temperature fixture compatibility

Date: 2026-09-27. Scope: shared tests only; no serving runtime changes, device execution, or server requests.

The installed vLLM0.26 `sampling_params.py:544-555` validates temperature as nonnegative and at most 2.0. Six request construction sites in `tests/tt/test_seeding_and_variety.py` used 5, 10.0, or 50, covering 13 parametrized test items. They were invalid before any model execution. The other shared `tests/tt/test_*.py` literal temperatures were within the accepted range.

The smallest repair replaces those six literals with **2.0**, the supported maximum, and updates four corresponding failure messages. No seed, prompt, `top_k`, generation length, repetition count, greedy comparison, or variety threshold changed. An AST comparison against the original file proved that temperature values and message text are the only semantic edits; evidence is `readiness_vllm/shared_temperature_source_proof.json`.

`test_topk` still sends a greedy half-batch at temperature 0 and a sampled half-batch with `top_k=10`, and retains all within/between-batch sequence and first-character variety assertions. Positive temperature scaling preserves the top-k ranking for fixed exact logits; temperature 2 produces a more concentrated distribution than 50, so observed variety is not presumed equivalent. The unchanged live assertions must still pass; CPU request validation does not establish model variety or seeded reproducibility.

Added `vllm/plugins/vllm-tt-plugin/tests/test_sampling_test_temperatures.py`. It intercepts each affected test's first batch before networking and constructs the actual installed `SamplingParams` for every request. It also checks all explicit literal temperatures in the shared test tree and retains three negative controls proving that the old values are rejected.

Before repair: **14 failed, 3 passed, 2 warnings in 5.22 seconds**. All 13 affected configurations and the all-files scan failed with `temperature must be in [0, 2]`; the three expected-rejection controls passed.

After repair: **17 passed, 2 warnings in 4.70 seconds**. Command, from `/workspace/tt-metal/vllm/plugins/vllm-tt-plugin`:

```bash
python -m pytest tests/test_sampling_test_temperatures.py -q --disable-warnings --tb=short
python -m black --check --target-version py310 tests/test_sampling_test_temperatures.py
python -m black --check --target-version py310 --line-ranges 185-205 --line-ranges 234-240 --line-ranges 259-310 tests/tt/test_seeding_and_variety.py
git diff --check
```

Checks passed. Baseline and repaired transcripts are `readiness_vllm/shared_temperature_baseline.log` and `readiness_vllm/shared_temperature_host_tests.log`; the active full-suite logs were left untouched. These Python-only test changes need no C++ build. The main stage will rerun the affected serving items against its full model and retain the original assertions.
