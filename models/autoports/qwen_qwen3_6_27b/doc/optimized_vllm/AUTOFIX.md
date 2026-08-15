# Optimized vLLM AutoFix report

## Trigger

The first post-optimization full plugin run passed every adjacent seed test but
failed `test_specific_seed_reproducible[999]`. Remaining stages were skipped and
the server shut down cleanly. The failure is retained under
`artifacts/autofix_seed_rng/`.

## Isolation

The failing request used `top_k=50`; Qwen advertises a maximum on-device top-k
of 32. It therefore exercised the plugin host-compatibility sampler, not the
adapter's device `SeedManager` or the newly cached device sampler parameters.
Adding seed to the device sampler key was refuted because
`reset_sampling_params` does not own seed state.

Source inspection found that `_build_host_generators` returned the live
canonical `torch.Generator` for a submitted row and immediately advanced that
same object. Async input preparation could mutate an earlier submission before
host sampling consumed it.

## Fix and proof

The plugin now clones every submitted host generator. It advances canonical
state exactly once for normal rows and leaves intermediate-prefill canonical
state unchanged. The host regression asserts identity separation, submitted
state preservation, deferred-draw stability, canonical advancement, and
intermediate preservation.

Focused command:

```bash
cd /home/mvasiljevic/vllm
python -m pytest -q plugins/vllm-tt-plugin/tests/test_lane_model_runner.py \
  -k build_host_generators
```

Result: 1 passed. The complete live suite then passed 72 tests with one skip,
including seed 999, same-seed cross-batch reproduction, different-seed variety,
mixed sampling parameters, and all device penalty tests. No unproven fix was
kept.

Separate active-Ethernet heartbeat stalls occurred only during later mesh open,
before model execution. Bounded `tt-smi -r` recovery followed by a 1x4 mesh
open/close smoke succeeded; software gates were rerun rather than waived.
