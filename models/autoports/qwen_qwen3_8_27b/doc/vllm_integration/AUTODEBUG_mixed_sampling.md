# AutoDebug: mixed-batch seeded sampling reproducibility

## Scope and observed failure

Initial source-only investigation on 2026-09-14. The subsequently authorized
CPU-only proof and native seed repair are recorded separately in
`AUTOFIX_native_sampling.md`. This agent performed no model execution, device
access, or shared-test changes. The operator-owned root `AUTODEBUG.md` was not
modified.

The parent runs the full 64-layer Qwen model with TP4, max sequences 32,
`QWEN_VLLM_HOST_COMPATIBILITY=1`, and the shared full sampling profile:

```bash
QWEN_VLLM_MAX_SEQS=32 QWEN_VLLM_HOST_COMPATIBILITY=1 \
  bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh \
  --stages serve --sampling-profile full
```

The completed log is preserved as `full_sampling_mixed_failure.log`:
**72 passed, 1 failed in 940.84 seconds**. The only failure is
`test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch`:

```text
Config:temp=0.01,seed=42
Run 1: '201'
Run 2: '197'
```

This identifies `Letter: `, max tokens 3. The first differing token ID and
per-step backend choice are not recorded by the original test. The parent owns
runtime follow-up and final validation.

Subsequent parent-owned controls on the original loaded source:

- `readiness_vllm/mixed_targeted_original.log`: unchanged isolated item passed
  in 22.47 seconds, establishing that the original failure is intermittent.
- `readiness_vllm/mixed_host_control.log`: same body with diagnostic
  `logprobs=0` on all requests passed in 22.31 seconds. Because the ordinary
  rerun also passed, this control alone does not prove routing caused the
  original failure.
- `readiness_vllm/full_concurrent_control_before_seed_fix.log`: four distinct
  chat prompts passed all 12 native-versus-forced-synchronous comparisons.
  These are additional state-correctness controls, not exact failing-prompt
  full-logit evidence.

## Source findings

1. **A seeded request can use different random algorithms depending on its
   neighbors.** `tt/generator_vllm.py:70-81` accepts device sampling only for
   greedy requests or stochastic `top_k` 1..32 without penalties. The plugin's
   `TTModelRunner.check_perform_device_sampling` (`model_runner.py:2589-2642`)
   returns one decision for the whole active batch. The original test sends
   concurrent requests, shuffles their submission order, and repeats. These
   are separate HTTP calls, so a concurrent burst does not guarantee that all
   requests share their first prefill or every subsequent decode batch.

   `Count: ` (greedy), `Word: ` (`top_k=1`), `Number: ` (`top_k=5`),
   `Letter: ` and `TopP: ` are device-supported. The last two omit `top_k`
   from their HTTP body, so they inherit the HF generation configuration's
   **top_k=20**, as recorded at `readiness_vllm/server.log:146`. Their requests
   explicitly supply `top_p`, overriding that configuration's default 0.95.
   The report's initial prediction that only `Number: ` was stochastic and
   device-supported was incorrect: it overlooked the server's generation
   defaults. The actual failing `Letter: ` is supported and can switch
   backends; its low temperature also makes raw-logit comparison especially
   important before attributing the mismatch to random draws.

   The plugin explicitly documents this limitation at
   `model_runner.py:1306-1311`. Host seeded generators are initialized per
   request (`input_batch.py:65-67`), move with requests during condensation
   (`input_batch.py:531-533`), and are passed to the host sampler only on host
   steps (`model_runner.py:1300-1305`). Host sampling uses PyTorch exponential
   random draws (`vllm/v1/sample/ops/topk_topp_sampler.py:311-333`). The TT
   platform is an out-of-tree platform, whose default custom-op dispatch
   reaches `forward_native`, not `forward_cpu`; the latter's different
   treatment of partial generator dictionaries is not the active source path.
   Native TT sampling instead calls `ttnn.manual_seed` immediately before
   `ttnn.sampling` (`models/common/sampling/tt_sampling.py:1109-1122`). No code
   translates RNG state between these algorithms. Even one switched prefill
   draw changes the host generator's later history.

2. **Native sampling also rewinds existing requests on batch changes.** The
   following describes the pre-fix source used for the CPU failure proof.
   `tt/generator_vllm.py:123-133` keys parameters by the whole batch and writes
   each row's original seed whenever `reset=True` or any row's parameters
   change. Prefill unconditionally requests this reset at line 155; decode
   requests it on layout reset or its first call after prefill at lines
   193-194. `QwenGenerator.set_batch_sampling_params` overwrites the full seed
   tensor (`tt/generator.py:400-407`). Normal sampling increments it once per
   actual sample (`tt/generator.py:347-349`). Neither adapter setup nor
   recurrent-slot remapping preserves or reconstructs each continuing
   request's consumed random draws.

   Thus an otherwise identical native request sees a different draw after a
   companion finishes, a request is admitted, or rows are compacted. This is
   a separate source-level defect; fixing it cannot make TT and PyTorch RNG
   streams equivalent. The existing common seed manager has a relevant
   precedent: `models/common/sampling/generator.py:969-1016` aligns explicit
   seed counters to authoritative decode positions. Its transformer caller
   deliberately avoids lagging host positions during overlapped steady decode
   (`models/tt_transformers/tt/generator.py:2339-2368`).

3. **No source evidence here establishes cache contamination.** Recurrent
   state reset targets only new prefill slots; decode applies the full slot
   permutation before stepping. Host decode cannot enter the steady overlap
   path (`async_decode.py:317-324`), and host-to-device transition forces
   authoritative token/position refresh in the adapter. The trace warmup
   backs up and restores the device seed tensor (`tt/generator.py:419-435`).
   These facts remove several obvious causes but do not prove model-state
   correctness. Raw logits must be compared before classifying an output
   difference as reproducibility-only.

## Minimal verify/refute experiments for the parent

1. **Completed assertion.** `Letter: `, temperature 0.01, seed 42 differs as
   `'201'` versus `'197'`. Preserve token IDs, earliest differing generation
   index, and backend choices in the focused rerun. The original failure
   alone does not distinguish RNG behavior from logit/state differences.

2. **Backend control with the existing server.** Repeat `Letter: ` with
   `temperature=0.01`, `top_k=20`, `seed=42`, max tokens 3 both alone and with an
   overlapping long host-only companion. Compare to an identical request
   explicitly asking for TP4 logprobs to force host sampling. Request full
   responses with token IDs, and temporarily record sampling mode, request
   ID, slot, and absolute position at each step if scheduling otherwise
   remains ambiguous. If the first differing draw corresponds to a backend
   change while raw logits match, finding 1 is confirmed. Merely observing
   different text under two random algorithms is insufficient to rule out a
   simultaneous state bug.

3. **One-backend compatibility control.** Without restarting the live server,
   rerun the original item, then invoke the same test body with a local
   `RequestConfig` wrapper adding `logprobs=0` to every request. TP4 logprobs
   force the existing host route. This is a diagnostic control, not a claim
   that the unchanged full profile passes. Retain both results and compare
   the actual failing config. The parent has selected this first control.

   If it confirms the routing mechanism, an explicit, separately labeled
   `QWEN_VLLM_HOST_COMPATIBILITY=all` mode could make the model capability hook
   return False even for supported requests, while allowing its existing
   host-logit boundary. At initial diagnosis `all` was not implemented: the
   checks accepted only `1`. The parent owns that separate configuration
   change. Host-test the mode policy first, then run the unchanged original
   isolation test on a freshly started server, followed by the unchanged
   full profile. This prevents batch neighbors from selecting another RNG
   backend without changing shared tests or their requested parameters.
   Passing proves that configuration's compatibility behavior; it does not
   repair or validate native RNG continuity, cross-backend equivalence, or
   device performance. Final native benchmarks must unset compatibility and
   retain canonical traced device sampling.

   Focused rerun, from the sibling vLLM checkout with the established
   environment, after the parent has prepared the chosen server mode:

   ```bash
   ../tt-metal/python_env/bin/python -m pytest \
     plugins/vllm-tt-plugin/tests/tt/test_request_isolation.py \
     --tt-server-url http://localhost:8000 \
     --tt-model-name Qwen/Qwen3.8-27B --tt-max-num-seqs 32 -v --tb=short
   ```

4. **Native RNG continuity independent of mixed mode.** Use two supported
   stochastic requests with deliberately unequal lengths, plus a longer
   survivor replayed alone. First use a fake generator that models seed
   increments: after several decode calls, introduce `reset_batch=True`
   with a companion departure and compare the survivor's next seed. Reuse
   `tests/test_vllm_adapter_host.py`'s fake generator, adding only a torch seed
   tensor and recording/incrementing it once in each fake decode. With seed
   42, three previous calls, and the same authoritative survivor position,
   the predicted fourth pre-draw seed is 45 in steady control and 42 after
   reset. A companion-parameter-only change with `reset_batch=False` should
   also rewind the survivor through the whole-batch key. Add a `[1,0]` remap
   case that compares the same request before and after moving rows. These
   probes exercise the real adapter's decisions, without cache computation.
   A runtime confirmation should record
   the same survivor's logits and pre-draw seed in identical absolute
   positions, so random state is isolated from model-state differences.
   The smallest coherent repair is to establish one consistent
   seed-plus-token-position convention for both prefill and decode, or retain
   per-request counters through remaps. Only re-anchor at authoritative
   refreshes; leave steady asynchronous replay's self-advancing seed intact.
   Changing a seed formula without this focused proof would be speculative.

5. **Logit/state localization if the one-backend control fails.** Compare
   normalized full-vocabulary logits for the exact failing prompt, both alone
   and across occupied slots/admission/departure schedules. Compare the first
   divergent decode distribution using the same forced token prefix, then
   compare that prefix against the standalone selected-policy generator.
   `tests/check_vllm_logit_determinism.py` currently samples two chat prompts
   at one prefill output; passing it alone does not cover an arbitrary legacy
   completion prompt or an observed later decode mismatch. Extend the probe
   to the actual boundary when required, without weakening the shared test.

## Verdict and remaining gate

Two mechanisms that can break exact seeded batch-order reproduction were
established in source: batch-level host/device routing and native whole-batch
seed resets. The latter is independently verified and repaired by CPU tests
in `AUTOFIX_native_sampling.md`. Discriminating runtime evidence is still
needed to attribute the original `Letter: ` failure; no hardware success is
claimed. Shared full-profile success, selected-policy standalone/serving logit
agreement, qualitative/correctness checks, and separately measured native
performance remain the parent's completion gates.
