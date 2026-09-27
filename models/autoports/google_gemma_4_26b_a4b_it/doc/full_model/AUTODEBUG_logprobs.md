# AutoDebug: TP4 logprob requests silently return `None`

Source-only diagnosis, 2026-09-27. No implementation edits or hardware runs.
Paths below are relative to the repository root unless abbreviated to this
model's directory, `models/autoports/google_gemma_4_26b_a4b_it/`.

## Verdict

Reject logprob requests explicitly at the model generator boundary on the
current TP4 configuration. This is the smallest justified change. The failure
is an explicit common-calculator topology gate, not evidence of a failed
log-softmax kernel or broken tuple unpacking. Both common sampler choices
share the gate. Extending shared support to four devices is plausible from
source, but numerical and trace qualification would be new work.

The full-model skill requires comparing sampler logprob handling
(`.agents/skills/full-model/SKILL.md:49`); it does not require adding TP4 logprob
support. The user has not separately required logprobs. Preserve device
sampling, penalties, and trace behavior, and document this optional limitation.
Do not silently return `None` for a request that asks for logprobs.

## Observed failure and causal source

- `doc/full_model/sampling_contract_final.log` terminates at
  `tests/check_full_sampling.py:116`: the assertion requiring non-`None`, finite
  `gen.last_log_probs` fails. The preceding penalty output-count assertion at
  line 115 completed. This supports the count invariant for that run; the
  combined test did not finish or produce its final success report.
- The test requests sampled generation with temperature 0.8, top-k 16,
  top-p 0.9, seed 0, three nondefault penalties, and `enable_log_probs=True`
  (`tests/check_full_sampling.py:103`). The trace log marks
  `penalties=True, log_probs=True`, but the key records requested mode, not
  whether the calculator supports that topology.
- `tt/generator.py:50` selects `use_topk_logprobs=False`. The normal sampler
  calls `calculate_log_probs(x, tt_out_tok)` when enabled
  (`models/common/sampling/tt_sampling.py:1125`). The generator retains the
  tuple's second element, so `None` propagates as designed by the common code.
- `models/common/sampling/tt_log_probs.py:419` returns false unless total
  device count is **8 or 32** and the sharded device dimension is at least two.
  Both `calculate_log_probs` and `calculate_topk_log_probs` return `None`
  immediately if that check fails. A `(1,4)` mesh therefore never executes the
  logprob numerical operations.

## Both logprob paths and the alternate sampler

| Path | Relevant contract | TP4 consequence |
| --- | --- | --- |
| Current scalar sampled-token path | `tt_log_probs.py:432` finds owning shard from float32 global token ID, gathers the local selected logit, masks other shards, and reduces. Full-vocabulary max and sum-exp give `logit - max - log(sum_exp)`. Returns preallocated BF16 `[1,1,1,32]`. | The topology guard returns `None` before these operations. |
| Optional top-k path | `tt_log_probs.py:607` narrows gathered candidates to 32, gathers global indices, and applies the same full-vocabulary normalization. Returns `LogProbsResult` with persistent `[1,1,32,32]` values/indices. | The same guard returns `None`. Flipping `use_topk_logprobs` cannot fix TP4 and changes the output contract. |
| `Sampling1D` alternative | `models/common/modules/sampling/sampling_1d.py:203` imports the same calculator through `models/common/utils.py:12`; its top-k path calls the scalar method at line 463. Its argmax path omits logprobs. | Switching common samplers does not provide TP4 logprobs. |

`models/common/sampling/generator.py:321` applies penalties before sampling.
The scalar logprob method receives those logits and normalizes over the full
vocabulary; it does not return a temperature/top-p/top-k-renormalized sampling
probability. Any future oracle must compare the intended logits at that exact
boundary rather than assume those two distributions coincide.

Both calculator modes allocate their outputs in the constructor
(`tt_log_probs.py:207-254`). Per-call normalization statistics are released
before return (`_release_global_stats`, line 504). There is no source basis
for blaming this observed `None` on a surviving logprob output allocation or
trace replay; the guard prevents computation entirely.

## Smallest appropriate repair and focused checks

1. Add pure request validation that rejects any formatted
   `enable_log_probs=True` lane or any `num_logprobs > 0` lane with a clear TP4
   unsupported error. Reject positive top-logprob counts even if the enable
   flag is false, so that contradictory requests are not silently ignored.
   Scalar and per-slot lists must behave consistently.
2. Call that validation from `generate()` **before** `reset()` and
   `_standalone_cache()`. Current `generate()` performs those mutations before
   entering `configure_sampling()` (`tt/generator.py:392-394`). Also call it
   from public `configure_sampling()` before `_release_trace()`, seed copies,
   penalty resets, or any other sampling-state mutation. A guard only in
   `configure_sampling()` does not preserve state for invalid `generate()`
   requests. A shared pure helper avoids divergent validation.
3. Change the penalty test to request no logprobs and retain the count
   assertion. Test rejection separately for scalar enable, a true per-slot
   flag, scalar positive count, and a per-slot positive count. Verify valid
   default/penalized parameters still pass validation.
4. A host-only boundary test can use an uninitialized generator shell or
   stubs: make `reset`, `_standalone_cache`, `_release_trace`, `_copy`, and
   sampler mutations fail if reached. An unsupported call must raise the
   documented error first. For direct configure calls, verify seeds, trace
   identifiers, and counters remain unchanged. No device is needed to prove
   early rejection.
5. Run the existing device sampling contract test after adapting its asserted
   capability; its normal penalty-count, seeded replay, device-feedback, and
   host-oracle checks remain relevant. No new hardware experiment is required
   merely to establish the topology gate. Record unsupported TP4 logprobs in
   the sampler comparison/capability documentation.

## Optional support-extension experiment, only if later required

The gate is not proof of a fundamental TP4 limitation. TP-axis inference,
shard masks, and global-stat reshapes use the variable device count
(`tt_log_probs.py:181-197, 231-244, 361-417`); four-device arithmetic is
plausible. The minimal candidate would admit four devices in `_is_supported`,
but that is not yet a qualified fix and would expose both shared paths.

Use a standalone oracle before involving full-model weights:

- TP4 mesh `(1,4)`, padded logical batch 32, local vocabulary 65,536, exact
  BF16 input logits. Exercise active batches 1, 3, and 32 with finite inactive
  rows. Use moderate values and nontrivial probability mass on every shard;
  a single overwhelming maximum does not check the normalizer adequately.
- Test scalar selected-token IDs around shard boundaries: 65,535/65,536,
  131,071/131,072, 196,607/196,608, and 262,143. Compare every replicated
  output to float32 `torch.log_softmax` on the exact BF16 logits presented to
  the calculator. Report maximum absolute logprob error and choose a justified
  BF16 tolerance; a finite-value assertion alone is insufficient.
- Qualify the top-k method separately with known unique candidates and exact
  global indices. Check values against the same full-vocabulary oracle.
  Do not silently substitute its result object for the scalar tensor API.
- Repeat changed-input A/B/A eager and traced execution with stable output
  addresses; toggle enabled/disabled requests and reset between requests.
  Enable `TT_METAL_TRACE_ALLOC_TRACKING=1` for lifecycle verification, and
  check a later eager prefill followed by replay for live temporary buffers.
  Tracker timings are not performance evidence.
- Only after those checks should the reduced full-model penalty case compare
  selected-token logprobs against its penalty-adjusted device logits. This
  future qualification is outside the current bounded diagnosis.

## Remaining uncertainty

Source proves why TP4 returns `None`, but does not prove numerical correctness
if the guard is widened. No hardware test, support extension, or implementation
change was performed by this diagnosis.
