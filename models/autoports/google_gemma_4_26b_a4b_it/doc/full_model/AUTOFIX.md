# AutoFix Report

## Starting evidence

`AUTODEBUG_prefill_l1.md`, `AUTODEBUG_trace_contract.md`, and
`bringup/artifacts/reference-fix/AUTODEBUG.md`.

## Hypothesis experiments

1. HF chat tokenizer returns mapping instead of token list. Tokenizer-only
   control verifies. Explicit `return_dict=False` repair passes10 host tests
   and the original100-token HF reference command.
2. Full-model prefill L1 collision is caused by extra semaphore sets alone.
   Reduced261-token probe reserves all30sets and passes: refuted.
3. Full-stack router persistent allocations depress shared L1 address frontier.
   Reduced probe adding28*8KiB exact-spec router storage reproduces the same
   SDPA overlap: verified. Model-owned serial sharing saves237568B; original
   all-layer readiness rerun passes both100-position top5/top100 gates. Decoder precision and geometry intact.
4. Split sampling leaves unsafe allocations after model capture. Combined
   reduced trace-allocation tracking passes; no unsafe survivor found on tested
   greedy path. The warning without tracking is generic, not evidence of
   corruption. Explicit sampling modes subsequently pass the final reduced contract test.
5. Reset dropped buffers instead of zeroing owned cache. Source verifies the
   contract violation. Repaired reset retains buffer/trace objects, zeroes owned
   KV and state; trace regression verifies identity, zero contents and repeated
   generation. External caches remain caller-owned.
6. Inactive negative positions reached unsigned RoPE lookup. Static active-slot
   capture now excludes inactive rows and freezes negative positions. MixedB3 inactive-slot and B32 independent-cache comparisons pass under trace
   allocation tracking; full30-layerB32 also passes.

7. Canonical sampler offsets own32 logical rows, while B3 indices had3. Raw
   standalone probe reproduces binary broadcast failure; actual device padding
   fixes selected greedy/sampled B1/B3/B32. Model trace now emits padded logits.
8. Public token slicing allocated a live buffer after trace capture. Tracker
   rejects it on next replay. Independent persistent-output probe proves B3/B32
   preallocation and output_tensor slice; original mixed-slot test now passes.
9. Host/device greedy completions differed on the reduced2-layer diagnostic.
   Common-prefix full-logit oracle finds44080–68885 exact maxima at the HF
   softcap30 on these incomplete-stack outputs. Both choices equal30 at every
   step. This is greedy tie-breaking within the common sampler's documented
   >32-ties limit, not a sampled-mode substitution. `sampling_greedy_oracle.json`.
   Full-model quality is evaluated only with all30 layers, separately.
10. Initial review found fixed seeds for omitted seed and silent host-policy
    mismatch. Fresh request entropy and explicit host-policy rejection are
    verified in `sampling_contract.json`. Explicitseed repeatability, mode
    alternation, device seed/token feedback and no-host-fallback audits pass.

## Final status

All identified runtime failures are fixed or resolved with controls. The
native unused-scatter initializer fix passes the original Watcher regression
and focused linear/ring controls. Optional TP4 logprob requests are explicitly
rejected before mutation, matching both common implementations' support gate;
this is a documented API limitation, not a host fallback. Penalty counts,
seeded sampling and host compatibility pass `sampling_contract_pass.log`.

The final host-first-token change exposed an obsolete tie-equality assertion.
The prefill/decode common-prefix oracle verifies both choices are exact maxima
at softcap30 on the reduced model. Full30-layer quality is separately passing.
Independent final stage review returns clean-pass (`stage_review_final.md`);
no result is inferred from source inspection alone.
